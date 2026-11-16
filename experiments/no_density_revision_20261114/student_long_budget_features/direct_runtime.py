"""Training-only frozen decoding/semantic supervision; student input is base only."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from model import GenerationRouter
from policies import sha
from quality_eval import Codec

B0_WEIGHTS = Path('/tmp/mixtok_light_backbones_20261111/efficientnet_b0_rwightman-7f5810bc.pth')
B0_SHA256 = '7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934'


class SemanticTeacher(nn.Module):
    """Reuse the preceding reconstruction experiments' frozen lightweight loss."""
    def __init__(self):
        super().__init__()
        from torchvision.models import efficientnet_b0
        assert sha(B0_WEIGHTS) == B0_SHA256
        self.net = efficientnet_b0(weights=None)
        self.net.load_state_dict(torch.load(B0_WEIGHTS, map_location='cpu', weights_only=True))
        self.net.eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, x01):
        x = (x01.float() - self.mean) / self.std
        middle = None
        for index, layer in enumerate(self.net.features):
            x = layer(x)
            if index == 5:
                middle = x.float()
        deep = x.float()
        logits = F.linear(deep.mean((2, 3)), self.net.classifier[1].weight,
                          self.net.classifier[1].bias)
        return middle, deep, logits.float()


class DirectRuntime:
    def __init__(self):
        self.codec = Codec()
        self.semantic = SemanticTeacher().cuda().eval().requires_grad_(False)
        self.semantic_versions = [(name, value, value._version) for name, value in
            [*self.semantic.named_parameters(), *self.semantic.named_buffers()]]

    @torch.no_grad()
    def prepare(self, images):
        # Only base leaves this method as the student's input. RGB/F2/semantic
        # targets below are training supervision, never student features.
        base = self.codec.base(images).detach()
        target = images * 2 - 1
        with torch.autocast('cuda', dtype=torch.bfloat16):
            local, _ = self.codec.ev.native_llamagen_feature(
                self.codec.model.llamagen_vq, target, 8, allow_encoder_grad=False)
            reference = self.semantic(images)
        return dict(base=base, local=local.detach(), target=target.detach(),
                    reference=tuple(value.detach() for value in reference))

    def decode(self, data, mask):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            m = mask.reshape(-1, 1, 16, 16).to(data['base'].dtype)
            reconstruction = self.codec.model.llamagen_vq.decoder(
                (1 - m) * data['base'] + m * data['local'])
        return reconstruction.float()

    def losses(self, prediction, data):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            middle, deep, logits = self.semantic(prediction.add(1).mul(.5).clamp(0, 1))
        real_middle, real_deep, real_logits = data['reference']
        mid_loss = (1 - F.cosine_similarity(middle, real_middle, dim=1)).mean((1, 2))
        deep_loss = (1 - F.cosine_similarity(deep, real_deep, dim=1)).mean((1, 2))
        probability = real_logits.float().softmax(-1)
        kl = (probability * (real_logits.float().log_softmax(-1) -
                            logits.float().log_softmax(-1))).sum(-1)
        semantic = .25 * mid_loss + deep_loss + .1 * kl
        rgb = (prediction - data['target']).square().mean((1, 2, 3))
        assert torch.isfinite(semantic).all() and torch.isfinite(rgb).all()
        return semantic, rgb

    def audit(self):
        codec = self.codec.audit()
        assert all(not value.requires_grad and value.grad is None and value._version == version
                   for _, value, version in self.semantic_versions)
        return dict(frozen_codec=codec, semantic_parameters_and_buffers_frozen=True,
                    B0_training_loss_only=True, B0_student_input=False,
                    B0_required_at_deployment=False,
                    Inception_or_LPIPS_used_for_training=False)

