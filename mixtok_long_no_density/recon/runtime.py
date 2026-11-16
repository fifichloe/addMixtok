"""Original frozen Long feature preparation; explicit asset paths."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .io import load_codec, sha256_file
B0_SHA = '7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934'
INPUT_CHANNELS = 3 * 256 + 12

def standard(x):
    return (x-x.mean((2, 3), keepdim=True))/x.std((2, 3), keepdim=True, correction=0).clamp_min(1e-6)

class FrozenSemantic(nn.Module):
    def __init__(self, weights):
        super().__init__()
        from torchvision.models import efficientnet_b0
        assert sha256_file(weights) == B0_SHA
        self.net = efficientnet_b0(weights=None)
        self.net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.net.eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, x01):
        z = (x01.float()-self.mean)/self.std
        middle = None
        for index, layer in enumerate(self.net.features):
            z = layer(z)
            if index == 5:
                middle = z.float()
        deep = z.float()
        logits = F.linear(deep.mean((2, 3)), self.net.classifier[1].weight,
                          self.net.classifier[1].bias)
        return middle, deep, logits

class FrozenInputs:
    def __init__(self, codec_args, semantic_weights, device='cuda'):
        self.model, self.ev, self.crop, self.meta = load_codec(codec_args, torch.device(device))
        del self.model.router
        self.detached_usage_buffers = []
        for name, module in self.model.named_modules():
            for key, value in list(module._buffers.items()):
                if value is not None and value.requires_grad:
                    assert key == 'codebook_used'
                    module._buffers[key] = value.detach()
                    self.detached_usage_buffers.append(name+'.'+key)
        self.semantic = FrozenSemantic(semantic_weights).to(device).eval().requires_grad_(False)
        self.versions = [(prefix+'.'+name, value, value._version)
            for prefix, module in [('codec', self.model), ('semantic', self.semantic)]
            for name, value in [*module.named_parameters(), *module.named_buffers()]]

    @torch.no_grad()
    def prepare(self, x01):
        target = x01*2-1
        with torch.autocast('cuda', dtype=torch.bfloat16):
            base, extra = self.model(x01)
            f1 = extra['f_1d_lg']
            f2, _ = self.ev.native_llamagen_feature(self.model.llamagen_vq, target, 8, allow_encoder_grad=False)
            semantic = self.semantic(torch.cat([x01, base.float().add(1).mul(.5).clamp(0, 1)]))
        n = len(x01)
        middle, deep, logits = semantic
        mid_real, mid_base = middle[:n], middle[n:]
        real, basic = deep[:n], deep[n:]
        real_logits = logits[:n]
        weights = self.semantic.net.classifier[1].weight[real_logits.argmax(1)]
        real_cam = torch.einsum('bc,bchw->bhw', weights, real).clamp_min(0)
        base_cam = torch.einsum('bc,bchw->bhw', weights, basic).clamp_min(0)
        missed = (real_cam-base_cam).clamp_min(0)[:, None]
        cosine = 1-F.cosine_similarity(f1.float(), f2.float(), dim=1).unsqueeze(1)
        error = target-base.float()
        pool = lambda v: F.adaptive_avg_pool2d(v, 16)
        resized = lambda v: F.interpolate(v, size=(16, 16), mode='bilinear', align_corners=False)
        rgb_mse = pool(error.square().mean(1, keepdim=True))
        maps = [cosine, pool(error.abs().mean(1, keepdim=True)), rgb_mse,
                torch.log1p((f2.float()-f1.float()).square().mean(1, keepdim=True)),
                torch.log1p(f1.float().square().mean(1, keepdim=True)),
                torch.log1p(f2.float().square().mean(1, keepdim=True)),
                resized(1-F.cosine_similarity(mid_real, mid_base, dim=1).unsqueeze(1)),
                resized(1-F.cosine_similarity(real, basic, dim=1).unsqueeze(1)),
                resized(missed), standard(resized(missed)), standard(rgb_mse), standard(cosine)]
        scale = float(f1.shape[1])**.5
        features = torch.cat([F.normalize(f1.float(), dim=1)*scale,
            F.normalize(f2.float(), dim=1)*scale,
            F.normalize((f2-f1).float(), dim=1)*scale, *maps], 1)
        assert features.shape[1:] == (INPUT_CHANNELS, 16, 16)
        return dict(features=features.detach(), f1=f1.detach(), f2=f2.detach(),
            target=target, base=base.detach(), reference=(mid_real.detach(), real.detach(), real_logits.detach()),
            independent_prior=.6*standard(cosine)+.75*standard(resized(missed))+.3*standard(rgb_mse))

    def decode(self, z, mask):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            m = mask.to(z['f1'].dtype)
            return self.model.llamagen_vq.decoder((1-m)*z['f1']+m*z['f2']).float()

    def losses(self, pred, z):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            middle, deep, logits = self.semantic(pred.add(1).mul(.5).clamp(0, 1))
        mr, dr, lr = z['reference']
        middle_loss = (1-F.cosine_similarity(middle, mr, dim=1)).mean((1, 2))
        deep_loss = (1-F.cosine_similarity(deep, dr, dim=1)).mean((1, 2))
        target_prob = lr.float().softmax(-1)
        kl = (target_prob*(lr.float().log_softmax(-1)-logits.float().log_softmax(-1))).sum(-1)
        semantic = .25*middle_loss+deep_loss+.1*kl
        rgb = (pred-z['target']).square().mean((1, 2, 3))
        return semantic, rgb

    def audit(self):
        assert not hasattr(self.model, 'router')
        failures=[dict(name=name,version_before=version,version_after=p._version,
            requires_grad=p.requires_grad,has_grad=p.grad is not None)
            for name,p,version in self.versions
            if p._version!=version or p.requires_grad or p.grad is not None]
        assert not failures, failures
        return dict(legacy_router_deleted=True, old_scores_used=False, old_masks_used=False,
            old_hidden_features_used=False, all_pretrained_weights_and_buffers_unchanged=True,
            usage_only_buffers_detached_during_setup=self.detached_usage_buffers)
