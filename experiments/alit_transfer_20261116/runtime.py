"""Prepare current Long Grid Scorer inputs from the frozen ALIT Mix codec."""

from pathlib import Path
import sys

import torch
from torch.nn import functional as F

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / 'addMixtok_metrics_work'))
from mixtok_long_no_density.recon.runtime import FrozenSemantic, standard


class ALITFrozenInputs:
    def __init__(self, model, semantic_weights, device='cuda'):
        self.model = model.to(device).eval().requires_grad_(False)
        self.semantic = FrozenSemantic(semantic_weights).to(device).eval().requires_grad_(False)
        self.versions = [(f'{owner}.{name}', tensor, tensor._version)
                         for owner, module in [('alit_mix', self.model), ('semantic', self.semantic)]
                         for name, tensor in [*module.named_parameters(), *module.named_buffers()]]

    @torch.no_grad()
    def prepare(self, x01):
        assert x01.ndim == 4 and x01.shape[1:] == (3, 256, 256)
        target = x01.float()*2-1
        with torch.autocast('cuda', dtype=torch.bfloat16):
            paths = self.model.encode_first_rollout(x01)
            f1, f2 = paths['z_1d'], paths['z_2d']
            base01 = self.model.decode(f1).float()
        assert f1.shape == f2.shape == (len(x01), 256, 16, 16)
        base = base01*2-1
        with torch.autocast('cuda', dtype=torch.bfloat16):
            middle, deep, logits = self.semantic(torch.cat([x01, base01.clamp(0, 1)]))
        n = len(x01)
        mid_real, mid_base = middle[:n], middle[n:]
        real, basic = deep[:n], deep[n:]
        real_logits = logits[:n]
        weights = self.semantic.net.classifier[1].weight[real_logits.argmax(1)]
        real_cam = torch.einsum('bc,bchw->bhw', weights, real).clamp_min(0)
        base_cam = torch.einsum('bc,bchw->bhw', weights, basic).clamp_min(0)
        missed = (real_cam-base_cam).clamp_min(0)[:, None]
        cosine = 1-F.cosine_similarity(f1.float(), f2.float(), dim=1).unsqueeze(1)
        error = target-base
        pool = lambda value: F.adaptive_avg_pool2d(value, 16)
        resized = lambda value: F.interpolate(value, size=(16,16), mode='bilinear', align_corners=False)
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
        assert features.shape == (n, 780, 16, 16)
        assert all(torch.isfinite(value).all() for value in (features, f1.float(), f2.float(), base))
        return dict(features=features.detach(), f1=f1.detach(), f2=f2.detach(),
                    target=target.detach(), base=base.detach(), base01=base01.detach(),
                    reference=(mid_real.detach(), real.detach(), real_logits.detach()))

    @torch.no_grad()
    def decode(self, f1, f2, mask):
        assert mask.shape == (len(f1), 1, 16, 16)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            mixed = (1-mask.to(f1.dtype))*f1+mask.to(f1.dtype)*f2
            return self.model.decode(mixed).float()

    def audit(self):
        bad = [name for name, tensor, version in self.versions
               if tensor._version != version or tensor.requires_grad or tensor.grad is not None]
        assert not bad, bad[:8]
        return dict(codec_and_semantic_frozen=True, old_router_scores_unused=True,
                    current_Long_inputs_780_channels=True)
