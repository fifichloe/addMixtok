"""Fresh single-score reconstruction routing: no legacy router execution/data."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'experiments/dynamic_recon_20261108'),
               str(ROOT/'experiments/light_grid_105_20261109'),
               str(ROOT/'experiments/light_grid_20261109')]
from cache_train_counterfactual import load_model, ImagePathDataset, sha256_file, EXPECTED_CHECKPOINT_SHA256
from lookup_budget_screen import config

B0_WEIGHTS = Path('/tmp/mixtok_light_backbones_20261111/efficientnet_b0_rwightman-7f5810bc.pth')
B0_SHA = '7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934'
INPUT_CHANNELS = 780


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def standard(x):
    return (x-x.mean((2, 3), keepdim=True))/x.std((2, 3), keepdim=True, correction=0).clamp_min(1e-6)


class FrozenSemantic(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import efficientnet_b0
        assert sha256_file(B0_WEIGHTS) == B0_SHA
        self.net = efficientnet_b0(weights=None)
        self.net.load_state_dict(torch.load(B0_WEIGHTS, map_location='cpu', weights_only=True))
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


class FreshScorer(nn.Module):
    def __init__(self, channels=INPUT_CHANNELS, width=64):
        super().__init__()
        self.register_buffer('input_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('input_std', torch.ones(1, channels, 1, 1))
        # No BatchNorm, old parameters, initial masks or intermediate selection.
        self.net = nn.Sequential(nn.Conv2d(channels, width, 1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, 1, 1))

    def forward(self, x):
        assert x.shape[1:] == (INPUT_CHANNELS, 16, 16)
        return self.net(((x.float()-self.input_mean)/self.input_std).clamp(-12, 12))


def select_once(score, threshold=None, fixed_k=None):
    flat = score.float().flatten(1)
    assert flat.shape[1] == 256 and torch.isfinite(flat).all()
    if fixed_k is None:
        assert threshold is not None
        k = (flat >= threshold).sum(1).clamp(48, 144)
    else:
        k = torch.full((len(flat),), fixed_k, dtype=torch.long, device=flat.device)
    order = flat.argsort(dim=1, descending=True, stable=True)
    mask = torch.zeros_like(flat).scatter(1, order,
        (torch.arange(256, device=flat.device)[None] < k[:, None]).float())
    return mask.reshape_as(score), k


class FrozenInputs:
    def __init__(self):
        cfg = config()
        assert sha256_file(cfg.checkpoint) == EXPECTED_CHECKPOINT_SHA256
        self.model, self.ev, self.crop, self.meta = load_model(cfg, torch.device('cuda'))
        # Load codec as released, then DELETE the old scorer. No old head is run,
        # used as input, or used to create labels. Codec forward does not call it.
        assert hasattr(self.model, 'router')
        del self.model.router
        assert not any('router' in name for name, _ in self.model.named_modules())
        # LlamaGen registers its usage-only buffer as nn.Parameter via
        # register_buffer; Module.requires_grad_(False) does not touch buffers.
        # Detach such buffers once during setup, preserving their values.
        self.detached_usage_buffers=[]
        for name,module in self.model.named_modules():
            for key,value in list(module._buffers.items()):
                if value is not None and value.requires_grad:
                    assert key=='codebook_used', (name,key)
                    replacement=value.detach()
                    assert torch.equal(value,replacement)
                    module._buffers[key]=replacement
                    self.detached_usage_buffers.append(name+'.'+key)
        self.semantic = FrozenSemantic().cuda().eval().requires_grad_(False)
        self.versions = []
        for prefix, module in [('codec', self.model), ('semantic', self.semantic)]:
            for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
                self.versions.append((prefix+'.'+name, tensor, tensor._version))

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
