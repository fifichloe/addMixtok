"""Density-free Long = gain + 0.5 white + 0.75 CAM + 0.9 U."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .gain_model import GainScorer
from .runtime import standard

class LongScorer:
    def __init__(self, gain_checkpoint, scales, utility, threshold=0.3803865313529968, device='cuda'):
        state = torch.load(gain_checkpoint, map_location='cpu', weights_only=False)
        self.head = GainScorer().to(device).eval().requires_grad_(False)
        self.head.load_state_dict(state['model'], strict=True)
        with np.load(scales) as d:
            self.mean = torch.as_tensor(d['mean'], device=device)[None,:,None,None]
            self.std = torch.as_tensor(d['std'], device=device)[None,:,None,None].clamp_min(1e-6)
        with np.load(utility) as d:
            self.utility = {k:torch.as_tensor(d[k], device=device, dtype=torch.float32)
                            for k in ('mean','std','beta','bias')}
        self.threshold = float(threshold)

    @torch.no_grad()
    def score(self, z):
        # Exact retained Long formula and rounding from teacher_final.py.
        a = (z['f1'].float()-self.mean)/self.std
        b = (z['f2'].float()-self.mean)/self.std
        white = standard(1-F.cosine_similarity(a, b, dim=1).unsqueeze(1))
        features = z['features'].half().float()
        head_input = torch.cat([features, white.half().float()], 1)
        pooled = torch.cat([features.mean((2, 3)), features.std((2, 3), correction=0)], 1)
        state = self.utility
        utility = (((pooled-state['mean'])/state['std']).clamp(-8, 8)
                   @ state['beta']+state['bias']).clamp(-3, 3).reshape(-1)
        gain = standard(self.head(head_input)[:, 1:2])
        local = gain+.5*white+.75*z['features'][:, 777:778]
        return local.flatten(1)+.9*utility[:, None]


    @torch.no_grad()
    def select(self, z, threshold=None, fixed_k=None):
        score = self.score(z)
        if fixed_k is None:
            tau = self.threshold if threshold is None else float(threshold)
            k = (score >= tau).sum(1).clamp(48, 144)
        else:
            assert 0 <= fixed_k <= 256
            k = torch.full((len(score),), fixed_k, device=score.device, dtype=torch.long)
        order = score.argsort(dim=1, descending=True, stable=True)
        mask = torch.zeros_like(score, dtype=torch.bool).scatter(1, order,
            torch.arange(256, device=score.device)[None] < k[:,None])
        return dict(score=score, K=k, mask=mask)
