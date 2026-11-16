"""Fresh base-only spatial scorer and image-wide score-offset predictor."""
import torch
from torch import nn
from torch.nn import functional as F

class GenerationRouter(nn.Module):
    def __init__(self, utility_weight, threshold, hidden=128, depth=3):
        super().__init__()
        self.utility_weight, self.threshold = float(utility_weight), float(threshold)
        self.f_proj = nn.Conv2d(256, hidden, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, hidden, 16, 16))
        self.trunk = nn.Sequential(*[nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.SiLU()) for _ in range(depth)])
        self.score_head = nn.Conv2d(hidden, 1, 1)
        nn.init.normal_(self.score_head.weight, std=1e-3)
        nn.init.zeros_(self.score_head.bias)
        self.offset_head = nn.Sequential(nn.Linear(2*hidden, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.offset_head[-1].weight); nn.init.zeros_(self.offset_head[-1].bias)

    def forward(self, base):
        assert base.shape[1:] == (256, 16, 16)
        h = self.trunk(self.f_proj(base)+self.pos_embed.to(base.dtype))
        local = self.score_head(h).float().flatten(1)
        # The teacher local maps have zero spatial mean. Its separate scalar
        # utility must not disappear through centering/softmax of final scores.
        local = local-local.mean(1, keepdim=True)
        pooled = torch.cat([h.float().mean((2, 3)), h.float().std((2, 3), correction=0)], 1)
        utility = 3*torch.tanh(self.offset_head(pooled).float().squeeze(1)/3)
        score = local+self.utility_weight*utility[:, None]
        return dict(local=local, utility=utility, score=score)

    @staticmethod
    def select_output(output, threshold):
        score = output['score']
        k = (score >= threshold).sum(1).clamp(48, 144)
        order = score.argsort(dim=1, descending=True, stable=True)
        mask = torch.zeros_like(score, dtype=torch.bool).scatter(1, order,
            torch.arange(256, device=score.device)[None] < k[:, None])
        return dict(K=k, mask=mask)

    @torch.no_grad()
    def select(self, base, threshold=None):
        output = self(base)
        return dict(**output, **self.select_output(output, self.threshold if threshold is None else threshold))

def distillation_loss(output, local, utility, utility_weight, threshold):
    local, utility = local.float(), utility.float()
    kl = F.kl_div((output['local']/2).log_softmax(1), (local/2).softmax(1), reduction='batchmean')
    shape = F.smooth_l1_loss(output['local'], local)
    shift = F.smooth_l1_loss(output['utility'], utility)
    # Distill the threshold boundary with soft targets, without a second
    # selection or a direct-count head. The fixed threshold is part of policy.
    teacher_score = local+utility_weight*utility[:, None]
    boundary = F.binary_cross_entropy_with_logits((output['score']-threshold)/.5,
        torch.sigmoid((teacher_score-threshold)/.5))
    total = kl+.05*shape+shift+.1*boundary
    return total, dict(kl=kl, shape=shape, utility=shift, boundary=boundary)
