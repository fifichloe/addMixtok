"""Base-only student: exact frozen local ranking, trainable budget features/U."""
import copy
import torch
from torch import nn

class GenerationRouter(nn.Module):
    def __init__(self,utility_weight,threshold,hidden=128,depth=3):
        super().__init__()
        self.utility_weight,self.threshold=float(utility_weight),float(threshold)
        self.f_proj=nn.Conv2d(256,hidden,1)
        self.pos_embed=nn.Parameter(torch.zeros(1,hidden,16,16),requires_grad=False)
        self.trunk=nn.Sequential(*[nn.Sequential(nn.Conv2d(hidden,hidden,3,padding=1),nn.GroupNorm(8,hidden),nn.SiLU()) for _ in range(depth)])
        self.score_head=nn.Conv2d(hidden,1,1)
        nn.init.normal_(self.score_head.weight,std=1e-3);nn.init.zeros_(self.score_head.bias)
        self.offset_head=nn.Sequential(nn.Linear(2*hidden,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.offset_head[-1].weight);nn.init.zeros_(self.offset_head[-1].bias)
        self.f_proj.requires_grad_(False);self.trunk.requires_grad_(False);self.score_head.requires_grad_(False)
        self.budget_f_proj=copy.deepcopy(self.f_proj).requires_grad_(True)
        self.budget_pos_embed=nn.Parameter(self.pos_embed.detach().clone(),requires_grad=True)
        self.budget_trunk=copy.deepcopy(self.trunk).requires_grad_(True)

    def load_parent_state_dict(self,parent):
        expected=set(self.state_dict())
        assert set(parent)=={key for key in expected if not key.startswith('budget_')}
        merged={key:parent[key.removeprefix('budget_')] for key in expected}
        self.load_state_dict(merged,strict=True)

    def forward(self,base):
        assert base.shape[1:]==(256,16,16)
        with torch.no_grad():
            h=self.trunk(self.f_proj(base)+self.pos_embed.to(base.dtype))
            local=self.score_head(h).float().flatten(1)
            local=local-local.mean(1,keepdim=True)
        # This is the new gradient path. Do not detach budget_h: the existing
        # U head and its copied input feature extractor learn together.
        budget_h=self.budget_trunk(self.budget_f_proj(base)+self.budget_pos_embed.to(base.dtype))
        features=budget_h.float()
        pooled=torch.cat([features.mean((2,3)),features.std((2,3),correction=0)],1)
        utility=3*torch.tanh(self.offset_head(pooled).float().squeeze(1)/3)
        score=local+self.utility_weight*utility[:,None]
        return dict(local=local,utility=utility,score=score)

    @staticmethod
    def select_output(output,threshold):
        score=output['score'];k=(score>=threshold).sum(1).clamp(48,144)
        order=score.argsort(dim=1,descending=True,stable=True)
        mask=torch.zeros_like(score,dtype=torch.bool).scatter(1,order,torch.arange(256,device=score.device)[None]<k[:,None])
        return dict(K=k,mask=mask)

    @torch.no_grad()
    def select(self,base,threshold=None):
        output=self(base)
        return dict(**output,**self.select_output(output,self.threshold if threshold is None else threshold))
