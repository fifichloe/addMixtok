"""A fresh small local head plus per-image global utility, no old-score inputs."""
import torch
from torch import nn


class GainScorer(nn.Module):
    def __init__(self):
        super().__init__();channels=781;width=128
        self.register_buffer('input_mean',torch.zeros(1,channels,1,1))
        self.register_buffer('input_std',torch.ones(1,channels,1,1))
        self.local=nn.Sequential(nn.Conv2d(channels,width,1),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(8,width),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(8,width),nn.SiLU(),nn.Conv2d(width,2,1))
        self.global_utility=nn.Sequential(nn.Linear(channels*2,64),nn.SiLU(),nn.Linear(64,2))

    def forward(self,features):
        assert features.shape[1:]==(781,16,16)
        x=((features.float()-self.input_mean)/self.input_std).clamp(-12,12)
        global_input=torch.cat([x.mean((2,3)),x.std((2,3),correction=0)],1)
        return self.local(x)+self.global_utility(global_input)[:,:,None,None]
