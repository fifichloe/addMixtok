"""One feed-forward map of raw image/feature errors; no selected-set feedback."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FRESH=ROOT/'experiments/fresh_reconstruction_scorer_20261112'
sys.path.insert(0,str(FRESH))
from runtime import FrozenInputs,ImagePathDataset,write_json,sha256_file,select_once,standard

SCALES=ROOT/'experiments/light_grid_105_20261109/input_scales.npz'


def policies():
    rows=[dict(name='pixel_only',white=0.,pixel=1.,cam=0.),
          dict(name='white_only',white=1.,pixel=0.,cam=0.)]
    for pixel in [1.,2.,4.]:
        for cam in [0.,.5,1.]:
            rows.append(dict(name=f'w1_p{pixel}_c{cam}',white=1.,pixel=pixel,cam=cam))
    rows += [dict(name='w05_p1_c1',white=.5,pixel=1.,cam=1.),
             dict(name='w1_p2_mid1',white=1.,pixel=2.,cam=0.,mid=1.),
             dict(name='w1_p2_c05_density025',white=1.,pixel=2.,cam=.5,density=.25),
             dict(name='w1_p2_c05_density05',white=1.,pixel=2.,cam=.5,density=.5),
             dict(name='w1_p2_c1_density05',white=1.,pixel=2.,cam=1.,density=.5),
             dict(name='w1_logp2_c05',white=1.,pixel=0.,cam=.5,logpixel=2.)]
    return rows


class AnalyticMaps:
    def __init__(self,device='cuda'):
        with np.load(SCALES) as value:
            self.mean=torch.as_tensor(value['mean'],device=device)[None,:,None,None]
            self.std=torch.as_tensor(value['std'],device=device)[None,:,None,None].clamp_min(1e-6)

    @torch.no_grad()
    def __call__(self,z):
        f1,f2=z['f1'].float(),z['f2'].float()
        a=(f1-self.mean)/self.std;b=(f2-self.mean)/self.std
        white=1-F.cosine_similarity(a,b,dim=1).unsqueeze(1)
        # Static density is independent of any scores, thresholds or masks.
        tokens=F.normalize(b,dim=1).flatten(2).transpose(1,2)
        sim=(tokens@tokens.transpose(1,2)).clamp_min(0).square()
        density=(sim.sum(-1)-sim.diagonal(dim1=1,dim2=2))/255
        features=z['features']
        pixel=features[:,770:771]
        result=dict(white=standard(white),pixel=standard(pixel),
            cam=features[:,777:778],mid=standard(features[:,774:775]),
            density=standard(density.reshape(-1,1,16,16)),logpixel=standard(torch.log1p(pixel)),
            fresh_prior=z['independent_prior'])
        assert all(torch.isfinite(value).all() for value in result.values())
        return result


def score(maps,policy):
    if policy['name']=='fresh_prior':return maps['fresh_prior']
    # One final score. No provisional mask or greedy/set-dependent update.
    return sum(policy.get(key,0.)*maps[key] for key in ['white','pixel','cam','mid','logpixel'])-policy.get('density',0.)*maps['density']
