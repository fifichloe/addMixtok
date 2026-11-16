"""B0-free full-validation scores from separately trained RGB-only gain head."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from ablation_common import HERE, OLD, VAL_PATHS, sha, write_json
from no_semantic import NoSemanticInputs
from gain_model import GainScorer
from routing import AnalyticMaps, standard, ImagePathDataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['short','long'],required=True)
    p.add_argument('--head',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    paths=VAL_PATHS.read_text().splitlines();assert len(paths)==len(set(paths))==50000
    group=np.array_split(np.arange(math.ceil(len(paths)/32)),a.shards)[a.shard]
    start=int(group[0])*32;end=min((int(group[-1])+1)*32,len(paths));paths=paths[start:end]
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'paths.txt').write_text('\n'.join(paths)+'\n')
    state=torch.load(a.head,map_location='cpu',weights_only=False)
    assert state['kind']==a.kind and state['semantic_target_used'] is False and state['semantic_channels_zero']==[774,775,776,777]
    for source,digest in state['source_sha256'].items():assert sha(source)==digest
    head=GainScorer().cuda().eval().requires_grad_(False);head.load_state_dict(state['model'],strict=True)
    runtime=NoSemanticInputs();maps_fn=AnalyticMaps()
    raw=np.lib.format.open_memmap(a.output/'no_semantic.npy',mode='w+',dtype='float32',shape=(len(paths),256))
    loader=DataLoader(ImagePathDataset(list(map(Path,paths)),runtime.crop),batch_size=32,num_workers=4,pin_memory=True)
    n=0;began=time.monotonic()
    with torch.no_grad():
        for images in loader:
            z=runtime.prepare(images.cuda(non_blocking=True));maps=maps_fn(z)
            x=torch.cat([z['features'].half().float(),maps['white'].half().float()],1)
            assert not x[:,774:778].count_nonzero()
            gain=standard(head(x)[:,1:2]);score=gain+.5*maps['white']-.25*maps['density']
            assert torch.isfinite(score).all();raw[n:n+len(images)]=score.flatten(1).cpu().numpy();n+=len(images)
            if n%320==0 or n==len(paths):
                progress=dict(done=n,total=len(paths),seconds=time.monotonic()-began)
                write_json(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    raw.flush();write_json(a.output/'report.json',dict(status='no_semantic_scores_complete',kind=a.kind,shard=a.shard,
        start=start,end=end,n=n,paths_sha256=sha(a.output/'paths.txt'),score_sha256=sha(a.output/'no_semantic.npy'),
        head=str(a.head),head_sha256=sha(a.head),quality_used=False,semantic_U_used=False,
        source_sha256={str(path):sha(path) for path in [Path(__file__),HERE/'no_semantic.py',HERE/'ablation_common.py',VAL_PATHS]},
        seconds=time.monotonic()-began,**runtime.audit()))


if __name__=='__main__':main()
