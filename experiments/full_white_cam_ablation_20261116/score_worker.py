"""Frozen-head input ablations on the accepted density-free Long teacher."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'no_density_revision_20261114'
sys.path.insert(0, str(SOURCE))
from density_free_config import VAL_PATHS, OUTPUT, sha, source_identities, write_json
from teacher_no_density import TwoTeachers, ImagePathDataset, standard

NAMES = ('without_white_all_explicit', 'without_cam_all_explicit')


def utility_from_pooled(pooled, state):
    return (((pooled-state['mean'])/state['std']).clamp(-8, 8)
            @ state['beta']+state['bias']).clamp(-3, 3).reshape(-1)


@torch.inference_mode()
def scores(teacher, z):
    full = teacher.score_prepared(z)
    reference = full['local'][:, 1] + .9*full['utility'][:, None]
    a = (z['f1'].float()-teacher.mean)/teacher.std
    b = (z['f2'].float()-teacher.mean)/teacher.std
    white = standard(1-F.cosine_similarity(a,b,dim=1).unsqueeze(1))
    features = z['features'].half().float()
    head = teacher.heads['long']
    head_input = torch.cat([features,white.half().float()],1)
    assert head_input.shape[1] == 781
    assert torch.equal(white.flatten(1),full['white'])
    assert torch.allclose(features[:,777:778].flatten(1),full['cam'],atol=5e-4,rtol=5e-4)

    # Whitened cosine is absent from both of its explicit destinations:
    # the direct score and the gain-head's final input channel.
    no_white_input = head_input.clone()
    no_white_input[:,780:781] = head.input_mean[:,780:781]
    gain_no_white = standard(head(no_white_input)[:,1:2])
    no_white = (gain_no_white+.75*z['features'][:,777:778]).flatten(1)+.9*full['utility'][:,None]

    # CAM has two feature channels (raw and standardized), a direct score,
    # and corresponding mean/std inputs to the image-utility regressor.
    no_cam_input = head_input.clone()
    no_cam_input[:,776:778] = head.input_mean[:,776:778]
    gain_no_cam = standard(head(no_cam_input)[:,1:2])
    pooled = torch.cat([features.mean((2,3)),features.std((2,3),correction=0)],1)
    no_cam_pooled = pooled.clone()
    for index in (776,777,780+776,780+777):
        no_cam_pooled[:,index] = teacher.utility['mean'][index]
    u_no_cam = utility_from_pooled(no_cam_pooled,teacher.utility)
    no_cam = (gain_no_cam+.5*white).flatten(1)+.9*u_no_cam[:,None]
    assert all(torch.isfinite(x).all() and x.shape==reference.shape for x in (reference,no_white,no_cam))
    return reference,no_white,no_cam


def main(args):
    torch.set_num_threads(4);torch.manual_seed(20261113)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    paths=VAL_PATHS.read_text().splitlines()
    assert len(paths)==len(set(paths))==50000
    group=np.array_split(np.arange(math.ceil(len(paths)/args.batch)),args.world)[args.rank]
    start=int(group[0])*args.batch;end=min((int(group[-1])+1)*args.batch,len(paths))
    selected=paths[start:end]
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'paths.txt').write_text('\n'.join(selected)+'\n')
    teacher=TwoTeachers()
    loader=DataLoader(ImagePathDataset(list(map(Path,selected)),teacher.runtime.crop),
        batch_size=args.batch,num_workers=4,pin_memory=True)
    arrays={name:np.lib.format.open_memmap(args.output/(name+'.npy'),mode='w+',dtype='float32',shape=(len(selected),256))
            for name in ('reference',*NAMES)}
    old=OUTPUT/'validation'/f'shard_{args.rank}'
    assert (old/'paths.txt').read_text().splitlines()==selected
    baseline_local=np.load(old/'local.npy',mmap_mode='r')[:,1]
    baseline_u=np.load(old/'utility.npy',mmap_mode='r')
    began=time.monotonic();done=0;max_reference_gap=0.
    for images in loader:
        z=teacher.runtime.prepare(images.cuda(non_blocking=True))
        values=scores(teacher,z)
        b=len(images)
        for name,value in zip(arrays,values):arrays[name][done:done+b]=value.cpu().numpy()
        expected=baseline_local[done:done+b]+np.float32(.9)*baseline_u[done:done+b,None]
        gap=float(np.abs(values[0].cpu().numpy()-expected).max())
        max_reference_gap=max(max_reference_gap,gap)
        assert gap < 5e-4, (args.rank,done,gap)
        done+=b
        if done%320==0 or done==len(selected):
            write_json(args.output/'progress.json',dict(done=done,total=len(selected),seconds=time.monotonic()-began,
                max_reference_gap=max_reference_gap))
    assert done==end-start
    for value in arrays.values():value.flush()
    write_json(args.output/'report.json',dict(status='complete',rank=args.rank,world=args.world,start=start,end=end,n=done,
        names=list(NAMES),max_reference_gap=max_reference_gap,source_sha256=source_identities(),
        worker_sha256=sha(Path(__file__)),paths_sha256=sha(args.output/'paths.txt'),
        scores_sha256={name:sha(args.output/(name+'.npy')) for name in arrays},seconds=time.monotonic()-began,
        head_retrained=False,white_direct_and_head_channel_removed=True,
        cam_direct_head_channels_and_utility_pooled_channels_removed=True,
        input_ablation_uses_training_mean_imputation=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rank',type=int,required=True)
    p.add_argument('--world',type=int,default=3)
    p.add_argument('--batch',type=int,default=32)
    main(p.parse_args())
