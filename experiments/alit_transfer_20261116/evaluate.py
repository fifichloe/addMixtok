"""ALIT-specific Long scorer, matched K=64 and mean-K=64 reconstruction."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh, sqrtm
from torch.utils.data import DataLoader

from common import PAPER, SEMANTIC, ImagePaths, load_alit
from runtime import ALITFrozenInputs

sys.path.insert(0, str(PAPER / 'addMixtok_metrics_work'))
from mixtok_long_no_density.recon.io import feature_tensor, sha256_file
from mixtok_long_no_density.recon.scorer import LongScorer


def masks(scores, k):
    order = np.argsort(-scores, axis=1, kind='stable')
    result = np.zeros(scores.shape, dtype='uint8')
    np.put_along_axis(result, order, (np.arange(256)[None] < k[:,None]).astype('uint8'), axis=1)
    return result


def threshold_for(scores, target=64):
    lo, hi = float(scores.min())-1, float(scores.max())+1
    best = (float('inf'), None, None)
    for _ in range(85):
        tau = float(np.float32((lo+hi)/2))
        k = np.clip((scores >= tau).sum(1), 48, 144)
        error = abs(float(k.mean())-target)
        if error < best[0]:
            best = (error, tau, k.copy())
        if k.mean() > target:
            lo = tau
        else:
            hi = tau
    if best[0] > .01:
        raise RuntimeError(f'Unable to match mean K={target}: {best[0]}')
    return best[1], best[2]


def moments(array):
    n = len(array)
    mean = sum((np.asarray(array[i:i+512], dtype='float64').sum(0)
                for i in range(0,n,512)), np.zeros(2048)) / n
    covariance = np.zeros((2048,2048),dtype='float64')
    for i in range(0,n,512):
        delta = np.asarray(array[i:i+512], dtype='float64')-mean
        covariance += delta.T@delta
    return mean, covariance/(n-1)


def fid(reference, fake):
    mr,cr=moments(reference); mf,cf=moments(fake)
    values, vectors=eigh(cr)
    root=(vectors*np.sqrt(np.maximum(values,0)))@vectors.T
    cross=root@cf@root; cross=(cross+cross.T)/2
    common=np.square(mr-mf).sum()+np.trace(cr)+np.trace(cf)
    result=float(common-2*np.sqrt(np.maximum(eigh(cross,eigvals_only=True),0)).sum())
    independent=sqrtm(cr@cf)
    if np.iscomplexobj(independent):
        assert np.abs(independent.imag).max()<1e-6
    check=float(common-2*np.trace(independent).real)
    assert abs(result-check)<1e-4, (result,check)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trained',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/rebuttal_20261106/reconstruction_50k/validation_files.txt'))
    parser.add_argument('--inception',type=Path,default=Path('/home/heyefei/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth'))
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--num-workers',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261116)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    paths=args.manifest.read_text().splitlines()
    assert len(paths)==len(set(paths))==50000
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    n=len(paths)
    model, checkpoint=load_alit()
    runtime=ALITFrozenInputs(model,SEMANTIC)
    scorer=LongScorer(args.trained/'gain_best.pt',args.trained/'whitening.npz',
                      args.trained/'semantic_rel.npz')
    torch.set_grad_enabled(False)
    loader=DataLoader(ImagePaths(paths),batch_size=args.batch_size,
                      num_workers=args.num_workers,pin_memory=True,
                      persistent_workers=args.num_workers>0)
    scores=np.lib.format.open_memmap(out/'scores.npy',mode='w+',dtype='float32',shape=(n,256))
    f1=np.lib.format.open_memmap(out/'f1.npy',mode='w+',dtype='float16',shape=(n,256,16,16))
    f2=np.lib.format.open_memmap(out/'f2.npy',mode='w+',dtype='float16',shape=(n,256,16,16))
    started=time.monotonic();cursor=0
    for images in loader:
        z=runtime.prepare(images.cuda(non_blocking=True))
        b=len(images)
        scores[cursor:cursor+b]=scorer.score(z).cpu().numpy()
        f1[cursor:cursor+b]=z['f1'].half().cpu().numpy()
        f2[cursor:cursor+b]=z['f2'].half().cpu().numpy()
        cursor+=b
        if cursor%400==0 or cursor==n:
            (out/'progress.json').write_text(json.dumps(dict(stage='scoring',completed=cursor,total=n,seconds=time.monotonic()-started)))
    assert cursor==n and np.isfinite(scores).all()
    for array in (scores,f1,f2):array.flush()
    tau, kd=threshold_for(scores)
    kf=np.full(n,64,dtype='int64')
    selected=dict(new_fixed64=masks(scores,kf),new_dynamic64=masks(scores,kd))
    for name,mask in selected.items():np.save(out/(name+'_mask.npy'),mask)
    np.save(out/'dynamic_K.npy',kd)
    policy=dict(target_mean_local_K=64,measured_mean_local_K=float(kd.mean()),
        threshold=tau,k_min=int(kd.min()),k_max=int(kd.max()),
        scorer='ALIT-trained density-free Long: gain + 0.5 white + 0.75 CAM + 0.9 U',
        calibration_uses_validation_scores_only=True,quality_not_computed_yet=True,
        original_ALIT_router_used_only_for_baseline=True,
        checkpoint=checkpoint,images_sha256=sha256_file(args.manifest),
        scorer_weights={name:sha256_file(args.trained/name) for name in
                        ('gain_best.pt','whitening.npz','semantic_rel.npz')})
    (out/'policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    from torchmetrics.image.fid import NoTrainInceptionV3
    inception=NoTrainInceptionV3(name='inception-v3-compat',features_list=['2048'],
        feature_extractor_weights_path=str(args.inception)).cuda().eval().requires_grad_(False)
    names=['old_router_fixed64','new_fixed64','new_dynamic64']
    real=np.lib.format.open_memmap(out/'real_features.npy',mode='w+',dtype='float32',shape=(n,2048))
    fake={name:np.lib.format.open_memmap(out/(name+'_features.npy'),mode='w+',
           dtype='float32',shape=(n,2048)) for name in names}
    psnr={name:0. for name in names}
    cursor=0
    for images in loader:
        images=images.cuda(non_blocking=True);b=len(images)
        a=torch.as_tensor(np.asarray(f1[cursor:cursor+b]),device='cuda')
        c=torch.as_tensor(np.asarray(f2[cursor:cursor+b]),device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            base=model.decode(a).float()
            logits=model.router(a,base*2-1,c).float().flatten(1)
        old_mask=torch.zeros_like(logits,dtype=torch.bool).scatter_(1,logits.topk(64,dim=1).indices,True).reshape(-1,1,16,16)
        candidates={'old_router_fixed64':old_mask,
                    'new_fixed64':torch.as_tensor(selected['new_fixed64'][cursor:cursor+b],device='cuda').reshape(-1,1,16,16),
                    'new_dynamic64':torch.as_tensor(selected['new_dynamic64'][cursor:cursor+b],device='cuda').reshape(-1,1,16,16)}
        real[cursor:cursor+b]=feature_tensor(inception,(images*255).round().to(torch.uint8))
        for name,mask in candidates.items():
            pred=runtime.decode(a,c,mask).clamp(0,1)
            mse=(pred-images).square().flatten(1).mean(1)
            psnr[name]+=float((-10*torch.log10(mse+1e-8)).sum())
            fake[name][cursor:cursor+b]=feature_tensor(inception,(pred*255).round().to(torch.uint8))
        cursor+=b
        if cursor%400==0 or cursor==n:
            (out/'progress.json').write_text(json.dumps(dict(stage='quality',completed=cursor,total=n,seconds=time.monotonic()-started)))
    assert cursor==n
    real.flush()
    for array in fake.values():array.flush()
    runtime.audit()
    results={name:dict(rFID=fid(real,fake[name]),PSNR=psnr[name]/n,
                       mean_local_K=64. if name!='new_dynamic64' else float(kd.mean()),
                       mean_content_tokens=128. if name!='new_dynamic64' else 64+float(kd.mean()))
             for name in names}
    final=dict(status='complete',n=n,checkpoint=checkpoint,policy=policy,
               results=results,seconds=time.monotonic()-started)
    (out/'results.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(final),flush=True)


if __name__=='__main__':main()
