"""GPU stage: one frozen input pass; no candidate reconstruction or quality use."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from ablation_common import HERE, PAPER, VAL_PATHS, Scores, load_inputs, load_file, sha, write_json, project_source_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['short','long'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=4);p.add_argument('--limit',type=int);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261114)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    paths=VAL_PATHS.read_text().splitlines();assert len(paths)==len(set(paths))==50000
    if a.limit is not None:paths=paths[:a.limit]
    groups=np.array_split(np.arange(math.ceil(len(paths)/32)),a.shards);group=groups[a.shard];assert len(group)
    start=int(group[0])*32;end=min((int(group[-1])+1)*32,len(paths));paths=paths[start:end]
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'paths.txt').write_text('\n'.join(paths)+'\n')
    runtime=load_inputs();score=Scores(a.kind,runtime)
    from runtime import ImagePathDataset
    baseline=load_file('original_recon_ablation_heuristics',PAPER/'experiments/rebuttal_20261106/original_recon_ablation.py')
    keys=['teacher','local','generation','legacy','without_gain','without_white','without_cam','without_density','gradient_entropy']
    values={key:np.lib.format.open_memmap(a.output/(key+'.npy'),mode='w+',dtype='float32',shape=(len(paths),256)) for key in keys}
    loader=DataLoader(ImagePathDataset(list(map(Path,paths)),runtime.crop),batch_size=32,num_workers=4,pin_memory=True)
    n=0;began=time.monotonic();checks=[]
    with torch.no_grad():
        for images in loader:
            images=images.cuda(non_blocking=True);z=runtime.prepare(images);scores=score(z)
            scores['gradient_entropy']=baseline.entropy_scores(z['target'])
            if n==0:
                single={key:value[:1] for key,value in z.items() if torch.is_tensor(value)}
                single_scores=score(single)
                for key in single_scores:
                    batch_value,one_value=scores[key][:1],single_scores[key]
                    difference=(batch_value.float()-one_value.float()).abs()
                    close=torch.allclose(batch_value,one_value,atol=1e-4,rtol=1e-4)
                    comparison=dict(score=key,single_image_score_close_to_batched=close,
                                    atol=1e-4,rtol=1e-4,max_absolute_difference=float(difference.max()),
                                    RMS_difference=float(difference.square().mean().sqrt()))
                    checks.append(comparison)
                    write_json(a.output/'first_batch_score_comparison.json',checks)
                    if key=='legacy' and not close:
                        # Preserve the published BF16 baseline's actual scores.
                        # Check per-image semantics separately in FP32 rather than
                        # claiming BF16 convolution kernels are batch-invariant.
                        with torch.autocast('cuda',enabled=False):
                            full=score.legacy(z['f1'].float(),z['base'].float(),z['f2'].float())[0][:1]
                            one=score.legacy(single['f1'].float(),single['base'].float(),single['f2'].float())[0]
                        fp32_close=torch.allclose(full,one,atol=1e-4,rtol=1e-4)
                        comparison.update(FP32_reference_single_image_close=fp32_close,
                                          FP32_max_absolute_difference=float((full-one).abs().max()),
                                          BF16_strict_comparison_failed=True,
                                          actual_cached_baseline_precision_unchanged='BF16')
                        write_json(a.output/'first_batch_score_comparison.json',checks)
                        assert fp32_close,'legacy FP32 per-image reference mismatch'
                    else:
                        assert close,key
            for key in keys:
                value=scores[key].flatten(1);assert value.shape==(len(images),256) and torch.isfinite(value).all()
                values[key][n:n+len(images)]=value.cpu().numpy()
            n+=len(images)
            if n%320==0 or n==len(paths):
                progress=dict(done=n,total=len(paths),seconds=time.monotonic()-began)
                write_json(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    for value in values.values():value.flush()
    sources={str(path):sha(path) for path in [Path(__file__),HERE/'ablation_common.py',PAPER/'experiments/rebuttal_20261106/original_recon_ablation.py',VAL_PATHS]}
    sources.update(project_source_hashes())
    write_json(a.output/'report.json',dict(status='scores_complete',kind=a.kind,shard=a.shard,shards=a.shards,start=start,end=end,n=n,
        total=50000 if a.limit is None else a.limit,paths_sha256=sha(a.output/'paths.txt'),validation_paths_sha256=sha(VAL_PATHS),
        quality_used=False,candidate_RGB_decoded=False,Inception_loaded=False,checks=checks,source_sha256=sources,
        score_sha256={key:sha(a.output/(key+'.npy')) for key in keys},seconds=time.monotonic()-began,
        precision='BF16 codec/features; FP16-rounded teacher head input then FP32; BF16 Fbase then FP32 student; legacy BF16',
        runtime_audit=runtime.audit(),scorers=score.audit()))


if __name__=='__main__':main()
