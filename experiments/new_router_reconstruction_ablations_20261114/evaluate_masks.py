"""GPU stage: reconstruct pre-frozen masks and retain raw Inception features."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from ablation_common import HERE, VAL_PATHS, load_inputs, sha, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--policies',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=4);p.add_argument('--names',nargs='+')
    p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261114)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    protocol=json.loads((a.policies/'protocol.json').read_text());assert protocol['status']=='policies_frozen_before_quality'
    paths=VAL_PATHS.read_text().splitlines();assert paths==(a.policies/'paths.txt').read_text().splitlines()
    assert len(paths)==len(set(paths))==50000 and protocol['paths_sha256']==sha(a.policies/'paths.txt')
    rows={row['name']:row for row in protocol['variants']};names=a.names or list(rows);assert set(names)<=set(rows)
    group=np.array_split(np.arange(math.ceil(len(paths)/32)),a.shards)[a.shard]
    start=int(group[0])*32;end=min((int(group[-1])+1)*32,len(paths));selected=paths[start:end]
    masks={};budgets={}
    for name in names:
        row=rows[name];assert sha(row['mask_file'])==row['mask_sha256'] and sha(row['K_file'])==row['K_sha256']
        masks[name]=np.load(row['mask_file'],mmap_mode='r');budgets[name]=np.load(row['K_file'],mmap_mode='r')
        assert masks[name].shape==(50000,256) and np.array_equal(masks[name].sum(1),budgets[name])
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'paths.txt').write_text('\n'.join(selected)+'\n')
    if names==['no_semantic_retrained_fixed96']:
        from no_semantic import NoSemanticInputs
        runtime=NoSemanticInputs()
    else:runtime=load_inputs()
    from runtime import ImagePathDataset
    from cache_train_counterfactual import feature_tensor
    from torchmetrics.image.fid import NoTrainInceptionV3
    inc=NoTrainInceptionV3(name='inception-v3-compat',features_list=['2048'],
        feature_extractor_weights_path='/home/heyefei/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth').cuda().eval()
    real=np.lib.format.open_memmap(a.output/'real.npy',mode='w+',dtype='float32',shape=(len(selected),2048))
    fake={name:np.lib.format.open_memmap(a.output/(name+'.features.npy'),mode='w+',dtype='float32',shape=(len(selected),2048)) for name in names}
    sums={name:0. for name in names};n=0;began=time.monotonic()
    loader=DataLoader(ImagePathDataset(list(map(Path,selected)),runtime.crop),batch_size=a.batch_size,num_workers=4,pin_memory=True)
    with torch.no_grad():
        for images in loader:
            images=images.cuda(non_blocking=True);z=runtime.prepare(images);b=len(images)
            real[n:n+b]=feature_tensor(inc,runtime.ev.to_uint8(z['target'],'minus1_1'))
            for name in names:
                mask=torch.as_tensor(np.array(masks[name][start+n:start+n+b],copy=True),device='cuda',dtype=torch.float32).reshape(b,1,16,16)
                pred=runtime.decode(z,mask)
                fake[name][n:n+b]=feature_tensor(inc,runtime.ev.to_uint8(pred,'minus1_1'))
                sums[name]+=float(runtime.ev.psnr_sum(pred,z['target'],'minus1_1'))
            n+=b
            if n%320==0 or n==len(selected):
                progress=dict(done=n,total=len(selected),seconds=time.monotonic()-began,PSNR_sums=sums)
                write_json(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    real.flush()
    for value in fake.values():value.flush()
    write_json(a.output/'report.json',dict(status='quality_features_complete',kind=protocol['kind'],shard=a.shard,shards=a.shards,
        start=start,end=end,n=n,names=names,PSNR_sums=sums,seconds=time.monotonic()-began,
        policy_protocol=str(a.policies/'protocol.json'),policy_sha256=sha(a.policies/'protocol.json'),
        paths_sha256=sha(a.output/'paths.txt'),feature_sha256={name:sha(a.output/(name+'.features.npy')) for name in names},
        real_sha256=sha(a.output/'real.npy'),source_sha256={str(path):sha(path) for path in [Path(__file__),HERE/'ablation_common.py',VAL_PATHS]},
        precision='BF16 frozen codec; FP32 Inception TF32 disabled; FP64 full pooled moments',runtime=runtime.audit()))


if __name__=='__main__':main()
