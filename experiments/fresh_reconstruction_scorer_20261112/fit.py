"""Fresh CNN regression of independent decoder/semantic utility gradients."""
import argparse, json, time, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from runtime import HERE, FreshScorer, INPUT_CHANNELS, write_json, sha256_file


class CacheDataset(Dataset):
    def __init__(self, folders, split):
        self.x=[]; self.y=[]; self.samples=[]
        self.paths=[]
        for folder in folders:
            report=json.loads((folder/'report.json').read_text())
            assert report['status']=='completed' and report['legacy_router_deleted']
            assert report['old_scores_used'] is False and report['old_masks_used'] is False
            x=np.load(folder/'inputs.npy',mmap_mode='r');y=np.load(folder/'gradients.npy',mmap_mode='r')
            indices=np.load(folder/'source_indices.npy')
            chosen=np.flatnonzero(indices%10 != 0 if split=='fit' else indices%10 == 0)
            slot=len(self.x);self.x.append(x);self.y.append(y)
            self.samples.extend((slot,int(i)) for i in chosen)
            paths=(folder/'paths.txt').read_text().splitlines()
            self.paths.extend(paths[i] for i in chosen)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        slot,index=self.samples[i]
        return (torch.from_numpy(np.array(self.x[slot][index],copy=True)),
                torch.from_numpy(np.array(self.y[slot][index],copy=True)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=128)
    a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261112);np.random.seed(20261112);random.seed(20261112)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    train=CacheDataset(a.cache,'fit');held=CacheDataset(a.cache,'held')
    assert len(train)==10843 and len(held)==1157 and not set(train.paths).intersection(held.paths)
    a.output.mkdir(parents=True,exist_ok=False)
    loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=2,pin_memory=True)
    validation=DataLoader(held,batch_size=a.batch_size,num_workers=2,pin_memory=True)
    rgb_weights=[.1,.5]
    heads=[FreshScorer().cuda() for _ in rgb_weights]
    # Deterministic, class-spanning FIT samples, never held/CAL/validation.
    sample=np.stack([train[int(i)][0].numpy() for i in np.linspace(0,len(train)-1,256,dtype=int)]).astype('float32')
    mean=sample.mean((0,2,3),dtype=np.float64).astype('float32')
    variance=np.square(sample,dtype=np.float64).mean((0,2,3))-mean.astype('float64')**2
    std=np.sqrt(np.maximum(variance,1e-6)).astype('float32')
    for head in heads:
        head.input_mean.copy_(torch.from_numpy(mean)[None,:,None,None].cuda())
        head.input_std.copy_(torch.from_numpy(std)[None,:,None,None].cuda())
    del sample
    optimizers=[torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=.01) for head in heads]
    # Target scaling from FIT labels only; absolute cross-image utility remains.
    collected=[]
    for y,slot in zip(train.y,range(len(train.y))):
        indices=[j for s,j in train.samples if s==slot]
        collected.append(np.asarray(y[indices],dtype='float32'))
    y=np.concatenate(collected)
    scales=[float(np.quantile(np.abs(y[:,0]+w*y[:,1]),.75)) for w in rgb_weights]
    assert all(v>0 and np.isfinite(v) for v in scales)
    def losses(pred,target):
        raw=F.smooth_l1_loss(pred,target)
        centered=F.mse_loss(pred-pred.mean((2,3),keepdim=True),target-target.mean((2,3),keepdim=True))
        return raw+.2*centered
    best=[float('inf')]*len(heads); history=[]; started=time.monotonic()
    source_sha={name:sha256_file(HERE/name) for name in ['runtime.py','fit.py']}
    for epoch in range(a.epochs):
        [head.train() for head in heads]
        sums=[0.]*len(heads); n=0
        for x,y in loader:
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            for i,(head,optimizer,w,scale) in enumerate(zip(heads,optimizers,rgb_weights,scales)):
                target=torch.asinh((y[:,0:1]+w*y[:,1:2])/scale)
                optimizer.zero_grad(set_to_none=True)
                pred=head(x)
                loss=losses(pred,target);assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.)
                optimizer.step();sums[i]+=float(loss.detach())*len(x)
            n+=len(x)
        [head.eval() for head in heads]
        totals=[0.]*len(heads); count=0
        with torch.no_grad():
            for x,y in validation:
                x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
                for i,(head,w,scale) in enumerate(zip(heads,rgb_weights,scales)):
                    target=torch.asinh((y[:,0:1]+w*y[:,1:2])/scale)
                    totals[i]+=float(losses(head(x),target))*len(x)
                count+=len(x)
        for i,(head,w,scale) in enumerate(zip(heads,rgb_weights,scales)):
            name=f'rgb{w}'
            row=dict(name=name,epoch=epoch,fit_loss=sums[i]/n,held_loss=totals[i]/count,seconds=time.monotonic()-started)
            history.append(row);print(json.dumps(row),flush=True)
            checkpoint=dict(model=head.state_dict(),rgb_weight=w,target_scale=scale,epoch=epoch,
                channels=INPUT_CHANNELS,parameters=sum(p.numel() for p in head.parameters()),
                source_sha256=source_sha,fit_n=len(train),held_n=len(held),legacy_router_used=False,
                initial_mask_or_legacy_teacher_used=False,metrics=row)
            if row['held_loss']<best[i]:
                best[i]=row['held_loss'];torch.save(checkpoint,a.output/(name+'_best.pt'))
            torch.save(checkpoint,a.output/(name+'_latest.pt'))
        write_json(a.output/'progress.json',dict(status='training',epoch=epoch,rows=history))
    write_json(a.output/'report.json',dict(status='completed',rows=history,best_held_loss=best,
        fit_n=len(train),held_n=len(held),fresh_random_initialization=True,legacy_router_used=False,
        initial_mask_or_legacy_teacher_used=False,seconds=time.monotonic()-started,source_sha256=source_sha))


if __name__=='__main__':main()
