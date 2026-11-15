"""Train a new scorer from non-routing-dependent endpoint reconstruction gains."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from routing import HERE,FRESH,write_json,sha256_file
from gain_model import GainScorer


class Data(Dataset):
    def __init__(self,folders,split):
        self.x=[];self.y=[];self.white=[];self.samples=[];self.paths=[]
        for folder in folders:
            report=json.loads((folder/'report.json').read_text())
            assert report['status']=='completed' and report['legacy_router_deleted'] and report['old_routing_or_mask_used'] is False
            x=np.load(report['source_inputs'],mmap_mode='r');y=np.load(folder/'gain.npy',mmap_mode='r')
            w=np.load(folder/'white.npy',mmap_mode='r');ids=np.load(report['source_indices'])
            assert len(x)==len(y)==len(w)==len(ids)
            chosen=np.flatnonzero(ids%10!=0 if split=='fit' else ids%10==0);slot=len(self.x)
            self.x.append(x);self.y.append(y);self.white.append(w)
            self.samples.extend((slot,int(i)) for i in chosen)
            paths=(folder/'paths.txt').read_text().splitlines();self.paths.extend(paths[i] for i in chosen)
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):
        slot,i=self.samples[index]
        x=np.concatenate([self.x[slot][i],self.white[slot][i]],0)
        return torch.from_numpy(x),torch.from_numpy(np.array(self.y[slot][i],copy=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=60);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261112);np.random.seed(20261112)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    fit=Data(a.cache,'fit');held=Data(a.cache,'held')
    assert len(fit)==10843 and len(held)==1157 and not set(fit.paths).intersection(held.paths)
    raw=[]
    for slot,value in enumerate(fit.y):
        ids=[i for s,i in fit.samples if s==slot];raw.append(np.asarray(value[ids]))
    raw=np.concatenate(raw);scales=np.quantile(np.abs(raw),.75,axis=(0,2,3)).clip(1e-6)
    a.output.mkdir(parents=True,exist_ok=False)
    model=GainScorer().cuda()
    # Only normalization statistics of independently prepared inputs are reused.
    old=torch.load(FRESH/'heads/rgb0.5_best.pt',map_location='cpu',weights_only=False)['model']
    model.input_mean[:,:780].copy_(old['input_mean'].cuda())
    model.input_std[:,:780].copy_(old['input_std'].cuda())
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=1e-5)
    loader=DataLoader(fit,batch_size=64,shuffle=True,num_workers=2,pin_memory=True)
    validation=DataLoader(held,batch_size=64,num_workers=2,pin_memory=True)
    scales_tensor=torch.tensor(scales,device='cuda',dtype=torch.float32)[None,:,None,None]
    def target_for(y):
        y=y/scales_tensor
        return torch.cat([torch.asinh(y[:,0:1]),torch.asinh(y[:,0:1]+.25*y[:,1:2])],1)
    def objective(pred,target):
        absolute=F.smooth_l1_loss(pred,target)
        ranking=F.kl_div((pred.flatten(2)/.5).log_softmax(-1),
            (target.flatten(2)/.5).softmax(-1),reduction='batchmean')*.25
        global_error=F.smooth_l1_loss(pred.mean((2,3)),target.mean((2,3)))
        return absolute+.2*ranking+.2*global_error
    best=float('inf');history=[];started=time.monotonic()
    sources={str(path):sha256_file(path) for path in [Path(__file__),HERE.parent/'gain_model.py',HERE/'routing.py',HERE.parent/'runtime.py']}
    for epoch in range(a.epochs):
        model.train();total=0.;n=0
        for x,y in loader:
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True);pred=model(x);loss=objective(pred,target_for(y))
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();total+=float(loss.detach())*len(x);n+=len(x)
        model.eval();h=0.;hn=0
        with torch.no_grad():
            for x,y in validation:
                x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
                h+=float(objective(model(x),target_for(y)))*len(x);hn+=len(x)
        row=dict(epoch=epoch,fit_loss=total/n,held_loss=h/hn,seconds=time.monotonic()-started)
        row['learning_rate']=optimizer.param_groups[0]['lr']
        scheduler.step()
        history.append(row);print(json.dumps(row),flush=True)
        checkpoint=dict(model=model.state_dict(),epoch=epoch,scales=scales.tolist(),source_sha256=sources,
            input='new independent raw features plus whitened cosine; no old scorer or selected set',
            parameters=sum(p.numel() for p in model.parameters()),metrics=row)
        torch.save(checkpoint,a.output/'latest.pt')
        if row['held_loss']<best:
            best=row['held_loss'];torch.save(checkpoint,a.output/'best.pt')
        write_json(a.output/'progress.json',dict(rows=history))
    write_json(a.output/'report.json',dict(status='completed',rows=history,fit_n=n,held_n=hn,
        best_held_loss=best,source_sha256=sources,seconds=time.monotonic()-started,
        legacy_scorer_or_routing_labels_used=False,Inception_training_supervision=False))


if __name__=='__main__':main()
