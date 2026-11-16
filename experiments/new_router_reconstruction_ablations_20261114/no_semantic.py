"""RGB-only gain training and B0-free deployment; explicit new-training control."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ablation_common import HERE, OLD, FRESH, PAPER, sha, write_json
sys.path.insert(0,str(OLD))
from gain_model import GainScorer
from fit_dense_gain import Data


class RGBData(Data):
    def __getitem__(self,index):
        x,y=super().__getitem__(index);x=x.clone();x[774:778]=0
        # Ignore the semantic label even though the archival cache contains it.
        return x,y[:1]


class NoSemanticInputs:
    """No B0 is instantiated or executed. Same BF16 codec and non-semantic maps."""
    def __init__(self):
        sys.path[:0]=[str(PAPER/'experiments/dynamic_recon_20261108'),str(PAPER/'experiments/light_grid_105_20261109')]
        from cache_train_counterfactual import load_model, EXPECTED_CHECKPOINT_SHA256
        from lookup_budget_screen import config
        cfg=config();assert sha(cfg.checkpoint)==EXPECTED_CHECKPOINT_SHA256
        self.model,self.ev,self.crop,self.meta=load_model(cfg,torch.device('cuda'));del self.model.router
        for module in self.model.modules():
            for key,value in list(module._buffers.items()):
                if value is not None and value.requires_grad:
                    assert key=='codebook_used';module._buffers[key]=value.detach()
        self.versions=[(t,t._version) for t in [*self.model.parameters(),*self.model.buffers()]]

    @torch.no_grad()
    def prepare(self,images):
        target=images*2-1
        with torch.autocast('cuda',dtype=torch.bfloat16):
            base,extra=self.model(images);f1=extra['f_1d_lg']
            f2,_=self.ev.native_llamagen_feature(self.model.llamagen_vq,target,8,allow_encoder_grad=False)
        from routing import standard
        pool=lambda v:F.adaptive_avg_pool2d(v,16)
        cosine=1-F.cosine_similarity(f1.float(),f2.float(),dim=1).unsqueeze(1)
        error=target-base.float();rgb_mse=pool(error.square().mean(1,keepdim=True));zero=torch.zeros_like(cosine)
        maps=[cosine,pool(error.abs().mean(1,keepdim=True)),rgb_mse,
              torch.log1p((f2.float()-f1.float()).square().mean(1,keepdim=True)),
              torch.log1p(f1.float().square().mean(1,keepdim=True)),torch.log1p(f2.float().square().mean(1,keepdim=True)),
              zero,zero,zero,zero,standard(rgb_mse),standard(cosine)]
        scale=f1.shape[1]**.5
        features=torch.cat([F.normalize(f1.float(),dim=1)*scale,F.normalize(f2.float(),dim=1)*scale,
                            F.normalize((f2-f1).float(),dim=1)*scale,*maps],1)
        assert features.shape[1:]==(780,16,16) and not features[:,774:778].count_nonzero()
        return dict(f1=f1,f2=f2,features=features,target=target,base=base,independent_prior=zero)

    def decode(self,z,mask):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            m=mask.to(z['f1'].dtype);return self.model.llamagen_vq.decoder((1-m)*z['f1']+m*z['f2']).float()

    def audit(self):
        assert all(t._version==v and not t.requires_grad and t.grad is None for t,v in self.versions)
        return dict(no_semantic_network_instantiated=True,no_semantic_features=True,no_semantic_targets=True,
                    codec_frozen=True,legacy_router_deleted=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['short','long'],required=True)
    p.add_argument('--cache',type=Path,nargs='+',default=[OLD/f'dense_gain_{i}' for i in range(4)])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20261112);a=p.parse_args()
    epochs=12 if a.kind=='short' else 60
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    fit=RGBData(a.cache,'fit');held=RGBData(a.cache,'held')
    assert len(fit)==10843 and len(held)==1157 and not set(fit.paths).intersection(held.paths)
    raw=np.concatenate([np.asarray(fit.y[slot][[i for s,i in fit.samples if s==slot],:1]) for slot in range(len(fit.y))])
    scale=float(np.quantile(np.abs(raw),.75).clip(1e-6))
    a.output.mkdir(parents=True,exist_ok=False);model=GainScorer().cuda()
    # Normalizers only, as in the original fresh gain training; no head weights reused.
    normal=torch.load(FRESH/'heads/rgb0.5_best.pt',map_location='cpu',weights_only=False)['model']
    model.input_mean[:,:780].copy_(normal['input_mean'].cuda());model.input_std[:,:780].copy_(normal['input_std'].cuda())
    model.input_mean[:,774:778]=0;model.input_std[:,774:778]=1
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5) if a.kind=='long' else None
    train=DataLoader(fit,batch_size=64,shuffle=True,num_workers=2,pin_memory=True)
    validation=DataLoader(held,batch_size=64,num_workers=2,pin_memory=True)
    def objective(pred,y):
        target=torch.asinh(y/scale).expand(-1,2,-1,-1)
        absolute=F.smooth_l1_loss(pred,target)
        ranking=F.kl_div((pred.flatten(2)/.5).log_softmax(-1),(target.flatten(2)/.5).softmax(-1),reduction='batchmean')*.25
        return absolute+.2*ranking+.2*F.smooth_l1_loss(pred.mean((2,3)),target.mean((2,3)))
    best=float('inf');rows=[];started=time.monotonic()
    sources={str(path):sha(path) for path in [Path(__file__),OLD/'gain_model.py',OLD/'fit_dense_gain.py']}
    cache_files=set()
    for folder in a.cache:
        report=json.loads((folder/'report.json').read_text())
        cache_files.update([folder/'report.json',folder/'paths.txt',folder/'gain.npy',folder/'white.npy',
                            Path(report['source_inputs']),Path(report['source_indices'])])
    cache_sha256={str(path):sha(path) for path in sorted(cache_files)}
    for epoch in range(epochs):
        model.train();total=0.;n=0
        for x,y in train:
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
            loss=objective(model(x),y);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step();total+=float(loss.detach())*len(x);n+=len(x)
        model.eval();held_loss=0.;hn=0
        with torch.no_grad():
            for x,y in validation:
                x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);held_loss+=float(objective(model(x),y))*len(x);hn+=len(x)
        row=dict(epoch=epoch+1,fit_loss=total/n,held_loss=held_loss/hn,learning_rate=opt.param_groups[0]['lr'],seconds=time.monotonic()-started)
        if scheduler is not None:scheduler.step()
        rows.append(row);print(json.dumps(row),flush=True)
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,kind=a.kind,RGB_gain_scale=scale,
                   source_sha256=sources,metrics=row,seed=a.seed,semantic_channels_zero=[774,775,776,777],semantic_target_used=False,
                   initialization='fresh GainScorer, normalization statistics only reused',matched_original_epochs=epochs,
                   cache_sha256=cache_sha256)
        torch.save(state,a.output/'latest.pt')
        if row['held_loss']<best:best=row['held_loss'];torch.save(state,a.output/'best.pt')
        write_json(a.output/'progress.json',dict(rows=rows))
    write_json(a.output/'report.json',dict(status='no_semantic_training_complete',kind=a.kind,epochs=epochs,steps=epochs*len(train),rows=rows,
        fit_n=len(fit),held_n=len(held),best_held_loss=best,source_sha256=sources,
        held_selection='RGB-only training held objective; never validation rFID',validation_images_used=0,
        semantic_input_used=False,semantic_label_used=False,B0_executed=False,Inception_executed=False,
        cache_sha256=cache_sha256,
        cache_note='archival source contains B0 channels/labels, but these are excluded before model or target use'))


if __name__=='__main__':main()
