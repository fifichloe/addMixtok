"""Student-only scan and reconstruction evaluation, with a frozen codec.

No reconstruction teacher, B0, original router or ground-truth-dependent
signals enter the student's scoring/offset forward pass.
"""
from pathlib import Path
import argparse, json, math, sys, time, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from policies import HERE, ROOT, POLICIES, sha, write_json
from model import GenerationRouter

from mixtok_long_no_density.recon.io import load_codec as load_model, ImagePathDataset, feature_tensor
from policies import codec_config as config
EXPECTED_CHECKPOINT_SHA256='457be008836e3f3702327795bb1953250f0032a0b35f74e26e6ffce9ef50e82e'
VAL_PATHS=Path(os.environ.get('MIXTOK_VALIDATION_MANIFEST','validation_files.txt'))
INC=Path(os.environ.get('MIXTOK_INCEPTION_WEIGHTS','weights-inception-2015-12-05-6726825d.pth'))

class Codec:
    def __init__(self):
        cfg=config();assert sha(cfg.checkpoint)==EXPECTED_CHECKPOINT_SHA256
        self.model,self.ev,self.crop,self.meta=load_model(cfg,torch.device('cuda'))
        del self.model.router
        assert not any('router' in name for name,_ in self.model.named_modules())
        for name,module in self.model.named_modules():
            for key,tensor in list(module._buffers.items()):
                if tensor is not None and tensor.requires_grad:
                    assert key=='codebook_used'
                    module._buffers[key]=tensor.detach()
        self.versions=[(name,p,p._version) for name,p in
                       [*self.model.named_parameters(),*self.model.named_buffers()]]

    def audit(self):
        assert all(not p.requires_grad and p.grad is None and p._version==version for _,p,version in self.versions)
        return dict(codec_parameters_and_buffers_frozen=True,original_router_deleted=True,
            B0_loaded=False,ground_truth_or_2D_features_enter_student=False)

    @torch.no_grad()
    def base(self,images):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,extra=self.model(images)
        return extra['f_1d_lg']

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['scan','quality'],required=True)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,required=True)
    p.add_argument('--frozen',type=Path);p.add_argument('--limit',type=int)
    args=p.parse_args();assert 0<=args.shard<args.shards
    torch.set_num_threads(4);torch.manual_seed(20261113)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    paths=VAL_PATHS.read_text().splitlines()
    assert len(paths)==len(set(paths))==50000
    assert all('/validation/' in v or '/val/' in v for v in paths)
    if args.limit is not None:paths=paths[:args.limit]
    batches=np.array_split(np.arange(math.ceil(len(paths)/32)),args.shards)[args.shard]
    start=int(batches[0])*32;end=min((int(batches[-1])+1)*32,len(paths));selected=paths[start:end];n=len(selected)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'paths.txt').write_text('\n'.join(selected)+'\n')
    codec=Codec();students={};weight_hash={};step={};training_calibration={}
    for kind,policy in POLICIES.items():
        path=args.training/kind/'best.pt';weight_hash[str(path)]=sha(path)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['config']['kind']==kind and saved['config']['input']=='f_base_only_Bx256x16x16'
        student=GenerationRouter(policy['weight'],policy['threshold']).cuda().eval().requires_grad_(False)
        student.load_state_dict(saved['model']);students[kind]=student;step[kind]=saved['step']
        training_calibration[kind]=saved['calibration']
    student_versions=[(kind,name,p,p._version) for kind,m in students.items() for name,p in m.named_parameters()]
    sources={str(path):sha(path) for path in [Path(__file__),HERE/'model.py',HERE/'policies.py',VAL_PATHS]}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename and str(filename).endswith('.py'):
            path=Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.is_file():
                sources[str(path)]=sha(path)
    for path in [config().checkpoint,config().titok_ckpt,config().llamagen_ckpt]:weight_hash[str(path)]=sha(path)
    raw={kind:np.lib.format.open_memmap(args.output/(kind+'.scores.npy'),mode='w+',dtype='float32',shape=(n,256)) for kind in students}
    if args.mode=='quality':
        assert args.frozen is not None
        frozen=json.loads(args.frozen.read_text())
        assert frozen['quality_used_for_thresholds'] is False and frozen['n']==len(paths)
        assert frozen['weight_sha256']==weight_hash
        for source,digest in frozen['source_sha256'].items():assert sha(source)==digest,source
        from torchmetrics.image.fid import NoTrainInceptionV3
        inc=NoTrainInceptionV3(name='inception-v3-compat',features_list=['2048'],
            feature_extractor_weights_path=str(INC)).cuda().eval().requires_grad_(False)
        sources[str(INC)]=sha(INC)
        real=np.lib.format.open_memmap(args.output/'real.npy',mode='w+',dtype='float32',shape=(n,2048))
        fake={kind:np.lib.format.open_memmap(args.output/(kind+'.features.npy'),mode='w+',dtype='float32',shape=(n,2048)) for kind in students}
        masks={kind:np.lib.format.open_memmap(args.output/(kind+'.masks.npy'),mode='w+',dtype='uint8',shape=(n,256)) for kind in students}
        budgets={kind:np.lib.format.open_memmap(args.output/(kind+'.K.npy'),mode='w+',dtype='int16',shape=(n,)) for kind in students}
        psnr={kind:0. for kind in students}
    checks=[];done=0;started=time.monotonic()
    loader=DataLoader(ImagePathDataset(list(map(Path,selected)),codec.crop),batch_size=32,num_workers=4,pin_memory=True)
    with torch.no_grad():
        for images in loader:
            images=images.cuda(non_blocking=True);base=codec.base(images);b=len(images)
            # FP32 scoring avoids BF16 score ties/threshold jitter when callers
            # use different student batch sizes. The frozen codec stays BF16.
            outputs={kind:m(base.float()) for kind,m in students.items()}
            if done==0:
                reverse={kind:m(base.flip(0).float()) for kind,m in students.items()}
                single={kind:m(base[:1].float()) for kind,m in students.items()}
                for kind in students:
                    assert torch.allclose(outputs[kind]['score'],reverse[kind]['score'].flip(0),atol=1e-4,rtol=1e-4)
                    assert torch.allclose(outputs[kind]['score'][:1],single[kind]['score'],atol=1e-4,rtol=1e-4)
            if args.mode=='quality':
                target=images*2-1
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    local,_=codec.ev.native_llamagen_feature(codec.model.llamagen_vq,target,8,allow_encoder_grad=False)
                real[done:done+b]=feature_tensor(inc,codec.ev.to_uint8(target,'minus1_1'))
            for kind,student in students.items():
                output=outputs[kind];raw[kind][done:done+b]=output['score'].cpu().numpy()
                if args.mode=='quality':
                    threshold=frozen['policies'][kind]['threshold']
                    chosen=student.select_output(output,threshold)
                    if done==0:
                        r=student.select_output(reverse[kind],threshold);s=student.select_output(single[kind],threshold)
                        assert torch.equal(chosen['K'],r['K'].flip(0)) and torch.equal(chosen['mask'],r['mask'].flip(0))
                        assert torch.equal(chosen['K'][:1],s['K']) and torch.equal(chosen['mask'][:1],s['mask'])
                        checks.append(dict(kind=kind,same_activations_singleton_reverse_masks_K_exact=True))
                    m=chosen['mask'].reshape(-1,1,16,16).to(base.dtype)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        reconstruction=codec.model.llamagen_vq.decoder((1-m)*base+m*local).float()
                    fake[kind][done:done+b]=feature_tensor(inc,codec.ev.to_uint8(reconstruction,'minus1_1'))
                    masks[kind][done:done+b]=chosen['mask'].cpu().numpy().astype('uint8')
                    budgets[kind][done:done+b]=chosen['K'].cpu().numpy()
                    psnr[kind]+=float(codec.ev.psnr_sum(reconstruction,target,'minus1_1'))
            done+=b
            if done%320==0 or done==n:
                row=dict(mode=args.mode,shard=args.shard,done=done,total=n,seconds=time.monotonic()-started)
                write_json(HERE/f'quality_{args.mode}_progress_{args.shard}.json',row)
                print(json.dumps(row),flush=True)
    for array in raw.values():array.flush()
    assert all(not p.requires_grad and p.grad is None and p._version==version for _,_,p,version in student_versions)
    result=dict(status='completed',mode=args.mode,shard=args.shard,shards=args.shards,n=n,start=start,end=end,
        validation_paths_sha256=sha(VAL_PATHS),paths_sha256=sha(args.output/'paths.txt'),weight_sha256=weight_hash,
        source_sha256=sources,student_steps=step,training_calibration=training_calibration,
        one_student_scoring_call_per_image=True,selection_feedback=False,student_input='base features only',
        students_frozen=True,student_deployment_precision='FP32',codec_precision='BF16 autocast',
        seconds=time.monotonic()-started,**codec.audit())
    if args.mode=='quality':
        real.flush()
        for arrays in [fake,masks,budgets]:
            for array in arrays.values():array.flush()
        result.update(PSNR_sums=psnr,checks=checks,frozen_sha256=sha(args.frozen),
            budgets_from_live_student_scores=True,Inception_used_for_evaluation_only=True)
    else:result.update(Inception_loaded=False,quality_metrics_computed=False)
    write_json(args.output/'report.json',result)

if __name__=='__main__':main()
