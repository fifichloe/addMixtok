"""Four-card execution of the original global-256, 20k Halton fine-tuning recipe.

Each rank accumulates two original 32-image microbatches. DDP averages four
ranks, so each microbatch is weighted 1/2, retaining eight logical groups.
Only the frozen route policy differs between the two formal runs.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Dataset
from common import *
from routing import control_identity
from bert2d.paths import atomic_json, sha256


class FourCardSampler(Sampler):
    def __init__(self,dataset,rank,seed):
        self.base=module('bucket_sampler').DynamicKBucketBatchSampler(dataset,256,True,True,seed)
        self.rank=rank;self.skip=0

    def __len__(self):return len(self.base)-self.skip

    def __iter__(self):
        for index,rows in enumerate(self.base):
            if index>=self.skip:yield rows[self.rank*64:(self.rank+1)*64]


class ProfileDataset(Dataset):
    """Real paired train codes, maximum-K stress, not a formal route cache."""
    def __init__(self):
        self.z1=np.load(SOURCE/'titok_codes.npy',mmap_mode='r')
        self.z2=np.load(SOURCE/'llamagen_codes.npy',mmap_mode='r')
        self.labels=np.load(SOURCE/'labels.npy',mmap_mode='r')
        self.rows=np.random.default_rng(20261114).choice(2562334,8192,replace=False)
        self.k_values=np.full(len(self.rows),144,dtype=np.uint16)

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        src,view=divmod(int(self.rows[index]),2)
        # Actual scorer ranking is applied on GPU before each profile objective.
        return dict(z1d=torch.from_numpy(np.array(self.z1[src,view],dtype=np.int64)),
            z2d=torch.from_numpy(np.array(self.z2[src,view,:144],dtype=np.int64)),
            full_z2d=torch.from_numpy(np.array(self.z2[src,view],dtype=np.int64)),
            route_indices=torch.arange(144),k=torch.tensor(144),
            label=torch.tensor(int(self.labels[src])),source_index=torch.tensor(src),augmentation=torch.tensor(view))


def run(args):
    rank,world,local=(int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK'])
    assert world==4
    torch.set_num_threads(4);torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=device)
    metadata=dist.new_group(backend='gloo',timeout=timedelta(minutes=30))
    torch.manual_seed(args.seed+rank);torch.cuda.manual_seed(args.seed+rank)
    public=module('train_halton');helpers=module('halton_upstream')
    upstream_hashes=helpers.verify_pinned_upstream(UPSTREAM)
    identity=router_identity(args.kind)
    if args.profile:
        dataset=ProfileDataset();cache_config=None;cache_summary=None
    else:
        cache_config=json.loads((args.routes/'config.json').read_text())
        cache_summary=json.loads((args.routes/'summary.json').read_text())
        assert cache_config['router_identity']==identity
        assert cache_summary['complete_coverage'] and cache_summary['route_mode']=='control_'+args.policy
        assert cache_config['control_identity']==control_identity(args.kind,args.policy,args.target_mean_k,args.route_seed,args.train_target_mean_k)
        dataset=module('data').ContinuousRouteDataset(SOURCE,SOURCE,args.routes)
        assert len(dataset)==2562334 and np.all((dataset.k_values>=48)&(dataset.k_values<=144))
    sampler=FourCardSampler(dataset,rank,args.seed)
    updates=args.profile_steps if args.profile else 20000
    config=dict(style='mixtok_new_router_control_halton_v1',kind=args.kind,policy=args.policy,seed=args.seed,
        control_identity=control_identity(args.kind,args.policy,args.target_mean_k,args.route_seed,args.train_target_mean_k),
        world=4,physical_gpus=GPUS,micro=32,global_batch=256,accumulation_by_rank=[2]*4,
        loss_scale_per_microbatch=.5,updates=updates,formal_updates=20000,profile=args.profile,
        lr=2e-5,warmup=2500,lr_schedule='budget-cosine',lr_hold_until=4000,
        lr_decay_until=20000,min_lr=2e-6,weight_decay=.03,betas=[.9,.999],clip_grad_norm=1.,
        loss='original masked-only CE; mean over eight logical 32-image microbatches',
        mask='original half arccos random, half Halton-prefix, including full MASK',
        initialization='official pretrained Halton; fresh AdamW; step zero',
        source_root=str(SOURCE),route_root=str(args.routes),
        router_identity=identity,route_config=cache_config,route_summary=cache_summary,
        adapter_sha256=sha256(ADAPTER),adapter_step=20000,adapter_reused=True,
        adapter_reuse_reason='same 661 codec tensors; adapter targets/inputs independent of routing',
        eval_adapter_checkpoint=str(ADAPTER),top_p_checkpoint=str(TOP),
        top_p_artifact_sha256=module('assets_top_p').verified_top_p_artifact(TOP),
        upstream_root=str(UPSTREAM),pretrained_sha256=helpers.PRETRAINED_SHA256,
        upstream_hashes=upstream_hashes,tokenizer_frozen=True,adapter_frozen=True,scorer_frozen=True,
        eval_stage1_steps=8,eval_stage2_steps=8,eval_cfg_stage1=4.5,eval_cfg_stage2=3.,
        fid_during_training=False,legacy_scorer_used=False,thresholds_recalibrated=False,
        precision='Halton/adapter BF16; original adapter Fbase FP32; new router Fbase BF16, scorer FP32; TF32 off',
        dataset_rows=len(dataset),steps_per_epoch=len(sampler.base),k_bucket_coverage=sampler.base.coverage(),
        implementation_sha256={n:sha256(HERE/n) for n in ['train.py','common.py','cache_routes.py','routing.py']},
        original_modules_sha256={n:sha256(Path(module(n).__file__)) for n in
            ['continuous_halton','halton_upstream','embedding_adapter','adapter_data','data','bucket_sampler','train_halton']})
    if rank==0:
        if args.resume:
            assert json.loads((args.output/'config.json').read_text())==config
        else:
            args.output.mkdir(parents=True,exist_ok=False)
            atomic_json(args.output/'config.json',config)
    dist.barrier()
    core=module('continuous_halton').ContinuousHalton(UPSTREAM)
    init=helpers.load_pretrained(core,PRETRAINED,True,verify_hash=(rank==0))
    core.to(device).train()
    optimizer=helpers.official_training_api(UPSTREAM,core,2e-5,2500,20000).optim
    checkpoint=None
    if args.resume:
        receipt=json.loads((args.output/'latest.json').read_text())
        assert sha256(args.output/'latest.pt')==receipt['sha256']
        checkpoint=torch.load(args.output/'latest.pt',map_location='cpu',weights_only=True)
        assert checkpoint['config']==config and checkpoint['step']==receipt['step']
        core.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer'])
    objective=DDP(module('continuous_halton').ContinuousObjective(core,UPSTREAM),device_ids=[local],broadcast_buffers=False)
    with torch.random.fork_rng(devices=[local]):
        frozen=module('assets_top_p').TopPFrozenAssets(ASSETS,TOP,'cpu',chunk=32)
        q=frozen.native.quantize.to(device);decoder=frozen.shell.latent_decoder.to(device)
        adapter,_=module('embedding_adapter').load_adapter(ADAPTER,device,module('assets_top_p').TOP_P_SHA256,20000)
        scorer,_=load_router(args.kind,device)
    make_features=module('adapter_data').make_features
    start_step=0;epoch=0;epoch_batch=0
    if checkpoint is not None:
        start_step=checkpoint['step'];epoch=checkpoint['cursor']['epoch'];epoch_batch=checkpoint['cursor']['batch_in_epoch']
        rng=checkpoint['rng_states'][rank]
        torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'],device)
        del checkpoint
    sampler.base.set_epoch(epoch);sampler.skip=epoch_batch
    loader=DataLoader(dataset,batch_sampler=sampler,collate_fn=module('data').collate_continuous,
        num_workers=2,pin_memory=True,persistent_workers=True,
        generator=torch.Generator().manual_seed(92026+rank))
    stream=iter(loader)
    frozen_versions=[(v,v._version) for m in [q,decoder,adapter,scorer] for v in m.parameters()]
    def save(step):
        states=[None]*world if rank==0 else None
        dist.gather_object(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device)),states,dst=0,group=metadata)
        if rank==0:
            public.save_checkpoint(args.output,core,optimizer,step,config,
                dict(epoch=epoch,batch_in_epoch=epoch_batch),states)
            receipt=json.loads((args.output/'latest.json').read_text())
            atomic_json(args.output/f'checkpoint_step{step}.json',receipt)
        dist.barrier()
    if not args.resume:
        if rank==0:atomic_json(args.output/'initialization.json',init)
        # Avoid a redundant untrained 1.6-GB checkpoint; initialization is hash-bound.
    started=time.monotonic();ticks=[]
    for step in range(start_step+1,updates+1):
        tic=time.monotonic()
        try:cpu=next(stream)
        except StopIteration:
            epoch+=1;epoch_batch=0;sampler.skip=0;sampler.base.set_epoch(epoch)
            stream=iter(loader);cpu=next(stream)
        epoch_batch+=1
        assert len(cpu['z1d'])==64
        batch={key:value.to(device,non_blocking=True) for key,value in cpu.items()}
        public.set_budget_cosine_lr(optimizer,step,2e-5,2500,4000,20000,2e-6)
        optimizer.zero_grad(set_to_none=True);stats=torch.zeros(7,device=device)
        for micro in range(2):
            sl=slice(32*micro,32*(micro+1))
            features=make_features(batch['z1d'][sl],q,decoder)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):context=adapter(features)
            del features
            indices=batch['route_indices'][sl];gt=batch['z2d'][sl];valid=batch['route_valid'][sl]
            if args.profile:
                base=router_features(batch['z1d'][sl],q,decoder)
                output=scorer(base.float())
                indices=output['score'].argsort(dim=1,descending=True,stable=True)[:,:144].sort(1).values
                gt=batch['full_z2d'][sl].gather(1,indices)
                del base,output
            sync=objective.no_sync() if micro==0 else nullcontext()
            with sync:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,metrics=objective(context,gt,batch['label'][sl],indices,valid)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite Halton training loss')
                (loss*.5).backward()
            stats[:5]+=metrics;stats[5]+=batch['k'][sl].sum();stats[6]+=len(gt)
        norm=torch.nn.utils.clip_grad_norm_(core.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        elapsed=time.monotonic()-tic;ticks.append(elapsed)
        if step<=2 or step%20==0 or step==updates:
            dist.all_reduce(stats)
            row=dict(step=step,kind=args.kind,policy=args.policy,epoch=epoch+epoch_batch/len(sampler.base),
                loss=float(stats[0]/8),masked_nll=float(stats[1]/stats[2].clamp_min(1)),
                mask_ratio=float(stats[3]/8),class_dropout=float(stats[4]/8),
                mean_K=float(stats[5]/stats[6]),global_batch=256,lr=optimizer.param_groups[0]['lr'],
                grad_norm=float(norm),seconds_per_step=elapsed,
                recent_mean_seconds=float(np.mean(ticks[-100:])),
                elapsed_seconds=time.monotonic()-started,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,pid=os.getpid(),status='running')
            atomic_json(args.output/f'rank{rank}.json',row)
            if rank==0:
                with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
        if (not args.profile and (step%2000==0 or epoch_batch==len(sampler.base))) or step==updates:save(step)
    assert all(not v.requires_grad and v.grad is None and v._version==version for v,version in frozen_versions)
    frozen.assert_frozen()
    assert all(int(state['step'])==updates for state in optimizer.state.values())
    # Actual restore of checkpoint model+optimizer verifies the resumable state.
    receipt=json.loads((args.output/'latest.json').read_text())
    loaded=torch.load(args.output/'latest.pt',map_location='cpu',mmap=True,weights_only=True)
    core.load_state_dict(loaded['model'],strict=True);optimizer.load_state_dict(loaded['optimizer'])
    assert all(int(state['step'])==updates for state in optimizer.state.values())
    dist.barrier()
    if rank==0:
        atomic_json(args.output/'training_complete.json',dict(status='complete',kind=args.kind,policy=args.policy,
            step=updates,profile=args.profile,initialization=init,global_batch=256,
            model_optimizer_restore_passed=True,all_frozen_components_unchanged=True,
            checkpoint_sha256=receipt['sha256'],seconds=time.monotonic()-started,
            steady_seconds_per_step=float(np.mean(ticks[min(5,len(ticks)-1):])),
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            fid_evaluated=False))
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=KINDS,required=True)
    p.add_argument('--policy',choices=POLICIES,required=True)
    p.add_argument('--target-mean-k',type=float,required=True)
    p.add_argument('--train-target-mean-k',type=float,required=True)
    p.add_argument('--route-seed',type=int,default=20261114)
    p.add_argument('--routes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--profile-steps',type=int,default=20)
    args=p.parse_args()
    try:run(args)
    except BaseException as e:
        if args.output.exists():atomic_json(args.output/f'failure_rank{os.environ.get("RANK","unknown")}.json',
            dict(type=type(e).__name__,message=str(e)))
        raise
