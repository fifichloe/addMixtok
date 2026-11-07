"""Public MixTok objective/optimizer on six cards, eight logical microbatches.

Only routing, six-card accumulation, local logging and resumable checkpointing
are changed. The tokenizer, adapter and pretrained Halton are unmodified.
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
from torch.utils.data import DataLoader, Sampler
from common import *
from routing import fixed_k, positions
from bert2d.paths import atomic_json, sha256

ACCUMULATION = [2,2,1,1,1,1]

class SixCardSampler(Sampler):
    """Partition the public 256-row global batch into eight 32-row groups."""
    def __init__(self,dataset,rank,seed):
        self.base=module('bucket_sampler').DynamicKBucketBatchSampler(dataset,256,True,True,seed)
        self.start=sum(ACCUMULATION[:rank])*32
        self.stop=self.start+ACCUMULATION[rank]*32
        self.skip=0

    def __len__(self): return len(self.base)-self.skip

    def __iter__(self):
        for index,rows in enumerate(self.base):
            if index>=self.skip:yield rows[self.start:self.stop]

def run(args):
    k = fixed_k(args.policy)
    rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE']); local=int(os.environ['LOCAL_RANK'])
    assert world==6
    torch.set_num_threads(4); torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=device)
    metadata=dist.new_group(backend='gloo',timeout=timedelta(minutes=30))
    torch.manual_seed(args.seed+rank); torch.cuda.manual_seed(args.seed+rank)
    cache_config=json.loads((args.routes/'config.json').read_text())
    cache_summary=json.loads((args.routes/'summary.json').read_text())
    assert cache_config['route_mode']==args.policy and cache_config['fixed_k']==k
    assert cache_summary['mean_k']==k and cache_summary['complete_coverage'] is True
    assert cache_config['scorer_sha256']==sha256(SCORER)
    assert cache_config['routing_sha256']==sha256(HERE/'routing.py')
    assert cache_config['selected_positions']==list(positions(args.policy))
    public=module('train_halton'); helpers=module('halton_upstream')
    upstream_hashes=helpers.verify_pinned_upstream(UPSTREAM)
    # Public dataset verifies complete coverage, metadata and every input-file hash.
    dataset=module('data').ContinuousRouteDataset(SOURCE,SOURCE,args.routes)
    assert len(dataset)==2562334 and np.all(dataset.k_values==k)
    sampler=SixCardSampler(dataset,rank,args.seed)
    steps_per_epoch=len(sampler.base)
    config=dict(style='mixtok_routing_ablation_v1',source_commit=COMMIT,
        route_mode=args.policy,fixed_k=k,mean_2d_tokens=k,content_tokens=k+(32 if args.policy=='uniform' else 0),
        diagnostic_pipeline_tokens=32+k,context_used=args.policy=='uniform',
        seed=args.seed,world=world,micro=32,global_batch=256,accumulation_by_rank=ACCUMULATION,
        loss_scale_per_microbatch=.75,updates=20000,lr=2e-5,warmup=2500,
        lr_schedule='budget-cosine',lr_hold_until=4000,lr_decay_until=20000,min_lr=2e-6,
        weight_decay=.03,betas=[.9,.999],clip_grad_norm=1.,
        loss='public masked-only CE, mean per logical microbatch',
        mask='public half arccos random, half Halton-prefix including full-MASK',
        initialization='official pretrained Halton, new optimizer, step zero',
        source_root=str(SOURCE),route_root=str(args.routes),
        route_meta_sha256=sha256(args.routes/'config.json'),
        route_summary_sha256=sha256(args.routes/'summary.json'),
        adapter_sha256=sha256(ADAPTER),scorer_sha256=sha256(SCORER),
        adapter_step=20000,eval_adapter_checkpoint=str(ADAPTER),eval_scorer_checkpoint=str(SCORER),
        top_p_checkpoint=str(TOP),upstream_root=str(UPSTREAM),
        top_p_condition_audit=dict(config=cache_config,summary=cache_summary),
        top_p_artifact_sha256=module('assets_top_p').verified_top_p_artifact(TOP),
        tokenizer_frozen=True,adapter_frozen=True,scorer_frozen=True,
        eval_router_mode=args.policy,eval_stage2_steps=8,eval_stage1_steps=8,
        eval_cfg_stage1=4.5,eval_cfg_stage2=3.0,fid_during_training=False,
        sampling=dict(order='official Halton filtered to fixed-K selection',steps=8,cfg_w=3.0,
            temperature=1.,exclude_mask_at_sampling=True,route_mode=args.policy,fixed_k=k),
        precision='BF16 Halton/adapter, FP32 base features, TF32 off',
        dataset_rows=len(dataset),steps_per_epoch=steps_per_epoch,
        k_bucket_coverage=sampler.base.coverage(),
        upstream_hashes=upstream_hashes,
        implementation_sha256={name:sha256(HERE/name) for name in ['train.py','common.py','cache_routes.py','routing.py']},
        public_module_sha256={name:sha256(Path(module(name).__file__)) for name in
            ['continuous_halton','halton_upstream','embedding_adapter','adapter_data','data','bucket_sampler','train_halton']})
    if rank==0:
        if args.resume:
            assert json.loads((args.output/'config.json').read_text())==config,'resume configuration changed'
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
        core.load_state_dict(checkpoint['model'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
    objective=DDP(module('continuous_halton').ContinuousObjective(core,UPSTREAM),device_ids=[local],broadcast_buffers=False)
    with torch.random.fork_rng(devices=[local]):
        frozen=module('assets_top_p').TopPFrozenAssets(ASSETS,TOP,'cpu',chunk=32)
        quantizer=frozen.native.quantize.to(device)
        decoder=frozen.shell.latent_decoder.to(device)
        adapter,_=module('embedding_adapter').load_adapter(ADAPTER,device,module('assets_top_p').TOP_P_SHA256,20000)
    make_features=module('adapter_data').make_features
    start_step=0; epoch=0; epoch_batch=0
    if checkpoint is not None:
        start_step=checkpoint['step']
        epoch=checkpoint['cursor']['epoch']; epoch_batch=checkpoint['cursor']['batch_in_epoch']
        rng=checkpoint['rng_states'][rank]
        torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda'],device)
        del checkpoint
    sampler.base.set_epoch(epoch); sampler.skip=epoch_batch
    loader_generator=torch.Generator().manual_seed(92026+rank)
    loader=DataLoader(dataset,batch_sampler=sampler,collate_fn=module('data').collate_continuous,
        num_workers=2,pin_memory=True,persistent_workers=True,generator=loader_generator)
    stream=iter(loader)
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
        save(0)
    started=time.monotonic()
    for step in range(start_step+1,20001):
        tic=time.monotonic()
        try:cpu=next(stream)
        except StopIteration:
            epoch+=1;epoch_batch=0;sampler.skip=0;sampler.base.set_epoch(epoch)
            stream=iter(loader);cpu=next(stream)
        epoch_batch+=1
        assert len(cpu['z1d'])==32*ACCUMULATION[rank] and bool((cpu['k']==k).all())
        assert bool((cpu['route_indices']==torch.tensor(positions(args.policy))[None]).all())
        batch={key:value.to(device,non_blocking=True) for key,value in cpu.items()}
        public.set_budget_cosine_lr(optimizer,step,2e-5,2500,4000,20000,2e-6)
        optimizer.zero_grad(set_to_none=True)
        stats=torch.zeros(5,device=device)
        for micro in range(ACCUMULATION[rank]):
            sl=slice(32*micro,32*(micro+1))
            features=make_features(batch['z1d'][sl],quantizer,decoder)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):context=adapter(features)
            del features
            assert not context.requires_grad and bool(torch.isfinite(context).all())
            sync=objective.no_sync() if micro+1<ACCUMULATION[rank] else nullcontext()
            with sync:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,metrics=objective(context,batch['z2d'][sl],batch['label'][sl],batch['route_indices'][sl],batch['route_valid'][sl])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                (loss*.75).backward()
            stats+=metrics
        norm=torch.nn.utils.clip_grad_norm_(core.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step<=2 or step%20==0 or step==20000:
            dist.all_reduce(stats);torch.cuda.synchronize()
            row=dict(step=step,epoch=epoch+epoch_batch/steps_per_epoch,loss=float(stats[0]/8),
                masked_nll=float(stats[1]/stats[2].clamp_min(1)),mask_ratio=float(stats[3]/8),
                class_dropout=float(stats[4]/8),selected_k=k,global_batch=256,
                lr=optimizer.param_groups[0]['lr'],grad_norm=float(norm),
                seconds_per_step=time.monotonic()-tic,elapsed_seconds=time.monotonic()-started,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            atomic_json(args.output/f'rank{rank}.json',row)
            if rank==0:
                with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
        if step%2000==0 or epoch_batch==steps_per_epoch or step==20000:save(step)
    frozen.assert_frozen()
    assert not any(p.requires_grad or p.grad is not None for p in adapter.parameters())
    dist.barrier()
    if rank==0:
        receipt=json.loads((args.output/'latest.json').read_text())
        assert receipt['step']==20000 and receipt['round_trip_exact']
        atomic_json(args.output/'training_complete.json',dict(status='complete',step=20000,policy=args.policy,
            fixed_k=k,checkpoint_sha256=receipt['sha256'],fid_evaluated=False))
    dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--policy',choices=['uniform','dense'],required=True)
    p.add_argument('--routes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    try:run(args)
    except BaseException as error:
        if args.output.exists():atomic_json(args.output/f'failure_rank{os.environ.get("RANK","unknown")}.json',dict(type=type(error).__name__,message=str(error)))
        raise
