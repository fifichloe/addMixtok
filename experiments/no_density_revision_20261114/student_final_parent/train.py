"""Train each base-only student from the full, shared, immutable teacher cache."""
from pathlib import Path
import argparse, json, os, random, time, math, shutil
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from policies import HERE, POLICIES, sha, write_json
from model import GenerationRouter, distillation_loss

class Shards(IterableDataset):
    def __init__(self, paths, batch, rank=0, world=1, infinite=True, streaming=False):
        self.paths, self.batch, self.rank, self.world, self.infinite = paths, batch, rank, world, infinite
        self.streaming=streaming

    def __iter__(self):
        info = get_worker_info(); worker = info.id if info else 0; workers = info.num_workers if info else 1
        epoch = 0
        while True:
            paths = list(self.paths)
            if not self.streaming or epoch>0:
                random.Random(20261113+epoch).shuffle(paths)
            owned = paths[self.rank::self.world][worker::workers]
            if self.infinite and not owned:
                raise RuntimeError('Too few cache shards for the requested GPU/data-worker count')
            for path in owned:
                # Writers atomically rename completed files. Wait for the next
                # prescribed full-manifest shard; never substitute an earlier
                # subset or sample only whichever chunks happen to be ready.
                while self.streaming and not path.exists():time.sleep(1)
                # Read a complete modest-sized chunk sequentially into RAM.
                # Random mmap indexing causes many small remote NFS reads.
                data = torch.load(path, map_location='cpu', weights_only=False, mmap=False)
                assert data['policies'] == POLICIES, 'Teacher labels do not belong to the new frozen policies'
                n = len(data['f_base'])
                order = torch.randperm(n, generator=torch.Generator().manual_seed(20261113+epoch*99991+int(path.stem)))
                for start in range(0, n, self.batch):
                    ix = order[start:start+self.batch]
                    if self.infinite and len(ix) < self.batch: continue
                    yield {key: data[key][ix] for key in ['f_base', 'local', 'utility']}
            if not self.infinite: return
            epoch += 1

@torch.no_grad()
def evaluate(module, loader, index, policy):
    module.eval(); n = 0; losses = 0.; intersection = union = 0
    pred_k, true_k, pred_u, true_u, scores, true_masks = [], [], [], [], [], []
    for cpu in loader:
        data = {key:value.cuda(non_blocking=True) for key,value in cpu.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16): output = module(data['f_base'])
        local, utility = data['local'][:, index], data['utility']
        loss, _ = distillation_loss(output, local, utility, policy['weight'], policy['threshold'])
        selected = module.select_output(output, policy['threshold'])
        target = module.select_output(dict(score=local+policy['weight']*utility[:,None]), policy['threshold'])
        intersection += int((selected['mask'] & target['mask']).sum())
        union += int((selected['mask'] | target['mask']).sum())
        b = len(utility); n += b; losses += float(loss)*b
        pred_k.extend(selected['K'].cpu().tolist()); true_k.extend(target['K'].cpu().tolist())
        pred_u.extend(output['utility'].cpu().tolist()); true_u.extend(utility.cpu().tolist())
        scores.append(output['score'].cpu().numpy())
        true_masks.append(target['mask'].cpu().numpy())
    def corr(a,b): return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
    metrics = dict(n=n, loss=losses/n, mask_IoU=intersection/union,
        K_MAE=float(np.abs(np.array(pred_k)-true_k).mean()), K_correlation=corr(pred_k,true_k),
        mean_K=float(np.mean(pred_k)), std_K=float(np.std(pred_k)), teacher_mean_K=float(np.mean(true_k)),
        utility_MAE=float(np.abs(np.array(pred_u)-true_u).mean()), utility_correlation=corr(pred_u,true_u))
    score_array=np.concatenate(scores)
    calibration=calibrate(score_array)
    calibrated_k=np.clip((score_array>=calibration['threshold']).sum(1),48,144)
    order=np.argsort(-score_array,axis=1,kind='stable')
    calibrated_mask=np.zeros_like(score_array,dtype=bool)
    np.put_along_axis(calibrated_mask,order,np.arange(256)[None]<calibrated_k[:,None],axis=1)
    target_masks=np.concatenate(true_masks)
    metrics['calibrated']=dict(mean_K=float(calibrated_k.mean()),std_K=float(calibrated_k.std()),
        K_MAE=float(np.abs(calibrated_k-true_k).mean()),K_correlation=corr(calibrated_k,true_k),
        mask_IoU=float((calibrated_mask & target_masks).sum()/(calibrated_mask | target_masks).sum()))
    module.train()
    return metrics, score_array

def calibrate(scores):
    # One scalar fixed using held TRAINING images only, without quality metrics.
    lo, hi = float(scores.min())-1, float(scores.max())+1
    best = (float('inf'), None, None)
    for _ in range(64):
        tau = float(np.float32((lo+hi)/2)); mean = float(np.clip((scores>=tau).sum(1),48,144).mean())
        candidate = (abs(mean-96), tau, mean)
        if candidate[0] < best[0]: best = candidate
        if mean > 96: lo = tau
        else: hi = tau
    return dict(threshold=best[1], held_mean_K=best[2], target_K=96,
        source='held_training_images_only', quality_metrics_used=False)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--kind', choices=list(POLICIES), required=True)
    p.add_argument('--cache', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=20000); p.add_argument('--batch', type=int, default=128)
    p.add_argument('--workers', type=int, default=2); p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--streaming-cache', action='store_true')
    p.add_argument('--expected-fit-chunks',type=int,default=621)
    p.add_argument('--memory-fraction',type=float)
    p.add_argument('--local-metrics',action='store_true')
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    rank, world, local = (int(os.environ.get(k,d)) for k,d in [('RANK','0'),('WORLD_SIZE','1'),('LOCAL_RANK','0')])
    torch.cuda.set_device(local); torch.set_num_threads(4); torch.manual_seed(20261113)
    if args.memory_fraction is not None:torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if world > 1: dist.init_process_group('nccl')
    policy = POLICIES[args.kind]; index = list(POLICIES).index(args.kind)
    model = GenerationRouter(policy['weight'], policy['threshold']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    step, best = 0, float('inf')
    if args.resume and (args.output/'latest.pt').exists():
        saved = torch.load(args.output/'latest.pt', map_location='cpu', weights_only=False)
        assert saved['config']['kind'] == args.kind and saved['config']['steps'] == args.steps
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        step, best = saved['step'], saved['best_held_loss']
    student = DDP(model, device_ids=[local], broadcast_buffers=False) if world>1 else model
    fit = ([args.cache/'fit'/f'{i:05d}.pt' for i in range(args.expected_fit_chunks)]
        if args.streaming_cache else sorted((args.cache/'fit').glob('*.pt')))
    held = sorted((args.cache/'held').glob('*.pt'))
    assert fit and held
    loader = DataLoader(Shards(fit,args.batch,rank,world,streaming=args.streaming_cache), batch_size=None, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers>0, prefetch_factor=2 if args.workers>0 else None)
    held_loader = DataLoader(Shards(held,args.batch,infinite=False), batch_size=None, num_workers=2,
        pin_memory=True, persistent_workers=True) if rank == 0 else None
    config = dict(kind=args.kind, policy=policy, steps=args.steps, micro_batch=args.batch,
        world_size=world, global_batch=args.batch*world, parameters=sum(p.numel() for p in model.parameters()),
        input='f_base_only_Bx256x16x16', fresh_initialization=True, direct_K_head=False,
        optimizer='AdamW lr=2e-4 wd=.01 constant', autocast='bfloat16',
        streaming_cache=args.streaming_cache,expected_fit_chunks=len(fit),
        epoch0_chunk_order='ascending_pre_shuffled_manifest' if args.streaming_cache else 'shuffled',
        cache_read='sequential_chunk_into_RAM',local_metrics=args.local_metrics,
        loss='KL(T2)+.05 localSmoothL1+utilitySmoothL1+.1 soft-thresholdBCE(T.5)',
        cache=str(args.cache), source_sha256={str(f):sha(f) for f in [Path(__file__),HERE/'model.py',HERE/'policies.py']})
    metrics_root=HERE/'live_metrics'/args.kind if args.local_metrics else args.output
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True); write_json(args.output/'config.json',config)
        metrics_root.mkdir(parents=True,exist_ok=True)
        if args.local_metrics:
            for name in ['training.jsonl','held.jsonl']:
                local_path=metrics_root/name
                local_path.touch(exist_ok=True)
                link=args.output/name
                if link.is_symlink():assert link.resolve()==local_path.resolve()
                elif link.exists():
                    assert not local_path.stat().st_size,'Cannot overwrite existing local training history'
                    shutil.copyfile(link,local_path);link.unlink();link.symlink_to(local_path.resolve())
                else:link.symlink_to(local_path.resolve())
    if world>1: dist.barrier()
    started=time.monotonic(); initial_step=step; checked=False
    def save(name, metrics=None, calibration=None):
        if rank: return
        path=args.output/name; temporary=path.with_suffix('.pt.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
            best_held_loss=best,config=config,metrics=metrics,calibration=calibration),temporary)
        temporary.replace(path)
    for cpu in loader:
        if step >= args.steps: break
        data={key:value.cuda(non_blocking=True) for key,value in cpu.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): output=student(data['f_base'])
        loss,terms=distillation_loss(output,data['local'][:,index],data['utility'],policy['weight'],policy['threshold'])
        assert torch.isfinite(loss)
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
        assert torch.isfinite(norm) and norm > 0
        optimizer.step(); step+=1
        if not checked:
            model.eval()
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                a=model.select(data['f_base']); b=model.select(data['f_base'].flip(0))
                assert torch.equal(a['K'],b['K'].flip(0)) and torch.equal(a['mask'],b['mask'].flip(0))
            model.train(); checked=True
            if rank==0: write_json(args.output/'input_audit.json',dict(base_features_only=True,
                per_image_independent=True, single_score_and_selection=True, original_router_not_loaded=True,
                no_B0_or_RGB_or_2D_features_in_student=True, fresh_initialization=initial_step==0))
        if rank==0 and (step<=4 or step%20==0):
            row=dict(step=step,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()},
                elapsed_seconds=time.monotonic()-started, steps_this_run=step-initial_step,
                peak_GiB=torch.cuda.max_memory_allocated()/2**30)
            with (metrics_root/'training.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            if world>1: dist.barrier()
            if rank==0:
                metrics,scores=evaluate(model,held_loader,index,policy)
                calibration=calibrate(scores)
                with (metrics_root/'held.jsonl').open('a') as f: f.write(json.dumps(dict(step=step,**metrics,calibration=calibration))+'\n')
                print(json.dumps(dict(stage='held',step=step,**metrics,calibration=calibration)),flush=True)
                if metrics['loss']<best:
                    best=metrics['loss']; save('best.pt',metrics,calibration)
                save('latest.pt',metrics,calibration)
            if world>1: dist.barrier()
        elif step%500==0: save('latest.pt')
    if rank==0:
        if args.local_metrics:
            for name in ['training.jsonl','held.jsonl']:
                temporary=args.output/(name+'.archive.tmp')
                shutil.copyfile(metrics_root/name,temporary)
                temporary.replace(args.output/name)
        write_json(args.output/'report.json',dict(status='completed',step=step,
            elapsed_seconds=time.monotonic()-started,best_held_loss=best,
            best_checkpoint_sha256=sha(args.output/'best.pt'), teacher_policy=policy,
            note='Scorer/offset distillation only; no MaskGIT fine-tuning or gFID evaluation performed.'))
    if world>1: dist.destroy_process_group()

if __name__=='__main__': main()
