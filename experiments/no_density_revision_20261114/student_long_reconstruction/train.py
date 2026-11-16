"""Base-only student fine-tuning through frozen RGB/semantic reconstruction."""
import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from direct_runtime import DirectRuntime, spatial_score, training_mask, objective, B0_WEIGHTS
from model import GenerationRouter
from quality_eval import config as codec_config
from policies import HERE, DATA, POLICIES, WARMSTART, WARMSTART_SHA256, INITIAL_STEP, FINETUNING_UPDATES, SAMPLER_SEED, LR, EXECUTION_GATE, PARENT_AUDIT, sha, write_json
from cache_train_counterfactual import ImagePathDataset


PARENT_MODEL_SOURCE=HERE.parent/'student_final_parent/model.py'
_spec=importlib.util.spec_from_file_location('long_genuine_native_parent_reference',PARENT_MODEL_SOURCE)
_parent=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_parent)
ParentRouter=_parent.GenerationRouter


def make_optimizer(module):
    spatial=[v for name,v in module.named_parameters() if not name.startswith('offset_head.')]
    offset=list(module.offset_head.parameters())
    assert len(spatial)==17 and len(offset)==4
    return torch.optim.AdamW([dict(params=spatial,role='spatial'),dict(params=offset,role='offset')],lr=LR,weight_decay=.01)


def calibrate(scores):
    lo, hi = float(scores.min()) - 1, float(scores.max()) + 1
    best = (float('inf'), None, None)
    for _ in range(64):
        threshold = float(np.float32((lo + hi) / 2))
        mean = float(np.clip((scores >= threshold).sum(1), 48, 144).mean())
        if abs(mean - 96) < best[0]:
            best = (abs(mean - 96), threshold, mean)
        if mean > 96:
            lo = threshold
        else:
            hi = threshold
    assert best[0] < .01
    return dict(threshold=best[1], held_mean_K=best[2], target_K=96,
        source='held_training_images_only', quality_metrics_used=False,
        score_array_sha256=hashlib.sha256(scores.tobytes()).hexdigest())


@torch.no_grad()
def evaluate(module, reference, runtime, loader, n, rank, world):
    module.eval()
    local_scores = torch.cat([module(runtime.codec.base(images.cuda(non_blocking=True)).float())['score']
                              for images in loader])
    assert len(local_scores) == n // world
    if world > 1:
        gathered = [torch.empty_like(local_scores) for _ in range(world)]
        dist.all_gather(gathered, local_scores)
    else:
        gathered = [local_scores]
    calibration = None
    if rank == 0:
        all_scores = np.empty((n, 256), dtype=np.float32)
        for owner, block in enumerate(gathered):
            all_scores[owner::world] = block.cpu().numpy()
        calibration = calibrate(all_scores)
    threshold_tensor = torch.tensor(calibration['threshold'] if rank == 0 else 0., device='cuda')
    if world > 1:
        dist.broadcast(threshold_tensor, 0)
    threshold = float(threshold_tensor)
    del gathered
    sums = torch.zeros(9, dtype=torch.float64, device='cuda')
    cursor = 0
    for images in loader:
        data = runtime.prepare(images.cuda(non_blocking=True))
        output = module(data['base'].float())
        assert torch.equal(output['score'], local_scores[cursor:cursor+len(images)])
        initial = reference(data['base'].float())
        selected = module.select_output(output, threshold)
        prediction = runtime.decode(data, selected['mask'].float())
        semantic, rgb = runtime.losses(prediction, data)
        _, terms = objective(output, initial, semantic, rgb)
        b = len(images)
        sums[:5] += torch.stack([terms[k].double() * b for k in
            ['spatial', 'semantic', 'rgb', 'spatial_anchor', 'utility_anchor']])
        sums[5] += runtime.codec.ev.psnr_sum(prediction, data['target'], 'minus1_1').double()
        k = selected['K'].double()
        sums[6] += b
        sums[7] += k.sum()
        sums[8] += k.square().sum()
        cursor += b
    assert cursor == len(local_scores)
    if world > 1:
        dist.all_reduce(sums)
    values = sums.cpu().tolist()
    assert values[6] == n
    module.train()
    if rank != 0:
        return None, None
    mean_k = values[7] / n
    assert abs(mean_k - calibration['held_mean_K']) < 1e-9
    return dict(n=n, loss=values[0]/n, semantic=values[1]/n, rgb=values[2]/n,
        spatial_anchor=values[3]/n, offset_loss=values[4]/n, PSNR=values[5]/n,
        mean_K=mean_k, std_K=float(max(0., values[8]/n - mean_k**2)**.5),
        checkpoint_criterion='held meanK96 semantic+.5RGB+.01local anchor; offset excluded',
        score_scan_quality_replay_exact=True, official_validation_images_used=0), calibration


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=FINETUNING_UPDATES)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--local-metrics', action='store_true')
    args = p.parse_args()
    rank, world, local = [int(os.environ.get(k, d)) for k,d in
        [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')]]
    assert list(POLICIES) == ['long'] and args.updates > 0 and world in [2,4]
    assert json.loads(EXECUTION_GATE.read_text())['status'] == 'user_authorized_long_only'
    assert json.loads(PARENT_AUDIT.read_text())['actual_all21_optimizer_state_steps']==INITIAL_STEP
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    torch.manual_seed(SAMPLER_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.88)
    if world > 1:
        dist.init_process_group('nccl')
    policy = POLICIES['long']
    assert sha(WARMSTART) == WARMSTART_SHA256
    inherited = torch.load(WARMSTART, map_location='cpu', weights_only=False)
    assert inherited['step'] == INITIAL_STEP
    module = GenerationRouter(policy['weight'], policy['threshold']).cuda()
    module.load_state_dict(inherited['model'], strict=True)
    reference = ParentRouter(policy['weight'], policy['threshold']).cuda().eval().requires_grad_(False)
    reference.load_state_dict(inherited['model'], strict=True)
    reference_versions = [(v, v._version) for v in reference.parameters()]
    spatial = [v for name,v in module.named_parameters() if not name.startswith('offset_head.')]
    offset = list(module.offset_head.parameters())
    optimizer = make_optimizer(module)
    student = DDP(module, device_ids=[local], broadcast_buffers=False) if world > 1 else module
    for name,value in module.state_dict().items():
        assert torch.equal(value.detach().cpu(),inherited['model'][name]),name
    initialization=dict(all_21_values_exact_original_long_parent_after_DDP=True,
        parent_step=INITIAL_STEP,parent_sha256=WARMSTART_SHA256,execution_gate_passed=True)
    runtime = DirectRuntime()
    fit = (DATA/'fit_paths.txt').read_text().splitlines()
    held = (DATA/'held_paths.txt').read_text().splitlines()
    assert len(fit) == len(set(fit)) == 1271167 and len(held) == len(set(held)) == 10000
    assert not set(fit).intersection(held) and len(held) % world == 0
    assert all('/train/' in path for path in fit + held)
    sampler = DistributedSampler(fit, world, rank, shuffle=True, seed=SAMPLER_SEED, drop_last=True)
    loader = DataLoader(ImagePathDataset(fit, runtime.codec.crop), batch_size=args.batch,
        sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    held_loader = DataLoader(ImagePathDataset(held[rank::world], runtime.codec.crop),
        batch_size=args.batch, num_workers=2, pin_memory=True)
    config = dict(kind='long', policy=policy, input='f_base_only_Bx256x16x16',
        steps=INITIAL_STEP+args.updates, initial_checkpoint_step=INITIAL_STEP,
        new_updates=args.updates, fresh_initialization=False, sampler_seed=SAMPLER_SEED,
        execution_gate_sha256=sha(EXECUTION_GATE),parent_source_audit_sha256=sha(PARENT_AUDIT),profile_weights_loaded=False,
        warmstart_checkpoint=str(WARMSTART), warmstart_sha256=WARMSTART_SHA256,
        warmstart_optimizer_loaded=False, micro_batch=args.batch, world_size=world,
        global_batch=args.batch*world, full_fit_manifest_images=len(fit), held_images=len(held),
        fit_images_seen_this_stage=args.updates*args.batch*world,
        new_stage_visits_entire_manifest=args.updates*args.batch*world>=len(fit),
        optimizer='FreshAdamW lr2e-5 wd.01; spatial/offset groups separately clipped1',
        autocast='FP32 student; BF16 frozen codec and semantic supervision',
        loss='B0semantic+.5RGB+.01localanchor; isolatedUanchor to initialstudent',
        checkpoint_selection='fixedmeanK96 held spatial objective; no officialval/FID',
        training_mask='Hardmaskforward; sigmoidSTgradient; training-onlybatchrateconstraint',
        deployment='singleforward basesonly; frozenscalar threshold; K48--144; stabletopK',
        B0_used_at_deployment=False, direct_K_head=False, direct_K_regression=False,per_image_K_supervision=False,offset_features_detached=True,
        asset_sha256={str(f):sha(f) for f in [codec_config().checkpoint,codec_config().titok_ckpt,codec_config().llamagen_ckpt]},
        source_sha256={str(f):sha(f) for f in [Path(__file__),HERE/'direct_runtime.py',
            HERE/'model.py',HERE/'policies.py',HERE/'quality_eval.py',B0_WEIGHTS,PARENT_MODEL_SOURCE,HERE/'acceptance_requirements.json']},
        manifest_sha256={str(f):sha(f) for f in [DATA/'fit_paths.txt',DATA/'held_paths.txt']})
    metrics_root = HERE/'live_metrics/long' if args.local_metrics else args.output
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        metrics_root.mkdir(parents=True, exist_ok=True)
        write_json(args.output/'config.json', config)
        write_json(args.output/'DDP_initialization_audit.json',initialization)
        for name in ['training.jsonl', 'held.jsonl']:
            path = metrics_root/name
            assert not path.exists()
            path.touch()
            if args.local_metrics:
                (args.output/name).symlink_to(path)
    if world > 1:
        dist.barrier()
    def audit():
        assert all(not v.requires_grad and v.grad is None and v._version == old for v,old in reference_versions)
        return dict(**runtime.audit(), reference_student_frozen=True)
    if args.baseline:
        metrics, calibration = evaluate(module, reference, runtime, held_loader, len(held), rank, world)
        if rank == 0:
            write_json(args.output/'baseline_held.json', dict(metrics=metrics, calibration=calibration,
                checkpoint_step=INITIAL_STEP, warmstart_sha256=WARMSTART_SHA256))
            print(json.dumps(dict(stage='baseline_held', **metrics, calibration=calibration)), flush=True)
    if world > 1:
        dist.barrier()
    started = time.monotonic()
    best, new_step = float('inf'), 0
    def save(name, metrics=None, calibration=None):
        if rank != 0:
            return
        path = args.output/name
        temporary = path.with_suffix('.pt.tmp')
        torch.save(dict(model=module.state_dict(), optimizer=optimizer.state_dict(),
            step=INITIAL_STEP+new_step, initial_step=INITIAL_STEP, new_updates=new_step,
            best_held_loss=best, config=config, metrics=metrics, calibration=calibration), temporary)
        temporary.replace(path)
    epoch = 0
    while new_step < args.updates:
        sampler.set_epoch(epoch)
        for images in loader:
            data = runtime.prepare(images.cuda(non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            output = student(data['base'].float())
            with torch.no_grad():
                initial = reference(data['base'].float())
            mask, k, threshold = training_mask(spatial_score(output, policy['weight']))
            prediction = runtime.decode(data, mask)
            semantic, rgb = runtime.losses(prediction, data)
            loss, terms = objective(output, initial, semantic, rgb)
            assert torch.isfinite(loss)
            loss.backward()
            spatial_norm = torch.nn.utils.clip_grad_norm_(spatial, 1.)
            offset_norm = torch.nn.utils.clip_grad_norm_(offset, 1.)
            assert torch.isfinite(spatial_norm) and spatial_norm > 0 and torch.isfinite(offset_norm)
            optimizer.step()
            new_step += 1
            if new_step == 1:
                assert terms['spatial_anchor']==0 and terms['utility_anchor']==0
                audit()
                module.eval()
                with torch.no_grad():
                    a = module.select(data['base'].float())
                    b = module.select(data['base'].float().flip(0))
                    assert torch.equal(a['K'], b['K'].flip(0)) and torch.equal(a['mask'], b['mask'].flip(0))
                module.train()
                if rank == 0:
                    write_json(args.output/'input_audit.json', dict(base_features_only=True,
                        per_image_independent=True, single_score_and_selection=True,
                        original_router_not_loaded_as_student=True, original_codec_router_deleted=True,
                        no_RGB_or_2D_or_B0_in_student=True, no_B0_at_deployment=True,
                        offset_features_detached=True, separate_parameter_group_clipping=True,
                        first_spatial_and_U_anchors_zero=True,DDP_initialized_values_exact_long_parent=True,
                        initial_checkpoint_step=INITIAL_STEP, warmstart_sha256=WARMSTART_SHA256))
            if rank == 0 and (new_step <= 4 or new_step % 20 == 0):
                row = dict(step=INITIAL_STEP+new_step, steps_this_run=new_step,
                    loss=float(loss.detach()), **{key:float(value.detach()) for key,value in terms.items()},
                    spatial_grad_norm=float(spatial_norm),U_head_grad_norm=float(offset_norm),
                    mean_K=float(k.float().mean()), training_threshold=float(threshold),
                    peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                    elapsed_seconds=time.monotonic()-started)
                with (metrics_root/'training.jsonl').open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
            if new_step % args.eval_every == 0 or new_step == args.updates:
                metrics, calibration = evaluate(module, reference, runtime, held_loader, len(held), rank, world)
                if rank == 0:
                    row = dict(step=INITIAL_STEP+new_step, new_updates=new_step, **metrics, calibration=calibration)
                    with (metrics_root/'held.jsonl').open('a') as handle:
                        handle.write(json.dumps(row)+'\n')
                    print(json.dumps(dict(stage='held', **row)), flush=True)
                    if metrics['loss'] < best:
                        best = metrics['loss']
                        save('best.pt', metrics, calibration)
                    save('latest.pt', metrics, calibration)
                    if args.updates==FINETUNING_UPDATES:
                        save(f'candidate_{new_step}.pt',metrics,calibration)
                if world > 1:
                    dist.barrier()
            elif new_step % 250 == 0:
                save('latest.pt')
            if new_step >= args.updates:
                break
        epoch += 1
    frozen_audit = audit()
    if rank == 0:
        if args.local_metrics:
            for name in ['training.jsonl','held.jsonl']:
                temporary = args.output/(name+'.archive.tmp')
                temporary.write_bytes((metrics_root/name).read_bytes())
                temporary.replace(args.output/name)
        write_json(args.output/'report.json', dict(status='completed', step=INITIAL_STEP+new_step,
            initial_checkpoint_step=INITIAL_STEP, new_finetuning_updates=new_step,
            warmstart_checkpoint=str(WARMSTART), warmstart_sha256=WARMSTART_SHA256,
            teacher_policy=policy, best_held_loss=best,
            best_checkpoint_sha256=sha(args.output/'best.pt'),
            elapsed_seconds=time.monotonic()-started, **frozen_audit))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
