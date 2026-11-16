"""Train copied budget features and U; preserve all local scores exactly."""
import argparse
import hashlib
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from model import GenerationRouter
from direct_runtime import DirectRuntime, B0_WEIGHTS
from quality_eval import config as codec_config
from budget_surrogate import training_mask
from policies import (HERE, DATA, POLICIES, WARMSTART, WARMSTART_SHA256,
                      SOURCE_STEPS, UPDATES, LR, FEATURE_LR, PARENT_LOCAL_FT_UPDATES, SAMPLER_SEED, PARENT_STEP, EXECUTION_GATE, sha, write_json)
from cache_train_counterfactual import ImagePathDataset


PARENT_MODEL_SOURCE = HERE.parent/'student_long_reconstruction/model.py'
_spec = importlib.util.spec_from_file_location('native_budget_parent_reference', PARENT_MODEL_SOURCE)
_parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent)
ParentRouter = _parent.GenerationRouter


def make_optimizer(module):
    features = [value for name,value in module.named_parameters() if value.requires_grad and name.startswith('budget_')]
    offset = list(module.offset_head.parameters())
    assert len(features)==15 and sum(v.numel() for v in features)==509184
    assert len(offset)==4 and sum(v.numel() for v in offset)==16513
    return torch.optim.AdamW([dict(params=features,role='budget_features',lr=FEATURE_LR),
        dict(params=offset,role='utility',lr=LR)],weight_decay=.01)


def calibrate(scores):
    assert scores.shape == (10000, 256) and np.isfinite(scores).all()
    lo, hi = float(scores.min())-1, float(scores.max())+1
    best = (float('inf'), None, None)
    for _ in range(64):
        threshold = float(np.float32((lo+hi)/2))
        mean = float(np.clip((scores >= threshold).sum(1), 48, 144).mean())
        if abs(mean-96) < best[0]:
            best = (abs(mean-96), threshold, mean)
        if mean > 96:
            lo = threshold
        else:
            hi = threshold
    assert best[0] < .01
    return dict(threshold=best[1], held_mean_K=best[2], target_K=96,
        source='TRAIN_held_only', quality_metrics_used=False,
        score_array_sha256=hashlib.sha256(scores.tobytes()).hexdigest())


@torch.no_grad()
def evaluate(module, reference, runtime, loader, n, rank, world):
    module.eval()
    local_scores = torch.cat([module(runtime.codec.base(images.cuda(non_blocking=True)).float())['score']
                              for images in loader])
    assert len(local_scores) == n//world
    gathered = [torch.empty_like(local_scores) for _ in range(world)] if world > 1 else [local_scores]
    if world > 1:
        dist.all_gather(gathered, local_scores)
    calibration = None
    if rank == 0:
        scores = np.empty((n, 256), dtype=np.float32)
        for owner, block in enumerate(gathered):
            scores[owner::world] = block.cpu().numpy()
        calibration = calibrate(scores)
    threshold = torch.tensor(calibration['threshold'] if rank == 0 else 0., device='cuda')
    if world > 1:
        dist.broadcast(threshold, 0)
    threshold = float(threshold)
    del gathered
    sums = torch.zeros(8, dtype=torch.float64, device='cuda')
    cursor = 0
    for images in loader:
        data = runtime.prepare(images.cuda(non_blocking=True))
        output, original = module(data['base'].float()), reference(data['base'].float())
        assert torch.equal(output['local'], original['local'])
        assert torch.equal(output['score'], local_scores[cursor:cursor+len(images)])
        chosen = module.select_output(output, threshold)
        prediction = runtime.decode(data, chosen['mask'].float())
        semantic, rgb = runtime.losses(prediction, data)
        anchor = F.smooth_l1_loss(output['utility'], original['utility'], reduction='none')
        sums[0] += (semantic+.5*rgb+.01*anchor).double().sum()
        sums[1] += semantic.double().sum()
        sums[2] += rgb.double().sum()
        sums[3] += anchor.double().sum()
        sums[4] += runtime.codec.ev.psnr_sum(prediction, data['target'], 'minus1_1').double()
        sums[5] += len(images)
        k = chosen['K'].double()
        sums[6] += k.sum()
        sums[7] += k.square().sum()
        cursor += len(images)
    assert cursor == len(local_scores)
    if world > 1:
        dist.all_reduce(sums)
    values = sums.cpu().tolist()
    assert values[5] == n
    module.train()
    if rank != 0:
        return None, None
    mean = values[6]/n
    assert abs(mean-calibration['held_mean_K']) < 1e-9
    return dict(n=n, loss=values[0]/n, semantic=values[1]/n, rgb=values[2]/n,
        U_anchor=values[3]/n, PSNR=values[4]/n, mean_K=mean,
        std_K=float(max(0., values[7]/n-mean**2)**.5),
        local_scores_exact_parent=True, score_scan_quality_replay_exact=True,
        official_validation_images_used=0), calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=__import__('pathlib').Path)
    parser.add_argument('--updates', type=int, default=UPDATES)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--local-metrics', action='store_true')
    args = parser.parse_args()
    rank, world, local = [int(os.environ.get(key, default)) for key, default in
                         [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')]]
    assert world in [2, 4] and args.updates > 0
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    torch.manual_seed(SAMPLER_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.88)
    dist.init_process_group('nccl')
    assert json.loads(EXECUTION_GATE.read_text())['status'] == 'user_authorized_long_only'
    assert sha(WARMSTART) == WARMSTART_SHA256
    inherited = torch.load(WARMSTART, map_location='cpu', weights_only=False)
    assert inherited['step']==PARENT_STEP and inherited['new_updates']==PARENT_LOCAL_FT_UPDATES and inherited['initial_step']==PARENT_STEP-PARENT_LOCAL_FT_UPDATES
    assert inherited['config']['kind']=='long' and inherited['config']['policy']['weight']==.9
    policy = POLICIES['long']
    module = GenerationRouter(policy['weight'], policy['threshold']).cuda()
    module.load_parent_state_dict(inherited['model'])
    active = [v for v in module.parameters() if v.requires_grad]
    assert len(active) == 19 and sum(value.numel() for value in active) == 525697
    assert len(list(module.parameters())) == 36
    assert all(value.requires_grad == name.startswith(('budget_','offset_head.')) for name,value in module.named_parameters())
    for name,value in module.named_parameters():
        if name.startswith('budget_'):
            assert value.data_ptr()!=dict(module.named_parameters())[name.removeprefix('budget_')].data_ptr()
    reference = ParentRouter(policy['weight'], policy['threshold']).cuda().eval().requires_grad_(False)
    reference.load_state_dict(inherited['model'], strict=True)
    reference_versions = [(value, value._version) for value in reference.parameters()]
    before_DDP_versions = {name:value._version for name, value in module.named_parameters()
                           if not value.requires_grad}
    optimizer = make_optimizer(module)
    student = DDP(module, device_ids=[local], broadcast_buffers=False)
    # DDP initializes by synchronizing parameters, including frozen tensors.
    # Verify values after that sync, then start the optimizer-freeze audit.
    for name, value in module.state_dict().items():
        assert torch.equal(value.detach().cpu(), inherited['model'][name.removeprefix('budget_')]), name
    frozen_versions = [(name, value, value._version) for name, value in module.named_parameters()
                       if not value.requires_grad]
    assert len(frozen_versions) == 17
    DDP_initialization = dict(all_36_parameter_values_exact_mapped_parent_after_DDP=True,
        frozen_snapshot_after_DDP=True, frozen_tensors=17,
        version_counters_changed_at_DDP=[name for name, value, _ in frozen_versions
                                        if value._version != before_DDP_versions[name]],
        before_frozen_versions=before_DDP_versions,
        after_frozen_versions={name:version for name, _, version in frozen_versions})
    runtime = DirectRuntime()
    fit = (DATA/'fit_paths.txt').read_text().splitlines()
    held = (DATA/'held_paths.txt').read_text().splitlines()
    assert len(fit) == len(set(fit)) == 1271167 and len(held) == len(set(held)) == 10000
    assert not set(fit).intersection(held) and len(held) % world == 0
    assert all('/train/' in path for path in fit+held)
    sampler = DistributedSampler(fit, world, rank, shuffle=True, seed=SAMPLER_SEED, drop_last=True)
    loader = DataLoader(ImagePathDataset(fit, runtime.codec.crop), batch_size=args.batch,
        sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    held_loader = DataLoader(ImagePathDataset(held[rank::world], runtime.codec.crop),
        batch_size=args.batch, num_workers=2, pin_memory=True)
    config = dict(kind='long', input='f_base_only_Bx256x16x16', policy=policy,
        new_U_updates=args.updates, parent_actual_step=PARENT_STEP,
        parent_checkpoint=str(WARMSTART), parent_checkpoint_sha256=WARMSTART_SHA256,
        parent_source_component_steps=SOURCE_STEPS, parent_local_FT_updates=PARENT_LOCAL_FT_UPDATES,
        new_spatial_updates=0, new_budget_feature_updates=args.updates, budget_features_detached=False, fresh_initialization=False, parent_optimizer_loaded=False,
        profile_weights_loaded=False, trainable_parameter_count=525697,
        trainable_parameter_tensors=[name for name, value in module.named_parameters() if value.requires_grad],
        frozen_parameter_tensors=[name for name, _, _ in frozen_versions],
        micro_batch=args.batch, world_size=world, global_batch=args.batch*world,
        fit_manifest_images=len(fit), held_images=len(held), fit_samples_this_stage=args.updates*args.batch*world,
        sampler_seed=SAMPLER_SEED, optimizer=dict(kind='FreshAdamW', feature_lr=FEATURE_LR, U_lr=LR, weight_decay=.01),
        loss='B0semantic+.5RGB+.01Uanchor; budgetFeaturesAndUTrainable',
        gradient='training-only normalized selected-boundary U surrogate with global raw-score-density projection',
        gradient_temperature=.25, direct_K_regression=False, per_image_K_supervision=False,
        deployment='split internal budget features; base-only, one local map/U scalar/final set',
        B0_at_deployment=False, Inception_or_LPIPS_training=False,
        asset_sha256={str(path):sha(path) for path in
            [codec_config().checkpoint, codec_config().titok_ckpt, codec_config().llamagen_ckpt]},
        source_sha256={str(path):sha(path) for path in
            [HERE/'train.py', HERE/'budget_surrogate.py', HERE/'direct_runtime.py', HERE/'model.py',
             HERE/'policies.py', HERE/'quality_eval.py', HERE/'acceptance_requirements.json', B0_WEIGHTS, PARENT_MODEL_SOURCE, EXECUTION_GATE]},
        manifest_sha256={str(path):sha(path) for path in [DATA/'fit_paths.txt', DATA/'held_paths.txt']})
    metrics_root = HERE/'live_metrics/long' if args.local_metrics else args.output
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        metrics_root.mkdir(parents=True, exist_ok=True)
        write_json(args.output/'config.json', config)
        write_json(args.output/'DDP_initialization_audit.json', DDP_initialization)
        for filename in ['training.jsonl', 'held.jsonl']:
            path = metrics_root/filename
            assert not path.exists()
            path.touch()
            if args.local_metrics:
                (args.output/filename).symlink_to(path)
    dist.barrier()

    def audit():
        assert all(not value.requires_grad and value.grad is None and value._version == version
                   for value, version in reference_versions)
        assert all(not value.requires_grad and value.grad is None and value._version == version
                   for _, value, version in frozen_versions)
        return dict(**runtime.audit(), reference_student_frozen=True,
                    all_17_local_tensors_frozen=True, local_scores_exact_parent_each_batch=True,
                    budget_features_and_existing_U_head_optimized=True)

    new_step, best, epoch = 0, float('inf'), 0
    started = time.monotonic()

    def save(filename, metrics, calibration):
        temporary = args.output/(filename+'.tmp')
        torch.save(dict(model=module.state_dict(), optimizer=optimizer.state_dict(),
            step=PARENT_STEP+new_step, component_training_steps=dict(spatial=PARENT_STEP,budget=PARENT_STEP+new_step), new_U_updates=new_step, new_spatial_updates=0, new_budget_feature_updates=new_step, parent_local_FT_updates=PARENT_LOCAL_FT_UPDATES,
            source_component_training_steps=SOURCE_STEPS,
            parent_checkpoint_sha256=WARMSTART_SHA256, config=config,
            best_held_loss=best, metrics=metrics, calibration=calibration,
            exact_data_iterator_resume_supported=False), temporary)
        temporary.replace(args.output/filename)

    while new_step < args.updates:
        sampler.set_epoch(epoch)
        for images in loader:
            data = runtime.prepare(images.cuda(non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            output = student(data['base'].float())
            with torch.no_grad():
                original = reference(data['base'].float())
            assert torch.equal(output['local'], original['local'])
            assert not output['local'].requires_grad and output['utility'].requires_grad
            mask, k, threshold = training_mask(output, policy['weight'], module.select_output)
            prediction = runtime.decode(data, mask)
            semantic, rgb = runtime.losses(prediction, data)
            task = semantic.mean()+.5*rgb.mean()
            anchor = F.smooth_l1_loss(output['utility'], original['utility'])
            loss = task+.01*anchor
            assert torch.isfinite(loss)
            if new_step == 0:
                assert torch.equal(output['utility'], original['utility']) and float(anchor.detach()) == 0
                gradient = torch.autograd.grad(task, output['utility'], retain_graph=True)[0].detach()
                relative_common_shift = gradient.sum().abs()/gradient.abs().sum().clamp_min(1e-12)
                assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                assert relative_common_shift < 1e-4
                if rank == 0:
                    write_json(args.output/'gradient_audit.json', dict(
                        task_U_gradient_nonzero=True, task_U_gradient_norm=float(gradient.norm()),
                        relative_common_U_shift_gradient=float(relative_common_shift),
                        first_U_anchor_exact_zero=True, local_scores_exact_parent=True,
                        hard_forward_exact=True, K_regression_or_per_image_K_labels=False))
            loss.backward()
            if new_step == 0:
                feature_grad = torch.stack([v.grad.square().sum() for name,v in module.named_parameters()
                    if name.startswith('budget_')]).sum().sqrt()
                head_grad = torch.stack([v.grad.square().sum() for v in module.offset_head.parameters()]).sum().sqrt()
                assert torch.isfinite(feature_grad) and feature_grad > 0 and torch.isfinite(head_grad) and head_grad > 0
                if rank == 0:
                    write_json(args.output/'budget_feature_gradient_audit.json', dict(
                        copied_budget_feature_gradient_finite_nonzero=True, U_head_gradient_finite_nonzero=True,
                        feature_gradient_norm=float(feature_grad),U_head_gradient_norm=float(head_grad),
                        budget_features_not_detached=True,local_branch_requires_grad=False))
            norm = torch.nn.utils.clip_grad_norm_(active, 1.)
            assert torch.isfinite(norm) and norm > 0
            optimizer.step()
            new_step += 1
            if new_step == 1:
                audit()
                module.eval()
                with torch.no_grad():
                    base = data['base'].float()
                    a = module.select(base, float(threshold))
                    reverse = module.select(base.flip(0), float(threshold))
                    singleton = module.select(base[:1], float(threshold))
                    assert torch.equal(a['K'], reverse['K'].flip(0)) and torch.equal(a['mask'], reverse['mask'].flip(0))
                    assert torch.equal(a['K'][:1], singleton['K']) and torch.equal(a['mask'][:1], singleton['mask'])
                module.train()
                if rank == 0:
                    write_json(args.output/'input_audit.json', dict(base_only=True,
                        same_base_singleton_reverse_masks_K_exact=True,
                        one_scoring_forward_one_final_set=True, no_extra_inference_model=True,
                        budget_features_and_U_trainable=True, no_direct_K_head=True,
                        RGB_F2_B0_not_student_inputs=True, original_codec_router_deleted=True))
            if rank == 0 and (new_step <= 4 or new_step % 20 == 0):
                row = dict(new_U_updates=new_step, new_budget_feature_updates=new_step, loss=float(loss.detach()),
                    semantic=float(semantic.mean().detach()), rgb=float(rgb.mean().detach()),
                    U_anchor=float(anchor.detach()), budget_features_and_U_grad_norm=float(norm),
                    mean_K=float(k.float().mean()), training_threshold=float(threshold),
                    peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                    elapsed_seconds=time.monotonic()-started)
                with (metrics_root/'training.jsonl').open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
            if new_step % args.eval_every == 0 or new_step == args.updates:
                metrics, calibration = evaluate(module, reference, runtime, held_loader, len(held), rank, world)
                if rank == 0:
                    row = dict(new_U_updates=new_step, new_budget_feature_updates=new_step, **metrics, calibration=calibration)
                    with (metrics_root/'held.jsonl').open('a') as handle:
                        handle.write(json.dumps(row)+'\n')
                    print(json.dumps(dict(stage='held', **row)), flush=True)
                    if metrics['loss'] < best:
                        best = metrics['loss']
                        save('best.pt', metrics, calibration)
                    save('latest.pt', metrics, calibration)
                    if new_step % 500 == 0:
                        save(f'candidate_{new_step}.pt', metrics, calibration)
                dist.barrier()
            if new_step == args.updates:
                break
        epoch += 1
    evidence = audit()
    if rank == 0:
        if args.local_metrics:
            for filename in ['training.jsonl', 'held.jsonl']:
                temporary = args.output/(filename+'.archive.tmp')
                temporary.write_bytes((metrics_root/filename).read_bytes())
                temporary.replace(args.output/filename)
        write_json(args.output/'report.json', dict(status='completed', new_U_updates=new_step, new_budget_feature_updates=new_step, parent_local_FT_updates=PARENT_LOCAL_FT_UPDATES,
            new_spatial_updates=0, step=PARENT_STEP+new_step, component_training_steps=dict(spatial=PARENT_STEP,budget=PARENT_STEP+new_step), source_component_training_steps=SOURCE_STEPS,
            best_held_loss=best, best_checkpoint_sha256=sha(args.output/'best.pt'),
            elapsed_seconds=time.monotonic()-started, **evidence))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
