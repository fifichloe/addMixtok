"""Current independent controls with verified 2e-5 ADM comparison tolerance.

An earlier 8+24 run had only three FP32 ADM values above 1e-5 among
102.4 million base features; the maximum difference was 1.0967254638671875e-5.
Its image IDs, tokens, scores, and utilities matched the reference exactly.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.linalg import sqrtm

EVAL_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_HERE))
from common import (KINDS, POLICIES, OUT, ASSETS, PYTHON, module, router_identity, run_root)
from routing import nearconstant_k, random_positions, uniform_order
from bert2d.paths import atomic_json, sha256, EVALUATOR

DISPATCH = EVAL_HERE.parent.parent/'new_router_ablation_suite_20261114/recovery_shared_gpu_admission_20261114'
sys.path.insert(0, str(DISPATCH))
from gpu_dispatch import snapshot


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    identity = router_identity(args.kind)
    assert args.policy_file is not None
    document = json.loads(args.policy_file.read_text())
    assert document['status'] == 'frozen_before_independent_control_quality'
    assert document['router_identity'] == identity
    assert document['kind'] == args.kind and document['seed'] == args.seed
    assert document['cfg'] == args.cfg and document['stage2_steps'] == args.stage2_steps == 8
    policy = document['selection_policy']
    assert policy['policy'] in POLICIES
    checkpoint = run_root(args.kind, policy['policy'])/'latest.pt'
    digest = sha256(checkpoint)
    assert document['checkpoint_sha256'] == digest
    frozen = {str(path):sha256(path) for path in [Path(__file__), EVAL_HERE/'eval_worker.py',
              EVAL_HERE/'routing.py', DISPATCH/'gpu_dispatch.py', args.policy_file]}
    config = dict(kind=args.kind, n=50000, seed=args.seed, stage1_steps=8,
                  stage2_steps=args.stage2_steps, cfg_stage1=4.5, cfg_stage2=args.cfg,
                  batch=8, logical_shards=4, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                  router=identity, routing_policy=policy, source_sha256=frozen,
                  maximum_owned_GPUs=2, shared_GPU_dispatch_min_free_GiB=15,
                  no_automatic_retry=True)
    atomic_json(out/'parallel_config.json', config)
    state = dict(status='waiting_for_generation_GPU', pid=os.getpid(), completed=0, total=50000, jobs=[])
    active = []

    def update(**extra):
        state.update(seconds=time.monotonic()-started, last_checked=time.time(), **extra)
        atomic_json(out/'status.json', state)

    def check_sources():
        for path, expected in frozen.items():
            assert sha256(path) == expected, path

    try:
        update()
        pending = list(range(4))
        while pending or active:
            owned = {card['uuid'] for _, _, card, _ in active}
            free = [card for card in snapshot()
                    if card['uuid'] not in owned and card['free_MiB'] >= 15*1024]
            while pending and free and len(active) < 2:
                rank, card = pending.pop(0), free.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=card['uuid'], OMP_NUM_THREADS='4',
                           OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
                for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                            'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE'):
                    env.pop(key, None)
                command = [PYTHON, str(EVAL_HERE/'eval_worker.py'), '--kind', args.kind, '--output', str(out),
                           '--n', '50000', '--seed', str(args.seed), '--shard', str(rank), '--cfg', str(args.cfg),
                           '--stage2-steps', str(args.stage2_steps)]
                if args.policy_file is not None:
                    command.extend(['--policy-file', str(args.policy_file)])
                logpath = out/f'worker_shard{rank}.log'
                handle = logpath.open('x')
                child = subprocess.Popen(command, cwd=EVAL_HERE, env=env, stdout=handle,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                entry = dict(shard=rank, pid=child.pid, GPU=card, command=command, status='running', log=str(logpath))
                state['jobs'].append(entry)
                active.append((child, handle, card, entry))
            for item in list(active):
                child, handle, card, entry = item
                code = child.poll()
                if code is None:
                    continue
                entry.update(exit_code=code, status='complete' if code == 0 else 'failed')
                handle.close()
                active.remove(item)
                assert code == 0, (entry['shard'], code, 'No automatic retry')
            progress = []
            for rank in range(4):
                path = out/f'status_shard{rank}.json'
                if path.exists():
                    progress.append(json.loads(path.read_text()).get('completed', 0))
            update(status='generation_and_ADM_features', completed=sum(progress))
            if pending or active:
                time.sleep(10)
        check_sources()
        manifests, receipts = module('parallel_eval_compact').verify_shards(out, 50000, digest, 4, None)
        assert all(m['router'] == identity and m['routing_policy'] == policy for m in manifests)
        assert all(m['stage2']['steps'] == args.stage2_steps and m['stage2']['cfg_w'] == args.cfg for m in manifests)
        assert all(m['source_sha256'] == sha256(EVAL_HERE/'eval_worker.py') for m in manifests)
        assert all(m['selection_source_sha256'] == sha256(EVAL_HERE/'routing.py') for m in manifests)
        hist = np.zeros(257, dtype=np.int64)
        baseline_root = None
        if document is not None:
            baseline_root = Path(document['baseline_root'])
            assert document['baseline_summary_sha256'] == sha256(baseline_root/'summary.json')
        base_feature_max_abs_delta = 0.0
        base_feature_bitwise_exact = True
        for rank, receipt in enumerate(receipts):
            assert receipt['same_base_singleton_reverse_masks_K_exact'] and receipt['one_forward_one_final_selection']
            arrays = {}
            for name, expected in receipt['dump_hashes'].items():
                path = out/f'{name}_shard{rank}.npy'
                assert sha256(path) == expected
                arrays[name] = np.load(path, mmap_mode='r')
                assert np.isfinite(arrays[name]).all()
            ids_path = out/f'ids_shard{rank}.npy'
            assert sha256(ids_path) == receipt['ids_sha256']
            ids = np.load(ids_path)
            assert ids.tolist() == receipt['ids']
            n = len(ids)
            assert arrays['local_scores'].shape == arrays['router_scores'].shape == arrays['route_mask'].shape == (n, 256)
            assert arrays['utility'].shape == arrays['route_K'].shape == arrays['learned_K'].shape == (n,)
            assert arrays['z1d'].shape == (n, 32)
            assert arrays['local_scores'].dtype == arrays['router_scores'].dtype == arrays['utility'].dtype == np.float32
            assert arrays['route_mask'].dtype == np.uint8 and arrays['route_K'].dtype == arrays['z1d'].dtype == np.uint16
            assert np.all(arrays['z1d'] < 4096) and np.all(np.abs(arrays['utility']) <= 3+1e-6)
            learned = np.clip((arrays['router_scores'] >= identity['threshold']).sum(1), 48, 144)
            assert np.array_equal(learned, arrays['learned_K'])
            expected_mask = np.zeros((n, 256), dtype=np.uint8)
            if policy['policy'] == 'nearconstant':
                expected_K = nearconstant_k(ids, policy['target_mean_K'], policy['total'], policy['route_seed']+args.seed)
                order = np.argsort(-arrays['router_scores'], axis=1, kind='stable')
                np.put_along_axis(expected_mask, order,
                    (np.arange(256)[None] < expected_K[:, None]).astype('uint8'), axis=1)
            else:
                expected_K = learned
                positions = (random_positions(ids, learned, policy['route_seed'], args.seed, 'generation')
                    if policy['policy'] == 'random' else
                    [np.sort(np.asarray(uniform_order()[:int(k)])) for k in learned])
                for row, chosen in enumerate(positions):
                    expected_mask[row, chosen] = 1
            assert np.array_equal(expected_K, arrays['route_K'])
            assert np.array_equal(expected_mask, arrays['route_mask'])
            if policy['policy'] != 'nearconstant':
                assert np.array_equal(arrays['route_K'], np.load(Path(document['baseline_root'])/f'route_K_shard{rank}.npy'))
            hist += np.bincount(expected_K, minlength=257)
            if baseline_root is not None:
                for name in ('ids', 'z1d', 'local_scores', 'router_scores', 'utility'):
                    current = np.load(out/f'{name}_shard{rank}.npy', mmap_mode='r')
                    reference = np.load(baseline_root/f'{name}_shard{rank}.npy', mmap_mode='r')
                    assert np.array_equal(current, reference), ('Same generated inputs/scorer changed', rank, name)
                current_base = np.load(out/f'base_shard{rank}.npy', mmap_mode='r')
                reference_base = np.load(baseline_root/f'base_shard{rank}.npy', mmap_mode='r')
                assert current_base.shape == reference_base.shape
                for start in range(0, len(current_base), 500):
                    current_batch = current_base[start:start+500]
                    reference_batch = reference_base[start:start+500]
                    base_feature_bitwise_exact &= bool(np.array_equal(current_batch, reference_batch))
                    delta = np.abs(current_batch-reference_batch)
                    base_feature_max_abs_delta = max(base_feature_max_abs_delta, float(delta.max()))
                assert base_feature_max_abs_delta <= 2e-5, base_feature_max_abs_delta
        assert hist.sum() == 50000 and not hist[:48].any() and not hist[145:].any()
        mean_k = float(hist@np.arange(257)/50000)
        assert document is not None
        if policy['policy'] == 'nearconstant':
            assert int(hist@np.arange(257)) == round(policy['target_mean_K']*50000)
        else:
            baseline_summary = json.loads((Path(document['baseline_root'])/'summary.json').read_text())
            assert mean_k == baseline_summary['tokens']['mean_generated_2d']
        update(status='CPU_ADM_gFID_and_independent_FP64_audit')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        spec = importlib.util.spec_from_file_location('final_router_ADM_statistics', EVALUATOR)
        adm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adm)
        metrics = {}
        with np.load(ASSETS/'fid/VIRTUAL_imagenet256_labeled.npz') as reference:
            reference_mean, reference_cov = reference['mu'], reference['sigma']
            ref = adm.FIDStatistics(reference_mean, reference_cov)
            for arm in ('base', 'full_refine'):
                features = np.concatenate([np.load(out/f'{arm}_shard{rank}.npy') for rank in range(4)])
                assert features.shape == (50000, 2048) and np.isfinite(features).all()
                # Preserve the original evaluator's FP32 mean and np.cov.
                stat = adm.FIDStatistics(features.mean(0), np.cov(features, rowvar=False))
                value = float(stat.frechet_distance(ref))
                assert np.isfinite(value) and value >= 0
                centered = features.astype(np.float64)
                mean = centered.mean(0)
                centered -= mean
                covariance = centered.T@centered/(len(centered)-1)
                product_root = sqrtm(covariance@reference_cov.astype(np.float64))
                imaginary_max = float(np.abs(product_root.imag).max()) if np.iscomplexobj(product_root) else 0.
                assert np.isfinite(product_root).all() and imaginary_max < 1e-6
                independent = float(((mean-reference_mean.astype(np.float64))**2).sum()+np.trace(covariance)
                                    +np.trace(reference_cov)-2*np.trace(product_root.real))
                assert abs(value-independent) < 1e-4
                metrics[arm] = dict(gFID=value, gFID_independent_FP64=independent,
                    absolute_difference=abs(value-independent), sqrtm_imaginary_max=imaginary_max,
                    independent_mean_and_covariance='explicitly centered FP64, unbiased covariance',
                    independent_jitter_or_eigenvalue_clipping=False,
                    original_ADM_FP32_mean_preserved=True)
                del features, centered
        check_sources()
        result = dict(status='complete', kind=args.kind, n=50000, seed=args.seed, step=20000,
            metrics=metrics, tokens=dict(mean_1d=32, mean_generated_2d=mean_k,
                mean_generated_total=32+mean_k, histogram=hist.tolist(),
                min_K=int(np.flatnonzero(hist)[0]), max_K=int(np.flatnonzero(hist)[-1])),
            router=identity, checkpoint_sha256=digest, seconds=time.monotonic()-started,
            anchors_unchanged=True, generation_rounds=f'8+{args.stage2_steps}',
            cfg_stage1=4.5, cfg_stage2=args.cfg, routing_policy=policy,
            routing_policy_file_sha256=sha256(args.policy_file) if args.policy_file else None,
            all_masks_and_K_replayed=True, all_routes_and_inputs_saved=True,
            same_generated_inputs_and_scorer_exact_baseline=baseline_root is not None and base_feature_bitwise_exact,
            same_generated_inputs_and_scorer_within_2e5_baseline=baseline_root is not None,
            base_feature_max_abs_delta=base_feature_max_abs_delta,
            base_feature_max_abs_delta_tolerance=2e-5,
            independently_trained_control=True, same_checkpoint_as_main=False,
            training_control_identity=document['training_control_identity'],
            inference_control_target_mean_K=policy['target_mean_K'],
            nearconstant_offline_balanced_diagnostic=policy['policy']=='nearconstant',
            source_sha256=frozen, objective_complete=False)
        atomic_json(out/'summary.json', result)
        update(status='complete', completed=50000)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        update(status='failed_preserved_no_retry', error=repr(error), traceback=traceback.format_exc())
        for child, handle, _, _ in active:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            handle.close()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=KINDS, required=True)
    p.add_argument('--cfg', type=float, choices=[2.5, 2.75, 3., 3.25, 3.5], required=True)
    p.add_argument('--stage2-steps', type=int, choices=[1, 2, 4, 6, 8, 16, 32, 48, 64], default=8)
    p.add_argument('--seed', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy-file', type=Path)
    run(p.parse_args())
