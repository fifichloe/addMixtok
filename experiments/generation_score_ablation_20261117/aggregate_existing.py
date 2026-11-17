"""Audit finished 50k shards and aggregate ADM gFID after numeric GPU drift.

The generation workers and their output files are never rerun or modified here.
The original launchers required bitwise equality of ADM base features across
different physical GPUs. This separate audit keeps exact generated inputs and
routing checks, but bounds the observed GPU floating point drift explicitly.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import sqrtm

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / 'no_density_revision_20261114/generation_eval'
sys.path.insert(0, str(EVAL.parent / 'maskgit'))
from common import ASSETS, module
from bert2d.paths import atomic_json, sha256, EVALUATOR

BASELINE = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/'
                'no_density_revision_20261114/generation_evaluation/long_cfg3_seed4_8p8')
LIMIT_MAX = 3e-4
LIMIT_MEAN = 5e-7


def check_same_base(out, rank):
    """The same generated sample may produce slightly different ADM FP32 activations."""
    current = np.load(out / f'base_shard{rank}.npy', mmap_mode='r')
    baseline = np.load(BASELINE / f'base_shard{rank}.npy', mmap_mode='r')
    assert current.shape == baseline.shape and current.dtype == baseline.dtype == np.float32
    difference = np.abs(np.asarray(current) - np.asarray(baseline))
    max_diff, mean_diff = float(difference.max()), float(difference.mean(dtype=np.float64))
    assert max_diff <= LIMIT_MAX and mean_diff <= LIMIT_MEAN, (rank, max_diff, mean_diff)
    return dict(shard=rank, max_abs=max_diff, mean_abs=mean_diff,
                exact=bool(np.array_equal(current, baseline)))


def run(out):
    start = time.monotonic()
    config = json.loads((out / 'parallel_config.json').read_text())
    policy = config['routing_policy']
    is_u0 = policy['policy'] == 'U_zero'
    kind = 'U_zero' if is_u0 else config['ablation']
    policy_file = HERE / ('u_zero_policy_seed4.json' if is_u0 else f'policy_{kind}_seed4.json')
    document = json.loads(policy_file.read_text())
    digest = config['checkpoint_sha256']
    assert config['n'] == 50000 and config['seed'] == 4
    assert config['stage1_steps'] == config['stage2_steps'] == 8
    assert config['cfg_stage1'] == 4.5 and config['cfg_stage2'] == 3.
    assert document['selection_policy'] == policy and document['checkpoint_sha256'] == digest
    assert document['baseline_summary_sha256'] == sha256(BASELINE / 'summary.json')
    assert digest == sha256(Path(config['checkpoint']))
    assert all(sha256(Path(path)) == expected for path, expected in config['source_sha256'].items())
    manifests, receipts = module('parallel_eval_compact').verify_shards(out, 50000, digest, 4, None)
    assert all(m['router'] == config['router'] and m['routing_policy'] == policy for m in manifests)
    assert all(m['stage2']['steps'] == 8 and m['stage2']['cfg_w'] == 3. for m in manifests)
    hist = np.zeros(257, dtype=np.int64)
    drift = []
    for rank, receipt in enumerate(receipts):
        assert receipt['same_base_singleton_reverse_masks_K_exact']
        assert receipt['one_forward_one_final_selection']
        arrays = {}
        for name, expected in receipt['dump_hashes'].items():
            path = out / f'{name}_shard{rank}.npy'
            assert sha256(path) == expected, (rank, name)
            arrays[name] = np.load(path, mmap_mode='r')
            assert np.isfinite(arrays[name]).all(), (rank, name)
        ids = np.load(out / f'ids_shard{rank}.npy', mmap_mode='r')
        assert sha256(out / f'ids_shard{rank}.npy') == receipt['ids_sha256']
        assert ids.tolist() == receipt['ids']
        n = len(ids)
        assert arrays['local_scores'].shape == arrays['router_scores'].shape == arrays['route_mask'].shape == (n, 256)
        assert arrays['utility'].shape == arrays['route_K'].shape == (n,)
        assert arrays['z1d'].shape == (n, 32)
        assert arrays['route_K'].dtype == arrays['z1d'].dtype == np.uint16
        assert arrays['route_mask'].dtype == np.uint8
        score = arrays['local_scores'] if is_u0 else arrays['router_scores']
        expected_k = np.clip((score >= policy['threshold']).sum(1), 48, 144)
        assert np.array_equal(expected_k, arrays['route_K'])
        order = np.argsort(-score, axis=1, kind='stable')
        expected_mask = np.zeros((n, 256), dtype=np.uint8)
        np.put_along_axis(expected_mask, order,
                           (np.arange(256)[None] < expected_k[:, None]).astype(np.uint8), axis=1)
        assert np.array_equal(expected_mask, arrays['route_mask'])
        hist += np.bincount(expected_k, minlength=257)
        for name in ('ids', 'z1d'):
            current = np.load(out / f'{name}_shard{rank}.npy', mmap_mode='r')
            baseline = np.load(BASELINE / f'{name}_shard{rank}.npy', mmap_mode='r')
            assert np.array_equal(current, baseline), (rank, name)
        if is_u0:
            for name in ('local_scores', 'router_scores', 'utility'):
                current = np.load(out / f'{name}_shard{rank}.npy', mmap_mode='r')
                baseline = np.load(BASELINE / f'{name}_shard{rank}.npy', mmap_mode='r')
                assert np.array_equal(current, baseline), (rank, name)
        drift.append(check_same_base(out, rank))
    assert hist.sum() == 50000 and not hist[:48].any() and not hist[145:].any()
    mean_k = float(hist @ np.arange(257) / 50000)
    assert abs(mean_k - document['target_mean_K']) <= .01
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    spec = importlib.util.spec_from_file_location('frozen_adm_statistics', EVALUATOR)
    adm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adm)
    metrics = {}
    with np.load(ASSETS / 'fid/VIRTUAL_imagenet256_labeled.npz') as reference:
        reference_mean, reference_cov = reference['mu'], reference['sigma']
        ref = adm.FIDStatistics(reference_mean, reference_cov)
        for arm in ('base', 'full_refine'):
            features = np.concatenate([np.load(out / f'{arm}_shard{rank}.npy') for rank in range(4)])
            assert features.shape == (50000, 2048) and np.isfinite(features).all()
            stat = adm.FIDStatistics(features.mean(0), np.cov(features, rowvar=False))
            value = float(stat.frechet_distance(ref))
            assert np.isfinite(value) and value >= 0
            centered = features.astype(np.float64)
            mean = centered.mean(0)
            centered -= mean
            covariance = centered.T @ centered / (len(centered) - 1)
            product_root = sqrtm(covariance @ reference_cov.astype(np.float64))
            imaginary_max = float(np.abs(product_root.imag).max()) if np.iscomplexobj(product_root) else 0.
            assert np.isfinite(product_root).all() and imaginary_max < 1e-6
            independent = float(((mean - reference_mean.astype(np.float64)) ** 2).sum()
                                + np.trace(covariance) + np.trace(reference_cov)
                                - 2 * np.trace(product_root.real))
            assert abs(value - independent) < 1e-4
            metrics[arm] = dict(gFID=value, gFID_independent_FP64=independent,
                                absolute_difference=abs(value - independent),
                                sqrtm_imaginary_max=imaginary_max,
                                independent_mean_and_covariance='explicitly centered FP64, unbiased covariance',
                                independent_jitter_or_eigenvalue_clipping=False,
                                original_ADM_FP32_mean_preserved=True)
            del features, centered
    result = dict(status='complete', kind=config['kind'], n=50000, seed=4, step=20000,
                  metrics=metrics, tokens=dict(mean_1d=32, mean_generated_2d=mean_k,
                  mean_generated_total=32 + mean_k, histogram=hist.tolist(),
                  min_K=int(np.flatnonzero(hist)[0]), max_K=int(np.flatnonzero(hist)[-1])),
                  router=config['router'], checkpoint_sha256=digest,
                  seconds=time.monotonic() - start, anchors_unchanged=True,
                  generation_rounds='8+8', cfg_stage1=4.5, cfg_stage2=3., routing_policy=policy,
                  routing_policy_file_sha256=sha256(policy_file), all_masks_and_K_replayed=True,
                  all_routes_and_inputs_saved=True, matched_baseline_mean_K=True,
                  same_generated_ids_and_1d_tokens_exact_baseline=True,
                  same_base_ADM_features_within_explicit_tolerance=True,
                  base_ADM_feature_drift=drift,
                  base_ADM_feature_tolerance=dict(max_abs=LIMIT_MAX, mean_abs=LIMIT_MEAN),
                  original_launcher_failed_on_bitwise_base_ADM_features=True,
                  original_launcher_source_sha256=config['source_sha256'],
                  aggregation_script_sha256=sha256(Path(__file__)), objective_complete=False)
    if is_u0:
        result['same_scorer_exact_baseline'] = True
    else:
        result['ablation'] = kind
        result['same_scorer_exact_baseline'] = False
    atomic_json(out / 'summary.json', result)
    old_status = json.loads((out / 'status.json').read_text())
    old_status['previous_failure'] = dict(error=old_status.get('error'),
                                          traceback=old_status.get('traceback'))
    old_status.update(status='complete_by_independent_aggregate', completed=50000,
                      aggregation_script_sha256=result['aggregation_script_sha256'])
    old_status.pop('error', None)
    old_status.pop('traceback', None)
    atomic_json(out / 'status.json', old_status)
    print(json.dumps(dict(kind=kind, gFID=metrics['full_refine']['gFID'], mean_K=mean_k,
                          max_base_feature_difference=max(x['max_abs'] for x in drift))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
