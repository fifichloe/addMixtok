"""Match each ablated router's generated-image mean K to the main run.

The calibration reads only the main run's previously generated 1D IDs. It
does not use RGB images, generated 2D tokens, or any quality metric.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / 'no_density_revision_20261114/generation_eval'
sys.path.insert(0, str(EVAL.parent / 'maskgit'))
from common import ASSETS, TOP, module, router_features
from ablation_router import KINDS, GENERATOR_CHECKPOINT, load_router, router_identity
from bert2d.paths import atomic_json, sha256

BASELINE = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/'
                'no_density_revision_20261114/generation_evaluation/long_cfg3_seed4_8p8')


def calibrate(scores, target):
    assert scores.shape == (50000, 256) and np.isfinite(scores).all()
    lo, hi = float(scores.min()) - 1., float(scores.max()) + 1.
    best = (float('inf'), None, None)
    for _ in range(64):
        threshold = float(np.float32((lo + hi) / 2))
        mean = float(np.clip((scores >= threshold).sum(1), 48, 144).mean())
        candidate = (abs(mean - target), threshold, mean)
        if candidate[0] < best[0]:
            best = candidate
        if mean > target:
            lo = threshold
        else:
            hi = threshold
    assert best[0] <= .01, best
    return dict(threshold=best[1], mean_K=best[2], target_mean_K=target,
                absolute_mean_error=best[0], source='same-seed 50000 generated 1D IDs',
                score_only=True, quality_used=False, no_batch_quota=True)


@torch.inference_mode()
def main(kind, output):
    assert kind in KINDS and not output.exists()
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    baseline = json.loads((BASELINE / 'summary.json').read_text())
    assert baseline['status'] == 'complete' and baseline['n'] == 50000
    assert baseline['seed'] == 4 and baseline['generation_rounds'] == '8+8'
    assert baseline['cfg_stage2'] == 3.
    target = baseline['tokens']['mean_generated_2d']
    assert abs(target - 106.38386) < 1e-9
    checkpoint_sha = sha256(GENERATOR_CHECKPOINT)
    assert checkpoint_sha == baseline['checkpoint_sha256']
    identity = router_identity(kind)
    device = torch.device('cuda', 0)
    assets = module('assets_top_p').TopPFrozenAssets(ASSETS, TOP, device, chunk=4)
    router, _ = load_router(kind, device)
    blocks = []
    source_hashes = {}
    for shard in range(4):
        z_path = BASELINE / f'z1d_shard{shard}.npy'
        ids_path = BASELINE / f'ids_shard{shard}.npy'
        ids = np.load(ids_path, mmap_mode='r')
        z = np.load(z_path, mmap_mode='r')
        expected = np.concatenate(module('parallel_eval_compact').worker_batches(
            50000, 4, shard, None))
        assert np.array_equal(ids, expected) and z.shape == (len(ids), 32)
        assert z.dtype == np.uint16 and np.all(z < 4096)
        source_hashes[str(z_path)] = sha256(z_path)
        source_hashes[str(ids_path)] = sha256(ids_path)
        values = []
        for start in range(0, len(z), 8):
            tokens = torch.as_tensor(np.asarray(z[start:start+8], dtype=np.int64).copy(),
                                     device=device)
            features = router_features(tokens, assets.native.quantize,
                                       assets.shell.latent_decoder)
            with torch.autocast('cuda', enabled=False):
                scores = router(features.float())['score']
            values.append(scores.float().cpu().numpy())
        blocks.append(np.concatenate(values))
        print(json.dumps(dict(stage='scored', kind=kind, shard=shard,
                              images=len(ids), elapsed_seconds=time.monotonic()-started)),
              flush=True)
    scores = np.concatenate(blocks)
    choice = calibrate(scores, target)
    result = dict(status='frozen_before_control_quality', time=time.time(),
        kind='long', ablation=kind, cfg=3., stage2_steps=8, seed=4,
        checkpoint_sha256=checkpoint_sha, router_identity=identity,
        selection_policy=dict(policy='dynamic', minimum_K=48, maximum_K=144,
                              threshold=choice['threshold']),
        baseline_root=str(BASELINE), baseline_summary_sha256=sha256(BASELINE/'summary.json'),
        target_mean_K=target, calibration=choice, source_z1d_and_ids_sha256=source_hashes,
        expected_same_router_outputs=False, quality_used_for_budget_or_threshold=False,
        scope='same-frozen-generator-score-ablation-matched-mean-K',
        elapsed_seconds=time.monotonic()-started)
    atomic_json(output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=KINDS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.kind, args.output)
