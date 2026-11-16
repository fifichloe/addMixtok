"""Freeze both scalar thresholds from all 50k fresh scores, without image quality."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from density_free_config import HERE, ROOT, OUTPUT, VAL_PATHS, KINDS, CAM, UTILITY_WEIGHT, sha, write_json, source_identities


def calibrate(scores, target=96):
    lo, hi = float(scores.min())-1, float(scores.max())+1
    best = (float('inf'), None, None)
    for _ in range(80):
        tau = float(np.float32((lo+hi)/2))
        k = np.clip((scores >= tau).sum(1), 48, 144)
        error = abs(float(k.mean())-target)
        if error < best[0]: best = (error, tau, k)
        if k.mean() > target: lo = tau
        else: hi = tau
    assert best[0] < .01
    return best[1], best[2]


def selection(scores, k):
    mask = np.zeros_like(scores, dtype='uint8')
    order = np.argsort(-scores, axis=1, kind='stable')
    np.put_along_axis(mask, order, (np.arange(256)[None] < k[:, None]).astype('uint8'), axis=1)
    assert np.array_equal(mask.sum(1), k)
    return mask


def main():
    p = argparse.ArgumentParser(); p.add_argument('--world', type=int, required=True)
    args = p.parse_args()
    folders = [OUTPUT/'validation'/f'shard_{i}' for i in range(args.world)]
    paths = []
    sources = source_identities()
    for i, folder in enumerate(folders):
        row = json.loads((folder/'report.json').read_text())
        assert row['status'] == 'fresh_validation_scores_complete' and row['rank'] == i and row['world'] == args.world
        assert not row['quality_used'] and not row['density_computed']
        assert row['source_sha256'] == sources
        assert sha(folder/'paths.txt') == row['paths_sha256']
        for name, digest in row['array_sha256'].items(): assert sha(folder/(name+'.npy')) == digest
        paths.extend((folder/'paths.txt').read_text().splitlines())
    assert paths == VAL_PATHS.read_text().splitlines() and len(paths) == len(set(paths)) == 50000
    local = np.concatenate([np.load(f/'local.npy') for f in folders])
    utility = np.concatenate([np.load(f/'utility.npy') for f in folders])
    assert local.shape == (50000, 2, 256) and utility.shape == (50000,)
    assert np.isfinite(local).all() and np.isfinite(utility).all()
    policies, replay = {}, {}
    directory = OUTPUT/'teacher_policies'; directory.mkdir(exist_ok=False)
    (directory/'paths.txt').write_text('\n'.join(paths)+'\n')
    for index, kind in enumerate(KINDS):
        scores = local[:, index] + UTILITY_WEIGHT[kind]*utility[:, None]
        tau, k = calibrate(scores)
        mask = selection(scores, k)
        np.save(directory/f'{kind}.scores.npy', scores)
        np.save(directory/f'{kind}.K.npy', k.astype('int16'))
        np.save(directory/f'{kind}.mask.npy', mask)
        policies[kind] = dict(name=f'{kind}_density_free_c{CAM[kind]}_u{UTILITY_WEIGHT[kind]}',
            weight=UTILITY_WEIGHT[kind], threshold=tau, density=0., cam=CAM[kind], white=.5,
            teacher_rFID=None, validation_mean_K=float(k.mean()), min_K=int(k.min()), max_K=int(k.max()),
            paths_sha256=sha(directory/'paths.txt'),
            mask_file=str(directory/f'{kind}.mask.npy'), mask_sha256=sha(directory/f'{kind}.mask.npy'),
            K_file=str(directory/f'{kind}.K.npy'), K_sha256=sha(directory/f'{kind}.K.npy'),
            source='fresh density-free computation; score-only 50k validation budget calibration',
            quality_used_for_threshold=False)
        # Historical subtraction-based caches are diagnostic evidence only.
        oldroot = OUTPUT.parent/'new_router_reconstruction_ablations_20261114'/kind/'scores'
        if all((oldroot/f'shard_{i}'/'without_density.npy').exists() for i in range(4)):
            oldpaths = sum([(oldroot/f'shard_{i}'/'paths.txt').read_text().splitlines() for i in range(4)], [])
            assert oldpaths == paths
            oldscores = np.concatenate([np.load(oldroot/f'shard_{i}'/'without_density.npy') for i in range(4)])
            oldtau, oldk = calibrate(oldscores)
            oldmask = selection(oldscores, oldk)
            replay[kind] = dict(max_absolute_score_difference=float(np.abs(scores-oldscores).max()),
                RMS_score_difference=float(np.sqrt(np.mean((scores.astype('float64')-oldscores)**2))),
                old_subtraction_cache_threshold=oldtau, new_threshold=tau,
                K_changed_images=int(np.count_nonzero(k != oldk)),
                mask_changed_images=int(np.count_nonzero(np.any(mask != oldmask, axis=1))),
                changed_locations=int(np.count_nonzero(mask != oldmask)),
                identical_masks=np.array_equal(mask, oldmask))
            if kind == 'long':
                trial = json.loads((HERE/'policies/protocol.json').read_text())['variants'][0]
                for field, hashfield in [('mask_file', 'mask_sha256'), ('K_file', 'K_sha256')]:
                    assert sha(trial[field]) == trial[hashfield]
                trial_mask = np.load(trial['mask_file']); trial_k = np.load(trial['K_file'])
                replay[kind].update(trial_K_exact=np.array_equal(k, trial_k),
                                    trial_mask_exact=np.array_equal(mask, trial_mask),
                                    quality_reuse_only_if_exact_masks=True)
    record = dict(status='full_validation_density_free_scores_and_policies_frozen', time=time.time(),
        validation_n=50000, validation_paths_sha256=sha(VAL_PATHS), source_sha256=sources,
        policies=policies, fresh_vs_subtraction_replay=replay,
        no_density_computation_or_label_subtraction=True, quality_used_for_calibration=False,
        calibration='80-iteration FP32 scalar bisection to mean K96; individual K clamped48..144',
        extraction_world=args.world,
        validation_reports=[dict(path=str(f/'report.json'), sha256=sha(f/'report.json')) for f in folders])
    assert not (HERE/'teacher_policies.json').exists()
    write_json(HERE/'teacher_policies.json', record)
    write_json(directory/'protocol.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__': main()
