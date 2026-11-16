"""Independent full-mask replay and full50k centered-covariance quality audit."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from density_free_config import HERE, OUTPUT, VAL_PATHS, UTILITY_WEIGHT, sha, write_json

ROOT = HERE/'reconstruction_controls'
HELPER = HERE.parent/'new_router_ablation_suite_20261114/analysis/audit_reconstruction_results.py'


def policies(kind):
    folder = ROOT/kind/'policies'
    output = ROOT/kind/'policy_independent_audit.json'
    assert not output.exists()
    protocol_path = folder/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['status'] == 'policies_frozen_before_quality' and protocol['kind'] == kind
    assert not protocol['quality_used_for_calibration'] and protocol['no_new_reconstruction_seeds']
    assert protocol['component_design_sha256'] == sha(HERE/'design_lock.json')
    assert sha(folder/'paths.txt') == protocol['paths_sha256']
    assert (folder/'paths.txt').read_text() == VAL_PATHS.read_text()
    design = json.loads((HERE/'design_lock.json').read_text())['teachers'][kind]
    index = ('short', 'long').index(kind)
    local = np.concatenate([np.load(OUTPUT/'validation'/f'shard_{i}/local.npy')[:, index] for i in range(3)])
    utility = np.concatenate([np.load(OUTPUT/'validation'/f'shard_{i}/utility.npy') for i in range(3)])
    scores = local+np.float32(UTILITY_WEIGHT[kind])*utility[:, None]
    native_K = np.minimum(144, np.maximum(48, np.sum(scores >= design['threshold'], axis=1)))
    order = np.argsort(-scores, axis=1, kind='stable')
    shuffle = np.random.default_rng(1).permutation(50000)
    random = np.argsort(np.random.default_rng(1).random((50000, 256)), axis=1, kind='stable')
    # Independent array implementation of greedy farthest spatial order.
    coordinates = np.stack(np.unravel_index(np.arange(256), (16, 16)), axis=1)
    distances = np.full(256, np.inf)
    chosen, point = [], 0
    for _ in range(256):
        chosen.append(point)
        distances = np.minimum(distances, np.square(coordinates-coordinates[point]).sum(1))
        distances[chosen] = -1
        point = int(np.argmax(distances))
    uniform = np.asarray(chosen)
    archive = OUTPUT.parent/'new_router_reconstruction_ablations_20261114/long/scores'
    entropy = np.concatenate([np.load(archive/f'shard_{i}/gradient_entropy.npy') for i in range(4)])
    entropy_order = np.argsort(-entropy, axis=1, kind='stable')
    rows = {row['name']:row for row in protocol['variants']}
    assert len(rows) == 19
    dynamic = {96:native_K}
    for target in (64, 128):
        tau = rows[f'dynamic{target}']['calibration']['threshold']
        dynamic[target] = np.maximum(48, np.minimum(144, np.sum(scores >= tau, axis=1)))
    result = {}
    for name, row in rows.items():
        ranking = order
        if name == 'teacher_dynamic96':
            k = native_K
        elif name.startswith('rank_'):
            k = native_K
            ranking = {'rank_random':random, 'rank_uniform':uniform,
                       'rank_gradient_entropy':entropy_order}[name]
        elif name.startswith('fixed'):
            k = np.full(50000, int(name[5:]))
        elif name.startswith('dynamic'):
            k = dynamic[int(name[7:])]
        elif name.startswith('shuffled'):
            k = dynamic[int(name[8:])][shuffle]
        elif name.startswith('matched_mean'):
            source = dynamic[int(name[12:])]
            low, remainder = divmod(int(source.sum()), 50000)
            k = np.full(50000, low)
            k[shuffle[:remainder]] += 1
            assert int(k.sum()) == int(source.sum())
        else:
            raise AssertionError(name)
        assert sha(row['mask_file']) == row['mask_sha256']
        assert sha(row['K_file']) == row['K_sha256']
        saved_k = np.load(row['K_file'])
        saved = np.load(row['mask_file'], mmap_mode='r')
        assert saved.shape == (50000, 256) and saved.dtype == np.uint8
        assert np.array_equal(k, saved_k) and np.array_equal(saved.sum(1), saved_k)
        ranking = np.broadcast_to(ranking, (50000, 256))
        for start in range(0, 50000, 1000):
            ii = np.arange(start, min(start+1000, 50000))
            expected = np.zeros((len(ii), 256), dtype=np.uint8)
            expected[np.arange(len(ii))[:, None], ranking[ii]] = np.arange(256)[None] < k[ii, None]
            assert np.array_equal(expected, saved[ii]), (kind, name, start)
        assert row['mean_K'] == float(k.mean())
        assert row['bpp_mask256'] == float((384+14*k.mean()+256)/65536)
        result[name] = dict(n=50000, full_mask_and_K_replay_exact=True,
                            mean_K=float(k.mean()), position_mask_included_in_bpp=True)
    write_json(output, dict(status='all19_current_reconstruction_masks_and_K_replayed',
        kind=kind, rows=result, protocol_sha256=sha(protocol_path),
        quality_used=False, GPU_used=False, new_reconstruction_seeds=False,
        auditor_source_sha256=sha(Path(__file__)), time=time.time()))


def quality(kind):
    output = ROOT/kind/'quality_independent_audit.json'
    assert not output.exists()
    spec = importlib.util.spec_from_file_location('current_reconstruction_independent_math', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    auditor = helper.Auditor(output)
    protocol_path = ROOT/kind/'policies/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    policy_audit = ROOT/kind/'policy_independent_audit.json'
    replay = json.loads(policy_audit.read_text())
    assert replay['status'] == 'all19_current_reconstruction_masks_and_K_replayed'
    assert replay['protocol_sha256'] == auditor.sha(protocol_path)
    result_path = ROOT/kind/'new_quality_results.json'
    new = json.loads(result_path.read_text())
    assert new['status'] == 'full_50000_quality_complete'
    assert new['protocol_sha256'] == auditor.sha(protocol_path)
    official = {row['name']:row for row in new['rows']}
    paths, real, fake = [], [], {name:[] for name in official}
    sums = {name:0. for name in official}
    for rank, item in enumerate(new['reports']):
        path = Path(item['path'])
        auditor.sha(path, item['sha256'])
        report = json.loads(path.read_text())
        assert report['status'] == 'quality_features_complete' and report['shard'] == rank
        assert set(report['names']) == set(official)
        assert report['policy_sha256'] == auditor.sha(protocol_path)
        auditor.hash_map(report['source_sha256'])
        folder = path.parent
        auditor.sha(folder/'paths.txt', report['paths_sha256'])
        paths.extend((folder/'paths.txt').read_text().splitlines())
        auditor.sha(folder/'real.npy', report['real_sha256'])
        helper.finite_array(folder/'real.npy', (report['n'], 2048), 'float32')
        real.append(folder/'real.npy')
        for name in official:
            fp = folder/(name+'.features.npy')
            auditor.sha(fp, report['feature_sha256'][name])
            helper.finite_array(fp, (report['n'], 2048), 'float32')
            fake[name].append(fp)
            sums[name] += report['PSNR_sums'][name]
    assert paths == VAL_PATHS.read_text().splitlines() and len(set(paths)) == 50000
    real_moments = helper.centered_moments(real)
    rows = {}
    for policy in protocol['variants']:
        name = policy['name']
        auditor.sha(policy['mask_file'], policy['mask_sha256'])
        auditor.sha(policy['K_file'], policy['K_sha256'])
        reuse = policy['exact_existing_quality_reuse']
        if reuse is not None:
            auditor.sha(reuse['audit'], reuse['audit_sha256'])
            prior = json.loads(Path(reuse['audit']).read_text())
            assert prior['protocol_sha256'] == auditor.sha(reuse['protocol'], reuse['protocol_sha256'])
            original_protocol = json.loads(Path(reuse['protocol']).read_text())
            original = next(row for row in original_protocol['variants'] if row['name'] == reuse['source_policy'])
            assert original['mask_sha256'] == policy['mask_sha256']
            assert np.array_equal(np.load(original['K_file']), np.load(policy['K_file']))
            values = prior['rows'][reuse['source_policy']]
            assert values == reuse['metrics']
            rfid, psnr = values['rFID'], values['PSNR']
            independent = values['independent_FID']
        else:
            original = official[name]
            independent = helper.independent_fid(real_moments, fake[name])
            rfid, psnr = original['rFID'], original['PSNR']
            assert abs(independent['rFID_sqrtm']-rfid) < 1e-7
            assert abs(sums[name]/50000-psnr) < 1e-12
        rows[name] = dict(**policy, n=50000, rFID=rfid, PSNR=psnr,
            independent_FID=independent, identical_mask_quality_reused=reuse is not None,
            PSNR_actual_shard_sums_remerged=True, PSNR_not_independently_redecoded=True)
        auditor.progress('current_reconstruction_independent_quality', completed=list(rows), total=19)
    write_json(output, dict(status='all19_current_reconstruction_qualities_independently_verified',
        kind=kind, rows=rows, n=50000, protocol_sha256=auditor.sha(protocol_path),
        result_sha256=auditor.sha(result_path), policy_audit_sha256=auditor.sha(policy_audit),
        verified_file_hashes=auditor.hashes, auditor_source_sha256=sha(Path(__file__)),
        helper_source_sha256=sha(HELPER), time=time.time(), objective_complete=False,
        interpretation='Frozen-codec inference controls, not independently trained per-K optimal curves',
        no_training_seed_variability_claim=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['short', 'long'], required=True)
    p.add_argument('--mode', choices=['policies', 'quality'], required=True)
    args = p.parse_args()
    (policies if args.mode == 'policies' else quality)(args.kind)
