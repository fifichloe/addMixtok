"""Freeze current-design experiments1/2/3, without extra reconstruction seeds.

Fresh density-free score tensors are authoritative. The cached image-only
gradient-entropy heuristic can be reused after checking its source/array/path
identities; no historical learned scorer, threshold, top-p, or mask is reused.
Existing component quality may be reused only for identical full50k masks/K.
"""
import json
from pathlib import Path
import time

import numpy as np

from density_free_config import (HERE, ROOT, OUTPUT, VAL_PATHS, KINDS,
    UTILITY_WEIGHT, source_identities, sha, write_json)
from final_teacher_config import load_design
from freeze_teachers import calibrate, selection

OUT = HERE/'reconstruction_controls'
ARCHIVE = OUTPUT.parent/'new_router_reconstruction_ablations_20261114/long/scores'
CONTROL_SEED = 1


def uniform_order():
    coords = [(r, c) for r in range(16) for c in range(16)]
    selected = [coords.pop(0)]
    while coords:
        index = max(range(len(coords)), key=lambda i:(
            min((coords[i][0]-r)**2+(coords[i][1]-c)**2 for r, c in selected),
            -coords[i][0], -coords[i][1]))
        selected.append(coords.pop(index))
    return np.asarray([r*16+c for r, c in selected], dtype=np.int16)


def main():
    assert not OUT.exists(), 'Never overwrite frozen masks or existing quality'
    design = load_design()
    sources = source_identities()
    folders = [OUTPUT/'validation'/f'shard_{rank}' for rank in range(3)]
    paths = []
    score_reports = []
    for rank, folder in enumerate(folders):
        report_path = folder/'report.json'
        report = json.loads(report_path.read_text())
        assert report['status'] == 'fresh_validation_scores_complete'
        assert report['rank'] == rank and report['world'] == 3
        assert not report['quality_used'] and not report['density_computed']
        assert report['source_sha256'] == sources
        assert sha(folder/'paths.txt') == report['paths_sha256']
        for name, expected in report['array_sha256'].items():
            assert sha(folder/(name+'.npy')) == expected, (rank, name)
        paths.extend((folder/'paths.txt').read_text().splitlines())
        score_reports.append(dict(path=str(report_path), sha256=sha(report_path)))
    assert paths == VAL_PATHS.read_text().splitlines()
    assert len(paths) == len(set(paths)) == 50000
    arrays = {name:np.concatenate([np.load(folder/(name+'.npy')) for folder in folders])
              for name in ('local', 'utility')}
    assert arrays['local'].shape == (50000, 2, 256) and arrays['utility'].shape == (50000,)
    assert all(np.isfinite(value).all() for value in arrays.values())

    # This heuristic uses only target RGB, so it is invariant to router changes.
    heuristic, heuristic_paths, heuristic_reports = [], [], []
    heuristic_source = ROOT/'experiments/rebuttal_20261106/original_recon_ablation.py'
    runtime_source = ROOT/'experiments/fresh_reconstruction_scorer_20261112/runtime.py'
    for rank in range(4):
        folder = ARCHIVE/f'shard_{rank}'
        report_path = folder/'report.json'
        report = json.loads(report_path.read_text())
        assert report['status'] == 'scores_complete' and report['shard'] == rank
        assert not report['quality_used'] and report['total'] == 50000
        assert sha(folder/'paths.txt') == report['paths_sha256']
        assert sha(heuristic_source) == report['source_sha256'][str(heuristic_source)]
        assert sha(runtime_source) == report['source_sha256'][str(runtime_source)]
        expected = report['score_sha256']['gradient_entropy']
        assert sha(folder/'gradient_entropy.npy') == expected
        heuristic.append(np.load(folder/'gradient_entropy.npy'))
        heuristic_paths.extend((folder/'paths.txt').read_text().splitlines())
        heuristic_reports.append(dict(path=str(report_path), sha256=sha(report_path),
            gradient_entropy_sha256=expected,
            reuse_scope='image-only gradient entropy, never historical learned scores'))
    assert heuristic_paths == paths
    entropy = np.concatenate(heuristic)
    assert entropy.shape == (50000, 256) and np.isfinite(entropy).all()
    entropy_order = np.argsort(-entropy, axis=1, kind='stable')
    uniform = uniform_order()
    random_order = np.argsort(np.random.default_rng(CONTROL_SEED).random((50000, 256)),
                              axis=1, kind='stable')
    permutation = np.random.default_rng(CONTROL_SEED).permutation(50000)
    manifests = {}
    OUT.mkdir()
    write_json(OUT/'predeclared_scope.json', dict(status='scope_frozen_before_new_control_quality',
        component_design_sha256=sha(HERE/'design_lock.json'), reconstruction_primary_kind='long',
        primary_reason='pre-existing full50k lower rFID; short PSNR tradeoff also reported',
        independent_of_generation_winner=True, fixed_K=[0, 32, 64, 96, 128, 196, 256],
        dynamic_target_K=[64, 96, 128], dynamic_limits=[48, 144],
        ranking=['teacher', 'random', 'uniform', 'gradient_entropy'], control_seed=CONTROL_SEED,
        one_selection_and_shuffle_seed=True, no_new_reconstruction_training_seeds=True,
        generation_student_ranking='separate policies after actual new student acceptance',
        B0_free_control='separate fresh score composition using independently RGB-trained head',
        old_top_p_and_legacy_router_excluded=True, quality_used_for_thresholds=False,
        curve='frozen codec/checkpoint inference curve, not independently trained per-K frontier'))
    for index, kind in enumerate(KINDS):
        folder = OUT/kind/'policies'
        folder.mkdir(parents=True)
        (folder/'paths.txt').write_text('\n'.join(paths)+'\n')
        record = design['teachers'][kind]
        assert record['white_direct_weight'] == .5
        scores = arrays['local'][:, index]+UTILITY_WEIGHT[kind]*arrays['utility'][:, None]
        order = np.argsort(-scores, axis=1, kind='stable')
        exact = np.clip((scores >= record['threshold']).sum(1), 48, 144)
        assert abs(float(exact.mean())-96) < .01
        gate_path = HERE/'design_gate'/kind/'policies/protocol.json'
        gate = json.loads(gate_path.read_text())
        gate_rows = {row['name']:row for row in gate['variants']}
        for row in gate_rows.values():
            assert sha(row['mask_file']) == row['mask_sha256']
            assert sha(row['K_file']) == row['K_sha256']
        rows = []

        def save(name, ranking, k, family, **metadata):
            k = np.asarray(k, dtype=np.int16)
            assert k.shape == (50000,) and 0 <= k.min() <= k.max() <= 256
            ranking = np.broadcast_to(ranking, (50000, 256))
            mask = np.zeros((50000, 256), dtype=np.uint8)
            np.put_along_axis(mask, ranking, (np.arange(256)[None] < k[:, None]).astype(np.uint8), axis=1)
            assert np.array_equal(mask.sum(1), k)
            mp, kp = folder/(name+'.mask.npy'), folder/(name+'.K.npy')
            np.save(mp, mask); np.save(kp, k)
            reuse = None
            for old in gate_rows.values():
                if sha(mp) == old['mask_sha256']:
                    assert np.array_equal(k, np.load(old['K_file']))
                    audit_path = HERE/'design_gate'/kind/'quality_independent_audit.json'
                    audit = json.loads(audit_path.read_text())
                    assert audit['status'] == 'all9_full50000_component_qualities_independently_verified'
                    assert audit['protocol_sha256'] == sha(gate_path)
                    reuse = dict(source_policy=old['name'], protocol=str(gate_path), protocol_sha256=sha(gate_path),
                        audit=str(audit_path), audit_sha256=sha(audit_path),
                        exact_mask_and_per_image_K=True,
                        proof_scope='same full50k inputs, codec, preprocessing and BF16 rendering protocol',
                        metrics=audit['rows'][old['name']])
                    break
            rows.append(dict(name=name, family=family, mean_K=float(k.mean()),
                min_K=int(k.min()), max_K=int(k.max()), mean_content_tokens=float(k.mean()+32),
                bpp_mask256=float((32*12+14*k.mean()+256)/65536),
                bpp_assumption='fixed-width 12/14-bit codewords plus raw256-bit position mask; no entropy coding',
                K256_still_accounts_for_32_global_tokens=True,
                mask_file=str(mp), mask_sha256=sha(mp), K_file=str(kp), K_sha256=sha(kp),
                density_removed=True, white_direct_weight=.5,
                exact_existing_quality_reuse=reuse, **metadata))

        save('teacher_dynamic96', order, exact, 'A01_A02_A03', threshold=record['threshold'])
        for label, ranking in [('uniform', uniform), ('random', random_order), ('gradient_entropy', entropy_order)]:
            save('rank_'+label, ranking, exact, 'A01', same_teacher_per_image_K=True,
                 selection_seed=CONTROL_SEED if label == 'random' else None)
        for k in (0, 32, 64, 96, 128, 196, 256):
            save(f'fixed{k}', order, np.full(50000, k), 'A02_A03', fixed_K=k,
                 frozen_checkpoint_inference_curve=True)
        for target in (64, 96, 128):
            if target == 96:
                tau, k = record['threshold'], exact
            else:
                tau, k = calibrate(scores, target)
                save(f'dynamic{target}', order, k, 'A03', calibration=dict(
                    threshold=tau, target=target, mean_K=float(k.mean()), quality_used=False))
            low, remainder = divmod(int(k.sum()), len(k))
            near = np.full(50000, low, dtype=np.int16)
            near[permutation[:remainder]] += 1
            assert near.sum() == k.sum()
            save(f'matched_mean{target}', order, near, 'A02_A03',
                 exact_dynamic_total_budget=True, content_independent_assignment=True,
                 not_strict_fixed_K=bool(remainder), assignment_seed=CONTROL_SEED)
            shuffled = k[permutation]
            assert np.array_equal(np.sort(shuffled), np.sort(k))
            save(f'shuffled{target}', order, shuffled, 'A02_A03',
                 permutation_seed=CONTROL_SEED, budget_histogram_exact=True)
        protocol_path = folder/'protocol.json'
        write_json(protocol_path, dict(status='policies_frozen_before_quality', kind=kind, n=50000,
            variants=rows, paths_sha256=sha(folder/'paths.txt'), source_reports=score_reports,
            image_only_heuristic_reports=heuristic_reports, source_sha256=sources,
            component_design_sha256=sha(HERE/'design_lock.json'),
            builder_source_sha256=sha(Path(__file__)), scope_sha256=sha(OUT/'predeclared_scope.json'),
            quality_used_for_calibration=False, current_dynamic_K_range=[48, 144],
            formal_teacher_policy=dict(kind=kind, threshold=record['threshold'],
                source='current density-free design lock', design_sha256=sha(HERE/'design_lock.json')),
            no_new_reconstruction_seeds=True, no_old_top_p_or_legacy=True,
            curve='frozen checkpoint inference, not separately trained per-budget optimum'))
        manifests[kind] = dict(protocol=str(protocol_path), sha256=sha(protocol_path),
            names=[row['name'] for row in rows],
            needs_new_quality=[row['name'] for row in rows if row['exact_existing_quality_reuse'] is None])
    write_json(OUT/'manifest.json', dict(status='both_current_teacher_control_policies_frozen',
        time=time.time(), kinds=manifests, component_design_sha256=sha(HERE/'design_lock.json'),
        reconstruction_primary_kind='long', objective_complete=False,
        no_new_GPU_context=True, B0_free_and_generation_student_policies_still_required=True))
    print(json.dumps(manifests), flush=True)


if __name__ == '__main__':
    main()
