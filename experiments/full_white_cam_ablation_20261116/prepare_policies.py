"""Freeze full-input White/CAM controls before looking at reconstruction quality."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'no_density_revision_20261114'
sys.path.insert(0, str(SOURCE))
from density_free_config import VAL_PATHS, sha, source_identities, write_json
from freeze_teachers import calibrate, selection


def main():
    folders = [HERE / f'score_shard{i}' for i in range(3)]
    paths = []
    reference = []
    maps = {'without_white_all_explicit': [], 'without_cam_all_explicit': []}
    frozen = source_identities()
    worker_digest = sha(HERE / 'score_worker.py')
    for i, folder in enumerate(folders):
        report = json.loads((folder / 'report.json').read_text())
        assert report['status'] == 'complete' and report['rank'] == i and report['world'] == 3
        assert report['source_sha256'] == frozen and report['worker_sha256'] == worker_digest
        assert sha(folder / 'paths.txt') == report['paths_sha256']
        paths.extend((folder / 'paths.txt').read_text().splitlines())
        for name, digest in report['scores_sha256'].items():
            assert sha(folder / (name + '.npy')) == digest
        reference.append(np.load(folder / 'reference.npy'))
        for name in maps:
            maps[name].append(np.load(folder / (name + '.npy')))
    assert paths == VAL_PATHS.read_text().splitlines() and len(paths) == 50000
    all_scores = {'full': np.concatenate(reference)}
    all_scores.update({name: np.concatenate(chunks) for name, chunks in maps.items()})
    target = HERE / 'policies'
    target.mkdir(parents=True, exist_ok=False)
    (target / 'paths.txt').write_text('\n'.join(paths) + '\n')
    rows = []
    for component, scores in all_scores.items():
        assert scores.shape == (50000, 256) and np.isfinite(scores).all()
        threshold, dynamic_k = calibrate(scores)
        for budget, k in [('dynamic96', dynamic_k), ('fixed96', np.full(50000, 96, dtype='int16'))]:
            name = component + '_' + budget
            mask = selection(scores, k)
            mask_path, k_path = target / (name + '.mask.npy'), target / (name + '.K.npy')
            np.save(mask_path, mask)
            np.save(k_path, k.astype('int16'))
            rows.append(dict(name=name, family='frozen_head_full_input_component_ablation', component=component,
                budget=budget, mean_K=float(k.mean()), min_K=int(k.min()), max_K=int(k.max()),
                mean_content_tokens=float(k.mean() + 32),
                bpp_mask256=float((32 * 12 + 14 * k.mean() + 256) / 65536),
                mask_file=str(mask_path), mask_sha256=sha(mask_path),
                K_file=str(k_path), K_sha256=sha(k_path),
                calibration=dict(threshold=threshold, target=96, mean_K=float(k.mean()), quality_used=False)
                    if budget == 'dynamic96' else dict(fixed_K=96),
                head_retrained=False, training_mean_imputation=True))
    protocol = dict(status='policies_frozen_before_quality', kind='long', n=50000, variants=rows,
        paths_sha256=sha(target / 'paths.txt'), source_sha256=frozen,
        worker_sha256=worker_digest, quality_used_for_calibration=False,
        current_dynamic_K_range=[48, 144], formal_teacher_policy=rows[0],
        note='White: direct term and whitened gain input removed. CAM: direct term, gain inputs, and utility inputs removed. Frozen pretrained heads; not retrained controls.')
    write_json(target / 'protocol.json', protocol)
    print(json.dumps({'protocol': str(target / 'protocol.json'), 'rows': [(r['name'], r['mean_K']) for r in rows]}), flush=True)


if __name__ == '__main__':
    main()
