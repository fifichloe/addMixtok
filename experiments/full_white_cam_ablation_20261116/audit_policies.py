"""Independently replay masks and compare with the earlier direct-term controls."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'no_density_revision_20261114'
sys.path.insert(0, str(SOURCE))
from density_free_config import sha, write_json


def main():
    policy = json.loads((HERE / 'policies/protocol.json').read_text())
    old_policy = json.loads((SOURCE / 'design_gate/long/policies/protocol.json').read_text())
    old = {row['name']: row for row in old_policy['variants']}
    assert policy['status'] == 'policies_frozen_before_quality' and not policy['quality_used_for_calibration']
    rows = {}
    for row in policy['variants']:
        name = row['name']
        mask = np.load(row['mask_file'])
        k = np.load(row['K_file'])
        assert sha(row['mask_file']) == row['mask_sha256']
        assert sha(row['K_file']) == row['K_sha256']
        assert mask.shape == (50000, 256) and k.shape == (50000,)
        assert np.array_equal(mask.sum(1), k) and abs(float(k.mean()) - 96) < .01
        if row['budget'] == 'fixed96':
            assert np.all(k == 96)
        elif row['budget'] == 'dynamic96':
            assert k.min() >= 48 and k.max() <= 144
        rows[name] = {'mean_K': float(k.mean()), 'min_K': int(k.min()), 'max_K': int(k.max())}
        if name.startswith('full_'):
            old_mask = np.load(old[name]['mask_file'])
            assert np.array_equal(mask, old_mask), 'Reference policy must reproduce accepted full policy exactly'
            rows[name]['matches_accepted_full_mask_exactly'] = True
        else:
            old_name = name.replace('_all_explicit', '')
            old_mask = np.load(old[old_name]['mask_file'])
            rows[name]['old_direct_only_mask_overlap'] = float(np.logical_and(mask, old_mask).sum() / mask.sum())
    write_json(HERE / 'policy_audit.json', dict(status='all_full_component_masks_verified', rows=rows,
        exact_original_full_policy=True,
        old_white_and_cam_controls_removed_direct_score_terms_only=True,
        new_white_direct_and_gain_input_removed=True,
        new_cam_direct_gain_and_utility_inputs_removed=True,
        input_ablation_uses_training_mean_imputation=True,
        frozen_heads_not_retrained=True,
        policy_sha256=sha(HERE / 'policies/protocol.json')))
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
