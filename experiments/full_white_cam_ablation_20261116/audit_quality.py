"""Recompute the reported 50k rFID by independent centered-moment sqrtm."""
import importlib.util
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / 'new_router_ablation_suite_20261114/analysis/audit_reconstruction_results.py'
spec = importlib.util.spec_from_file_location('independent_fid_helper', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', choices=('dynamic', 'fixed'), default='dynamic')
    args = parser.parse_args()
    result = json.loads((HERE / f'results_{args.budget}.json').read_text())
    assert result['status'] == 'full_50000_quality_complete' and result['kind'] == 'long'
    assert len(result['reports']) == 3 and len(result['rows']) == 3
    folders = [HERE / f'quality_{args.budget}' / f'shard_{i}' for i in range(3)]
    real = helper.centered_moments([folder / 'real.npy' for folder in folders])
    rows = {}
    for row in result['rows']:
        name = row['name']
        computed = helper.independent_fid(real, [folder / (name + '.features.npy') for folder in folders])
        error = abs(computed['rFID_sqrtm'] - row['rFID'])
        assert error < 1e-7, (name, error)
        rows[name] = dict(reported_rFID=row['rFID'], independent=computed, absolute_error=error)
        print(name, error, flush=True)
    out = HERE / f'quality_{args.budget}_independent_audit.json'
    out.write_text(json.dumps(dict(status='all_three_50k_rFID_independently_verified', rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
