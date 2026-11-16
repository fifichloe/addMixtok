"""Archive only completed generation ablation receipts and compact results."""
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
DEST = HERE.parent.parent / 'addMixtok_metrics_work/experiment_provenance/actual_runs/generation_score_ablation_20261117'
BASELINE = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/'
                'no_density_revision_20261114/generation_evaluation/long_cfg3_seed4_8p8/summary.json')
KINDS = ('without_gain', 'without_white_all_explicit', 'without_cam_all_explicit')


def load(path):
    return json.loads(path.read_text())


def copy(source, name):
    assert source.is_file(), source
    shutil.copyfile(source, DEST / name)


def main():
    baseline = load(BASELINE)
    assert baseline['status'] == 'complete' and baseline['n'] == 50000
    assert baseline['seed'] == 4 and baseline['generation_rounds'] == '8+8'
    assert baseline['cfg_stage2'] == 3.
    target_k = baseline['tokens']['mean_generated_2d']
    records = []
    for kind in ('main', 'U_zero', *KINDS):
        if kind == 'main':
            result = baseline
            policy = None
        elif kind == 'U_zero':
            result = load(HERE / 'u_zero_seed4_50k/summary.json')
            policy = load(HERE / 'u_zero_policy_seed4.json')
        else:
            result = load(HERE / f'generation_{kind}_seed4_50k/summary.json')
            policy = load(HERE / f'policy_{kind}_seed4.json')
            assert result['ablation'] == kind and result['matched_baseline_mean_K']
            for stage, field, expected in [('train', 'steps', 20000),
                                           ('spatial', 'new_finetuning_updates', 2000),
                                           ('budget', 'new_U_updates', 2000)]:
                report = load(HERE / f'{stage}_{kind}/report.json')
                assert report[field] == expected and report['status'] in ('complete', 'completed')
        assert result['status'] == 'complete' and result['n'] == 50000
        assert result['seed'] == 4 and result['generation_rounds'] == '8+8'
        assert result['cfg_stage2'] == 3.
        assert result['checkpoint_sha256'] == baseline['checkpoint_sha256']
        observed_k = result['tokens']['mean_generated_2d']
        assert abs(observed_k - target_k) <= .01
        if policy is not None:
            assert policy['quality_used_for_budget_or_threshold'] is False
            assert abs(policy['target_mean_K'] - target_k) < 1e-9
        records.append(dict(kind=kind, n=result['n'], seed=result['seed'],
                            generation_rounds=result['generation_rounds'],
                            cfg_stage2=result['cfg_stage2'],
                            gFID=result['metrics']['full_refine']['gFID'],
                            mean_generated_2D_tokens=observed_k,
                            delta_gFID_vs_main=result['metrics']['full_refine']['gFID'] -
                                baseline['metrics']['full_refine']['gFID'],
                            generator_checkpoint_sha256=result['checkpoint_sha256']))
    DEST.mkdir(parents=True, exist_ok=True)
    copy(BASELINE, 'main_summary.json')
    copy(HERE / 'u_zero_seed4_50k/summary.json', 'U_zero_summary.json')
    copy(HERE / 'u_zero_policy_seed4.json', 'U_zero_policy.json')
    for kind in KINDS:
        copy(HERE / f'generation_{kind}_seed4_50k/summary.json', f'{kind}_summary.json')
        copy(HERE / f'policy_{kind}_seed4.json', f'{kind}_policy.json')
        for stage in ('train', 'spatial', 'budget'):
            copy(HERE / f'{stage}_{kind}/report.json', f'{kind}_{stage}_report.json')
            copy(HERE / f'{stage}_{kind}/config.json', f'{kind}_{stage}_config.json')
        resume_history = HERE / f'train_{kind}/resume_history.jsonl'
        if resume_history.exists():
            copy(resume_history, f'{kind}_resume_history.jsonl')
    (DEST / 'results.json').write_text(json.dumps(dict(status='complete',
        protocol='same frozen generator; seed 4; 50000 images; 8+8; stage2 CFG 3.0; '
                 'same realized average 2D-token budget', rows=records),
        indent=2, allow_nan=False) + '\n')
    lines = ['# Generation score and U ablations', '',
             'The MaskGIT generator is frozen in all rows. Each score ablation '
             're-trains the base-only generation router, the spatial stage, and the U stage. '
             'The U=0 row uses the main router without its U shift.', '',
             'Thresholds were calibrated from the same 50,000 generated 1D token sequences '
             'to match the main run\'s average K. Calibration used scores only, before gFID.', '',
             '| Variant | gFID | Mean generated 2D tokens | Delta gFID |',
             '| --- | ---: | ---: | ---: |']
    for row in records:
        lines.append(f"| {row['kind']} | {row['gFID']:.4f} | "
                     f"{row['mean_generated_2D_tokens']:.4f} | "
                     f"{row['delta_gFID_vs_main']:+.4f} |")
    lines.extend(['', 'Run configs, stage reports, policy receipts, and 50k summary JSON '
                  'are archived in this directory. ImageNet images, large feature dumps, '
                  'and checkpoints are excluded.', ''])
    (DEST / 'README.md').write_text('\n'.join(lines))
    print(DEST / 'README.md')


if __name__ == '__main__':
    main()
