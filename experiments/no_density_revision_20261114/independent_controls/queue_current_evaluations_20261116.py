"""Evaluate the three actually running independent Long controls after 20k."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
REV = HERE.parent
PAPER = REV.parents[1]
RUN = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/no_density_revision_20261114')
CONTROLS = RUN / 'independent_controls_20261116'
LINKS = RUN / 'generation_controls/independent'
LOCAL_BASE = REV / 'recovered_generation_20261116'
EVALUATOR = HERE / 'evaluate_current_shared_20261116.py'
POLICIES = ('random', 'uniform', 'nearconstant')
STATUS = HERE / 'queue_current_evaluations_20261116.json'
POLICY_DIR = HERE / 'current_evaluation_policies_20261116'
THRESHOLD = 0.05


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def running_training(policy):
    progress = CONTROLS / policy / 'rank0.json'
    if not progress.exists():
        return False
    pid = read(progress).get('pid')
    if not pid:
        return False
    proc = Path(f'/proc/{pid}/cmdline')
    try:
        command = proc.read_bytes().replace(b'\0', b' ').decode()
        state = Path(f'/proc/{pid}/stat').read_text().split()[2]
    except FileNotFoundError:
        return False
    return state not in ('Z', 'X') and 'independent_controls/train.py' in command and f'--policy {policy}' in command


def baseline_root(seed):
    if seed == 4:
        return RUN / 'generation_evaluation/long_cfg3_seed4_8p8'
    return LOCAL_BASE / f'long_cfg3_seed{seed}_8p8_abs'


def main():
    assert not STATUS.exists(), 'Preserve previous execution; no silent retry'
    started = time.time()
    state = dict(status='running', stage='waiting_for_three_20000_step_trainings',
                 started_unix=started, results={}, no_automatic_retry=True)

    def update(**fields):
        state.update(fields, checked_unix=time.time())
        write(STATUS, state)

    try:
        update()
        while True:
            progress = {}
            for policy in POLICIES:
                root = CONTROLS / policy
                done = root / 'training_complete.json'
                row = read(root / 'rank0.json') if (root / 'rank0.json').exists() else None
                progress[policy] = dict(step=row.get('step') if row else None,
                                        complete=done.exists(), live=running_training(policy))
                if not done.exists() and row is not None and not progress[policy]['live']:
                    raise RuntimeError(f'{policy} training ended without training_complete.json')
            update(training_progress=progress)
            if all(item['complete'] for item in progress.values()):
                break
            time.sleep(30)

        assert not (CONTROLS / 'evaluations').exists(), 'No old control quality reuse'
        POLICY_DIR.mkdir(exist_ok=False)
        baselines = {}
        for seed in (1, 2, 3, 4):
            root = baseline_root(seed)
            summary = read(root / 'summary.json')
            assert summary['status'] == 'complete' and summary['n'] == 50000
            assert summary['seed'] == seed and summary['routing_policy']['policy'] == 'dynamic'
            assert summary['cfg_stage1'] == 4.5 and summary['cfg_stage2'] == 3.0
            assert summary['generation_rounds'] == '8+8'
            baselines[seed] = dict(root=root, summary=summary,
                                   summary_sha256=digest(root / 'summary.json'))

        for policy in POLICIES:
            root = CONTROLS / policy
            report = read(root / 'training_complete.json')
            receipt = read(root / 'latest.json')
            config = read(root / 'config.json')
            assert report['status'] == 'complete' and report['step'] == receipt['step'] == 20000
            assert report['model_optimizer_restore_passed'] and report['all_frozen_components_unchanged']
            assert report['checkpoint_sha256'] == receipt['sha256']
            assert report['kind'] == config['kind'] == 'long'
            assert report['policy'] == config['policy'] == policy
            assert config['updates'] == 20000 and config['global_batch'] == 256
            assert config['eval_cfg_stage2'] == 3.0 and config['seed'] == 0
            assert LINKS.joinpath(f'long_{policy}_seed0').resolve() == root.resolve()
            assert math.isclose(config['control_identity']['target_mean_K'],
                                baselines[4]['summary']['tokens']['mean_generated_2d'], abs_tol=1e-9)
            for seed, base in baselines.items():
                assert config['router_identity'] == base['summary']['router']
                selection = dict(policy=policy, minimum_K=48, maximum_K=144,
                                 target_mean_K=base['summary']['tokens']['mean_generated_2d'],
                                 total=50000, route_seed=config['control_identity']['route_seed'])
                document = dict(status='frozen_before_independent_control_quality',
                    kind='long', cfg=3.0, stage2_steps=8, seed=seed,
                    checkpoint_sha256=receipt['sha256'],
                    router_identity=base['summary']['router'],
                    selection_policy=selection, baseline_root=str(base['root']),
                    baseline_summary_sha256=base['summary_sha256'],
                    training_control_identity=config['control_identity'],
                    training_complete_sha256=digest(root / 'training_complete.json'),
                    quality_used_for_budget_or_positions=False,
                    matched_per_generated_image_K=policy != 'nearconstant',
                    nearconstant_offline_balanced_diagnostic=policy == 'nearconstant',
                    independently_trained_control=True, original_global256_20000_recipe=True)
                write(POLICY_DIR / f'{policy}_seed{seed}.json', document)
        update(stage='all_policies_frozen_before_control_quality',
               policy_dir=str(POLICY_DIR), evaluation_source_sha256=digest(EVALUATOR))
        (CONTROLS / 'evaluations').mkdir(exist_ok=False)

        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', USE_TF='0')
        for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE'):
            env.pop(key, None)
        base_fids = [baselines[seed]['summary']['metrics']['full_refine']['gFID']
                     for seed in (1, 2, 3, 4)]
        mean = sum(base_fids) / 4
        sample_std = math.sqrt(sum((value - mean) ** 2 for value in base_fids) / 3)
        threshold = max(THRESHOLD, 3 * sample_std)
        update(repeat_threshold=threshold, repeat_rule='seed4 gain below threshold => seeds 1-3')

        for policy in POLICIES:
            rows = {}
            for seed in (4, 1, 2, 3):
                if seed != 4 and rows[4]['control_minus_main_gFID'] >= threshold:
                    break
                output = CONTROLS / 'evaluations' / f'{policy}_seed{seed}'
                command = [sys.executable, str(EVALUATOR), '--kind', 'long', '--cfg', '3.0',
                           '--stage2-steps', '8', '--seed', str(seed), '--output', str(output),
                           '--policy-file', str(POLICY_DIR / f'{policy}_seed{seed}.json')]
                log_path = HERE / f'eval_{policy}_seed{seed}_20261116.log'
                with log_path.open('x') as log:
                    child = subprocess.Popen(command, cwd=HERE, env=env, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    while child.poll() is None:
                        progress_path = output / 'status.json'
                        progress = read(progress_path) if progress_path.exists() else {}
                        update(stage='evaluating', policy=policy, seed=seed, child_pid=child.pid,
                               evaluation_status=progress.get('status'),
                               generated=progress.get('completed', 0))
                        time.sleep(30)
                if child.returncode != 0:
                    raise RuntimeError(f'{policy} seed {seed} evaluation failed: {child.returncode}')
                summary_path = output / 'summary.json'
                summary = read(summary_path)
                assert summary['status'] == 'complete' and summary['n'] == 50000
                assert summary['seed'] == seed and summary['independently_trained_control']
                assert summary['all_masks_and_K_replayed']
                assert summary['same_generated_inputs_and_scorer_within_1e5_baseline']
                assert summary['base_feature_max_abs_delta'] <= 1e-5
                assert abs(summary['tokens']['mean_generated_2d'] -
                           baselines[seed]['summary']['tokens']['mean_generated_2d']) < 1e-9
                control = summary['metrics']['full_refine']['gFID']
                baseline = baselines[seed]['summary']['metrics']['full_refine']['gFID']
                rows[seed] = dict(control_gFID=control, main_gFID=baseline,
                                  control_minus_main_gFID=control-baseline,
                                  mean_local_K=summary['tokens']['mean_generated_2d'],
                                  summary=str(summary_path), summary_sha256=digest(summary_path))
                state['results'][policy] = rows
                update(stage='evaluated', policy=policy, seed=seed, results=state['results'])
        update(status='complete', stage='all_required_independent_control_gFID_complete',
               completed_unix=time.time(), results=state['results'])
    except BaseException as error:
        update(status='failed_preserved_no_retry', error=repr(error),
               traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
