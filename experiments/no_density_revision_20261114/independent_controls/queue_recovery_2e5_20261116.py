"""Continue frozen independent-control evaluations after the seed-4 GPU OOM.

The failed random attempt and partial shards are archived. Completed shards
0/1 remain in place; evaluate_recovery_shared_2e5_20261116.py verifies their
hashes and reruns only shards 2/3. Later policies use the same frozen recipe.
"""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from queue_current_evaluations_2e5_20261116 import (
    CONTROLS, HERE, POLICIES, POLICY_DIR, baseline_root, digest, read, write,
)

STATUS = HERE / 'queue_recovery_2e5_20261116.json'
EVALUATOR = HERE / 'evaluate_recovery_shared_2e5_20261116.py'
ORIGINAL = HERE / 'queue_current_evaluations_2e5_20261116.json'
GPU_UUIDS = (
    'GPU-2fd4aa9c-58f6-0bea-efbf-d4a65cc3e8cd,'
    'GPU-721b39e7-aa5e-e60d-ac55-3c16d203eb43'
)


def main():
    assert not STATUS.exists(), 'Preserve previous recovery; no silent retry'
    original = read(ORIGINAL)
    assert original['status'] == 'failed_preserved_no_retry'
    assert original['policy'] == 'random' and original['seed'] == 4
    assert 'random seed 4 evaluation failed' in original['error']
    assert POLICY_DIR.is_dir()
    evaluator_sha256 = digest(EVALUATOR)

    baselines = {}
    for seed in (1, 2, 3, 4):
        root = baseline_root(seed)
        summary = read(root / 'summary.json')
        assert summary['status'] == 'complete' and summary['n'] == 50000
        assert summary['seed'] == seed and summary['generation_rounds'] == '8+8'
        assert summary['cfg_stage1'] == 4.5 and summary['cfg_stage2'] == 3.0
        baselines[seed] = summary

    for policy in POLICIES:
        root = CONTROLS / policy
        complete = read(root / 'training_complete.json')
        checkpoint = read(root / 'latest.json')
        config = read(root / 'config.json')
        assert complete['status'] == 'complete' and complete['step'] == 20000
        assert checkpoint['step'] == 20000 and checkpoint['round_trip_exact']
        assert complete['checkpoint_sha256'] == checkpoint['sha256']
        assert complete['model_optimizer_restore_passed']
        assert complete['all_frozen_components_unchanged']
        assert config['policy'] == policy and config['updates'] == 20000
        assert config['global_batch'] == 256 and config['eval_cfg_stage2'] == 3.0
        for seed in (1, 2, 3, 4):
            document = read(POLICY_DIR / f'{policy}_seed{seed}.json')
            assert document['status'] == 'frozen_before_independent_control_quality'
            assert document['checkpoint_sha256'] == checkpoint['sha256']
            assert document['training_control_identity'] == config['control_identity']
            assert document['kind'] == 'long' and document['cfg'] == 3.0
            assert document['seed'] == seed and document['stage2_steps'] == 8
            assert document['selection_policy']['policy'] == policy
            assert document['baseline_summary_sha256'] == digest(baseline_root(seed) / 'summary.json')

    base_fids = [baselines[s]['metrics']['full_refine']['gFID'] for s in (1, 2, 3, 4)]
    mean = sum(base_fids) / 4
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in base_fids) / 3)
    threshold = max(0.05, 3 * sample_std)
    state = dict(status='running', stage='waiting_for_seed4_shard_recovery',
                 started_unix=time.time(), original_failure_status_sha256=digest(ORIGINAL),
                 evaluator_sha256=evaluator_sha256, repeat_threshold=threshold,
                 repeat_rule='seed4 gain below threshold => seeds 1-3',
                 fixed_GPU_UUIDs=GPU_UUIDS, results={}, no_automatic_retry=True)

    def update(**fields):
        state.update(fields, checked_unix=time.time())
        write(STATUS, state)

    try:
        update()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                   USE_TF='0', MIXTOK_RECOVERY_GPU_UUIDS=GPU_UUIDS)
        for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE'):
            env.pop(key, None)

        for policy in POLICIES:
            rows = {}
            for seed in (4, 1, 2, 3):
                if seed != 4 and rows[4]['control_minus_main_gFID'] >= threshold:
                    break
                output = CONTROLS / 'evaluations' / f'{policy}_seed{seed}'
                summary_path = output / 'summary.json'
                if policy == 'random' and seed == 4:
                    assert output.is_dir()
                    while not summary_path.exists():
                        progress_path = output / 'status.json'
                        progress = read(progress_path) if progress_path.exists() else {}
                        if progress.get('status') == 'failed_preserved_no_retry':
                            raise RuntimeError('Manual random seed-4 recovery failed')
                        update(stage='waiting_for_seed4_shard_recovery', policy=policy,
                               seed=seed, generated=progress.get('completed', 0),
                               evaluation_status=progress.get('status'))
                        time.sleep(30)
                else:
                    assert not output.exists(), output
                    assert digest(EVALUATOR) == evaluator_sha256
                    command = [sys.executable, str(EVALUATOR), '--kind', 'long',
                               '--cfg', '3.0', '--stage2-steps', '8', '--seed', str(seed),
                               '--output', str(output), '--policy-file',
                               str(POLICY_DIR / f'{policy}_seed{seed}.json')]
                    log_path = HERE / f'eval_{policy}_seed{seed}_recovery_20261116.log'
                    with log_path.open('x') as log:
                        child = subprocess.Popen(command, cwd=HERE, env=env,
                                                 stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True)
                        while child.poll() is None:
                            progress_path = output / 'status.json'
                            progress = read(progress_path) if progress_path.exists() else {}
                            update(stage='evaluating', policy=policy, seed=seed,
                                   child_pid=child.pid,
                                   evaluation_status=progress.get('status'),
                                   generated=progress.get('completed', 0))
                            time.sleep(30)
                    if child.returncode != 0:
                        raise RuntimeError(f'{policy} seed {seed} evaluation failed: {child.returncode}')

                summary = read(summary_path)
                assert summary['status'] == 'complete' and summary['n'] == 50000
                assert summary['seed'] == seed and summary['independently_trained_control']
                assert summary['all_masks_and_K_replayed']
                assert summary['same_generated_inputs_and_scorer_within_2e5_baseline']
                assert summary['base_feature_max_abs_delta'] <= 2e-5
                assert summary['base_feature_max_abs_delta_tolerance'] == 2e-5
                assert abs(summary['tokens']['mean_generated_2d'] -
                           baselines[seed]['tokens']['mean_generated_2d']) < 1e-9
                control = summary['metrics']['full_refine']['gFID']
                baseline = baselines[seed]['metrics']['full_refine']['gFID']
                rows[seed] = dict(control_gFID=control, main_gFID=baseline,
                                  control_minus_main_gFID=control - baseline,
                                  mean_local_K=summary['tokens']['mean_generated_2d'],
                                  summary=str(summary_path),
                                  summary_sha256=digest(summary_path))
                state['results'][policy] = rows
                update(stage='evaluated', policy=policy, seed=seed,
                       results=state['results'])

        update(status='complete', stage='all_required_independent_control_gFID_complete',
               completed_unix=time.time(), results=state['results'])
    except BaseException as error:
        update(status='failed_preserved_no_retry', error=repr(error),
               traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
