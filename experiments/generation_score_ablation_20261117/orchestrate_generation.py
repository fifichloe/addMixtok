"""Evaluate the three frozen-generator ablations as soon as U stages finish."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
KINDS = ('without_gain', 'without_white_all_explicit', 'without_cam_all_explicit')
STATUS = HERE / 'generation_orchestrator_status.json'
QUALITY_STATUS = HERE / 'quality_orchestrator_status.json'


def write(**values):
    values['updated_at_unix'] = time.time()
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(values, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(values), flush=True)


def wait_file(path, kind):
    while not path.exists():
        write(status='waiting_for_U_training', kind=kind, expected=str(path))
        time.sleep(60)


def free_calibration_gpu():
    reserved = set()
    if QUALITY_STATUS.exists():
        state = json.loads(QUALITY_STATUS.read_text())
        if state.get('scheduler') == 'parallel_gpu_pairs_v1' and state.get('status') in ('running', 'stage_complete'):
            try:
                os.kill(int(state['pid']), 0)
                reserved = {str(index) for index in state.get('reserved_GPUs', [])}
            except (OSError, KeyError, ValueError):
                pass
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    cards = []
    for line in result.stdout.splitlines():
        index, uuid, free = (value.strip() for value in line.split(','))
        if int(free) >= 16 * 1024 and index not in reserved:
            cards.append((int(free), index, uuid))
    return max(cards) if cards else None


def calibrate(kind):
    policy = HERE / f'policy_{kind}_seed4.json'
    if policy.exists():
        saved = json.loads(policy.read_text())
        assert saved['status'] == 'frozen_before_control_quality' and saved['ablation'] == kind
        return policy
    while True:
        card = free_calibration_gpu()
        if card is not None:
            break
        write(status='waiting_for_calibration_GPU', kind=kind, minimum_free_MiB=16 * 1024)
        time.sleep(20)
    _, index, uuid = card
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    logpath = HERE / f'calibration_{kind}.log'
    command = [sys.executable, str(HERE / 'calibrate_generated_budget.py'),
               '--kind', kind, '--output', str(policy)]
    with logpath.open('x') as log:
        child = subprocess.Popen(command, cwd=HERE.parent.parent, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT)
        write(status='calibration_running', kind=kind, GPU=index, pid=child.pid,
              log=str(logpath))
        code = child.wait()
    if code:
        raise RuntimeError(f'{kind} generated-budget calibration failed: {code}; see {logpath}')
    saved = json.loads(policy.read_text())
    assert saved['status'] == 'frozen_before_control_quality' and saved['ablation'] == kind
    assert abs(saved['calibration']['mean_K'] - saved['target_mean_K']) <= .01
    write(status='calibration_complete', kind=kind, GPU=index,
          threshold=saved['selection_policy']['threshold'],
          mean_K=saved['calibration']['mean_K'], policy=str(policy))
    return policy


def main():
    with (HERE / 'generation_orchestrator.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for kind in KINDS:
                report = HERE / f'budget_{kind}/report.json'
                wait_file(report, kind)
                saved = json.loads(report.read_text())
                assert saved['status'] == 'completed' and saved['new_U_updates'] == 2000
                policy = calibrate(kind)
                output = HERE / f'generation_{kind}_seed4_50k'
                command = [sys.executable, str(HERE / 'evaluate.py'),
                    '--kind', 'long', '--ablation', kind,
                    '--cfg', '3.0', '--stage2-steps', '8', '--seed', '4',
                    '--output', str(output), '--policy-file', str(policy)]
                logpath = HERE / f'generation_{kind}_launcher.log'
                with logpath.open('x') as log:
                    child = subprocess.Popen(command, cwd=HERE.parent.parent,
                                             stdout=log, stderr=subprocess.STDOUT)
                    write(status='generation_running', kind=kind, pid=child.pid,
                          output=str(output), log=str(logpath))
                    code = child.wait()
                if code:
                    raise RuntimeError(f'{kind} generation evaluation failed: {code}; see {logpath}')
                summary = json.loads((output / 'summary.json').read_text())
                assert summary['status'] == 'complete' and summary['n'] == 50000
                assert summary['ablation'] == kind and summary['generation_rounds'] == '8+8'
                write(status='generation_complete', kind=kind,
                      gFID=summary['metrics']['full_refine']['gFID'],
                      mean_generated_2d=summary['tokens']['mean_generated_2d'],
                      summary=str(output / 'summary.json'))
            write(status='complete', all_kinds=list(KINDS), MaskGIT_retrained=False)
        except BaseException as error:
            write(status='failed_no_retry', error=repr(error),
                  traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
