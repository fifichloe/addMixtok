"""Run the approved spatial and U stages once all three distillations finish."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
KINDS = ('without_gain', 'without_white_all_explicit', 'without_cam_all_explicit')
STATUS = HERE / 'quality_orchestrator_status.json'


def write(**values):
    values['updated_at_unix'] = time.time()
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(values, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(values), flush=True)


def free_cards():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    cards = []
    for line in result.stdout.splitlines():
        index, free = (int(value.strip()) for value in line.split(','))
        if free >= 27 * 1024:
            cards.append((free, index))
    return [index for _, index in sorted(cards, reverse=True)]


def wait_file(path, kind, stage):
    while not path.exists():
        write(status='waiting', kind=kind, stage=stage, expected=str(path))
        time.sleep(60)


def gpu_group(kind, stage):
    while True:
        cards = free_cards()
        if len(cards) >= 2:
            selected = cards[:4] if len(cards) >= 4 else cards[:2]
            write(status='GPU_admitted', kind=kind, stage=stage, GPUs=selected,
                  minimum_free_MiB=27 * 1024)
            return ','.join(map(str, selected))
        write(status='waiting_for_two_GPUs', kind=kind, stage=stage,
              available_cards=cards, minimum_free_MiB=27 * 1024)
        time.sleep(30)


def run_stage(kind, stage):
    preparation = HERE / f'prepare_{stage}.py'
    result = subprocess.run([sys.executable, str(preparation), '--kind', kind],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{kind} {stage} preparation failed: {result.stderr}')
    directory = HERE / f'{stage}_{kind}'
    GPUs = gpu_group(kind, stage)
    with (directory / 'launch.log').open('x') as log:
        child = subprocess.Popen(['bash', str(HERE / 'run_quality_stage.sh'),
                                  stage, kind, GPUs], cwd=HERE.parent.parent,
                                 stdout=log, stderr=subprocess.STDOUT)
        write(status='running', kind=kind, stage=stage, GPUs=GPUs,
              pid=child.pid, log=str(directory / 'launch.log'))
        code = child.wait()
    if code:
        raise RuntimeError(f'{kind} {stage} failed with exit code {code}; see launch.log')
    report = json.loads((directory / 'report.json').read_text())
    assert report['status'] == 'completed'
    if stage == 'spatial':
        assert report['new_finetuning_updates'] == 2000
    else:
        assert report['new_U_updates'] == 2000
    write(status='stage_complete', kind=kind, stage=stage, GPUs=GPUs,
          report=str(directory / 'report.json'))


def main():
    with (HERE / 'quality_orchestrator.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for kind in KINDS:
                wait_file(HERE / f'train_{kind}/report.json', kind, 'distill')
                report = json.loads((HERE / f'train_{kind}/report.json').read_text())
                assert report['status'] == 'complete' and report['steps'] == 20000
                run_stage(kind, 'spatial')
                run_stage(kind, 'budget')
            write(status='complete', all_kinds=list(KINDS),
                  MaskGIT_retrained=False, generation_gFID_still_pending=True)
        except BaseException as error:
            write(status='failed_no_retry', error=repr(error),
                  traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
