"""Run independent score-ablation quality stages on disjoint free GPU pairs."""
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
STATUS = HERE / 'quality_orchestrator_status.json'
MINIMUM_FREE_MIB = 27 * 1024


def write(status, active, **extra):
    value = dict(status=status, scheduler='parallel_gpu_pairs_v1', pid=os.getpid(),
                 active=[dict(kind=kind, stage=stage, GPUs=list(gpus), pid=child.pid)
                         for kind, (stage, gpus, child, _) in active.items()],
                 reserved_GPUs=sorted({gpu for _, gpus, _, _ in active.values()
                                       for gpu in gpus}), updated_at_unix=time.time(), **extra)
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(value), flush=True)


def free_cards(active):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True)
    reserved = {gpu for _, gpus, _, _ in active.values() for gpu in gpus}
    cards = []
    for line in result.stdout.splitlines():
        index, free = (int(item.strip()) for item in line.split(','))
        if free >= MINIMUM_FREE_MIB and index not in reserved:
            cards.append((free, index))
    return [index for _, index in sorted(cards, reverse=True)]


def stage_for(kind):
    budget_report = HERE / f'budget_{kind}/report.json'
    if budget_report.exists():
        report = json.loads(budget_report.read_text())
        assert report['status'] == 'completed' and report['new_U_updates'] == 2000
        return 'complete'
    spatial_report = HERE / f'spatial_{kind}/report.json'
    if spatial_report.exists():
        report = json.loads(spatial_report.read_text())
        assert report['status'] == 'completed' and report['new_finetuning_updates'] == 2000
        return 'budget'
    distill_report = HERE / f'train_{kind}/report.json'
    if distill_report.exists():
        report = json.loads(distill_report.read_text())
        assert report['status'] == 'complete' and report['steps'] == 20000
        return 'spatial'
    return 'waiting_for_distillation'


def launch(kind, stage, gpus, active):
    preparation = subprocess.run([sys.executable, str(HERE / f'prepare_{stage}.py'),
                                  '--kind', kind], capture_output=True, text=True)
    if preparation.returncode:
        raise RuntimeError(f'{kind} {stage} preparation failed: {preparation.stderr}')
    directory = HERE / f'{stage}_{kind}'
    log = (directory / 'launch.log').open('x')
    command = ['bash', str(HERE / 'run_quality_stage.sh'), stage, kind,
               ','.join(map(str, gpus))]
    try:
        child = subprocess.Popen(command, cwd=HERE.parent.parent, stdout=log,
                                 stderr=subprocess.STDOUT)
    except BaseException:
        log.close()
        raise
    active[kind] = (stage, tuple(gpus), child, log)
    write('running', active, just_launched=dict(kind=kind, stage=stage, GPUs=gpus))


def main():
    active = {}
    completed = set()
    with (HERE / 'quality_orchestrator.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while len(completed) != len(KINDS):
                for kind, (stage, gpus, child, log) in list(active.items()):
                    code = child.poll()
                    if code is None:
                        continue
                    log.close()
                    del active[kind]
                    if code:
                        raise RuntimeError(f'{kind} {stage} failed with exit code {code}; '
                                           f'see {stage}_{kind}/launch.log')
                    assert stage_for(kind) == ('budget' if stage == 'spatial' else 'complete')
                    write('stage_complete', active, kind=kind, stage=stage,
                          GPUs=list(gpus), report=str(HERE / f'{stage}_{kind}/report.json'))
                stages = {kind: stage_for(kind) for kind in KINDS}
                completed = {kind for kind, stage in stages.items() if stage == 'complete'}
                cards = free_cards(active)
                for kind in KINDS:
                    stage = stages[kind]
                    if kind in active or stage not in ('spatial', 'budget'):
                        continue
                    if len(cards) < 2:
                        break
                    pair, cards = cards[:2], cards[2:]
                    launch(kind, stage, pair, active)
                write('running' if active else 'waiting', active,
                      completed_kinds=sorted(completed), stages=stages,
                      free_admissible_GPUs=cards, minimum_free_MiB=MINIMUM_FREE_MIB)
                if len(completed) != len(KINDS):
                    time.sleep(20)
            write('complete', active, completed_kinds=sorted(completed),
                  MaskGIT_retrained=False)
        except BaseException as error:
            write('failed_no_retry', active, error=repr(error),
                  traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
