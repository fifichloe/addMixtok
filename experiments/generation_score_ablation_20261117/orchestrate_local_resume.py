"""Resume the three score ablations from exact checkpoints on a local cache.

The transfer and all teacher-label ranks must finish before the switch. The
current trainers remain active until then, so this does not idle their GPUs.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
LOCAL = HERE / 'base_local'
SOURCE = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/no_density_revision_20261114/final/full/cache')
PYTHON = '/home/heyefei/miniconda3/envs/MoT/bin/python'
JOBS = {
    'without_gain': (3622851, '6'),
    'without_cam_all_explicit': (3622859, '0'),
    'without_white_all_explicit': (3622862, '1'),
}


def status(**kwargs):
    path = HERE / 'local_resume_status.json'
    path.write_text(json.dumps(dict(time=time.time(), **kwargs), indent=2) + '\n')


def alive(pid):
    try:
        raw = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return 'train_router.py' in raw


def checkpoint_step(kind):
    path = HERE / f'train_{kind}' / 'latest.pt'
    return int(torch.load(path, map_location='cpu', weights_only=False)['step'])


def last_logged_step(kind):
    path = HERE / f'train_{kind}' / 'training.jsonl'
    return json.loads(path.read_text().splitlines()[-1])['step']


def cache_ready():
    if not (LOCAL / 'copy.complete').exists():
        return False
    if not all((HERE / 'cache' / f'rank{rank}' / 'report.json').exists()
               for rank in range(4)):
        return False
    for split, count in [('held', 5), ('fit', 621)]:
        original = sorted((SOURCE / split).glob('*.pt'))
        mirrored = sorted((LOCAL / split).glob('*.pt'))
        if len(original) != count or len(mirrored) != count:
            raise RuntimeError(f'cache file count mismatch: {split}')
        for a, b in zip(original, mirrored):
            if a.name != b.name or a.stat().st_size != b.stat().st_size:
                raise RuntimeError(f'cache copy mismatch: {a} / {b}')
    return True


def launch(kind, gpu):
    output = HERE / f'train_{kind}'
    log = (output / 'local_resume.log').open('w')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu
    env['MIXTOK_GEN_ABL_BASE_CACHE'] = str(LOCAL)
    command = [PYTHON, str(HERE / 'train_router.py'), '--kind', kind,
               '--steps', '20000', '--batch', '512', '--eval-every', '1000',
               '--output', str(output), '--resume']
    process = subprocess.Popen(command, cwd=HERE.parent.parent, env=env,
                               stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    log.close()
    return process.pid


def main():
    while not cache_ready():
        status(phase='waiting_for_local_and_teacher_cache')
        time.sleep(30)
    status(phase='cache_ready', file_count=626)
    for kind, (pid, gpu) in JOBS.items():
        output = HERE / f'train_{kind}'
        if (output / 'report.json').exists():
            continue
        # Preserve the latest optimizer state and at most 20 uncheckpointed
        # training-log steps. The old process has an isolated process group.
        while alive(pid) and last_logged_step(kind) - checkpoint_step(kind) > 20:
            time.sleep(5)
        if alive(pid):
            os.killpg(pid, signal.SIGTERM)
            for _ in range(60):
                if not alive(pid):
                    break
                time.sleep(1)
            if alive(pid):
                raise RuntimeError(f'old trainer did not stop: {kind} pid={pid}')
        if not (output / 'report.json').exists():
            new_pid = launch(kind, gpu)
            status(phase='resumed', kind=kind, old_pid=pid, new_pid=new_pid,
                   gpu=gpu, resume_step=checkpoint_step(kind))
    status(phase='all_switched_to_local')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status(phase='error', error=repr(exc))
        raise
