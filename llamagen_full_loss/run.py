#!/usr/bin/env python3
"""Probe peak GPU memory with the full loss, then train the LlamaGen control."""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MEMORY = re.compile(r'MEMORY world=(\d+) batch=(\d+) peak_used_mib=([\d.]+) '
                    r'total_mib=([\d.]+) fraction=([\d.]+)')


def gpu_inventory() -> list[tuple[int, int, int]]:
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.total,memory.used', '--format=csv,noheader,nounits'],
        text=True,
    )
    return [tuple(int(part.strip()) for part in line.split(','))
            for line in output.splitlines() if line.strip()]


def run_logged(command: list[str], logfile: Path, env: dict[str, str],
               timeout: int | None, append: bool = False) -> int:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a' if append else 'w') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def forward_sigterm(signum, _frame):
            if process.poll() is None:
                os.killpg(process.pid, signum)
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, forward_sigterm)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f'Command timed out after {timeout} s: {logfile}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--titok-root', type=Path, required=True)
    parser.add_argument('--llamagen-root', type=Path, required=True)
    parser.add_argument('--llamagen-ckpt', type=Path, required=True)
    parser.add_argument('--dino-repo', type=Path, required=True)
    parser.add_argument('--clip-cache-dir', type=Path, required=True)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--gpu-ids', help='Comma-separated physical GPU indices; otherwise use first idle GPUs.')
    parser.add_argument('--target-memory', type=float, default=0.92)
    parser.add_argument('--max-batch', type=int, default=128)
    parser.add_argument('--probe-steps', type=int, default=12)
    parser.add_argument('--probe-timeout', type=int, default=900)
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit-samples', type=int, default=0, help='Smoke runs only.')
    parser.add_argument('--train-max-steps', type=int, default=0, help='Smoke runs only; skips checkpoint.')
    parser.add_argument('--wandb-project', default='mixtok-rebuttal')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    args = parser.parse_args()
    if args.gpus < 1 or args.max_batch < 1 or args.probe_steps < 2 or args.accum < 1:
        parser.error('GPU count, max batch, probe steps and accumulation must be positive')
    if not 0.0 < args.target_memory < 1.0:
        parser.error('--target-memory must be between zero and one')
    paths = (args.data_root, args.source_root, args.titok_root,
             args.llamagen_root, args.llamagen_ckpt, args.dino_repo)
    for path in paths:
        if not path.exists():
            parser.error(f'required path does not exist: {path}')
    inventory = gpu_inventory()
    available = {index: (total, used) for index, total, used in inventory}
    if args.gpu_ids:
        selected = [int(part) for part in args.gpu_ids.split(',')]
        if len(selected) != args.gpus or len(set(selected)) != len(selected):
            parser.error('--gpu-ids must list exactly --gpus distinct indices')
    else:
        selected = [index for index, total, used in inventory if used <= 1024][:args.gpus]
    if len(selected) != args.gpus:
        parser.error(f'Only {len(selected)} idle GPUs available; requested {args.gpus}')
    for index in selected:
        if index not in available or available[index][1] > 1024:
            parser.error(f'GPU {index} is missing or has more than 1 GiB in use')
    totals = [available[index][0] for index in selected]
    if max(totals) - min(totals) > 512:
        parser.error('Selected GPUs have different memory capacities; choose matching GPUs')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(index) for index in selected)
    env.setdefault('OMP_NUM_THREADS', '2')
    base = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={args.gpus}', str(ROOT / 'train.py'),
            '--data-root', str(args.data_root.resolve()),
            '--source-root', str(args.source_root.resolve()),
            '--titok-root', str(args.titok_root.resolve()),
            '--llamagen-root', str(args.llamagen_root.resolve()),
            '--llamagen-ckpt', str(args.llamagen_ckpt.resolve()),
            '--dino-repo', str(args.dino_repo.resolve()),
            '--clip-cache-dir', str(args.clip_cache_dir.resolve()),
            '--dataset-manifest', str(args.output / 'dataset_manifest.pt'),
            '--workers', str(args.workers)]
    selection_path = args.output / 'batch_selection.json'
    latest = args.output / 'latest.pt'
    if latest.exists():
        if not selection_path.exists():
            raise RuntimeError('latest.pt exists but batch_selection.json is missing; cannot safely resume')
        selection = json.loads(selection_path.read_text())
        if selection['gpu_count'] != args.gpus or selection['accum'] != args.accum:
            raise RuntimeError('Resume requires the original GPU count and accumulation')
        batch = int(selection['batch_size'])
        print(f'Resuming with prior batch={batch}; no new memory probe', flush=True)
    else:
        records: list[dict] = []

        def probe(batch: int) -> bool:
            name = f'batch_{batch:03d}'
            log = args.output / 'probes' / f'{name}.log'
            probe_output = args.output / 'probes' / name
            command = base + ['--output', str(probe_output), '--batch-size', str(batch),
                              '--accum', str(args.accum), '--epochs', str(args.epochs),
                              '--max-steps', str(args.probe_steps), '--force-gan',
                              '--wandb-mode', 'disabled']
            print(f'Probing batch={batch} on GPUs {selected} with full GAN loss', flush=True)
            exit_code = run_logged(command, log, env, args.probe_timeout)
            output = log.read_text(errors='replace')
            match = MEMORY.search(output)
            if exit_code != 0:
                if re.search(r'out of memory|cuda error: out of memory', output, re.IGNORECASE):
                    records.append({'batch_size': batch, 'status': 'oom', 'log': str(log)})
                    print(f'  batch={batch}: out of memory', flush=True)
                    return False
                raise RuntimeError(f'Probe batch={batch} failed (exit {exit_code}): {log}')
            if not match:
                raise RuntimeError(f'Probe batch={batch} produced no MEMORY marker: {log}')
            fraction = float(match.group(5))
            if int(match.group(1)) != args.gpus or int(match.group(2)) != batch:
                raise RuntimeError(f'Probe batch={batch} returned inconsistent memory marker: {log}')
            record = {'batch_size': batch, 'status': 'fit' if fraction <= args.target_memory else 'over_target',
                      'peak_used_mib': float(match.group(3)), 'total_mib': float(match.group(4)),
                      'fraction': fraction, 'log': str(log)}
            records.append(record)
            print(f'  batch={batch}: {fraction:.2%} GPU memory', flush=True)
            return fraction <= args.target_memory

        lower = 0
        upper = None
        candidate = min(8, args.max_batch)
        while True:
            if probe(candidate):
                lower = candidate
                if candidate == args.max_batch:
                    break
                candidate = min(candidate * 2, args.max_batch)
            else:
                upper = candidate
                break
        if upper is not None:
            while upper - lower > 1:
                candidate = (upper + lower) // 2
                if probe(candidate):
                    lower = candidate
                else:
                    upper = candidate
        if lower == 0:
            raise RuntimeError(f'No batch fits the {args.target_memory:.0%} memory target')
        batch = lower
        selection = {'gpu_ids': selected, 'gpu_count': args.gpus, 'batch_size': batch,
                     'accum': args.accum, 'target_memory': args.target_memory,
                     'probes': records, 'selected_peak_fraction':
                     next(record['fraction'] for record in records if record['batch_size'] == batch)}
        selection_path.write_text(json.dumps(selection, indent=2) + '\n')
        print(f'Selected per-GPU batch={batch} at {selection["selected_peak_fraction"]:.2%} memory', flush=True)
    command = base + ['--output', str(args.output), '--batch-size', str(batch),
                      '--accum', str(args.accum), '--epochs', str(args.epochs),
                      '--limit-samples', str(args.limit_samples),
                      '--max-steps', str(args.train_max_steps),
                      '--wandb-project', args.wandb_project, '--wandb-mode', args.wandb_mode]
    print(f'Starting training: GPUs={selected}, batch={batch}, accum={args.accum}, epochs={args.epochs}', flush=True)
    train_log = args.output / 'train.log'
    exit_code = run_logged(command, train_log, env, timeout=None, append=True)
    if exit_code:
        raise RuntimeError(f'Training exited with code {exit_code}; see {train_log}')
    print(f'Training finished; see {train_log}', flush=True)


if __name__ == '__main__':
    main()
