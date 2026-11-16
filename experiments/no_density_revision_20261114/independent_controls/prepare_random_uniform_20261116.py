"""Prepare the two CPU-only independent generation-control route caches."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/no_density_revision_20261114')
ROOT = Path('/tmp/no_density_control_routes_20261116')
ROUTE_SEED = 20261114


def run(command, log):
    with log.open('a') as output:
        result = subprocess.run(command, cwd=HERE, stdout=output, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1'))
    if result.returncode:
        raise RuntimeError(f'{command[2]} failed with code {result.returncode}: {log}')


def main():
    train = json.loads((BASE/'maskgit/routes/long/summary.json').read_text())
    generated = json.loads((BASE/'generation_evaluation/long_cfg3_seed4_8p8/summary.json').read_text())
    assert train['complete_coverage'] and train['rows'] == 2562334
    assert generated['status'] == 'complete' and generated['n'] == 50000 and generated['seed'] == 4
    assert generated['router']['density_component_removed'] is True
    train_target = train['mean_k']
    generation_target = generated['tokens']['mean_generated_2d']
    ROOT.mkdir(exist_ok=True)
    for policy in ('random', 'uniform'):
        output = ROOT/policy
        log = ROOT/f'{policy}.log'
        common = ['--kind', 'long', '--policy', policy,
                  '--target-mean-k', str(generation_target),
                  '--train-target-mean-k', str(train_target),
                  '--route-seed', str(ROUTE_SEED), '--output', str(output), '--batch', '1024']
        script = str(HERE/'cache_routes.py')
        if not output.exists():
            run([sys.executable, script, 'prepare', *common], log)
        assert (output/'config.json').exists()
        children = []
        for rank in range(4):
            child_log = (ROOT/f'{policy}_rank{rank}.log').open('a')
            cmd = [sys.executable, script, 'extract', *common, '--rank', str(rank)]
            child = subprocess.Popen(cmd, cwd=HERE, stdout=child_log,
                                     stderr=subprocess.STDOUT,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1'))
            children.append((child, child_log))
        statuses = [child.wait() for child, _ in children]
        for _, handle in children:
            handle.close()
        if any(statuses):
            raise RuntimeError(f'{policy} route extraction failed: {statuses}')
        if not (output/'summary.json').exists():
            run([sys.executable, script, 'finalize', *common], log)
        summary = json.loads((output/'summary.json').read_text())
        assert summary['complete_coverage'] and summary['rows'] == 2562334
        assert summary['route_mode'] == f'control_{policy}'
        assert summary['matched_per_source_view_K']
        print(json.dumps(dict(policy=policy, status='complete', output=str(output),
                              mean_k=summary['mean_k'], elapsed=time.time())), flush=True)


if __name__ == '__main__':
    main()
