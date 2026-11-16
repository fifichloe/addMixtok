"""Audit a completed distillation and prepare the unchanged spatial stage."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
DATA = PAPER / 'experiments/generation_dynamic_router_20261112/data'
KINDS = ('without_gain', 'without_white_all_explicit', 'without_cam_all_explicit')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(kind):
    assert kind in KINDS
    run = HERE / f'train_{kind}'
    report = json.loads((run / 'report.json').read_text())
    assert report['status'] == 'complete' and report['steps'] == 20000
    # Match the main router's 20,000-step parent checkpoint before the
    # identical reconstruction-guided spatial stage.
    checkpoint = run / 'latest.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['step'] == 20000
    assert saved['config']['kind'] == kind
    assert len(saved['model']) == 21
    steps = {int(state['step']) for state in saved['optimizer']['state'].values()}
    assert len(steps) == 1 and next(iter(steps)) == saved['step']
    assert len(saved['optimizer']['state']) == 21
    # The original spatial stage expects a checkpoint at its declared parent step.
    # Its optimizer is newly initialized; only the model state is transferred.
    policy = json.loads((HERE / 'teacher_policies.json').read_text())[kind]
    stage = HERE / f'spatial_{kind}'
    stage.mkdir(exist_ok=False)
    policy_path = stage / 'teacher_policies.json'
    write(policy_path, {'long': {'weight': policy['utility_weight'],
                                 'threshold': policy['threshold']}})
    gate = stage / 'execution_gate.json'
    write(gate, {'status': 'user_authorized_long_only', 'ablation': kind,
                 'MaskGIT_retrained': False})
    audit = stage / 'parent_audit.json'
    write(audit, {'actual_all21_optimizer_state_steps': saved['step'],
                  'all21_optimizer_state_steps_equal': True,
                  'checkpoint_sha256': sha(checkpoint),
                  'ablation': kind})
    config = {'data': str(DATA), 'teacher_policies': str(policy_path),
              'warmstart': str(checkpoint), 'warmstart_sha256': sha(checkpoint),
              'execution_gate': str(gate), 'parent_step': saved['step'],
              'parent_audit': str(audit), 'ablation': kind}
    write(stage / 'stage_config.json', config)
    print(json.dumps(config), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=KINDS, required=True)
    main(parser.parse_args().kind)
