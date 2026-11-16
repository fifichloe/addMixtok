"""Audit the spatial-stage checkpoint and configure the unchanged U stage."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
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
    spatial = HERE / f'spatial_{kind}'
    checkpoint = spatial / 'best.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['config']['kind'] == 'long'
    assert saved['new_updates'] > 0 and saved['new_updates'] <= 2000
    assert saved['step'] == saved['initial_step'] + saved['new_updates']
    assert len(saved['model']) == 21
    policy = json.loads((spatial / 'teacher_policies.json').read_text())
    budget = HERE / f'budget_{kind}'
    budget.mkdir(exist_ok=False)
    gate = budget / 'execution_gate.json'
    write(gate, {'status': 'user_authorized_long_only', 'ablation': kind,
                 'MaskGIT_retrained': False})
    config = {'data': json.loads((spatial / 'stage_config.json').read_text())['data'],
              'teacher_policies': str(spatial / 'teacher_policies.json'),
              'warmstart': str(checkpoint), 'warmstart_sha256': sha(checkpoint),
              'execution_gate': str(gate), 'parent_step': saved['step'],
              'parent_local_updates': saved['new_updates'], 'ablation': kind,
              'policy': policy['long']}
    write(budget / 'stage_config.json', config)
    print(json.dumps(config), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=KINDS, required=True)
    main(parser.parse_args().kind)
