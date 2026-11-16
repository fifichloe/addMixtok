"""Load a component-ablated base-only router with its held-set calibration."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
KINDS = ('without_gain', 'without_white_all_explicit', 'without_cam_all_explicit')
MODEL = HERE.parent.parent / 'addMixtok_metrics_work/mixtok_long_no_density/gene/scorer_training/budget/model.py'
GENERATOR_CHECKPOINT = Path('/tmp/mixtok_gen_score_weights/long_seed0/latest.pt')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(kind):
    assert kind in KINDS
    checkpoint = HERE / f'budget_{kind}/best.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['config']['kind'] == 'long'
    assert saved['new_U_updates'] > 0 and saved['new_U_updates'] <= 2000
    assert saved['config']['policy']['weight'] == .9
    calibration = saved['calibration']
    assert calibration['source'] == 'TRAIN_held_only' and abs(calibration['held_mean_K'] - 96) < .01
    return checkpoint, saved, calibration


def router_identity(kind):
    checkpoint, saved, calibration = read(kind)
    return dict(kind=kind, checkpoint=str(checkpoint), weights_sha256=sha(checkpoint),
        model_sha256=sha(MODEL), threshold=calibration['threshold'], utility_weight=.9,
        minimum_K=48, maximum_K=144, target_held_mean_K=96,
        student_input='base_features_only', one_forward_one_final_selection=True,
        explicit_component_removed=kind, MaskGIT_retrained=False,
        spatial_parent_sha256=saved['parent_checkpoint_sha256'])


def load_router(kind, device):
    checkpoint, saved, calibration = read(kind)
    spec = importlib.util.spec_from_file_location(f'ablation_model_{kind}', MODEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    router = module.GenerationRouter(.9, calibration['threshold'])
    router.load_state_dict(saved['model'], strict=True)
    return router.to(device).eval().requires_grad_(False), saved['config']
