"""Reuse the frozen final-design assets/features, with separate control outputs."""
import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
_path = HERE.parent/'maskgit/common.py'
_spec = importlib.util.spec_from_file_location('final_design_maskgit_assets_for_controls', _path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

PAPER = _base.PAPER
GEN = _base.GEN
BASE_OUT = _base.OUT
OUT = BASE_OUT.parent/'generation_controls/independent'
POLICIES = ('random', 'uniform', 'nearconstant')
KINDS = _base.KINDS
SOURCE = _base.SOURCE
ASSETS = _base.ASSETS
WEIGHTS = _base.WEIGHTS
TOP = _base.TOP
ADAPTER = _base.ADAPTER
PRETRAINED = _base.PRETRAINED
UPSTREAM = _base.UPSTREAM
PYTHON = _base.PYTHON
GPUS = os.environ.get('MIXTOK_CONTROL_GPUS', '0,1,2,3')
assert len(GPUS.split(',')) == len(set(GPUS.split(','))) == 4
assert set(GPUS.split(',')) <= {'0', '1', '2', '3', '4'}

# These functions keep their original globals bound to the frozen main module.
# In particular, capsules remain the actual newly accepted final-design ones.
module = _base.module
bundle = _base.bundle
load_router = _base.load_router
router_identity = _base.router_identity
router_features = _base.router_features
verify_final_design = _base.verify_final_design


def environment():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPUS, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTORCH_ALLOC_CONF='expandable_segments:True', MOTAR_ASSETS=str(ASSETS),
               PYTHONUNBUFFERED='1', WANDB_MODE='offline')
    return env


def distributed(script, *args):
    return [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
            str(script), *map(str, args)]


def run_root(kind, policy):
    assert kind in KINDS and policy in POLICIES
    return OUT/f'{kind}_{policy}_seed0'
