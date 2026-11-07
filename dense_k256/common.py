"""Portable paths for the MixTok fixed-routing generation control."""
import importlib
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
COMMIT = '6623ccdc7a48688497f7b1f312548d1f14107e4d'


def required_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'Set {name} to the required local path (see README.md).')
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'{name}={path}')
    return path


REPO = required_path('MIXTOK_REPO')
GEN = REPO / 'generation'
SOURCE = required_path('MIXTOK_SOURCE_CACHE')
ASSETS = required_path('MIXTOK_ASSETS')
TOP = required_path('MIXTOK_TOP_CHECKPOINT')
ADAPTER = required_path('MIXTOK_ADAPTER_CHECKPOINT')
SCORER = required_path('MIXTOK_SCORER_CHECKPOINT')
PRETRAINED = required_path('MIXTOK_HALTON_PRETRAINED')
UPSTREAM = required_path('MIXTOK_HALTON_UPSTREAM')

os.environ.setdefault('MOTAR_ASSETS', str(ASSETS))
sys.path.insert(0, str(GEN))


def module(name):
    return importlib.import_module('experiments.continuous_halton_embedding_20260923.' + name)
