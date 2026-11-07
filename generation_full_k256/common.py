"""Portable MixTok and Halton paths for the full-grid generation control."""
import importlib
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
COMMIT = '6623ccdc7a48688497f7b1f312548d1f14107e4d'


def required_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'Set {name} to a local path (see README.md).')
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'{name}={path}')
    return path


REPO = required_path('MIXTOK_REPO')
GEN = REPO / 'generation'
UPSTREAM = required_path('MIXTOK_HALTON_UPSTREAM')
PRETRAINED = required_path('MIXTOK_HALTON_PRETRAINED')
sys.path.insert(0, str(GEN))


def module(name):
    return importlib.import_module('experiments.continuous_halton_embedding_20260923.' + name)
