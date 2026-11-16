"""Reuse only the original measurement primitives, never historical runtimes."""
import ast
import importlib.util
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
REV = HERE.parent
PAPER = REV.parents[1]
ORIGINAL = PAPER/'experiments/new_router_efficiency_20261114/profile_components.py'
spec = importlib.util.spec_from_file_location('current_density_free_measurement_primitives', ORIGINAL)
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
Meter = original.Meter
parameter_groups = original.parameter_groups
mask_from_score = original.mask_from_score
sha = original.sha
write_json = original.write_json


def source(path, name):
    assert name not in sys.modules
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deployment_helpers(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('render', 'mixed_direct')]
    assert sorted(node.name for node in nodes) == ['mixed_direct', 'render']
    scope = {'torch':torch}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), 'exec'), scope)
    return scope['render'], scope['mixed_direct'], ast.dump(module, include_attributes=False)
