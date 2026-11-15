"""Load the accepted density-free, base-only generation router."""
import hashlib
import importlib.util
import json
from pathlib import Path
import torch


def load_router(bundle, device='cuda'):
    bundle=Path(bundle)
    config=json.loads((bundle/'config.json').read_text())
    assert config['student']=='long' and config['density_component_removed'] is True
    for name,field in [('model.py','exported_model_sha256'),('student_state.pt','exported_weights_sha256')]:
        assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==config[field], name
    spec=importlib.util.spec_from_file_location('accepted_no_density_Long',bundle/'model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.GenerationRouter(config['utility_weight'],config['threshold'])
    model.load_state_dict(torch.load(bundle/'student_state.pt',map_location='cpu',weights_only=True),strict=True)
    return model.to(device).eval().requires_grad_(False),config


@torch.no_grad()
def select(router, base):
    # base is the BF16 spatial expansion used during training, cast to FP32 here.
    with torch.autocast(base.device.type,enabled=False):
        return router.select(base.float())
