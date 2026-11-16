"""Independent control generators with frozen accepted routers and original recipe."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
GEN = Path('/home/heyefei/lichenge/MixTok-Official/generation')
BASE_OUT = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/generation_two_router_maskgit_20261114')
OUT = BASE_OUT.parent/'new_router_generation_controls_20261114'
POLICIES = ('random', 'uniform', 'nearconstant')
SOURCE = OUT.parent/'rebuttal_training_20261107/paired_top_p_codes'
ASSETS = GEN.parent/'assets/generation'
WEIGHTS = OUT.parent/'continuous_pruned_hf_stage_20260924'
TOP = WEIGHTS/'top_p_ema_inference_v1_step247482.pt'
ADAPTER = WEIGHTS/'adapter_step20000.pt'
PRETRAINED = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/external_weights/halton_maskgit/ImageNet_256_base.pth')
UPSTREAM = Path('/home/heyefei/lichenge/MoT_maskgit/third_party/Halton-MaskGIT')
PYTHON = '/home/heyefei/miniconda3/envs/MoT/bin/python'
GPUS = os.environ.get('MIXTOK_CONTROL_GPUS', '0,1,2,4')
assert len(GPUS.split(',')) == len(set(GPUS.split(','))) == 4
assert set(GPUS.split(',')) <= {'0', '1', '2', '3', '4'}
KINDS = ('short', 'long')
sys.path.insert(0, str(GEN))
os.environ.setdefault('MOTAR_ASSETS', str(ASSETS))
os.environ.setdefault('USE_TF', '0')


def module(name):
    return importlib.import_module('experiments.continuous_halton_embedding_20260923.'+name)


def bundle(kind):
    assert kind in KINDS
    return OUT.parent/f'generation_{kind}_budget_feature_selection_20261114'/f'accepted_{kind}_inference'


def load_router(kind, device):
    folder = bundle(kind)
    spec = importlib.util.spec_from_file_location(f'accepted_{kind}_router', folder/'model.py')
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    import torch
    config = json.loads((folder/'config.json').read_text())
    assert config['student'] == kind and config['Kmin'] == 48 and config['Kmax'] == 144
    scorer = implementation.GenerationRouter(config['utility_weight'], config['threshold'])
    scorer.load_state_dict(torch.load(folder/'student_state.pt', map_location='cpu', weights_only=True), strict=True)
    return scorer.to(device).eval().requires_grad_(False), config


def router_identity(kind):
    from bert2d.paths import sha256
    folder = bundle(kind)
    config = json.loads((folder/'config.json').read_text())
    assert sha256(folder/'model.py') == config['exported_model_sha256']
    assert sha256(folder/'student_state.pt') == config['exported_weights_sha256']
    return dict(kind=kind, bundle=str(folder), model_sha256=sha256(folder/'model.py'),
                weights_sha256=sha256(folder/'student_state.pt'), config_sha256=sha256(folder/'config.json'),
                threshold=config['threshold'], utility_weight=config['utility_weight'],
                minimum_K=48, maximum_K=144, target_validation_mean_K=96,
                policy='one forward, scalar U shift, fixed threshold count, stable top K',
                scorer_precision='FP32', input_feature_precision='BF16 frozen spatial expansion')


def router_features(ids, quantizer, decoder):
    """Match the BF16 feature grid used to train/accept the new scorers.

    The original frozen adapter separately retains its original FP32 features.
    Neither path sees RGB or GT local codes. Both are pure functions of 1D IDs.
    """
    import torch
    with torch.no_grad(), torch.autocast(ids.device.type, enabled=False):
        z = quantizer.get_codebook_entry(ids.reshape(-1))
        z = z.reshape(len(ids), 1, 32, -1).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad(), torch.autocast(ids.device.type, dtype=torch.bfloat16):
        features = decoder(z.float())
    assert features.shape == (len(ids), 256, 16, 16) and features.dtype == torch.bfloat16
    return features.detach()


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
    return OUT/f"{kind}_{policy}_seed0"
