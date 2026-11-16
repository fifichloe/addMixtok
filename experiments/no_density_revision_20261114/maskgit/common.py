"""Original continuous Halton recipe with the two accepted frozen routers."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[2]
GEN = Path('/home/heyefei/lichenge/MixTok-Official/generation')
OUT = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/no_density_revision_20261114/maskgit')
DATA_ROOT = OUT.parent.parent
REVISION = HERE.parent
SOURCE = DATA_ROOT/'rebuttal_training_20261107/paired_top_p_codes'
ASSETS = GEN.parent/'assets/generation'
WEIGHTS = DATA_ROOT/'continuous_pruned_hf_stage_20260924'
TOP = WEIGHTS/'top_p_ema_inference_v1_step247482.pt'
ADAPTER = WEIGHTS/'adapter_step20000.pt'
PRETRAINED = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/external_weights/halton_maskgit/ImageNet_256_base.pth')
UPSTREAM = Path('/home/heyefei/lichenge/MoT_maskgit/third_party/Halton-MaskGIT')
PYTHON = '/home/heyefei/miniconda3/envs/MoT/bin/python'
GPUS = os.environ.get('MIXTOK_TRAIN_GPUS', '0,1,2,3')
_devices = GPUS.split(',')
assert len(_devices) == len(set(_devices)) == 4 and all(item.isdecimal() for item in _devices)
KINDS = ('long',)
sys.path.insert(0, str(GEN))
os.environ.setdefault('MOTAR_ASSETS', str(ASSETS))
os.environ.setdefault('USE_TF', '0')


def module(name):
    return importlib.import_module('experiments.continuous_halton_embedding_20260923.'+name)


def bundle(kind):
    assert kind in KINDS
    return OUT.parent/f'accepted_{kind}_inference'


def load_router(kind, device):
    folder = bundle(kind)
    verify_final_design(kind)
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
    config = verify_final_design(kind)
    assert sha256(folder/'model.py') == config['exported_model_sha256']
    assert sha256(folder/'student_state.pt') == config['exported_weights_sha256']
    return dict(kind=kind, bundle=str(folder), model_sha256=sha256(folder/'model.py'),
                weights_sha256=sha256(folder/'student_state.pt'), config_sha256=sha256(folder/'config.json'),
                threshold=config['threshold'], utility_weight=config['utility_weight'],
                minimum_K=48, maximum_K=144, target_validation_mean_K=96,
                policy='one forward, scalar U shift, fixed threshold count, stable top K',
                scorer_precision='FP32', input_feature_precision='BF16 frozen spatial expansion',
                component_design_sha256=config['component_design_sha256'],
                density_component_removed=True,
                white_direct_weight=config['white_direct_weight'])


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


def verify_final_design(kind):
    """No new training can bind a historical or unfinalized routing capsule."""
    from bert2d.paths import sha256
    design_path = REVISION/'design_lock.json'
    design = json.loads(design_path.read_text())
    assert design['status'] == 'density_free_component_design_locked'
    assert design['density_component_removed'] is True
    expected = design['teachers'][kind]
    config = json.loads((bundle(kind)/'config.json').read_text())
    assert config['density_component_removed'] is True
    assert config['component_design_sha256'] == sha256(design_path)
    assert config['white_direct_weight'] == expected['white_direct_weight']
    assert config['teacher_policy_sha256'] == expected['teacher_policy_sha256']
    assert config['student'] == kind
    assert config['exported_model_sha256'] == sha256(bundle(kind)/'model.py')
    assert config['exported_weights_sha256'] == sha256(bundle(kind)/'student_state.pt')
    return config
