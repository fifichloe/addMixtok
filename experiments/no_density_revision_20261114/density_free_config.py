"""New density-free experiment identities; historical experiments stay immutable."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'experiments/generation_dynamic_router_20261112/data'
OUTPUT = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/no_density_revision_20261114')
VAL_PATHS = OUTPUT.parent / 'rebuttal_20261106/reconstruction_50k/validation_files.txt'
OLD = ROOT / 'experiments/independent_onepass_goal110_20261112'
FRESH = ROOT / 'experiments/fresh_reconstruction_scorer_20261112'
HEADS = {'short': OLD / 'dense_gain_head/best.pt',
         'long': ROOT / 'experiments/independent_onepass_goal105_20261112/long_head/best.pt'}
KINDS = ('short', 'long')
CAM = {'short': .5, 'long': .75}
UTILITY_WEIGHT = {'short': .3, 'long': .9}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_identities():
    paths = [HERE / 'density_free_config.py', HERE / 'teacher_no_density.py', FRESH / 'runtime.py',
             OLD / 'gain_model.py', OLD / 'image_utility/semantic_rel.npz',
             ROOT / 'experiments/light_grid_105_20261109/input_scales.npz', *HEADS.values(),
             Path('/home/heyefei/lichenge/MixTok-Official/assets/released/recon/mixtok_recon.pt'),
             Path('/home/heyefei/lichenge/1d-tokenizer/tokenizer_titok_l32.bin'),
             Path('/home/heyefei/lichenge/LlamaGen/pretrained_models/vq_ds16_c2i.pt'),
             Path('/tmp/mixtok_light_backbones_20261111/efficientnet_b0_rwightman-7f5810bc.pth')]
    return {str(path): sha(path) for path in paths}


def policies():
    record = HERE / 'teacher_policies.json'
    value = json.loads(record.read_text())
    assert value['status'] == 'full_validation_density_free_scores_and_policies_frozen'
    assert value['validation_n'] == 50000 and not value['quality_used_for_calibration']
    assert value['source_sha256'] == source_identities()
    return value['policies']
