"""Frozen-codec reconstruction ablations. Importing this file does not use CUDA."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
OLD = PAPER/'experiments/independent_onepass_goal110_20261112'
TEACHERS = PAPER/'experiments/independent_onepass_goal100_20261113'
FRESH = PAPER/'experiments/fresh_reconstruction_scorer_20261112'
OUT = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/new_router_reconstruction_ablations_20261114')
VAL_PATHS = OUT.parent/'rebuttal_20261106/reconstruction_50k/validation_files.txt'
CODEC = Path('/home/heyefei/lichenge/MixTok-Official/assets/released/recon/mixtok_recon.pt')
PYTHON = '/home/heyefei/miniconda3/envs/MoT/bin/python'
KINDS = ('short', 'long')
POLICIES = dict(short=dict(name='short_c0.5_u0.3', cam=.5, utility_weight=.3, threshold=.3232019245624542,
                          frozen_policy_name='u_semantic_rel_0.3',
                          threshold_manifest=str(PAPER/'experiments/independent_onepass_goal105_20261112/val_budget_frozen.json')),
                long=dict(name='long_c0.75_u0.9', cam=.75, utility_weight=.9, threshold=.4321809709072113,
                          frozen_policy_name='long_c0.75_u0.9',
                          threshold_manifest=str(TEACHERS/'val_budget_frozen.json')))


def formal_teacher_policy(kind):
    """Use accepted validation score-only deployment thresholds, not FIT thresholds."""
    row=POLICIES[kind];manifest=Path(row['threshold_manifest']);frozen=json.loads(manifest.read_text())
    assert frozen['status']=='frozen_score_only_validation_budget' and frozen['no_quality_used']
    assert frozen['policies'][row['frozen_policy_name']]['threshold']==row['threshold']
    assert frozen['validation_paths_sha256']==sha(VAL_PATHS)
    return dict(row,threshold_manifest_sha256=sha(manifest),threshold_source='accepted_validation_score_only_frozen_deployment')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16*1024*1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_inputs():
    sys.path[:0] = [str(OLD), str(FRESH)]
    import routing
    return routing.FrozenInputs()


def project_source_hashes():
    """Fingerprint modules actually imported by the running entrypoint."""
    paths=set()
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename and str(filename).endswith('.py'):
            path=Path(filename).resolve()
            if str(path).startswith('/home/heyefei/lichenge/') and path.is_file():paths.add(path)
    return {str(path):sha(path) for path in sorted(paths)}


class Scores:
    """The decomposition is copied exactly from the accepted one-pass formula."""
    def __init__(self, kind, runtime):
        import torch
        sys.path.insert(0, str(OLD))
        from routing import AnalyticMaps, standard, SCALES
        from gain_model import GainScorer
        self.kind, self.policy, self.maps, self.standard = kind, formal_teacher_policy(kind), AnalyticMaps(), standard
        source = OLD/'dense_gain_head/best.pt' if kind == 'short' else PAPER/'experiments/independent_onepass_goal105_20261112/long_head/best.pt'
        state = torch.load(source, map_location='cpu', weights_only=False)
        for path, digest in state['source_sha256'].items(): assert sha(path) == digest, path
        self.head = GainScorer().cuda().eval().requires_grad_(False); self.head.load_state_dict(state['model'], strict=True)
        import numpy as np
        with np.load(OLD/'image_utility/semantic_rel.npz') as data:
            self.utility = {key: torch.as_tensor(data[key], device='cuda', dtype=torch.float32) for key in ['mean','std','beta','bias']}
        bundle = OUT.parent/f'generation_{kind}_budget_feature_selection_20261114'/f'accepted_{kind}_inference'
        self.student_config = json.loads((bundle/'config.json').read_text())
        assert sha(bundle/'model.py') == self.student_config['exported_model_sha256']
        assert sha(bundle/'student_state.pt') == self.student_config['exported_weights_sha256']
        model = load_file(f'ablations_student_{kind}', bundle/'model.py')
        self.student = model.GenerationRouter(self.student_config['utility_weight'], self.student_config['threshold']).cuda().eval().requires_grad_(False)
        self.student.load_state_dict(torch.load(bundle/'student_state.pt', map_location='cpu', weights_only=True), strict=True)
        from train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic import DynamicBudgetRouter
        self.legacy = DynamicBudgetRouter(latent_channels=256, hidden_dim=128, depth=3, target_ratio=.375,
                                         min_ratio=0., max_ratio=1., detach_inputs=True).cuda().eval().requires_grad_(False)
        original = torch.load(CODEC, map_location='cpu', mmap=True, weights_only=True)['model_ema']
        router = {key.removeprefix('router.'): value for key,value in original.items() if key.startswith('router.')}
        # Original release stores parameters only; its position embedding is a parameter too.
        self.legacy.load_state_dict(router, strict=True)
        self.sources = {str(path): sha(path) for path in [source, OLD/'image_utility/semantic_rel.npz',
                        bundle/'model.py', bundle/'student_state.pt', bundle/'config.json', CODEC, SCALES,
                        FRESH/'runtime.py', OLD/'routing.py', OLD/'gain_model.py', Path(self.policy['threshold_manifest'])]}
        self.sources.update({str(path):sha(path) for path in [
            Path('/tmp/mixtok_light_backbones_20261111/efficientnet_b0_rwightman-7f5810bc.pth'),
            Path('/home/heyefei/lichenge/1d-tokenizer/tokenizer_titok_l32.bin'),
            Path('/home/heyefei/lichenge/LlamaGen/pretrained_models/vq_ds16_c2i.pt')]})
        self.versions = [(name,t,t._version) for module_name,module in [('head',self.head),('student',self.student),('legacy',self.legacy)]
                         for name,t in [(module_name+'.'+name,t) for name,t in [*module.named_parameters(),*module.named_buffers()]]]

    def __call__(self, z):
        import torch
        maps = self.maps(z); x = z['features'].half().float()
        gain = self.standard(self.head(torch.cat([x, maps['white'].half().float()], 1))[:,1:2])
        pooled = torch.cat([x.mean((2,3)),x.std((2,3),correction=0)],1); u = self.utility
        utility = ((((pooled-u['mean'])/u['std']).clamp(-8,8))@u['beta']+u['bias']).clamp(-3,3)[:,None,None,None]
        components = dict(gain=gain, white=.5*maps['white'], cam=self.policy['cam']*maps['cam'], density=-.25*maps['density'])
        local = sum(components.values()); teacher = local+self.policy['utility_weight']*utility
        with torch.autocast('cuda', enabled=False): student = self.student(z['f1'].float())
        with torch.autocast('cuda', dtype=torch.bfloat16): legacy,_ = self.legacy(z['f1'], z['base'], z['f2'])
        return dict(teacher=teacher, local=local, generation=student['score'].reshape(-1,1,16,16),
                    legacy=legacy.float(), **{'without_'+key: teacher-value for key,value in components.items()})

    def audit(self):
        assert all(t._version==v and not t.requires_grad and t.grad is None for _,t,v in self.versions)
        return dict(all_scorers_frozen=True, scorer_asset_sha256=self.sources,
                    original_scorer_only_used_as_explicit_A01_baseline=True,formal_teacher_policy=self.policy)
