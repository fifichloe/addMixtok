"""One image-local feed-forward score; no redundancy map is computed or loaded."""
import sys
import numpy as np
import torch
from torch.nn import functional as F
from density_free_config import ROOT, OLD, FRESH, HEADS, KINDS, CAM, UTILITY_WEIGHT, sha, source_identities

sys.path[:0] = [str(FRESH), str(OLD)]
from runtime import FrozenInputs, ImagePathDataset, standard
from gain_model import GainScorer


class TwoTeachers:
    def __init__(self):
        self.runtime = FrozenInputs()
        scales = ROOT / 'experiments/light_grid_105_20261109/input_scales.npz'
        with np.load(scales) as data:
            self.mean = torch.as_tensor(data['mean'], device='cuda')[None, :, None, None]
            self.std = torch.as_tensor(data['std'], device='cuda')[None, :, None, None].clamp_min(1e-6)
        self.heads = {}
        for kind, path in HEADS.items():
            state = torch.load(path, map_location='cpu', weights_only=False)
            for source, digest in state['source_sha256'].items(): assert sha(source) == digest, source
            head = GainScorer().cuda().eval().requires_grad_(False)
            head.load_state_dict(state['model'], strict=True)
            self.heads[kind] = head
        with np.load(OLD / 'image_utility/semantic_rel.npz') as data:
            self.utility = {key: torch.as_tensor(data[key], device='cuda', dtype=torch.float32)
                            for key in ('mean', 'std', 'beta', 'bias')}
        self.versions = [(kind + '.' + name, value, value._version)
                         for kind, head in self.heads.items()
                         for name, value in [*head.named_parameters(), *head.named_buffers()]]
        self.identities = source_identities()

    @torch.no_grad()
    def score_prepared(self, z):
        # Preserve the original whitening, feature rounding, CAM and U exactly.
        a = (z['f1'].float() - self.mean) / self.std
        b = (z['f2'].float() - self.mean) / self.std
        white = standard(1 - F.cosine_similarity(a, b, dim=1).unsqueeze(1))
        features = z['features'].half().float()
        head_input = torch.cat([features, white.half().float()], 1)
        pooled = torch.cat([features.mean((2, 3)), features.std((2, 3), correction=0)], 1)
        state = self.utility
        utility = (((pooled - state['mean']) / state['std']).clamp(-8, 8)
                   @ state['beta'] + state['bias']).clamp(-3, 3).reshape(-1)
        local, gain = [], []
        for kind in KINDS:
            g = standard(self.heads[kind](head_input)[:, 1:2])
            gain.append(g.flatten(1))
            local.append((g + .5 * white + CAM[kind] * z['features'][:, 777:778]).flatten(1))
        return dict(f_base=z['f1'].detach(), local=torch.stack(local, 1), utility=utility,
                    gain=torch.stack(gain, 1), white=white.flatten(1),
                    cam=z['features'][:, 777:778].flatten(1))

    @torch.no_grad()
    def __call__(self, images):
        z = self.runtime.prepare(images)
        return self.score_prepared(z), z

    @torch.no_grad()
    def audit_labels(self, result, z):
        # Test the score's per-image semantics on identical prepared features;
        # separately disclose codec batching precision in validation replay.
        single = {key: value[:1] for key, value in z.items() if torch.is_tensor(value)}
        one = self.score_prepared(single)
        differences = {}
        for key in ('local', 'utility', 'gain', 'white', 'cam'):
            differences[key] = float((result[key][:1] - one[key]).abs().max())
            assert torch.allclose(result[key][:1], one[key], atol=1e-4, rtol=1e-4), (key, differences[key])
        assert not any(name == 'routing' for name in sys.modules), 'Historical analytic-map implementation was imported'
        return dict(no_redundancy_map_or_similarity_matrix_computed=True,
                    per_image_score_on_prepared_features_verified=True,
                    single_image_max_absolute_differences=differences,
                    no_provisional_mask_or_selected_set_feedback=True,
                    score='standard(gain)+0.5*white+cam_weight*CAM+utility_weight*U',
                    utility_weight=UTILITY_WEIGHT, **self.audit())

    def audit(self):
        assert all(value._version == version and not value.requires_grad and value.grad is None
                   for _, value, version in self.versions)
        return dict(heads_frozen_and_unchanged=True, density_removed_from_computation=True,
                    **self.runtime.audit())
