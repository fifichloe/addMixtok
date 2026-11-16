"""Profile the locked density-free reconstruction formula and unchanged codec."""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from profiling_helpers import (REV, ORIGINAL, original, Meter, parameter_groups,
                               mask_from_score, sha)
sys.path.insert(0, str(REV))
from final_teacher_config import (HEADS, CAM, UTILITY_WEIGHT, VAL_PATHS, policies,
                                   source_identities)
from teacher_final import TwoTeachers, ImagePathDataset
from runtime import standard


class Reconstruction:
    # The feature definition is identical across density revisions; the old
    # method only prepares raw features/CAM and never calls a density/scorer.
    feature_maps = original.Reconstruction.feature_maps

    def __init__(self, args):
        self.kind = args.kind
        self.engine = TwoTeachers()
        self.runtime = self.engine.runtime
        self.model = self.runtime.model
        self.head = self.engine.heads[args.kind]
        self.policy = policies()[args.kind]
        self.tau = self.policy['threshold']
        self.white_weight = self.engine.white_weight[args.kind]
        assert self.white_weight == .5
        paths = list(map(Path, VAL_PATHS.read_text().splitlines()[:args.repeats]))
        assert len(paths) == args.repeats
        ds = ImagePathDataset(paths, self.runtime.crop)
        self.images = torch.stack([ds[i] for i in range(len(ds))]).cuda()
        self.metadata = dict(kind=args.kind, policy=self.policy,
            density_computed=False, white_direct_weight=self.white_weight,
            parameters=parameter_groups(dict(
                one_d_encoder_and_quantizer=[self.model.titok.encoder, self.model.titok.quantize, self.model.titok.latent_tokens],
                spatial_expansion=self.model.latent_decoder,
                dense_2d_encoder_quantizer_projection=[self.model.llamagen_vq.encoder,
                    self.model.llamagen_vq.quant_conv, self.model.llamagen_vq.quantize,
                    self.model.llamagen_vq.post_quant_conv],
                shared_rgb_decoder=self.model.llamagen_vq.decoder,
                semantic_B0=self.runtime.semantic, gain_CNN=self.head)),
            utility_coefficients_and_bias=self.engine.utility['beta'].numel()+self.engine.utility['bias'].numel(),
            input_paths=[str(p) for p in paths], input_paths_sha256=sha(VAL_PATHS),
            source_sha256={**source_identities(), str(Path(__file__)):sha(Path(__file__)),
                           str(ORIGINAL):sha(ORIGINAL)},
            precision='BF16 codec/B0; FP32 scoring with original FP16 input rounding; TF32off',
            semantic_inference='B0 original/base concatenated batch2 at reconstruction batch1',
            endpoint_scope='All K retain dense2D/base decode/B0/scoring/sorting/final decode; no optimized K0 bypass',
            inactive_other_teacher_head_loaded_for_reference_audit=True,
            inactive_other_teacher_head_not_executed_in_timed_path=True,
            inactive_reference_head_excluded_from_deployed_parameters_but_present_in_memory=True)
        self.metadata['total_deployed_parameters'] = (
            self.metadata['parameters']['total_unique_parameters']+self.metadata['utility_coefficients_and_bias'])

    def run(self, k, index, mode):
        m = Meter(mode)
        x = self.images[index:index+1]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z, _ = m.call('1d_encoder_and_quantizer', lambda:self.model.encode_titok(x))
            f1 = m.call('spatial_expansion', lambda:self.model.latent_decoder(z))
            base = m.call('base_rgb_decoder', lambda:self.model.llamagen_vq.decoder(f1))
            f2, _ = m.call('dense_2d_encoder_quantizer_projection',
                lambda:self.runtime.ev.native_llamagen_feature(self.model.llamagen_vq, x*2-1, 8, False))
            semantic = m.call('semantic_B0_original_and_base',
                lambda:self.runtime.semantic(torch.cat([x, base.float().add(1).mul(.5).clamp(0, 1)])))
        parts = m.call('raw_feature_and_semantic_maps', lambda:self.feature_maps(x, base, f1, f2, semantic))

        def whiten():
            a, b = (f1.float()-self.engine.mean)/self.engine.std, (f2.float()-self.engine.mean)/self.engine.std
            return standard(1-F.cosine_similarity(a, b, dim=1).unsqueeze(1))

        white = m.call('whitened_feature_cosine', whiten)
        rounded = m.call('gain_inputs_fp16_rounding',
            lambda:torch.cat([parts['features'].half().float(), white.half().float()], 1))
        gain = m.call('learned_gain_CNN', lambda:standard(self.head(rounded)[:, 1:2]))

        def utility():
            features = parts['features'].half().float()
            pooled = torch.cat([features.mean((2, 3)), features.std((2, 3), correction=0)], 1)
            state = self.engine.utility
            return (((pooled-state['mean'])/state['std']).clamp(-8, 8)@state['beta']+state['bias']).clamp(-3, 3).reshape(-1)

        u = m.call('image_utility_statistics_and_linear', utility)
        local = m.call('local_score_combination',
            lambda:(gain+self.white_weight*white+CAM[self.kind]*parts['features'][:, 777:778]).flatten(1))
        score = m.call('final_image_utility_offset', lambda:local+UTILITY_WEIGHT[self.kind]*u[:, None])
        mask, counts = m.call('threshold_count_and_stable_sort', lambda:mask_from_score(score, self.tau, k))

        def mix():
            weight = mask.reshape(-1, 1, 16, 16).to(f1.dtype)
            return (1-weight)*f1+weight*f2

        mixed = m.call('feature_mixing', mix)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            rgb = m.call('final_rgb_decoder', lambda:self.model.llamagen_vq.decoder(mixed).float())
        return rgb, int(counts.item()), m, dict(score=score, mask=mask, prepared=parts)

    def audit(self):
        rgb, count, _, split = self.run(-1, 0, 'off')
        reference, prepared = self.engine(self.images[:1])
        index = ('short', 'long').index(self.kind)
        score = reference['local'][:, index]+UTILITY_WEIGHT[self.kind]*reference['utility'][:, None]
        mask, counts = mask_from_score(score, self.tau, -1)
        assert torch.equal(score, split['score']) and torch.equal(mask, split['mask'])
        assert int(counts.item()) == count
        expected = self.runtime.decode(prepared, mask.reshape(-1, 1, 16, 16).float())
        assert torch.equal(rgb, expected)
        return dict(split_path_matches_current_density_free_deployment_exactly=True,
                    density_computed=False, **self.engine.audit())
