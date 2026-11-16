"""Native fixed-budget baselines from the verified completed checkpoints."""
import json
from pathlib import Path

import numpy as np
import torch

from profiling_helpers import REV, Meter, parameter_groups, source, deployment_helpers, sha


class ReusedFixedGeneration:
    def __init__(self, args):
        self.args = args
        assert args.policy in ('random', 'uniform', 'dense') and args.budgets == [-1]
        assert args.policy_file is not None
        audit = json.loads(args.policy_file.read_text())
        assert audit['status'] == 'existing_random_uniform_Dense_all20k_and50k_complete_verified'
        record = audit['checkpoints'][args.policy]
        checkpoint = Path(record['checkpoint'])
        assert sha(checkpoint) == record['sha256']
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        config = saved['config']
        assert saved['step'] == 20000 and config['route_mode'] == args.policy
        assert config['fixed_k'] == (256 if args.policy == 'dense' else 96)
        self.c = source(REV/'maskgit/common.py', 'verified_assets_for_reused_fixed_profile')
        c = self.c
        assert config['adapter_sha256'] == sha(c.ADAPTER)
        self.model = c.module('continuous_halton').ContinuousHalton(c.UPSTREAM)
        self.model.load_state_dict(saved['model'], strict=True)
        self.model = self.model.cuda().eval().requires_grad_(False)
        del saved
        dense = args.policy == 'dense'
        self.assets = c.module('assets_top_p').TopPFrozenAssets(c.ASSETS, c.TOP, 'cpu' if dense else 'cuda', chunk=4)
        if dense:
            # All positions use discrete embeddings: context cannot affect x
            # in ContinuousHalton.forward's torch.where(selected, embedding, context).
            self.assets.shell.llamagen_vq.cuda()
            del self.assets.shell.latent_decoder
            del self.assets.native
        else:
            from bert2d.paths import ONE_D
            from bert2d.eval import ImageBert, OfficialSamplingView
            from h20.assets import official_model
            self.adapter, _ = c.module('embedding_adapter').load_adapter(c.ADAPTER, 'cuda',
                c.module('assets_top_p').TOP_P_SHA256, 20000)
            self.one_d, _ = official_model(c.ASSETS)
            for name in ('embedding_2d', 'pos_embedding_2d', 'budget_embedding', 'output_2d'):
                delattr(self.one_d, name)
            self.one_d = self.one_d.cuda().eval().requires_grad_(False)
            self.view = OfficialSamplingView(self.one_d).eval()
            self.ImageBert = ImageBert
            self.stage1 = dict(ONE_D, guidance_scale=4.5, randomize_temperature=9.5)
        self.render, self.mixed_direct, _ = deployment_helpers(REV/'generation_eval/worker.py')
        self.uniform = None
        if args.policy == 'uniform':
            routing = source(REV.parent/'uniform_dense_training_20261107/routing.py', 'native_uniform96_profile_positions')
            self.uniform = torch.tensor(routing.positions('uniform'), device='cuda')
        vq = self.assets.shell.llamagen_vq
        modules = dict(stage2_halton=self.model, local_codebook_projection=[vq.quantize, vq.post_quant_conv],
                       shared_rgb_decoder=vq.decoder)
        if not dense:
            modules.update(stage1_generator=self.one_d, context_adapter=self.adapter,
                           spatial_expansion=self.assets.shell.latent_decoder, one_d_codebook=self.assets.native.quantize)
        self.metadata = dict(policy=args.policy, fixed_K=config['fixed_k'], checkpoint=str(checkpoint),
            checkpoint_sha256=record['sha256'], training_step=20000, reused_completed_training=True,
            stage1_rounds=0 if dense else 8, stage2_rounds=args.stage2_steps, cfg_stage2=args.cfg,
            scorer_loaded=False, density_computed=False, parameters=parameter_groups(modules),
            dense_native_pipeline_omits_unused_stage1_base_expansion_and_adapter=dense,
            dense_historical_quality_pipeline_computed_unused_stage1=True,
            dense_equivalence='For selected=True at every location, continuous context is algebraically excluded from model input and final grid.',
            effective_content_tokens=256 if dense else 128,
            profile_inputs='Batch1 classes0..49, stage1 seed4+index, stage2 reset seed5+index; timing cohort, not historical50k cohort',
            random_selection='Uniform without replacement with independent per-input NumPy RNG; not historical cohort RNG replay',
            precision='BF16 generation/adapter, FP32 base expansion and RGB projection/decoding, TF32off',
            additional_models_at_inference='None; no scoring, B0, DINO, LPIPS or Inception',
            source_sha256={str(path):sha(path) for path in [Path(__file__), REV/'maskgit/common.py',
                REV/'generation_eval/worker.py', Path(c.module('continuous_halton').__file__)]})
        parts = [('stage2', self.model), ('vq', vq)]
        if not dense:
            parts += [('stage1', self.one_d), ('adapter', self.adapter),
                      ('expansion', self.assets.shell.latent_decoder), ('native', self.assets.native)]
        self.versions = [(prefix+'.'+name, value, value._version) for prefix, module in parts
                         for name, value in [*module.named_parameters(), *module.named_buffers()]]

    def run(self, k, index, mode):
        assert k == -1
        c, m = self.c, Meter(mode)
        labels = torch.full((1,), index % 1000, device='cuda', dtype=torch.long)
        dense = self.args.policy == 'dense'
        if dense:
            context = torch.zeros((1, 256, 768), device='cuda', dtype=torch.bfloat16)
            mask = torch.ones((1, 256), device='cuda', dtype=torch.bool)
            base = torch.zeros((1, 256, 16, 16), device='cuda')
        else:
            torch.manual_seed(4+index)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ids = m.call('stage1_generation', lambda:self.ImageBert.generate(self.view, condition=labels, **self.stage1))
            base = m.call('base_expansion_FP32', lambda:c.module('adapter_data').make_features(
                ids, self.assets.native.quantize, self.assets.shell.latent_decoder))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                context = m.call('context_embedding_adapter', lambda:self.adapter(base))
            def select():
                selected = torch.zeros((1, 256), device='cuda', dtype=torch.bool)
                positions = (self.uniform if self.uniform is not None else torch.tensor(
                    np.random.default_rng(20261107+index).choice(256, 96, replace=False), device='cuda'))
                selected[:, positions] = True
                return selected
            mask = m.call('fixed_spatial_selection', select)
        torch.manual_seed(5+index)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            completed = m.call('stage2_generation', lambda:c.module('continuous_halton').sample_continuous_completion(
                self.model, context, mask, labels, c.UPSTREAM, steps=self.args.stage2_steps,
                cfg_w=self.args.cfg, temperature=1.))
        mixed = m.call('local_codebook_projection_and_mixing', lambda:self.mixed_direct(self.assets, base, completed, mask))
        rgb = m.call('final_rgb_decoder', lambda:self.render(self.assets, mixed))
        return rgb, 256 if dense else 96, m, dict(mask=mask)

    def audit(self):
        rgb, count, _, extra = self.run(-1, 0, 'off')
        assert torch.isfinite(rgb).all() and int(extra['mask'].sum()) == count
        return dict(fixed_selection_replayed=True, finite_final_RGB=True,
                    no_generation_scorer_or_density=True, frozen_checkpoint_reused=True,
                    dense_context_excluded_at_all_locations=self.args.policy == 'dense')

    def final_audit(self):
        assert all(not value.requires_grad and value.grad is None and value._version == version
                   for _, value, version in self.versions)
        return dict(all_loaded_parameters_and_buffers_frozen_unchanged=True,
                    no_new_training=True, baseline_checkpoint_reused=True)
