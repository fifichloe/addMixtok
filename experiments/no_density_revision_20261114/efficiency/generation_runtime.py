"""Current main and separately trained controls, with the actual generation path."""
import json
from pathlib import Path

import numpy as np
import torch

from profiling_helpers import (REV, ORIGINAL, original, Meter, parameter_groups,
    mask_from_score, source, deployment_helpers, sha)


class Generation:
    router_output = original.Generation.router_output
    final_audit = original.Generation.final_audit

    def __init__(self, args):
        self.args = args
        variant = getattr(args, 'checkpoint_variant', 'main')
        source_folder = 'fixed96_control' if variant == 'fixed96' else 'maskgit'
        self.c = source(REV/source_folder/'common.py', 'current_density_free_generation_profile_common')
        c = self.c
        from bert2d.paths import ONE_D
        from bert2d.eval import ImageBert, OfficialSamplingView
        from h20.assets import official_model
        self.policy = args.policy
        self.document = None
        self.control_routing = None
        if self.policy == 'learned':
            checkpoint = c.OUT/f'{args.kind}_seed0/latest.pt'
            implementation = REV/source_folder
            suffix = '_fixed96' if variant == 'fixed96' else ''
            audit_path = REV/f'final_execution/{args.kind}{suffix}_MaskGIT_training_audit.json'
            expected_audit = 'formal_20000_training_independently_audited'
            assert args.policy_file is None
        else:
            assert args.policy_file is not None and args.budgets == [-1]
            self.document = json.loads(args.policy_file.read_text())
            assert self.document['status'] == 'frozen_before_independent_control_quality'
            assert self.document['kind'] == args.kind and self.document['cfg'] == args.cfg
            assert self.document['seed'] == 4 and self.document['stage2_steps'] == args.stage2_steps == 8
            assert self.document['selection_policy']['policy'] == self.policy
            implementation = REV/'independent_controls'
            checkpoint = c.OUT.parent/'generation_controls/independent'/f'{args.kind}_{self.policy}_seed0/latest.pt'
            audit_path = Path(self.document['actual_training_audit'])
            assert sha(audit_path) == self.document['actual_training_audit_sha256']
            expected_audit = 'formal_20000_control_training_independently_audited'
            self.control_routing = source(implementation/'routing.py', 'current_control_profile_routing')
        audit = json.loads(audit_path.read_text())
        assert audit['status'] == expected_audit
        receipt = json.loads(checkpoint.with_suffix('.json').read_text())
        digest = sha(checkpoint)
        assert receipt['step'] == 20000 and receipt['sha256'] == digest
        assert audit['checkpoint'] == str(checkpoint) and audit['checkpoint_sha256'] == digest
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        assert state['step'] == 20000 and state['config']['kind'] == args.kind
        assert not state['config']['profile'] and state['config']['router_identity'] == c.router_identity(args.kind)
        if variant == 'fixed96':
            assert state['config']['fixed_k'] == 96 and args.budgets == [96]
        for name, expected in state['config']['implementation_sha256'].items():
            assert sha(implementation/name) == expected, name
        for name, expected in state['config']['original_modules_sha256'].items():
            assert sha(Path(c.module(name).__file__)) == expected, name
        if self.document is not None:
            assert self.document['checkpoint_sha256'] == digest
            assert self.document['training_control_identity'] == state['config']['control_identity']
        c.module('halton_upstream').verify_pinned_upstream(c.UPSTREAM)
        self.model = c.module('continuous_halton').ContinuousHalton(c.UPSTREAM)
        self.model.load_state_dict(state['model'], strict=True)
        self.model = self.model.cuda().eval().requires_grad_(False)
        del state
        self.assets = c.module('assets_top_p').TopPFrozenAssets(c.ASSETS, c.TOP, 'cuda', chunk=4)
        self.adapter, _ = c.module('embedding_adapter').load_adapter(c.ADAPTER, 'cuda',
            c.module('assets_top_p').TOP_P_SHA256, 20000)
        self.scorer, _ = c.load_router(args.kind, 'cuda')
        self.one_d, one_d_audit = official_model(c.ASSETS)
        for name in ('embedding_2d', 'pos_embedding_2d', 'budget_embedding', 'output_2d'):
            delattr(self.one_d, name)
        assert sum(p.numel() for p in self.one_d.parameters()) == 177202432
        self.one_d = self.one_d.cuda().eval().requires_grad_(False)
        self.view = OfficialSamplingView(self.one_d).eval()
        self.ImageBert = ImageBert
        self.stage1 = dict(ONE_D, guidance_scale=4.5, randomize_temperature=9.5)
        assert self.stage1['num_sample_steps'] == 8
        self.cfg, self.steps = args.cfg, args.stage2_steps
        self.render, self.mixed_direct, helper_ast = deployment_helpers(REV/'generation_eval/worker.py')
        _, _, control_ast = deployment_helpers(REV/'independent_controls/eval_worker.py')
        assert helper_ast == control_ast
        vq, scorer = self.assets.shell.llamagen_vq, self.scorer
        self.metadata = dict(kind=args.kind, policy=self.policy, checkpoint=str(checkpoint),
            checkpoint_sha256=digest, actual_training_audit=str(audit_path),
            actual_training_audit_sha256=sha(audit_path), router=c.router_identity(args.kind),
            one_d_audit=one_d_audit, stage1=self.stage1,
            stage2=dict(steps=self.steps, cfg_w=self.cfg, temperature=1.),
            assets=dict(self.assets.audit), adapter_sha256=sha(c.ADAPTER),
            parameters=parameter_groups(dict(stage1_generator=self.one_d, stage2_halton=self.model,
                spatial_expansion=self.assets.shell.latent_decoder, one_d_codebook=self.assets.native.quantize,
                local_CNN=[scorer.f_proj, scorer.pos_embed, scorer.trunk, scorer.score_head],
                budget_CNN_and_offset=[scorer.budget_f_proj, scorer.budget_pos_embed,
                    scorer.budget_trunk, scorer.offset_head], context_adapter=self.adapter,
                local_codebook_projection=[vq.quantize, vq.post_quant_conv], shared_rgb_decoder=vq.decoder)),
            source_sha256={str(p):sha(p) for p in [Path(__file__), ORIGINAL,
                REV/'maskgit/common.py', REV/'generation_eval/worker.py',
                REV/'independent_controls/eval_worker.py',
                Path(c.module('continuous_halton').__file__), Path(c.module('adapter_data').__file__),
                Path(c.module('embedding_adapter').__file__)]},
            precision='BF16 stage1/stage2/adapter; FP32 adapter expansion; separate BF16 router expansion; FP32 scorer/projection/decoder; TF32off',
            additional_models_at_inference='None; no B0, DINO, LPIPS, Inception or quality evaluator',
            base_rgb_control_decode_included=False,
            inputs='Batch1, classes0..49, torch seeds4..53, identical across profiled checkpoints; not the full50k quality batch plan',
            endpoint_scope='Frozen checkpoint fixed-K diagnostics retain routing/adapter/both expansions; no optimized K0 bypass',
            inference_policy=self.document['selection_policy'] if self.document else 'current learned dynamic or specified fixed-K diagnostic',
            checkpoint_variant=variant,
            policy_file_sha256=sha(args.policy_file) if args.policy_file else None,
            actual_control_training_identity=self.document['training_control_identity'] if self.document else None,
            no_new_training_or_budget_calibration=True)
        if self.document:
            self.metadata['source_sha256'][str(implementation/'routing.py')] = sha(implementation/'routing.py')
        self.versions = [(prefix+'.'+name, tensor, tensor._version)
            for prefix, part in [('stage1', self.one_d), ('stage2', self.model),
                ('adapter', self.adapter), ('router', self.scorer),
                ('native', self.assets.native), ('codec', self.assets.shell)]
            for name, tensor in [*part.named_parameters(), *part.named_buffers()]]

    def selection(self, output, index, k):
        if self.policy == 'learned':
            if getattr(self.args, 'checkpoint_variant', 'main') == 'fixed96':
                k = 96
            return mask_from_score(output['score'], self.scorer.threshold, k)
        assert k == -1
        counts = (output['score'] >= self.scorer.threshold).sum(1).clamp(48, 144)
        # Shape scaffold only: the control constructs the sole final mask.
        scaffold = dict(**output, K=counts, mask=torch.empty_like(output['score'], dtype=torch.bool))
        policy = self.document['selection_policy']
        return self.control_routing.select_control(self.policy, scaffold, np.array([index]),
            policy['target_mean_K'], policy['route_seed'], 4, policy['total'], 'generation')

    def run(self, k, index, mode):
        c, m, seed = self.c, Meter(mode), 4+index
        labels = torch.full((1,), index % 1000, device='cuda', dtype=torch.long)
        torch.manual_seed(seed)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ids = m.call('stage1_generation', lambda:self.ImageBert.generate(self.view, condition=labels, **self.stage1))
        base = m.call('base_expansion_FP32_for_adapter_and_render',
            lambda:c.module('adapter_data').make_features(ids, self.assets.native.quantize, self.assets.shell.latent_decoder))
        scoring_base = m.call('separate_base_expansion_BF16_for_router',
            lambda:c.router_features(ids, self.assets.native.quantize, self.assets.shell.latent_decoder))
        with torch.autocast('cuda', enabled=False):
            output = self.router_output(scoring_base.float(), m)
        mask, counts = m.call('threshold_and_single_final_selection', lambda:self.selection(output, index, k))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            context = m.call('context_embedding_adapter', lambda:self.adapter(base))
        torch.manual_seed(seed+1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            completed = m.call('stage2_generation', lambda:c.module('continuous_halton').sample_continuous_completion(
                self.model, context, mask, labels, c.UPSTREAM, steps=self.steps, cfg_w=self.cfg, temperature=1.))
        mixed = m.call('local_codebook_projection_and_mixing',
            lambda:self.mixed_direct(self.assets, base, completed, mask) if bool(mask.any()) else base)
        rgb = m.call('final_rgb_decoder', lambda:self.render(self.assets, mixed))
        return rgb, int(counts.item()), m, dict(output=output, mask=mask, scoring_base=scoring_base)

    def audit(self):
        rgb, count, _, split = self.run(-1, 0, 'off')
        reference = self.scorer.select(split['scoring_base'].float())
        for key in ('local', 'utility', 'score'):
            assert torch.equal(reference[key], split['output'][key])
        if self.policy == 'learned':
            if getattr(self.args, 'checkpoint_variant', 'main') == 'fixed96':
                mask, counts = mask_from_score(reference['score'], self.scorer.threshold, 96)
            else:
                mask, counts = reference['mask'], reference['K']
        else:
            policy = self.document['selection_policy']
            mask, counts = self.control_routing.select_control(self.policy, reference, np.array([0]),
                policy['target_mean_K'], policy['route_seed'], 4, policy['total'], 'generation')
        assert torch.equal(split['mask'], mask) and count == int(counts.item())
        assert torch.isfinite(rgb).all()
        return dict(split_scorer_matches_exported_current_student_exactly=True,
            actual_control_selection_function_replayed=True, finite_RGB=True,
            no_GT_or_2d_inputs=True, final_RGB_only_decode=True)
