"""Actual new-router deployment paths: synchronized latency and counted FLOPs.

Does not run on import. Launch only after the experiment coordinator grants a
GPU. Timing excludes model startup, disk/PIL preprocessing, and quality metrics.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
RECON=PAPER/'experiments/independent_onepass_goal100_20261113'
GEN=PAPER/'experiments/generation_two_router_maskgit_20261114'
DEFAULT_OUT=Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/new_router_efficiency_20261114')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def source(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


class Meter:
    def __init__(self,mode):self.mode=mode;self.values={};self.operator_counts={}

    def call(self,name,fn):
        if self.mode=='time':
            torch.cuda.synchronize();start=time.perf_counter();out=fn();torch.cuda.synchronize()
            self.values[name]=(time.perf_counter()-start)*1000
        elif self.mode=='flops':
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as counter:out=fn()
            self.values[name]=counter.get_total_flops()
            self.operator_counts[name]={str(k):int(v) for k,v in counter.get_flop_counts().get('Global',{}).items()}
        else:out=fn()
        return out


def parameter_groups(groups):
    result={};all_seen=set()
    for name,items in groups.items():
        items=items if isinstance(items,list) else [items]
        params=[]
        for item in items:
            params.extend(item.parameters() if isinstance(item,torch.nn.Module) else [item])
        unique={id(p):p for p in params}
        result[name]=dict(parameters=sum(p.numel() for p in unique.values()),
            fresh_unique_parameters=sum(p.numel() for ident,p in unique.items() if ident not in all_seen),
            trainable_at_inference=sum(p.numel() for p in unique.values() if p.requires_grad))
        all_seen.update(unique)
    result['total_unique_parameters']=sum(v['fresh_unique_parameters'] for v in result.values())
    return result


def mask_from_score(score,tau,k):
    flat=score.float().flatten(1)
    if k<0:counts=(flat>=tau).sum(1).clamp(48,144)
    else:counts=torch.full((len(flat),),k,dtype=torch.long,device=flat.device)
    order=flat.argsort(dim=1,descending=True,stable=True)
    mask=torch.zeros_like(flat,dtype=torch.bool).scatter(1,order,
        torch.arange(256,device=flat.device)[None]<counts[:,None])
    return mask,counts


class Reconstruction:
    def __init__(self,args):
        sys.path.insert(0,str(RECON))
        from onepass_inference import OnePassEngine
        from validate_full import VAL_PATHS
        policy_root=RECON if args.kind=='long' else PAPER/'experiments/independent_onepass_goal105_20261112'
        self.accepted=json.loads((policy_root/'accepted_policy.json').read_text())
        for path,digest in self.accepted['shared_pretrained_assets'].items():assert sha(path)==digest,path
        for field in ['scorer_checkpoint','utility_checkpoint']:
            assert sha(self.accepted[field])==self.accepted[field+'_sha256']
        # The combined entrypoint names the original accepted short formula by
        # an alias; its checkpoint/maps/coefficients are unchanged. Do not use
        # the early TRAIN-calibrated short threshold from the search registry.
        self.name=self.accepted['name'] if args.kind=='long' else 'short_c0.5_u0.3'
        self.tau=self.accepted['threshold']
        frozen=json.loads(Path(self.accepted['frozen_source_and_threshold_manifest']).read_text())
        assert frozen['policies'][self.accepted['name']]['threshold']==self.tau
        self.engine=OnePassEngine(self.name);self.runtime=self.engine.runtime
        assert sha(self.engine.head_checkpoint)==self.accepted['scorer_checkpoint_sha256']
        self.model=self.runtime.model;self.scorer=self.engine.scorer;self.row=self.engine.row
        priority=self.accepted['priority']
        assert self.row['white']==priority['whitened_feature_cosine'] and self.row['cam']==priority['B0_CAM_missing']
        assert self.row['density']==priority['static_feature_density_penalty']
        assert self.row['utility_weight']==priority['independent_semantic_relative_image_utility']
        from routing import ImagePathDataset
        paths=list(map(Path,VAL_PATHS.read_text().splitlines()[:args.repeats]))
        assert len(paths)==args.repeats and all('/val/' in str(p) or '/validation/' in str(p) for p in paths)
        ds=ImagePathDataset(paths,self.runtime.crop)
        self.images=torch.stack([ds[i] for i in range(len(ds))]).cuda()
        self.paths=paths
        self.metadata=dict(kind=args.kind,policy=self.accepted,deployment_alias=self.name,
            accepted_policy_path=str(policy_root/'accepted_policy.json'),
            accepted_policy_sha256=sha(policy_root/'accepted_policy.json'),
            threshold_source=str(self.accepted['frozen_source_and_threshold_manifest']),
            parameters=parameter_groups(dict(one_d_encoder_and_quantizer=[self.model.titok.encoder,self.model.titok.quantize,self.model.titok.latent_tokens],
                spatial_expansion=self.model.latent_decoder,
                dense_2d_encoder_quantizer_projection=[self.model.llamagen_vq.encoder,self.model.llamagen_vq.quant_conv,self.model.llamagen_vq.quantize,self.model.llamagen_vq.post_quant_conv],
                shared_rgb_decoder=self.model.llamagen_vq.decoder,
                semantic_B0=self.runtime.semantic,gain_CNN=self.engine.head)),
            utility_coefficients_and_bias=sum(state['beta'].numel()+state['bias'].numel() for state in self.scorer.utility.values()),
            input_paths=[str(p) for p in paths],input_paths_sha256=sha(VAL_PATHS),
            source_sha256={str(p):sha(p) for p in [RECON/'candidates.py',RECON/'onepass_inference.py',RECON/'accepted_policy.json',
                PAPER/'experiments/fresh_reconstruction_scorer_20261112/runtime.py',
                PAPER/'experiments/independent_onepass_goal110_20261112/routing.py',
                PAPER/'experiments/independent_onepass_goal110_20261112/gain_model.py']},
            precision='Encoder, expansion, dense2D, B0, base/final RGB decoder BF16 autocast; score maps/gain/linear utility FP32 with original FP16 input rounding; TF32 disabled',
            semantic_inference='One concatenated batch [original RGB, base RGB], B0 batch=2 at reconstruction batch=1',
            endpoint_scope='All fixed K retain full dense2D, base decoding, B0, scoring, sorting, mixing, final decoding. K=0 is NOT an optimized base-only bypass.')
        self.metadata['total_deployed_parameters']=self.metadata['parameters']['total_unique_parameters']+self.metadata['utility_coefficients_and_bias']

    def feature_maps(self,x,base,f1,f2,semantic):
        # Exact FrozenInputs.prepare postprocessing, no legacy scorer.
        middle,deep,logits=semantic;n=len(x)
        mid_real,mid_base=middle[:n],middle[n:];real,basic=deep[:n],deep[n:]
        weights=self.runtime.semantic.net.classifier[1].weight[logits[:n].argmax(1)]
        missed=(torch.einsum('bc,bchw->bhw',weights,real).clamp_min(0)-
            torch.einsum('bc,bchw->bhw',weights,basic).clamp_min(0)).clamp_min(0)[:,None]
        cosine=1-F.cosine_similarity(f1.float(),f2.float(),dim=1).unsqueeze(1)
        error=x*2-1-base.float()
        pool=lambda v:F.adaptive_avg_pool2d(v,16)
        resized=lambda v:F.interpolate(v,size=(16,16),mode='bilinear',align_corners=False)
        from routing import standard
        mse=pool(error.square().mean(1,keepdim=True))
        maps=[cosine,pool(error.abs().mean(1,keepdim=True)),mse,
            torch.log1p((f2.float()-f1.float()).square().mean(1,keepdim=True)),
            torch.log1p(f1.float().square().mean(1,keepdim=True)),torch.log1p(f2.float().square().mean(1,keepdim=True)),
            resized(1-F.cosine_similarity(mid_real,mid_base,dim=1).unsqueeze(1)),
            resized(1-F.cosine_similarity(real,basic,dim=1).unsqueeze(1)),resized(missed),standard(resized(missed)),standard(mse),standard(cosine)]
        scale=float(f1.shape[1])**.5
        features=torch.cat([F.normalize(f1.float(),dim=1)*scale,F.normalize(f2.float(),dim=1)*scale,
            F.normalize((f2-f1).float(),dim=1)*scale,*maps],1)
        return dict(features=features.detach(),f1=f1.detach(),f2=f2.detach(),
            independent_prior=.6*standard(cosine)+.75*standard(resized(missed))+.3*standard(mse))

    def run(self,k,index,mode):
        m=Meter(mode);x=self.images[index:index+1]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z,_=m.call('1d_encoder_and_quantizer',lambda:self.model.encode_titok(x))
            f1=m.call('spatial_expansion',lambda:self.model.latent_decoder(z))
            base=m.call('base_rgb_decoder',lambda:self.model.llamagen_vq.decoder(f1))
            f2,_=m.call('dense_2d_encoder_quantizer_projection',lambda:self.runtime.ev.native_llamagen_feature(self.model.llamagen_vq,x*2-1,8,False))
            semantic=m.call('semantic_B0_original_and_base',lambda:self.runtime.semantic(torch.cat([x,base.float().add(1).mul(.5).clamp(0,1)])))
        parts=m.call('raw_feature_and_semantic_maps',lambda:self.feature_maps(x,base,f1,f2,semantic))
        maps=m.call('whitened_cosine_and_static_density',lambda:self.scorer.maps(parts))
        features=m.call('gain_inputs_fp16_rounding',lambda:torch.cat([parts['features'].half().float(),maps['white'].half().float()],1))
        from routing import standard
        gain=m.call('learned_gain_CNN',lambda:standard(self.engine.head(features)[:,1:2]))
        def utility():
            original=parts['features'].half().float()
            gf=torch.cat([original.mean((2,3)),original.std((2,3),correction=0)],1)
            state=self.scorer.utility[self.row['utility']]
            normal=((gf-state['mean'])/state['std']).clamp(-8,8)
            return (normal@state['beta']+state['bias']).clamp(-3,3)[:,None,None,None]
        u=m.call('image_utility_statistics_and_linear',utility)
        score=m.call('final_score_combination',lambda:gain+self.row['white']*maps['white']+self.row['cam']*maps['cam']-
            self.row['density']*maps['density']+self.row['utility_weight']*u)
        from routing import select_once
        mask,counts=m.call('threshold_count_and_stable_sort',lambda:select_once(score,self.tau,None if k<0 else k))
        def mix():
            weight=mask.reshape(-1,1,16,16).to(f1.dtype)
            return (1-weight)*f1+weight*f2
        mixed=m.call('feature_mixing',mix)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            rgb=m.call('final_rgb_decoder',lambda:self.model.llamagen_vq.decoder(mixed).float())
        return rgb,int(counts.item()),m,dict(score=score,mask=mask.flatten(1),parts=parts)

    def audit(self):
        rgb,k,_,parts=self.run(-1,0,'off')
        reference=self.engine(self.images[:1],self.tau)
        assert torch.equal(parts['mask'],reference['mask'].flatten(1)) and k==int(reference['K'].item())
        assert torch.equal(parts['score'],reference['score']) and torch.equal(rgb,reference['reconstruction'])
        return dict(split_path_matches_actual_deployment_exactly=True,**self.engine.audit())


class Generation:
    def __init__(self,args):
        sys.path.insert(0,str(GEN))
        self.c=source(GEN/'common.py','efficiency_generation_common')
        c=self.c
        from bert2d.paths import sha256,ONE_D
        from bert2d.eval import ImageBert,OfficialSamplingView
        from h20.assets import official_model
        ckpt=c.OUT/f'{args.kind}_seed0/latest.pt'
        receipt=json.loads(ckpt.with_suffix('.json').read_text());digest=sha256(ckpt)
        assert receipt['step']==20000 and receipt['sha256']==digest
        state=torch.load(ckpt,map_location='cpu',mmap=True,weights_only=True)
        assert state['step']==20000 and not state['config']['profile'] and state['config']['kind']==args.kind
        assert state['config']['router_identity']==c.router_identity(args.kind)
        for n,d in state['config']['implementation_sha256'].items():assert sha256(c.HERE/n)==d
        for n,d in state['config']['original_modules_sha256'].items():assert sha256(Path(c.module(n).__file__))==d
        c.module('halton_upstream').verify_pinned_upstream(c.UPSTREAM)
        self.model=c.module('continuous_halton').ContinuousHalton(c.UPSTREAM)
        self.model.load_state_dict(state['model'],strict=True);self.model=self.model.cuda().eval().requires_grad_(False);del state
        self.assets=c.module('assets_top_p').TopPFrozenAssets(c.ASSETS,c.TOP,'cuda',chunk=4)
        self.adapter,_=c.module('embedding_adapter').load_adapter(c.ADAPTER,'cuda',c.module('assets_top_p').TOP_P_SHA256,20000)
        self.scorer,_=c.load_router(args.kind,'cuda')
        one_d,audit=official_model(c.ASSETS)
        for n in ['embedding_2d','pos_embedding_2d','budget_embedding','output_2d']:delattr(one_d,n)
        self.one_d=one_d.cuda().eval().requires_grad_(False);self.view=OfficialSamplingView(self.one_d).eval()
        self.ImageBert=ImageBert;self.stage1=dict(ONE_D,guidance_scale=4.5,randomize_temperature=9.5)
        self.cfg=args.cfg;self.steps=args.stage2_steps
        self.helper=source(GEN/'eval_worker.py','efficiency_eval_helpers')
        vq=self.assets.shell.llamagen_vq;s=self.scorer
        self.metadata=dict(kind=args.kind,checkpoint=str(ckpt),checkpoint_sha256=digest,
            router=c.router_identity(args.kind),one_d_audit=audit,stage1=self.stage1,
            stage2=dict(steps=self.steps,cfg_w=self.cfg,temperature=1.),
            assets=dict(self.assets.audit),adapter_sha256=sha256(c.ADAPTER),
            parameters=parameter_groups(dict(stage1_generator=self.one_d,stage2_halton=self.model,
                spatial_expansion=self.assets.shell.latent_decoder,one_d_codebook=self.assets.native.quantize,
                local_CNN=[s.f_proj,s.pos_embed,s.trunk,s.score_head],
                budget_CNN_and_offset=[s.budget_f_proj,s.budget_pos_embed,s.budget_trunk,s.offset_head],
                context_adapter=self.adapter,local_codebook_projection=[vq.quantize,vq.post_quant_conv],shared_rgb_decoder=vq.decoder)),
            source_sha256={str(p):sha(p) for p in [GEN/'common.py',GEN/'eval_worker.py',Path(c.module('continuous_halton').__file__),
                Path(c.module('adapter_data').__file__),Path(c.module('embedding_adapter').__file__)]},
            precision='Stage1/2 and adapter BF16; adapter/base rendering expansion FP32; separate router expansion BF16; scorer FP32; codebook projection and RGB decoding FP32; TF32 disabled',
            endpoint_scope='K=0 retains both expansions, both scoring trunks and adapter; sampler follows original empty-selection path. Not an optimized stage1-only bypass.',
            additional_models_at_inference='None: no B0, DINO, LPIPS, or Inception',
            base_rgb_control_decode_included=False,
            inputs='Batch1 classes 0..repeats-1 modulo1000; seeds 4..(4+repeats-1); not the 50k evaluation batch plan')
        self.versions=[(prefix+'.'+name,value,value._version) for prefix,part in
            [('stage1',self.one_d),('stage2',self.model),('adapter',self.adapter),('router',self.scorer),
             ('native',self.assets.native),('codec',self.assets.shell)]
            for name,value in [*part.named_parameters(),*part.named_buffers()]]

    def router_output(self,base,m):
        s=self.scorer
        def local():
            h=s.trunk(s.f_proj(base)+s.pos_embed.to(base.dtype))
            score=s.score_head(h).float().flatten(1)
            return score-score.mean(1,keepdim=True)
        ranking=m.call('generation_local_ranking_CNN',local)
        h=m.call('generation_budget_feature_CNN',lambda:s.budget_trunk(s.budget_f_proj(base)+s.budget_pos_embed.to(base.dtype)))
        def utility():
            features=h.float();pooled=torch.cat([features.mean((2,3)),features.std((2,3),correction=0)],1)
            return 3*torch.tanh(s.offset_head(pooled).float().squeeze(1)/3)
        u=m.call('generation_image_utility_head',utility)
        score=m.call('generation_score_combination',lambda:ranking+s.utility_weight*u[:,None])
        return dict(local=ranking,utility=u,score=score)

    def run(self,k,index,mode):
        c=self.c;m=Meter(mode);seed=4+index
        labels=torch.full((1,),index%1000,device='cuda',dtype=torch.long)
        torch.manual_seed(seed)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            ids=m.call('stage1_generation',lambda:self.ImageBert.generate(self.view,condition=labels,**self.stage1))
        base=m.call('base_expansion_FP32_for_adapter_and_render',lambda:c.module('adapter_data').make_features(ids,self.assets.native.quantize,self.assets.shell.latent_decoder))
        scoring_base=m.call('separate_base_expansion_BF16_for_router',lambda:c.router_features(ids,self.assets.native.quantize,self.assets.shell.latent_decoder))
        with torch.autocast('cuda',enabled=False):output=self.router_output(scoring_base.float(),m)
        mask,counts=m.call('threshold_count_and_stable_sort',lambda:mask_from_score(output['score'],self.scorer.threshold,k))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            context=m.call('context_embedding_adapter',lambda:self.adapter(base))
        torch.manual_seed(seed+1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            ids2=m.call('stage2_generation',lambda:c.module('continuous_halton').sample_continuous_completion(
                self.model,context,mask,labels,c.UPSTREAM,steps=self.steps,cfg_w=self.cfg,temperature=1.))
        mixed=m.call('local_codebook_projection_and_mixing',lambda:self.helper.mixed_direct(self.assets,base,ids2,mask) if bool(mask.any()) else base)
        rgb=m.call('final_rgb_decoder',lambda:self.helper.render(self.assets,mixed))
        return rgb,int(counts.item()),m,dict(output=output,mask=mask,scoring_base=scoring_base)

    def audit(self):
        rgb,k,_,parts=self.run(-1,0,'off')
        reference=self.scorer.select(parts['scoring_base'].float())
        assert torch.equal(parts['mask'],reference['mask']) and k==int(reference['K'].item())
        assert all(torch.equal(parts['output'][key],reference[key]) for key in ['local','utility','score'])
        assert torch.isfinite(rgb).all()
        return dict(split_scorer_matches_exported_student_exactly=True,finite_final_RGB=True,
            no_GT_images_or_2d_inputs=True,actual_final_only_RGB_decode=True)

    def final_audit(self):
        self.assets.assert_frozen()
        assert all(value._version==version for _,value,version in self.versions)
        assert all(not p.requires_grad and p.grad is None for part in [self.one_d,self.model,self.adapter,self.scorer]
                   for p in part.parameters())
        return dict(all_loaded_weights_and_buffers_unchanged=True,all_inference_modules_frozen=True,
            no_optimizer_or_training_state_created=True,checkpoint_restore_scope='Strict stage2 model state at step20000, strict exported student state, hash-checked frozen adapter and official codec/stage1 assets. No optimizer restore: profiling performs no training.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['reconstruction','generation'],required=True)
    p.add_argument('--kind',choices=['short','long'],default='long')
    p.add_argument('--cfg',type=float,default=3.)
    p.add_argument('--stage2-steps',type=int,default=8)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--repeats',type=int,default=50)
    p.add_argument('--budgets',type=int,nargs='+',default=[-1,0,32,64,96,128,196,256])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert a.warmup>=0 and a.repeats>1 and all(-1<=k<=256 for k in a.budgets)
    assert len(set(a.budgets))==len(a.budgets) and a.cfg>0 and a.stage2_steps>0
    assert len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))==1 and os.environ.get('CUDA_VISIBLE_DEVICES')
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(status='loading',mode=a.mode,kind=a.kind,
        batch=1,warmup=a.warmup,repeats=a.repeats,hardware=torch.cuda.get_device_name(),
        physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,source_sha256=sha(__file__),
        CUDA=torch.version.cuda,cuDNN=torch.backends.cudnn.version(),
        scope='Images/labels already on GPU; excludes input disk/PIL, startup, ADM/Inception and all metrics. Includes CPU launch overhead.',
        wallclock='CUDA synchronized wall-clock; components and independent end-to-end are separate passes, alternating pass order.',
        TF32=False,cudnn_benchmark=False,
        counted_FLOPs='PyTorch FlopCounterMode; supported convolution/matmul/attention, 2 FLOPs per MAC. Operator-level count stored.',
        FLOPs_limitations='Normalization, nonlinearities, sorting, thresholding, sampling, transfer and other unsupported operations may not be counted. Zero counted FLOPs does not mean zero computational cost.',
        FLOPs_input_scope='One representative input index0 per budget; its K is reported separately from latency sample mean and actual full50k distributions.',
        comparison='Old TF32-enabled profiling cannot be directly compared as an identical protocol; reprofile baselines with these same precision settings when making speed claims.',
        frozen_fixed_K_diagnostic=True,no_new_training=True)
    write_json(a.output/'protocol.json',protocol)
    # Construct ordinary tensors so frozen-weight version counters remain
    # available to the existing deployment and final immutability audits.
    pipeline=Reconstruction(a) if a.mode=='reconstruction' else Generation(a)
    with torch.inference_mode():
        protocol.update(pipeline=pipeline.metadata,deployment_audit=pipeline.audit(),status='profiling')
        write_json(a.output/'protocol.json',protocol)
        results={}
        for k in a.budgets:
            arm='dynamic' if k<0 else 'fixed_'+str(k)
            for i in range(a.warmup):pipeline.run(k,i%a.repeats,'off')
            rows=[];torch.cuda.reset_peak_memory_stats()
            for i in range(a.repeats):
                parts=None
                if i%2==0:rgb,count,m,_=pipeline.run(k,i,'time');parts=m.values;del rgb
                torch.cuda.synchronize();start=time.perf_counter()
                rgb,actual,_,_=pipeline.run(k,i,'off');torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000
                assert torch.isfinite(rgb).all();del rgb
                if i%2:rgb,count,m,_=pipeline.run(k,i,'time');parts=m.values;del rgb
                assert count==actual
                rows.append(dict(input_index=i,K=actual,e2e_ms=elapsed,components_ms=parts))
            memory=dict(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
            flop_status='complete'
            try:
                rgb,fk,fm,_=pipeline.run(k,0,'flops');del rgb
                flop_report=dict(K=fk,components=fm.values,operator_counts=fm.operator_counts,total=sum(fm.values.values()))
            except Exception as error:
                flop_status='failed';flop_report=dict(error_type=type(error).__name__,message=str(error))
            result=dict(latency_status='complete',FLOPs_status=flop_status,mean_K=float(np.mean([r['K'] for r in rows])),
                e2e_median_ms=float(np.median([r['e2e_ms'] for r in rows])),
                e2e_mean_ms=float(np.mean([r['e2e_ms'] for r in rows])),
                e2e_std_ms=float(np.std([r['e2e_ms'] for r in rows],ddof=1)),
                e2e_p90_ms=float(np.quantile([r['e2e_ms'] for r in rows],.9)),
                component_median_ms={key:float(np.median([r['components_ms'][key] for r in rows])) for key in rows[0]['components_ms']},
                FLOPs=flop_report,memory=memory,rows=rows)
            results[arm]=result
            write_json(a.output/'result.json',dict(protocol=protocol,results=results,complete=False))
            print(json.dumps(dict(arm=arm,median_ms=result['e2e_median_ms'],mean_K=result['mean_K'],FLOPs_status=flop_status)),flush=True)
        complete=all(r['FLOPs_status']=='complete' for r in results.values())
        final_audit=pipeline.engine.audit() if a.mode=='reconstruction' else pipeline.final_audit()
        protocol['status']='complete' if complete else 'latency_complete_FLOPs_incomplete'
        write_json(a.output/'protocol.json',protocol)
        write_json(a.output/'result.json',dict(protocol=protocol,results=results,complete=complete,
            final_frozen_audit=final_audit,status='complete' if complete else 'latency_complete_FLOPs_incomplete'))
        if not complete:raise RuntimeError('Latency saved, but at least one FLOPs measurement failed; inspect result.json')


if __name__=='__main__':main()
