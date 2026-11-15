"""Original evaluation with an explicit second-stage CFG argument."""
import argparse
import importlib.util
import json
import os
import time
import numpy as np
import torch
from pathlib import Path
import sys
EVAL_HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(EVAL_HERE.parent/'maskgit'))
from common import *
sys.path.insert(0,str(EVAL_HERE.parent))
from generation_control_policy import select_output as select_policy_output
from bert2d.paths import atomic_json, sha256, EVALUATOR, ONE_D
from bert2d.eval import ImageBert, OfficialSamplingView, to_uint8
from h20.assets import official_model


def render(assets,features):
    with torch.autocast('cuda',enabled=False):
        return ((assets.shell.llamagen_vq.decoder(features.float())+1)*.5).clamp(0,1)


def mixed_direct(assets,base,completed,selected):
    result=base.flatten(2).transpose(1,2).clone()
    batch,spatial=selected.nonzero(as_tuple=True);vq=assets.shell.llamagen_vq
    with torch.autocast('cuda',enabled=False):
        emb=vq.quantize.get_codebook_entry(completed[selected])
        projected=vq.post_quant_conv(emb.t()[None,:,None,:]).squeeze(0).squeeze(1).t()
    result[batch,spatial]=projected.to(result.dtype)
    return result.transpose(1,2).reshape_as(base)


@torch.inference_mode()
def main(args):
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda',0);out=args.output
    status_path=out/f'status_shard{args.shard}.json'
    assert not status_path.exists()
    atomic_json(status_path,dict(status='loading',pid=os.getpid(),completed=0))
    started=time.monotonic()
    identity=router_identity(args.kind)
    policy=dict(policy='dynamic',minimum_K=48,maximum_K=144,threshold=identity['threshold'])
    policy_sha=None
    if args.policy_file is not None:
        document=json.loads(args.policy_file.read_text())
        assert document['status']=='frozen_before_control_quality'
        assert document['router_identity']==identity and document['seed']==args.seed
        assert document['kind']==args.kind and document['cfg']==args.cfg
        assert document['stage2_steps']==args.stage2_steps
        policy=document['selection_policy'];policy_sha=sha256(args.policy_file)
    assert policy['policy'] in ('dynamic','fixed','U_zero')
    checkpoint=OUT/f'{args.kind}_seed0/latest.pt'
    receipt=json.loads(checkpoint.with_suffix('.json').read_text());digest=sha256(checkpoint)
    assert receipt['step']==20000 and receipt['sha256']==digest
    if args.policy_file is not None:assert document['checkpoint_sha256']==digest
    state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
    config=state['config']
    assert state['step']==20000 and config['kind']==args.kind and not config['profile']
    assert same_router_identity(identity, config['router_identity']) and config['adapter_sha256']==sha256(ADAPTER)
    for name,digest_expected in config['original_modules_sha256'].items():
        assert sha256(Path(module(name).__file__))==digest_expected, f'original module changed: {name}'
    for name,digest_expected in config['implementation_sha256'].items():
        actual = sha256(HERE/name)
        if name == 'common.py' and actual != digest_expected:
            original = json.loads((PACKAGE/'configs/original_training_sources.json').read_text())
            assert original['implementation_sha256'][name] == digest_expected
        else:
            assert actual == digest_expected, f'training implementation changed: {name}'
    module('halton_upstream').verify_pinned_upstream(UPSTREAM)
    model=module('continuous_halton').ContinuousHalton(UPSTREAM)
    model.load_state_dict(state['model'],strict=True)
    model=model.to(device).eval().requires_grad_(False);del state
    assets=module('assets_top_p').TopPFrozenAssets(ASSETS,TOP,device,chunk=4)
    adapter,_=module('embedding_adapter').load_adapter(ADAPTER,device,module('assets_top_p').TOP_P_SHA256,20000)
    scorer,_=load_router(args.kind,device)
    one_d,one_d_audit=official_model(ASSETS)
    before=sum(p.numel() for p in one_d.parameters());removed=0
    for name in ['embedding_2d','pos_embedding_2d','budget_embedding','output_2d']:
        m=getattr(one_d,name);removed+=sum(p.numel() for p in m.parameters());delattr(one_d,name)
    after=sum(p.numel() for p in one_d.parameters())
    assert (before,removed,after)==(202585088,25382656,177202432)
    one_d=one_d.to(device).eval().requires_grad_(False)
    view=OfficialSamplingView(one_d).eval()
    stage1=dict(ONE_D,guidance_scale=4.5,randomize_temperature=9.5)
    from bert2d.assets import FID,check
    for item in FID:check(ASSETS/item['local_path'],item)
    spec=importlib.util.spec_from_file_location('new_router_adm',EVALUATOR)
    adm=importlib.util.module_from_spec(spec);spec.loader.exec_module(adm)
    adm.INCEPTION_V3_PATH=str(ASSETS/'fid/classify_image_graph_def.pb')
    tf=adm.tf;tf.disable_eager_execution()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tfcfg=tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=2)
    tfcfg.gpu_options.allow_growth=True;tfcfg.gpu_options.visible_device_list='0'
    batches=module('parallel_eval_compact').worker_batches(args.n,4,args.shard,args.reference_total)
    ids_all=np.concatenate(batches)
    manifest=dict(n=args.n,step=20000,checkpoint_sha256=digest,batch=8,seed=args.seed,
        screen_batch_plan_total=args.reference_total,stage1=stage1,one_d_audit=one_d_audit,
        router=identity,stage2=dict(steps=args.stage2_steps,cfg_w=args.cfg,temperature=1.,order='official Halton filtered to selection'),
        context=dict(mode='original frozen continuous adapter',adapter_sha256=sha256(ADAPTER),
            adapter_features='FP32',router_features='BF16',scorer='FP32'),
        decoder_identities=dict(assets.audit),reference=str(ASSETS/'fid/VIRTUAL_imagenet256_labeled.npz'),
        source_sha256=sha256(Path(__file__)),no_gt_images_or_2d_used=True,
        physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],primary='full_refine',
        routing_policy=policy,routing_policy_file_sha256=policy_sha,
        selection_source_sha256=sha256(EVAL_HERE.parent/'generation_control_policy.py'),
        rng_protocol='stage1 seed+2*firstID; stage2 seed+2*firstID+1; original four logical shards/batch8',
        raw_route_scores_saved=True,one_final_selection_per_image=True)
    atomic_json(out/f'manifest_shard{args.shard}.json',manifest)
    dump_specs=dict(local_scores=((len(ids_all),256),np.float32),
        router_scores=((len(ids_all),256),np.float32),utility=((len(ids_all),),np.float32),
        route_K=((len(ids_all),),np.uint16),route_mask=((len(ids_all),256),np.uint8),
        z1d=((len(ids_all),32),np.uint16))
    dumps={name:np.lib.format.open_memmap(out/f'{name}_shard{args.shard}.npy',mode='w+',dtype=dtype,shape=shape)
        for name,(shape,dtype) in dump_specs.items()}
    np.save(out/f'ids_shard{args.shard}.npy',ids_all,allow_pickle=False)
    histogram=np.zeros(257,dtype=np.int64);offset=0;gpu_checked=False
    with tf.Session(config=tfcfg) as session:
        evaluator=adm.Evaluator(session,batch_size=8)
        pools={arm:np.lib.format.open_memmap(out/f'{arm}_shard{args.shard}.npy',mode='w+',
            dtype=np.float32,shape=(len(ids_all),2048)) for arm in ['base','full_refine']}
        for ids in batches:
            labels=torch.as_tensor(ids%1000,device=device)
            torch.manual_seed(args.seed+2*int(ids[0]))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z1=ImageBert.generate(view,condition=labels,**stage1)
            assert z1.shape==(len(ids),32) and bool(((z1>=0)&(z1<4096)).all())
            base=module('adapter_data').make_features(z1,assets.native.quantize,assets.shell.latent_decoder)
            router_base=router_features(z1,assets.native.quantize,assets.shell.latent_decoder)
            with torch.autocast('cuda',enabled=False):
                output=scorer(router_base.float())
                selection=dict(**output,**select_policy_output(output,policy))
            assert torch.isfinite(base).all() and torch.isfinite(router_base).all()
            assert all(torch.isfinite(selection[key]).all() for key in ['local','utility','score'])
            selected=selection['mask'];counts=selection['K']
            assert torch.equal(selected.sum(1),counts) and bool(((counts>=48)&(counts<=144)).all())
            if offset==0:
                # Extra independent-input probes are evaluation audits only;
                # deployment takes one forward and directly one final set.
                single=select_policy_output(scorer(router_base[:1].float()),policy)
                reverse=select_policy_output(scorer(router_base.flip(0).float()),policy)
                assert torch.equal(single['mask'],selected[:1]) and torch.equal(single['K'],counts[:1])
                assert torch.equal(reverse['mask'].flip(0),selected) and torch.equal(reverse['K'].flip(0),counts)
            sl=slice(offset,offset+len(ids))
            dumps['local_scores'][sl]=output['local'].cpu().numpy()
            dumps['router_scores'][sl]=output['score'].cpu().numpy()
            dumps['utility'][sl]=output['utility'].cpu().numpy()
            dumps['route_K'][sl]=counts.cpu().numpy().astype(np.uint16)
            dumps['route_mask'][sl]=selected.cpu().numpy().astype(np.uint8)
            dumps['z1d'][sl]=z1.cpu().numpy().astype(np.uint16)
            histogram+=torch.bincount(counts,minlength=257).cpu().numpy()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                context=torch.cat([adapter(part) for part in base.split(4)])
            assert torch.isfinite(context).all()
            anchors=context[~selected].clone()
            base_rgb=torch.cat([render(assets,part) for part in base.split(4)])
            torch.manual_seed(args.seed+2*int(ids[0])+1)
            trace=[]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                completed=module('continuous_halton').sample_continuous_completion(
                    model,context,selected,labels,UPSTREAM,steps=args.stage2_steps,cfg_w=args.cfg,temperature=1.,trace=trace)
            assert torch.equal(context[~selected],anchors)
            mixed=mixed_direct(assets,base,completed,selected)
            assert torch.equal(mixed.flatten(2).transpose(1,2)[~selected],base.flatten(2).transpose(1,2)[~selected])
            refined_rgb=torch.cat([render(assets,part) for part in mixed.split(4)])
            assert torch.isfinite(base_rgb).all() and torch.isfinite(refined_rgb).all()
            pixels=dict(base=to_uint8(base_rgb),full_refine=to_uint8(refined_rgb))
            if offset==0:
                np.savez_compressed(out/f'first_shard{args.shard}.npz',ids=ids,
                    completed=completed.cpu().numpy(),selected=selected.cpu().numpy(),**pixels)
                atomic_json(out/f'trace_shard{args.shard}.json',trace)
            for arm,images in pixels.items():
                kwargs={}
                if not gpu_checked:
                    metadata=tf.RunMetadata()
                    kwargs=dict(options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=metadata)
                features=session.run(evaluator.pool_features,{evaluator.image_input:images.astype(np.float32)},**kwargs).reshape(len(ids),2048)
                if not gpu_checked:
                    devices=[d.device for d in metadata.step_stats.dev_stats if 'GPU' in d.device.upper()
                             and any('conv' in n.node_name.lower() for n in d.node_stats)]
                    assert devices,'ADM convolutions must actually run on GPU'
                    atomic_json(out/f'adm_gpu_shard{args.shard}.json',dict(devices=devices));gpu_checked=True
                assert np.isfinite(features).all();pools[arm][offset:offset+len(ids)]=features
            offset+=len(ids)
            atomic_json(status_path,dict(status='running',completed=offset,total=len(ids_all),
                sum_k=int(histogram@np.arange(257)),seconds=time.monotonic()-started,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        for pool in pools.values():pool.flush()
    assert offset==len(ids_all)
    for value in dumps.values():value.flush()
    counts=dict(n=offset,sum_k=int(histogram@np.arange(257)),
        min_k=int(np.flatnonzero(histogram)[0]),max_k=int(np.flatnonzero(histogram)[-1]),histogram=histogram.tolist())
    result=dict(status='complete',shard=args.shard,n=offset,ids=ids_all.tolist(),step=20000,
        checkpoint_sha256=digest,anchors_unchanged=True,route_counts=counts,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        hashes={arm:sha256(out/f'{arm}_shard{args.shard}.npy') for arm in pools},
        dump_hashes={name:sha256(out/f'{name}_shard{args.shard}.npy') for name in dumps},
        ids_sha256=sha256(out/f'ids_shard{args.shard}.npy'),
        routing_policy=policy,routing_policy_file_sha256=policy_sha,
        same_base_singleton_reverse_masks_K_exact=True,
        one_forward_one_final_selection=True,seconds=time.monotonic()-started)
    atomic_json(out/f'shard{args.shard}.json',result)
    atomic_json(status_path,dict(status='complete',completed=offset,total=offset,seconds=result['seconds']))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=KINDS,required=True)
    p.add_argument('--cfg',type=float,choices=[2.5,2.75,3.0,3.25,3.5],required=True)
    p.add_argument('--stage2-steps',type=int,choices=[1,2,4,6,8,16,32,48,64],default=8)
    p.add_argument('--policy-file',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[32,50000],required=True)
    p.add_argument('--reference-total',type=int,default=None)
    p.add_argument('--seed',type=int,default=4)
    p.add_argument('--shard',type=int,choices=range(4),required=True)
    main(p.parse_args())
