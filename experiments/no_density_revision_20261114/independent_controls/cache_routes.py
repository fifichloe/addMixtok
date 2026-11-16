"""Control route caches; source/view K retained for random/uniform controls."""
import argparse
import json
import os
import time
from types import SimpleNamespace
import numpy as np
from common import *
from routing import control_identity, nearconstant_k, random_positions, uniform_order, select_control
from bert2d.paths import atomic_json, sha256

ROWS=2562334


def verified_base(kind):
    root=BASE_OUT/'routes'/kind
    config=json.loads((root/'config.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    assert summary['complete_coverage'] and summary['rows']==ROWS
    assert config['router_identity']==router_identity(kind)
    assert sha256(root/'config.json')==summary['config_sha256']
    assert sha256(root/'route_k.npy')==summary['route_k_sha256']
    assert sha256(root/'route_indices.npy')==summary['route_indices_sha256']
    module('cache_routes').source_codes(SOURCE)
    for filename,key in [('meta.json','source_meta_sha256'),('titok_codes.npy','source_titok_sha256'),
                         ('llamagen_codes.npy','source_gt2d_sha256'),('labels.npy','source_labels_sha256')]:
        assert sha256(SOURCE/filename)==config[key]
    return root,config,summary


def prepare(args):
    root,source,summary=verified_base(args.kind)
    assert abs(args.train_target_mean_k-summary['mean_k'])<1e-10
    args.output.mkdir(parents=True,exist_ok=False)
    config=dict(source)
    config.update(world_size=4,batch_size=args.batch,route_mode='control_'+args.policy,
        control_identity=control_identity(args.kind,args.policy,args.target_mean_k,args.route_seed,args.train_target_mean_k),
        base_route_root=str(root),base_route_config_sha256=sha256(root/'config.json'),
        base_route_indices_sha256=summary['route_indices_sha256'],
        base_route_k_sha256=summary['route_k_sha256'],implementation_sha256=sha256(Path(__file__)),
        routing_implementation_sha256=sha256(HERE/'routing.py'),fixed_k=None)
    for name,shape,dtype in [('route_indices.npy',(ROWS,256),np.uint16),('route_k.npy',(ROWS,),np.uint16)]:
        array=np.lib.format.open_memmap(args.output/name,mode='w+',dtype=dtype,shape=shape)
        array[:]=0;array.flush()
    for rank in range(4):
        array=np.lib.format.open_memmap(args.output/f'written_rank{rank}.npy',mode='w+',
            dtype=np.bool_,shape=(len(range(rank,ROWS,4)),))
        array[:]=False;array.flush()
    atomic_json(args.output/'config.json',config)


def extract(args):
    # Random/uniform are CPU-only; nearconstant ranking needs frozen BF16 grid.
    rank=int(os.environ.get('RANK',args.rank));world=int(os.environ.get('WORLD_SIZE',4))
    assert world==4 and 0<=rank<4
    config=json.loads((args.output/'config.json').read_text())
    assert config['control_identity']==control_identity(args.kind,args.policy,args.target_mean_k,args.route_seed,args.train_target_mean_k)
    assert config['implementation_sha256']==sha256(Path(__file__))
    assert config['routing_implementation_sha256']==sha256(HERE/'routing.py')
    root,_,_=verified_base(args.kind)
    base_k=np.load(root/'route_k.npy',mmap_mode='r')
    routes=np.load(args.output/'route_indices.npy',mmap_mode='r+')
    counts=np.load(args.output/'route_k.npy',mmap_mode='r+')
    written=np.load(args.output/f'written_rank{rank}.npy',mmap_mode='r+')
    if args.policy=='nearconstant':
        import torch
        torch.set_num_threads(4);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;device=torch.device('cuda',local)
        source=np.load(SOURCE/'titok_codes.npy',mmap_mode='r')
        frozen=module('assets_top_p').TopPFrozenAssets(ASSETS,TOP,'cpu',chunk=32)
        q=frozen.native.quantize.to(device);decoder=frozen.shell.latent_decoder.to(device)
        scorer,_=load_router(args.kind,device)
    all_ids=np.arange(rank,ROWS,4,dtype=np.int64);start=time.monotonic()
    for begin in range(0,len(all_ids),args.batch):
        end=min(begin+args.batch,len(all_ids));ids=all_ids[begin:end]
        if bool(written[begin:end].all()):continue
        if args.policy=='nearconstant':
            im,view=np.divmod(ids,2)
            with torch.inference_mode():
                z1=torch.as_tensor(np.array(source[im,view],dtype=np.int64),device=device)
                features=router_features(z1,q,decoder)
                with torch.autocast('cuda',enabled=False):
                    output=scorer(features.float())
                    # Budget counts only; no provisional learned selected set.
                    selection=dict(**output,K=(output['score']>=scorer.threshold).sum(1).clamp(48,144),
                        mask=torch.empty_like(output['score'],dtype=torch.bool))
                mask,ks=select_control(args.policy,selection,ids,args.train_target_mean_k,args.route_seed,0,ROWS,'train')
                ks=ks.cpu().numpy();mask=mask.cpu().numpy()
            positions=[np.flatnonzero(row) for row in mask]
        else:
            ks=np.asarray(base_k[ids],dtype=np.int64)
            positions=(random_positions(ids,ks,args.route_seed,0,'train') if args.policy=='random'
                       else [np.sort(np.asarray(uniform_order()[:int(k)])) for k in ks])
        for row,k,chosen in zip(ids,ks,positions):
            assert len(chosen)==k and len(np.unique(chosen))==k
            routes[row,:int(k)]=chosen;counts[row]=k
        routes.flush();counts.flush();written[begin:end]=True;written.flush()
        atomic_json(args.output/f'status_rank{rank}.json',dict(status='running',pid=os.getpid(),
            completed=int(written.sum()),total=len(all_ids),seconds=time.monotonic()-start))
    assert bool(written.all())
    if args.policy=='nearconstant':frozen.assert_frozen()
    atomic_json(args.output/f'status_rank{rank}.json',dict(status='complete',completed=len(all_ids),
        total=len(all_ids),seconds=time.monotonic()-start))


def finalize(args):
    module('cache_routes').finalize(SimpleNamespace(output=args.output,world_size=4))
    root,_,_=verified_base(args.kind)
    counts=np.load(args.output/'route_k.npy',mmap_mode='r')
    if args.policy!='nearconstant':
        assert np.array_equal(counts,np.load(root/'route_k.npy',mmap_mode='r'))
    else:assert int(np.asarray(counts,dtype=np.int64).sum())==round(args.train_target_mean_k*ROWS)
    summary=json.loads((args.output/'summary.json').read_text())
    summary.update(route_mode='control_'+args.policy,router_identity=router_identity(args.kind),
        matched_per_source_view_K=args.policy!='nearconstant',
        control_identity=control_identity(args.kind,args.policy,args.target_mean_k,args.route_seed,args.train_target_mean_k))
    atomic_json(args.output/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','extract','finalize'])
    p.add_argument('--kind',choices=KINDS,required=True)
    p.add_argument('--policy',choices=POLICIES,required=True)
    p.add_argument('--target-mean-k',type=float,required=True)
    p.add_argument('--train-target-mean-k',type=float,required=True)
    p.add_argument('--route-seed',type=int,default=20261114)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--rank',type=int,default=0)
    a=p.parse_args();globals()[a.mode](a)
