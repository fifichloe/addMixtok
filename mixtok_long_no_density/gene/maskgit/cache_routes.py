"""Build both full-train route caches in one frozen feature extraction pass."""
import argparse
import json
import os
import time
import numpy as np
import torch
from common import *
from bert2d.paths import atomic_json, sha256

ROWS = 2*1281167


def prepare(args):
    module('cache_routes').source_codes(SOURCE)
    source_hashes = dict(source_meta_sha256=sha256(SOURCE/'meta.json'),
        source_titok_sha256=sha256(SOURCE/'titok_codes.npy'),
        source_gt2d_sha256=sha256(SOURCE/'llamagen_codes.npy'),
        source_labels_sha256=sha256(SOURCE/'labels.npy'))
    for kind in KINDS:
        out = args.output/kind
        out.mkdir(parents=True, exist_ok=False)
        identity = router_identity(kind)
        config = dict(format=module('data').FORMAT, shape=[ROWS,256], dtype='uint16',
            flat_layout='flat_id=source_index*num_aug+augmentation', world_size=4,
            batch_size=args.batch, source_root=str(SOURCE), source_shape=[1281167,2,32],
            top_p_checkpoint_sha256=module('assets_top_p').TOP_P_SHA256,
            top_p_artifact_sha256=module('assets_top_p').verified_top_p_artifact(TOP),
            scorer_sha256=identity['weights_sha256'], router_identity=identity,
            route_mode='new_dynamic_'+kind, fixed_k=None, p=None, temperature=None,
            implementation_sha256=sha256(Path(__file__)), **source_hashes)
        for name,shape,dtype in [('route_indices.npy',(ROWS,256),np.uint16),
                                 ('route_k.npy',(ROWS,),np.uint16)]:
            a=np.lib.format.open_memmap(out/name,mode='w+',dtype=dtype,shape=shape)
            a.flush()
        for rank in range(4):
            a=np.lib.format.open_memmap(out/f'written_rank{rank}.npy',mode='w+',
                dtype=np.bool_,shape=(len(range(rank,ROWS,4)),))
            a[:]=False; a.flush()
        atomic_json(out/'config.json',config)


@torch.inference_mode()
def extract(args):
    rank,local,world=(int(os.environ[k]) for k in ['RANK','LOCAL_RANK','WORLD_SIZE'])
    assert world==4
    torch.set_num_threads(4);torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda',local)
    source=module('cache_routes').source_codes(SOURCE)
    frozen=module('assets_top_p').TopPFrozenAssets(ASSETS,TOP,'cpu',chunk=32)
    q=frozen.native.quantize.to(device);decoder=frozen.shell.latent_decoder.to(device)
    models={};arrays={}
    for kind in KINDS:
        out=args.output/kind
        cfg=json.loads((out/'config.json').read_text())
        assert cfg['router_identity']==router_identity(kind) and cfg['batch_size']==args.batch
        assert cfg['implementation_sha256']==sha256(Path(__file__))
        models[kind],_=load_router(kind,device)
        arrays[kind]=(np.load(out/'route_indices.npy',mmap_mode='r+'),
            np.load(out/'route_k.npy',mmap_mode='r+'),
            np.load(out/f'written_rank{rank}.npy',mmap_mode='r+'))
    ids_all=np.arange(rank,ROWS,4,dtype=np.int64)
    start=time.monotonic();probe_done=False
    for begin in range(0,len(ids_all),args.batch):
        end=min(begin+args.batch,len(ids_all))
        if all(bool(arrays[k][2][begin:end].all()) for k in KINDS):continue
        ids=ids_all[begin:end];im,view=np.divmod(ids,2)
        codes=torch.from_numpy(np.array(source[im,view],dtype=np.int64)).to(device)
        base=router_features(codes,q,decoder)
        for kind in KINDS:
            scorer=models[kind]
            with torch.autocast('cuda',enabled=False):output=scorer.select(base.float())
            mask=output['mask'];ks=output['K']
            if not probe_done:
                # Same real feature input: batch order/size cannot change the policy.
                single=scorer.select(base[:1].float())
                assert torch.equal(single['K'],ks[:1]) and torch.equal(single['mask'],mask[:1])
                assert not any(p.requires_grad or p.grad is not None for p in scorer.parameters())
            index=torch.arange(256,device=device)[None].expand(len(ids),-1)
            selected=index.masked_fill(~mask,256).sort(1).values.masked_fill(~(index<ks[:,None]),0)
            routes,counts,written=arrays[kind]
            routes[ids]=selected.cpu().numpy().astype(np.uint16)
            counts[ids]=ks.cpu().numpy().astype(np.uint16)
            # Rows must reach disk before marking them committed.
            routes.flush();counts.flush();written[begin:end]=True;written.flush()
            if begin//args.batch%10==0 or end==len(ids_all):
                row=dict(status='running',rank=rank,completed=int(written.sum()),
                    total=len(ids_all),seconds=time.monotonic()-start,
                    rows_per_second=end/max(time.monotonic()-start,1),
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                atomic_json(args.output/kind/f'status_rank{rank}.json',row)
                if rank==0:print(json.dumps(dict(kind=kind,**row)),flush=True)
        probe_done=True
    frozen.assert_frozen()
    for kind in KINDS:
        assert bool(arrays[kind][2].all())
        atomic_json(args.output/kind/f'status_rank{rank}.json',
            dict(status='complete',rank=rank,completed=len(ids_all),total=len(ids_all),
                 seconds=time.monotonic()-start,singleton_probe_passed=probe_done))


def finalize(args):
    from types import SimpleNamespace
    for kind in KINDS:
        out=args.output/kind
        module('cache_routes').finalize(SimpleNamespace(output=out,world_size=4))
        summary=json.loads((out/'summary.json').read_text())
        assert 48<=summary['min_k']<=summary['max_k']<=144
        summary.update(route_mode='new_dynamic_'+kind,router_identity=router_identity(kind),
            thresholds_recalibrated=False,legacy_scorer_used=False)
        atomic_json(out/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','extract','finalize'])
    p.add_argument('--output',type=Path,default=OUT/'routes')
    p.add_argument('--batch',type=int,default=64)
    a=p.parse_args();globals()[a.mode](a)
