"""Construct and verify static routing using the existing public cache format."""
import argparse
import json
from types import SimpleNamespace
import numpy as np
from common import *
from routing import fixed_k, positions, description
from bert2d.paths import atomic_json, sha256

def run(args):
    k=fixed_k(args.policy)
    public=module('cache_routes')
    settings=SimpleNamespace(source_root=SOURCE,output=args.output,assets_root=ASSETS,
        top_p_ckpt=TOP,scorer_ckpt=SCORER,p=.75,temperature=2.,world_size=6,batch=8192)
    public.prepare(settings)
    config=json.loads((args.output/'config.json').read_text())
    config.update(route_mode=args.policy,fixed_k=k,p=None,temperature=None,
        scorer_used_for_selection=False,route_seed=None,source_commit=COMMIT,
        selection=description(args.policy),selected_positions=positions(args.policy),
        route_persistence='identical fixed positions for every source/view and epoch',
        implementation_sha256=sha256(Path(__file__)),routing_sha256=sha256(HERE/'routing.py'))
    atomic_json(args.output/'config.json',config)
    routes=np.load(args.output/'route_indices.npy',mmap_mode='r+')
    ks=np.load(args.output/'route_k.npy',mmap_mode='r+')
    selected=np.asarray(positions(args.policy),dtype=np.uint16)
    for begin in range(0,len(ks),8192):
        end=min(begin+8192,len(ks))
        routes[begin:end,:k]=selected
        routes[begin:end,k:]=0
        ks[begin:end]=k
    routes.flush();ks.flush()
    # Independent full-row readback before declaring coverage complete.
    for begin in range(0,len(ks),8192):
        assert np.all(routes[begin:begin+8192,:k]==selected)
        assert np.all(ks[begin:begin+8192]==k)
    for rank in range(6):
        written=np.load(args.output/f'written_rank{rank}.npy',mmap_mode='r+')
        written[:]=True;written.flush()
    public.finalize(settings)
    summary=json.loads((args.output/'summary.json').read_text())
    assert summary['min_k']==summary['max_k']==summary['mean_k']==k
    summary.update(route_mode=args.policy,fixed_k=k,static_positions_verified=True)
    atomic_json(args.output/'summary.json',summary)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',choices=['uniform','dense'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
