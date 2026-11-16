"""Four-GPU CFG evaluation with original batches, seeds, budgets and ADM."""
import argparse
import importlib.util
import json
import os
import subprocess
import time
from common import *
from routing import control_identity
from bert2d.paths import atomic_json, sha256, EVALUATOR


def run(args):
    assert args.output is not None
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();reference_total=None
    # Fixed four cards: no fifth GPU while waiting or extracting features.
    state=dict(status='waiting_for_gpu',last_checked=time.time())
    while True:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        used={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()}
        if all(used[int(g)]<1536 for g in GPUS.split(',')):break
        atomic_json(args.output/'status.json',dict(status='waiting_for_gpu',gpu_usage=used));time.sleep(20)
    checkpoint=run_root(args.kind,args.policy)/'latest.pt';digest=sha256(checkpoint)
    completion=json.loads((checkpoint.parent/'training_complete.json').read_text())
    assert completion['step']==20000 and completion['model_optimizer_restore_passed']
    assert completion['all_frozen_components_unchanged'] and completion['checkpoint_sha256']==digest
    control=control_identity(args.kind,args.policy,args.target_mean_k,args.route_seed,args.train_target_mean_k)
    atomic_json(args.output/'parallel_config.json',dict(kind=args.kind,policy=args.policy,control_identity=control,n=args.n,seed=args.seed,gpus=GPUS,
        checkpoint=str(checkpoint),checkpoint_sha256=digest,reference_total=reference_total,cfg_stage2=args.cfg,
        router=router_identity(args.kind),source_sha256=sha256(Path(__file__))))
    workers=[]
    for shard,gpu in enumerate(GPUS.split(',')):
        env=environment();env['CUDA_VISIBLE_DEVICES']=gpu
        for name in ['RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE']:
            env.pop(name,None)
        command=[PYTHON,str(HERE/'eval_worker.py'),'--kind',args.kind,'--output',str(args.output),
            '--n',str(args.n),'--seed',str(args.seed),'--shard',str(shard),'--cfg',str(args.cfg),'--policy',args.policy,'--target-mean-k',str(args.target_mean_k),
            '--route-seed',str(args.route_seed),'--train-target-mean-k',str(args.train_target_mean_k)]
        if reference_total:command+=['--reference-total',str(reference_total)]
        log=(args.output/f'worker_shard{shard}.log').open('w')
        proc=subprocess.Popen(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append((proc,log))
    atomic_json(args.output/'processes.json',dict(coordinator_pid=os.getpid(),pids=[p.pid for p,_ in workers]))
    failure_seen=False
    while True:
        codes=[p.poll() for p,_ in workers]
        statuses=[json.loads(p.read_text()) if p.exists() else {} for p in
            [args.output/f'status_shard{i}.json' for i in range(4)]]
        failure_seen=failure_seen or any(c is not None and c!=0 for c in codes)
        atomic_json(args.output/'status.json',dict(status='worker_failed' if failure_seen else 'running',
            codes=codes,completed=sum(s.get('completed',0) for s in statuses),total=args.n,
            seconds=time.monotonic()-started,worker_pids=[p.pid for p,_ in workers]))
        if all(c is not None for c in codes):break
        time.sleep(20)
    for _,log in workers:log.close()
    if failure_seen:raise RuntimeError(f'generation worker failed: {codes}; logs retained; no retry')
    manifests,receipts=module('parallel_eval_compact').verify_shards(args.output,args.n,digest,4,reference_total)
    identity=router_identity(args.kind)
    assert all(m['router']==dict(teacher=identity,control=control) for m in manifests)
    import numpy as np
    hist=np.sum([np.array(r['route_counts']['histogram'],dtype=np.int64) for r in receipts],axis=0)
    assert hist.sum()==args.n and not hist[:48].any() and not hist[145:].any()
    learned_hist=np.zeros(257,dtype=np.int64)
    for shard,receipt in enumerate(receipts):
        path=args.output/f'route_K_shard{shard}.npy'
        assert sha256(path)==receipt['per_image_K_sha256']
        pairs=np.load(path)
        assert pairs.shape==(receipt['n'],2) and pairs.dtype==np.uint16
        learned_hist+=np.bincount(pairs[:,0].astype(np.int64),minlength=257)
        assert np.array_equal(np.bincount(pairs[:,1].astype(np.int64),minlength=257),
                              receipt['route_counts']['histogram'])
        if args.policy!='nearconstant':assert np.array_equal(pairs[:,0],pairs[:,1])
    if args.seed==4 and args.n==50000:
        reference=json.loads((BASE_OUT/f'{args.kind}_eval_seed4_50000/summary.json').read_text())
        assert np.array_equal(learned_hist,reference['tokens']['histogram'])
    tokens=dict(mean_1d=32,mean_generated_2d=float(hist@np.arange(257)/args.n),
        mean_generated_total=32+float(hist@np.arange(257)/args.n),histogram=hist.tolist(),
        min_K=int(np.flatnonzero(hist)[0]),max_K=int(np.flatnonzero(hist)[-1]))
    metrics={}
    if args.n==50000:
        # Evaluation-only Inception; original ADM reference and covariance estimator.
        os.environ['CUDA_VISIBLE_DEVICES']=''
        spec=importlib.util.spec_from_file_location('new_router_adm_stats',EVALUATOR)
        adm=importlib.util.module_from_spec(spec);spec.loader.exec_module(adm)
        with np.load(ASSETS/'fid/VIRTUAL_imagenet256_labeled.npz') as reference:
            ref=adm.FIDStatistics(reference['mu'],reference['sigma'])
            for arm in ['base','full_refine']:
                features=np.concatenate([np.load(args.output/f'{arm}_shard{i}.npy') for i in range(4)])
                assert features.shape==(50000,2048) and np.isfinite(features).all()
                stat=adm.FIDStatistics(features.mean(0),np.cov(features,rowvar=False))
                value=float(stat.frechet_distance(ref))
                assert np.isfinite(value) and value>=0, f'invalid ADM gFID for {arm}: {value}'
                metrics[arm]=dict(gFID=value)
    result=dict(status='complete',kind=args.kind,policy=args.policy,control_identity=control,n=args.n,seed=args.seed,step=20000,
        metrics=metrics,tokens=tokens,router=dict(teacher=identity,control=control),checkpoint_sha256=digest,
        seconds=time.monotonic()-started,anchors_unchanged=True,
        generation_rounds='8+8',guidance=f'4.5/{args.cfg:g}',cfg_stage2=args.cfg)
    result['per_image_learned_K_matches_actual']=args.policy!='nearconstant'
    result['learned_K_histogram']=learned_hist.tolist()
    if args.policy=='nearconstant' and args.n==50000:
        assert int(hist@np.arange(257))==round(args.target_mean_k*50000)
    atomic_json(args.output/'summary.json',result)
    atomic_json(args.output/'status.json',dict(status='complete',completed=args.n,total=args.n))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=KINDS,required=True)
    p.add_argument('--policy',choices=POLICIES,required=True)
    p.add_argument('--target-mean-k',type=float,required=True)
    p.add_argument('--train-target-mean-k',type=float,required=True)
    p.add_argument('--route-seed',type=int,default=20261114)
    p.add_argument('--cfg',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[50000],required=True)
    p.add_argument('--seed',type=int,default=4)
    run(p.parse_args())
