"""Sequential controls; launch only after parent scheduler grants four GPUs.

No retries, unrelated process termination, source edits, or remote writes.
"""
import argparse
import json
import os
import subprocess
import time
from common import *
from bert2d.paths import atomic_json, sha256


def admission(state):
    while True:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
            '--format=csv,noheader,nounits'],text=True)
        memory={int(row.split(',')[0]):int(row.split(',')[1]) for row in raw.splitlines()}
        if all(memory[int(gpu)]<1536 for gpu in GPUS.split(',')):return
        state.update(status='waiting_for_gpu',gpu_memory_mib=memory,last_checked=time.time())
        atomic_json(HERE/'current.json',state);time.sleep(20)


def run(args):
    assert not (HERE/'current.json').exists(),'Existing queue state: inspect before creating another controller'
    assert args.cfg>0 and args.cfg<float('inf')
    baseline=json.loads((BASE_OUT/f'{args.kind}_eval_seed4_50000/summary.json').read_text())
    source_summary=json.loads((BASE_OUT/'routes'/args.kind/'summary.json').read_text())
    assert baseline['status']=='complete' and baseline['n']==50000 and baseline['seed']==4
    assert abs(baseline['tokens']['mean_generated_2d']-args.target_mean_k)<1e-9
    assert source_summary['complete_coverage']
    assert abs(source_summary['mean_k']-args.train_target_mean_k)<1e-10
    assert len(args.policies)==len(set(args.policies))
    state=dict(status='starting',pid=os.getpid(),kind=args.kind,cfg=args.cfg,
        target_mean_K=args.target_mean_k,train_target_mean_K=args.train_target_mean_k,
        main_selection_file=str(args.main_selection) if args.main_selection else None,
        main_selection_sha256=sha256(args.main_selection) if args.main_selection else None,
        policies=args.policies,stages=[],started=time.time(),gpus=GPUS)
    atomic_json(HERE/'current.json',state)
    def stage(name,command,gpu=False):
        if gpu:admission(state)
        log_path=HERE/f'{name}.log'
        with log_path.open('x') as log:
            child=subprocess.Popen(command,cwd=HERE,env=environment(),stdout=log,stderr=subprocess.STDOUT)
            record=dict(name=name,command=command,pid=child.pid,log=str(log_path),started=time.time())
            state['stages'].append(record);state.update(status='running',stage=name)
            while child.poll() is None:
                state['last_checked']=time.time();atomic_json(HERE/'current.json',state);time.sleep(20)
            record.update(returncode=child.returncode,finished=time.time())
            atomic_json(HERE/'current.json',state)
            if child.returncode!=0:raise RuntimeError(f'{name} exited {child.returncode}; log retained; no retry')
    results={}
    for policy in args.policies:
        root=OUT/'routes'/f'{args.kind}_{policy}'
        shared=['--kind',args.kind,'--policy',policy,'--target-mean-k',str(args.target_mean_k),
                '--train-target-mean-k',str(args.train_target_mean_k),'--route-seed',str(args.route_seed)]
        cache=[*shared,'--output',str(root),'--batch','64']
        stage(f'{policy}_prepare',[PYTHON,str(HERE/'cache_routes.py'),'prepare',*cache])
        if policy=='nearconstant':
            stage(f'{policy}_extract',distributed(HERE/'cache_routes.py','extract',*cache),gpu=True)
        else:
            for rank in range(4):
                stage(f'{policy}_extract_rank{rank}',[PYTHON,str(HERE/'cache_routes.py'),'extract',*cache,'--rank',str(rank)])
        stage(f'{policy}_finalize',[PYTHON,str(HERE/'cache_routes.py'),'finalize',*cache])
        stage(f'{policy}_train',distributed(HERE/'train.py',*shared,'--routes',root,
            '--output',run_root(args.kind,policy),'--seed','0'),gpu=True)
        completed=json.loads((run_root(args.kind,policy)/'training_complete.json').read_text())
        assert completed['step']==20000 and completed['model_optimizer_restore_passed']
        assert completed['all_frozen_components_unchanged']
        assert completed['checkpoint_sha256']==sha256(run_root(args.kind,policy)/'latest.pt')
        out=OUT/f'{args.kind}_{policy}_eval_seed4_50000'
        stage(f'{policy}_eval_50000',[PYTHON,str(HERE/'evaluate.py'),*shared,'--cfg',str(args.cfg),
            '--output',str(out),'--n','50000','--seed','4'],gpu=True)
        results[policy]=json.loads((OUT/f'{args.kind}_{policy}_eval_seed4_50000/summary.json').read_text())
        atomic_json(HERE/'results.json',results)
    state.update(status='complete',finished=time.time(),results=str(HERE/'results.json'))
    atomic_json(HERE/'current.json',state)
    atomic_json(OUT/'controls_complete.json',dict(status='complete',kind=args.kind,
        cfg_stage2=args.cfg,train_target_mean_K=args.train_target_mean_k,
        generation_target_mean_K=args.target_mean_k,policies=args.policies,
        results=results,controller_pid=os.getpid(),finished=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--main-selection',type=Path)
    p.add_argument('--kind',choices=KINDS)
    p.add_argument('--cfg',type=float)
    p.add_argument('--target-mean-k',type=float)
    p.add_argument('--train-target-mean-k',type=float)
    p.add_argument('--route-seed',type=int,default=20261114)
    p.add_argument('--policies',choices=POLICIES,nargs='+',default=list(POLICIES))
    args=p.parse_args()
    if args.main_selection:
        selected=json.loads(args.main_selection.read_text())
        # Minimal explicit scheduler contract; no automatic gFID-based selection.
        for attr,key in [('kind','kind'),('cfg','cfg_stage2'),
                         ('target_mean_k','mean_generated_2d'),('train_target_mean_k','train_mean_K')]:
            value=selected[key]
            assert getattr(args,attr) in (None,value),f'CLI/main-selection mismatch: {attr}'
            setattr(args,attr,value)
    assert args.kind in KINDS and all(getattr(args,k) is not None for k in
        ['cfg','target_mean_k','train_target_mean_k']),'Supply --main-selection or all four explicit settings'
    try:run(args)
    except BaseException as exc:
        state=json.loads((HERE/'current.json').read_text()) if (HERE/'current.json').exists() else {}
        state.update(status='failed',error=repr(exc),failed_at=time.time())
        atomic_json(HERE/'failure.json',state)
        # Keep active current.json untouched if assertion found a previous queue.
        if state.get('pid')==os.getpid():atomic_json(HERE/'current.json',state)
        raise
