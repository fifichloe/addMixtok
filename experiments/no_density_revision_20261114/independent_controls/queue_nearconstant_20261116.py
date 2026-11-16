"""Finish nearconstant routes, then start its independent 20k-step training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROUTES = Path('/tmp/no_density_control_routes_20261116/nearconstant')
OUT = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/no_density_revision_20261114/independent_controls_20261116/nearconstant')
LOG = HERE/'train_nearconstant_20261116.log'
STATE = HERE/'queue_nearconstant_20261116.json'
COMMON = ['--kind','long','--policy','nearconstant',
          '--target-mean-k','106.38386','--train-target-mean-k','93.64123880805546',
          '--route-seed','20261114','--output',str(ROUTES),'--batch','64']


def update(**kwargs):
    STATE.write_text(json.dumps(dict(time=time.time(),**kwargs),indent=2)+'\n')


def free_cards():
    result = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                      '--format=csv,noheader,nounits'],text=True)
    cards=[]
    for line in result.splitlines():
        index,free=[int(part.strip()) for part in line.split(',')]
        if free>=14*1024:
            cards.append((free,index))
    return [index for _,index in sorted(cards,reverse=True)[:4]]


def main():
    if OUT.exists():
        raise RuntimeError(f'Fresh nearconstant training output already exists: {OUT}')
    while True:
        rows=[]
        for rank in range(4):
            path=ROUTES/f'status_rank{rank}.json'
            rows.append(json.loads(path.read_text()) if path.exists() else None)
        if all(row and row.get('status')=='complete' and row.get('completed')==row.get('total') for row in rows):
            break
        update(status='waiting_for_nearconstant_route_extraction',ranks=rows)
        time.sleep(30)
    final = [sys.executable,str(HERE/'cache_routes.py'),'finalize',*COMMON]
    with (HERE/'nearconstant_finalize_20261116.log').open('w') as log:
        subprocess.run(final,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
    summary=json.loads((ROUTES/'summary.json').read_text())
    assert summary['complete_coverage'] and summary['rows']==2562334
    assert summary['route_mode']=='control_nearconstant'
    while True:
        selected=free_cards()
        if len(selected)==4:
            break
        update(status='waiting_for_four_GPUs_with_14GiB_free',available=selected)
        time.sleep(30)
    chosen=','.join(map(str,selected))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=chosen,OMP_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',
             WANDB_MODE='offline',PYTORCH_ALLOC_CONF='expandable_segments:True')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
             str(HERE/'train.py'),'--kind','long','--policy','nearconstant',
             '--target-mean-k','106.38386','--train-target-mean-k','93.64123880805546',
             '--route-seed','20261114','--routes',str(ROUTES),'--output',str(OUT),
             '--seed','0','--eval-cfg-stage2','3']
    update(status='training',GPUs=selected,route_summary=str(ROUTES/'summary.json'))
    with LOG.open('w') as log:
        result=subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
    update(status='complete' if result.returncode==0 else 'failed',
           GPUs=selected,exit_code=result.returncode,output=str(OUT))
    if result.returncode:
        raise RuntimeError(f'Nearconstant training failed with code {result.returncode}; see {LOG}')


if __name__=='__main__':
    main()
