"""Four offline new-label workers followed by one small fresh-head fit."""
import os,sys,subprocess,json,time
from routing import HERE,ROOT,write_json


def main():
    devices=[0,2,3,6];processes=[]
    status=json.loads((HERE/'STATUS.json').read_text())
    status.update(status='independent_dense_gain_labels_running',generation='paused',devices=devices)
    write_json(HERE/'STATUS.json',status)
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={int(row.split(',')[0]):int(row.split(',')[1]) for row in raw.strip().splitlines()}
    assert all(memory[device]<1024 for device in devices),memory
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',TMPDIR='/tmp/mxfr-1112')
    for shard,device in enumerate(devices):
        command=[sys.executable,str(HERE/'cache_dense_gain.py'),'--shard',str(shard),'--output',str(HERE/f'dense_gain_{shard}')]
        log=(HERE/f'dense_gain_{shard}.log').open('w')
        child=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(device)),stdout=log,stderr=subprocess.STDOUT)
        processes.append((child,log));print(json.dumps(dict(stage='labels',shard=shard,pid=child.pid,device=device)),flush=True)
    while any(child.poll() is None for child,_ in processes):
        if any(child.poll() not in [None,0] for child,_ in processes):
            status.update(status='dense_gain_worker_failed');write_json(HERE/'STATUS.json',status)
            raise RuntimeError('Worker failed; no retry or new workers started.')
        time.sleep(2)
    for child,log in processes:assert child.returncode==0;log.close()
    status.update(status='fresh_dense_gain_head_training');write_json(HERE/'STATUS.json',status)
    command=[sys.executable,str(HERE/'fit_dense_gain.py'),'--cache',*[str(HERE/f'dense_gain_{i}') for i in range(4)],
        '--output',str(HERE/'dense_gain_head'),'--epochs','12']
    with (HERE/'fit_dense_gain.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT)
    status.update(status='dense_gain_head_ready_for_actual_quality' if result.returncode==0 else 'dense_gain_fit_failed')
    write_json(HERE/'STATUS.json',status)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
