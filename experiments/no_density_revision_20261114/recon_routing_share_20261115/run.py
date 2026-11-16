"""Brief dedicated-card profiling; hold only the idle fixed106 dispatcher."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
REV=HERE.parent
sys.path.insert(0,str(REV))
from continue_short_U import process
from density_free_config import write_json
from gpu_dispatch import snapshot
QUEUE=986758
STATE=HERE/'execution.json'


def main():
    assert not STATE.exists(), 'No automatic retry'
    identity=process(QUEUE)
    assert identity and identity['state'] not in ('T','t','Z')
    assert str(REV/'fixed106_seed4_20261115/run.py') in identity['command']
    state=dict(status='preparing',pid=os.getpid(),started=time.time(),measurement='current Long no-density reconstruction',
        fixed106_queue_PID=QUEUE,fixed106_queue_identity=identity,queue_resumed=False,
        no_training=True,no_automatic_retry=True,max_owned_GPUs=4,foreign_GPU_processes_seen=[])
    held=False;active=None
    def update(**extra):
        state.update(extra,checked=time.time());write_json(STATE,state)
    def resume():
        nonlocal held
        if held:
            actual=process(QUEUE)
            assert actual and actual['start_ticks']==identity['start_ticks']
            os.kill(QUEUE,signal.SIGCONT);held=False;update(queue_resumed=True)
    try:
        update()
        os.kill(QUEUE,signal.SIGSTOP);held=True;time.sleep(.2)
        assert process(QUEUE)['state'] in ('T','t')
        record=json.loads((REV/'fixed106_seed4_20261115/execution.json').read_text())
        assert record['status']=='waiting_for_current_CFG_sweep' and not record['jobs']
        assert not Path(f'/proc/{QUEUE}/task/{QUEUE}/children').read_text().strip()
        # The last CFG evaluation has one remaining GPU worker. Do not touch it.
        old=json.loads((REV/'cfg_screen5k_20261115/execution.json').read_text())
        owned=set()
        for job in old['jobs']:
            f=Path(job['output'])/'status.json'
            if f.exists():
                for worker in json.loads(f.read_text()).get('jobs',[]):
                    p=process(worker['pid'])
                    if p and p['state']!='Z' and 'worker.py' in p['command']:
                        owned.add(worker['GPU']['uuid'])
        cards=[c for c in snapshot() if c['used_MiB']==0 and c['utilization']<=5]
        assert cards,'Need an exclusive empty card for a valid timing measurement'
        card=cards[0]
        assert len(owned|{card['uuid']})<=4
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=card['uuid'],OMP_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',USE_TF='0',PYTHONUNBUFFERED='1')
        for key in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE']:
            environment.pop(key,None)
        with (HERE/'measurement.log').open('x') as log:
            active=subprocess.Popen([sys.executable,str(HERE/'measure.py')],cwd=HERE,env=environment,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            update(status='measuring',worker_PID=active.pid,GPU=card,other_owned_GPU_UUIDs=sorted(owned))
            while active.poll() is None:
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                    '--format=csv,noheader,nounits'],text=True)
                extra=[]
                for line in raw.splitlines():
                    parts=[v.strip() for v in line.split(',')]
                    if len(parts)==3 and parts[1]==card['uuid'] and int(parts[0])!=active.pid:
                        extra.append(dict(pid=int(parts[0]),GPU=parts[1],used_MiB=parts[2]))
                if extra:state['foreign_GPU_processes_seen'].append(dict(time=time.time(),processes=extra))
                update(worker_process=process(active.pid))
                time.sleep(2)
            update(worker_exit_code=active.returncode)
            assert active.returncode==0, 'Measurement failed, preserve evidence; no automatic retry'
        result=json.loads((HERE/'result.json').read_text())
        assert result['status']=='complete' and result['n']==200
        assert not state['foreign_GPU_processes_seen'],'Concurrent foreign work detected; result must not be quoted as exclusive-card timing'
        update(status='complete',finished=time.time(),result=str(HERE/'result.json'),exclusive_GPU_verified=True)
    except BaseException as e:
        update(status='failed_preserved_no_retry',error=repr(e),traceback=traceback.format_exc())
        raise
    finally:
        if active and active.poll() is None:active.wait()
        resume()


if __name__=='__main__':main()
