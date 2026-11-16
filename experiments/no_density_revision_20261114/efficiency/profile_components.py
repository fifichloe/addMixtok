"""Current density-free model costs using the original synchronized timing harness.

No GPU launch occurs on import. Runtime classes bind only current checkpoints
and accepted scorers; original helper reuse is confined to measurement/raw-map
and two-CNN decomposition code, without historical model loading or density.
"""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from profiling_helpers import sha, write_json
from reconstruction_runtime import Reconstruction
from generation_runtime import Generation
from reused_fixed_runtime import ReusedFixedGeneration

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['reconstruction','generation'],required=True)
    p.add_argument('--kind',choices=['short','long'],default='long')
    p.add_argument('--cfg',type=float,default=3.)
    p.add_argument('--stage2-steps',type=int,default=8)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--repeats',type=int,default=50)
    p.add_argument('--budgets',type=int,nargs='+',default=[-1,0,32,64,96,128,196,256])
    p.add_argument('--policy',choices=['learned','random','uniform','dense'],default='learned')
    p.add_argument('--checkpoint-variant',choices=['main','fixed96'],default='main')
    p.add_argument('--policy-file',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert a.warmup>=0 and a.repeats>1 and all(-1<=k<=256 for k in a.budgets)
    if a.policy!='learned':assert a.mode=='generation' and a.budgets==[-1] and a.policy_file is not None
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
    pipeline=(Reconstruction(a) if a.mode=='reconstruction' else
              Generation(a) if a.policy=='learned' else ReusedFixedGeneration(a))
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
