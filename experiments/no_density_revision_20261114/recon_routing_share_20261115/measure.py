"""Fresh batch1 reconstruction timing: CUDA-event attribution and clean wall time."""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
REV=HERE.parent
sys.path.insert(0,str(REV/'efficiency'))
import reconstruction_runtime as rr
from profiling_helpers import sha,write_json

N=200
WARMUP=20
PREP=['base_rgb_decoder','semantic_B0_original_and_base','raw_feature_and_semantic_maps']
GRID=['whitened_feature_cosine','gain_inputs_fp16_rounding','learned_gain_CNN','local_score_combination']
BUDGET=['image_utility_statistics_and_linear','final_image_utility_offset','threshold_count_and_stable_sort']
POOL=[]


class EventMeter:
    def __init__(self,mode):
        self.mode=mode;self.values={};self.records=[]
    def call(self,name,fn):
        if self.mode=='events':
            start,end=POOL[len(self.records)]
            start.record();out=fn();end.record()
            self.records.append((name,start,end))
            return out
        return fn()
    def finish(self):
        self.values={name:float(a.elapsed_time(b)) for name,a,b in self.records}
        return self.values


def stats(values):
    a=np.asarray(values,dtype='float64')
    assert np.isfinite(a).all()
    return dict(mean=float(a.mean()),median=float(np.median(a)),std=float(a.std(ddof=1)),
        p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)))


def main():
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    assert not (HERE/'result.json').exists()
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.manual_seed(20261115)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    source_paths=rr.VAL_PATHS
    all_paths=source_paths.read_text().splitlines()
    assert len(all_paths)==len(set(all_paths))==50000
    ids=np.arange(N)*250
    manifest=HERE/'input_manifest.txt'
    manifest.write_text('\n'.join(all_paths[int(i)] for i in ids)+'\n')
    rr.VAL_PATHS=manifest
    rr.Meter=EventMeter
    args=SimpleNamespace(kind='long',repeats=N)
    pipeline=rr.Reconstruction(args)
    audit=pipeline.audit()
    # Create reusable events outside measured regions and initialize their CUDA resources.
    for _ in range(32):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();b.record();POOL.append((a,b))
    total_start=torch.cuda.Event(enable_timing=True);total_end=torch.cuda.Event(enable_timing=True)
    total_start.record();total_end.record();torch.cuda.synchronize()
    protocol=dict(variant='current_Long_no_density',batch=1,images=N,warmup=WARMUP,
        physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],GPU=torch.cuda.get_device_name(),torch=torch.__version__,
        CUDA=torch.version.cuda,TF32=False,cudnn_benchmark=False,
        image_indices=ids.tolist(),input_manifest_sha256=sha(manifest),original_manifest_sha256=sha(source_paths),
        selection='one image every250 in sorted50000 manifest, frozen before timing',
        precision='BF16 codec/B0, FP32 gain/scoring with original FP16 input rounding',
        scope='GPU-resident image to reconstructed RGB; excludes startup/disk/PIL, metrics and transfers to CPU',
        measurement='CUDA events per existing component, one final sync; separate uninstrumented synchronized wall-clock pass, alternating order',
        ratio_denominator='full same-pass CUDA event timeline; independent clean wall-clock reported separately',
        per_component_synchronization=False,reruns_or_training=False,
        groups=dict(scoring_feature_preparation=PREP,grid_core=GRID,K_and_selection=BUDGET),
        pipeline=pipeline.metadata,deployment_audit=audit)
    write_json(HERE/'protocol.json',protocol)
    rows=[]
    def clean(i):
        torch.cuda.synchronize();start=time.perf_counter()
        rgb,k,_,_=pipeline.run(-1,i,'off')
        torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        assert torch.isfinite(rgb).all();del rgb
        return k,elapsed
    def instrumented(i):
        torch.cuda.synchronize();start=time.perf_counter();total_start.record()
        rgb,k,m,_=pipeline.run(-1,i,'events')
        total_end.record();torch.cuda.synchronize()
        wall=(time.perf_counter()-start)*1000
        total=float(total_start.elapsed_time(total_end))
        parts=m.finish();assert torch.isfinite(rgb).all();del rgb
        return k,total,wall,parts
    with torch.inference_mode():
        for i in range(WARMUP):pipeline.run(-1,i,'off')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.time()
        for i in range(N):
            if i%2==0:
                k,total,wall,parts=instrumented(i);ck,cwall=clean(i)
            else:
                ck,cwall=clean(i);k,total,wall,parts=instrumented(i)
            assert k==ck
            groups={name:sum(parts[n] for n in names) for name,names in [('preparation',PREP),('grid_core',GRID),('K_selection',BUDGET)]}
            groups['core_grid_and_K']=groups['grid_core']+groups['K_selection']
            groups['Grid_including_preparation']=groups['preparation']+groups['grid_core']
            groups['full_routing']=groups['preparation']+groups['grid_core']+groups['K_selection']
            assert sum(parts.values())<=total+1e-3
            rows.append(dict(index=i,K=k,clean_e2e_wall_ms=cwall,instrumented_e2e_wall_ms=wall,
                event_e2e_ms=total,components_ms=parts,groups_ms=groups,
                groups_percent={n:100*v/total for n,v in groups.items()}))
            if (i+1)%10==0:
                write_json(HERE/'progress.json',dict(completed=i+1,total=N,seconds=time.time()-started))
    frozen=pipeline.engine.audit()
    result=dict(status='complete',n=N,batch=1,
        clean_e2e_wall_ms=stats([r['clean_e2e_wall_ms'] for r in rows]),
        instrumented_e2e_wall_ms=stats([r['instrumented_e2e_wall_ms'] for r in rows]),
        event_e2e_ms=stats([r['event_e2e_ms'] for r in rows]),
        components_ms={name:stats([r['components_ms'][name] for r in rows]) for name in rows[0]['components_ms']},
        groups_ms={name:stats([r['groups_ms'][name] for r in rows]) for name in rows[0]['groups_ms']},
        groups_percent={name:stats([r['groups_percent'][name] for r in rows]) for name in rows[0]['groups_percent']},
        aggregate_groups_percent={name:100*sum(r['groups_ms'][name] for r in rows)/sum(r['event_e2e_ms'] for r in rows) for name in rows[0]['groups_ms']},
        mean_K=float(np.mean([r['K'] for r in rows])),rows=rows,protocol_sha256=sha(HERE/'protocol.json'),
        final_frozen_audit=frozen,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
    write_json(HERE/'result.json',result)
    lines=['# 当前无density Long：重建中Grid与K选区耗时','',
        'RTX5090，batch1，200张预先选定验证图，warmup20；图像预置GPU，排除加载/裁剪、启动和质量指标。',
        '',f"独立、无分项插桩的完整重建wall-clock均值：{result['clean_e2e_wall_ms']['mean']:.4f} ms/图。",'',
        '| 部分 | 均值 ms | 完整CUDA时间线占比 |','|---|---:|---:|']
    for key,label in [('preparation','评分专用准备：base RGB+B0+差异图'),('grid_core','Grid核心：white+gain CNN+局部分数组合'),('K_selection','K与最终选区：U+阈值计数+排序'),('core_grid_and_K','Grid核心与K合计'),('full_routing','包含准备的完整路由合计')]:
        lines.append(f"| {label} | {result['groups_ms'][key]['mean']:.4f} | {result['aggregate_groups_percent'][key]:.2f}% |")
    lines+=['','分项使用CUDA events，不在每个小组件之间加同步；占比以同次插桩的完整CUDA时间线为分母。独立wall-clock另报，避免把多个分项同步开销混进总时间。',
        '完整重建包含1D编码/量化、空间展开、base解码、dense2D编码/投影、B0/特征准备、评分与K选区、特征混合和最终RGB解码。',
        '评分专用准备属于完整路由开销；只引用CNN/排序的小占比会漏掉这些代价。完整路由百分比是时间归因，不是删掉模块后保证获得的加速比例。']
    (HERE/'RESULTS_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ['status','clean_e2e_wall_ms','event_e2e_ms','aggregate_groups_percent']}),flush=True)


if __name__=='__main__':main()
