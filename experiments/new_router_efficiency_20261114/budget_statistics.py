"""CPU-only K and fixed-width bitrate accounting from actual completed runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
RECON = PAPER/'experiments/independent_onepass_goal100_20261113'
GEN = PAPER/'experiments/generation_two_router_maskgit_20261114'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def summarize(histogram):
    hist = np.asarray(histogram, dtype=np.int64)
    assert hist.shape == (257,) and np.all(hist >= 0) and hist.sum() > 0
    k = np.arange(257); n = int(hist.sum()); mean = float(hist@k/n)
    occupied = np.flatnonzero(hist)
    values = np.repeat(k, hist)
    # Bitmap representation is self-delimiting: its population count gives K.
    mask_bits = 256
    global_bits = 32*12
    return dict(n=n, mean_K=mean, mean_content_tokens=32+mean,
        min_K=int(occupied[0]), max_K=int(occupied[-1]),
        std_K=float(values.std(ddof=1)) if n > 1 else 0.,
        quantiles={str(q):float(np.quantile(values,q)) for q in [0,.1,.25,.5,.75,.9,1]},
        fraction_K48=float(hist[48]/n), fraction_K144=float(hist[144]/n),
        histogram=hist.tolist(),
        mean_content_bits=global_bits+14*mean,
        mean_mask_bits=mask_bits,
        mean_fixed_width_bits=global_bits+14*mean+mask_bits,
        mean_fixed_width_bpp=(global_bits+14*mean+mask_bits)/(256*256),
        alternative_explicit_8bit_index_bpp=(global_bits+(14+8)*mean)/(256*256),
        alternative_explicit_index_note='Diagnostic only; assumes K externally known. The primary bitmap representation needs no extra K field.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=HERE/'actual_budget_statistics')
    p.add_argument('--generation-results',type=Path,default=GEN/'final_results.json')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows={};sources={}
    for kind,root in [('long',RECON),('short',PAPER/'experiments/independent_onepass_goal105_20261112')]:
        accepted=json.loads((root/'accepted_policy.json').read_text()); name=accepted['name']
        all_k=[]
        for shard in range(4):
            path=root/f'val_quality_{shard}'/(name+'.K.npy')
            values=np.load(path); masks=np.load(path.with_name(name+'.masks.npy'),mmap_mode='r')
            assert np.array_equal(values,masks.sum(1))
            all_k.append(values);sources[str(path)]=sha(path)
        all_k=np.concatenate(all_k);assert len(all_k)==50000
        rows['reconstruction_'+kind+'_'+name]=summarize(np.bincount(all_k,minlength=257))
    results=json.loads(a.generation_results.read_text());sources[str(a.generation_results)]=sha(a.generation_results)
    assert results['status']=='both_trained_and_evaluated'
    for kind,row in results['results'].items():
        assert row['status']=='complete' and row['n']==50000 and not row['smoke']
        stats=summarize(row['tokens']['histogram'])
        assert stats['n']==row['n'] and abs(stats['mean_K']-row['tokens']['mean_generated_2d'])<1e-9
        stats.update(gFID=row['metrics']['full_refine']['gFID'],seed=row['seed'],
                     sampling_rounds=row['generation_rounds'],guidance=row['guidance'])
        rows['generation_'+kind]=stats
    report=dict(status='computed_from_completed_actual_50k_runs',rows=rows,source_sha256=sources,
        accounting=dict(global_tokens=32,global_codebook=4096,global_bits_per_id=12,
            local_codebook=16384,local_bits_per_id=14,spatial_mask_bits=256,image_pixels=65536,
            primary_formula='(32*12 + mean_K*14 + 256)/(256*256)',
            entropy_coding=False,container_bytes_included=False,model_weights_included=False,
            class_label_bits_included=False,optimized_empty_or_dense_endpoint_coding=False),
        note='These are real 50k distributions, not the 50 profiling inputs, not threshold calibration, and not measured entropy-coded file sizes.')
    (a.output/'statistics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fields=['name','n','mean_K','mean_content_tokens','mean_fixed_width_bits','mean_fixed_width_bpp','min_K','max_K']
    with (a.output/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name,row in rows.items():w.writerow(dict(name=name,**{key:row[key] for key in fields[1:]}))
    for name,row in rows.items():
        with (a.output/(name+'.histogram.csv')).open('w') as f:
            w=csv.writer(f);w.writerow(['K','images','fraction'])
            w.writerows((k,n,n/row['n']) for k,n in enumerate(row['histogram']) if n)
    lines=['# 实际 K 分布与码率统计','','来自完整 50k 推理结果，未重新生成图片。',
        '', '| 场景 | 平均局部 K | 平均总 content tokens | 含位置开销 bpp |', '|---|---:|---:|---:|']
    for name,row in rows.items():lines.append(f"| {name} | {row['mean_K']:.5f} | {row['mean_content_tokens']:.5f} | {row['mean_fixed_width_bpp']:.8f} |")
    lines+=['','固定宽度估算：32 个 12-bit 全局 ID、每个局部 ID 14 bits，以及 256-bit 空间掩码。',
        '不包含模型参数、类别标签和文件封装；不是实际压缩文件大小，也未使用熵编码。',
        '该预算统计与 batch=1 的计时输入分布分开报告。']
    (a.output/'README_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({name:{k:row[k] for k in ['n','mean_K','mean_fixed_width_bpp']} for name,row in rows.items()}))


if __name__=='__main__':main()
