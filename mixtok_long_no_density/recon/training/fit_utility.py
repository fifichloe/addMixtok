"""CPU-only independent image utility heads as a fallback budget experiment.

Uses only the new no-router feature/endpoint-gain caches. No validation, old
scores/masks, original router hidden states, metric networks or GPU are read.
"""
import json,time,os
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import spearmanr

HERE=Path(os.environ['MIXTOK_GAIN_CACHE']);FRESH=Path(os.environ['MIXTOK_BASE_FEATURE_CACHE'])


def main():
    output=HERE/'image_utility';output.mkdir(exist_ok=False)
    arrays=[];labels=[];indices=[];paths=[];started=time.monotonic()
    # Fixed units from FIT endpoint labels, same as the new gain head.
    all_gain=[];all_fit=[]
    for shard in range(4):
        all_gain.append(np.load(HERE/f'dense_gain_{shard}/gain.npy',mmap_mode='r'))
        all_fit.append(np.load(FRESH/f'targets_{shard}/source_indices.npy')%10!=0)
    fit_gain=np.concatenate([np.asarray(value[mask]) for value,mask in zip(all_gain,all_fit)])
    scales=np.quantile(np.abs(fit_gain),.75,axis=(0,2,3)).clip(1e-6)
    for shard,gain in enumerate(all_gain):
        report=json.loads((HERE/f'dense_gain_{shard}/report.json').read_text())
        assert report['legacy_router_deleted'] and report['old_routing_or_mask_used'] is False
        x=np.load(report['source_inputs'],mmap_mode='r');blocks=[];targets=[]
        for start in range(0,len(x),32):
            value=np.asarray(x[start:start+32],dtype='float64')
            mean=value.mean((2,3));std=value.std((2,3),ddof=0)
            blocks.append(np.concatenate([mean,std],1))
            y=np.asarray(gain[start:start+32],dtype='float64')
            rg=y[:,0].mean((1,2));sg=y[:,1].mean((1,2))
            base_rgb=mean[:,770];base_sem=.25*mean[:,774]+mean[:,775]
            mixed=rg/scales[0]+.25*sg/scales[1]
            base_mixed=base_rgb/scales[0]+.25*base_sem/scales[1]
            targets.append(np.stack([rg,rg/(base_rgb+1e-4),mixed,mixed/(base_mixed+1e-3),
                np.maximum(y[:,0],0).mean((1,2))/(base_rgb+1e-4),sg/(base_sem+1e-4)],1))
        arrays.append(np.concatenate(blocks));labels.append(np.concatenate(targets))
        indices.append(np.load(report['source_indices']))
        paths.extend((HERE/f'dense_gain_{shard}/paths.txt').read_text().splitlines())
        print(json.dumps(dict(stage='feature_statistics',shard=shard,seconds=time.monotonic()-started)),flush=True)
    x=np.concatenate(arrays);y=np.concatenate(labels);ids=np.concatenate(indices);fit=ids%10!=0;held=~fit
    assert len(x)==len(y)==len(ids)==len(set(paths))==12000
    assert int(fit.sum())==10843 and int(held.sum())==1157
    assert np.isfinite(x).all() and np.isfinite(y).all()
    np.save(output/'image_features.npy',x);np.save(output/'targets.npy',y);np.save(output/'source_indices.npy',ids)
    (output/'paths.txt').write_text('\n'.join(paths)+'\n')
    mean=x[fit].mean(0);std=x[fit].std(0).clip(1e-3)
    normalized=((x-mean)/std).clip(-8,8);center=normalized[fit].mean(0)
    xf=normalized[fit]-center;xh=normalized[held]-center
    target_mean=y[fit].mean(0);target_std=y[fit].std(0).clip(1e-6)
    yf=(y[fit]-target_mean)/target_std;yh=(y[held]-target_mean)/target_std
    gram=xf.T@xf;cross=xf.T@yf;names=['rgb_abs','rgb_rel','mixed_abs','mixed_rel','positive_rgb_rel','semantic_rel']
    best=[None]*len(names);rows=[]
    for relative in [.001,.01,.1,1.]:
        factor=cho_factor(gram+relative*len(xf)*np.eye(xf.shape[1]),check_finite=False)
        beta=cho_solve(factor,cross,check_finite=False);prediction=xh@beta
        mse=((prediction-yh)**2).mean(0)
        for column,name in enumerate(names):
            row=dict(name=name,ridge_relative=relative,held_MSE=float(mse[column]),
                held_Spearman=float(spearmanr(prediction[:,column],yh[:,column]).statistic))
            rows.append(row)
            if best[column] is None or row['held_MSE']<best[column]['held_MSE']:
                best[column]=row
                np.savez(output/(name+'.npz'),mean=mean,std=std,beta=beta[:,column],bias=-center@beta[:,column],
                    target_mean=target_mean[column],target_std=target_std[column],clip=8.)
    report=dict(status='trained_independent_image_utilities_not_quality_validated',n=12000,fit=10843,held=1157,
        feature_dimension=x.shape[1],target_names=names,best=best,rows=rows,
        features='FP64 mean/std over positions of FP16-rounded new 780-channel inputs; image axis never reduced at inference',
        labels='Absolute/relative new base-vs-dense RGB and B0 endpoint gains; not marginal K48-to144 gains',
        original_scorer_used=False,old_mask_used=False,validation_used=False,metric_network_supervision=False,
        seconds=time.monotonic()-started)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
