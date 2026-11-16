"""CPU stage: full 50k pooled feature statistics, never mean-of-shard FIDs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from ablation_common import VAL_PATHS, sha, write_json


def moments(arrays):
    n=0;total=np.zeros(2048,dtype='float64');outer=np.zeros((2048,2048),dtype='float64')
    for array in arrays:
        for start in range(0,len(array),512):
            x=np.asarray(array[start:start+512],dtype='float64');assert x.shape[1]==2048 and np.isfinite(x).all()
            n+=len(x);total+=x.sum(0);outer+=x.T@x
    mean=total/n;cov=(outer-n*np.outer(mean,mean))/(n-1)
    return mean,(cov+cov.T)/2,n


def main():
    p=argparse.ArgumentParser();p.add_argument('--quality-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shards',type=int,default=4);a=p.parse_args()
    folders=[a.quality_root/f'shard_{i}' for i in range(a.shards)];reports=[json.loads((f/'report.json').read_text()) for f in folders]
    paths=sum([(f/'paths.txt').read_text().splitlines() for f in folders],[])
    assert paths==VAL_PATHS.read_text().splitlines() and len(paths)==len(set(paths))==50000
    names=reports[0]['names'];policy_path=Path(reports[0]['policy_protocol']);policy=json.loads(policy_path.read_text())
    for i,(folder,r) in enumerate(zip(folders,reports)):
        assert r['status']=='quality_features_complete' and r['names']==names and r['shard']==i
        assert r['policy_sha256']==sha(policy_path) and r['paths_sha256']==sha(folder/'paths.txt')
        assert r['real_sha256']==sha(folder/'real.npy')
        for path,digest in r['source_sha256'].items():assert sha(path)==digest
        for name,digest in r['feature_sha256'].items():assert sha(folder/(name+'.features.npy'))==digest
    mr,cr,n=moments([np.load(f/'real.npy',mmap_mode='r') for f in folders]);assert n==50000
    eig,vec=eigh(cr,check_finite=False);root=(vec*np.sqrt(np.maximum(eig,0)))@vec.T;rows=[]
    source_rows={row['name']:row for row in policy['variants']}
    for name in names:
        mf,cf,nf=moments([np.load(f/(name+'.features.npy'),mmap_mode='r') for f in folders]);assert nf==n
        ev=eigh(root@cf@root,eigvals_only=True,check_finite=False)
        fid=float(np.square(mr-mf).sum()+np.trace(cr)+np.trace(cf)-2*np.sqrt(np.maximum(ev,0)).sum())
        assert np.isfinite(fid) and fid>=0
        independently_trained_head=bool(source_rows[name].get('independently_retrained_RGB_only_head',False))
        if independently_trained_head:
            assert all(r['runtime'].get('no_semantic_network_instantiated') for r in reports), 'No-semantic quality must use the B0-free runtime'
        row=dict(source_rows[name],rFID=fid,PSNR=sum(r['PSNR_sums'][name] for r in reports)/n,n=n,
                 codec_retrained=False,gain_head_independently_retrained=independently_trained_head,
                 inference_control_on_original_head=not independently_trained_head)
        rows.append(row);print(json.dumps(row),flush=True)
    write_json(a.output,dict(status='full_50000_quality_complete',kind=policy['kind'],rows=rows,
        full_unique_validation_coverage=True,no_shard_FID_averaging=True,
        protocol=str(policy_path),protocol_sha256=sha(policy_path),reports=[dict(path=str(f/'report.json'),sha256=sha(f/'report.json')) for f in folders],
        formal_teacher_policy=policy.get('formal_teacher_policy'),
        codec_retrained=False,
        inference_controls_only=not any(row['gain_head_independently_retrained'] for row in rows),
        independent_retraining=any(row['gain_head_independently_retrained'] for row in rows),
        independent_retraining_components=['RGB_only_gain_head'] if any(row['gain_head_independently_retrained'] for row in rows) else [],
        newly_trained_head_runs=1 if any(row['gain_head_independently_retrained'] for row in rows) else 0,
        training_seed_variability_evaluated=False,
        interpretation='No-semantic row uses an independently trained RGB-only head on the unchanged frozen codec; other rows are frozen-head inference controls.'))


if __name__=='__main__':main()
