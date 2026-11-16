"""CPU stage: freeze budget-only calibrations and every mask before quality eval."""
import argparse
import json
from pathlib import Path
import numpy as np
from ablation_common import HERE, POLICIES, VAL_PATHS, sha, write_json, formal_teacher_policy


def calibrate(scores,target):
    lo,hi=float(scores.min())-1,float(scores.max())+1;best=(1e9,None,None)
    for _ in range(80):
        tau=float(np.float32((lo+hi)/2));k=np.clip(np.count_nonzero(scores>=tau,axis=1),48,144)
        item=(abs(float(k.mean())-target),tau,k)
        if item[0]<best[0]:best=item
        if k.mean()>target:lo=tau
        else:hi=tau
    assert best[0]<.01,(target,best[0]);return best[2],dict(threshold=best[1],mean_K=float(best[2].mean()),target=target)


def top_p_budget(local,target,temperature=1.):
    z=np.asarray(local,dtype=np.float64)/temperature;z-=z.max(1,keepdims=True)
    z=np.exp(z);z/=z.sum(1,keepdims=True);sorted_p=np.sort(z,axis=1)[:,::-1]
    before=np.cumsum(sorted_p,axis=1)-sorted_p
    lo,hi=0.,1.;best=(1e9,None,None)
    for _ in range(70):
        p=(lo+hi)/2;k=np.clip(np.count_nonzero(before<p,axis=1),48,144)
        error=abs(float(k.mean())-target)
        if error<best[0]:best=(error,p,k)
        if k.mean()>target:hi=p
        else:lo=p
    assert best[0]<.01,(target,best[0]);return best[2],dict(p=best[1],temperature=temperature,target=target,mean_K=float(best[2].mean()))


def uniform_order():
    coords=[(r,c) for r in range(16) for c in range(16)];selected=[coords.pop(0)]
    while coords:
        i=max(range(len(coords)),key=lambda i:(min((coords[i][0]-r)**2+(coords[i][1]-c)**2 for r,c in selected),-coords[i][0],-coords[i][1]))
        selected.append(coords.pop(i))
    return np.asarray([r*16+c for r,c in selected],dtype='int16')


def near_constant(k,seed=20261114):
    low,remainder=divmod(int(k.sum()),len(k));result=np.full(len(k),low,dtype='int16')
    result[np.random.default_rng(seed).permutation(len(k))[:remainder]]+=1
    assert result.sum()==k.sum();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['short','long'],required=True)
    p.add_argument('--score-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shards',type=int,default=4);p.add_argument('--no-semantic-root',type=Path);a=p.parse_args()
    folders=[a.score_root/f'shard_{i}' for i in range(a.shards)];reports=[json.loads((f/'report.json').read_text()) for f in folders]
    paths=sum([(f/'paths.txt').read_text().splitlines() for f in folders],[])
    assert paths==VAL_PATHS.read_text().splitlines() and len(paths)==len(set(paths))==50000
    for i,(folder,report) in enumerate(zip(folders,reports)):
        assert report['status']=='scores_complete' and report['kind']==a.kind and report['shard']==i
        assert report['paths_sha256']==sha(folder/'paths.txt') and not report['quality_used']
        for key,digest in report['score_sha256'].items():assert sha(folder/(key+'.npy'))==digest
        for source,digest in report['source_sha256'].items():assert sha(source)==digest
        for source,digest in report['scorers']['scorer_asset_sha256'].items():assert sha(source)==digest
    keys=list(reports[0]['score_sha256']);scores={key:np.concatenate([np.load(f/(key+'.npy'),mmap_mode='r') for f in folders]) for key in keys}
    assert all(value.shape==(50000,256) and np.isfinite(value).all() for value in scores.values())
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'paths.txt').write_text('\n'.join(paths)+'\n')
    teacher=scores['teacher'];ranking=np.argsort(-teacher,axis=1,kind='stable');n=len(teacher);rows=[]
    def save(name,order,k,family,**meta):
        k=np.asarray(k,dtype='int16');assert k.shape==(n,) and k.min()>=0 and k.max()<=256
        order=np.broadcast_to(order,(n,256));mask=np.zeros((n,256),dtype='uint8')
        np.put_along_axis(mask,order,(np.arange(256)[None]<k[:,None]).astype('uint8'),axis=1)
        assert np.array_equal(mask.sum(1),k)
        np.save(a.output/(name+'.mask.npy'),mask);np.save(a.output/(name+'.K.npy'),k)
        rows.append(dict(name=name,family=family,mean_K=float(k.mean()),min_K=int(k.min()),max_K=int(k.max()),
            mean_content_tokens=float(k.mean()+32),bpp_mask256=float((32*12+14*k.mean()+256)/65536),
            mask_file=str(a.output/(name+'.mask.npy')),mask_sha256=sha(a.output/(name+'.mask.npy')),
            K_file=str(a.output/(name+'.K.npy')),K_sha256=sha(a.output/(name+'.K.npy')),**meta))
    formal=formal_teacher_policy(a.kind)
    exact=np.clip((teacher>=formal['threshold']).sum(1),48,144)
    assert float(exact.mean())==96.,('Accepted teacher must replay mean96; investigate precision/path differences',float(exact.mean()))
    save('teacher_dynamic96',ranking,exact,'A01_A02_A03',threshold=POLICIES[a.kind]['threshold'])
    for name in ['generation','legacy','gradient_entropy']:
        save('rank_'+name,np.argsort(-scores[name],axis=1,kind='stable'),exact,'A01',same_teacher_per_image_K=True)
    save('rank_uniform',uniform_order()[None],exact,'A01',same_teacher_per_image_K=True)
    for seed in [1,2,3]:
        # Selection seeds are control seeds, separate from training/sampling seeds.
        order=np.argsort(np.random.default_rng(seed).random((n,256)),axis=1,kind='stable')
        save(f'rank_random_seed{seed}',order,exact,'A01',same_teacher_per_image_K=True,selection_seed=seed)
    for k in [0,32,64,96,128,196,256]:save(f'fixed{k}',ranking,np.full(n,k),'A02_A03',frozen_checkpoint_inference_curve=True)
    for target in [64,96,128]:
        if target==96:k,cal=exact,dict(threshold=POLICIES[a.kind]['threshold'],mean_K=float(exact.mean()),target=96)
        else:
            k,cal=calibrate(teacher,target);save(f'dynamic{target}',ranking,k,'A03',calibration=cal)
        save(f'matched_mean{target}',ranking,near_constant(k),'A02_A03',exact_dynamic_total_budget=True,content_independent_assignment=True)
        for seed in [1,2,3]:
            shuffled=k[np.random.default_rng(seed).permutation(n)]
            assert np.array_equal(np.sort(shuffled),np.sort(k))
            save(f'shuffled{target}_seed{seed}',ranking,shuffled,'A02_A03',permutation_seed=seed,budget_histogram_exact=True)
    k,cal=calibrate(scores['local'],96);save('U_zero_recalibrated',ranking,k,'A02',calibration=cal)
    k,cal=top_p_budget(scores['local'],96);save('same_local_top_p',ranking,k,'A02',calibration=cal,uniform_U_shift_cancels_in_softmax=True)
    for component in ['gain','white','cam','density']:
        order=np.argsort(-scores['without_'+component],axis=1,kind='stable')
        save('without_'+component+'_fixed96',order,np.full(n,96),'B01',direct_term_removed_only=True,
             semantic_features_and_training_targets_still_present=True)
    if a.no_semantic_root is not None:
        no_folders=[a.no_semantic_root/f'shard_{i}' for i in range(a.shards)]
        assert sum([(f/'paths.txt').read_text().splitlines() for f in no_folders],[])==paths
        for f in no_folders:
            r=json.loads((f/'report.json').read_text())
            assert r['status']=='no_semantic_scores_complete' and r['kind']==a.kind and not r['quality_used']
            assert r['no_semantic_network_instantiated'] and r['score_sha256']==sha(f/'no_semantic.npy')
            assert r['head_sha256']==sha(r['head'])
            for source,digest in r['source_sha256'].items():assert sha(source)==digest
        value=np.concatenate([np.load(f/'no_semantic.npy',mmap_mode='r') for f in no_folders])
        assert value.shape==(50000,256) and np.isfinite(value).all()
        save('no_semantic_retrained_fixed96',np.argsort(-value,axis=1,kind='stable'),np.full(n,96),'B01',
             independently_retrained_RGB_only_head=True,semantic_features_removed=True,semantic_objective_removed=True,
             semantic_U_removed=True,B0_not_instantiated_in_scoring=True,
             source_reports=[dict(path=str(f/'report.json'),sha256=sha(f/'report.json')) for f in no_folders])
    write_json(a.output/'protocol.json',dict(status='policies_frozen_before_quality',kind=a.kind,n=n,variants=rows,
        source_reports=[dict(path=str(f/'report.json'),sha256=sha(f/'report.json')) for f in folders],
        source_sha256={str(path):sha(path) for path in [Path(__file__),HERE/'ablation_common.py',VAL_PATHS]},
        paths_sha256=sha(a.output/'paths.txt'),quality_used_for_calibration=False,
        calibration='score distributions only; no rFID/PSNR; inference each image independent after frozen scalar',
        formal_teacher_policy=formal,
        current_dynamic_K_range=[48,144],curves='frozen checkpoint; not separately trained per-budget optimal frontier',
        no_semantic_ablation='RGB-only head separately trained' if a.no_semantic_root is not None else 'requires separate RGB-only head training; direct CAM removal is not no-B0'))


if __name__=='__main__':main()
