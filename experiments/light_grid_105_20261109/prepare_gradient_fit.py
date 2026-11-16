"""Build fit-only population FID gradient and class-balanced gradient examples."""
import json,hashlib
from pathlib import Path
import numpy as np
from common import PREV,OLD,CAL,HEAD
from fit_router_budget_head import moments
from fid_math import fid_gradient
from fit_encoder_ridge import predict_ridge
from light_grid_selector import allocate_pool
from cache_train_counterfactual import sha256_file,EXPECTED_CHECKPOINT_SHA256

HERE=Path(__file__).resolve().parent
out=HERE/'gradient_fit'
if out.exists():raise FileExistsError(out)
paths=(PREV/'expanded_train100k_paths.txt').read_text().splitlines()
roles=np.load(PREV/'expanded_encoder_ridge/roles.npy');fit=np.flatnonzero(roles==0)
assert len(paths)==100000 and len(fit)==84996 and all('/train/' in p for p in paths)
cal=set((CAL/'paths.txt').read_text().splitlines());assert not cal.intersection(paths)
byclass={}
for i in fit:byclass.setdefault(Path(paths[i]).parent.name,[]).append(int(i))
assert len(byclass)==1000
idx=[]
for key in sorted(byclass):
    choices=sorted(byclass[key],key=lambda j:hashlib.sha256(paths[j].encode()).digest())
    assert len(choices)>=12;idx.extend(choices[:12])
idx=np.asarray(idx);assert len(idx)==12000 and len(set(idx))==12000
arrays={}
for kind in ['real','k96']:
    x=np.empty((100000,2048),dtype='float32')
    x[:50000]=np.load(OLD/f'train50k_assembled_5090/inception_{kind}.npy',mmap_mode='r')
    for s in range(2):
        d=PREV/f'extra_train50k/shard_{s}';ids=np.load(d/'indices.npy')
        assert (d/'paths.txt').read_text().splitlines()==[paths[50000+j] for j in ids]
        x[50000+ids]=np.load(d/f'inception_{kind}.npy',mmap_mode='r')
    arrays[kind]=x
mr,cr=moments(arrays['real'][fit],1024);mf,cf=moments(arrays['k96'][fit],1024)
gm,gc=fid_gradient(mr,cr,mf,cf,1e-6)
gc=gc.astype('float64')*len(fit)/(len(fit)-1);gm=2*(mf-mr)-2*gc@mf
out.mkdir();np.savez(out/'fid_gradient.npz',gm=gm.astype('float32'),gc=gc.astype('float32'))
np.save(out/'indices.npy',idx);(out/'paths.txt').write_text('\n'.join(paths[i] for i in idx)+'\n')
head=np.load(HEAD);x=np.load(PREV/'expanded_encoder_ridge/encoder_features.npy',mmap_mode='r')[idx]
score=predict_ridge(head,x)[:,0];policy=dict(kind='power',low=32,high=192,score='cosine',weight=.3)
np.save(out/'budget.npy',allocate_pool(score,policy))
report=dict(status='completed',n=12000,fit_population=84996,per_class=12,validation_images_used=0,
    calibration_images_used_for_fit=0,checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,head_sha256=sha256_file(HEAD),
    source_paths_sha256=sha256_file(PREV/'expanded_train100k_paths.txt'),files_sha256=sha256_file(out/'paths.txt'),
    fid_gradient_reference='fit-only original-ranking fixed96 reconstructions; not a calibration or validation gradient',
    source_roles_sha256=sha256_file(PREV/'expanded_encoder_ridge/roles.npy'))
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
