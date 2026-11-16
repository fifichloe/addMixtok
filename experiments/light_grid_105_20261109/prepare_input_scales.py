import json
import numpy as np
from common import PREV
from pathlib import Path
from cache_train_counterfactual import sha256_file,EXPECTED_CHECKPOINT_SHA256
HERE=Path(__file__).resolve().parent
out=HERE/'input_scales.npz'
if out.exists():raise FileExistsError(out)
roles=np.load(PREV/'expanded_encoder_ridge/roles.npy');idx=np.flatnonzero(roles==0)
x=np.load(PREV/'expanded_encoder_ridge/encoder_features.npy',mmap_mode='r')
m1=np.asarray(x[idx,:256],dtype='float64');s1=np.asarray(x[idx,256:512],dtype='float64')
m2=np.asarray(x[idx,512:768],dtype='float64');s2=np.asarray(x[idx,768:1024],dtype='float64')
mean=(m1.mean(0)+m2.mean(0))/2
second=((s1**2+m1**2).mean(0)+(s2**2+m2**2).mean(0))/2
std=np.sqrt(np.maximum(second-mean**2,1e-8))
np.savez(out,mean=mean.astype('float32'),std=std.astype('float32'))
(HERE/'input_scales.json').write_text(json.dumps(dict(n_fit=len(idx),validation_images_used=0,
    calibration_images_used=0,checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
    source_roles_sha256=sha256_file(PREV/'expanded_encoder_ridge/roles.npy'),
    source='Equal mixture of f1/f2 per-channel first and second moments on fit-only train images'),indent=2)+'\n')
print('Fit-only input scales saved',len(idx),flush=True)
