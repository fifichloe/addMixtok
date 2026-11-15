"""Evaluate frozen density-free Long reconstruction from explicit local assets.

Run as python -m mixtok_long_no_density.recon.evaluate from the repository root.
Budget calibration, when requested, finishes before quality is measured.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from scipy.linalg import eigh, sqrtm
from .io import ImagePathDataset, feature_tensor, sha256_file
from .runtime import FrozenInputs
from .scorer import LongScorer

PACKAGE = Path(__file__).resolve().parents[1]


def main(args):
    torch.set_num_threads(4)
    torch.manual_seed(20261115)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=False)
    paths = args.images.read_text().splitlines()
    assert len(paths) == len(set(paths)) == 50000, 'Use the complete unique validation manifest'
    public_repo = Path(os.environ['MIXTOK_REPO']).expanduser().resolve()
    titok_root = Path(os.environ['MIXTOK_TITOK_ROOT']).expanduser().resolve()
    llamagen_root = Path(os.environ['MIXTOK_LLAMAGEN_ROOT']).expanduser().resolve()
    cfg = SimpleNamespace(reconstruction_root=public_repo/'reconstruction', checkpoint=args.codec,
        titok_root=titok_root,
        titok_config=titok_root/'configs/infer/TiTok/titok_l32.yaml',
        titok_ckpt=args.titok, llamagen_root=llamagen_root, llamagen_ckpt=args.llamagen)
    assert sha256_file(args.codec) == '457be008836e3f3702327795bb1953250f0032a0b35f74e26e6ffce9ef50e82e', 'Use the released main codec; fixed20 is a separate diagnostic'
    runtime = FrozenInputs(cfg, args.semantic)
    scorer = LongScorer(args.gain, args.scales, args.utility)
    loader = torch.utils.data.DataLoader(ImagePathDataset(list(map(Path, paths)), runtime.crop),
        batch_size=args.batch, num_workers=4, shuffle=False, pin_memory=True, persistent_workers=True)
    scores = np.lib.format.open_memmap(args.output/'scores.npy', mode='w+', dtype='float32', shape=(50000,256))
    cursor = 0
    with torch.no_grad():
        for images in loader:
            values = scorer.score(runtime.prepare(images.cuda(non_blocking=True)))
            scores[cursor:cursor+len(images)] = values.cpu().numpy()
            cursor += len(images)
    assert cursor == 50000 and np.isfinite(scores).all()
    scores.flush()
    tau = scorer.threshold
    if args.calibrate_mean_K:
        lo, hi = float(scores.min())-1, float(scores.max())+1
        best = (float('inf'), None)
        for _ in range(80):
            trial = float(np.float32((lo+hi)/2))
            mean = np.clip((scores >= trial).sum(1),48,144).mean()
            if abs(mean-96) < best[0]: best = (abs(mean-96),trial)
            if mean > 96: lo = trial
            else: hi = trial
        assert best[0] < .01
        tau = best[1]
    k = np.clip((scores >= tau).sum(1),48,144)
    if args.fixed_K is not None: k = np.full(50000,args.fixed_K,dtype=np.int64)
    order = np.argsort(-scores,axis=1,kind='stable')
    mask = np.zeros((50000,256),dtype='uint8')
    np.put_along_axis(mask,order,(np.arange(256)[None] < k[:,None]).astype('uint8'),axis=1)
    np.save(args.output/'K.npy',k); np.save(args.output/'mask.npy',mask)
    protocol = dict(density=False,threshold=tau,mean_K=float(k.mean()),fixed_K=args.fixed_K,
        calibration_uses_scores_only=True,quality_not_computed_yet=True,
        images_sha256=sha256_file(args.images),codec_sha256=sha256_file(args.codec),
        weights={n:sha256_file(getattr(args,n)) for n in ['gain','scales','utility','semantic','titok','llamagen','inception']},
        mask_sha256=sha256_file(args.output/'mask.npy'),batch=args.batch)
    (args.output/'policy.json').write_text(json.dumps(protocol,indent=2)+'\n')
    from torchmetrics.image.fid import NoTrainInceptionV3
    inc = NoTrainInceptionV3(name='inception-v3-compat',features_list=['2048'],
        feature_extractor_weights_path=str(args.inception)).cuda().eval().requires_grad_(False)
    real = np.lib.format.open_memmap(args.output/'real.npy',mode='w+',dtype='float32',shape=(50000,2048))
    fake = np.lib.format.open_memmap(args.output/'fake.npy',mode='w+',dtype='float32',shape=(50000,2048))
    cursor, psnr = 0, 0.
    with torch.no_grad():
        for images in loader:
            images = images.cuda(non_blocking=True); target = images*2-1
            with torch.autocast('cuda',dtype=torch.bfloat16):
                _, extra = runtime.model(images)
                f2,_ = runtime.ev.native_llamagen_feature(runtime.model.llamagen_vq,target,8,False)
            z = dict(target=target,f1=extra['f_1d_lg'],f2=f2)
            m = torch.as_tensor(mask[cursor:cursor+len(images)],device='cuda').reshape(-1,1,16,16)
            pred = runtime.decode(z,m)
            psnr += float(runtime.ev.psnr_sum(pred,target,'minus1_1'))
            real[cursor:cursor+len(images)] = feature_tensor(inc,runtime.ev.to_uint8(target,'minus1_1'))
            fake[cursor:cursor+len(images)] = feature_tensor(inc,runtime.ev.to_uint8(pred,'minus1_1'))
            cursor += len(images)
            if cursor % 320 == 0 or cursor == 50000:
                (args.output/'progress.json').write_text(json.dumps(dict(completed=cursor,total=50000)))
    assert cursor == 50000
    real.flush();fake.flush();runtime.audit()
    def moments(x):
        mean = sum((np.asarray(x[i:i+512],dtype='float64').sum(0) for i in range(0,len(x),512)),np.zeros(2048))/len(x)
        cov = np.zeros((2048,2048),dtype='float64')
        for i in range(0,len(x),512):
            d = np.asarray(x[i:i+512],dtype='float64')-mean
            cov += d.T@d
        return mean,cov/(len(x)-1)
    mr,cr=moments(real);mf,cf=moments(fake)
    vals,vec=eigh(cr);root=(vec*np.sqrt(np.maximum(vals,0)))@vec.T
    cross=root@cf@root;cross=(cross+cross.T)/2
    common=np.square(mr-mf).sum()+np.trace(cr)+np.trace(cf)
    fid=float(common-2*np.sqrt(np.maximum(eigh(cross,eigvals_only=True),0)).sum())
    independent_root=sqrtm(cr@cf)
    assert not np.iscomplexobj(independent_root) or np.abs(independent_root.imag).max()<1e-6
    independent=float(common-2*np.trace(independent_root).real)
    assert abs(fid-independent)<1e-4
    result=dict(status='complete',n=50000,rFID=fid,independent_rFID=independent,
        PSNR=psnr/50000,mean_K=float(k.mean()),mean_content_tokens=float(k.mean()+32),
        density=False,policy=protocol,precision='BF16 codec; FP32 scorer/Inception; TF32off')
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['images','output','codec','titok','llamagen','semantic','gain','scales','utility','inception']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--calibrate-mean-K',dest='calibrate_mean_K',action='store_true')
    p.add_argument('--fixed-K',dest='fixed_K',type=int,choices=range(257))
    main(p.parse_args())
