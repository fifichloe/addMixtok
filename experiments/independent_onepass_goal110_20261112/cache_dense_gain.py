"""Offline base-vs-dense gain labels, independent of any routing policy."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from routing import HERE,FRESH,FrozenInputs,ImagePathDataset,write_json,sha256_file,AnalyticMaps


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261112)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    source=FRESH/f'targets_{a.shard}';report=json.loads((source/'report.json').read_text())
    assert report['status']=='completed' and report['legacy_router_deleted'] and report['old_scores_used'] is False
    paths=(source/'paths.txt').read_text().splitlines()
    if a.limit is not None:paths=paths[:a.limit]
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'paths.txt').write_text('\n'.join(paths)+'\n')
    n=len(paths);cached=np.load(source/'inputs.npy',mmap_mode='r')
    targets=np.lib.format.open_memmap(a.output/'gain.npy',mode='w+',dtype='float32',shape=(n,2,16,16))
    white=np.lib.format.open_memmap(a.output/'white.npy',mode='w+',dtype='float16',shape=(n,1,16,16))
    runtime=FrozenInputs();map_fn=AnalyticMaps()
    loader=DataLoader(ImagePathDataset(list(map(Path,paths)),runtime.crop),batch_size=8,num_workers=4,pin_memory=True)
    done=0;started=time.monotonic();parity=[]
    with torch.no_grad():
        for images in loader:
            images=images.cuda(non_blocking=True);z=runtime.prepare(images);b=len(images)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                native=runtime.model.llamagen_vq.decoder(z['f2']).float()
                middle,deep,_=runtime.semantic(native.add(1).mul(.5).clamp(0,1))
            rgb_native=F.adaptive_avg_pool2d((native-z['target']).square().mean(1,keepdim=True),16)
            rgb_gain=z['features'][:,770:771]-rgb_native
            mid,refdeep,_=z['reference']
            mid_error=1-F.cosine_similarity(mid,middle,dim=1).unsqueeze(1)
            deep_error=1-F.cosine_similarity(refdeep,deep,dim=1).unsqueeze(1)
            interp=lambda x:F.interpolate(x,size=(16,16),mode='bilinear',align_corners=False)
            semantic_gain=.25*(z['features'][:,774:775]-interp(mid_error))+(z['features'][:,775:776]-interp(deep_error))
            target=torch.cat([rgb_gain,semantic_gain],1)
            assert torch.isfinite(target).all()
            targets[done:done+b]=target.cpu().numpy();white[done:done+b]=map_fn(z)['white'].cpu().numpy().astype('float16')
            if done==0:
                current=z['features'].cpu().numpy().astype('float16')
                stored=np.asarray(cached[:b])
                maximum=float(np.max(np.abs(current.astype('float32')-stored.astype('float32'))))
                parity.append(dict(first_batch_cached_input_exact=bool(np.array_equal(current,stored)),max_abs_difference=maximum))
                # Same source, image order and native B8; don't accept a changed input definition.
                assert np.allclose(current,stored,atol=.02,rtol=.02),maximum
            done+=b
            if done%128==0 or done==n:print(json.dumps(dict(done=done,total=n,seconds=time.monotonic()-started)),flush=True)
    targets.flush();white.flush()
    write_json(a.output/'report.json',dict(status='completed',n=n,source_inputs=str(source/'inputs.npy'),
        source_indices=str(source/'source_indices.npy'),paths_sha256=sha256_file(a.output/'paths.txt'),
        labels='signed RGB local error reduction; signed B0 middle/deep local cosine-error reduction',
        extra_dense_decode_at_inference=False,old_routing_or_mask_used=False,Inception_used=False,
        parity=parity,source_sha256={str(path):sha256_file(path) for path in [Path(__file__),HERE/'routing.py',FRESH/'runtime.py']},
        seconds=time.monotonic()-started,**runtime.audit()))


if __name__=='__main__':main()
