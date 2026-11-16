"""Join completed ALIT training-cache shards in original manifest order."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--shards',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    folders=[a.source/f'shard_{i}' for i in range(a.shards)]
    reports=[json.loads((folder/'report.json').read_text()) for folder in folders]
    for i,r in enumerate(reports):
        assert r['status']=='completed' and r['shard']==i and r['shards']==a.shards
        assert r['source_span'][1]-r['source_span'][0]==r['n']
        assert i==0 or reports[i-1]['source_span'][1]==r['source_span'][0]
    assert reports[0]['source_span'][0]==0 and reports[-1]['source_span'][1]==12000
    assert len({r['checkpoint_sha256'] for r in reports})==1
    a.output.mkdir(parents=True,exist_ok=False)
    specifications={'features':('float16',(780,16,16)),
                    'f1':('float16',(256,16,16)),
                    'f2':('float16',(256,16,16)),
                    'gain':('float32',(2,16,16))}
    for name,(dtype,shape) in specifications.items():
        destination=np.lib.format.open_memmap(a.output/(name+'.npy'),mode='w+',
            dtype=dtype,shape=(12000,*shape))
        for folder,r in zip(folders,reports):
            source=np.load(folder/(name+'.npy'),mmap_mode='r')
            assert source.shape==(r['n'],*shape)
            left,right=r['source_span']
            destination[left:right]=source
        destination.flush()
    ids=np.concatenate([np.load(folder/'source_indices.npy') for folder in folders])
    paths=sum(((folder/'paths.txt').read_text().splitlines() for folder in folders),[])
    assert len(ids)==len(paths)==len(set(paths))==12000
    np.save(a.output/'source_indices.npy',ids)
    (a.output/'paths.txt').write_text('\n'.join(paths)+'\n')
    (a.output/'report.json').write_text(json.dumps(dict(status='completed',n=12000,
        source='merged ALIT shards',checkpoint_sha256=reports[0]['checkpoint_sha256'],
        validation_used=False,original_router_labels_used=False),indent=2)+'\n')
    print(json.dumps(dict(status='completed',n=12000,output=str(a.output))),flush=True)


if __name__=='__main__':main()
