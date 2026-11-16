"""Cache ALIT-specific independent scorer inputs and endpoint gain labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common import HERE, PAPER, SEMANTIC, ImagePaths, load_alit
from runtime import ALITFrozenInputs


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261116)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = PAPER / 'experiments/fresh_reconstruction_scorer_20261112'
    manifest = [source / f'targets_{i}/paths.txt' for i in range(4)]
    paths = sum((p.read_text().splitlines() for p in manifest), [])
    source_ids = [np.load(source / f'targets_{i}/source_indices.npy') for i in range(4)]
    ids = np.concatenate(source_ids)
    assert len(paths) == len(ids) == len(set(paths)) == 12000
    assert args.shards >= 1 and 0 <= args.shard < args.shards
    left = (12000 * args.shard) // args.shards
    right = (12000 * (args.shard + 1)) // args.shards
    paths, ids = paths[left:right], ids[left:right]
    if args.limit:
        paths, ids = paths[:args.limit], ids[:args.limit]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / 'paths.txt').write_text('\n'.join(paths) + '\n')
    np.save(out / 'source_indices.npy', ids)
    n = len(paths)
    features = np.lib.format.open_memmap(out / 'features.npy', mode='w+', dtype='float16', shape=(n, 780, 16, 16))
    first = np.lib.format.open_memmap(out / 'f1.npy', mode='w+', dtype='float16', shape=(n, 256, 16, 16))
    second = np.lib.format.open_memmap(out / 'f2.npy', mode='w+', dtype='float16', shape=(n, 256, 16, 16))
    gain = np.lib.format.open_memmap(out / 'gain.npy', mode='w+', dtype='float32', shape=(n, 2, 16, 16))
    model, checkpoint = load_alit()
    runtime = ALITFrozenInputs(model, SEMANTIC)
    torch.set_grad_enabled(False)
    loader = DataLoader(ImagePaths(paths), batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    done = 0
    started = time.monotonic()
    for images in loader:
        images = images.cuda(non_blocking=True)
        z = runtime.prepare(images)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            native = model.decode(z['f2']).float()
            middle, deep, _ = runtime.semantic(native.clamp(0, 1))
        rgb_native = F.adaptive_avg_pool2d((native * 2 - 1 - z['target']).square().mean(1, keepdim=True), 16)
        rgb_gain = z['features'][:, 770:771] - rgb_native
        mid, refdeep, _ = z['reference']
        interp = lambda v: F.interpolate(v, size=(16, 16), mode='bilinear', align_corners=False)
        mid_error = 1 - F.cosine_similarity(mid, middle, dim=1).unsqueeze(1)
        deep_error = 1 - F.cosine_similarity(refdeep, deep, dim=1).unsqueeze(1)
        semantic_gain = .25 * (z['features'][:, 774:775] - interp(mid_error)) + (z['features'][:, 775:776] - interp(deep_error))
        b = len(images)
        features[done:done+b] = z['features'].half().cpu().numpy()
        first[done:done+b] = z['f1'].half().cpu().numpy()
        second[done:done+b] = z['f2'].half().cpu().numpy()
        gain[done:done+b] = torch.cat((rgb_gain, semantic_gain), dim=1).cpu().numpy()
        done += b
        if done % 400 == 0 or done == n:
            print(json.dumps(dict(done=done, total=n, seconds=time.monotonic()-started)), flush=True)
    for array in (features, first, second, gain):
        array.flush()
    report = dict(status='completed', n=n, shard=args.shard, shards=args.shards,
                  source_span=[left,right], checkpoint=checkpoint,
                  checkpoint_sha256=sha256('/tmp/mixtok_alit20_20261116/alit64_vqgan_mix_fixed64_l1_3_h200_20epoch/epoch_0020.pt'),
                  source_paths_sha256=sha256(out/'paths.txt'),
                  label='base minus full native 2D signed RGB and EfficientNet-B0 semantic endpoint errors',
                  original_router_labels_used=False, validation_used=False,
                  audit=runtime.audit(), seconds=time.monotonic()-started)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
