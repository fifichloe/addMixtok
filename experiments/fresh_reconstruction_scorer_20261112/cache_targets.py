"""Train-only utility gradients and new-router inputs; never load legacy scores."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from runtime import HERE, ROOT, FrozenInputs, ImagePathDataset, INPUT_CHANNELS, write_json, sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_num_threads(4); torch.manual_seed(20261112)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    source = ROOT/'experiments/light_grid_105_20261109/gradient_fit'
    paths = (source/'paths.txt').read_text().splitlines()
    indices = np.load(source/'indices.npy')
    assert len(paths) == len(indices) == 12000 and len(set(paths)) == len(paths)
    cal = (ROOT/'experiments/dynamic_recon_20261109/extra_train100k/shard_1/paths.txt').read_text().splitlines()
    assert not set(paths).intersection(cal) and all('/train/' in path for path in paths)
    subset = np.arange(len(paths))[a.shard::a.shards]
    if a.limit is not None:
        subset = subset[:a.limit]
    selected = [Path(paths[i]) for i in subset]
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'paths.txt').write_text('\n'.join(map(str, selected))+'\n')
    np.save(a.output/'source_indices.npy', indices[subset])
    n = len(selected)
    x = np.lib.format.open_memmap(a.output/'inputs.npy', mode='w+', dtype='float16', shape=(n, INPUT_CHANNELS, 16, 16))
    labels = np.lib.format.open_memmap(a.output/'gradients.npy', mode='w+', dtype='float32', shape=(n, 2, 16, 16))
    losses = np.lib.format.open_memmap(a.output/'losses.npy', mode='w+', dtype='float32', shape=(n, 2))
    runtime = FrozenInputs()
    loader = DataLoader(ImagePathDataset(selected, runtime.crop), batch_size=a.batch_size,
        num_workers=4, pin_memory=True)
    done, started = 0, time.monotonic()
    for images in loader:
        images = images.cuda(non_blocking=True)
        z = runtime.prepare(images)
        mask = torch.full((len(images), 1, 16, 16), .375, device='cuda', requires_grad=True)
        pred = runtime.decode(z, mask)
        semantic, rgb = runtime.losses(pred, z)
        semantic_gradient = torch.autograd.grad(semantic.sum(), mask, retain_graph=True)[0]
        rgb_gradient = torch.autograd.grad(rgb.sum(), mask)[0]
        target = -torch.cat([semantic_gradient, rgb_gradient], 1)
        assert torch.isfinite(target).all() and torch.isfinite(z['features']).all()
        b = len(images)
        x[done:done+b] = z['features'].cpu().numpy().astype('float16')
        labels[done:done+b] = target.detach().cpu().numpy()
        losses[done:done+b] = torch.stack([semantic, rgb], 1).detach().cpu().numpy()
        done += b
        if done <= a.batch_size or done % 128 == 0 or done == n:
            print(json.dumps(dict(done=done, total=n, seconds=time.monotonic()-started,
                semantic=float(semantic.detach().mean()), RGB=float(rgb.detach().mean()),
                peak_GiB=torch.cuda.max_memory_allocated()/2**30)), flush=True)
        if done == b:
            write_json(a.output/'first_batch_audit.json', runtime.audit())
    x.flush(); labels.flush(); losses.flush()
    write_json(a.output/'report.json', dict(status='completed', n=done, shard=a.shard,
        shards=a.shards, paths_sha256=sha256_file(a.output/'paths.txt'),
        fit=int((indices[subset]%10 != 0).sum()), held=int((indices[subset]%10 == 0).sum()),
        source_sha256={name:sha256_file(HERE/name) for name in ['runtime.py', 'cache_targets.py']},
        training_only_uniform_continuous_mask=.375, legacy_score_or_mask_labels_used=False,
        Inception_LPIPS_DINO_used=False, seconds=time.monotonic()-started,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30, **runtime.audit()))


if __name__ == '__main__':
    main()
