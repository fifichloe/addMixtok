"""Cache gain/White/CAM removal targets aligned to the existing TRAIN base cache."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REV = HERE.parent / 'no_density_revision_20261114'
sys.path.insert(0, str(REV))
from density_free_config import DATA, sha, write_json
from teacher_no_density import TwoTeachers, ImagePathDataset, standard

ORIGINAL = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/no_density_revision_20261114/final/full/cache')
KINDS = ('without_gain', 'without_white_all_explicit', 'without_cam_all_explicit')


def utility_from_pooled(pooled, state):
    return (((pooled - state['mean']) / state['std']).clamp(-8, 8)
            @ state['beta'] + state['bias']).clamp(-3, 3).reshape(-1)


@torch.inference_mode()
def labels(teacher, z):
    full = teacher.score_prepared(z)
    a, b = (z['f1'].float() - teacher.mean) / teacher.std, (z['f2'].float() - teacher.mean) / teacher.std
    white = standard(1 - F.cosine_similarity(a, b, dim=1).unsqueeze(1))
    features = z['features'].half().float()
    head = teacher.heads['long']
    head_input = torch.cat([features, white.half().float()], 1)
    assert torch.equal(white.flatten(1), full['white'])
    gain = standard(head(head_input)[:, 1:2])
    assert torch.equal(gain.flatten(1), full['gain'][:, 1])
    cam = z['features'][:, 777:778]

    white_input = head_input.clone()
    white_input[:, 780:781] = head.input_mean[:, 780:781]
    gain_no_white = standard(head(white_input)[:, 1:2])

    cam_input = head_input.clone()
    cam_input[:, 776:778] = head.input_mean[:, 776:778]
    gain_no_cam = standard(head(cam_input)[:, 1:2])
    pooled = torch.cat([features.mean((2, 3)), features.std((2, 3), correction=0)], 1)
    cam_pooled = pooled.clone()
    for index in (776, 777, 1556, 1557):
        cam_pooled[:, index] = teacher.utility['mean'][index]
    utility_no_cam = utility_from_pooled(cam_pooled, teacher.utility)
    result = dict(
        without_gain=(.5 * white + .75 * cam, full['utility']),
        without_white_all_explicit=(gain_no_white + .75 * cam, full['utility']),
        without_cam_all_explicit=(gain_no_cam + .5 * white, utility_no_cam))
    baseline = full['local'][:, 1] + .9 * full['utility'][:, None]
    assert torch.isfinite(baseline).all()
    for local, utility in result.values():
        assert torch.isfinite(local).all() and torch.isfinite(utility).all()
        assert local.shape[1:] == (1, 16, 16) and utility.shape == (len(local),)
    return z['f1'], full['local'][:, 1], full['utility'], result


def main(args):
    torch.set_num_threads(4)
    torch.manual_seed(20261113)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    teacher = TwoTeachers()
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    progress = dict(rank=args.rank, world=args.world, completed_chunks=0, completed_images=0, split=None)
    for split in ('held', 'fit'):
        paths = (DATA / f'{split}_paths.txt').read_text().splitlines()
        directory = args.output / split
        directory.mkdir()
        source = ORIGINAL / split
        chunks = sorted(source.glob('*.pt'))[args.rank::args.world]
        expected_count = 5 if split == 'held' else 621
        assert len(list(source.glob('*.pt'))) == expected_count
        for original_path in chunks:
            original = torch.load(original_path, map_location='cpu', mmap=True, weights_only=False)
            indices = original['indices']
            selected = [paths[int(index)] for index in indices]
            loader = DataLoader(ImagePathDataset(list(map(Path, selected)), teacher.runtime.crop),
                batch_size=args.batch, num_workers=4, pin_memory=True)
            values = {kind: {'local': [], 'utility': []} for kind in KINDS}
            cursor = 0
            maximum_reference_gap = 0.
            for images in loader:
                z = teacher.runtime.prepare(images.cuda(non_blocking=True))
                f_base, baseline_local, baseline_utility, output = labels(teacher, z)
                size = len(images)
                reference_local = original['local'][cursor:cursor + size, 1].float().cuda()
                reference_utility = original['utility'][cursor:cursor + size].float().cuda()
                gap = max(float((baseline_local - reference_local).abs().max()),
                          float((baseline_utility - reference_utility).abs().max()))
                maximum_reference_gap = max(maximum_reference_gap, gap)
                assert gap < 5e-4, (split, original_path.name, cursor, gap)
                if cursor == 0:
                    assert torch.equal(f_base.cpu(), original['f_base'][:size])
                for kind, (local, utility) in output.items():
                    values[kind]['local'].append(local.flatten(1).cpu())
                    values[kind]['utility'].append(utility.cpu())
                cursor += size
            assert cursor == len(indices)
            packed = dict(indices=indices.clone(), split=split, source_cache=str(original_path),
                source_cache_size_bytes=original_path.stat().st_size,
                maximum_reference_gap=maximum_reference_gap,
                targets={kind: {key: torch.cat(chunks) for key, chunks in data.items()}
                         for kind, data in values.items()})
            destination = directory / original_path.name
            temporary = destination.with_suffix('.pt.tmp')
            torch.save(packed, temporary)
            temporary.replace(destination)
            progress.update(completed_chunks=progress['completed_chunks'] + 1,
                completed_images=progress['completed_images'] + cursor, split=split,
                last_chunk=original_path.name, elapsed_seconds=time.monotonic() - began)
            write_json(args.output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    write_json(args.output / 'report.json', dict(status='complete', **progress,
        kinds=list(KINDS), teacher_source_sha256=teacher.identities,
        original_cache=str(ORIGINAL), no_official_validation_images=True,
        MaskGIT_or_generation_model_retrained=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world', type=int, default=4)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
