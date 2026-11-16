"""Select faithful MixTok reconstruction comparisons from ImageNet validation.

The shortlist uses cached density-free routing scores only. Final ranking uses
newly decoded 1D and mixed images from the released reconstruction checkpoint.
No image synthesis, manual retouching, or metric computed on the publication
panels is used in the reconstruction itself.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mixtok_long_no_density.recon.io import ImagePathDataset, sha256_file
from mixtok_long_no_density.recon.runtime import FrozenInputs
from mixtok_long_no_density.recon.scorer import LongScorer

EXPECTED_CODEC_SHA = '457be008836e3f3702327795bb1953250f0032a0b35f74e26e6ffce9ef50e82e'
THRESHOLD = 0.3803865313529968
SEED = 20261116


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def records(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(args):
    args.work.mkdir(parents=True, exist_ok=False)
    all_rows = []
    source_sha = {}
    for shard in range(4):
        directory = args.score_cache / f'shard_{shard}'
        score_file = directory / 'without_density.npy'
        path_file = directory / 'paths.txt'
        score = np.load(score_file, mmap_mode='r')
        paths = path_file.read_text().splitlines()
        assert score.shape == (len(paths), 256)
        source_sha[f'shard_{shard}_scores'] = sha256_file(score_file)
        source_sha[f'shard_{shard}_paths'] = sha256_file(path_file)
        k = np.clip((score >= THRESHOLD).sum(1), 48, 144)
        top = np.sort(score, axis=1)[:, ::-1]
        for offset, name in enumerate(paths):
            p = Path(name)
            assert p.parent.name.startswith('n') and p.suffix.lower() == '.jpeg'
            all_rows.append(dict(validation_index=len(all_rows), source_path=str(p),
                                 relative_path=str(Path(p.parent.name) / p.name),
                                 class_id=p.parent.name, predicted_K=int(k[offset]),
                                 predicted_top_score=float(top[offset, :k[offset]].mean())))
    assert len(all_rows) == 50000
    assert len({row['relative_path'] for row in all_rows}) == 50000
    order = sorted(all_rows, key=lambda row: row['predicted_top_score'], reverse=True)
    rng = np.random.default_rng(SEED)
    chosen = []
    per_class = {}
    for row in order:
        if per_class.get(row['class_id'], 0) >= 6:
            continue
        chosen.append(row)
        per_class[row['class_id']] = per_class.get(row['class_id'], 0) + 1
        if len(chosen) == 1200:
            break
    seen = {row['validation_index'] for row in chosen}
    remainder = [row for row in all_rows if row['validation_index'] not in seen]
    random_indices = rng.choice(len(remainder), size=800, replace=False)
    chosen.extend(remainder[int(i)] for i in random_indices)
    assert len(chosen) == 2000 and len({r['validation_index'] for r in chosen}) == 2000
    atomic_json(args.work / 'candidates.json', dict(status='prepared', seed=SEED,
                selection='1200 highest mean selected routing scores, at most 6 per class; plus 800 uniformly sampled remaining validation images',
                score_threshold=THRESHOLD, source_sha256=source_sha,
                candidate_count=2000, candidates=chosen))
    print(json.dumps(dict(candidate_count=len(chosen), mean_predicted_K=float(np.mean([r['predicted_K'] for r in chosen])))))


def make_runtime(args):
    assert sha256_file(args.codec) == EXPECTED_CODEC_SHA
    config = SimpleNamespace(reconstruction_root=args.mixtok_repo / 'reconstruction', checkpoint=args.codec,
        titok_root=args.titok_root, titok_config=args.titok_root / 'configs/infer/TiTok/titok_l32.yaml',
        titok_ckpt=args.titok, llamagen_root=args.llamagen_root, llamagen_ckpt=args.llamagen)
    runtime = FrozenInputs(config, args.semantic)
    scorer = LongScorer(args.gain, args.scales, args.utility, threshold=THRESHOLD)
    return runtime, scorer


def uint8(x):
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def metric_rows(target, base, mixed, k, rows):
    target, base, mixed = [v.float().clamp(-1, 1) for v in (target, base, mixed)]
    mse_b = (target - base).square().mean((1, 2, 3)).clamp_min(1e-12)
    mse_m = (target - mixed).square().mean((1, 2, 3)).clamp_min(1e-12)
    def gradient_error(image):
        dx = image[:, :, :, 1:] - image[:, :, :, :-1]
        dy = image[:, :, 1:, :] - image[:, :, :-1, :]
        return dx, dy
    gx, gy = gradient_error(target)
    bx, by = gradient_error(base)
    mx, my = gradient_error(mixed)
    edge_b = ((gx - bx).square().mean((1, 2, 3)) + (gy - by).square().mean((1, 2, 3))) / 2
    edge_m = ((gx - mx).square().mean((1, 2, 3)) + (gy - my).square().mean((1, 2, 3))) / 2
    change = (mixed - base).abs().mean((1, 2, 3))
    out = []
    for j, row in enumerate(rows):
        psnr_gain = float(10 * torch.log10(mse_b[j] / mse_m[j]))
        edge_gain = float(10 * torch.log10(edge_b[j].clamp_min(1e-12) / edge_m[j].clamp_min(1e-12)))
        change_value = float(change[j])
        out.append(dict(validation_index=row['validation_index'], relative_path=row['relative_path'],
            class_id=row['class_id'], K=int(k[j]),
            base_PSNR_dB=float(10 * torch.log10(4 / mse_b[j])),
            mixed_PSNR_dB=float(10 * torch.log10(4 / mse_m[j])),
            PSNR_gain_dB=psnr_gain, edge_gain_dB=edge_gain,
            mean_absolute_RGB_change=change_value,
            ranking_score=psnr_gain + 0.35 * edge_gain + 4 * change_value))
    return out


def source_receipt(args):
    return {name: sha256_file(getattr(args, name)) for name in
            ('codec', 'titok', 'llamagen', 'semantic', 'gain', 'scales', 'utility')}


def score(args):
    candidate_data = json.loads((args.work / 'candidates.json').read_text())
    chosen = candidate_data['candidates'][:args.limit] if args.limit else candidate_data['candidates']
    output = args.work / ('smoke_records.jsonl' if args.limit else 'scored_records.jsonl')
    existing = records(output)
    done = {row['validation_index'] for row in existing}
    assert len(done) == len(existing)
    runtime, scorer = make_runtime(args)
    torch.set_num_threads(4)
    receipt = source_receipt(args)
    with torch.no_grad(), output.open('a') as handle:
        for start in range(0, len(chosen), args.batch):
            batch = chosen[start:start + args.batch]
            if all(row['validation_index'] in done for row in batch):
                continue
            assert not any(row['validation_index'] in done for row in batch)
            dataset = ImagePathDataset([Path(row['source_path']) for row in batch], runtime.crop)
            image = torch.stack([dataset[i] for i in range(len(dataset))]).cuda()
            z = runtime.prepare(image)
            selected = scorer.select(z)
            mixed = runtime.decode(z, selected['mask'].reshape(-1, 1, 16, 16))
            values = metric_rows(z['target'], z['base'], mixed, selected['K'], batch)
            for row in values:
                handle.write(json.dumps(row, sort_keys=True) + '\n')
            handle.flush()
            done.update(row['validation_index'] for row in values)
            atomic_json(args.work / ('smoke_progress.json' if args.limit else 'score_progress.json'),
                dict(status='scoring', completed=len(done), total=len(chosen), sources_sha256=receipt))
    audit = runtime.audit()
    assert len(done) == len(chosen)
    atomic_json(args.work / ('smoke_complete.json' if args.limit else 'score_complete.json'),
        dict(status='complete', n=len(done), records_sha256=sha256_file(output),
             sources_sha256=receipt, runtime_audit=audit))
    print(json.dumps(dict(status='complete', n=len(done))))


def panel(target, base, mixed, mask, labels=True):
    rgb = [Image.fromarray(uint8(image).permute(1, 2, 0).cpu().numpy(), 'RGB')
           for image in (target, base, mixed)]
    mask_image = Image.fromarray((mask.cpu().numpy().reshape(16, 16) * 255).astype('uint8'), 'L')
    mask_image = mask_image.resize((256, 256), Image.Resampling.NEAREST)
    overlay = Image.blend(rgb[0], Image.new('RGB', (256, 256), '#af3c1d'), 0.38)
    grid = Image.composite(overlay, rgb[0], mask_image)
    draw = ImageDraw.Draw(grid)
    for x in range(0, 257, 16):
        draw.line((x, 0, x, 256), fill=(255, 255, 255), width=1)
    for y in range(0, 257, 16):
        draw.line((0, y, 256, y), fill=(255, 255, 255), width=1)
    result = Image.new('RGB', (1024, 284 if labels else 256), 'white')
    for index, picture in enumerate((rgb[0], rgb[1], grid, rgb[2])):
        result.paste(picture, (256 * index, 28 if labels else 0))
    if labels:
        draw = ImageDraw.Draw(result)
        for index, title in enumerate(('Ground truth', '1D base', 'Selected 2D locations', 'MixTok refined')):
            draw.text((256 * index + 5, 6), title, fill='black', font=ImageFont.load_default())
    return result


def export(args):
    scored = records(args.work / 'scored_records.jsonl')
    assert len(scored) == 2000 and len({r['validation_index'] for r in scored}) == 2000
    scored.sort(key=lambda r: r['ranking_score'], reverse=True)
    selected = []
    class_count = {}
    for row in scored:
        if row['PSNR_gain_dB'] <= 0 or row['edge_gain_dB'] <= 0:
            continue
        if class_count.get(row['class_id'], 0) >= 2:
            continue
        selected.append(row)
        class_count[row['class_id']] = class_count.get(row['class_id'], 0) + 1
        if len(selected) == 200:
            break
    assert len(selected) == 200
    args.output.mkdir(parents=True, exist_ok=False)
    runtime, scorer = make_runtime(args)
    torch.set_num_threads(4)
    files = []
    with torch.no_grad():
        for start in range(0, 200, args.batch):
            batch = selected[start:start + args.batch]
            paths = [args.dataset_root / row['relative_path'] for row in batch]
            dataset = ImagePathDataset(paths, runtime.crop)
            image = torch.stack([dataset[i] for i in range(len(dataset))]).cuda()
            z = runtime.prepare(image)
            picked = scorer.select(z)
            mixed = runtime.decode(z, picked['mask'].reshape(-1, 1, 16, 16))
            verify = metric_rows(z['target'], z['base'], mixed, picked['K'], batch)
            for index, (row, check) in enumerate(zip(batch, verify)):
                assert row['K'] == check['K']
                assert abs(row['PSNR_gain_dB'] - check['PSNR_gain_dB']) < 1e-4
                filename = f"{start + index + 1:03d}_val{row['validation_index']:05d}_{row['class_id']}.png"
                final_path = args.output / filename
                panel(z['target'][index], z['base'][index], mixed[index],
                      picked['mask'][index]).save(final_path, optimize=True)
                files.append(dict(rank=start + index + 1, filename=filename,
                                  sha256=sha256_file(final_path), **row))
    atomic_json(args.output / 'manifest.json', dict(status='complete', count=200,
        figure='Figure 4 reconstruction candidate pool; Figure 5 requires separately generated samples',
        candidate_count=2000, selection_seed=SEED,
        selection='Positive PSNR and gradient-error gains; rank by PSNR gain + 0.35*gradient gain + 4*mean absolute RGB change; at most two examples per ImageNet class',
        rendering='Original 256x256 center crop; base and mixed are direct frozen-model outputs; orange locations mark selected 16x16 cells; no retouching',
        scoring_source_sha256=sha256_file(args.work / 'scored_records.jsonl'),
        assets_sha256=source_receipt(args), files=files))
    (args.output / 'README.md').write_text(
        '# Reconstruction comparison candidates\n\n'
        'These 200 ImageNet validation examples show ground truth, 1D base, selected 2D locations, '
        'and the MixTok reconstruction. They are deliberately selected for a visible 2D refinement effect, '
        'so they are qualitative examples rather than a representative random sample. '
        'The exact validation index, class, token count, per-image metrics, and PNG hash are in `manifest.json`.\n\n'
        'The outputs use the density-free Long reconstruction scorer and the released MixTok reconstruction '
        'checkpoint identified by SHA256 in `manifest.json`. Panels include center crops of the ImageNet validation images; no model weights or standalone dataset files are included.\n')
    print(json.dumps(dict(status='complete', count=len(files), first=files[0]['filename'])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'score', 'export'))
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--score-cache', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--dataset-root', type=Path)
    p.add_argument('--mixtok-repo', type=Path)
    p.add_argument('--titok-root', type=Path)
    p.add_argument('--llamagen-root', type=Path)
    for name in ('codec', 'titok', 'llamagen', 'semantic', 'gain', 'scales', 'utility'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.phase == 'prepare':
        assert args.score_cache
        prepare(args)
    elif args.phase == 'score':
        score(args)
    else:
        assert args.output and args.dataset_root
        export(args)


if __name__ == '__main__':
    main()
