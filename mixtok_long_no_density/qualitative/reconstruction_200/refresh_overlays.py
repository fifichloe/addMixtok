"""Refresh existing comparison panels and export the manually chosen shortlist.

The original red panel stores the exact binary 16x16 location mask. Recover it
cell by cell, verify its count against the saved K, then change only that panel.
The ground truth, base reconstruction, and refined reconstruction stay exact.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from mixtok_long_no_density.qualitative.reconstruction_200.overlay import (
    WHITE_OPACITY, paint_selected_grid,
)


SHORTLIST = (
    2, 3, 20, 23, 29, 32, 37, 39, 46, 48, 50, 56, 59, 61, 63, 67,
    69, 70, 74, 77, 78, 82, 83, 84, 92, 95, 96, 97, 113, 121, 127,
    130, 134, 135, 139, 140, 148, 151, 154, 155, 156, 157, 171,
    173, 174, 175, 176, 179, 180, 184, 189, 191, 192, 195, 197,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def recover_mask(original: Image.Image, k: int) -> np.ndarray:
    pixels = np.asarray(original.convert('RGB'))
    if pixels.shape != (284, 1024, 3):
        raise ValueError(f'Unexpected panel dimensions: {pixels.shape}')
    target = pixels[28:284, 0:256].astype(np.int16)
    old_grid = pixels[28:284, 512:768].astype(np.int16)
    expected_red = np.asarray(Image.blend(
        Image.fromarray(target.astype('uint8')),
        Image.new('RGB', (256, 256), '#af3c1d'), 0.38)).astype(np.int16)
    selected = np.zeros((16, 16), dtype=bool)
    for y in range(16):
        for x in range(16):
            # The first row and column contain the original white grid lines.
            ys = slice(y * 16 + 1, y * 16 + 16)
            xs = slice(x * 16 + 1, x * 16 + 16)
            region = old_grid[ys, xs]
            red_error = np.abs(region - expected_red[ys, xs]).mean()
            bare_error = np.abs(region - target[ys, xs]).mean()
            if abs(red_error - bare_error) < 1:
                raise ValueError(f'Ambiguous old cell ({x}, {y})')
            selected[y, x] = red_error < bare_error
    if int(selected.sum()) != k:
        raise ValueError(f'Recovered {selected.sum()} locations, expected K={k}')
    return selected


def contact_sheet(rows, root: Path, output: Path) -> None:
    height = ((len(rows) + 1) // 2) * 174
    sheet = Image.new('RGB', (1040, height), '#f8f8f8')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        left, top = (i % 2) * 520, (i // 2) * 174
        draw.text((left + 8, top + 8), row['filename'].removesuffix('.png'),
                  fill='#222222')
        with Image.open(root / row['filename']) as original:
            thumb = original.convert('RGB').resize((512, 142), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (left + 4, top + 30))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=91, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source, args.output
    if output.exists():
        raise FileExistsError(output)
    manifest_path = source / 'manifest.json'
    previous = json.loads(manifest_path.read_text())
    rows = previous['files']
    if previous['count'] != 200 or len(rows) != 200:
        raise ValueError('Expected the original 200-image set')
    if len(SHORTLIST) != 55 or len(set(SHORTLIST)) != 55:
        raise ValueError('Shortlist must contain exactly 55 unique ranks')

    # Preflight every source file and all location masks before creating output.
    masks = {}
    for row in rows:
        path = source / row['filename']
        if sha256(path) != row['sha256']:
            raise ValueError(f'Source SHA256 changed: {path}')
        with Image.open(path) as original:
            masks[row['rank']] = recover_mask(original, row['K'])
    if set(SHORTLIST) - set(masks):
        raise ValueError('A selected rank is missing from the source manifest')

    output.mkdir(parents=True)
    new_manifest = copy.deepcopy(previous)
    new_manifest['source_manifest_sha256'] = sha256(manifest_path)
    new_manifest['rendering'] = (
        'Original 256x256 center crop; base and mixed are exact frozen-model outputs; '
        'selected 16x16 cells have white fill at 55% opacity with thin dark outlines; '
        'only the location-visualization panel changed'
    )
    new_manifest['selected_shortlist_ranks'] = list(SHORTLIST)
    new_manifest['visualization_only_change'] = True
    for row in new_manifest['files']:
        original_path = source / row['filename']
        with Image.open(original_path) as image:
            image = image.convert('RGB')
            target = image.crop((0, 28, 256, 284))
            image.paste(paint_selected_grid(target, masks[row['rank']]), (512, 28))
            destination = output / row['filename']
            image.save(destination, optimize=True)
        row['previous_sha256'] = row['sha256']
        row['sha256'] = sha256(destination)

    sheets = output / 'contact_sheets'
    for i in range(0, len(rows), 20):
        contact_sheet(rows[i:i + 20], output, sheets / f'contact_{i // 20 + 1:02d}.jpg')
    new_manifest['contact_sheets_sha256'] = {
        path.name: sha256(path) for path in sorted(sheets.glob('*.jpg'))
    }
    write_json(output / 'manifest.json', new_manifest)

    guide = (source / 'README.md').read_text()
    guide = guide.replace('Browse all 200 examples in ten contact sheets:',
        f'The selected locations are now shown with white fill at {WHITE_OPACITY:.0%} opacity '
        'and thin dark outlines. Only the location-visualization panel changed. '
        '[Browse the 55 examples chosen by the authors](selected_55/README.md).\n\n'
        'Browse all 200 examples in ten contact sheets:')
    (output / 'README.md').write_text(guide)

    chosen = [row for row in new_manifest['files'] if row['rank'] in set(SHORTLIST)]
    if [row['rank'] for row in chosen] != list(SHORTLIST):
        raise ValueError('Shortlist ordering differs from source rank order')
    shortlisted = output / 'selected_55'
    shortlisted.mkdir()
    for row in chosen:
        shutil.copy2(output / row['filename'], shortlisted / row['filename'])
    shortlist_sheets = shortlisted / 'contact_sheets'
    for i in range(0, len(chosen), 20):
        contact_sheet(chosen[i:i + 20], shortlisted,
                      shortlist_sheets / f'contact_{i // 20 + 1:02d}.jpg')
    write_json(shortlisted / 'manifest.json', dict(
        status='complete', count=len(chosen), original_ranks=list(SHORTLIST),
        selected_from_source_manifest_sha256=new_manifest['source_manifest_sha256'],
        white_opacity=WHITE_OPACITY,
        files=[dict(rank=row['rank'], filename=row['filename'], K=row['K'],
                    PSNR_gain_dB=row['PSNR_gain_dB'], sha256=row['sha256'])
               for row in chosen],
        contact_sheets_sha256={path.name: sha256(path)
                               for path in sorted(shortlist_sheets.glob('*.jpg'))}))
    (shortlisted / 'README.md').write_text(
        '# Author-selected reconstruction examples\n\n'
        'These 55 images are the author-selected subset of the 200 reconstruction '
        'comparisons. The numbers in every filename are the original 1–200 ranks, '
        'so they match the shared selection list. Only the visualization of selected '
        '2D locations changed to white fill with thin dark outlines; the ground '
        'truth, 1D base, and MixTok refined panels remain unchanged.\n\n'
        'Contact sheets: [1–20](contact_sheets/contact_01.jpg), '
        '[21–40](contact_sheets/contact_02.jpg), '
        '[41–55](contact_sheets/contact_03.jpg).\n\n'
        f'Original ranks: {", ".join(map(str, SHORTLIST))}.\n')
    print(json.dumps(dict(status='complete', all=len(rows), shortlisted=len(chosen),
                          output=str(output))))


if __name__ == '__main__':
    main()
