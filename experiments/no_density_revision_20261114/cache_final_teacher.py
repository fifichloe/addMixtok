"""Fresh validation scores or full TRAIN teacher supervision, atomically saved."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from final_teacher_config import HERE, OUTPUT, DATA, VAL_PATHS, KINDS, UTILITY_WEIGHT, sha, write_json, source_identities, policies
from teacher_final import TwoTeachers, ImagePathDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('validation', 'train'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--world', type=int, required=True)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--chunk', type=int, default=2048)
    args = p.parse_args()
    assert args.mode == 'train', 'Final teacher cache requires the already audited full validation component policies'
    assert args.output.resolve() == (OUTPUT/'full/cache').resolve(), 'Final labels must use their own cache'
    assert 0 <= args.rank < args.world and args.chunk % args.batch == 0
    sidecars = HERE/'final_execution'
    sidecars.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(20261113)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    frozen = source_identities()
    teacher = TwoTeachers()
    start_time = time.monotonic()
    first_audit = None
    def run(images):
        nonlocal first_audit
        target, z = teacher(images.cuda(non_blocking=True))
        if first_audit is None:
            first_audit = teacher.audit_labels(target, z)
            write_json(sidecars / f'{args.mode}_input_audit_rank{args.rank}.json', first_audit)
        assert all(torch.isfinite(t.float()).all() for t in target.values())
        return target
    def progress(**row):
        row.update(mode=args.mode, rank=args.rank, elapsed=time.monotonic()-start_time,
                   peak_GiB=torch.cuda.max_memory_allocated()/2**30)
        write_json(sidecars / f'{args.mode}_progress_rank{args.rank}.json', row)
        print(json.dumps(row), flush=True)
    if args.mode == 'validation':
        paths = VAL_PATHS.read_text().splitlines()
        assert len(paths) == len(set(paths)) == 50000
        group = np.array_split(np.arange(math.ceil(len(paths)/args.batch)), args.world)[args.rank]
        start = int(group[0])*args.batch; end = min((int(group[-1])+1)*args.batch, len(paths))
        selected = paths[start:end]
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'paths.txt').write_text('\n'.join(selected)+'\n')
        arrays = {key: np.lib.format.open_memmap(args.output/(key+'.npy'), mode='w+', dtype='float32',
                                                shape=shape)
                  for key, shape in [('local', (len(selected), 2, 256)), ('utility', (len(selected),)),
                                     ('gain', (len(selected), 2, 256)), ('white', (len(selected), 256)),
                                     ('cam', (len(selected), 256))]}
        loader = DataLoader(ImagePathDataset(list(map(Path, selected)), teacher.runtime.crop),
                            batch_size=args.batch, num_workers=4, pin_memory=True)
        done = 0
        for images in loader:
            target = run(images)
            for key, array in arrays.items(): array[done:done+len(images)] = target[key].cpu().numpy()
            done += len(images)
            if done % 320 == 0 or done == len(selected): progress(done=done, total=len(selected))
        for array in arrays.values(): array.flush()
        assert done == end - start
        write_json(args.output/'report.json', dict(status='fresh_validation_scores_complete', n=done,
            rank=args.rank, world=args.world, start=start, end=end,
            paths_sha256=sha(args.output/'paths.txt'), validation_paths_sha256=sha(VAL_PATHS),
            array_sha256={key: sha(args.output/(key+'.npy')) for key in arrays},
            source_sha256=frozen, quality_used=False, density_computed=False, **teacher.audit()))
    else:
        identity = policies()
        manifest = json.loads((DATA/'manifest.json').read_text())
        for filename, digest in manifest['files'].items(): assert sha(DATA/filename) == digest
        args.output.mkdir(parents=True, exist_ok=True)
        total = 0
        for split, seed in [('held', 20261114), ('fit', 20261113)]:
            paths = (DATA/(split+'_paths.txt')).read_text().splitlines()
            order = np.random.default_rng(seed).permutation(len(paths))
            chunks = [(i, order[i*args.chunk:(i+1)*args.chunk])
                      for i in range(math.ceil(len(order)/args.chunk))][args.rank::args.world]
            directory = args.output/split; directory.mkdir(exist_ok=True)
            assert not any((directory/f'{i:05d}.pt').exists() for i, _ in chunks), 'No silent resume'
            selected = [Path(paths[int(j)]) for _, indices in chunks for j in indices]
            loader = iter(DataLoader(ImagePathDataset(selected, teacher.runtime.crop), batch_size=args.batch,
                                    num_workers=4, pin_memory=True, persistent_workers=True))
            for chunk_id, indices in chunks:
                values = {key: [] for key in ('f_base', 'local', 'utility')}
                done = 0
                while done < len(indices):
                    images = next(loader); target = run(images)
                    assert target['f_base'].dtype == torch.bfloat16
                    for key in values: values[key].append(target[key].detach().cpu())
                    done += len(images); total += len(images)
                assert done == len(indices)
                packed = {key: torch.cat(value) for key, value in values.items()}
                packed.update(indices=torch.as_tensor(indices), split=split, policies=identity,
                              paths_sha256=manifest['files'][split+'_paths.txt'])
                path = directory/f'{chunk_id:05d}.pt'; tmp = path.with_suffix('.pt.tmp')
                torch.save(packed, tmp); tmp.replace(path)
                progress(split=split, chunk=chunk_id, images=total)
            assert sum(len(torch.load(directory/f'{i:05d}.pt', map_location='cpu', mmap=True,
                                      weights_only=False)['indices']) for i, _ in chunks) == sum(len(ix) for _, ix in chunks)
        write_json(args.output/f'complete_rank{args.rank}.json', dict(status='completed', mode='train',
            rank=args.rank, world=args.world, images=total, policies=identity,
            source_sha256=frozen, no_official_validation_images=True,
            f_base_lossless_bfloat16=True, density_computed=False, **teacher.audit()))
    assert source_identities() == frozen, 'Sources changed during extraction'


if __name__ == '__main__': main()
