"""Deterministic train-only ImageNet manifest with ten held images per class."""
from pathlib import Path
from collections import defaultdict
import hashlib, json

HERE = Path(__file__).resolve().parent
INDEX = Path('/home/heyefei/lichenge/imagenet_index.txt')
ROOT = Path('/home/heyefei/ImageNet/train')


def main():
    output = HERE/'data'
    assert not output.exists()
    groups = defaultdict(list)
    for line in INDEX.read_text().splitlines():
        path = Path(line.strip())
        if str(path).startswith(str(ROOT)+'/'):
            groups[path.parent.name].append(str(path))
    assert len(groups) == 1000
    fit, held, profile = [], [], []
    for name, paths in sorted(groups.items()):
        assert len(set(paths)) == len(paths) and len(paths) >= 20
        ordered = sorted(paths, key=lambda path: (hashlib.sha256(Path(path).name.encode()).digest(), path))
        held.extend(ordered[:10]); fit.extend(ordered[10:])
        profile.append(ordered[10])
    assert len(held) == 10000 and not set(fit) & set(held)
    output.mkdir(parents=True)
    for role, paths in [('fit', fit), ('held', held), ('profile', profile)]:
        (output/(role+'_paths.txt')).write_text('\n'.join(paths)+'\n')
    report = dict(status='prepared_train_only_ImageNet_dynamic_router_manifest', n_fit=len(fit),
        n_held=len(held), n_profile=len(profile), held_per_class=10, classes=1000,
        crop='Existing exact reconstruction-teacher ADM center crop at256; no random flip, following original scorer',
        source_index=str(INDEX), source_index_sha256=hashlib.sha256(INDEX.read_bytes()).hexdigest(),
        validation_images_used=0, inference_image_lookup=False,
        files={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.glob('*.txt')})
    (output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
