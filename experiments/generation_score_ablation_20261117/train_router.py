"""Retrain a base-only generation router against one frozen score ablation."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

HERE = Path(__file__).resolve().parent
MODEL_SOURCE = HERE.parent.parent / 'addMixtok_metrics_work/mixtok_long_no_density/gene/scorer_training/distill/model.py'
spec = importlib.util.spec_from_file_location('score_ablation_generation_model', MODEL_SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GenerationRouter, distillation_loss = module.GenerationRouter, module.distillation_loss

ORIGINAL_CACHE = Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/no_density_revision_20261114/final/full/cache')
BASE_CACHE = Path(os.environ.get('MIXTOK_GEN_ABL_BASE_CACHE', str(ORIGINAL_CACHE)))
TARGET_CACHE = HERE / 'cache'
POLICIES = json.loads((HERE / 'teacher_policies.json').read_text())


class JoinedShards(IterableDataset):
    def __init__(self, split, kind, batch, infinite, workers_per_gpu=2):
        self.split, self.kind, self.batch, self.infinite = split, kind, batch, infinite
        self.count = 621 if split == 'fit' else 5

    def __iter__(self):
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        epoch = 0
        while True:
            chunk_ids = list(range(self.count))
            if epoch:
                random.Random(20261113 + epoch).shuffle(chunk_ids)
            owned = chunk_ids[worker::workers]
            for chunk_id in owned:
                base_path = BASE_CACHE / self.split / f'{chunk_id:05d}.pt'
                target_path = TARGET_CACHE / f'rank{chunk_id % 4}' / self.split / f'{chunk_id:05d}.pt'
                while not (target_path.exists() and base_path.exists()):
                    time.sleep(5)
                base = torch.load(base_path, map_location='cpu', weights_only=False, mmap=False)
                target = torch.load(target_path, map_location='cpu', weights_only=False)
                assert torch.equal(base['indices'], target['indices']) and target['split'] == self.split
                assert target['source_cache'] == str(ORIGINAL_CACHE / self.split / base_path.name)
                assert target['source_cache_size_bytes'] == base_path.stat().st_size
                assert self.kind in target['targets']
                labels = target['targets'][self.kind]
                n = len(base['f_base'])
                assert labels['local'].shape == (n, 256) and labels['utility'].shape == (n,)
                order = torch.randperm(n, generator=torch.Generator().manual_seed(
                    20261113 + epoch * 99991 + chunk_id)) if self.split == 'fit' else torch.arange(n)
                for start in range(0, n, self.batch):
                    ix = order[start:start + self.batch]
                    if self.infinite and len(ix) < self.batch:
                        continue
                    yield dict(f_base=base['f_base'][ix], local=labels['local'][ix], utility=labels['utility'][ix])
            if not self.infinite:
                return
            epoch += 1


def calibrate(scores):
    assert scores.shape == (10000, 256) and np.isfinite(scores).all()
    lo, hi = float(scores.min()) - 1, float(scores.max()) + 1
    best = (math.inf, None, None)
    for _ in range(64):
        threshold = float(np.float32((lo + hi) / 2))
        mean = float(np.clip((scores >= threshold).sum(1), 48, 144).mean())
        if abs(mean - 96) < best[0]:
            best = (abs(mean - 96), threshold, mean)
        if mean > 96:
            lo = threshold
        else:
            hi = threshold
    assert best[0] < .01
    return dict(threshold=best[1], held_mean_K=best[2], target_K=96,
        source='TRAIN-held score-only', quality_used=False)


@torch.inference_mode()
def evaluate(model, loader, policy):
    model.eval()
    total = 0
    loss_sum = 0.
    score_chunks = []
    for cpu in loader:
        base = cpu['f_base'].cuda(non_blocking=True)
        local = cpu['local'].cuda(non_blocking=True)
        utility = cpu['utility'].cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(base)
        loss, _ = distillation_loss(output, local, utility,
            policy['utility_weight'], policy['threshold'])
        loss_sum += float(loss) * len(base)
        total += len(base)
        score_chunks.append(output['score'].float().cpu().numpy())
    assert total == 10000
    scores = np.concatenate(score_chunks)
    metrics = dict(n=total, held_loss=loss_sum / total)
    metrics.update(calibrate(scores))
    model.train()
    return metrics


def main(args):
    assert args.kind in POLICIES and args.steps > 0 and args.batch > 0
    torch.set_num_threads(4)
    torch.manual_seed(20261113)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = POLICIES[args.kind]
    model = GenerationRouter(policy['utility_weight'], policy['threshold']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(kind=args.kind, steps=args.steps, batch=args.batch, policy=policy,
        optimizer='AdamW lr2e-4 wd0.01', init_seed=20261113,
        label_source='frozen reconstruction teacher with explicit component removed',
        input='f_base only', MaskGIT_retrained=False,
        original_base_cache=str(BASE_CACHE), target_cache=str(TARGET_CACHE),
        model_source=str(MODEL_SOURCE))
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    train_loader = DataLoader(JoinedShards('fit', args.kind, args.batch, True),
        batch_size=None, num_workers=2, pin_memory=True, persistent_workers=True)
    held_loader = DataLoader(JoinedShards('held', args.kind, args.batch, False),
        batch_size=None, num_workers=2, pin_memory=True, persistent_workers=True)
    best = math.inf
    started = time.monotonic()
    def save(name, step, metrics):
        path = args.output / name
        temporary = path.with_suffix('.pt.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            step=step, best_held_loss=best, metrics=metrics, config=config), temporary)
        temporary.replace(path)
    step = 0
    for cpu in train_loader:
        base = cpu['f_base'].cuda(non_blocking=True)
        local = cpu['local'].cuda(non_blocking=True)
        utility = cpu['utility'].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(base)
        loss, terms = distillation_loss(output, local, utility,
            policy['utility_weight'], policy['threshold'])
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(norm) and norm > 0
        optimizer.step()
        step += 1
        if step <= 4 or step % 20 == 0:
            row = dict(step=step, loss=float(loss.detach()),
                seconds=time.monotonic() - started,
                **{key: float(value.detach()) for key, value in terms.items()})
            with (args.output / 'training.jsonl').open('a') as file:
                file.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, held_loader, policy)
            metrics['step'] = step
            with (args.output / 'held.jsonl').open('a') as file:
                file.write(json.dumps(metrics) + '\n')
            if metrics['held_loss'] < best:
                best = metrics['held_loss']
                save('best.pt', step, metrics)
            save('latest.pt', step, metrics)
            print(json.dumps(dict(stage='held', **metrics)), flush=True)
        elif step % 500 == 0:
            save('latest.pt', step, None)
        if step >= args.steps:
            break
    (args.output / 'report.json').write_text(json.dumps(dict(status='complete',
        kind=args.kind, steps=step, best_held_loss=best,
        seconds=time.monotonic() - started, MaskGIT_retrained=False), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=list(POLICIES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--eval-every', type=int, default=1000)
    main(parser.parse_args())
