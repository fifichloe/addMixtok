#!/usr/bin/env python3
"""Evaluate the fine-tuned pure LlamaGen tokenizer on ImageNet val50k."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--llamagen-root', type=Path, required=True)
    p.add_argument('--llamagen-ckpt', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--num-images', type=int, default=50000)
    p.add_argument('--wandb-project', default='mixtok-rebuttal')
    p.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='disabled')
    args = p.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    from train_titok_llamagen_recon import add_path, load_llamagen_vq
    from eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic import (
        EvalImageDataset, psnr_sum, to_uint8,
    )
    device = torch.device('cuda:0')
    add_path(args.llamagen_root.resolve())
    from dataset.augmentation import center_crop_arr

    dataset = EvalImageDataset(str(args.data_root.resolve()), 256, center_crop_arr)
    if args.num_images < 2 or args.num_images > len(dataset):
        raise ValueError(f'num-images={args.num_images} outside [2, {len(dataset)}]')
    dataset = Subset(dataset, range(args.num_images))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    model = load_llamagen_vq(str(args.llamagen_root.resolve()),
                             str(args.llamagen_ckpt.resolve()),
                             device, 16384, 8)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint['model']
    for name, tensor in checkpoint['model_ema'].items():
        if name not in state:
            raise KeyError(f'EMA parameter missing from model state: {name}')
        state[name] = tensor
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    count = 0
    psnr_total = 0.0
    with torch.inference_mode():
        for x01, _ in loader:
            x = x01.to(device, non_blocking=True).mul_(2).sub_(1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y, _ = model(x)
            fid.update(to_uint8(x, 'minus1_1'), real=True)
            fid.update(to_uint8(y, 'minus1_1'), real=False)
            psnr_total += psnr_sum(y, x, 'minus1_1').item()
            count += x.shape[0]
            if count % 5000 == 0 or count == args.num_images:
                print(f'evaluated {count}/{args.num_images}', flush=True)
    result = {'images': count, 'rFID': fid.compute().item(), 'PSNR_dB': psnr_total / count,
              'checkpoint': str(args.checkpoint), 'model_ema': True, 'fid_feature': 2048,
              'preprocessing': 'LlamaGen center_crop_arr 256', 'codebook_size': 16384,
              'content_tokens': 256}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if args.wandb_mode != 'disabled':
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f'{args.checkpoint.parent.name}-val{count}',
                         group=args.checkpoint.parent.name,
                         mode=args.wandb_mode,
                         config={'model': 'original LlamaGen VQ-16',
                                 'checkpoint_step': int(checkpoint['step']),
                                 'images': count,
                                 'fid_feature': 2048,
                                 'preprocessing': result['preprocessing'],
                                 'model_ema': True})
        run.log({'eval/rFID': result['rFID'], 'eval/PSNR_dB': result['PSNR_dB'],
                 'eval/images': count}, step=int(checkpoint['step']))
        run.finish()


if __name__ == '__main__':
    main()
