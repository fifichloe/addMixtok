#!/usr/bin/env python3
"""Pure LlamaGen VQ fine-tuning with the MixTok reconstruction loss recipe."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

CONFIG = Path(__file__).with_name('loss_config.yaml')


def atomic_save(payload, path: Path):
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cosine_lr(initial, step, total):
    return initial * 0.5 * (1.0 + math.cos(math.pi * min(step / max(total, 1), 1.0)))


def synchronize_gradients(parameters, world):
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    for grad, reduced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        grad.copy_(reduced / world)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True,
                   help='MixTok/reconstruction at the pinned public commit')
    p.add_argument('--titok-root', type=Path, required=True,
                   help='TiTok source, needed by the projected discriminator')
    p.add_argument('--llamagen-root', type=Path, required=True)
    p.add_argument('--llamagen-ckpt', type=Path, required=True)
    p.add_argument('--dino-repo', type=Path, required=True)
    p.add_argument('--clip-cache-dir', type=Path, required=True)
    p.add_argument('--dataset-manifest', type=Path)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--force-gan', action='store_true', help='Include all losses in short throughput probes.')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--limit-samples', type=int, default=0, help='Short smoke runs only; zero uses all images.')
    p.add_argument('--wandb-project', default='mixtok-rebuttal')
    p.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    args_cli = p.parse_args()
    if args_cli.batch_size <= 0 or args_cli.accum <= 0 or args_cli.epochs <= 0:
        raise ValueError('batch size, accumulation and epochs must be positive')
    source = args_cli.source_root.resolve()
    if not (source / 'train_titok_llamagen_recon.py').is_file():
        raise FileNotFoundError(f'Invalid MixTok reconstruction source: {source}')
    sys.path.insert(0, str(source))
    from train_titok_llamagen_recon import (  # noqa: E402
        TrainableEMA, build_clip_feature_loss, build_dino_feature_loss,
        build_discriminator, build_perceptual_loss, compute_image_loss,
        compute_lecam_loss, compute_perceptual_loss,
        discriminator_feature_matching_loss, gan_g_loss_from_logits,
        hinge_d_loss, load_llamagen_vq, set_requires_grad,
        weighted_logits_mean,
    )
    from train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic import (  # noqa: E402
        build_cached_imagefolder, distributed_setup,
    )
    distributed, rank, world, local_rank, device = distributed_setup()
    main_rank = rank == 0
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    manifest = args_cli.dataset_manifest or args_cli.output / 'dataset_manifest.pt'
    config.update(
        data_path=str(args_cli.data_root.resolve()),
        output_dir=str(args_cli.output.resolve()),
        dataset_manifest=str(manifest.resolve()),
        rebuild_dataset_manifest=False,
        titok_root=str(args_cli.titok_root.resolve()),
        llamagen_root=str(args_cli.llamagen_root.resolve()),
        llamagen_ckpt=str(args_cli.llamagen_ckpt.resolve()),
        clip_cache_dir=str(args_cli.clip_cache_dir.resolve()),
        dino_repo=str(args_cli.dino_repo.resolve()),
        limit_samples=args_cli.limit_samples,
    )
    cfg = SimpleNamespace(**config)
    if main_rank:
        args_cli.output.mkdir(parents=True, exist_ok=True)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        source_commit = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                       capture_output=True, text=True, check=False)
        (args_cli.output / 'run_config.json').write_text(json.dumps({
            'source_commit': source_commit.stdout.strip() if source_commit.returncode == 0 else 'unknown',
            'model': 'original LlamaGen VQ-16, no TiTok or router',
            'training': vars(args_cli),
            'loss_config': config,
        }, indent=2, default=str))
    if distributed:
        dist.barrier()
    dataset = build_cached_imagefolder(cfg, distributed, rank)
    if args_cli.limit_samples:
        if args_cli.limit_samples > len(dataset):
            raise ValueError('--limit-samples exceeds the dataset size')
        dataset = Subset(dataset, range(args_cli.limit_samples))
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True, seed=0) if distributed else None
    loader = DataLoader(dataset, batch_size=args_cli.batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=args_cli.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args_cli.workers > 0)
    steps_per_epoch = len(loader) // args_cli.accum
    if steps_per_epoch <= 0:
        raise ValueError('Dataset is smaller than one accumulated optimizer step')
    total_steps = steps_per_epoch * args_cli.epochs
    run = None
    if main_rank and not args_cli.max_steps and args_cli.wandb_mode != 'disabled':
        import wandb
        run = wandb.init(project=args_cli.wandb_project,
                         name=args_cli.output.name,
                         dir=str(args_cli.output),
                         mode=args_cli.wandb_mode,
                         config={'batch_size_per_gpu': args_cli.batch_size,
                                 'accum': args_cli.accum, 'world_size': world,
                                 'epochs': args_cli.epochs, 'total_steps': total_steps})
    model = load_llamagen_vq(cfg.llamagen_root, cfg.llamagen_ckpt, device, cfg.codebook_size, cfg.codebook_embed_dim)
    model.requires_grad_(False)
    for name in ('encoder', 'quant_conv', 'post_quant_conv', 'decoder'):
        getattr(model, name).requires_grad_(True).train()
    trainable = [x for x in model.parameters() if x.requires_grad]
    groups = [
        {'params': list(model.encoder.parameters()) + list(model.quant_conv.parameters()),
         'lr': cfg.lr_llamagen_encoder, 'initial_lr': cfg.lr_llamagen_encoder},
        {'params': list(model.post_quant_conv.parameters()) + list(model.decoder.parameters()),
         'lr': cfg.lr_llamagen, 'initial_lr': cfg.lr_llamagen},
    ]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=cfg.weight_decay_llamagen)
    ema = TrainableEMA(model, decay=cfg.ema_decay)
    lpips = build_perceptual_loss(cfg, device)
    dino = build_dino_feature_loss(cfg, device)
    clip = build_clip_feature_loss(cfg, device)
    disc = build_discriminator(cfg, device)
    disc_trainable = [x for x in disc.parameters() if x.requires_grad]
    opt_d = torch.optim.AdamW(disc_trainable, lr=cfg.lr_d, betas=(0.5, 0.9), weight_decay=cfg.weight_decay_llamagen)
    lecam_real = torch.zeros((), device=device)
    lecam_fake = torch.zeros((), device=device)
    first_epoch = 0
    step = 0
    latest = args_cli.output / 'latest.pt'
    if latest.exists() and not args_cli.max_steps:
        ckpt = torch.load(latest, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'], strict=True)
        ema.load_state_dict(ckpt['model_ema'], device=device)
        disc.load_state_dict(ckpt['discriminator'], strict=True)
        opt.load_state_dict(ckpt['optimizer'])
        opt_d.load_state_dict(ckpt['optimizer_d'])
        first_epoch = int(ckpt['epoch']) + 1
        step = int(ckpt['step'])
        lecam_real = ckpt['lecam_real'].to(device)
        lecam_fake = ckpt['lecam_fake'].to(device)
    if distributed:
        model_ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        model_ddp = model
    if main_rank:
        print(f'dataset={len(dataset)} world={world} per_gpu_batch={args_cli.batch_size} accum={args_cli.accum} '
              f'global_batch={world * args_cli.batch_size * args_cli.accum} steps_per_epoch={steps_per_epoch} '
              f'epochs={args_cli.epochs} total_steps={total_steps} start_step={step}', flush=True)
    started = time.monotonic()
    benchmark_started = None
    for epoch in range(first_epoch, args_cli.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _epoch_step in range(steps_per_epoch):
            step += 1
            for group in opt.param_groups:
                group['lr'] = cosine_lr(group['initial_lr'], step - 1, total_steps)
            opt.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            gan_on = args_cli.force_gan or (epoch + _epoch_step / steps_per_epoch >= cfg.gan_start_epoch)
            d_warmup = gan_on and not args_cli.force_gan and (epoch + _epoch_step / steps_per_epoch < cfg.gan_start_epoch + cfg.d_warmup_epochs)
            for micro in range(args_cli.accum):
                x01, _ = next(iterator)
                x = x01.to(device, non_blocking=True).mul_(2).sub_(1)
                sync_ctx = model_ddp.no_sync() if distributed and micro + 1 < args_cli.accum else contextlib.nullcontext()
                with sync_ctx, torch.autocast('cuda', dtype=torch.bfloat16):
                    y, _ = model_ddp(x)
                    g_loss = 2.0 * (compute_image_loss(y, x, cfg.image_loss)
                                    + cfg.lambda_perceptual * compute_perceptual_loss(lpips, cfg.perceptual_loss, y, x))
                    g_loss = g_loss + cfg.lambda_dino_feat * dino(y, x, pred_range='minus1_1', target_range='minus1_1')
                    g_loss = g_loss + cfg.lambda_clip_feat * clip(y, x, pred_range='minus1_1', target_range='minus1_1')
                    fake01 = (y.float() + 1) * 0.5
                    real01 = (x.float() + 1) * 0.5
                    if gan_on and not d_warmup:
                        set_requires_grad(disc, False)
                        g_loss = g_loss + cfg.lambda_gan * gan_g_loss_from_logits(disc(fake01))
                        g_loss = g_loss + cfg.lambda_disc_feature_matching * discriminator_feature_matching_loss(disc, fake01, real01)
                (g_loss / args_cli.accum).backward()
                if gan_on:
                    set_requires_grad(disc, True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        real_logits = disc(real01.detach())
                        fake_logits = disc(fake01.detach())
                        d_loss = hinge_d_loss(real_logits, fake_logits)
                        real_mean = weighted_logits_mean(real_logits)
                        fake_mean = weighted_logits_mean(fake_logits)
                        if cfg.lecam_regularization_weight > 0:
                            d_loss = d_loss + cfg.lecam_regularization_weight * compute_lecam_loss(
                                real_mean, fake_mean, lecam_real, lecam_fake)
                        lecam_real = lecam_real * cfg.lecam_ema_decay + real_mean.detach() * (1 - cfg.lecam_ema_decay)
                        lecam_fake = lecam_fake * cfg.lecam_ema_decay + fake_mean.detach() * (1 - cfg.lecam_ema_decay)
                    (d_loss / args_cli.accum).backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            opt.step()
            ema.update(model)
            if gan_on:
                if distributed:
                    synchronize_gradients(disc_trainable, world)
                opt_d.step()
            if main_rank and (step <= 2 or step % args_cli.log_every == 0):
                d_text = f'{d_loss.item():.5f}' if gan_on else 'off'
                print(f'epoch={epoch + 1} step={step}/{total_steps} loss_g={g_loss.item():.5f} '
                      f'loss_d={d_text} elapsed_s={time.monotonic() - started:.1f}', flush=True)
                if run is not None:
                    metrics = {'train/loss_g': g_loss.item(), 'train/epoch': epoch + 1,
                               'train/progress': step / total_steps}
                    if gan_on:
                        metrics['train/loss_d'] = d_loss.item()
                    run.log(metrics, step=step)
            if args_cli.max_steps and step == 10:
                torch.cuda.synchronize()
                if distributed:
                    dist.barrier()
                benchmark_started = time.monotonic()
            if args_cli.max_steps and step >= args_cli.max_steps:
                torch.cuda.synchronize()
                if distributed:
                    dist.barrier()
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                used_bytes = max(total_bytes - free_bytes, torch.cuda.max_memory_reserved(device))
                peak = torch.tensor([used_bytes], device=device, dtype=torch.int64)
                if distributed:
                    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                if main_rank:
                    print(f'MEMORY world={world} batch={args_cli.batch_size} '
                          f'peak_used_mib={peak.item() / 2**20:.1f} '
                          f'total_mib={total_bytes / 2**20:.1f} '
                          f'fraction={peak.item() / total_bytes:.5f}', flush=True)
                if main_rank and benchmark_started is not None and step > 10:
                    seconds_per_step = (time.monotonic() - benchmark_started) / (step - 10)
                    print(f'THROUGHPUT world={world} batch={args_cli.batch_size} accum={args_cli.accum} '
                          f'seconds_per_step={seconds_per_step:.5f} '
                          f'images_per_second={world * args_cli.batch_size * args_cli.accum / seconds_per_step:.2f}', flush=True)
                if distributed:
                    dist.destroy_process_group()
                return
        if main_rank:
            payload = {
                'epoch': epoch, 'step': step, 'model': model.state_dict(),
                'model_ema': ema.state_dict(), 'discriminator': disc.state_dict(),
                'optimizer': opt.state_dict(), 'optimizer_d': opt_d.state_dict(),
                'lecam_real': lecam_real.detach().cpu(), 'lecam_fake': lecam_fake.detach().cpu(),
            }
            atomic_save(payload, latest)
            if epoch + 1 == 10:
                atomic_save(payload, args_cli.output / 'epoch_10.pt')
            print(f'completed epoch={epoch + 1} step={step} elapsed_h={(time.monotonic() - started) / 3600:.2f}', flush=True)
            if run is not None:
                run.log({'train/completed_epoch': epoch + 1}, step=step)
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
