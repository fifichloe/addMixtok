#!/usr/bin/env python3
"""Top-p MoT Grid Scorer/Budget Controller ablations.

This evaluator loads the requested MoT checkpoint once, pre-scans validation
images to cache each image's top-p K, then evaluates spatial-rank and budget
controls with the same per-image K histogram.  It reports rFID and PSNR.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path("/home/heyefei/lichenge/MixTok-Official/reconstruction")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic as ev
from train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic import DynamicBudgetRouter
from train_titok_llamagen_recon import (
    autocast_dtype,
    convert_image_range,
    image_to_zero_one,
    load_llamagen_vq,
    load_titok,
)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--cache-k", required=True)
    p.add_argument("--num-images", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.91)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--mixed-precision", default="bf16", choices=["bf16", "fp16", "none"])
    p.add_argument("--titok-root", required=True)
    p.add_argument("--titok-config", required=True)
    p.add_argument("--titok-ckpt", required=True)
    p.add_argument("--llamagen-root", required=True)
    p.add_argument("--llamagen-ckpt", required=True)
    p.add_argument("--inception-weights", required=True)
    p.add_argument("--entropy-bins", type=int, default=32)
    p.add_argument("--skip-prepass", action="store_true")
    return p.parse_args()


def entropy_scores(target: torch.Tensor, bins: int = 32) -> torch.Tensor:
    b, _, h, w = target.shape
    with torch.autocast(device_type=target.device.type, enabled=False):
        image = target.float().add(1).mul(0.5).clamp(0, 1)
        weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        gray = (image * weights).sum(1, keepdim=True)
        kernels = image.new_tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        ]).unsqueeze(1).div(4)
        grad = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernels)
        magnitude = grad.square().sum(1).sqrt()
        ids = (magnitude * (bins / math.sqrt(2))).floor().long().clamp(0, bins - 1)
        pixels = ids.reshape(b, 16, h // 16, 16, w // 16)
        pixels = pixels.permute(0, 1, 3, 2, 4).reshape(b, 256, -1)
        hist = image.new_zeros(b, 256, bins)
        hist.scatter_add_(2, pixels, torch.ones_like(pixels, dtype=torch.float32))
        prob = hist / pixels.shape[-1]
        ent = -(prob * prob.clamp_min(1e-12).log2()).sum(-1)
    return ent


def uniform_order(device):
    coords = [(r, c) for r in range(16) for c in range(16)]
    selected = [coords.pop(0)]
    while coords:
        best_i = max(
            range(len(coords)),
            key=lambda i: (min((coords[i][0] - r) ** 2 + (coords[i][1] - c) ** 2 for r, c in selected), -coords[i][0], -coords[i][1]),
        )
        selected.append(coords.pop(best_i))
    return torch.tensor([r * 16 + c for r, c in selected], device=device, dtype=torch.long)


def topk_mask(order: torch.Tensor, k: torch.Tensor, shape):
    b, n = order.shape
    rank = torch.arange(n, device=order.device).view(1, n)
    keep = rank < k.view(-1, 1)
    flat = torch.zeros((b, n), device=order.device, dtype=torch.float32)
    flat.scatter_(1, order, keep.float())
    return flat.view(shape)


def make_model(a, device):
    ev.add_path(a.llamagen_root)
    from dataset.augmentation import center_crop_arr
    titok = load_titok(a.titok_root, a.titok_config, a.titok_ckpt, device)
    vq = load_llamagen_vq(a.llamagen_root, a.llamagen_ckpt, device, 16384, 8)
    model = ev.TiTokLlamaGenStage2(titok, vq, lg_latent_channels=256, head_channels=256,
                                   head_mode="feature", codebook_size=16384,
                                   codebook_temperature=1.0).to(device)
    model.router = DynamicBudgetRouter(latent_channels=256, hidden_dim=128, depth=3,
                                       target_ratio=0.375, min_ratio=0.0, max_ratio=1.0,
                                       detach_inputs=True).to(device)
    step, epoch, copied, threshold, ratio_ema = ev.load_trainable_params(
        model, a.ckpt, require_latent_decoder=True, use_model_ema=True
    )
    model.eval().requires_grad_(False)
    return model, center_crop_arr, {"step": step, "epoch": epoch, "copied": copied,
                                    "selector_threshold": threshold, "ratio_ema": ratio_ema}


class Stats:
    def __init__(self, n, feature_extractor, device, feature_dir):
        self.n = n
        self.feat = feature_extractor
        self.device = device
        self.feature_dir = Path(feature_dir)
        self.feature_dir.parent.mkdir(parents=True, exist_ok=True)
        self.real_path = self.feature_dir.with_name(self.feature_dir.name + "_real.dat")
        self.real = np.memmap(self.real_path, mode="w+", dtype="float32", shape=(n, 2048))
        self.fake = {}
        self.psnr = {k: 0.0 for k in ["learned_grid", "random", "uniform", "gradient_entropy", "fixed_k96", "shuffled_k"]}
        self.count = 0

    def start_fake(self, names):
        for name in names:
            path = self.feature_dir.with_name(self.feature_dir.name + f"_{name}.dat")
            self.fake[name] = np.memmap(path, mode="w+", dtype="float32", shape=(self.n, 2048))

    @torch.no_grad()
    def add_real(self, x, start):
        z = self.feat(x)
        # torch-fidelity returns a one-element feature list, whereas
        # torchmetrics NoTrainInceptionV3 returns the feature tensor itself.
        if isinstance(z, (list, tuple)):
            z = z[0]
        z = z.detach().cpu().numpy()
        self.real[start:start + len(z)] = z

    @torch.no_grad()
    def add_fake(self, name, pred, target, start):
        z = self.feat(ev.to_uint8(pred, "minus1_1"))
        if isinstance(z, (list, tuple)):
            z = z[0]
        z = z.detach().cpu().numpy()
        self.fake[name][start:start + len(z)] = z
        pred01 = image_to_zero_one(pred, "minus1_1").clamp(0, 1)
        tgt01 = image_to_zero_one(target, "minus1_1").clamp(0, 1)
        mse = (pred01 - tgt01).pow(2).flatten(1).mean(1).clamp_min(1e-8)
        self.psnr[name] += float((-10 * mse.log10()).sum().item())

    def finish(self):
        self.real.flush()
        mu_r = np.asarray(self.real).mean(0, dtype=np.float64)
        xc = np.asarray(self.real, dtype=np.float64) - mu_r
        cov_r = (xc.T @ xc) / max(self.n - 1, 1)
        out = {}
        from scipy.linalg import sqrtm
        for name, arr in self.fake.items():
            arr.flush()
            x = np.asarray(arr)
            mu = x.mean(0, dtype=np.float64)
            xc = np.asarray(x, dtype=np.float64) - mu
            cov = (xc.T @ xc) / max(self.n - 1, 1)
            prod = cov_r @ cov
            covmean = sqrtm(prod)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fid = float((mu_r - mu) @ (mu_r - mu) + np.trace(cov_r + cov - 2 * covmean))
            out[name] = {"rFID": fid, "PSNR": self.psnr[name] / self.n}
        return out


def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device)
    model, center_crop_arr, ckpt_meta = make_model(a, device)
    ds = ev.EvalImageDataset(a.data_path, a.image_size, center_crop_arr)
    n = min(a.num_images, len(ds)) if a.num_images > 0 else len(ds)
    ds = Subset(ds, range(n))
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                        pin_memory=(device.type == "cuda"), drop_last=False)
    dtype = autocast_dtype(a.mixed_precision)
    use_amp = device.type == "cuda" and a.mixed_precision != "none"
    cache = Path(a.cache_k)
    if not a.skip_prepass:
        ks = []
        with torch.no_grad():
            for x01, _ in tqdm(loader, desc="prepass_top_p", dynamic_ncols=True, disable=bool(os.environ.get("MOT_ABLATION_TQDM_DISABLE"))):
                x01 = x01.to(device, non_blocking=True)
                xt = convert_image_range(x01, "zero_1")
                target = convert_image_range(x01, "minus1_1")
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    x_base, extra = model(xt)
                    f2d, _ = ev.native_llamagen_feature(model.llamagen_vq, target, 8, allow_encoder_grad=False)
                    logits, _ = model.router(extra["f_1d_lg"], x_base, f2d)
                flat = logits.float().flatten(1)
                prob = torch.softmax(flat / a.temperature, 1)
                vals = torch.sort(prob, 1, descending=True).values
                cum_before = torch.cumsum(vals, 1) - vals
                ks.extend((cum_before < a.top_p).sum(1).cpu().numpy().astype(np.int64).tolist())
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.asarray(ks, dtype=np.int64))
    ks_all = np.load(cache if cache.suffix == ".npy" else str(cache) + ".npy")[:n]
    if len(ks_all) != n:
        raise RuntimeError(f"cached K length {len(ks_all)} != n={n}")
    rng = np.random.default_rng(a.seed)
    shuffled_all = ks_all.copy(); rng.shuffle(shuffled_all)
    order_uniform = uniform_order(device)
    names = ["learned_grid", "random", "uniform", "gradient_entropy", "fixed_k96", "shuffled_k"]
    from torchmetrics.image.fid import NoTrainInceptionV3 as FeatureExtractorInceptionV3
    feat = FeatureExtractorInceptionV3(name="inception-v3-compat", features_list=["2048"],
        feature_extractor_weights_path=a.inception_weights).to(device).eval()
    stats = Stats(n, feat, device, Path(a.output_json).with_suffix(".features"))
    stats.start_fake(names)
    cursor = 0
    perm_gen = torch.Generator(device=device); perm_gen.manual_seed(a.seed)
    with torch.no_grad():
        for x01, _ in tqdm(loader, desc="eval_ablation", dynamic_ncols=True, disable=bool(os.environ.get("MOT_ABLATION_TQDM_DISABLE"))):
            b = x01.shape[0]; x01 = x01.to(device, non_blocking=True)
            xt = convert_image_range(x01, "zero_1"); target = convert_image_range(x01, "minus1_1")
            stats.add_real(ev.to_uint8(target, "minus1_1"), cursor)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                x_base, extra = model(xt)
                f1d = extra["f_1d_lg"]
                f2d, _ = ev.native_llamagen_feature(model.llamagen_vq, target, 8, allow_encoder_grad=False)
                logits, _ = model.router(f1d, x_base, f2d)
            flat = logits.float().flatten(1)
            order = torch.argsort(flat, dim=1, descending=True, stable=True)
            k = torch.as_tensor(ks_all[cursor:cursor + b], device=device, dtype=torch.long)
            k_sh = torch.as_tensor(shuffled_all[cursor:cursor + b], device=device, dtype=torch.long)
            masks = {}
            masks["learned_grid"] = topk_mask(order, k, logits.shape)
            noise = torch.rand((b, 256), device=device, generator=perm_gen)
            masks["random"] = topk_mask(torch.argsort(noise, 1, descending=True), k, logits.shape)
            masks["uniform"] = topk_mask(order_uniform.view(1, -1).expand(b, -1), k, logits.shape)
            ent = entropy_scores(target, a.entropy_bins).flatten(1)
            masks["gradient_entropy"] = topk_mask(torch.argsort(ent, 1, descending=True), k, logits.shape)
            masks["fixed_k96"] = topk_mask(order, torch.full((b,), 96, device=device, dtype=torch.long), logits.shape)
            masks["shuffled_k"] = topk_mask(order, k_sh, logits.shape)
            for name in names:
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    pred = model.llamagen_vq.decoder((1 - masks[name].to(f1d.dtype)) * f1d + masks[name].to(f1d.dtype) * f2d)
                stats.add_fake(name, pred.float(), target.float(), cursor)
                del pred
                if device.type == "cuda": torch.cuda.empty_cache()
            cursor += b
    result = {
        "checkpoint": str(a.ckpt), "checkpoint_meta": ckpt_meta,
        "data_path": str(a.data_path), "num_images": n,
        "top_p": a.top_p, "temperature": a.temperature,
        "k_mean": float(ks_all.mean()), "k_std": float(ks_all.std()),
        "k_min": int(ks_all.min()), "k_max": int(ks_all.max()),
        "k_histogram": {str(int(k)): int((ks_all == k).sum()) for k in np.unique(ks_all)},
        "shuffled_k_seed": a.seed,
        "metrics": stats.finish(),
    }
    out = Path(a.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main(parser())
