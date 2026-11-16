"""Fit fresh ALIT Long gain, whitened feature statistics, and image utility."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from common import PAPER

sys.path.insert(0, str(PAPER / 'addMixtok_metrics_work'))
from mixtok_long_no_density.recon.gain_model import GainScorer


def blocks(n, step):
    for start in range(0, n, step):
        yield slice(start, min(start + step, n))


def update_moments(total, squared, count, values):
    x = np.asarray(values, dtype=np.float64)
    total += x.sum(axis=(0, 2, 3))
    squared += np.square(x).sum(axis=(0, 2, 3))
    count += x.shape[0] * x.shape[2] * x.shape[3]
    return count


class GainData(Dataset):
    def __init__(self, cache, white, indices):
        self.features = np.load(cache / 'features.npy', mmap_mode='r')
        self.labels = np.load(cache / 'gain.npy', mmap_mode='r')
        self.white = np.load(white, mmap_mode='r')
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        x = np.concatenate((self.features[i], self.white[i]), axis=0)
        return torch.from_numpy(x), torch.from_numpy(np.array(self.labels[i], copy=True))


def prepare_stats(cache, out):
    report = json.loads((cache / 'report.json').read_text())
    assert report['status'] == 'completed' and not report['validation_used']
    ids = np.load(cache / 'source_indices.npy')
    fit = np.flatnonzero(ids % 10 != 0)
    held = np.flatnonzero(ids % 10 == 0)
    assert len(ids) == 12000 and len(fit) == 10843 and len(held) == 1157
    f1 = np.load(cache / 'f1.npy', mmap_mode='r')
    f2 = np.load(cache / 'f2.npy', mmap_mode='r')
    features = np.load(cache / 'features.npy', mmap_mode='r')
    gain = np.load(cache / 'gain.npy', mmap_mode='r')
    sums = np.zeros(256); sq = np.zeros(256); count = 0
    input_sum = np.zeros(780); input_sq = np.zeros(780); input_count = 0
    scale_blocks = []
    start = time.monotonic()
    for idx in blocks(len(fit), 32):
        choice = fit[idx]
        count = update_moments(sums, sq, count, np.concatenate((f1[choice], f2[choice]), axis=0))
        input_count = update_moments(input_sum, input_sq, input_count, features[choice])
        scale_blocks.append(np.asarray(gain[choice]))
    mean = sums / count
    std = np.sqrt(np.maximum(sq / count - mean**2, 1e-12))
    np.savez(out / 'whitening.npz', mean=mean.astype('float32'), std=std.astype('float32'))
    input_mean = input_sum / input_count
    input_std = np.sqrt(np.maximum(input_sq / input_count - input_mean**2, 1e-6))
    scales = np.quantile(np.abs(np.concatenate(scale_blocks)), .75, axis=(0, 2, 3)).clip(1e-6)
    white = np.lib.format.open_memmap(out / 'white.npy', mode='w+', dtype='float16', shape=(len(ids), 1, 16, 16))
    fm = torch.tensor(mean, dtype=torch.float32, device='cuda')[None, :, None, None]
    fs = torch.tensor(std, dtype=torch.float32, device='cuda')[None, :, None, None]
    for idx in blocks(len(ids), 64):
        a = torch.from_numpy(np.asarray(f1[idx])).to('cuda', dtype=torch.float32)
        b = torch.from_numpy(np.asarray(f2[idx])).to('cuda', dtype=torch.float32)
        cos = 1 - F.cosine_similarity((a-fm)/fs, (b-fm)/fs, dim=1).unsqueeze(1)
        standardized = (cos-cos.mean((2,3), keepdim=True))/cos.std((2,3), keepdim=True, correction=0).clamp_min(1e-6)
        white[idx] = standardized.half().cpu().numpy()
    white.flush()
    np.savez(out/'input_stats.npz', mean=input_mean.astype('float32'), std=input_std.astype('float32'), scales=scales.astype('float32'))
    print(json.dumps(dict(stage='stats', fit=len(fit), held=len(held), seconds=time.monotonic()-start)), flush=True)
    return fit, held, input_mean, input_std, scales


def fit_gain(cache, out, fit, held, mean, std, scales, epochs):
    torch.manual_seed(20261116)
    model = GainScorer().cuda()
    with torch.no_grad():
        model.input_mean[0, :780, 0, 0] = torch.as_tensor(mean, device='cuda')
        model.input_std[0, :780, 0, 0] = torch.as_tensor(std, device='cuda')
    train = DataLoader(GainData(cache, out/'white.npy', fit), batch_size=64,
                       shuffle=True, num_workers=2, pin_memory=True)
    held_loader = DataLoader(GainData(cache, out/'white.npy', held), batch_size=64,
                             num_workers=2, pin_memory=True)
    scale = torch.as_tensor(scales, dtype=torch.float32, device='cuda')[None, :, None, None]
    def target(y):
        y = y / scale
        return torch.cat((torch.asinh(y[:, 0:1]), torch.asinh(y[:, 0:1] + .25*y[:, 1:2])), dim=1)
    def loss_fn(pred, y):
        absolute = F.smooth_l1_loss(pred, y)
        ranking = F.kl_div((pred.flatten(2)/.5).log_softmax(-1),
                           (y.flatten(2)/.5).softmax(-1), reduction='batchmean')*.25
        global_error = F.smooth_l1_loss(pred.mean((2,3)), y.mean((2,3)))
        return absolute + .2*ranking + .2*global_error
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best = float('inf')
    rows = []
    for epoch in range(epochs):
        model.train(); total = 0.; count = 0
        for x, y in train:
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), target(y))
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            total += float(loss.detach())*len(x); count += len(x)
        model.eval(); htotal = 0.; hcount = 0
        with torch.no_grad():
            for x,y in held_loader:
                x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
                htotal += float(loss_fn(model(x), target(y)))*len(x); hcount += len(x)
        row = dict(epoch=epoch+1, train_loss=total/count, held_loss=htotal/hcount)
        rows.append(row)
        state = dict(model=model.state_dict(), epoch=epoch+1, scales=scales.tolist(),
                     trained_on='ALIT train images and independent endpoint gain labels')
        torch.save(state, out/'gain_latest.pt')
        if row['held_loss'] < best:
            best = row['held_loss']
            torch.save(state, out/'gain_best.pt')
        schedule.step()
        (out/'gain_history.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps(dict(stage='gain', **row)), flush=True)


def fit_utility(cache, out, fit, held, scales):
    features = np.load(cache/'features.npy', mmap_mode='r')
    gain = np.load(cache/'gain.npy', mmap_mode='r')
    blocks_x = []; blocks_y = []
    for idx in blocks(len(features), 32):
        x = np.asarray(features[idx], dtype=np.float64)
        mean = x.mean((2,3)); std = x.std((2,3), ddof=0)
        blocks_x.append(np.concatenate((mean,std), axis=1))
        y = np.asarray(gain[idx], dtype=np.float64)
        rgb = y[:,0].mean((1,2)); sem = y[:,1].mean((1,2))
        base_rgb = mean[:,770]; base_sem = .25*mean[:,774]+mean[:,775]
        mixed = rgb/scales[0]+.25*sem/scales[1]
        base_mixed = base_rgb/scales[0]+.25*base_sem/scales[1]
        blocks_y.append(np.stack((rgb, rgb/(base_rgb+1e-4), mixed,
            mixed/(base_mixed+1e-3), np.maximum(y[:,0],0).mean((1,2))/(base_rgb+1e-4),
            sem/(base_sem+1e-4)), axis=1))
    x = np.concatenate(blocks_x); y = np.concatenate(blocks_y)
    mean = x[fit].mean(0); std = x[fit].std(0).clip(1e-3)
    normalized = ((x-mean)/std).clip(-8,8)
    center = normalized[fit].mean(0)
    xf = normalized[fit]-center; xh = normalized[held]-center
    tmean = y[fit].mean(0); tstd = y[fit].std(0).clip(1e-6)
    yf = (y[fit]-tmean)/tstd; yh=(y[held]-tmean)/tstd
    gram = xf.T@xf; cross = xf.T@yf
    names = ['rgb_abs','rgb_rel','mixed_abs','mixed_rel','positive_rgb_rel','semantic_rel']
    best = {}
    for strength in (.001,.01,.1,1.):
        factor = cho_factor(gram + strength*len(xf)*np.eye(xf.shape[1]), check_finite=False)
        beta = cho_solve(factor, cross, check_finite=False)
        pred = xh@beta
        mse = ((pred-yh)**2).mean(0)
        for column,name in enumerate(names):
            row = dict(strength=strength, held_mse=float(mse[column]),
                       held_spearman=float(spearmanr(pred[:,column],yh[:,column]).statistic))
            if name not in best or row['held_mse'] < best[name]['held_mse']:
                best[name] = row
                np.savez(out/(name+'.npz'), mean=mean, std=std, beta=beta[:,column],
                         bias=-center@beta[:,column], target_mean=tmean[column],
                         target_std=tstd[column], clip=8.)
    (out/'utility_report.json').write_text(json.dumps(best, indent=2)+'\n')
    print(json.dumps(dict(stage='utility', selected=best['semantic_rel'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True, exist_ok=True)
    fit, held, mean, std, scales = prepare_stats(args.cache, args.output)
    fit_gain(args.cache, args.output, fit, held, mean, std, scales, args.epochs)
    fit_utility(args.cache, args.output, fit, held, scales)
    (args.output/'report.json').write_text(json.dumps(dict(status='completed', fit=len(fit), held=len(held),
        gain_epochs=args.epochs, original_router_weights_used=False, validation_used=False), indent=2)+'\n')


if __name__=='__main__':
    main()
