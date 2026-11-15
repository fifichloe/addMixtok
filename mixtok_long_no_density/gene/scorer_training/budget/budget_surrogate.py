"""Training-only U-gradient surrogate; deployed routing stays unchanged.

The trained U head and its copied budget features are optimized. Frozen local scores set
the selected boundary; a straight-through approximation supplies U gradients
near that boundary. This is a surrogate, not an exact discrete-loss gradient.
No per-image K regression target or additional inference model is introduced.
"""
import torch


def training_mask(output, utility_weight, select_output, temperature=.25):
    score = output['score']
    with torch.no_grad():
        lo, hi = score.amin()-1, score.amax()+1
        for _ in range(24):
            midpoint = (lo+hi)/2
            count = (score >= midpoint).sum(1).clamp(48, 144).float().mean()
            lo = torch.where(count > 96, midpoint, lo)
            hi = torch.where(count > 96, hi, midpoint)
        threshold = (lo+hi)/2
        selected = select_output(dict(score=score), threshold)
        hard = selected['mask'].float()
        local = output['local'].detach()
        centered = local-local.mean(1, keepdim=True)
        scale = centered.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
        normalized = centered/scale
        inside = normalized.masked_fill(~selected['mask'], float('inf')).amin(1, keepdim=True)
        outside = normalized.masked_fill(selected['mask'], -float('inf')).amax(1, keepdim=True)
        cutoff = (inside+outside)/2
        margin = (normalized-cutoff)/temperature
    utility = output['utility']
    offset_gradient = ((utility-utility.detach())[:, None]
                       * float(utility_weight)/(scale*temperature))
    soft = (margin+offset_gradient).sigmoid()
    # Raw-score threshold density accounts for different per-image score units.
    density = (soft*(1-soft)/(scale*temperature)).detach()
    share = density/density.sum().clamp_min(1e-12)
    surrogate = soft-share*(soft.sum()-selected['K'].sum())
    mask = hard+(surrogate-surrogate.detach())
    assert torch.equal(mask.detach(), hard)
    return mask, selected['K'], threshold
