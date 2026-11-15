"""Inference controls on ONE final generation router, without provisional sets.

Calibration is offline and score-only. No stochastic selection, batch quotas,
original-router fallback, or image-to-image feedback exists in final selection.
"""
import math
import numpy as np


def calibrate_u_zero(local_scores, target_mean_K, minimum=48, maximum=144):
    assert local_scores.ndim == 2 and local_scores.shape[1] == 256
    assert local_scores.dtype == np.float32 and np.isfinite(local_scores).all()
    assert minimum <= target_mean_K <= maximum and len(local_scores) == 50000
    lo, hi = float(local_scores.min())-1., float(local_scores.max())+1.
    best = (math.inf, None, None)
    for _ in range(64):
        threshold = float(np.float32((lo+hi)/2))
        counts = np.clip((local_scores >= threshold).sum(1), minimum, maximum)
        mean = float(counts.mean())
        if abs(mean-target_mean_K) < best[0]:
            best = abs(mean-target_mean_K), threshold, mean
        if mean > target_mean_K:
            lo = threshold
        else:
            hi = threshold
    assert best[0] <= .01, ('Threshold cannot match mean within tolerance without changing policy', best)
    return dict(threshold=best[1], mean_K=best[2], target_mean_K=target_mean_K,
                absolute_mean_error=best[0], quality_used=False, input='same-seed 50000 generated base-only local scores',
                offline_score_only_calibration=True, no_batch_quota=True)


def select_output(output, policy):
    """Use a single scoring output to construct exactly one final selected set."""
    import torch
    name = policy['policy']
    assert name in ('dynamic', 'fixed', 'U_zero')
    minimum, maximum = policy['minimum_K'], policy['maximum_K']
    assert (minimum, maximum) == (48, 144)
    assert output['score'].shape == output['local'].shape and output['score'].shape[1] == 256
    score = output['local'] if name == 'U_zero' else output['score']
    assert torch.isfinite(score).all()
    if name == 'fixed':
        k = policy['fixed_K']
        assert isinstance(k, int) and minimum <= k <= maximum
        counts = torch.full((len(score),), k, dtype=torch.long, device=score.device)
    else:
        threshold = policy['threshold']
        assert math.isfinite(threshold)
        counts = (score >= threshold).sum(1).clamp(minimum, maximum)
    order = score.argsort(dim=1, descending=True, stable=True)
    mask = torch.zeros_like(score, dtype=torch.bool).scatter(
        1, order, torch.arange(256, device=score.device)[None] < counts[:, None])
    assert torch.equal(mask.sum(1), counts)
    return dict(K=counts, mask=mask, selection_score=score)
