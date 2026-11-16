"""Deterministic control policies, independent of batch size and order.

Nearconstant is an offline balanced diagnostic, NOT a learned/deployable budget
policy: floor/ceil counts are assigned by source/sample ID, never image content.
"""
from functools import lru_cache
import math
import numpy as np


@lru_cache(None)
def uniform_order():
    # Same greedy farthest-point rule/tie breaks as old uniform96; extend to256.
    remaining=[(r,c) for r in range(16) for c in range(16)]
    chosen=[remaining.pop(0)]
    while remaining:
        index=max(range(len(remaining)),key=lambda i:(
            min((remaining[i][0]-r)**2+(remaining[i][1]-c)**2 for r,c in chosen),
            -remaining[i][0],-remaining[i][1]))
        chosen.append(remaining.pop(index))
    return tuple(r*16+c for r,c in chosen)


def nearconstant_k(ids,target,total,seed):
    """Exact declared rounded sum over IDs0..total-1; no batch quota or RGB.

    A bijective affine permutation assigns ceil K to a predeclared number of
    IDs. Source size is a known control-design constant, not current-batch data.
    """
    assert math.isfinite(target) and 48<=target<=144 and total>0
    ids=np.asarray(ids,dtype=np.int64)
    assert bool(((ids>=0)&(ids<total)).all())
    lo=math.floor(target);high=round((target-lo)*total)
    a=104729
    while math.gcd(a,total)!=1:a+=2
    b=(int(seed)*2654435761)%total
    permutation=(a*ids+b)%total
    return (lo+(permutation<high)).astype(np.int64)


def random_positions(ids,counts,route_seed,sample_seed,domain):
    """Uniform without replacement, separate NumPy RNG per immutable row ID."""
    domain_id={'train':0,'generation':1}[domain]
    rows=[]
    for image_id,k in zip(ids,counts):
        rng=np.random.default_rng(np.random.SeedSequence(
            [int(route_seed),domain_id,int(sample_seed),int(image_id)]))
        rows.append(np.sort(rng.choice(256,int(k),replace=False)))
    return rows


def control_identity(kind,policy,target,route_seed,train_target):
    assert kind in ('short','long') and policy in ('random','uniform','nearconstant')
    assert math.isfinite(target) and 48<=target<=144
    assert math.isfinite(train_target) and 48<=train_target<=144
    return dict(kind=kind,policy=policy,route_seed=int(route_seed),
        target_mean_K=float(target),train_target_mean_K=float(train_target),
        nearconstant_split_targets=(dict(train=float(train_target),generation=float(target))
                                   if policy=='nearconstant' else None),
        budget=('exact per-source/view or generated-image new-scorer K' if policy!='nearconstant'
            else 'content-independent affine-ID allocation of floor/ceil(split target); '
                 'exact rounded target sum per declared split, offline balanced control'),
        spatial=('same learned stable top-K ranking' if policy=='nearconstant' else
                 'per-ID NumPy uniform without replacement' if policy=='random' else
                 'prefix of fixed greedy farthest-point order, original row/col tie breaks'),
        rounding='Python round to nearest integer, ties to even',
        no_batch_quota=True,per_image_independent_given_fixed_design=True)


def select_control(policy,selection,ids,target,route_seed,sample_seed,total,domain):
    import torch
    device=selection['score'].device
    counts=(nearconstant_k(ids,target,total,route_seed+sample_seed) if policy=='nearconstant'
            else selection['K'].detach().cpu().numpy())
    counts=np.asarray(counts,dtype=np.int64)
    if policy=='nearconstant':
        order=selection['score'].argsort(dim=1,descending=True,stable=True)
        mask=torch.zeros_like(selection['mask'])
        valid=torch.arange(256,device=device)[None]<torch.as_tensor(counts,device=device)[:,None]
        mask.scatter_(1,order,valid)
    else:
        rows=(random_positions(ids,counts,route_seed,sample_seed,domain) if policy=='random'
              else [np.sort(np.asarray(uniform_order()[:int(k)])) for k in counts])
        mask=torch.zeros_like(selection['mask'])
        for row,positions in enumerate(rows):
            mask[row,torch.as_tensor(positions,device=device)]=True
    ks=torch.as_tensor(counts,device=device)
    assert torch.equal(mask.sum(1),ks)
    if policy!='nearconstant':assert torch.equal(ks,selection['K'])
    return mask,ks
