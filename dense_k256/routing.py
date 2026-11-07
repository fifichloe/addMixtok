"""Image-independent positions shared by cache construction and evaluation."""
from functools import lru_cache

def fixed_k(policy):
    return {'uniform': 96, 'dense': 256}[policy]

@lru_cache(None)
def positions(policy):
    if policy == 'dense':
        return tuple(range(256))
    if policy != 'uniform':
        raise ValueError(policy)
    # Identical farthest-point rule to the previous uniform routing ablation.
    coords = [(r, c) for r in range(16) for c in range(16)]
    selected = [coords.pop(0)]
    while len(selected) < 96:
        i = max(range(len(coords)), key=lambda i: (
            min((coords[i][0]-r)**2+(coords[i][1]-c)**2 for r,c in selected),
            -coords[i][0], -coords[i][1]))
        selected.append(coords.pop(i))
    result = tuple(sorted(r*16+c for r,c in selected))
    assert len(result) == len(set(result)) == 96
    return result

def description(policy):
    return {'uniform': 'fixed greedy farthest-point spatial coverage, first 96 positions',
            'dense': 'all 256 spatial positions; no 1D context used by generator or final decoder'}[policy]
