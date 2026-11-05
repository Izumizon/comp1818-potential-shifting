"""Explicit exponential reconstructions and a fixed graph-smoothing extension.

No exact recovery of the authors' normalisation is claimed. Both definitions
are retained as a prespecified sensitivity factor. Smoothing is a familiar
Laplacian operation, not a claim of a novel algorithm.
"""
import hashlib
import numpy as np
from .environment import GridSpec, model, distances


def valid_nonterminal(spec):
    mask = np.ones(spec.size**2, dtype=bool)
    mask[list(spec.walls)] = False
    mask[spec.goal] = False
    return np.flatnonzero(mask)


def exponential(spec: GridSpec, normalisation="unit", base=32.0):
    if normalisation not in ("unit", "literal") or base <= 1:
        raise ValueError("invalid exponential definition")
    rows, cols = np.divmod(np.arange(spec.size**2), spec.size)
    gr, gc = divmod(spec.goal, spec.size)
    d = np.abs(rows - gr) + np.abs(cols - gc)
    valid = valid_nonterminal(spec)
    maximum = float(max(d[valid].max(), d[spec.goal]))
    h = 1.0 - d / maximum
    phi = base ** (h - (1.0 if normalisation == "unit" else 0.0))
    phi[spec.goal] = 0.0
    phi[list(spec.walls)] = 0.0
    return phi.astype(np.float64)


def corrupt(phi, spec, fraction, rng):
    if not 0 <= fraction <= 1:
        raise ValueError("corruption fraction outside [0,1]")
    result = phi.copy()
    valid = valid_nonterminal(spec)
    count = int(np.floor(fraction * len(valid)))
    if count >= 2:
        chosen = rng.choice(valid, size=count, replace=False)
        result[chosen] = result[rng.permutation(chosen)]
    return result


def smooth(phi, spec, strength=1.0):
    """One synchronous Euler graph-diffusion step, excluding goal/walls.

    phi' = phi - strength * L phi / 4. For an undirected grid, this
    preserves the non-terminal mean and cannot expand the value range.
    It does not use BFS distances, rewards, rollouts or test outcomes.
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength outside [0,1]")
    nxt, _, _ = model(spec)
    result = phi.copy()
    valid = valid_nonterminal(spec)
    active = set(map(int, valid))
    for state in valid:
        neighbours = {int(t) for t in nxt[state] if int(t) in active and int(t) != state}
        lap = sum(phi[state] - phi[t] for t in neighbours)
        result[state] -= strength * lap / 4.0
    result[spec.goal] = 0.0
    return result


def shift(phi, spec, q_initial, gamma, intermediate_reward):
    if not 0 <= gamma < 1:
        raise ValueError("gamma outside [0,1)")
    b = (1.0 - gamma) * q_initial - intermediate_reward
    result = phi.copy()
    result[valid_nonterminal(spec)] += b / (gamma - 1.0)
    result[spec.goal] = 0.0
    result[list(spec.walls)] = 0.0
    return result, b


def describe(phi, reference, spec):
    nxt, _, _ = model(spec)
    dist = distances(spec)
    valid = valid_nonterminal(spec)
    active = set(map(int, valid))
    agrees = ties = edges = 0
    energy = 0.0
    for s in valid:
        for t in set(map(int, nxt[s])):
            if t not in active or t == s or not (np.isfinite(dist[s]) and np.isfinite(dist[t])):
                continue
            if dist[s] == dist[t]:
                continue
            edges += 1
            delta = phi[t] - phi[s]
            ties += int(abs(delta) <= 1e-12)
            agrees += int(abs(delta) > 1e-12 and np.sign(delta) == np.sign(dist[s] - dist[t]))
            energy += delta**2
    values = phi[valid]
    return {"directional_consistency": agrees / edges if edges else None,
            "tie_fraction": ties / edges if edges else None,
            "edge_energy": energy / edges if edges else None,
            "changed_fraction": float(np.mean(np.abs(values - reference[valid]) > 1e-12)),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()),
            "sha256": hashlib.sha256(phi.tobytes()).hexdigest()}
