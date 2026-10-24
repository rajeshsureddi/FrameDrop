"""Synthetic frame-drop patterns, expressed as display index maps.

A drop at tick t means the display repeats the previous frame while game time
keeps advancing, so after a freeze the content jumps forward. Drops are applied
AFTER encoding by re-indexing decoded frames, which means features of an encode
are computed once and reused for any number of drop patterns.

The family is deliberately broader than any single test protocol: burst
lengths 1-60, drops anywhere (optionally GOP-anchored), Bernoulli, Gilbert-
Elliott, periodic pacing judder, and mixtures.
"""
import numpy as np


def drops_to_index(drop):
    drop = np.asarray(drop, bool).copy()
    drop[0] = False
    idx = np.arange(len(drop))
    for t in range(1, len(drop)):
        if drop[t]:
            idx[t] = idx[t - 1]
    return idx


def _burst(T, rng):
    d = np.zeros(T, bool)
    for _ in range(int(rng.integers(1, 6))):
        L = int(round(np.exp(rng.uniform(0.0, np.log(60)))))
        start = int(rng.integers(1, T))          # anywhere: no keyframe anchoring
        d[start:start + L] = True
    return d


def _bernoulli(T, rng):
    p = rng.uniform(0.03, 0.7)
    d = rng.random(T) < p
    if rng.random() < 0.5:                           # only inside a window
        a = int(rng.integers(0, T - 1))
        b = int(rng.integers(a + 1, T + 1))
        w = np.zeros(T, bool)
        w[a:b] = True
        d &= w
    return d


def _gilbert(T, rng):
    p_gb, p_bg = rng.uniform(0.005, 0.15), rng.uniform(0.1, 0.9)
    loss_bad = rng.uniform(0.5, 1.0)
    d = np.zeros(T, bool)
    bad = False
    for t in range(T):
        bad = (rng.random() > p_bg) if bad else (rng.random() < p_gb)
        d[t] = bad and rng.random() < loss_bad
    return d


def _pacing(T, rng):
    m = int(rng.choice([2, 3, 4, 5, 6]))
    d = np.zeros(T, bool)
    a = int(rng.integers(0, T // 2))
    b = int(rng.integers(a + m + 1, T + 1))
    d[a:b][::m] = True
    return d


FAMILIES = {"burst": _burst, "bernoulli": _bernoulli,
            "gilbert": _gilbert, "pacing": _pacing}


def sample_pattern(T, rng):
    """Returns (drop mask, family name)."""
    r = rng.random()
    if r < 0.08:
        return np.zeros(T, bool), "none"
    if r < 0.2:
        a, b = rng.choice(list(FAMILIES), 2, replace=False)
        d = FAMILIES[a](T, rng) | FAMILIES[b](T, rng)
        name = f"{a}+{b}"
    else:
        name = str(rng.choice(list(FAMILIES)))
        d = FAMILIES[name](T, rng)
    d[0] = False
    return d, name


def superset(drop, rng):
    """A strictly-worse pattern: the same drops plus extra ones.
    Gives a ranking pair with a known order and no human labels."""
    for _ in range(10):
        extra, _ = sample_pattern(len(drop), rng)
        out = drop | extra
        out[0] = False
        if out.sum() > drop.sum():
            return out
    out = drop.copy()
    free = np.where(~out[1:])[0] + 1
    if len(free):
        out[rng.choice(free, max(1, len(free) // 10), replace=False)] = True
    return out
