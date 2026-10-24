"""Temporal branch: perceptual surprise in a frozen feature space.

For each displayed frame we compare the latent motion the viewer *expects*
(from a causal predictor trained on clean video only) with what is shown.

Per-frame features (columns of F):
 0 freeze_surprise : ||expected motion|| on a repeated frame (motion that should
                     have happened but did not). 0 in a static scene -> an
                     invisible freeze costs nothing.
 1 catchup_excess  : on the first new frame after a freeze, how much larger the
                     latent jump is than one normal step (the visible "skip").
 2 pred_err        : ||observed jump - gap * expected step|| (general surprise:
                     judder, irregular pacing, cuts).
 3 curvature       : 1 - cos(current step, previous step)  (trajectory bending;
                     perceptual-straightening view).
 4 dup             : repeated-frame flag.
 5 motion          : ||per-frame latent motion|| (drives temporal masking).
 6 mad             : luma thumb difference (pixel-level sanity signal).

The predictor is any object with init() -> state and
step(state, m) -> (expected_next_m, state). Passing predictor=None uses a
constant-velocity predictor, a zero-training ablation.
"""
import numpy as np
from .common import dup_flags, run_lengths, SEG

N_FRAME_FEATS = 7
N_SEG_FEATS = 12
EPS = 1e-6


class ConstantVelocity:
    def init(self):
        return None

    def step(self, state, m):
        return m, state


def per_frame_features(z, thumbs, predictor=None):
    """z: (T,D) normalized dense embeddings of DISPLAYED frames.
    thumbs: (T,h,w) uint8 luma thumbs of displayed frames."""
    predictor = predictor or ConstantVelocity()
    T, D = z.shape
    dup, mad = dup_flags(thumbs)
    F = np.zeros((T, N_FRAME_FEATS), np.float32)
    F[:, 6] = mad
    state = predictor.init()
    exp_m = np.zeros(D, np.float32)
    prev_m = np.zeros(D, np.float32)
    last_unique = z[0]
    gap = 0
    for t in range(1, T):
        if dup[t]:
            gap += 1
            F[t, 0] = np.linalg.norm(exp_m)
            F[t, 4] = 1.0
            continue
        g = gap + 1
        dz = z[t] - last_unique
        m = dz / g
        n_dz, n_exp = np.linalg.norm(dz), np.linalg.norm(exp_m)
        if gap > 0:
            F[t, 1] = max(0.0, n_dz - n_exp)
        F[t, 2] = np.linalg.norm(dz - g * exp_m)
        n_m, n_p = np.linalg.norm(m), np.linalg.norm(prev_m)
        if n_m > EPS and n_p > EPS:
            F[t, 3] = 1.0 - float(m @ prev_m) / (n_m * n_p)
        F[t, 5] = n_m
        exp_m, state = predictor.step(state, m)
        prev_m, last_unique, gap = m, z[t], 0
    return F, dup


def segment_stats(F, dup, seg=SEG):
    """Aggregate per-frame features into (S, 12) segment descriptors."""
    T = len(F)
    rl = run_lengths(dup)
    starts = dup & ~np.r_[False, dup[:-1]]
    S = max(1, T // seg)
    out = np.zeros((S, N_SEG_FEATS), np.float32)
    for s in range(S):
        a, b = s * seg, (s + 1) * seg if s < S - 1 else T
        f = F[a:b]
        nd = f[:, 4] < 0.5
        out[s] = [
            np.log1p(f[:, 0].sum()), f[:, 0].max(),
            np.log1p(f[:, 1].sum()), f[:, 1].max(),
            f[nd, 2].mean() if nd.any() else 0.0,
            np.percentile(f[nd, 2], 95) if nd.any() else 0.0,
            f[nd, 3].mean() if nd.any() else 0.0,
            f[:, 4].mean(),
            rl[a:b].max() / seg,
            starts[a:b].sum(),
            f[nd, 5].mean() if nd.any() else 0.0,
            f[:, 6].mean() * 100.0,
        ]
    return out


def gate_inputs(F, dup):
    """Per-frame inputs for the spatial temporal-masking gate:
    log freeze length so far, and the most recent real latent motion."""
    rl = run_lengths(dup)
    motion = F[:, 5].copy()
    for t in range(1, len(motion)):          # hold last real motion through freezes
        if dup[t]:
            motion[t] = motion[t - 1]
    return np.stack([np.log1p(rl), motion], 1).astype(np.float32)
