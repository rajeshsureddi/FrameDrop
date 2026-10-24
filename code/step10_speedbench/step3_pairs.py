"""Ground-truth order for L_rank (METHOD_PLAN.md, revised 2026-10-14).

A pair is (better, worse), identical except along one axis. Only the order is
used; nothing here is ever a model input.

Rate: the encode with the larger ACTUAL file size is better. Valid only if the
  size ratio is >= 1.5x AND the nominal ladder order agrees (higher rank =
  higher target bitrate / lower CRF); identical files or disagreeing orders are
  not used. Both encodes share source, encoder settings, span, drops, tiles
  (clean ladder rungs; or the two rungs of one capture plan at one severity).
Capture severity: plan.with_severity(s) better than with_severity(s'), s' > s.
  Severities are compared in OP units (op base severity x plan multiplier):
  the largest per-op gap must be >= 0.25; if the plan contains an op that can
  look better at low strength (RISKY_OPS), that op's worse severity must be
  >= 0.6 and its gap >= 0.5.
Drops: P better than superset(P); the superset must add >= 3 dropped frames or
  a run of >= 2 consecutive added drops.
Indistinguishable: mean |A - B| < 0.5 gray levels -> pair dropped (computed on
  the 384x216 frames for video, on the 256x256 core for tiles).
"""
from __future__ import annotations

import numpy as np

RATE_MIN_RATIO = 1.5
CRF_MIN_GAP = 6
SEV_MIN_GAP = 0.25
RISKY_OPS = {"halo", "low_contrast", "white_balance", "hue_shift"}   # + saturation with k > 0
RISKY_MIN_WORSE = 0.6
RISKY_MIN_GAP = 0.5
INDIST_THRESH = 0.5
DROP_MIN_ADDED = 3
DROP_MIN_BURST = 2


def rate_order(a, b):
    """a, b: version dicts with rank, size_bytes (and crf for 2f clips).
    Returns (better, worse) or None."""
    if a["size_bytes"] is None or b["size_bytes"] is None or a["size_bytes"] == b["size_bytes"]:
        return None
    big, small = (a, b) if a["size_bytes"] > b["size_bytes"] else (b, a)
    if big["size_bytes"] < RATE_MIN_RATIO * small["size_bytes"]:
        return None
    if big.get("crf") is not None and small.get("crf") is not None:      # 2f CRF pairs
        if small["crf"] - big["crf"] < CRF_MIN_GAP:
            return None
        return big, small
    if big["rank"] is None or small["rank"] is None or big["rank"] <= small["rank"]:
        return None                       # nominal ladder order disagrees (or tie)
    return big, small


def _risky(name, base_sev, P=None):
    if name in RISKY_OPS:
        return True
    if name == "saturation":
        return P is None or P.get("k", 1.0) > 0
    return False


def severity_order(ops, s_a, s_b):
    """ops: [(name, base severity[, P])] of ONE plan; s_a, s_b: multipliers.
    Returns +1 if a is better (less severe), -1 if b is better, 0 if invalid."""
    if s_a == s_b:
        return 0
    lo, hi = min(s_a, s_b), max(s_a, s_b)
    gaps = []
    for o in ops:
        name, base = o[0], float(o[1])
        P = o[2] if len(o) > 2 else None
        e_lo, e_hi = min(1.0, base * lo), min(1.0, base * hi)
        if _risky(name, base, P) and (e_hi < RISKY_MIN_WORSE or e_hi - e_lo < RISKY_MIN_GAP):
            return 0
        gaps.append(e_hi - e_lo)
    if not gaps or max(gaps) < SEV_MIN_GAP:
        return 0
    return 1 if s_a < s_b else -1


def drops_superset_ok(p, q):
    """p, q: bool drop masks over the same displayed window, q supposed superset."""
    p, q = np.asarray(p, bool), np.asarray(q, bool)
    if (p & ~q).any():
        return False
    added = q & ~p
    if added.sum() >= DROP_MIN_ADDED:
        return True
    run = best = 0
    for v in added:
        run = run + 1 if v else 0
        best = max(best, run)
    return best >= DROP_MIN_BURST


def indistinguishable(a, b):
    """uint8 arrays of identical shape; mean abs difference in gray levels."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()
    return d < INDIST_THRESH, float(d)
