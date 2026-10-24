"""Shared constants and timeline helpers (numpy only).

Terminology
-----------
source frame   : a frame of the (compressed) video before any drop is applied
displayed frame: what the viewer sees at display tick t (after drops -> repeats)
idx            : int array, idx[t] = source frame shown at display tick t
dup[t]         : True if displayed frame t repeats displayed frame t-1
"""
import numpy as np

FPS = 60
SEG = 60                 # frames per temporal segment (1 s at 60 fps)
DUP_MEAN = 0.25 / 255.0  # repeat if mean |diff| on luma thumbs is below this ...
DUP_MAX = 4               # ... AND no pixel differs by more than this (gray levels)
SAMPLE_EVERY = 30        # spatial branch: 1 of every K *unique* frames (~2/s)
THUMB_WH = (128, 72)
LOW_WH = (256, 144)


def thumb_mad(thumbs):
    """Mean absolute luma difference between consecutive thumbs, in [0,1]."""
    th = thumbs.astype(np.float32) / 255.0
    mad = np.zeros(len(th), np.float32)
    mad[1:] = np.abs(th[1:] - th[:-1]).mean(axis=(1, 2))
    return mad


def is_dup_pair(prev, cur):
    """Repeated frame test. Mean AND max criteria: small genuine motion on a
    static background has a tiny mean difference but large local changes."""
    d = np.abs(cur.astype(np.int16) - prev.astype(np.int16))
    return d.mean() / 255.0 < DUP_MEAN and d.max() <= DUP_MAX


def dup_flags(thumbs):
    mad = thumb_mad(thumbs)
    dup = np.zeros(len(thumbs), bool)
    for t in range(1, len(thumbs)):
        dup[t] = is_dup_pair(thumbs[t - 1], thumbs[t])
    return dup, mad


def sample_mask(dup, k=SAMPLE_EVERY):
    """Which displayed frames get a (costly) spatial feature.

    Every k-th unique frame, plus the frame that is *held* at the start of
    every freeze (the image the viewer stares at). Identical logic is used
    in streaming inference and when building training samples.
    """
    T = len(dup)
    valid = np.zeros(T, bool)
    valid[0] = True
    cnt = 0
    for t in range(1, T):
        if not dup[t]:
            cnt += 1
            if cnt % k == 0:
                valid[t] = True
        elif not dup[t - 1]:
            valid[t - 1] = True
    return valid


def ffill_index(valid):
    """For every t, the last index <= t where valid is True."""
    pos = np.where(valid, np.arange(len(valid)), 0)
    return np.maximum.accumulate(pos)


def run_lengths(dup):
    """rl[t] = number of consecutive repeated frames ending at t (0 if not dup)."""
    rl = np.zeros(len(dup), np.int32)
    for t in range(1, len(dup)):
        rl[t] = rl[t - 1] + 1 if dup[t] else 0
    return rl
