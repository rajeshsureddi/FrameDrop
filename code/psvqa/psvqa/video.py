"""Video decoding (PyAV) and native-resolution fragment sampling.

For production real-time use, swap the CPU decoder for NVDEC and do the
resizes on GPU; the rest of the pipeline is unchanged.
"""
import av
import numpy as np
from .common import LOW_WH, THUMB_WH


def decode(path, low_wh=LOW_WH, thumb_wh=THUMB_WH):
    """Yields (t, low_rgb HxWx3 uint8, thumb_luma hxw uint8, av.VideoFrame, fps)."""
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    fps = float(stream.average_rate) if stream.average_rate else 60.0
    try:
        for t, fr in enumerate(container.decode(stream)):
            low = fr.reformat(width=low_wh[0], height=low_wh[1],
                              format="rgb24").to_ndarray()
            th = fr.reformat(width=thumb_wh[0], height=thumb_wh[1],
                             format="gray").to_ndarray()
            yield t, low, th, fr, fps
    finally:
        container.close()


def fragments(frame, grid=7, patch=32, seed=0):
    """FAST-VQA-style fragment: a grid x grid mosaic of patch x patch crops
    taken at NATIVE resolution (compression artifacts survive, cost is fixed
    at 224x224 regardless of 1080p/4K)."""
    rgb = frame.to_ndarray(format="rgb24") if not isinstance(frame, np.ndarray) else frame
    H, W, _ = rgb.shape
    rng = np.random.default_rng(seed)
    ch, cw = H // grid, W // grid
    out = np.empty((grid * patch, grid * patch, 3), np.uint8)
    for i in range(grid):
        for j in range(grid):
            y = i * ch + int(rng.integers(0, max(1, ch - patch)))
            x = j * cw + int(rng.integers(0, max(1, cw - patch)))
            out[i * patch:(i + 1) * patch, j * patch:(j + 1) * patch] = \
                rgb[y:y + patch, x:x + patch]
    return out
