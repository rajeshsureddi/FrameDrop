"""
Utility functions for the frame drop package
"""

import subprocess
from typing import List


def run_cmd(cmd: List[str], quiet: bool = False, check: bool = True):
    """Run command"""
    if not quiet:
        print(f"[CMD] {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def format_resolution_fps(width: int, height: int, fps: float) -> str:
    """Format resolution and fps for filename (e.g., 1920x1088_60fps)"""
    fps_rounded = int(round(fps))
    return f"{width}x{height}_{fps_rounded}fps"


def format_keyframes(keyframes: List[int]) -> str:
    """Format keyframe numbers for filename (e.g., [203, 528] -> _drop_203_528).
    Keyframes are expected to be 1-based as stored in keyframe_summary.json."""
    if not keyframes:
        return ""
    keyframes_str = "_".join(str(kf) for kf in keyframes)
    return f"_drop_{keyframes_str}"


def frame_size_bytes(width: int, height: int, pixel_format: str) -> int:
    """
    Compute bytes per frame for common planar YUV pixel formats.
    Supports 8/10/12/16-bit variants of 420/422/444.
    """
    if not pixel_format:
        raise ValueError("pixel_format is required for raw YUV processing")
    fmt = pixel_format.lower()
    if "420" in fmt:
        multiplier = 1.5
    elif "422" in fmt:
        multiplier = 2.0
    elif "444" in fmt:
        multiplier = 3.0
    else:
        raise ValueError(f"Unsupported pixel_format '{pixel_format}' for frame size calculation")

    # streaming is done in 8 bit
    bytes_per_sample = 1 #streaming is done in 8 bit
    return int(width * height * multiplier * bytes_per_sample)


def frame_count_from_filesize(video_path: str, width: int, height: int, pixel_format: str) -> int:
    """Derive frame count for raw YUV using file size."""
    import os

    size = os.path.getsize(video_path)
    size_per_frame = frame_size_bytes(width, height, pixel_format)
    if size_per_frame == 0:
        return 0
    return int(size // size_per_frame)

