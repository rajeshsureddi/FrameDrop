"""
Frame processing operations for YUV in-place frame drops.
All active operations work directly on raw YUV byte streams; no PNG extraction.
"""

import random
from pathlib import Path
from typing import List
from utils import frame_size_bytes


def process_consecutive_drop_yuv(input_path: Path, output_path: Path, keyframes_1based: List[int],
                                 consecutive_frames: int, total_frames: int, width: int,
                                 height: int, pixel_format: str, frame_size: int = None) -> tuple:
    """Apply consecutive frame drop directly on raw YUV without PNG extraction."""
    if frame_size is None:
        frame_size = frame_size_bytes(width, height, pixel_format)

    frames_to_replace = set()
    # Build replacement list per segment
    for i in range(len(keyframes_1based) - 1):
        start_kf = keyframes_1based[i]
        end_kf = keyframes_1based[i + 1]
        between_len = max(0, end_kf - start_kf - 1)
        if between_len >= consecutive_frames:
            for j in range(consecutive_frames):
                target_frame = start_kf + 1 + j
                if target_frame < end_kf:
                    frames_to_replace.add(target_frame)

    if keyframes_1based:
        last_kf = keyframes_1based[-1]
        tail_len = max(0, total_frames - last_kf)
        if tail_len >= 60:     # Ensure enough frames to drop at the end atleast on second duration
            for j in range(consecutive_frames):
                target_frame = last_kf + 1 + j
                if target_frame <= total_frames:
                    frames_to_replace.add(target_frame)

    frames_dropped = 0
    last_keyframe_bytes = None
    frames_to_replace = sorted(frames_to_replace)
    with open(input_path, "rb") as src, open(output_path, "wb") as dst:
        for frame_idx in range(1, total_frames + 1):
            frame_bytes = src.read(frame_size)
            if len(frame_bytes) != frame_size:
                raise IOError("Unexpected EOF while reading frame {}".format(frame_idx))

            if frame_idx in keyframes_1based:
                last_keyframe_bytes = frame_bytes

            if frame_idx in frames_to_replace and last_keyframe_bytes is not None:
                dst.write(last_keyframe_bytes)
                frames_dropped += 1
            else:
                dst.write(frame_bytes)

    return frames_dropped, sorted(frames_to_replace)


def process_random_drop_yuv(input_path: Path, output_path: Path, keyframes_1based: List[int],
                            drop_percentage: float, total_frames: int, width: int,
                            height: int, pixel_format: str, frame_size: int = None) -> tuple:
    """Apply random frame drop directly on raw YUV without PNG extraction."""
    if frame_size is None:
        frame_size = frame_size_bytes(width, height, pixel_format)

    frames_to_drop = set()
    # Build drop list per segment
    for i in range(len(keyframes_1based) - 1):
        start_kf = keyframes_1based[i]
        end_kf = keyframes_1based[i + 1]
        between_frames = list(range(start_kf + 1, end_kf))
        if between_frames:
            num_to_drop = int(len(between_frames) * drop_percentage / 100)
            selected = random.sample(between_frames, min(num_to_drop, len(between_frames)))
            frames_to_drop.update(selected)

    if keyframes_1based:
        last_kf = keyframes_1based[-1]
        between_frames = list(range(last_kf + 1, total_frames + 1))
        if len(between_frames) >= 60:  # Ensure enough frames to drop at the end atleast on second duration
            num_to_drop = int(len(between_frames) * drop_percentage / 100)
            selected = random.sample(between_frames, min(num_to_drop, len(between_frames)))
            frames_to_drop.update(selected)

    frames_dropped = 0
    frames_to_drop = sorted(frames_to_drop)
    last_written = None

    with open(input_path, "rb") as src, open(output_path, "wb") as dst:
        for frame_idx in range(1, total_frames + 1):
            frame_bytes = src.read(frame_size)
            if len(frame_bytes) != frame_size:
                raise IOError("Unexpected EOF while reading frame {}".format(frame_idx))

            if last_written is None:
                out_bytes = frame_bytes
            elif frame_idx in frames_to_drop:
                out_bytes = last_written
                frames_dropped += 1
            else:
                out_bytes = frame_bytes

            dst.write(out_bytes)
            last_written = out_bytes

    return frames_dropped, frames_to_drop

