"""
Consecutive and Random Frame Drop Package
=========================================
A modular package for frame extraction, consecutive/random frame dropping, HEVC encoding, and video compression.
"""

from .processor import VideoFrameProcessor
from .compression import VideoCompressor

__all__ = ['VideoFrameProcessor', 'VideoCompressor']

