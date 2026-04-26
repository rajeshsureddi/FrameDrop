"""
CSV logging functionality
"""

import csv
from typing import Dict


def init_csv(log_file: str):
    """Initialize CSV with headers - always create fresh"""
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow([
            "input_video", "output_video", "input_fps", "output_fps",
            "input_frames", "output_frames", "input_resolution", "output_resolution",
            "keyframes", "operation"
        ])


def log_operation(log_file: str, data: Dict):
    """Log operation to CSV"""
    with open(log_file, 'a', newline='') as f:
        csv.writer(f).writerow([
            data.get("input_video", ""), data.get("output_video", ""),
            data.get("input_fps", ""), data.get("output_fps", ""),
            data.get("input_frames", ""), data.get("output_frames", ""),
            data.get("input_resolution", ""), data.get("output_resolution", ""),
            data.get("keyframes", ""), data.get("operation", ""),
        ])

