#!/usr/bin/env python3
"""
Video Compression Script with Bitrate and HD Options
====================================================
Can be run directly from the package folder:
    python compress_videos.py --mode original --input_folder /path/to/videos --output_folder compressed_original/
    python compress_videos.py --mode hd --input_folder /path/to/videos --output_folder compressed_hd/
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path so we can import package modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Now import from the package
from consecutive_and_random_package.compression import VideoCompressor


def main():
    # Default output directory is in the package folder
    default_output = Path(__file__).parent / "output" / "compressed"
    
    parser = argparse.ArgumentParser(description='Compress videos with bitrate and HD options')
    parser.add_argument('--mode', choices=['original', 'hd'], required=True, help='Compression mode')
    parser.add_argument('--input_folder', required=True, help='Input folder containing videos')
    parser.add_argument('--output_folder', default=str(default_output), 
                        help=f'Output folder for compressed videos (default: {default_output})')
    parser.add_argument('--bitrate_values', nargs='+', type=float, 
                       help='Bitrate values in Mbps to use (default: [1.5, 6, 10] for original, [0.8, 2, 6] for hd)')
    parser.add_argument('--scene_summary', type=str,
                       default="/scratch/09997/rs64952/FrameDrop/video-freeze-injector/samples_json/scene_threshold_summary.json",
                       help='Path to original scene_threshold_summary.json file')
    args = parser.parse_args()
    
    if not Path(args.input_folder).exists():
        print(f"Error: Input folder does not exist: {args.input_folder}")
        return 1
    
    compressor = VideoCompressor(args.input_folder, args.output_folder, args.mode, 
                                 scene_summary_path=args.scene_summary)
    if args.bitrate_values:
        compressor.bitrate_values = args.bitrate_values
    compressor.process_videos()
    
    return 0


if __name__ == "__main__":
    exit(main())

