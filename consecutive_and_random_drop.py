#!/usr/bin/env python3
"""
Consecutive and Random Frame Drop Tool - Standalone Script
==========================================================
Can be run directly from the package folder:
    python consecutive_and_random_drop.py --input-folder /path/to/videos --all
"""

import sys
import os
from pathlib import Path
from datetime import datetime


# Add parent directory to path so we can import package modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Now import from the YUV package
from processor import VideoFrameProcessor
import argparse


def calculate_vmaf_for_video(processor: VideoFrameProcessor, reference_video_path: str, 
                            distorted_video_path: str, args) -> bool:
    """
    Calculate VMAF for a distorted video compared to reference.
    Returns True if calculation was successful, False otherwise.
    """
    if not args.calculate_vmaf:
        return False
    
    try:
        print(f"  Calculating VMAF for {Path(distorted_video_path).name}...")
        json_path, png_path = processor.calculate_vmaf_for_output(
            reference_video_path=reference_video_path,
            distorted_video_path=distorted_video_path,
            ffmpeg_path=args.ffmpeg_path,
            vmaf_resolution=args.vmaf_resolution,
            num_threads=args.vmaf_threads
        )
        print(f"  ✓ VMAF results saved: {json_path}, {png_path}")
        return True
    except Exception as e:
        print(f"  ✗ Error calculating VMAF: {e}")
        return False


def process_single_video(video_path: str, processor: VideoFrameProcessor, args):
    """Process a single video file"""
    print(f"\n{'='*60}")
    print(f"Processing: {Path(video_path).name}")
    print(f"{'='*60}")
    
    try:
        vmaf_calculated = False  # Track if any VMAF calculations were done
        
        # Consecutive drops
        if args.consecutive_drop or args.all:
            print("\n=== CONSECUTIVE FRAME DROPPING ===")
            values = args.consecutive_drop if args.consecutive_drop else [5, 15, 30]
            for frames in values:
                expected = processor.get_output_path(video_path, "consecutive_drop", consecutive_frames=frames)
                if args.skip_existing and Path(expected).exists():
                    print(f"Skipping {frames} frames - output exists")
                    if args.calculate_vmaf and calculate_vmaf_for_video(processor, video_path, expected, args):
                        vmaf_calculated = True
                    continue
                output = processor.drop_consecutive_frames(video_path, frames)
                if output is None:
                    print("Skipping consecutive drop - no keyframes available")
                    continue
                validation = processor.validate_properties(video_path, output)
                print(f"Validation: Res={validation['resolution_match']}, FPS={validation['fps_match']}")
                
                # Calculate VMAF for processed video as soon as output is prepared
                if args.calculate_vmaf and calculate_vmaf_for_video(processor, video_path, output, args):
                    vmaf_calculated = True
        
        # Random drops
        if args.random_drop or args.all:
            print("\n=== RANDOM FRAME DROPPING ===")
            percentages = args.random_drop if args.random_drop else [30.0, 60.0, 90.0]
            for pct in percentages:
                expected = processor.get_output_path(video_path, "random_drop", drop_percentage=pct)
                if args.skip_existing and Path(expected).exists():
                    print(f"Skipping {pct}% - output exists")
                    if args.calculate_vmaf and calculate_vmaf_for_video(processor, video_path, expected, args):
                        vmaf_calculated = True
                    continue
                output = processor.drop_random_frames(video_path, pct)
                if output is None:
                    print("Skipping random drop - no keyframes available")
                    continue
                validation = processor.validate_properties(video_path, output)
                print(f"Validation: Res={validation['resolution_match']}, FPS={validation['fps_match']}")
                if args.calculate_vmaf and calculate_vmaf_for_video(processor, video_path, output, args):
                    vmaf_calculated = True

        # Clean up temporary reference YUV after all operations for this video
        if vmaf_calculated:
            processor.cleanup_temp_reference_yuv(video_path)

        print(f"✓ Completed: {Path(video_path).name}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {video_path}: {e}")
        return False


def main():
    default_output = Path("/mnt/Expansion_mnt/reimplement_600frames_cuts/framedrop_output")
    default_output.mkdir(parents=True, exist_ok=True)
    default_logs = default_output / "logs"
    default_logs.mkdir(exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_log_filename = f"frame_processing_log_{timestamp}.csv"
    
    parser = argparse.ArgumentParser(description="Consecutive and Random Frame Drop Tool")
    parser.add_argument("--input-folder", default="/mnt/Expansion_mnt/600frames_cuts/clips_yuv420p", help="Input folder containing YUV files")
    parser.add_argument("--output-dir", default=str(default_output), help="Output directory")
    parser.add_argument("--log-file", default=str(default_logs / default_log_filename), 
                       help="CSV log file (saved in logs directory with timestamp by default)")
    parser.add_argument("--width", type=int, required=False, help="YUV frame width (required for .yuv inputs)")
    parser.add_argument("--height", type=int, required=False, help="YUV frame height (required for .yuv inputs)")
    parser.add_argument("--fps", type=float, required=False, help="YUV frame rate (required for .yuv inputs)")
    parser.add_argument("--pixel-format", default="yuv420p", help="YUV pixel format (e.g., yuv420p, yuv422p)")
    parser.add_argument("--consecutive-drop", type=int, nargs="*", default=[5, 15, 25], 
                       help="Consecutive frames to drop (e.g., 5 15 30)")
    parser.add_argument("--random-drop", type=float, nargs="*", default=[15.0, 30.0, 60.0],
                       help="Random drop percentages (e.g., 30.0 60.0 90.0)")
    parser.add_argument("--all", action="store_true", help="Run all operations")
    parser.add_argument("--skip-existing", default=True, action="store_true", help="Skip existing outputs")
    parser.add_argument("--delete-consecutive-temp", action="store_true", 
                       help="Delete temporary frames directory after consecutive drop operations (default: keep it)")
    parser.add_argument("--calculate-vmaf", default=True, action="store_true", 
                       help="Calculate VMAF for processed videos (output saved to output/vmaf)")
    parser.add_argument("--ffmpeg-path", default="/mnt/Expansion_mnt/yuv_consecutive_and_random_framedrop_package_new/ffmpeg-6.0-amd64-static",
                       help="Path to ffmpeg static build directory")
    parser.add_argument("--vmaf-resolution", type=int, nargs=2, default=[3840, 2160],
                       metavar=('WIDTH', 'HEIGHT'),
                       help="Resolution for VMAF calculation (default: 3840 2160)")
    parser.add_argument("--vmaf-threads", type=int, default=8,
                       help="Number of threads for VMAF calculation (default: 8)")
    parser.add_argument("--scene-summary", type=str,
                       default="/mnt/Expansion_mnt/reimplement_600frames_cuts/keyframe_summary/keyframe_summary.json",
                       help="Path to keyframe_summary.json")
    parser.add_argument("--temp-dir", type=str, default="/mnt/Expansion_mnt/temp",
                       help="Custom temporary directory for MP4 to YUV conversion (default: system temp)")
    
    args = parser.parse_args()
    # Convert vmaf_resolution list to tuple
    args.vmaf_resolution = tuple(args.vmaf_resolution)
    
    # Ensure log file is in logs directory if it's a relative path
    log_file_path = Path(args.log_file)
    if not log_file_path.is_absolute():
        # If relative path, put it in logs directory
        args.log_file = str(default_logs / log_file_path.name)
    else:
        # If absolute path, ensure the directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    processor = VideoFrameProcessor(
        args.output_dir,
        args.log_file,
        scene_summary_path=args.scene_summary,
        delete_consecutive_temp=args.delete_consecutive_temp,
        raw_input=True,
        width=args.width,
        height=args.height,
        fps=args.fps,
        pixel_format=args.pixel_format,
        temp_dir=args.temp_dir,
        ffmpeg_path=args.ffmpeg_path
    )
    
    input_folder = Path(args.input_folder)
    video_extensions = {'.yuv', '.mp4'}
    
    # Get video files from folder
    if not input_folder.is_dir():
        print(f"Error: {input_folder} is not a directory")
        return 1
    
    video_files = [str(f) for f in input_folder.iterdir() 
                  if f.is_file() and f.suffix.lower() in video_extensions]
    
    if not video_files:
        print(f"Error: No video files found in {input_folder}")
        return 1
    
    # Validate required metadata for raw YUV inputs
    any_yuv = any(Path(v).suffix.lower() == ".yuv" for v in video_files)
    if any_yuv and (args.width is None or args.height is None or args.fps is None):
        print("Error: --width, --height, and --fps are required for .yuv inputs.")
        return 1
    
    video_files.sort()
    
    print(f"Found {len(video_files)} video file(s) to process")
    
    # Process each video file
    successful, failed = 0, 0
    for video_file in video_files:
        if process_single_video(video_file, processor, args):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {successful} successful, {failed} failed")
    print(f"Log: {args.log_file}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
