"""
Main VideoFrameProcessor class
"""

import random
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional
from utils import format_resolution_fps, format_keyframes, frame_size_bytes, run_cmd, frame_count_from_filesize
from video_info import get_basic_video_info, get_video_info, SceneSummaryLoader
from csv_logging import init_csv, log_operation
from frame_operations import process_consecutive_drop_yuv, process_random_drop_yuv
from vmaf import calculate_vmaf


class VideoFrameProcessor:
    def __init__(self, output_dir: str = "output", log_file: str = "frame_processing_log.csv",
                 scene_summary_path: str = "/mnt/Expansion_mnt/reimplement_600frames_cuts/keyframe_summary/keyframe_summary.json",
                 delete_consecutive_temp: bool = False,
                 raw_input: bool = True,
                 width: int = None,
                 height: int = None,
                 fps: float = None,
                 pixel_format: str = "yuv420p",
                 temp_dir: str = None,
                 ffmpeg_path: str = "/mnt/Expansion_mnt/yuv_consecutive_and_random_framedrop_package_new/ffmpeg-6.0-amd64-static"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.raw_input = raw_input
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format
        self._frame_size_bytes = None
        self.temp_dir = Path(temp_dir) if temp_dir else None
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure log file path is absolute or relative to output directory
        log_file_path = Path(log_file)
        if not log_file_path.is_absolute():
            # If relative path, put it in output directory
            self.log_file = str(self.output_dir / log_file)
        else:
            self.log_file = log_file
        self.scene_summary_path = Path(scene_summary_path)
        self.delete_consecutive_temp = delete_consecutive_temp
        # Create consecutive_operation directory
        self.consecutive_output_dir = self.output_dir / "consecutive_operation"
        self.consecutive_output_dir.mkdir(exist_ok=True)
        
        # Create random_operation directory
        self.random_output_dir = self.output_dir / "random_operation"
        self.random_output_dir.mkdir(exist_ok=True)
        
        # Create VMAF output directory
        self.vmaf_output_dir = self.output_dir / "vmaf"
        self.vmaf_output_dir.mkdir(exist_ok=True)
        
        self.ffmpeg_path = ffmpeg_path
        self.scene_loader = SceneSummaryLoader(str(self.scene_summary_path))
        
        init_csv(self.log_file)
        
        # Cache for reference YUV paths (original_video_path -> yuv_path)
        # Used to avoid re-converting MP4 to YUV for VMAF
        self._reference_yuv_cache = {}
    
    def _get_basic_video_info(self, video_path: str, raw_input: bool = None,
                              width: int = None, height: int = None,
                              fps: float = None, pixel_format: str = None) -> Dict:
        """Get basic video info without keyframes"""
        return get_basic_video_info(
            video_path,
            raw_input=self.raw_input if raw_input is None else raw_input,
            width=self.width if width is None else width,
            height=self.height if height is None else height,
            fps=self.fps if fps is None else fps,
            pixel_format=self.pixel_format if pixel_format is None else pixel_format
        )
    
    def _get_video_info(self, video_path: str, raw_input: bool = None,
                        basic_info_override: Dict = None) -> Dict:
        """Get video info and keyframes (for input videos only)"""
        basic_info = basic_info_override or self._get_basic_video_info(video_path, raw_input=raw_input)
        return get_video_info(
            video_path,
            self.scene_loader,
            basic_info_override=basic_info,
            raw_input=self.raw_input if raw_input is None else raw_input
        )
    
    def _process_frames(self, video_path: str, operation: str, **kwargs) -> Optional[str]:
        """Generic frame processing method"""
        start_time = time.time()
        video_name = Path(video_path).stem
        is_raw = Path(video_path).suffix.lower() == ".yuv"
        info = self._get_video_info(video_path, raw_input=is_raw)
        keyframes = info.get("keyframes", [])

        if operation == "consecutive_drop":
            consecutive_frames = kwargs["consecutive_frames"]
            print(f"Processing consecutive drop: {consecutive_frames} frames")
        else:  # random_drop
            drop_percentage = kwargs["drop_percentage"]
            print(f"Processing random drop: {drop_percentage}%")

        if not keyframes:
            fallback = random.randint(100, 500)
            print(f"No keyframes in summary — using randomly sampled fallback keyframe: {fallback}")
            keyframes = [fallback]

        # keyframes are 1-based as stored in keyframe_summary.json
        keyframes_1based = list(keyframes)
        print(f"Found {len(keyframes_1based)} keyframes (1-based): {keyframes_1based}{'...' if len(keyframes_1based) > 10 else ''}")
        print(f"Total video frames: {info.get('frame_count', 0)}")

        keyframes_str = format_keyframes(keyframes)

        # Prepare input (handles mp4->yuv conversion and metadata)
        yuv_path, info, cleanup_fn = self._prepare_input(video_path, info)
        try:
            width = info.get("width", self.width)
            height = info.get("height", self.height)
            fps_val = info.get("fps", self.fps)
            pix_fmt = info.get("pixel_format", self.pixel_format)
            # Use post-prepare frame count (derived from actual YUV file size)
            total_frames = info.get("frame_count", 0)

            if not all([width, height, fps_val, pix_fmt]):
                print("Missing width/height/fps/pixel_format for processing.")
                return None

            # Generate output path with resolution and fps
            res_fps_str = format_resolution_fps(width, height, fps_val)
            if operation == "consecutive_drop":
                output_path = self.consecutive_output_dir / f"{video_name}_consecutive_drop_{consecutive_frames:02d}frames{keyframes_str}_{res_fps_str}.yuv"
            else:  # random_drop
                output_path = self.random_output_dir / f"{video_name}_random_drop_{drop_percentage:04.1f}pct{keyframes_str}_{res_fps_str}.yuv"

            frame_size = frame_size_bytes(width, height, pix_fmt)

            # Perform YUV in-place style operations
            # import pdb; pdb.set_trace()
            if operation == "consecutive_drop":
                frames_dropped, dropped_frame_indices = process_consecutive_drop_yuv(
                    Path(yuv_path),
                    output_path,
                    keyframes_1based,
                    consecutive_frames,
                    total_frames,
                    width,
                    height,
                    pix_fmt,
                    frame_size
                )
            else:
                frames_dropped, dropped_frame_indices = process_random_drop_yuv(
                    Path(yuv_path),
                    output_path,
                    keyframes_1based,
                    drop_percentage,
                    total_frames,
                    width,
                    height,
                    pix_fmt,
                    frame_size
                )

            if not output_path.exists():
                print(f"Error: Output video was not created: {output_path}")
                return None

            output_info = {
                "frame_count": total_frames,
                "fps": fps_val,
                "resolution": f"{width}x{height}"
            }
            processing_time = time.time() - start_time

            input_frame_count = info.get("frame_count", 0)
            output_frame_count = output_info.get("frame_count", 0)
            if input_frame_count != output_frame_count:
                print(f"WARNING: Frame count mismatch! Input: {input_frame_count}, Output: {output_frame_count}")
            else:
                print(f"✓ Frame count verified: {input_frame_count} frames (input) = {output_frame_count} frames (output)")

            log_operation(self.log_file, {
                "operation": operation,
                "input_video": video_path, "output_video": str(output_path),
                "input_fps": info.get("fps", ""), "output_fps": output_info.get("fps", ""),
                "input_frames": input_frame_count,
                "output_frames": output_frame_count,
                "input_resolution": info.get("resolution", ""), "output_resolution": output_info.get("resolution", ""),
                "keyframes": len(keyframes),
            })

            print(f"Created {output_path} with {frames_dropped} frames processed ({processing_time:.2f}s)")
            print(f"Total frames in video: {info.get('frame_count', 0)}, Frames processed (frozen): {frames_dropped}")
            return str(output_path)
        finally:
            cleanup_fn()
    
    def drop_consecutive_frames(self, video_path: str, consecutive_frames: int) -> Optional[str]:
        return self._process_frames(video_path, "consecutive_drop", consecutive_frames=consecutive_frames)
    
    def drop_random_frames(self, video_path: str, drop_percentage: float) -> Optional[str]:
        return self._process_frames(video_path, "random_drop", drop_percentage=drop_percentage)

    def _prepare_input(self, video_path: str, info: Dict):
        """
        Ensure we have a raw YUV path to process and accurate info.
        Returns: (path_to_process, info_dict, cleanup_fn)
        """
        suffix = Path(video_path).suffix.lower()

        # If already YUV, require metadata
        if suffix == ".yuv":
            width = info.get("width") or self.width
            height = info.get("height") or self.height
            fps_val = info.get("fps") or self.fps
            pix_fmt = info.get("pixel_format") or self.pixel_format
            if not all([width, height, fps_val, pix_fmt]):
                raise ValueError("YUV input requires width, height, fps, and pixel_format.")
            # Update frame_count using file size for accuracy
            frame_count = frame_count_from_filesize(video_path, width, height, pix_fmt)
            info = dict(info)
            info.update({
                "width": width,
                "height": height,
                "fps": fps_val,
                "pixel_format": pix_fmt,
                "frame_count": frame_count,
                "resolution": f"{width}x{height}"
            })
            return video_path, info, (lambda: None)

        # Otherwise, transcode MP4 to raw YUV temp
        # Check cache first
        if video_path in self._reference_yuv_cache:
            cached_yuv_path, cached_info = self._reference_yuv_cache[video_path]
            return cached_yuv_path, cached_info, (lambda: None)
        
        # Use custom temp directory or system default
        if self.temp_dir:
            # Use custom temp directory
            temp_dir = self.temp_dir / f"mp4_to_yuv_{int(time.time() * 1000000) % 1000000}"
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use system default temp directory
            temp_dir = Path(tempfile.mkdtemp(prefix="mp4_to_yuv_"))
        yuv_path = temp_dir / f"{Path(video_path).stem}.yuv"
        width = info.get("width")
        height = info.get("height")
        fps_val = info.get("fps")
        pix_fmt = info.get("pixel_format") or self.pixel_format or "yuv420p"

        if not all([width, height, fps_val]):
            raise ValueError("Could not determine width/height/fps from input video.")

        ffmpeg_exec = str(Path(self.ffmpeg_path) / "ffmpeg") if (Path(self.ffmpeg_path) / "ffmpeg").exists() else self.ffmpeg_path
        cmd = [
            ffmpeg_exec, "-y",
            "-i", video_path,
            "-pix_fmt", pix_fmt,
            "-vsync", "0",
            "-f", "rawvideo",
            str(yuv_path)
        ]
        run_cmd(cmd)

        frame_count = frame_count_from_filesize(str(yuv_path), width, height, pix_fmt)
        info = dict(info)
        info.update({
            "width": width,
            "height": height,
            "fps": fps_val,
            "pixel_format": pix_fmt,
            "frame_count": frame_count,
            "resolution": f"{width}x{height}"
        })

        # Store in cache for VMAF reuse
        self._reference_yuv_cache[video_path] = (str(yuv_path), info)
        
        # Store temp_dir for later cleanup
        if not hasattr(self, '_temp_dirs_to_cleanup'):
            self._temp_dirs_to_cleanup = []
        self._temp_dirs_to_cleanup.append(temp_dir)

        def cleanup():
            # Don't cleanup here - will be cleaned up after VMAF
            pass

        return str(yuv_path), info, cleanup
    
    def get_output_path(self, video_path: str, operation: str, **kwargs) -> str:
        """Generate output path — must match the filename format in _process_frames exactly."""
        video_name = Path(video_path).stem
        is_raw = Path(video_path).suffix.lower() == ".yuv"
        info = self._get_video_info(video_path, raw_input=is_raw)
        width = info.get("width")
        height = info.get("height")
        fps = info.get("fps")
        keyframes = info.get("keyframes", [])
        res_fps_str = format_resolution_fps(width, height, fps)
        keyframes_str = format_keyframes(keyframes)

        if operation == "consecutive_drop":
            return str(self.consecutive_output_dir / f"{video_name}_consecutive_drop_{kwargs['consecutive_frames']:02d}frames{keyframes_str}_{res_fps_str}.yuv")
        elif operation == "random_drop":
            return str(self.random_output_dir / f"{video_name}_random_drop_{kwargs['drop_percentage']:04.1f}pct{keyframes_str}_{res_fps_str}.yuv")
        return str(self.output_dir / f"{video_name}_{res_fps_str}_{operation}.yuv")
    
    def validate_properties(self, input_path: str, output_path: str) -> Dict:
        """Validate video properties match"""
        input_is_raw = Path(input_path).suffix.lower() == ".yuv"
        input_info = self._get_video_info(input_path, raw_input=input_is_raw)
        output_info = self._get_basic_video_info(
            output_path,
            raw_input=True,
            width=input_info.get("width"),
            height=input_info.get("height"),
            fps=input_info.get("fps"),
            pixel_format=input_info.get("pixel_format", self.pixel_format)
        )
        return {
            "resolution_match": input_info.get("resolution") == output_info.get("resolution"),
            "fps_match": abs(input_info.get("fps", 0) - output_info.get("fps", 0)) < 0.1,
            "frame_count_match": input_info.get("frame_count") == output_info.get("frame_count")
        }
    
    def calculate_vmaf_for_output(self, reference_video_path: str, distorted_video_path: str,
                                 ffmpeg_path: str = None,
                                 vmaf_resolution: tuple = (3840, 2160),
                                 num_threads: int = 8) -> tuple:
        """
        Calculate VMAF for a processed video compared to reference.
        Uses cached reference YUV if available (from _prepare_input conversion).
        
        Args:
            reference_video_path: Path to original reference video (MP4 or YUV)
            distorted_video_path: Path to processed/distorted video (should be YUV)
            ffmpeg_path: Path to ffmpeg executable (default: /scratch/09997/rs64952/FrameDrop/ffmpegEx/ffmpegEE)
            vmaf_resolution: Resolution for VMAF calculation (default: (3840, 2160))
            num_threads: Number of threads for VMAF (default: 8)
        
        Returns:
            Tuple of (json_path, png_path) for VMAF results
        """
        # Get video info to extract keyframes (from original metadata / scene summary)
        is_raw_ref = Path(reference_video_path).suffix.lower() == ".yuv"
        info = self._get_video_info(reference_video_path, raw_input=is_raw_ref)
        keyframes = info.get("keyframes", [])
        
        # Generate video name from distorted video path
        video_name = Path(distorted_video_path).stem

        # Use cached reference YUV if available (from _prepare_input)
        if reference_video_path in self._reference_yuv_cache:
            vmaf_ref_path, ref_info = self._reference_yuv_cache[reference_video_path]
            width = ref_info.get("width")
            height = ref_info.get("height")
            fps_val = ref_info.get("fps")
            pix_fmt = ref_info.get("pixel_format", self.pixel_format) or "yuv420p"
            print(f"[VMAF] Using cached reference YUV: {vmaf_ref_path}")
            print(f"[VMAF] Reference YUV metadata: {width}x{height}, {fps_val}fps, {pix_fmt}")
        else:
            ref_path = Path(reference_video_path)
            if ref_path.suffix.lower() == ".yuv":
                # Already raw — use directly
                vmaf_ref_path = reference_video_path
                ref_info = self._get_video_info(reference_video_path, raw_input=True)
                print(f"[VMAF] Using direct reference YUV: {vmaf_ref_path}")
            else:
                # MP4 not yet converted (e.g. skip-existing path) — convert now
                print(f"[VMAF] Reference MP4 not in cache, converting: {reference_video_path}")
                basic = self._get_basic_video_info(reference_video_path, raw_input=False)
                vmaf_ref_path, ref_info, _ = self._prepare_input(reference_video_path, basic)
                print(f"[VMAF] Converted reference YUV: {vmaf_ref_path}")
            width = ref_info.get("width")
            height = ref_info.get("height")
            fps_val = ref_info.get("fps")
            pix_fmt = ref_info.get("pixel_format", self.pixel_format) or "yuv420p"

        # Distorted path should already be YUV
        dist_path = Path(distorted_video_path)
        if dist_path.suffix.lower() != ".yuv":
            raise ValueError(f"Distorted video must be YUV format, got: {distorted_video_path}")
        vmaf_dist_path = distorted_video_path
        print(f"[VMAF] Distorted YUV path: {vmaf_dist_path}")
        print(f"[VMAF] Both inputs are YUV format - calculating VMAF in raw mode")

        if not all([width, height, fps_val, pix_fmt]):
            raise ValueError("Missing width/height/fps/pixel_format for VMAF calculation.")

        # Both inputs are now YUV - calculate VMAF in raw mode
        json_path, png_path = calculate_vmaf(
            reference_video_path=vmaf_ref_path,
            distorted_video_path=vmaf_dist_path,
            output_dir=self.vmaf_output_dir,
            video_name=video_name,
            keyframes=keyframes,
            ffmpeg_path=ffmpeg_path or self.ffmpeg_path,
            vmaf_resolution=vmaf_resolution,
            num_threads=num_threads,
            raw_video=True,
            input_resolution=(width, height),
            frame_rate=fps_val,
            pixel_format=pix_fmt
        )
        
        return json_path, png_path
    
    def cleanup_temp_reference_yuv(self, reference_video_path: str = None):
        """
        Clean up temporary reference YUV files after VMAF calculations are done.
        If reference_video_path is provided, only clean up that specific one.
        Otherwise, clean up all cached temp dirs.
        """
        if reference_video_path and reference_video_path in self._reference_yuv_cache:
            # Get YUV path before removing from cache
            yuv_path_str, _ = self._reference_yuv_cache[reference_video_path]
            # Remove from cache
            del self._reference_yuv_cache[reference_video_path]
            # Find and remove corresponding temp dir
            if yuv_path_str:
                yuv_path = Path(yuv_path_str)
                temp_dir = yuv_path.parent
                if temp_dir.exists() and temp_dir.name.startswith("mp4_to_yuv_"):
                    print(f"Cleaning up temporary reference YUV: {temp_dir}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
        elif hasattr(self, '_temp_dirs_to_cleanup'):
            # Clean up all temp dirs
            for temp_dir in self._temp_dirs_to_cleanup:
                if temp_dir.exists():
                    print(f"Cleaning up temporary reference YUV: {temp_dir}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
            self._temp_dirs_to_cleanup = []
            self._reference_yuv_cache = {}

