"""
Video information retrieval and keyframe detection
"""

import json
from pathlib import Path
from typing import Dict, List
from utils import run_cmd, frame_count_from_filesize


def get_basic_video_info(video_path: str, raw_input: bool = False, width: int = None,
                         height: int = None, fps: float = None,
                         pixel_format: str = None) -> Dict:
    """Get basic video info; supports raw YUV when raw_input is True."""
    try:
        if not Path(video_path).exists():
            print(f"Error: Video file does not exist: {video_path}")
            return {}

        if raw_input:
            if width is None or height is None or fps is None or pixel_format is None:
                print("Missing required metadata for raw YUV input (width/height/fps/pixel_format).")
                return {}
            frame_count = frame_count_from_filesize(video_path, width, height, pixel_format)
            return {
                "frame_count": frame_count,
                "width": int(width),
                "height": int(height),
                "fps": float(fps),
                "duration": frame_count / float(fps) if fps else 0,
                "resolution": f"{width}x{height}",
                "pixel_format": pixel_format
            }

        # MP4 or other container path: rely on ffprobe
        result = run_cmd([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=duration,width,height,r_frame_rate,pix_fmt",
            "-of", "json", video_path
        ], quiet=True, check=False)

        if result.returncode != 0:
            print(f"Error running ffprobe: {result.stderr}")
            return {}

        stream = json.loads(result.stdout)["streams"][0]

        fps_str = stream.get("r_frame_rate", "0/1")
        fps_val = float(fps_str.split("/")[0]) / float(fps_str.split("/")[1]) if "/" in fps_str else float(fps_str)
        duration = float(stream.get("duration", 0))

        if duration > 0 and fps_val > 0:
            frame_count = int(round(duration * fps_val))
        else:
            frame_count = int(stream.get("nb_frames", 0))

        return {
            "frame_count": frame_count,
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "fps": fps_val,
            "duration": duration,
            "resolution": f"{stream.get('width', 0)}x{stream.get('height', 0)}",
            "pixel_format": stream.get("pix_fmt")
        }
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {}


class SceneSummaryLoader:
    """Handles loading and querying keyframe_summary.json data"""

    def __init__(self, scene_summary_path: str):
        self.scene_summary_path = Path(scene_summary_path)
        self._scene_data = None

    def get_keyframes_from_summary(self, video_path: str) -> List[int]:
        """Get 1-based keyframe indices from keyframe_summary.json.
        Returns: keyframes list or [] if not found.
        """
        if self._scene_data is None:
            if not self.scene_summary_path.exists():
                print(f"Error: Keyframe summary file not found: {self.scene_summary_path}")
                return []
            try:
                with open(self.scene_summary_path, 'r') as f:
                    self._scene_data = json.load(f)
            except Exception as e:
                print(f"Error loading keyframe summary: {e}")
                return []
        keyframe_data = self._scene_data
        
        video_name = Path(video_path).name
        video_stem = Path(video_path).stem
        
        # Find video in keyframe summary
        video_entry = None
        for video in keyframe_data.get("videos", []):
            # Match by video_name or video_path
            if (video.get("video_name") == video_name or 
                video.get("video_name") == f"{video_stem}.mp4" or
                Path(video.get("video_path", "")).name == video_name):
                video_entry = video
                break
        
        if not video_entry:
            print(f"Video not found in keyframe summary: {video_name}")
            return []
        
        # Get keyframes directly from the first valid_thresholds entry
        all_keyframes = []
        valid_thresholds = video_entry.get("valid_thresholds", [])
        if valid_thresholds:
            all_keyframes = valid_thresholds[0].get("keyframe_frames", [])
        
        if all_keyframes:
            print(f"Using keyframes: {len(all_keyframes)} keyframes")
            print(f"Keyframe indices: {all_keyframes[:10]}{'...' if len(all_keyframes) > 10 else ''}")
        else:
            print(f"No keyframes found for video: {video_name}")
        
        return all_keyframes


def get_video_info(video_path: str, scene_loader: SceneSummaryLoader,
                  basic_info_override: Dict = None, raw_input: bool = False) -> Dict:
    """Get video info and keyframes (for input videos only)"""
    try:
        if not Path(video_path).exists():
            print(f"Error: Video file does not exist: {video_path}")
            return {"keyframes": []}

        basic_info = dict(basic_info_override) if basic_info_override is not None else get_basic_video_info(video_path, raw_input=raw_input)
        if not basic_info:
            return {"keyframes": []}

        keyframes = scene_loader.get_keyframes_from_summary(video_path)

        if not keyframes:
            print("No keyframes available - skipping video")

        basic_info["keyframes"] = keyframes
        basic_info["num_keyframes"] = len(keyframes)
        return basic_info
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {"keyframes": []}

