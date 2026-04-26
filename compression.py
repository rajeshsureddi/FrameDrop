"""
Video Compression Module with Bitrate and HD Options
"""

import os
import subprocess
import json
import csv
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime


class VideoCompressor:
    def __init__(self, input_folder: str, output_folder: str, mode: str, 
                 scene_summary_path: str = "/mnt/Expansion_mnt/reimplement_600frames_cuts/keyframe_summary/keyframe_summary.json"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.mode = mode.lower()
        # Default bitrates in Mbps: [1.5, 6, 10] for original, [0.8, 2, 6] for hd
        if self.mode == 'original':
            self.bitrate_values = [1.5, 6, 10]  # Mbps
        else:  # hd mode
            self.bitrate_values = [0.8, 2, 6]  # Mbps
        self.video_extensions = {'.mp4'}
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.scene_summary_path = Path(scene_summary_path)
        self.original_scene_data = None
        self.compressed_video_entries = []  # Store entries for compressed videos
        self.scene_summary_output = self.output_folder / 'scene_threshold_summary_compressed.json'
    
    def get_video_info(self, video_path: Path) -> Dict:
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', str(video_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except:
            return {}
    
    def get_video_resolution(self, video_path: Path) -> Tuple[int, int]:
        info = self.get_video_info(video_path)
        if not info or 'streams' not in info:
            return None, None
        for stream in info['streams']:
            if stream.get('codec_type') == 'video':
                return int(stream.get('width', 0)), int(stream.get('height', 0))
        return None, None
    
    def get_video_bitrate(self, video_path: Path) -> int:
        info = self.get_video_info(video_path)
        if not info:
            return None
        if 'format' in info and 'bit_rate' in info['format']:
            return int(info['format']['bit_rate'])
        if 'streams' in info:
            for stream in info['streams']:
                if stream.get('codec_type') == 'video' and 'bit_rate' in stream:
                    return int(stream['bit_rate'])
        return None
    
    def is_hd_resolution(self, width: int, height: int) -> bool:
        return abs(width - 1920) <= 20 and abs(height - 1080) <= 20
    
    def load_scene_summary(self):
        """Load the original scene threshold summary JSON"""
        if self.original_scene_data is None:
            if not self.scene_summary_path.exists():
                print(f"Warning: Scene summary file not found: {self.scene_summary_path}")
                return None
            try:
                with open(self.scene_summary_path, 'r') as f:
                    self.original_scene_data = json.load(f)
                print(f"Loaded scene summary from: {self.scene_summary_path}")
            except Exception as e:
                print(f"Error loading scene summary: {e}")
                return None
        return self.original_scene_data
    
    def find_original_video_entry(self, video_name: str):
        """Find the original video entry in scene summary by video name"""
        scene_data = self.load_scene_summary()
        if not scene_data:
            return None
        
        for video in scene_data.get("videos", []):
            # Match by video_name or video_path
            if (video.get("video_name") == video_name or 
                Path(video.get("video_path", "")).name == video_name):
                return video
        return None
    
    def create_compressed_video_entry(self, original_entry: Dict, compressed_video_path: Path, 
                                     compressed_video_name: str, original_video_path: Path = None):
        """Create a new entry for compressed video with updated paths but same keyframe data"""
        # Deep copy the original entry
        compressed_entry = json.loads(json.dumps(original_entry))
        
        # Update video path and name
        compressed_entry["video_path"] = str(compressed_video_path)
        compressed_entry["video_name"] = compressed_video_name
        
        # Add original video path if provided
        if original_video_path:
            compressed_entry["original_video_path"] = str(original_video_path)
        
        # Note: We keep all keyframe information (valid_thresholds, keyframe_frames, etc.) as-is
        # since compressed videos have the same keyframes as originals
        
        return compressed_entry
    
    def save_compressed_entry_to_json(self, compressed_entry: Dict):
        """Save a compressed video entry to JSON file immediately"""
        # Read existing JSON if it exists
        if self.scene_summary_output.exists():
            try:
                with open(self.scene_summary_output, 'r') as f:
                    data = json.load(f)
                videos = data.get("videos", [])
            except (json.JSONDecodeError, FileNotFoundError):
                videos = []
        else:
            videos = []
        
        # Append new entry
        videos.append(compressed_entry)
        
        # Write back to file
        output_data = {"videos": videos}
        with open(self.scene_summary_output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.compressed_video_entries.append(compressed_entry)
    
    def compress_original(self, video_path: Path, bitrate_mbps: float) -> Path:
        bitrate_str = f"{bitrate_mbps}M"
        output_path = self.output_folder / f"{video_path.stem}_original_{bitrate_mbps}Mbps.mp4"
        cmd = ['ffmpeg', '-i', str(video_path), '-c:v', 'hevc_nvenc', '-b:v', bitrate_str,
               '-preset', 'p1', '-tune', 'ull', '-an', '-y', str(output_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except:
            return None
    
    def compress_hd_workflow(self, video_path: Path, bitrate_mbps: float) -> Path:
        orig_width, orig_height = self.get_video_resolution(video_path)
        if not orig_width or not orig_height:
            return None
        
        # Determine target resolution
        if self.is_hd_resolution(orig_width, orig_height):
            target_width, target_height = 3840, 2160
            print(f"Input is HD resolution, will upsample to 4K: {target_width}x{target_height}")
        else:
            target_width, target_height = orig_width, orig_height
            print(f"Will upsample back to original resolution: {target_width}x{target_height}")
        
        # Create temp folder and files
        bitrate_str = f"{bitrate_mbps}M"
        temp_folder = self.output_folder / f"temp_{video_path.stem}_{bitrate_mbps}Mbps"
        temp_folder.mkdir(exist_ok=True)
        hd_temp_path = temp_folder / "hd_temp.mp4"
        hd_compressed_path = temp_folder / "hd_compressed.mp4"
        final_output_path = self.output_folder / f"{video_path.stem}_hd_{bitrate_mbps}Mbps.mp4"
        
        try:
            # Step 1: Downsample to HD
            downsample_cmd = ['ffmpeg', '-i', str(video_path), '-vf', 'scale=1920:1080', '-c:v', 'hevc_nvenc',
                              '-preset', 'p1', '-tune', 'ull', '-an', '-y', str(hd_temp_path)]
            subprocess.run(downsample_cmd, check=True, capture_output=True)

            # Step 2: Compress HD video with specified bitrate
            compress_cmd = ['ffmpeg', '-i', str(hd_temp_path), '-c:v', 'hevc_nvenc', '-b:v', bitrate_str,
                            '-preset', 'p1', '-tune', 'ull', '-an', '-y', str(hd_compressed_path)]
            subprocess.run(compress_cmd, check=True, capture_output=True)

            # Step 3: Upsample to target resolution
            upsample_cmd = ['ffmpeg', '-i', str(hd_compressed_path), '-vf', f'scale={target_width}:{target_height}', '-c:v', 'hevc_nvenc',
                            '-preset', 'p1', '-tune', 'ull', '-an', '-y', str(final_output_path)]
            subprocess.run(upsample_cmd, check=True, capture_output=True)
            
            shutil.rmtree(temp_folder)
            return final_output_path
        except:
            if temp_folder.exists():
                shutil.rmtree(temp_folder)
            return None
    
    def process_videos(self):
        video_files = [f for f in self.input_folder.iterdir() if f.is_file() and f.suffix.lower() in self.video_extensions]
        if not video_files:
            print(f"Warning: No video files found in {self.input_folder}")
            return
        
        print(f"Found {len(video_files)} video files to process")
        log_data = []
        
        for video_file in video_files:
            print(f"\nProcessing: {video_file.name}")
            orig_size = video_file.stat().st_size
            orig_bitrate = self.get_video_bitrate(video_file)
            orig_width, orig_height = self.get_video_resolution(video_file)
            effective_mode = self.mode
            
            # Determine bitrate values based on effective mode
            if effective_mode == 'original':
                bitrate_values = [1.5, 6, 10]  # Mbps for original mode
            else:  # hd mode
                bitrate_values = [0.8, 2, 6]  # Mbps for HD mode
            
            # Auto-switch to HD mode if input resolution is close to HD
            if self.mode == 'original' and orig_width and orig_height and self.is_hd_resolution(orig_width, orig_height):
                effective_mode = 'hd'
                print(f"Input resolution {orig_width}x{orig_height} is close to HD, switching to HD mode")
                bitrate_values = [0.8, 2, 6]  # Mbps for HD mode
            
            for bitrate_mbps in bitrate_values:
                start_time = datetime.now()
                output_path = self.compress_original(video_file, bitrate_mbps) if effective_mode == 'original' else self.compress_hd_workflow(video_file, bitrate_mbps)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                if output_path and output_path.exists():
                    compressed_size = output_path.stat().st_size
                    compressed_bitrate = self.get_video_bitrate(output_path)
                    compression_ratio = (1 - compressed_size / orig_size) * 100 if orig_size > 0 else 0
                    
                    log_entry = {
                        'input_file': video_file.name, 'output_file': output_path.name, 'mode': effective_mode, 'target_bitrate_mbps': bitrate_mbps,
                        'original_size_mb': round(orig_size / (1024*1024), 2), 'compressed_size_mb': round(compressed_size / (1024*1024), 2),
                        'original_bitrate_kbps': round(orig_bitrate / 1000, 2) if orig_bitrate else None,
                        'compressed_bitrate_kbps': round(compressed_bitrate / 1000, 2) if compressed_bitrate else None,
                        'compression_ratio_pct': round(compression_ratio, 2), 'processing_time_sec': round(processing_time, 2),
                        'timestamp': start_time.isoformat()
                    }
                    log_data.append(log_entry)
                    
                    # Create entry for compressed video in scene summary and save immediately
                    original_entry = self.find_original_video_entry(video_file.name)
                    if original_entry:
                        compressed_entry = self.create_compressed_video_entry(
                            original_entry, output_path, output_path.name, video_file
                        )
                        self.save_compressed_entry_to_json(compressed_entry)
                        print(f"  Saved keyframe entry for compressed video: {output_path.name}")
                    else:
                        print(f"  Warning: Could not find keyframe data for original video: {video_file.name}")
                    
                    bitrate_info = f", bitrate: {compressed_bitrate/1000:.0f}kbps ({(1-compressed_bitrate/orig_bitrate)*100:.1f}% reduction)" if orig_bitrate and compressed_bitrate else ""
                    print(f"  Bitrate {bitrate_mbps}Mbps: {compressed_size / (1024*1024):.2f} MB ({compression_ratio:.1f}% reduction){bitrate_info} in {processing_time:.1f}s")
                else:
                    print(f"  Bitrate {bitrate_mbps}Mbps: Failed to compress")
        
        # Save log
        if log_data:
            log_file = self.output_folder / 'compression_log.csv'
            with open(log_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=log_data[0].keys())
                writer.writeheader()
                writer.writerows(log_data)
            print(f"\nCompression log saved to: {log_file}")
        
        # Print summary of saved entries
        if self.compressed_video_entries:
            print(f"\nScene threshold summary for compressed videos: {self.scene_summary_output}")
            print(f"Total compressed video entries saved: {len(self.compressed_video_entries)}")
        
        print(f"\nProcessing complete! Results saved to {self.output_folder}")

