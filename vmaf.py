"""
VMAF (Video Multi-Method Assessment Fusion) quality estimation module
"""

import subprocess
import json
import shlex
import os
from pathlib import Path
from typing import Tuple, Optional, List, Union

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


def quality_estimation_command(
    raw_video: bool = False,
    input_resolution: Tuple = None,
    frame_rate: float = None,
    pixel_format: str = None,
    reference_video_path: str = None,
    distored_video_path: str = None,
    vmaf_resolution: tuple = (3840, 2160),
    scaling_algo: str = 'lanczos',
    PSNR: bool = False,
    SSIM: bool = False,
    MS_SSIM: bool = False,
    logfile_path: str = None,
    num_threads: int = 8,
    ffmpeg_path: str = "ffmpeg"
):
    """
    Function to generate ffmpeg command for video-quality estimations. The VMAF value is calculated as default.
    Args:
        raw_video (bool): Whether input video is YUV format or not. (Default: False)
        input_resolution (tuple): The input resolution needs to provided when video format is yuv as (width, height) as frame size is not stored in the input file. (Default: None)
        frame_rate (float): The frame-rate of the video. (Default: None)
        pixel_format (str): The input video pixel format. Use command `ffmpeg -pix_fmts` to get all the options. (Default: None)
        reference_video_path (str): The path to the reference video. (Default: None)
        distored_video_path (str): The path to the distored video. (Default: None)
        vmaf_resolution(tuple): The resolution at which VMAF is calculated. (Default: (3840,2160))
        scaling_algo (str): The scaling algorithm. Refer to this link for all the options. Link: https://ffmpeg.org/ffmpeg-scaler.html#scaler_005foptions. (Default: "lanczos")
        PSNR (bool): Whether to estimate PSNR or not. (Default: False)
        SSIM (bool): Whether to estimate SSIM or not. (Default: False)
        MS_SSIM (bool): Whether to estimate MS_SSIM or not. (Default: False)
        logfile_path (str): File to logfile which contains the estimated quality values. (Default: None)
        num_threads (int): No.of threads used to run ffmpeg commands. Generally no.of threads are set between 4-8 for ffmpeg. (Default: 8)
        ffmpeg_path (str): The path to ffmpeg. (Default: "ffmpeg")
    Returns:
        cmd (str): The ffmpeg command to estimate quality of the given video.
    """
    # Display Resolution: VMAF resolution i.e VMAF model used for predicting Quality of compressed video
    # Try to find VMAF model - check if ffmpeg_path is a directory with models, or use system models
    ffmpeg_path_obj = Path(ffmpeg_path)
    if (ffmpeg_path_obj / "model").exists():
        # Custom ffmpeg with models
        if vmaf_resolution == (3840, 2160):
            vmaf_model_path = str(ffmpeg_path_obj / "model" / "vmaf_4k_v0.6.1.json")
        elif vmaf_resolution == (1920, 1080):
            vmaf_model_path = str(ffmpeg_path_obj / "model" / "vmaf_v0.6.1.json")
        else:
            assert False, 'Select a valid VMAF resolution i.e (1920,1080) or (3840,2160).'
        cmd_ffmpeg = str(ffmpeg_path_obj / "ffmpeg")
    else:
        # System ffmpeg - use system VMAF models (usually in /usr/share/ffmpeg/libvmaf/models/)
        # Or try common locations
        if vmaf_resolution == (3840, 2160):
            vmaf_model_path = "/usr/share/ffmpeg/libvmaf/models/vmaf_4k_v0.6.1.json"
        elif vmaf_resolution == (1920, 1080):
            vmaf_model_path = "/usr/share/ffmpeg/libvmaf/models/vmaf_v0.6.1.json"
        else:
            assert False, 'Select a valid VMAF resolution i.e (1920,1080) or (3840,2160).'
        cmd_ffmpeg = ffmpeg_path  # Use as-is (should be "ffmpeg" or full path)

    # Input reference and distored videos
    if raw_video:
        # Assertions: The following are necessary for rawvideo inputs.
        assert input_resolution is not None, "For a 'YUV' format video input, resolution is required are inputs."
        assert frame_rate is not None, "For a 'YUV' format video input, frame-rate is required are inputs."
        assert pixel_format is not None, "For a 'YUV' format video input, pixel_format is required are inputs."
        
        common = f"-f rawvideo -pix_fmt {pixel_format} -s {input_resolution[0]}x{input_resolution[1]} -r {frame_rate}"
        cmd_inputs = f"{common} -i {distored_video_path} {common} -i {reference_video_path}"
    else:
        cmd_inputs = f"-i {distored_video_path} -i {reference_video_path}"

    # Upscaling of reference/distored videos
    cmd_upscale = '-lavfi "[0:v]scale={}x{}:flags={}[distorted]; [1:v]scale={}x{}:flags={}[reference];[distorted][reference]'.format(
        vmaf_resolution[0], vmaf_resolution[1], scaling_algo, vmaf_resolution[0], vmaf_resolution[1], scaling_algo)
    
    # Multi-Threading
    if num_threads is not None:
        cmd_quality = 'libvmaf=log_fmt=json:log_path={}:model_path={}:n_threads={}:psnr={}:ssim={}:ms_ssim={}" -f null -'.format(
            logfile_path, vmaf_model_path, num_threads, int(PSNR), int(SSIM), int(MS_SSIM))
    else:
        cmd_quality = 'libvmaf=log_fmt=json:log_path={}:model_path={}:psnr={}:ssim={}:ms_ssim={}" -f null -'.format(
            logfile_path, vmaf_model_path, int(PSNR), int(SSIM), int(MS_SSIM))

    # Final Command
    cmd_prefix = "/usr/bin/time"
    cmd = " ".join([cmd_prefix, cmd_ffmpeg, cmd_inputs, cmd_upscale, cmd_quality])

    return cmd


def plot_vmaf_data(json_file_path: str, output_png_path: Optional[str] = None, keyframes_list: Optional[Union[List, str]] = None):
    """
    Read VMAF JSON file and plot frame number vs VMAF score.
    Keyframes are 1-based as stored in keyframe_summary.json.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Skipping VMAF plot.")
        return

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    frame_numbers = [frame['frameNum'] for frame in data['frames']]
    vmaf_values   = [frame['metrics']['vmaf'] for frame in data['frames']]

    keyframes = list(keyframes_list) if (keyframes_list is not None and not isinstance(keyframes_list, str)) else []

    # --- Conference-paper style ---
    plt.rcParams.update({
        'font.family':        'serif',
        'font.size':          13,
        'axes.labelsize':     14,
        'axes.labelweight':   'bold',
        'legend.fontsize':    11,
        'xtick.labelsize':    12,
        'ytick.labelsize':    12,
        'axes.linewidth':     1.2,
        'xtick.major.width':  1.2,
        'ytick.major.width':  1.2,
        'xtick.direction':    'in',
        'ytick.direction':    'in',
    })

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.plot(frame_numbers, vmaf_values, color='#0055CC', linewidth=1.6, linestyle='-',
            label='VMAF Score', zorder=2)

    if keyframes:
        for kf in keyframes:
            ax.axvline(x=kf, color='#008800', linestyle=':', linewidth=1.4, alpha=0.75, zorder=4)
        ax.plot([], [], color='#008800', linestyle=':', linewidth=1.4,
                label='Keyframes')

    ax.set_xlabel('Frame Index', fontweight='bold')
    ax.set_ylabel('VMAF Score', fontweight='bold')
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#AAAAAA',
              loc='upper center', bbox_to_anchor=(0.5, 1.13),
              ncol=2, borderaxespad=0.)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    if vmaf_values:
        ax.set_ylim(max(0, min(vmaf_values) - 2), 101)
    ax.set_xlim(0, max(frame_numbers) if frame_numbers else 1)

    fig.tight_layout()
    if not output_png_path:
        output_png_path = json_file_path.replace('.json', '.png')
    fig.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"Plot saved to: {output_png_path}")


def calculate_vmaf(reference_video_path: str, distorted_video_path: str, 
                   output_dir: Path, video_name: str, keyframes: Optional[List[int]] = None,
                   ffmpeg_path: str = "ffmpeg", vmaf_resolution: Tuple = (3840, 2160),
                   num_threads: int = 8, raw_video: bool = False,
                   input_resolution: Tuple = None, frame_rate: float = None,
                   pixel_format: str = None) -> Tuple[str, str]:
    """
    Calculate VMAF for a distorted video compared to reference video
    
    Args:
        reference_video_path: Path to reference video
        distorted_video_path: Path to distorted video
        output_dir: Directory to save VMAF outputs
        video_name: Name for output files
        keyframes: List of keyframe indices (1-based) for plotting
        ffmpeg_path: Path to ffmpeg executable
        vmaf_resolution: Resolution for VMAF calculation
        num_threads: Number of threads for VMAF calculation
    
    Returns:
        Tuple of (json_path, png_path) for the VMAF results
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output file paths
    json_path = output_dir / f"{video_name}_vmaf.json"
    png_path = output_dir / f"{video_name}_vmaf.png"
    
    # Generate VMAF command
    cmd = quality_estimation_command(
        reference_video_path=reference_video_path,
        distored_video_path=distorted_video_path,
        vmaf_resolution=vmaf_resolution,
        logfile_path=str(json_path),
        num_threads=num_threads,
        ffmpeg_path=ffmpeg_path,
        raw_video=raw_video,
        input_resolution=input_resolution,
        frame_rate=frame_rate,
        pixel_format=pixel_format
    )
    
    print(f"Calculating VMAF for {video_name}...")
    print(f"Command: {cmd}")
    
    # Run VMAF calculation
    try:
        subprocess.run(shlex.split(cmd), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running VMAF calculation: {e}")
        raise
    
    # keyframes are already 1-based as stored in keyframe_summary.json
    plot_vmaf_data(str(json_path), str(png_path), keyframes_list=keyframes if keyframes else None)
    
    return str(json_path), str(png_path)

