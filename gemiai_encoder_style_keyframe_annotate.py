import cv2
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
from collections import deque

def create_segments_from_keyframes(keyframes, total_frames, fps):
    """
    Create segments from keyframes and calculate statistics.
    
    Args:
        keyframes: List of keyframe frame indices (1-indexed)
        total_frames: Total number of frames in video
        fps: Frames per second
    
    Returns:
        Dictionary with segments and statistics
    """
    if not keyframes:
        # No keyframes detected, single segment
        segments = [{
            "start_time": 0.0,
            "end_time": total_frames / fps if fps > 0 else 0.0,
            "duration": total_frames / fps if fps > 0 else 0.0,
            "start_frame": 0,
            "end_frame": total_frames - 1,
            "frame_count": total_frames,
            "segment_index": 0
        }]
        return {
            "segments": segments,
            "segment_count": 1,
            "avg_segment_duration": total_frames / fps if fps > 0 else 0.0,
            "min_segment_duration": total_frames / fps if fps > 0 else 0.0,
            "max_segment_duration": total_frames / fps if fps > 0 else 0.0,
            "avg_segment_frame_count": total_frames,
            "min_segment_frame_count": total_frames,
            "max_segment_frame_count": total_frames
        }
    
    segments = []
    keyframe_times = [kf / fps if fps > 0 else 0.0 for kf in keyframes]
    
    # First segment: from start to first keyframe
    if keyframes[0] > 0:
        start_frame = 0
        end_frame = keyframes[0] - 1
        start_time = 0.0
        end_time = keyframe_times[0]
        duration = end_time - start_time
        frame_count = end_frame - start_frame + 1
        
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": frame_count,
            "segment_index": len(segments)
        })
    
    # Middle segments: between keyframes
    for i in range(len(keyframes) - 1):
        start_frame = keyframes[i]
        end_frame = keyframes[i + 1] - 1
        start_time = keyframe_times[i]
        end_time = keyframe_times[i + 1]
        duration = end_time - start_time
        frame_count = end_frame - start_frame + 1
        
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": frame_count,
            "segment_index": len(segments)
        })
    
    # Last segment: from last keyframe to end
    last_keyframe = keyframes[-1]
    if last_keyframe < total_frames:
        start_frame = last_keyframe
        end_frame = total_frames - 1
        start_time = keyframe_times[-1]
        end_time = total_frames / fps if fps > 0 else 0.0
        duration = end_time - start_time
        frame_count = end_frame - start_frame + 1
        
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": frame_count,
            "segment_index": len(segments)
        })
    
    # Calculate statistics
    durations = [seg["duration"] for seg in segments]
    frame_counts = [seg["frame_count"] for seg in segments]
    
    return {
        "segments": segments,
        "segment_count": len(segments),
        "avg_segment_duration": np.mean(durations) if durations else 0.0,
        "min_segment_duration": min(durations) if durations else 0.0,
        "max_segment_duration": max(durations) if durations else 0.0,
        "avg_segment_frame_count": np.mean(frame_counts) if frame_counts else 0.0,
        "min_segment_frame_count": min(frame_counts) if frame_counts else 0,
        "max_segment_frame_count": max(frame_counts) if frame_counts else 0
    }

def process_video_folder(input_folder, output_root_folder, save_video=False):
    """
    Scans a folder for videos and processes them one by one.
    
    Args:
        input_folder: Folder containing video files
        output_root_folder: Folder to save outputs
        save_video: If True, saves annotated video. If False (default), only saves images.
    """
    extensions = ['*.mp4']
    video_files = []
    
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    video_files.sort()
    
    if not video_files:
        print(f"No video files found in: {input_folder}")
        return

    print(f"Found {len(video_files)} videos. Starting batch process...\n")
    print(f"Mode: {'Video + Images' if save_video else 'Images only (default)'}")
    print("-" * 50)

    all_videos_data = []
    
    for i, video_path in enumerate(video_files):
        print(f"[{i+1}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
        video_data = robust_encoder_keyframe_detection(video_path, output_root_folder, save_video=save_video)
        if video_data:
            all_videos_data.append(video_data)
        print("-" * 50)
    
    # Save combined summary JSON file
    if all_videos_data:
        summary_json_path = os.path.join(output_root_folder, "keyframe_summary.json")
        summary_data = {"videos": all_videos_data}
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\n -> Saved combined summary to: {summary_json_path}")

def robust_encoder_keyframe_detection(video_path, output_folder, save_video=False):
    # 1. SETUP DIRECTORIES
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_folder, video_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 2. PARAMETERS (tuned for 60fps PlayStation recordings)
    WINDOW_SIZE = 90   # 1.5 s of history at 60fps
    LAG         = 6    # 0.1 s lag at 60fps
    MIN_MAD     = 5.0

    # --- TEXT SETTINGS ---
    TEXT_DURATION_SEC = 0.5
    text_frame_count = int(fps * TEXT_DURATION_SEC)
    # ---------------------

    ret, prev_frame = cap.read()
    if not ret: return None

    prev_y = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2YUV)[:,:,0]

    # Sparse pre-scan: sample consecutive frame pairs every 30 frames to estimate
    # the video's motion variability (CV = std/mean), then map CV -> THRESHOLD_Z.
    # Calibrated so CV=0.5 (typical content) yields Z=3.5 (the original default).
    sparse_mads = []
    SAMPLE_STEP = 30
    for fi in range(SAMPLE_STEP, total_frames - 1, SAMPLE_STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, fa = cap.read()
        if not ret: continue
        ret, fb = cap.read()
        if not ret: continue
        fa_y = cv2.cvtColor(fa, cv2.COLOR_BGR2YUV)[:,:,0]
        fb_y = cv2.cvtColor(fb, cv2.COLOR_BGR2YUV)[:,:,0]
        sparse_mads.append(float(np.mean(cv2.absdiff(fb_y, fa_y))))
    if len(sparse_mads) > 1 and np.mean(sparse_mads) > 0:
        cv_val = float(np.std(sparse_mads) / np.mean(sparse_mads))
    else:
        cv_val = 1.0
    THRESHOLD_Z = float(np.clip(2.5 + cv_val * 2.0, 2.5, 4.5))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)  # reset for main loop
    print(f" -> Adaptive THRESHOLD_Z: {THRESHOLD_Z:.2f}  (CV = {cv_val:.3f})")

    history = deque(maxlen=WINDOW_SIZE)

    mad_scores = []
    thresholds = []
    keyframes = []

    print(f" -> Phase 1: Analyzing Motion Statistics...")

    curr_idx = 1
    last_keyframe = -100

    # --- PHASE 1: ANALYSIS ---
    while True:
        ret, frame = cap.read()
        if not ret: break

        curr_y = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:,:,0]

        diff = cv2.absdiff(curr_y, prev_y)
        mad = np.mean(diff)

        if len(history) == WINDOW_SIZE:
            hist_array = np.array(history)
            valid_history = hist_array[:-LAG]

            mean = np.mean(valid_history)
            std = np.std(valid_history)

            adaptive_thresh = mean + (THRESHOLD_Z * std)
            adaptive_thresh = max(adaptive_thresh, MIN_MAD)

            if mad > adaptive_thresh:
                if (curr_idx - last_keyframe) > (fps * 0.5):
                    keyframes.append(curr_idx)
                    last_keyframe = curr_idx
                    history.clear()
        else:
            adaptive_thresh = max(mad * 2, 20.0)

        history.append(mad)
        mad_scores.append(mad)
        thresholds.append(adaptive_thresh)
        prev_y = curr_y
        curr_idx += 1
        
        if curr_idx % 500 == 0:
            print(f"    Scanning frame {curr_idx}/{total_frames}...", end='\r')

    print(f"\n -> Detected {len(keyframes)} Keyframes.")

    # Calculate video duration
    video_duration = total_frames / fps if fps > 0 else 0.0
    
    # Convert keyframes to times
    keyframe_times = [kf / fps if fps > 0 else 0.0 for kf in keyframes]
    
    # Create segments and statistics
    segment_data = create_segments_from_keyframes(keyframes, total_frames, fps)
    
    # Prepare JSON data structure
    video_data = {
        "video_path": video_path,
        "video_name": os.path.basename(video_path),
        "video_duration": video_duration,
        "video_fps": fps,
        "total_frames": total_frames,
        "valid_thresholds": [{
            "threshold": f"adaptive_z{THRESHOLD_Z}_min{MIN_MAD}",
            "num_keyframes": len(keyframes),
            "keyframe_times": keyframe_times,
            "keyframe_frames": keyframes,
            "segments": segment_data["segments"],
            "segment_count": segment_data["segment_count"],
            "avg_segment_duration": segment_data["avg_segment_duration"],
            "min_segment_duration": segment_data["min_segment_duration"],
            "max_segment_duration": segment_data["max_segment_duration"],
            "avg_segment_frame_count": segment_data["avg_segment_frame_count"],
            "min_segment_frame_count": segment_data["min_segment_frame_count"],
            "max_segment_frame_count": segment_data["max_segment_frame_count"]
        }],
        "num_valid_thresholds": 1
    }
    
    # Save JSON file
    json_path = os.path.join(save_dir, f"{video_name}_keyframe_summary.json")
    with open(json_path, 'w') as f:
        json.dump(video_data, f, indent=2)
    print(f" -> Saved keyframe details to: {json_path}")

    # --- PHASE 2: VISUALIZATION (PLOT) ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    fig, ax = plt.subplots(figsize=(12, 4.5))

    frame_indices = list(range(len(mad_scores)))

    ax.plot(frame_indices, mad_scores, color='#222222', linewidth=0.9,
            label='MAD Signal', zorder=2)
    ax.plot(frame_indices, thresholds, color='#E07B00', linewidth=2.0,
            linestyle='--', label='Adaptive Threshold', zorder=3)

    # Fix: scatter uses 0-based index (k-1) to align with the MAD signal line
    scatter_x = [k - 1 for k in keyframes if 0 < k <= len(mad_scores)]
    scatter_y = [mad_scores[x] for x in scatter_x]
    ax.scatter(scatter_x, scatter_y, color='#CC0000', s=70, zorder=5,
               label='Detected Scene Cut', marker='o', edgecolors='white',
               linewidths=0.6)

    ax.set_xlabel("Frame Index", fontweight='bold')
    ax.set_ylabel("Mean Absolute Difference (MAD)", fontweight='bold')
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#AAAAAA',
              loc='upper right')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    ax.set_xlim(0, len(mad_scores))
    ax.margins(y=0.1)

    fig.tight_layout()
    plot_path = os.path.join(save_dir, f"{video_name}_graph.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)

    # --- PHASE 3: SAVE KEYFRAME IMAGES (DEFAULT) ---
    print(" -> Phase 2: Saving Keyframe Images...")
    
    cap.release()
    cap = cv2.VideoCapture(video_path)
    
    keyframe_set = set(keyframes)
    curr_frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_idx = curr_frame_idx + 1 
        
        # Save image if keyframe detected
        if display_idx in keyframe_set:
            image_path = os.path.join(save_dir, f"frame_{display_idx:06d}.png")
            cv2.imwrite(image_path, frame)
            saved_count += 1
        
        curr_frame_idx += 1
        
        if curr_frame_idx % 500 == 0:
            print(f"    Processing {curr_frame_idx}/{total_frames}...", end='\r')
    
    cap.release()
    print(f"\n -> Saved {saved_count} keyframe images to: {save_dir}")

    # --- PHASE 4: SAVE ANNOTATED VIDEO (OPTIONAL) ---
    if save_video:
        print(" -> Phase 3: Generating Annotated Video...")
        
        cap = cv2.VideoCapture(video_path)
        
        output_video_path = os.path.join(save_dir, f"{video_name}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        curr_frame_idx = 0 
        
        active_message = ""
        frames_remaining = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            display_idx = curr_frame_idx + 1 
            
            # 1. Trigger Overlay if Keyframe Detected
            if display_idx in keyframe_set:
                active_message = f"KEYFRAME: {display_idx}"
                frames_remaining = text_frame_count 
                
                # Red Border Flash
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 20)

            # 2. Draw Persistent Text (BIGGER FONT)
            if frames_remaining > 0:
                # Position: Center-ish Bottom (80 pixels from bottom edge to accommodate larger text)
                text_pos = (50, height - 80)
                
                # White Outline (Thickness increased to 10)
                cv2.putText(frame, active_message, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                            2.5, (255, 255, 255), 10, cv2.LINE_AA) 
                # RED Text (Scale increased to 2.5, Thickness to 5)
                cv2.putText(frame, active_message, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                            2.5, (0, 0, 255), 5, cv2.LINE_AA) 
                
                frames_remaining -= 1

            out.write(frame)
            curr_frame_idx += 1
            
            if curr_frame_idx % 500 == 0:
                print(f"    Writing {curr_frame_idx}/{total_frames}...", end='\r')

        cap.release()
        out.release()
        print(f"\n -> Done. Saved annotated video to: {output_video_path}")
    
    return video_data
# --- CONFIGURATION & EXECUTION ---
INPUT_FOLDER = "/mnt/Expansion_mnt/600frames_cuts/clips_yuv420p" 
OUTPUT_FOLDER = "/mnt/Expansion_mnt/reimplement_600frames_cuts/keyframe_summary_new_algo4"

# Set to True if you want to save annotated videos, False for images only (default)
SAVE_VIDEO = False

process_video_folder(INPUT_FOLDER, OUTPUT_FOLDER, save_video=SAVE_VIDEO)