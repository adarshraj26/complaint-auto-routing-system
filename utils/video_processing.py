import os
import cv2
import numpy as np

def extract_keyframes(video_path: str, max_keyframes: int = 5) -> list:
    """
    Extracts keyframes from a video file using OpenCV.
    Identifies frames that are spaced evenly across the video.
    Returns:
        List of dicts: [{'timestamp_sec': float, 'image_rgb': np.ndarray, 'frame_idx': int}]
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0 or fps <= 0:
        cap.release()
        return []

    duration_sec = total_frames / fps
    
    # Calculate frame indices to extract
    # We want to extract max_keyframes evenly spaced across the video
    if total_frames <= max_keyframes:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * (total_frames - 1) / (max_keyframes - 1)) for i in range(max_keyframes)]

    keyframes = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Convert BGR to RGB (OpenCV default is BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = idx / fps
        
        keyframes.append({
            'frame_idx': idx,
            'timestamp_sec': round(timestamp, 2),
            'image_rgb': frame_rgb
        })
        
    cap.release()
    return keyframes

def extract_audio_from_video(video_path: str, audio_output_path: str) -> bool:
    """
    Extracts the audio track from a video file and saves it as a WAV file.
    Tries MoviePy first, and falls back to ffmpeg subprocess if moviepy fails.
    """
    if not os.path.exists(video_path):
        return False
        
    # Attempt MoviePy first
    try:
        import moviepy.editor as mp
        clip = mp.VideoFileClip(video_path)
        if clip.audio is not None:
            # Save audio file at 16kHz mono (standard for Whisper)
            clip.audio.write_audiofile(
                audio_output_path, 
                fps=16000, 
                nbytes=2, 
                codec='pcm_s16le', 
                ffmpeg_params=["-ac", "1"],
                logger=None
            )
            clip.close()
            return True
        else:
            clip.close()
            print("Video has no audio track.")
            return False
    except Exception as e:
        print(f"Moviepy audio extraction failed: {e}. Trying FFmpeg subprocess fallback...")
        
        # Subprocess FFmpeg fallback
        import subprocess
        try:
            # cmd: extract audio to WAV, 16kHz, mono
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                audio_output_path
            ]
            # Run command, suppressing output
            result = subprocess.run(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                check=True
            )
            return os.path.exists(audio_output_path)
        except Exception as sub_err:
            print(f"FFmpeg subprocess audio extraction failed: {sub_err}")
            return False
