import os
import cv2
import subprocess
from pathlib import Path


def get_video_metadata(video_path):
    """
    Get metadata from the input video file.
    
    Returns:
        dict: A dictionary containing video metadata such as total frames,
        fps, width, height, and duration.
    """
    print(f"\n\nGetting metadata for video: {video_path}")
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error: Cannot open video: {video_path}.")
    else:
        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get frames per second
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get frame width and height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate duration in seconds
        duration = total_frames / fps

        cap.release()
        
        return {"total_frames": total_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "duration": duration
                }

def load_video(video_path: str) -> cv2.VideoCapture:
    """Load video file and return VideoCapture object"""
    # Validate video format first
    # self._validate_video_format(video_path)
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    return cap

def _validate_video_format(video_path: str, supported_formats: list) -> str:
    """
    Validate video format and return file extension
    Args:
        video_path: Path to video file
    Returns:
        file_extension: Validated file extension
    Raises:
        ValueError: If format is not supported
    """
    file_ext = Path(video_path).suffix.lower()

    if file_ext not in supported_formats:
        supported_formats_str = ", ".join(supported_formats)
        raise ValueError(
            f"Unsupported video format: {file_ext}\n"
            f"Supported formats: {supported_formats_str}\n"
            f"Please convert your video to one of the supported formats."
        )

    return file_ext

def _has_audio(video_path: str) -> bool:
    """
    Check if video has audio stream using ffprobe
    Args:
        video_path: Path to video file
    Returns:
        bool: True if audio exists, False otherwise
    """
    try:
        cmd = [
            "ffprobe",
            "-i",
            video_path,
            "-show_streams",
            "-select_streams",
            "a",
            "-loglevel",
            "error",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return len(result.stdout) > 0
    except Exception as e:
        print(f"Error checking audio: {e}")
        return False

def _merge_audio_fast(
    video_no_audio: str, original_video: str, output_path: str
) -> bool:
    """
    Merge audio from original video to processed video using ffmpeg (fastest method)
    Args:
        video_no_audio: Path to processed video without audio
        original_video: Path to original video with audio
        output_path: Path for final output with audio
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("Merging audio with ffmpeg...")
        cmd = [
            "ffmpeg",
            "-i",
            video_no_audio,  # Processed video (no audio)
            "-i",
            original_video,  # Original video (with audio)
            "-c:v",
            "copy",  # Copy video codec (no re-encoding)
            "-c:a",
            "aac",  # Use AAC audio codec
            "-map",
            "0:v:0",  # Use video from first input
            "-map",
            "1:a:0",  # Use audio from second input
            "-shortest",  # Match shortest stream duration
            "-y",  # Overwrite output
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        print(f"Audio merged successfully to: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error merging audio: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False