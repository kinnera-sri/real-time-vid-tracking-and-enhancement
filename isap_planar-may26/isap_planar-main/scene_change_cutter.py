#!/usr/bin/env python3
"""
Scene change detection and video cutting.

This module provides functions to:
- Detect scene changes in a video using FFmpeg
- Cut the video into clips at scene boundaries
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

# How many seconds before start_time to seek.  A larger value means the seek
# lands on a keyframe further back, so FFmpeg needs to decode more frames to
# reach trim_start – but it also improves the chance of hitting a keyframe.
# 12s is a practical sweet spot for typical GOP sizes.
SEEK_BUFFER = 12.0


class FfmpegError(RuntimeError):
    """Raised when FFmpeg operations fail."""
    pass


def _require_ffmpeg() -> None:
    """Check if ffmpeg is available on the system."""
    if shutil.which("ffmpeg") is None:
        raise FfmpegError("ffmpeg not found on PATH. Install it and try again.")


def _run_ffmpeg(cmd: list[str], capture_stderr: bool = True) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg command and return the result."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if capture_stderr else subprocess.STDOUT,
        text=True,
        check=False
    )
    return result


def detect_scene_changes(
    video_path: str | Path,
    threshold: float = 0.4,
) -> List[float]:
    """
    Detect scene changes in a video using FFmpeg's scene filter.

    Args:
        video_path: Path to the input video file
        threshold: Scene change detection threshold (0.0-1.0).
                  Lower values = more sensitive (more cuts detected)

    Returns:
        List of timestamps (in seconds) where scene changes occur
    """
    _require_ffmpeg()
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null",
        "-"
    ]

    result = _run_ffmpeg(cmd)

    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip() or "ffmpeg failed"
        raise FfmpegError(f"Scene detection failed: {error_msg}")

    print("--- RAW FFMPEG STDERR (showinfo lines only) ---")
    for line in result.stderr.splitlines():
        if "Parsed_showinfo" in line:
            print(line)
    print("--- END RAW FFMPEG STDERR ---")

    # Handle truncated decimals like "pts_time:45." and normal "pts_time:45.123"
    pts_time_pattern = re.compile(r"pts_time:([\d.]+)")
    timestamps: List[float] = []

    for line in result.stderr.splitlines():
        if "Parsed_showinfo" in line:
            match = pts_time_pattern.search(line)
            if match:
                raw_val = match.group(1)
                parsed = float(raw_val)
                print(f"  matched pts_time raw='{raw_val}' -> float={parsed}")
                timestamps.append(parsed)

    timestamps = sorted(set(timestamps))
    print(f"Detected scene times (raw floats): {timestamps}")
    return timestamps


def get_video_duration(video_path: str | Path) -> int:
    """Get the duration of a video in seconds."""
    _require_ffmpeg()
    video_path = Path(video_path)

    #cmd = [
    #    "ffprobe",
    #    "-v", "error",
    #    "-show_entries", "format=duration",
    #    "-of", "default=noprint_wrappers=1:nokey=1",
    #    str(video_path)
    #]
    #cmd = [
    #    "ffprobe",
    #    "-v", "error",
    #    "-select_streams", "v:0",
    #    "-count_frames",
    #    "-show_entries",
    #    "stream=nb_read_frames",
    #    "-of", "csv=p=0",
    #    str(video_path)
    #        ]
    cmd = [ "ffmpeg",
           "-i", str(video_path),
           "-map", "0:v:0",
           "-f", "null", "-"
           ]

    result = _run_ffmpeg(cmd, capture_stderr=False)

    if result.returncode != 0:
        raise FfmpegError(f"Failed to get video duration: {result.stdout}")

    try:
        #return float(result.stdout.strip())
        matches = re.findall(r"frame=\s+\d+\s", result.stdout)
        match = matches[-1]
        frame_count = re.findall(r"\d+", match)
        frame_count = int(frame_count[0])
        return frame_count
    except ValueError:
        raise FfmpegError(f"Could not parse video duration: {result.stdout}")


def get_video_fps(video_path: str | Path) -> float:
    """Get the frame rate (fps) of a video."""
    _require_ffmpeg()
    video_path = Path(video_path)

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    result = _run_ffmpeg(cmd, capture_stderr=False)

    if result.returncode != 0:
        raise FfmpegError(f"Failed to get video fps: {result.stdout}")

    raw = result.stdout.strip()
    if "/" in raw:
        num, den = raw.split("/", 1)
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            raise FfmpegError(f"Could not parse video fps: {raw}")
    try:
        return float(raw)
    except ValueError:
        raise FfmpegError(f"Could not parse video fps: {raw}")


def cut_video_segment(
    video_path: str | Path,
    start_time: float,
    duration: float,
    output_path: str | Path,
    reencode: bool = False,
    bitrate: str | None = None,
    strip_audio: bool = False,
) -> None:
    """
    Cut a video segment using hybrid fast+accurate method.

    Strategy:
      1. Input-side seek (-ss before -i) jumps to the keyframe nearest to
         (start_time - SEEK_BUFFER), skipping decode of the earlier part of the
         video entirely.
      2. trim/atrim filters within the seeked segment provide frame-accurate cuts
         with no keyframe-snap artefacts at the start.
      3. setpts=PTS-STARTPTS / asetpts=PTS-STARTPTS reset PTS to 0, preventing
         any freeze or playback stall at the end.
      4. trim:end= (exclusive) avoids the boundary-bleed caused by trim:duration=.

    Args:
        video_path:   Path to the input video.
        start_time:   Start time in seconds (absolute in the source).
        duration:     Duration in seconds.
        output_path:  Destination file path.
        reencode:     Ignored – always re-encodes for correct PTS.
        bitrate:      Optional video bitrate (e.g. "1000k").
        strip_audio:  If True, drop the audio stream from the output.
    """
    _require_ffmpeg()
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Hybrid seek math ---
    # coarse_seek: where we tell FFmpeg to start reading (snaps to prior keyframe)
    seek_buf = min(start_time, SEEK_BUFFER)
    coarse_seek = start_time - seek_buf   # absolute time of the coarse seek

    # trim offsets are relative to the coarse_seek position
    trim_start = seek_buf                 # = start_time - coarse_seek
    end_time = start_time + duration if duration is not None else None
    trim_end = (trim_start + duration) if duration is not None else None

    if trim_end is not None:
        v_filter = f"trim=start={trim_start:.6f}:end={trim_end:.6f},setpts=PTS-STARTPTS"
        a_filter = f"atrim=start={trim_start:.6f}:end={trim_end:.6f},asetpts=PTS-STARTPTS"
    else:
        v_filter = f"trim=start={trim_start:.6f},setpts=PTS-STARTPTS"
        a_filter = f"atrim=start={trim_start:.6f},asetpts=PTS-STARTPTS"

    print(f"    hybrid seek: coarse_seek={coarse_seek:.6f}  trim_start={trim_start:.6f}  "
          f"trim_end={trim_end}  (seek_buf={seek_buf:.6f})")

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-ss", f"{coarse_seek:.6f}",   # fast input-side seek to nearby keyframe
        "-i", str(video_path),
    ]

    if strip_audio:
        cmd.extend(["-vf", v_filter, "-an"])
    else:
        cmd.extend([
            "-filter_complex", f"[0:v]{v_filter}[v];[0:a]{a_filter}[a]",
            "-map", "[v]",
            "-map", "[a]",
        ])

    cmd.extend([
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-movflags", "faststart",
    ])

    if not strip_audio:
        cmd.extend(["-c:a", "aac"])

    if bitrate:
        cmd.extend(["-b:v", bitrate])

    cmd.extend(["-y", str(output_path)])

    print(f"    ffmpeg cmd: {' '.join(cmd)}")
    result = _run_ffmpeg(cmd)

    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip() or "ffmpeg failed"
        raise FfmpegError(f"Failed to cut video: {error_msg}")


def detect_and_cut_scenes(
    video_path: str | Path,
    output_dir: str | Path,
    threshold: float = 0.4,
    clip_prefix: str = "clip",
    clip_format: str = "mp4",
    reencode: bool = False,
    bitrate: str | None = None,
    strip_audio: bool = False,
    trim_end: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Detect scene changes in a video and cut it into clips.

    Args:
        video_path:   Path to the input video file.
        output_dir:   Directory where clips will be saved.
        threshold:    Scene change detection threshold (0.0-1.0).
        clip_prefix:  Prefix for output clip filenames.
        clip_format:  Output video format (e.g. "mp4", "mkv").
        reencode:     Ignored – always re-encodes.
        bitrate:      Optional bitrate (e.g. "1000k").
        strip_audio:  If True, strip audio from output clips.
        trim_end:     Trim this many seconds from the end of each clip.

    Returns:
        List of dicts with keys: clip_name, duration_frames, sequence_number,
        start_frame, duration_seconds, start_seconds, end_seconds.
    """
    _require_ffmpeg()
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detecting scene changes in {video_path}...")
    scene_times = detect_scene_changes(video_path, threshold)
    print(f"Found {len(scene_times)} scene changes")
    print(f"Scene times: {scene_times}")

    duration = get_video_duration(video_path)
    fps = get_video_fps(video_path)
    print(f"Video duration (raw): {duration}  fps (raw): {fps}")

    original_filename = video_path.name
    clip_infos: Dict[str, Any] = {}

    # ---- Print cutting plan ----
    print(f"\n--- CUTTING PLAN ---")
    print(f"{'idx':>4}  {'start':>18}  {'end':>18}  {'duration':>18}")
    time_from = 0.0
    for i, ts in enumerate(scene_times):
        ts_clamped = min(ts, duration)
        dur = ts_clamped - time_from - trim_end
        flag = 'SKIP(<=0)' if dur <= 0 else ('SKIP(<0.5s)' if dur < 0.5 else '')
        print(f"{i:>4}  {time_from:>18}  {ts_clamped:>18}  {dur:>18}  {flag}")
        time_from = ts_clamped
    final_dur = duration - time_from
    print(f"{'fin':>4}  {time_from:>18}  {duration:>18}  {final_dur:>18}")
    print(f"--- END CUTTING PLAN ---\n")

    # ---- Cutting loop ----
    time_from = 0.0
    clip_idx = 0
    MIN_CLIP_DURATION = 0.5  # skip false double-detections

    print("Cutting video into clips...")

    for timestamp in scene_times:
        timestamp = min(timestamp, duration)
        clip_duration = timestamp - time_from - trim_end

        print(f"  [clip {clip_idx}] time_from={time_from}  timestamp={timestamp}  "
              f"clip_duration={clip_duration}  trim_end={trim_end}")

        if clip_duration < MIN_CLIP_DURATION:
            print(f"    -> SKIPPED (duration {clip_duration} < MIN_CLIP_DURATION {MIN_CLIP_DURATION}), "
                  f"merging into next clip")
            # Do NOT update time_from – absorb this short scene into the next clip
            continue

        if clip_prefix:
            output_filename = f"{clip_idx:04d}_{clip_prefix}_{video_path.stem}.{clip_format}"
        else:
            output_filename = f"{clip_idx:04d}_{original_filename}"
        output_path = output_dir / output_filename

        print(f"    -> cutting: start={time_from}  duration={clip_duration}  out={output_filename}")

        try:
            cut_video_segment(
                video_path, time_from, clip_duration, output_path,
                reencode=reencode, bitrate=bitrate, strip_audio=strip_audio
            )
            print(f"    -> done")
            clip_infos[output_filename]= {
                "clip_path" : str(output_path),
                "duration_frames": int(round(clip_duration * fps)),
                "sequence_number": clip_idx,
                "start_frame": int(round(time_from * fps)),
                "duration_seconds": clip_duration,
                "start_seconds": time_from,
                "end_seconds": timestamp,
            }
            clip_idx += 1           # Only increment on success
            time_from = timestamp   # Only advance on success
        except FfmpegError as e:
            print(f"    -> FAILED: {e}")
            time_from = timestamp   # Still advance to avoid retrying the same segment

    # ---- Final clip (last scene change → end of video) ----
    if time_from < duration:
        if clip_prefix:
            output_filename = f"{clip_idx:04d}_{clip_prefix}_{video_path.stem}.{clip_format}"
        else:
            output_filename = f"{clip_idx:04d}_{video_path.name}"
        output_path = output_dir / output_filename

        final_duration_sec = duration - time_from
        print(f"  [final clip {clip_idx}] time_from={time_from}  duration={duration}  "
              f"final_duration_sec={final_duration_sec}  -> {output_filename}")

        try:
            cut_video_segment(
                video_path, time_from, final_duration_sec, output_path,
                reencode=reencode, bitrate=bitrate, strip_audio=strip_audio
            )
            final_end_time = min(duration, time_from + final_duration_sec)
            clip_infos[output_filename] = {
                "clip_path" : str(output_path),
                "duration_frames": int(round(final_duration_sec * fps)),
                "sequence_number": clip_idx,
                "start_frame": int(round(time_from * fps)),
                "duration_seconds": final_duration_sec,
                "start_seconds": time_from,
                "end_seconds": final_end_time,
            }
            print(f"    -> done (final)")
        except FfmpegError as e:
            print(f"    -> FAILED (final): {e}")

    print(f"Successfully created {len(clip_infos)} clips in {output_dir}")
    return clip_infos


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect scene changes and cut video into clips (hybrid fast+accurate)"
    )
    parser.add_argument("-f", "--file", type=str, required=True, dest="video",
                        help="Input video file")
    parser.add_argument("-o", "--out", type=str, default="./clips", dest="output",
                        help="Output directory for clips (default: ./clips)")
    parser.add_argument("-d", "--diff", type=float, default=0.2, dest="threshold",
                        help="Scene change detection threshold (0.0-1.0, default: 0.2)")
    parser.add_argument("--format", type=str, default=None,
                        help="Output video format (default: same as input)")
    parser.add_argument("-b", "--bitrate", type=str, default=None,
                        help="Bitrate for encoding (e.g., '1000k')")
    parser.add_argument("-sa", "--strip-audio", action="store_true",
                        help="Strip audio from output clips")
    parser.add_argument("--trim", type=float, default=0.0,
                        help="Trim this many seconds from the end of each clip (default: 0.0)")
    parser.add_argument("--seek-buffer", type=float, default=SEEK_BUFFER,
                        help=f"Seconds to seek before start_time for keyframe alignment "
                             f"(default: {SEEK_BUFFER})")

    args = parser.parse_args()

    # Allow overriding SEEK_BUFFER from command line
    import scene_change_cutter_hybrid as _self
    _self.SEEK_BUFFER = args.seek_buffer

    video_path = Path(args.video)
    output_format = args.format or (video_path.suffix.lstrip('.') or 'mp4')

    try:
        clip_infos = detect_and_cut_scenes(
            args.video,
            args.output,
            threshold=args.threshold,
            clip_prefix="",
            clip_format=output_format,
            reencode=True,
            bitrate=args.bitrate,
            strip_audio=args.strip_audio,
            trim_end=args.trim
        )
        print(f"\nDone! Created {len(clip_infos)} clips.")

        scenes_data = {"input_video": video_path.name, "clip_infos": clip_infos}
        scenes_path = Path(args.output) / "scenes.json"
        with open(scenes_path, "w") as f:
            json.dump(scenes_data, f, indent=2)
        print(f"Wrote {scenes_path}")

        if clip_infos:
            best = max(clip_infos, key=lambda x: x["duration_frames"])
            print(f"\nClip with maximum duration:")
            print(f"  Clip name:       {best['clip_name']}")
            print(f"  Duration:        {best['duration_frames']} frames")
            print(f"  Sequence number: {best['sequence_number']}")
            print(f"  Start frame:     {best['start_frame']}")
    except Exception as e:
        import sys
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
