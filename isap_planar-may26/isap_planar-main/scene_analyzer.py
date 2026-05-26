import argparse
import json
import sys
from pathlib import Path
import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("./models/LightGlueGyrus")

from scene_change_cutter import detect_and_cut_scenes, get_video_duration
from robustpoint.robustpoint_top import RobustPointTop  # noqa: E402  # type: ignore[reportMissingImports]
from lightglue_gy import LightGlueGy  # noqa: E402  # type: ignore[reportMissingImports]
from lightglue_gy.utils import rbd
from lightglue_gy import viz2d
import scene_cluster_simple as scene_cluster


##############################################################################
# Some constants
##############################################################################
OUTPUT_DIR = "staging_dir"
VIDEO_INFO_FILE = f"{OUTPUT_DIR}/video_info.json"
ROBUSTPOINT_CKPT_FILE = "assets/checkpoints/tracker/robustpoint_ckpt.pth"
LIGHT_GLUE_CKPT_FILE = "assets/checkpoints/tracker/checkpoint_lgrp.tar"
DO_SCENE_CLUSTER=True

def _add_lightglue_infer_to_syspath() -> None:
    """Make `LightGlue_infer/` importable as top-level packages (lightglue, robustpoint)."""
    repo_root = Path(__file__).resolve().parent
    infer_dir = repo_root / "LightGlue_infer"
    if infer_dir.exists():
        infer_dir_str = str(infer_dir)
        if infer_dir_str not in sys.path:
            sys.path.insert(0, infer_dir_str)


def _read_nth_frame_cv2(clip_path: Path, n=10) -> np.ndarray:
    """Read the first frame of a clip into a NumPy array (BGR) using OpenCV."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {clip_path}")
    for i in range(n-1):
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"OpenCV could not read {n} frame: {clip_path}")
    return frame

def _read_frames_cv2(clip_path: Path, frame_indices) -> np.ndarray:
    """Read frames of a clip into a NumPy array (BGR) using OpenCV."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {clip_path}")
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        frames.append(frame)
    cap.release()
    return frames




def pre_proc_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (640, 480))
    image = image[None]  # add channel axis
    image = image[None]  # add batch axis
    image = torch.tensor(image / 255.0, dtype=torch.float)
    return image


def main() -> int:
    p = argparse.ArgumentParser(description="Cut video into scene-based clips and report max-frame clip.")
    p.add_argument("-i", "--ip", required=True, help="Input video")
    p.add_argument("-t", "--threshold", type=float, default=0.2, help="Scene threshold (0..1)")
    p.add_argument("-b", "--bitrate", default=None, help="Bitrate when re-encoding (e.g. 1000k)")
    p.add_argument("--strip-audio", action="store_true", help="Drop audio in output clips")
    p.add_argument("--trim-end", type=float, default=0.0, help="Trim N seconds from end of each clip")
    p.add_argument("--format", default=None, help="Output format/extension (default: same as input)")
    p.add_argument("--prefix", default="clip", help="Filename prefix (default: clip). Use '' to mimic bash naming.")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, file_ext = os.path.splitext(args.ip)

    if os.path.exists(OUTPUT_DIR):
        error_str = f"\nERROR! {OUTPUT_DIR} exists! Perhaps it contains results from previous run\n"
        error_str += "Please delete/move this folder and run again..."
        raise Exception(error_str)
    else:
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

    ############################################################################
    # 1. model loading. 
    # These models may be required for analyzing if the scenes are same or not
    ############################################################################
    if 0:
        robustpoint_conf = {"max_num_keypoints": 512,
                        "force_num_keypoints": True,
                        "weights": ROBUSTPOINT_CKPT_FILE, }
        extractor = RobustPointTop(robustpoint_conf).eval().to(device)
        matcher = LightGlueGy(weights=LIGHT_GLUE_CKPT_FILE, features=None).eval().to(device)

    video_path = Path(args.ip)
    video_filename = os.path.split(video_path)[1]
    video_filename = os.path.splitext(video_filename)[0]
    clip_format = args.format or (video_path.suffix.lstrip(".") or "mp4")

    ############################################################################
    # 1. Detect scenes changes and cut the video into clips
    ############################################################################
    clip_infos = detect_and_cut_scenes(
        video_path=video_path,
        output_dir=OUTPUT_DIR,
        threshold=args.threshold,
        clip_prefix=args.prefix,
        clip_format=clip_format,
        reencode=True,
        bitrate=args.bitrate,
        strip_audio=args.strip_audio,
        trim_end=args.trim_end,
    )

    if not clip_infos:
        print("No clips created. The entire video contains only one scene")
        return 2

    ############################################################################
    # 1.b Let us get the clip_info by decoding the cut videos
    # We will first update the duration_frames for each clip
    # After this we will again traverse the list to update the start_frame
    # using the accurate duration frames
    ############################################################################
    # CAUTION:: Here we are assuming ordered list
    print(clip_infos)
    for idx, (clip_name, cl_info) in enumerate(clip_infos.items()):
        assert idx == int(cl_info["sequence_number"])
        dur_frames =  get_video_duration(cl_info["clip_path"])
        cl_info["duration_frames"] = dur_frames

    start_frame = 0
    for idx, (clip_name, cl_info) in enumerate(clip_infos.items()):
        if idx==0:
            start_frame = cl_info["duration_frames"]
            continue
        cl_info["start_frame"] = start_frame
        start_frame += cl_info["duration_frames"]


    #max_clip_info = max(clip_infos, key=lambda d: d.get("duration_frames", 0))
    ############################################################################
    # 2. Need some more analysis of the clips here
    # a> Figure out if the scenes are similar, construct a map of similar scenes
    #    scenes_map is an array of dictonary with key as the clip number,
    #    values is a list of clip numbers this clip is similar to 
    ############################################################################
    scene_map = {}
    for i in range(len(clip_infos)):
        scene_map[i] = [i]

    ############################################################################
    # 4. Extract nth fram from each clip and save them in "clip_frames" folder
    ############################################################################
    i = 0
    for clip_name, info in clip_infos.items():
        clip_frames = info.get("duration_frames")
        if clip_frames < 21: #we need some minimum number of frames
            i+=1
            continue
        clip_path = out_dir / clip_name
        n_frames = info.get("duration_frames")-1
        #frame_indices = list(range(0,n_frames,n_frames//2))
        # We will extract 5th frame from each clip and dump them
        frame_indices = [5]
        frames = _read_frames_cv2(clip_path, frame_indices)
        for k,j in enumerate(frame_indices):
            frame_path = out_dir / "clip_frames" / (Path(clip_name).stem + f"_{j:03d}.png")
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            if type(frames[k]) is not type(None):
                cv2.imwrite(str(frame_path), frames[k])

        i += 1


    if DO_SCENE_CLUSTER:
        scene_cluster_info = scene_cluster.run_full_pipeline(input_video=args.ip,
                                                          output_folder=OUTPUT_DIR,
                                                          run_split_video=False)


    ############################################################################
    # 3. Dump the results into json file
    ############################################################################
    #video_info_path = Path(VIDEO_INFO_FILE)
    video_info_json_path = Path(os.path.join(out_dir.as_posix(), f"{video_filename}_video_info.json"))
    op_dict = {"input_video": video_path.name, 
               "clip_infos": clip_infos,
               "scene_map": scene_map}
    if DO_SCENE_CLUSTER:
        op_dict["scene_cluster_info"] = scene_cluster_info
    json_record = json.dumps(op_dict, indent=2)
    video_info_json_path.write_text(json_record + "\n")


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
