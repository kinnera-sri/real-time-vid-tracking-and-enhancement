import os
import re
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from utils.video_util import get_video_metadata, load_video
from utils.time_util import seconds_to_timestamp

from utils import blend_util as blend
from utils.ad_util import AdManager
from utils.keypoint_util import is_roi_out_of_bounds
from tqdm import tqdm


class ADBlender:
    def __init__(self, video_root, staging_dir):
        """
        Docstring for __init__

        Args:
            video_root: Root directory where input videos are stored
            staging_dir: Directory where intermediate files 
            (clips_info.json, clips_roi.json) are stored
        """
        self.video_root = video_root
        self.staging_dir = staging_dir

        # get clips_info and clips_roi file path
        self.clips_info_json = os.path.join(staging_dir, "clips_info.json")
        self.clips_roi_json = os.path.join(staging_dir, "clips_roi.json")

        # blending parameters
        self.feather_blend_px = 5
        self.warp_border_mode = cv2.BORDER_REPLICATE
        self.warp_border_value = [255, 255, 255]

        # video config
        # Supported video formats
        self.SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mkv"]
        self.VIDEO_CODECS = {
            ".mp4": "mp4v",
            ".avi": "XVID",
            ".mkv": "mp4v",  # Use mp4v codec for mkv as well
        }

        # ad manager
        self.ad_manager = AdManager()

    def _validate_video_format(self, video_path: str) -> str:
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

        if file_ext not in self.SUPPORTED_VIDEO_FORMATS:
            supported_formats_str = ", ".join(self.SUPPORTED_VIDEO_FORMATS)
            raise ValueError(
                f"Unsupported video format: {file_ext}\n"
                f"Supported formats: {supported_formats_str}\n"
                f"Please convert your video to one of the supported formats."
            )

        return file_ext

    def skip_ad_blending_for_clip(self, cap, out, clip_info):
        """
        Save original frames for the clip without blending when 
        no valid ROI is found or APO is not trackable.
        
        Args:
            cap: OpenCV VideoCapture object for input video
            out: OpenCV VideoWriter object for output video
            clip_info: Dictionary containing clip information 
                       (start_frame, end_frame, etc.)
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_info["start_frame"])
        for idx in tqdm(range(clip_info["start_frame"], clip_info["end_frame"])):
            ret, frame = cap.read()
            if not ret:
                print(f"End of video reached at frame {idx}. Stopping.")
                exit(0)
            out.write(frame)

    def blend_ad_in_clip(self, cap, out, clip_info, clip_roi):
        """
        Blend ad into the clip frames using the provided ROI information.
        Args:
            cap: OpenCV VideoCapture object for input video
            out: OpenCV VideoWriter object for output video
            clip_info: Dictionary containing clip information 
                       (start_frame, end_frame, etc.)
            clip_roi: Dictionary containing ROI information for the clip
        """

        total_frames = clip_info["duration_frames"]
        _, img0 = cap.read()
        img_H, img_W = img0.shape[:2]

        # template frame
        self.template_frame = img0.copy()

        roi = np.array(
            clip_roi["APOs"][0]["APO_list"]["tracked_rois"][0], dtype=np.float32
        ).reshape(-1, 2)[
            :4
        ]  # Get the first 4 points for the ROI
        roi_points = roi.astype(np.float32)

        ad_img_pil = self.ad_manager.ad_frames[0]
        ad_img = cv2.cvtColor(np.array(ad_img_pil), cv2.COLOR_RGBA2BGRA)

        ad_img_H, ad_img_W = ad_img.shape[:2]

        # create the ad mask
        ad_img_pts = np.array(
            [[0, 0], [ad_img_W, 0], [ad_img_W, ad_img_H], [0, ad_img_H]],
            dtype=np.float32,
        )

        # crete the ad mask using roi points
        ad_mask = np.zeros((img_H, img_W), dtype=np.uint8)
        cv2.fillPoly(ad_mask, [roi_points.astype(np.int32)], [255])

        H_adInsert = cv2.getPerspectiveTransform(ad_img_pts, roi_points)

        ad_bgra_warped = cv2.warpPerspective(
            ad_img,
            H_adInsert,
            (img_W, img_H),
            borderMode=self.warp_border_mode,
            borderValue=(*self.warp_border_value, 0),
        )
        ad_img_warped = ad_bgra_warped[:, :, :3]  # BGR
        ad_alpha_warped = ad_bgra_warped[:, :, 3]

        # blending
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_info["start_frame"])
        L_ref = blend.estimate_lighting(cap, roi_points, self.feather_blend_px, 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_info["start_frame"] + 1)

        img0_blended, S_prev = blend.retinex_blend_feathered(
            frame=img0,
            ad_mask=ad_mask,
            I_ins_warped=ad_img_warped,
            alpha_mask=ad_alpha_warped,
            lighting_ref=L_ref,
            feather_blend_px=self.feather_blend_px,
            roi_points=roi_points,
            S_prev=None,
        )
        out.write(img0_blended)

        if False:
            cv2.imshow("img0_blended", img0_blended)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        for idx in tqdm(range(clip_info["start_frame"] + 1, clip_info["end_frame"])):
            ret, img1 = cap.read()
            if not ret:
                print(f"End of video reached at frame {idx}. Stopping.")
                exit(0)

            current_time = (idx - clip_info["start_frame"]) / self.video_meta["fps"]
            ad_img_pil = self.ad_manager.get_ad_frame_for_time(current_time)
            ad_img = cv2.cvtColor(np.array(ad_img_pil), cv2.COLOR_RGBA2BGRA)

            roi_points_trans = clip_roi["APOs"][0]["APO_list"]["tracked_rois"][
                idx - clip_info["start_frame"]
            ].copy()

            if not is_roi_out_of_bounds(roi_points_trans, (img_H, img_W)):
                roi_points_trans = np.array(roi_points_trans, dtype=np.float32).reshape(
                    -1, 2
                )[:4]

                ad_mask = np.zeros((img_H, img_W), dtype=np.uint8)
                cv2.fillPoly(ad_mask, [roi_points_trans.astype(np.int32)], [255])

                H_comb = cv2.getPerspectiveTransform(ad_img_pts, roi_points_trans)
                # H = cv2.getPerspectiveTransform(roi_points, roi_points_trans)
                # H_comb = H @ H_adInsert

                ad_bgra_warped = cv2.warpPerspective(
                    ad_img,
                    H_comb,
                    (img_W, img_H),
                    borderMode=self.warp_border_mode,
                    borderValue=(*self.warp_border_value, 0),
                )
                ad_img_warped = ad_bgra_warped[:, :, :3]
                ad_alpha_warped = ad_bgra_warped[:, :, 3]

                # print(f"ROI Points Trans: {roi_points_trans}")

                L_ref = blend.estimate_lighting_knr([img0, img1], roi_points_trans, 5)

                img_blended, S_prev = blend.retinex_blend_feathered(
                    frame=img1,
                    ad_mask=ad_mask,
                    I_ins_warped=ad_img_warped,
                    alpha_mask=ad_alpha_warped,
                    lighting_ref=L_ref,
                    feather_blend_px=self.feather_blend_px,
                    roi_points=roi_points_trans,
                    S_prev=None,
                )

                # ANNOTATE ROI_POINTS_TRANS
                if False:
                    for point in roi_points_trans:
                        cv2.circle(
                            img_blended,
                            (int(point[0]), int(point[1])),
                            5,
                            (0, 255, 0),
                            -1,
                        )

                out.write(img_blended)

                if False:
                    cv2.imshow("img_blended", img_blended)
                    cv2.waitKey(40)
            else:
                print(
                    f"ROI points out of bounds for frame {idx}. Skipping blending for this frame."
                )
                out.write(img1)

        cv2.destroyAllWindows()

    def insert_ad(self, cap, out, clip_info, clip_roi, is_found):
        """
        Insert ad into the clip frames based on the provided ROI information.
        If no valid ROI is found or APO is not trackable, save original frames without blending.
        
        Args:
            cap: OpenCV VideoCapture object for input video
            out: OpenCV VideoWriter object for output video
            clip_info: Dictionary containing clip information 
                       (start_frame, end_frame, etc.)
            clip_roi: Dictionary containing ROI information for the clip
            is_found: Boolean indicating whether clip ROI information was found for the clip
        returns:
            cap: Updated OpenCV VideoCapture object after processing the clip
        """
        print(f"\nClip Info:")
        for key, value in clip_info.items():
            print(f" {key}: {value}")
        print()

        if not is_found:
            print(
                f"No ROI information found for clip {clip_info['clip_name']}. Skipping ad blending for this clip."
            )
            # save original frames for the clip
            self.skip_ad_blending_for_clip(cap, out, clip_info)
            return cap

        if len(clip_roi["APOs"]) > 0:
            print(f"APOs found ...")
            if clip_roi["APOs"][0]["APO_list"]["is_trackable"]:
                print(f"APO is trackable.")
                # ad blending code goes here.
                self.blend_ad_in_clip(cap, out, clip_info, clip_roi)
            else:
                print(f"APO is not trackable.")
                # save original frames for the clip
                self.skip_ad_blending_for_clip(cap, out, clip_info)
            print(
                f"Number of tracked ROIs: {len(clip_roi['APOs'][0]['APO_list']['tracked_rois'])}"
            )
        else:
            print(
                f"No APOs found for clip {clip_info['clip_name']}. Skipping ad blending for this clip."
            )
            # save original frames for the clip
            self.skip_ad_blending_for_clip(cap, out, clip_info)

        print("\n\n===========================================\n\n")

        return cap

    def process(self, ad_path, output_dir="./staging_dir/blended"):
        """
         Main processing function to read input video, blend ad
         content based on clip and ROI information, and save output video.
         
         Args:
             ad_path: Path to the ad content (image) to be blended into the clips
             output_dir: Directory where the blended output video will be saved
         """
        # load clips_info
        with open(self.clips_info_json, "r") as f:
            clips_info = json.load(f)

        os.makedirs(output_dir, exist_ok=True)

        # get input video path
        input_video = clips_info[0]["input_video"]
        print(f"processing input_video: {input_video}")
        input_video_path = os.path.join(self.video_root, input_video)

        video_ext = self._validate_video_format(input_video_path)

        self.video_meta = get_video_metadata(input_video_path)

        # print video metadata
        print(
            f"Video properties:\n Resolution: {self.video_meta['width']}x{self.video_meta['height']}"
        )
        print(f" FPS: {self.video_meta['fps']:.2f}")
        print(f" Total Frames: {self.video_meta['total_frames']}")
        print(f" Video format: {video_ext}")
        print(f" Video duration: {seconds_to_timestamp(self.video_meta['duration'])}")
        # print("\n")

        # load video
        cap = load_video(input_video_path)
        codec = self.VIDEO_CODECS.get(video_ext, "mp4v")  # Default to mp4v if not found
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_video_path = os.path.join(
            output_dir, f"{input_video.replace(video_ext, '')}_blended{video_ext}"
        )

        print(f"Output video will be saved to: {output_video_path}\n")

        out = cv2.VideoWriter(
            output_video_path,
            fourcc,
            self.video_meta["fps"],
            (self.video_meta["width"], self.video_meta["height"]),
        )

        if not out.isOpened():
            raise ValueError(
                f"Could not open output video file for writing: {output_video_path}"
            )

        # load ad content here
        # <Placeholder for ad content loading>
        self.ad_manager.setup_ad_template(ad_path)

        # load clips roi
        with open(self.clips_roi_json, "r") as f:
            clips_roi = json.load(f)

        global_frame_idx = 0
        seq_no = 0
        seq_len = len(clips_info[0]["clip_infos"])

        for seq in range(seq_len):
            print(f"Processing clip: {seq + 1:03d}/{seq_len:03d}")
            clip_info = self.get_clip_info_by_seq_no(clips_info, seq)
            clip_info["end_frame"] = (
                clip_info["start_frame"] + clip_info["duration_frames"]
            )

            clip_roi, is_found = self.get_clip_roi_by_name(
                clips_roi, clip_info["clip_name"]
            )
            cap = self.insert_ad(cap, out, clip_info, clip_roi, is_found)

            seq_no += 1

            global_frame_idx += clip_info["duration_frames"]
            print(f"Finished processing clip with seq_no {seq}.")

        cap.release()
        out.release()

    def get_clip_info_by_seq_no(self, clips_info, seq_no):
        """
        Retrieve clip information by its sequence number.
        Args:
            clips_info: List of clip information dictionaries loaded from clips_info.json
            seq_no: Sequence number of the clip to retrieve information for
        Returns:
            clip_info: Dictionary containing information for the specified clip sequence number
        Raises:
            ValueError: If clip information with the specified sequence number is not found
        """
        for clip_info in clips_info[0]["clip_infos"]:
            if clip_info["sequence_number"] == seq_no:
                return clip_info
        raise ValueError(f"Clip info with seq_no {seq_no} not found in clips_info.")

    def get_clip_roi_by_name(self, clips_roi, clip_name):
        """
        Retrieve clip ROI information by clip name.
        Args:
            clips_roi: List of clip ROI dictionaries loaded from clips_roi.json
            clip_name: Name of the clip to retrieve ROI information for
        Returns:
            clip_roi: Dictionary containing ROI information for the specified clip name
            is_found: Boolean indicating whether the clip ROI information was found
        """
        for clip_roi in clips_roi:
            if clip_roi["filename"] == clip_name.replace(".mkv", "_005.jpg"):
                print(f"Found matching clip_roi for clip_name {clip_name}.")
                return clip_roi, True
        # raise ValueError(f"Clip ROI with name {clip_name} not found in clips_roi.")
        return None, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, default="test_data/full_video/videos")
    parser.add_argument("--staging_dir", type=str, default="./staging_dir")
    parser.add_argument("--ad_path", type=str, default="assets/ads/lego.png")
    args = parser.parse_args()

    video_root = args.video_root
    staging_dir = args.staging_dir
    ad_path = args.ad_path

    blender = ADBlender(video_root, staging_dir)
    blender.process(ad_path)
