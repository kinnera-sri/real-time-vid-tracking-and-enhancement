import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import glob
from ultralytics import YOLO

sys.path.append("LightGlueGyrus//")


import cv2
from roi_tracker import ROITracker

DBG_DIR = "staging_dir/debug/tracking"


class YOLOPersonDetector:
    def __init__(self, model_name="yolo26n-seg.pt", device="cuda:0"):
        self.model = YOLO(model_name)
        self.model.to(device)
        self.device = device

    def detect_human(self, img):
        results = self.model.predict(
            source=img, classes=[0], verbose=False
        )  # Class 0 is 'person' in the COCO dataset
        people_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # people_box_mask = np.zeros(img_.shape[:2], dtype=np.uint8)

        # Process results
        for result in results:
            if result.masks is not None:
                masks = result.masks.xy  # Segmentation masks as (x, y) points
                for i, mask in enumerate(masks):
                    # Create a binary mask for the person
                    b_mask = np.zeros(result.orig_img.shape[:2], np.uint8)
                    # Reshape the mask points for cv2.drawContours
                    contour = mask.astype(np.int32).reshape(-1, 1, 2)
                    # Draw the contour on the binary mask
                    cv2.drawContours(b_mask, [contour], -1, 1, cv2.FILLED)

                    people_mask = cv2.bitwise_or(b_mask, people_mask)
        return people_mask


def check_occlusion(people_mask, ad_roi, track_roi):
    # Create masks for ad_roi and track_roi
    ad_mask = np.zeros(people_mask.shape, dtype=np.uint8)
    track_mask = np.zeros(people_mask.shape, dtype=np.uint8)

    # Draw filled polygons for each ROI
    cv2.fillPoly(ad_mask, [ad_roi], 1)
    cv2.fillPoly(track_mask, [track_roi], 1)

    # Calculate occlusion: intersection of ROI with people_mask
    ad_overlap = cv2.bitwise_and(ad_mask, people_mask)
    track_overlap = cv2.bitwise_and(track_mask, people_mask)

    # Count pixels of overlap
    ad_occlusion_ratio = (
        np.sum(ad_overlap) / np.sum(ad_mask) if np.sum(ad_mask) > 0 else 0
    )
    track_occlusion_ratio = (
        np.sum(track_overlap) / np.sum(track_mask) if np.sum(track_mask) > 0 else 0
    )

    # Consider occluded if overlap ratio exceeds threshold (e.g., 10%)
    ad_occcl_thresh = 0.05
    track_occcl_thresh = 0.2
    ad_occluded = ad_occlusion_ratio > ad_occcl_thresh
    track_occluded = track_occlusion_ratio > track_occcl_thresh

    return ad_occluded, track_occluded


def process_video(tracker, vid_file, ad_roi, track_roi, output_dbg_video=False):
    

    cap = cv2.VideoCapture(vid_file)

    ret, img0 = cap.read()

    img_H, img_W = img0.shape[:2]
    # img0 = cv2.resize(img0, (img_W, img_H), interpolation=cv2.INTER_AREA)

    if output_dbg_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        _, filename = os.path.split(vid_file)
        dbg_out_file = f"{DBG_DIR}/{filename}"
        vid_out = cv2.VideoWriter(dbg_out_file, fourcc, 30.0, (img_W, img_H))
        print(f"writing to file {dbg_out_file}")
        vid_out.write(img0)

    tracked_ad_rois = {}
    tracked_ad_rois[0] = ad_roi
    idx = 0

    ad_roi_0 = np.array(ad_roi).astype(np.float32)[np.newaxis, :, :]
    track_roi_0 = np.array(track_roi).astype(np.float32)[np.newaxis, :, :]

    # initialize the person detector
    person_detector = YOLOPersonDetector("yolo26n-seg.pt")

    ad_roi_occl = False
    track_roi_occl = False

    while True:
        ret, img1 = cap.read()
        if not ret:
            print("stream end? Exiting ...")
            break

        idx = idx + 1
        # img1 = cv2.resize(img1, (img_W, img_H), interpolation=cv2.INTER_AREA)

        Hg, is_fine = tracker.run(img0, img1, track_roi_0, ad_roi_0)
        if Hg is not None:
            ad_roi_i = cv2.perspectiveTransform(ad_roi_0, Hg)[0]
            ad_roi_i = np.round(ad_roi_i).astype(int)

            # tracker roi in the i_th frame
            track_roi_i = cv2.perspectiveTransform(track_roi_0, Hg)[0]
            track_roi_i = np.round(track_roi_i).astype(int)

            # check if ad_roi_i is occluded by people
            people_mask = person_detector.detect_human(img1)

            assert (
                img1.shape[:2] == people_mask.shape[:2]
            ), f"Image shape {img1.shape} and people mask shape {people_mask.shape} do not match"

            ad_occl, track_occl = check_occlusion(people_mask, ad_roi_i, track_roi_i)

            if ad_occl:
                print(f"Frame {idx}: Ad ROI is occluded by people.")
                ad_roi_occl = True
            if track_occl:
                print(f"Frame {idx}: Track ROI is occluded by people.")
                track_roi_occl = True

            if 0:
                cv2.imshow("people_mask", people_mask.astype(np.uint8) * 255)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if output_dbg_video:
                img1 = cv2.fillPoly(img1, [ad_roi_i], (0, 0, 255))
            tracked_ad_rois[idx] = ad_roi_i.tolist()

        if output_dbg_video:
            vid_out.write(img1)
    cap.release()
    if output_dbg_video:
        vid_out.release()

    return tracked_ad_rois, ad_roi_occl, track_roi_occl


def main():
    parser = argparse.ArgumentParser(description="Main application")
    parser.add_argument(
        "--staging_dir",
        type=str,
        help="Path to input video file",
        default="staging_dir",
    )
    args = parser.parse_args()

    if not os.path.exists(args.staging_dir):
        raise Exception(f"Error: wrong directory: {args.staging_dir}")

    clips_roi_file = os.path.join(args.staging_dir, "clips_roi.json")
    if not os.path.exists(clips_roi_file):
        raise Exception(f"Error: clips roi file not found: {clips_roi_file}")

    op_dir = os.path.join(args.staging_dir, "tracked_ad_rois")
    os.makedirs(op_dir, exist_ok=True)
    os.makedirs(f"{DBG_DIR}", exist_ok=True)
    tracker = ROITracker()

    with open(clips_roi_file, "r") as f:
        clips_roi_data = json.load(f)

    op_json_file = os.path.join(args.staging_dir, f"tracked_ad_roi.json")
    op_json_fh = open(op_json_file, "w")
    op_json_fh.write("[")
    first_line = True
    for i, clip_roi in enumerate(clips_roi_data):
        filename = clip_roi["filename"]
        vid_file_noext = re.sub(r"_(\d+).jpg", "", filename)
        vid_file = glob.glob(args.staging_dir + "/" + vid_file_noext + ".*")[0]
        op_json_file = os.path.join(op_dir, f"{vid_file_noext}.json")
        APOs = clip_roi["APOs"]
        clip_meta = clip_roi["clip_meta"]

        # output debug video once 10 times
        # if i%10 == 0:
        if True:
            output_dbg_video = True

        for APO in APOs:  # per plane APO
            APO_0 = APO["APO_list"][0]  # First APO in this plane
            if APO_0["is_trackable"]:
                ad_roi = APO_0["ad_roi"]
                track_roi = APO_0["trackable_roi0"]
                
                # convert sfcake from proc_img_size to orig_img_size
                scale_x = clip_meta["scale_x"]
                scale_y = clip_meta["scale_y"]
                ad_roi = (np.array(ad_roi) / [scale_x, scale_y]).tolist()
                track_roi = (np.array(track_roi) / [scale_x, scale_y]).tolist()
                
                tracked_ad_rois, ad_roi_occl, track_roi_occl = process_video(
                    tracker, vid_file, ad_roi, track_roi, output_dbg_video
                )
                if output_dbg_video:
                    output_dbg_video = False
                
                # print("ad_roi :", ad_roi)
                # print("track_roi :", track_roi)
                # print("tracked_ad_rois :", tracked_ad_rois)
                out_dict = {
                
                    "filename": filename,
                    "tracked_ad_rois": tracked_ad_rois,
                    "ad_roi_occl": ad_roi_occl,
                    "track_roi_occl": track_roi_occl,
                }
                out_json_record = json.dumps(out_dict, default=int, indent=1)
                if not first_line:
                    out_json_record = ",\n" + out_json_record
                else:
                    first_line = False
                op_json_fh.write(out_json_record)
        op_json_fh.flush()
    op_json_fh.write("]")
    op_json_fh.close()
    exit()

    # bb court
    if 0:
        track_points = np.array(
            [[400, 24], [770, 34], [770, 88], [400, 70]], dtype=np.float32
        )
        roi_points = np.array(
            [[500, 100], [900, 100], [900, 190], [505, 200]], dtype=np.float32
        )
    else:
        # "ad_bb": [1046, 479, 1262, 641], "trackable_bb": [544, 32, 768, 256]
        track_points = np.array(
            [[544, 32], [768, 32], [768, 256], [544, 256]], dtype=np.float32
        )
        roi_points = np.array(
            [[1046, 479], [1262, 479], [1262, 641], [1046, 641]], dtype=np.float32
        )

    # pita_s_bobi
    # roi_points_f = np.array([[1375,350],[1704,358],[1700,498],[1375,498]]).astype(np.int32)
    # pita_s_bobi_1
    # roi_points_f = np.array([[53,126],[338,120],[342,190],[53,196]]).astype(np.int32)
    # roi_points_f = np.array([[1610,200],[1820,215],[1810,302],[1600,280]]).astype(np.int32)
    # roi_points_f = np.array([[453,444],[652,438],[682,474],[456,488]]).astype(np.int32)
    # roi_points_f = np.array([[690,478],[790,472],[1030,680],[870,700]]).astype(np.int32)
    # roi_points_f = np.array([[978,95],[1228,100],[1220,270],[970,268]]).astype(np.int32)
    # roi_points_f = np.array([[1290,268],[1668,258],[1662,393],[1291,388]]).astype(np.int32)

    tracker.reset_template_cache()

    cap = cv2.VideoCapture(args.vid)

    ret, img0 = cap.read()
    plt.imshow(img0)
    plt.show()
    img_H, img_W = img0.shape[:2]
    if ret is None:
        raise Exception("Empty video file")

    # create the ad mask
    ad_img_pts = np.array(
        [[0, 0], [ad_img_W - 1, 0], [ad_img_W - 1, ad_img_H - 1], [0, ad_img_H - 1]],
        dtype=np.float32,
    )
    ad_mask = np.zeros((img_H, img_W), dtype=np.uint8)
    cv2.fillPoly(ad_mask, [roi_points.astype(np.int32)], [255], lineType=cv2.LINE_AA)

    roi_plane_bb = np.array([[0, 0], [img_W - 1, img_H - 1]])

    update_template = False
    cv2.imshow("tracking", img0)
    cv2.waitKey(50)
    print(img_H, img_W)

    ######################################################################
    # Setup optical flow module
    ######################################################################

    flow_yx = None
    op_dir = "op/tracking"
    # dbg_dir = "debug/tracking/roi_frames"
    # os.makedirs(f"{dbg_dir}", exist_ok=True)
    os.makedirs(f"{op_dir}", exist_ok=True)

    track_points0 = track_points.copy().astype(np.int32)
    roi_points0 = roi_points.copy().astype(np.int32)
    S_prev = None

    idx = 0
    img_prev = img0
    while True:
        ret, img1 = cap.read()
        img1_copy = img1.copy()
        if not ret:
            print("stream end? Exiting ...")
            break

        Hg, Hg_ocl, is_fine = tracker.run(
            img_f,
            img0,
            img1,
            track_points0,
            roi_points0,
            None,
            update_template,
            idx,
            "dbg_logs",
        )

        roi_points0_ = roi_points0.astype(np.float32)
        roi_points1 = cv2.perspectiveTransform(roi_points0_[np.newaxis, :, :], Hg)[0]
        roi_points1 = np.round(roi_points1).astype(int)

        # add a 3 pixel thickness black border to ad_img
        if True:  # not is_roi_empty:
            ad_img_pil = ad_manager.ad_frames[idx % len(ad_manager.ad_frames)]
            ad_img = cv2.cvtColor(np.array(ad_img_pil), cv2.COLOR_RGBA2BGRA)
            ad_img = cv2.copyMakeBorder(
                ad_img, 7, 7, 7, 7, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
            )

        ad_mask = np.zeros((img_H, img_W), dtype=np.uint8)
        cv2.fillPoly(ad_mask, [roi_points1], [255])

        cv2.fillPoly(img1, [roi_points1], [255])
        img1_blended = img1

        cv2.imshow("tracking", img1_blended)
        cv2.waitKey(1)
        cv2.imwrite(f"{op_dir}/frame_{idx:03d}.png", img1_blended)

        idx += 1
        img_prev = img1
        update_template = False


if __name__ == "__main__":
    main()
