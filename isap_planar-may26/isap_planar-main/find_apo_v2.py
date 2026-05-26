import os
import argparse
import torch
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import glob
import re
import json
import subprocess
import time
from PIL import Image
import tqdm
import colorama 
#sys.path.append("models/third_party/TinySAM")

#from models.third_party.TinySAM.demo_hierachical_everything import show_anns_rect

# RectangleDetection utilities
sys.path.insert(0, "utils/RectangleDetection")
from mask_filter_utils import (
    create_parent_upward_mask,
    create_parent_downward_mask,
    filter_masks_by_dsine_variance,
    get_person_mask_from_yolo,
    remove_outliers,
)
from rectangle_detection import filter_masks_with_lcnn
from vis_utils import visualize_lcnn_masks

from model_loader import load_models, OneFormerSemanticSegmentation


# from robustpoint.robustpoint_top import RobustPointTop
from image_util import draw_matches, draw_keypoints
from lcnn.infer_lcnn import infer_single_image as lcnn_infer
from fit_bb_2_roi import get_rois, expand_bb, show_proposals, check_if_bb_surrounded_by_kpts

from image_util import show_sam_anns, draw_matches, draw_keypoints
from seg_planes import seg_depth, detect_humans, draw_bb_roi
from warp_bbox_v2 import warp_bbox_using_depth_v1, update_X_and_Y_axes
# from tracker import PlaneTracker, ROITrackerCoarse
from tracker_romav2 import PlaneTracker, ROITracker
from ultralytics import YOLO

global DETECT_RECTANGLE

DBG_DIR = "staging_dir/debug/apo"
MIN_EMPTY_AREA = 20000
class FAPO_Config:
    """Hyperparameters """
    TRACKABLE_RATIO = 0.95
    HG_MAT_MAX_SCALE_OR_ROT = 0.1
    MAX_NUM_KEYPOINTS = 4096 * 2
    PEOPLE_MASK_THRESH = 0.6
    MIN_CLIP_DURATION_SEC = 2


class YOLOPersonDetector:
    def __init__(self, model=None, model_name="yolo26n-seg.pt", device="cuda:0"):
        if model is None:
            self.model = YOLO(model_name)
            self.model.to(device)
        else:
            self.model = model
        self.device = device

    def detect_human(self, img):
        results = self.model.predict(
            source=img, classes=[0], verbose=False
        )  # Class 0 is 'person' in the COCO dataset
        people_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # people_box_mask = np.zeros(img_.shape[:2], dtype=np.uint8)
        found = False

        # Process results
        for result in results:
            if result.masks is not None:
                masks = result.masks.xy  # Segmentation masks as (x, y) points
                found = True
                for i, mask in enumerate(masks):
                    # Create a binary mask for the person
                    b_mask = np.zeros(result.orig_img.shape[:2], np.uint8)
                    # Reshape the mask points for cv2.drawContours
                    contour = mask.astype(np.int32).reshape(-1, 1, 2)
                    # Draw the contour on the binary mask
                    cv2.drawContours(b_mask, [contour], -1, 1, cv2.FILLED)

                    people_mask = cv2.bitwise_or(b_mask, people_mask)
        return found, people_mask


def draw_lines(img, lines, color=1, th=1):
    for i in range(len(lines)):
        pt1, pt2 = lines[i]

        x1, y1 = pt1
        x2, y2 = pt2

        # Convert to integer coordinates
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw on output image
        # cv2.line(output_image, (x1, y1), (x2, y2), 1, 1, lineType=16)  # red
        cv2.line(img, (x1, y1), (x2, y2), color, th)
    return img


def get_all_lines(
    img, depth_img, sn_img, kp_extractor, lcnn_model, device, debug=False
):

    depth_rgb = np.repeat(depth_img[:, :, np.newaxis], 3, axis=2)

    # detecting lines
    im_lines, sn_lines, depth_lines = [], [], []
    im_lines = lcnn_infer(lcnn_model, device, img, thresholds=[0.99])
    im_lines = np.array(im_lines)
    if len(im_lines):
        im_lines = im_lines[:, :, ::-1]  # y,x to x,y order

    sn_lines = lcnn_infer(lcnn_model, device, sn_img, thresholds=[0.99])
    sn_lines = np.array(sn_lines)
    if len(sn_lines):
        sn_lines = sn_lines[:, :, ::-1]  # y,x to x,y order

    depth_lines = lcnn_infer(lcnn_model, device, depth_rgb, thresholds=[0.99])
    depth_lines = np.array(depth_lines)
    if len(depth_lines):
        depth_lines = depth_lines[:, :, ::-1]  # y,x to x,y order

    if debug:
        img_vis = img.copy()
        img_vis = draw_lines(img_vis, im_lines, color=(255, 0, 0), th=3)
        img_vis = draw_lines(img_vis, sn_lines, color=(0, 255, 0), th=2)
        img_vis = draw_lines(img_vis, depth_lines, color=(0, 0, 255), th=1)
        img_vis = draw_keypoints(img_vis, kpts, radius=1, color=(0, 255, 0))

    all_lines = np.vstack(
        [im_lines.reshape(-1, 4), sn_lines.reshape(-1, 4), depth_lines.reshape(-1, 4)]
    )
    all_lines = all_lines.reshape(-1, 2, 2)

    return all_lines


def debug_vis_all_planes(img, planes, kpts, filename, out_dir):

    for i, plane in enumerate(planes):
        apo_present = False

        img_vis = np.ascontiguousarray(img.copy()[:,:,::-1])

        if "kpts" in plane:
            plane_kpts_list = plane['kpts']
            if len(plane_kpts_list):
                plane_kpts = plane_kpts_list[0]
                if len(plane_kpts) > 2:
                    img_vis = draw_keypoints(img_vis, plane_kpts.astype(np.int32), radius=3, color=(255, 255, 0))


        if "mask_unoccl" in plane:
            plane_mask = plane["mask_unoccl"][0]
            empty_mask = plane["empty_region_unoccl"] 
            dbg_str = "Unoccl. " + plane["dbg_str"]  
        else:
            plane_mask = plane["segmentation"].astype(bool) 
            empty_mask = plane["empty_region"] 
            dbg_str = plane["dbg_str"]  


        # we will draw only the first bb
        if "ad_rois" in plane:
            apo_present = True
            ad_rois = plane["ad_rois"]
            if "warped_rois" in plane and len(plane["warped_rois"]):
                ad_rois = plane["warped_rois"]
            ad_rois = np.array(ad_rois).reshape(-1, 4, 2)
            x0, y0 = ad_rois[0,0]
            x1, y1 = ad_rois[0,2]
            img_vis = cv2.rectangle(img_vis, (x0, y0), (x1, y1), (100, 255, 255), 3)

        cv2.putText(img_vis, dbg_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        fig, ax = plt.subplots(nrows=2, figsize=(12,10))
        ax[0].set_title("Un occluded plane mask")
        ax[0].imshow(img_vis)
        show_proposals(img_vis, plane_mask, ax[0])

        ax[1].set_title("Un occluded empty mask")
        ax[1].imshow(img_vis)
        show_proposals(img_vis, empty_mask, ax[1])

        for axis in ax.ravel():
            axis.axis("off")
        plt.tight_layout()

        if filename is not None:
            if apo_present:
                dbg_filename = re.sub(r".png", f"_APO_present_{i}.jpg", filename)
            else:
                dbg_filename = re.sub(r".png", f"_APO_absent_{i}.jpg", filename)
            save_file = os.path.join(out_dir, f"{dbg_filename}")
            plt.savefig(save_file)
            plt.close()
        else:
            plt.show()
    return


# We will do this check on a single ROI
def check_if_rois_are_sane(tracked_roi, img_shape):
    res = True
    img_H, img_W = img_shape
    # maximum distance any point travels in x/y direction
    max_dx = 0
    max_dy = 0
    n_frames = tracked_roi.shape[0]

    tracked_pts = tracked_roi.reshape(n_frames,-1,2)

    dx = tracked_pts[0:n_frames-1,:,0] - tracked_pts[1:,:,0]
    dy = tracked_pts[0:n_frames-1,:,1] - tracked_pts[1:,:,1]
    print(dx.shape, dy.shape, n_frames)
    # double derivative
    ddx = dx[0:n_frames-2] - dx[1:]
    ddy = dy[0:n_frames-2] - dy[1:]

    if False:
        fig, ax = plt.subplots(nrows=2, ncols=4)
        for i in range(4):
            ax[0,i].scatter(np.arange(n_frames-1), dx[:,i])
            ax[0,i].scatter(np.arange(n_frames-1), dy[:,i])
            ax[1,i].scatter(np.arange(n_frames-2), ddx[:,i])
            ax[1,i].scatter(np.arange(n_frames-2), ddy[:,i])
        plt.show()

    max_accl_x = np.max(np.abs(ddx)) 
    max_accl_y = np.max(np.abs(ddy)) 
    if max_accl_x > 20 or max_accl_y > 20:
        print("deleting this APO")
        res = False
    return res






##### KNR:TODO need to handle the case where we don't get Hg for more than a
# few frame. There is no need to continue with the full video in such case
def track_ad_rois(roi_tracker, ref_img, vid_file, plane_mask, rois, output_dbg_video=True, plane_idx=0):
    is_trackable=True
    tracked_count = 0

    #img0 = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
    img0 = ref_img
    img_H, img_W = img0.shape[:2]

    roi_tracker.set_img0(img0, plane_mask, rois, dbg_dir=DBG_DIR)


    cap = cv2.VideoCapture(vid_file)

    if output_dbg_video:
        hg_type = "coarse"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        _, filename = os.path.split(vid_file)
        filename, file_xtn = os.path.splitext(filename)
        dbg_out_file = f"{DBG_DIR}/{filename}_ad_in_plane_{plane_idx}_{hg_type}{file_xtn}"
        vid_out = cv2.VideoWriter(dbg_out_file, fourcc, 30.0, (img_W, img_H))
        print(f"writing to file {dbg_out_file}")

    frame_idx = 0
    rois_i = rois
    tracked_rois = []
    Hg_list = []
    prev_Hg = np.eye(3)
    prev_res = []
    frame_idx_ = None
    while True:
        ret, img1 = cap.read()
        if not ret:
            print("stream end? Exiting ...")
            break


        if output_dbg_video:
            frame_idx_ = frame_idx
        res, Hg = roi_tracker.run(img1, idx=frame_idx_)
        # when the number of keypoints is less then we return empty array.
        # Let us use the previous results in that case, we update the
        # rois_i only if we get vald results
        if len(res) > 0:
            tracked_count +=1
            prev_Hg = Hg
            prev_res = res
        else:
            Hg = prev_Hg
            res = prev_res
        Hg_list.append(Hg)
        rois_i = res

        if output_dbg_video:
            roi_i0 = rois_i[0:4,:]
            img1 = cv2.fillPoly(img1, [roi_i0], (0, 0, 255))
                
            # debug: for the first image let us plot the plane mask also 
            if frame_idx==0:
                img_vis = img1.copy()
                img_vis[plane_mask] //= 2
                vid_out.write(img_vis)
            else:
                vid_out.write(img1)
        tracked_rois.append(rois_i)

        frame_idx = frame_idx+1

    if output_dbg_video:
        vid_out.release()
    cap.release
    ########################################################################## 
    # Run some statistics to ensure the roi is trackable
    ########################################################################## 

    # the number of untrackable frames must be very less < 5 %
    ########################################################################## 
    trackable_ratio = tracked_count/frame_idx
    if trackable_ratio < FAPO_Config.TRACKABLE_RATIO:
        is_trackable = False

    # Check if the homographies of all frames make sense
    ########################################################################## 
    # We will do processing on only the first 2 rows of Homography matrix
    Hg_0to6_arr = np.array(Hg_list).reshape(-1,9)[:,:6]
    Hg_0to6_mean = np.mean(Hg_0to6_arr, axis=0)
    Hg_0to6_std = np.std(Hg_0to6_arr, axis=0)

    scale_or_rot = Hg_0to6_std.reshape(2,3)[:,:2].reshape(-1)

    # if the scale or rotation is beyond some amount 
    if np.max(scale_or_rot) > FAPO_Config.HG_MAT_MAX_SCALE_OR_ROT:
        is_trackable = False

    tracked_rois = np.array(tracked_rois)
    return tracked_rois, is_trackable, tracked_count, Hg_0to6_std



def get_empty_region(img, yolo_model, fe_model, device):
    img_blur = cv2.medianBlur(img, 3)
    img_t = torch.tensor((img_blur / 255.0).transpose(2, 0, 1), dtype=torch.float32).to(
        device
    )
    feats = fe_model(img_t.unsqueeze(0))
    print(feats["feats"].shape)
    feats = feats["feats"][0].sum(dim=0).detach().cpu().numpy().astype(np.uint8)
    # feats = cv2.medianBlur(feats, 3)

    # feats_bin = cv2.adaptiveThreshold(feats,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)
    ret2, feats_bin = cv2.threshold(feats, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    feats_bin = cv2.resize(feats_bin, (img.shape[1], img.shape[0])).astype(bool)

    people_mask, people_box_mask = detect_humans(yolo_model, img)

    empty_region_mask = np.logical_and(1 - feats_bin, 1 - people_mask)

    return empty_region_mask, people_mask


def debug_vis_rectangle_rois(img, sn_img, lcnn_masks, kpts, filename, out_dir):
    """Save a visualisation of the detected rectangular ROIs to disk.

    Draws each detected rectangle's four side-lines and its corner points on a
    2×2 grid: image+lines, mask, image+keypoints, surface-normal image.
    Output is saved to <out_dir>/<filename stem>_rect_rois.jpg.
    Nothing is displayed interactively.
    """
    if len(lcnn_masks) == 0:
        return

    import matplotlib
    matplotlib.use("Agg")   # ensure off-screen rendering

    stem = re.sub(r"\.(png|jpg)$", "", filename, flags=re.IGNORECASE)
    save_path = os.path.join(out_dir, f"{stem}_rect_rois.jpg")

    colors_per_side = {'left': 'red', 'right': 'blue', 'up': 'green', 'down': 'orange'}

    for mask_idx, mask_dict in enumerate(lcnn_masks):
        x, y, w, h = mask_dict['bbox']
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        mask      = mask_dict['segmentation'].astype(np.uint8)
        side_lines = mask_dict['side_lines']
        side_scores = mask_dict['side_scores']

        # Keypoints inside the bounding box
        mask_kpts = kpts[
            (kpts[:, 0] >= x1) & (kpts[:, 0] <= x2) &
            (kpts[:, 1] >= y1) & (kpts[:, 1] <= y2)
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Rect ROI #{mask_idx}  |  {filename}", fontsize=11)

        # --- panel 1: image + 4 side lines ---
        ax = axes[0, 0]
        ax.imshow(img)
        for side_name, line in side_lines.items():
            if line is not None:
                (y_p1, x_p1), (y_p2, x_p2) = line[0], line[1]
                score = side_scores.get(side_name, 0.0)
                ax.plot([x_p1, x_p2], [y_p1, y_p2],
                        color=colors_per_side[side_name], linewidth=3,
                        label=f"{side_name} ({score:.2f})", zorder=10)
                ax.scatter([x_p1, x_p2], [y_p1, y_p2], s=30,
                           c=colors_per_side[side_name], edgecolors='white',
                           linewidth=1.5, zorder=11)
        rect_patch = plt.Rectangle((x1, y1), w, h, linewidth=2,
                                   edgecolor='cyan', fill=False, linestyle='--', alpha=0.7)
        ax.add_patch(rect_patch)
        ax.set_title("Detected rectangle lines")
        ax.legend(loc='upper right', fontsize=9)
        ax.axis('off')

        # --- panel 2: segmentation mask ---
        ax = axes[0, 1]
        ax.imshow(mask, cmap='gray')
        ax.set_title("Segmentation mask")
        ax.axis('off')

        # --- panel 3: image + keypoints ---
        ax = axes[1, 0]
        ax.imshow(img)
        x_mid, y_mid = (x1 + x2) / 2, (y1 + y2) / 2
        ax.axvline(x=x_mid, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=y_mid, color='purple', linestyle='--', linewidth=1, alpha=0.5)
        if len(mask_kpts) > 0:
            ax.scatter(mask_kpts[:, 0], mask_kpts[:, 1], s=10,
                       c='red', alpha=0.7, edgecolors='yellow', linewidth=0.5)
        kd = mask_dict.get('keypoint_dist', {})
        title = (f"Keypoints ({len(mask_kpts)})  "
                 f"L:{kd.get('vert_num_of_point_left','-')} "
                 f"R:{kd.get('vert_num_of_point_right','-')}")
        ax.set_title(title)
        ax.axis('off')

        # --- panel 4: surface normal (DSINE) ---
        ax = axes[1, 1]
        ax.imshow(sn_img)
        ax.set_title("Surface normal (DSINE)")
        ax.axis('off')

        plt.tight_layout()
        p = os.path.join(out_dir, f"{stem}_rect_roi_{mask_idx}.jpg")
        plt.savefig(p, bbox_inches='tight', dpi=100)
        plt.close(fig)

    print(f"[rect] saved {len(lcnn_masks)} rect-ROI debug image(s) → {out_dir}/{stem}_rect_roi_*.jpg")


def get_rectangle_roi(img, sn_img, yolo_model, sam_mask_generator, kp_extractor, lcnn_model, device):

    # 1. SAM masks
    masks = sam_mask_generator.hierarchical_generate(img)

    # 2. Person mask (dilated)
    
    kernel = np.ones((10, 10), np.uint8)
    person_mask_uint8 = get_person_mask_from_yolo(img, yolo_model)
    person_mask_uint8 = cv2.dilate(person_mask_uint8, kernel, iterations=2)
    person_mask = person_mask_uint8.astype(bool)

    # 3. Remove non-connected masks
    masks, _, _ = remove_outliers(masks, min_component_ratio=0.01)

    # 4. Floor / ceiling exclusion masks
    parent_upward_mask   = create_parent_upward_mask(masks, sn_img, normal_threshold=0.6)
    parent_downward_mask = create_parent_downward_mask(masks, sn_img, normal_threshold=0.6)

    exclusion_mask = (person_mask |
                      parent_upward_mask.astype(bool) |
                      parent_downward_mask.astype(bool))
    exclusion_mask = person_mask
    # 5. Filter by overlap with exclusion mask (>60% → reject)
    filtered_masks = []
    removed_masks = []
    for idx, mask_dict in enumerate(masks):
        seg = mask_dict['segmentation']
        if seg.shape != exclusion_mask.shape:
            seg = cv2.resize(seg.astype(np.uint8),
                             (exclusion_mask.shape[1], exclusion_mask.shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        valid = seg.astype(bool)
        total = int(valid.sum())
        if total == 0:
            continue
        if int(exclusion_mask[valid].sum()) / total <= 0.6:
            filtered_masks.append(mask_dict)
        else:
            removed_masks.append((idx, mask_dict,  int(exclusion_mask[valid].sum())))

    if 0:

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        axes[0, 0].imshow(exclusion_mask, cmap='gray')
        axes[0, 0].set_title(f'people mask', fontsize=12)
        axes[0, 0].axis('off')

        axes[0, 1].set_title(f'All Masks ({len(masks)})', fontsize=12)
        axes[0, 1].axis('off')
        show_anns_rect(img, masks, ax=axes[0, 1])

        axes[1, 0].set_title(f'Filtered Masks: {len(filtered_masks)}', fontsize=12)
        axes[1, 0].axis('off')
        if len(filtered_masks) > 0:
            show_anns_rect(img, filtered_masks, ax=axes[1, 0])
        else:
            axes[1, 0].imshow(img)

        axes[1, 1].set_title(f'Removed Masks: {len(removed_masks)}', fontsize=12)
        axes[1, 1].axis('off')
        removed_masks_only = [m[1] for m in removed_masks]
        if len(removed_masks_only) > 0:
            show_anns_rect(img, removed_masks_only, ax=axes[1, 1])
        else:
            axes[1, 1].imshow(img)
        
        plt.tight_layout()
        plt.show()
        
    # 6. DSINE variance filter (keep uniform/planar regions)
    low_var_masks, _ = filter_masks_by_dsine_variance(
        filtered_masks, sn_img, variance_threshold=0.03
    )


    # 7. Keypoint distribution filter (≥2 kpts on each of 4 sides)
    img_t = torch.tensor(
        (img / 255.0).transpose(2, 0, 1), dtype=torch.float32
    ).to(device)
    feats = kp_extractor(data={"image": img_t})
    kpts = feats["keypoints"].detach().cpu().numpy()[0]  # (N, 2) x,y

    threshold_num_points = 2
    point_distribution_masks = []
    for mask_dict in low_var_masks:
        x, y, w, h = mask_dict['bbox']
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        in_bbox = kpts[
            (kpts[:, 0] >= x1) & (kpts[:, 0] <= x2) &
            (kpts[:, 1] >= y1) & (kpts[:, 1] <= y2)
        ]
        if len(in_bbox) == 0:
            continue
        x_mid, y_mid = (x1 + x2) / 2, (y1 + y2) / 2
        n_l = int(np.sum(in_bbox[:, 0] <  x_mid))
        n_r = int(np.sum(in_bbox[:, 0] >= x_mid))
        n_u = int(np.sum(in_bbox[:, 1] <  y_mid))
        n_d = int(np.sum(in_bbox[:, 1] >= y_mid))
        if n_l >= threshold_num_points and n_r >= threshold_num_points \
                and n_u >= threshold_num_points and n_d >= threshold_num_points:
            md = mask_dict.copy()
            md['keypoint_dist'] = {
                'vert_num_of_point_left':  n_l,
                'vert_num_of_point_right': n_r,
                'hori_num_of_point_up':    n_u,
                'hori_num_of_point_down':  n_d,
                'total_keypoints_in_bbox': len(in_bbox),
            }
            point_distribution_masks.append(md)
    # 8. LCNN rectangle confirmation
    lcnn_masks, _, _ = filter_masks_with_lcnn(
        lcnn_model,
        device,
        point_distribution_masks,
        img,
        person_mask_uint8,
        score_threshold=0.95,
        apply_postprocess=False,
        dsine_image=sn_img,
    )   


    # # final masks with pred rects
    # if len(lcnn_masks) > 0:
    #     visualize_lcnn_masks(lcnn_masks, img, kpts, dsine_image=sn_img, top_k=1)


    return lcnn_masks


def _lcnn_mask_to_corners(mask_dict):
    """Extract corner points [TL, TR, BR, BL] in (x, y) order from an lcnn_mask dict.

    `side_lines` stores points as (y, x) (row, col) — this function flips each
    corner to (x, y) so it matches the convention expected by the tracker and
    cv2.perspectiveTransform (same as convert_bb_2_rois output).

    Returns a (4, 2) numpy array of float corners in (x, y) order, or None.
    """
    side_lines = mask_dict.get('side_lines')
    if side_lines is None:
        return None
    # side_lines format from refine_corners_by_detection (points are (y, x)):
    #   'up':    [TL, TR]   → row 0 is TL, row 1 is TR
    #   'down':  [BL, BR]   → row 0 is BL, row 1 is BR
    #   'left':  [TL, BL]
    #   'right': [TR, BR]
    try:
        def yx_to_xy(pt):
            """Flip (y, x) → (x, y)."""
            a = np.asarray(pt, dtype=float)
            return np.array([a[1], a[0]])

        TL = yx_to_xy(side_lines['up'][0])
        TR = yx_to_xy(side_lines['up'][1])
        BR = yx_to_xy(side_lines['down'][1])
        BL = yx_to_xy(side_lines['down'][0])
        return np.stack([TL, TR, BR, BL], axis=0)  # (4, 2) — TL, TR, BR, BL in (x, y)
    except (KeyError, IndexError, TypeError):
        return None


def process_clip(
    img,
    depth_img,
    pts_3d,
    sn_img,
    device,
    vp_detector,
    kp_extractor,
    sam_mask_generator,
    lcnn_model,
    yolo_model,
    fe_model,
    filename,
    out_dir=None,
):


    #####################################################################
    # 2. Get planes
    #####################################################################
    save_file = False
    depth_segs, img_segs, _ = seg_depth(
        kp_extractor,
        sam_mask_generator,
        None,
        yolo_model,
        img,
        depth_img,
        pts_3d,
        sn_img,
        device,
        filename=filename,
        dbg_dir=DBG_DIR,
    )
    if len(depth_segs) == 0:
        return [], [], None

       
    return depth_segs, img_segs, pts_3d


def gen_sn_images(staging_dir, sn_img_folder):
    command = [sys.executable, "run_surface_normal.py", "--imgs",
               f"{staging_dir}/clip_frames/", "--op_dir", f"{sn_img_folder}"]
    print(f"Running command {command}")
    try:
        # Run the command and wait for it to complete
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Script failed with exit code {e.returncode}")
        print("Error output:", e.stderr)
        exit()

def gen_watermark_bboxes(staging_dir, wm_bbox_json_file, wm_bbox_folder):
    command = [sys.executable, "run_watermark_detection.py", "--imgs",
               f"{staging_dir}/clip_frames/", "--op_json", wm_bbox_json_file, "--op_bbox_dir", wm_bbox_folder]
    print(f"Running command {command}")
    try:
        # Run the command and wait for it to complete
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f'Watermark bounding boxes saved to {wm_bbox_json_file}')
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Script failed with exit code {e.returncode}")
        print("Error output:", e.stderr)




def plot_available_empty_planes(img, plane_segs, empty_mask, save_file=None):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    img_vis = np.ones((img.shape[0], img.shape[1], 4))
    img_vis[:, :, 3] = 0
    color_mask = np.array([1, 0, 0, 0.30])
    img_vis[empty_mask] = color_mask
    fig.suptitle('FAPO::Empty regions')
    ax[0, 0].set_title("image")
    ax[0, 0].imshow(img)
    ax[0, 1].set_title("empty region")
    ax[0, 1].imshow(img)
    ax[0, 1].imshow(img_vis)
    ax[1, 0].set_title("sam plane segments")
    ax[1, 0].imshow(img)
    show_sam_anns(plane_segs, ax[1, 0], show_idx=True)

    ax[1, 1].set_title("planar regions available for ad placement")
    ax[1, 1].imshow(img)
    show_sam_anns(plane_segs, ax[1, 1], show_idx=True, valid_mask=empty_mask)
    for axis in ax.ravel():
        axis.axis("off")
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()
    plt.close()
    return


def classify_indoor_outdoor_oneformer(oneformer_model, img):
    json_path = "assets/config/oneformer_indoor_outdoor.json"
    with open(json_path, "r") as f:
        classes_data = json.load(f)
    
    indoor_1_classes = set(classes_data.get("indoor_1", []))
    indoor_3_classes = set(classes_data.get("indoor_3", []))
    outdoor_1_classes = set(classes_data.get("outdoor_1", []))
    outdoor_3_classes = set(classes_data.get("outdoor_3", []))
    
    if isinstance(img, np.ndarray):
        original_img = img.copy()
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        original_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_pil = img
        
    seg_map, id2label = oneformer_model.predict(img_pil)
    
    detected_classes = np.unique(seg_map)
    out_img = original_img.copy()
    
    indoor_count = 0
    outdoor_count = 0
    centers_to_label = []
    person_mask = (seg_map == 13).astype(np.uint8)

    for cls_idx in detected_classes:
        if cls_idx == 0: continue
        
        if cls_idx in indoor_1_classes:
            indoor_count += 1
            color = (0, 255, 0)
        elif cls_idx in indoor_3_classes:
            indoor_count += 3
            color = (0, 255, 0)
        elif cls_idx in outdoor_1_classes:
            outdoor_count += 1
            color = (0, 0, 255)
        elif cls_idx in outdoor_3_classes:
            outdoor_count += 3
            color = (0, 0, 255)
        else:
            continue
            
        mask = (seg_map == cls_idx).astype(np.uint8)
        out_img[mask == 1] = out_img[mask == 1] * 0.9 + np.array(color) * 0.1
        
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            label_name = id2label.get(cls_idx, id2label.get(str(cls_idx), "Unknown"))
            centers_to_label.append((cX, cY, label_name))
            
    final_class = "indoor" if indoor_count > outdoor_count else "outdoor"
    if indoor_count == 0 and outdoor_count == 0:
        final_class = "outdoor"
        
    for (cX, cY, label_name) in centers_to_label:
        cv2.putText(out_img, label_name, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
    pad_h = 50
    final_img = cv2.copyMakeBorder(out_img, pad_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    text_size = cv2.getTextSize(final_class, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    tX = (final_img.shape[1] - text_size[0]) // 2
    cv2.putText(final_img, final_class, (tX, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    # add person percetage on the top left corner
    person_percentage = (person_mask.sum() / (person_mask.shape[0] * person_mask.shape[1])) * 100
    person_text = f"Person: {person_percentage:.1f}%"
    cv2.putText(final_img, person_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return final_class, final_img, person_mask

def get_occlusions(ref_img, clips, plane_mask, empty_mask, yolo_model,
                      plane_tracker, plane_idx, wm_bbox_json_file, dbg_dir, process_interval=5):
    #we wil have one empty unoccluded region for the whole clip, multipe (one
    # per clip) unoccluded plane region
    person_detector =  YOLOPersonDetector(yolo_model)

    #img0 = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
    img0 = ref_img
    img_H, img_W = img0.shape[:2]

    knl = np.ones((7, 7), np.uint8)

    plane_mask_unoccl = np.ones([len(clips), img_H, img_W], dtype=bool)
    empty_mask_unoccl = np.ones([img_H, img_W], dtype=bool)

    #dilate the plane mask to include boundary regions. This will help with
    # tracking this plane across multiple clips of the scene
    plane_mask_dil = cv2.dilate(plane_mask.astype(np.uint8), knl, iterations=2)

    empty_mask_unoccl = np.logical_and(empty_mask_unoccl, empty_mask.astype(bool))
    #for i in range(len(clips)):
    #    plane_mask_unoccl[i] = np.logical_and(plane_mask_unoccl[i], plane_mask_dil.astype(bool))

    plane_tracker.set_img0(img0, plane_mask_dil, empty_mask)

    # read bounding box json file
    if 0: #will enable later. Checkpoint is missing
        with open(wm_bbox_json_file, 'r') as f:
            wm_bboxes_dict = json.load(f)
    else:
        wm_bboxes_dict = {}

    print(f"getting occlusions.....")
    for clip_id, vid_file in enumerate(tqdm.tqdm(clips)):
        _, vid_filename = os.path.split(vid_file)
        vid_filename = os.path.splitext(vid_filename)[0]
        dbg_filename = vid_filename+f"_{plane_idx}_matching.jpg"
        save_file = os.path.join(DBG_DIR, f"{dbg_filename}")
        cap = cv2.VideoCapture(vid_file)
        wm_bboxes = wm_bboxes_dict.get(vid_filename, [])
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx=0
        while True:
            ret, img1 = cap.read()
            if not ret:
                #print("stream end? Exiting ...")
                break

            #NOTE: uncomment to use watermark mask
            # watermark_mask = np.zeros(img1.shape[:2], dtype=bool)
            # if wm_bboxes != []:
            #     for bbox in wm_bboxes:
            #         x0, y0 = bbox[0]
            #         x1, y1 = bbox[2]
            #         watermark_mask[y0:y1, x0:x1] = 1

            if idx > 0 and idx % process_interval==0:
                people_found, people_mask = person_detector.detect_human(img1)

                # If any people are found, remove this region from plane_mask and empty mask
                if people_found:
    
                    Hg, plane_mask1, empty_mask1 = plane_tracker.run(img1, save_file)
                    save_file = None
    
                    if plane_mask1 is not None and Hg is not None:
                        people_mask = cv2.dilate(people_mask.astype(np.uint8), knl, iterations=1).astype(bool)
                        plane_mask1 = np.logical_and(plane_mask1, 1-people_mask)
                        empty_mask1 = np.logical_and(empty_mask1, 1-people_mask)
    
                        HgI = np.linalg.inv(Hg)
                        plane_mask1to0 = cv2.warpPerspective(plane_mask1.astype(np.uint8), HgI, (img_W,img_H),
                                                             borderMode=cv2.BORDER_REPLICATE,
                                                             flags=cv2.INTER_NEAREST)
                        empty_mask1to0 = cv2.warpPerspective(empty_mask1.astype(np.uint8), HgI, (img_W,img_H),
                                                             borderMode=cv2.BORDER_REPLICATE, 
                                                             flags=cv2.INTER_NEAREST)
                        plane_mask_unoccl[clip_id] = np.logical_and(plane_mask_unoccl[clip_id], plane_mask1to0.astype(bool))
                        empty_mask_unoccl = np.logical_and(empty_mask_unoccl, empty_mask1to0.astype(bool))
                        if 0:
                            fig, ax = plt.subplots(nrows=3)
                            ax[0].imshow(img1)
                            ax[1].imshow(empty_mask1to0)
                            ax[2].imshow(people_mask)
                            plt.show()
            idx = idx + 1
            # We will not process the first and last frame
            if idx+process_interval >= n_frames-1:
                break
        cap.release()
    # cleanup the plane_mask 
    for i in range(len(clips)):
        plane_mask_unoccl[i] = cv2.dilate(plane_mask_unoccl[i].astype(np.uint8), knl, iterations=2)
        plane_mask_unoccl[i] = cv2.erode(plane_mask_unoccl[i].astype(np.uint8), knl, iterations=2).astype(bool)
    return plane_mask_unoccl, empty_mask_unoccl


def get_lines_belonging_to_this_plane(plane_mask, lines):
    lines_in_plane = []
    for line in lines:
        line_img = np.zeros([plane_mask.shape[0], plane_mask.shape[1]], dtype=np.uint8)
        line_img = draw_lines(line_img, [line]).astype(bool)
        intrs_region = np.logical_and(line_img, plane_mask)
        # We need some good length of the line passing through the seg
        score = np.sum(intrs_region)/np.sum(line_img)
        if score > 0.7:
            lines_in_plane.append(line)
    return lines_in_plane




def main():
    parser = argparse.ArgumentParser(description="Main application")
    parser.add_argument("--staging_dir", type=str, help="Path to stagind dir", default="staging_dir")
    args = parser.parse_args()

    # Initialize to clear color after each print
    colorama.init(autoreset=True)

    # Set this to True to run rectangle/screen detection on each scene.
    # When a rectangle ROI is found its corners are used directly as the ad
    # placement region (bypassing the normal plane-based flow for that scene).
    DETECT_RECTANGLE = False

    APO_process_frame_num = 5

    if not os.path.exists(args.staging_dir):
        raise Exception(f"Error: Directory not present: {args.staging_dir}")

    video_info_json_file = glob.glob(args.staging_dir+"/*_video_info.json")
    if not video_info_json_file:
        error_str = f"Error: video info file _video_info.json not found\n"
        error_str += "You forgot running the scene_analyzer script?"
        raise Exception(error_str)

    video_info_json_file = video_info_json_file[0]

    op_apo_json_file = re.sub(r"_video_info.json", "_apo_info.json", video_info_json_file)

    os.makedirs(f"{DBG_DIR}/tracking", exist_ok=True)
    op_apo_fh =  open(op_apo_json_file, "w")
    op_apo_fh.write("[")


    with open(video_info_json_file, "r") as f:
        video_info = json.load(f)
    clips_info = video_info["clip_infos"]

    img_folder = os.path.join(args.staging_dir, "clip_frames")
    sn_img_folder = os.path.join(args.staging_dir, "sn")
    wm_bbox_folder = os.path.join(args.staging_dir, "watermark_bboxes")
    wm_bbox_json_file = re.sub(r"_video_info.json", "_watermarks_bboxes.json", video_info_json_file)

    if not os.path.exists(img_folder):
        raise Exception(f"Error: img dir not found: {img_folder}")

    if not os.path.exists(sn_img_folder):
        print("surface normals not found. Generating them. Please wait....")
        gen_sn_images(args.staging_dir, sn_img_folder)

    if not os.path.exists(wm_bbox_json_file):
        print("watermark bboxes not found. Generating them. Please wait....")
        gen_watermark_bboxes(args.staging_dir, wm_bbox_json_file, wm_bbox_folder)

    time.sleep(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{DBG_DIR}", exist_ok=True)

    vp_detector = None

    kp_extractor, sam_mask_generator, lcnn_model, yolo_model, fe_model, da_model = (
        load_models(True, True, True, True, True, True)
    )
    plane_tracker = PlaneTracker()
    roi_tracker = ROITracker()

    scene_cluster_info = video_info["scene_cluster_info"]

    first_line=True
    for grp_id, cluster in scene_cluster_info.items():
        print(f"\n\nProcessing groupd {cluster}")
        # Per scene roi info
        APO_info = {}
        APO_for_this_scene_found = False
        clip0_name = cluster[0]
        clip0_path = os.path.join(args.staging_dir, clip0_name)

        clip_name_noext = os.path.splitext(clip0_name)[0]
        img_filename = f"{clip_name_noext}_{APO_process_frame_num:03d}.png"
        img_file = os.path.join(img_folder, img_filename)
        sn_file = os.path.join(sn_img_folder, f"{clip_name_noext}_{APO_process_frame_num:03d}.png")

        img_bgr = cv2.imread(img_file)
        #img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = img_bgr

        sn_img = cv2.imread(sn_file)

        img_H, img_W = np.array(img).shape[:2]
        assert img_H == 720 and img_W == 1280, f"Video resolution not proper, expected (1290x480), got ({img_W},{img_H})"

        ########################### CAUTION::: ONLY FOR DEBUG RUNS ##########
        ## Continue running from previous pause/ctrl-c
        #####################################################################
        img_name_ = re.sub(r".png", "", img_filename)
        files = glob.glob(DBG_DIR+f"/{img_name_}*")
        if files:
            print(f"Skipping processing for {img_name_}")
            continue
        #######################################
        # check if the image is blank
        if np.sum(img) == 0 or np.sum(1-img) == 0:
            print("Blank image! Either completely white or completely black")
            continue
        print(f"{colorama.Fore.GREEN}Processing clip file {img_name_}")

        if True:
            ######################################################################
            # check if indoor or outdoor using oneformer
            ######################################################################
            oneformer_model = OneFormerSemanticSegmentation()
            indoor_outdoor_class, oneformer_vis, person_mask = classify_indoor_outdoor_oneformer(oneformer_model, img)
            oneformer_model.unload()
            print(f"{colorama.Fore.GREEN}Scene is {indoor_outdoor_class}")

            oneformer_vis_file = os.path.join(DBG_DIR, f"{img_name_}_indoor_outdoor_classification.png")
            cv2.imwrite(oneformer_vis_file, oneformer_vis)
    
            # skip if outdoor
            if indoor_outdoor_class == "outdoor":
                print(f"{colorama.Fore.RED}Skipping {img_name_}, outdoor scene")
                continue

            # skip if too many people are present
            if person_mask.sum() / (img_H*img_W) > 0.5:
                print(f"{colorama.Fore.RED}Skipping {img_name_}, people are occupying ({person_mask.sum() / (img_H*img_W):.2f}) % of the scene")
                continue


        # UniDepth inference
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
        with torch.no_grad():
            depth_preds = da_model.infer(rgb_tensor)
        depth_img = depth_preds["depth"].squeeze().cpu().numpy()

        pts_3d_img = depth_preds["points"].squeeze().permute(1, 2, 0).cpu().numpy()

        # rescale depth results to processed image size
        #depth_img = cv2.GaussianBlur(depth_img, (5, 5), 0)
        depth_img = cv2.resize(depth_img, (img_W, img_H), interpolation=cv2.INTER_AREA)

    
        ######################################################################
        # 2. detecting keypoints
        ######################################################################
        img_t = torch.tensor((img / 255.0).transpose(2, 0, 1), dtype=torch.float32).to(device)
        feats = kp_extractor(data={"image": img_t})
        kpts = feats["keypoints"].detach().cpu().numpy().astype(int)[0]

        ######################################################################
        # We will do plane processing only on the reference image
        ######################################################################
        planes, img_segs, _ = process_clip(img, depth_img, pts_3d_img,
                                      sn_img, device, vp_detector,
                                      kp_extractor, sam_mask_generator, 
                                      lcnn_model, yolo_model,
                                      fe_model, img_filename, out_dir=DBG_DIR)

        ######################################################################
        # Rectangle ROI detection (screens, photos, boards …)
        # When detect_rectangle=True and rectangles are found, their corners
        # are added directly as APO candidates and the normal plane flow is
        # skipped for this scene cluster.
        ######################################################################
        rect_masks = []
        if DETECT_RECTANGLE:
            print('DETECT_RECTANGLE')
            rect_masks = get_rectangle_roi(
                img, sn_img, yolo_model, sam_mask_generator,
                kp_extractor, lcnn_model, device
            )
        if DETECT_RECTANGLE and rect_masks:
            print(f"[rect] {len(rect_masks)} rectangular ROI(s) detected → using as APOs")
            debug_vis_rectangle_rois(img, sn_img, rect_masks, kpts,
                                        img_filename, DBG_DIR)

            clips_paths = [os.path.join(args.staging_dir, cn) for cn in cluster]
            rect_apo_list = []
            for rect_idx, mask_dict in enumerate(rect_masks):
                corners = _lcnn_mask_to_corners(mask_dict)
                if corners is None:
                    continue
                # corners: (4,2) in (x,y) order — TL, TR, BR, BL
                ad_rois = corners.reshape(-1, 2)  # (4,2)
                for clip_idx, clip_path in enumerate(clips_paths):

                    # plt.imshow(mask_dict['segmentation'].astype(np.uint8))
                    # plt.show()
                    tracked_rois, is_trackable, tracked_count, Hg_std = track_ad_rois(
                        roi_tracker, img, clip_path,
                        mask_dict['segmentation'].astype(bool),
                        ad_rois, plane_idx=rect_idx,
                    )

                    if is_trackable:
                        print("is is_trackable")
                        clip_name = cluster[clip_idx]
                        clip_info = clips_info.get(clip_name, {})
                        rect_apo_list.append({
                            "filename":        clip_name,
                            "start_frame":     clip_info.get("start_frame", 0),
                            "duration_frames": clip_info.get("duration_frames", 0),
                            "roi":             np.array(tracked_rois).tolist(),
                            "tracked_count":   tracked_count,
                            "Hg_0to6_std":     Hg_std.tolist(),
                            "rect_roi_index":  rect_idx,
                        })
                        APO_for_this_scene_found = True
                        print("\n+++++++++++++++APO (RECT) FOUND+++++++++++++++\n")

            if rect_apo_list:
                APO_info["APO_rect"] = rect_apo_list
            # Skip normal plane processing for this cluster since rect ROIs were found
            if APO_for_this_scene_found:
                
                # plane["dbg_str"] = "Skipped – rectangle ROI detected"
                # debug_vis_all_planes(img, planes, kpts, img_filename, DBG_DIR)
            
                json_record = json.dumps(APO_info, default=int, indent=1)
                if not first_line:
                    json_record = ",\n" + json_record
                else:
                    first_line = False
                op_apo_fh.write(json_record)
                op_apo_fh.flush()
                # continue  
        
        else:
            ######################################################################
            # 1. check empty region using a resnet based feature extractor
            ######################################################################
            empty_region_mask, people_mask = get_empty_region(img, yolo_model, fe_model, device)
            if True:
                dbg_filename = re.sub(r".png", "_empty_planes.jpg", img_filename)
                save_file = os.path.join(DBG_DIR, f"{dbg_filename}")
                plot_available_empty_planes(img[:,:,::-1], planes, empty_region_mask, save_file)


            clips_paths = []
            for clip_name in cluster:
                clip_path = os.path.join(args.staging_dir, clip_name)
                clips_paths.append(clip_path)

            ######################################################################
            # 2. Get all lines for this image
            ######################################################################
            all_lines = get_all_lines(img, depth_img, sn_img, kp_extractor, lcnn_model, device)

            for plane_idx, plane in enumerate(planes):
                plane_mask = plane["segmentation"].astype(bool)
                plane_lines = get_lines_belonging_to_this_plane(plane_mask, all_lines)
                plane["plane_lines"] = plane_lines

            update_X_and_Y_axes(img, depth_preds, pts_3d_img, planes, all_lines)

            ######################################################################
            # Per plane processing
            ######################################################################
            for plane_idx, plane in enumerate(planes):
                process_next_steps = True
                dbg_str = ""
                # Per plane roi info
                plane_apo_list = []
                plane_mask = plane["segmentation"].astype(bool)
                plane["APO_present"] = False

                ##################################################################
                # Plane must be of minimum size
                ##################################################################
                if plane_mask.sum() < MIN_EMPTY_AREA:
                    dbg_str = f"Min plane area failure:: plane area < min empty area :: {plane_mask.sum()} < {MIN_EMPTY_AREA}:"
                    process_next_steps = False

                plane_empty_mask = np.logical_and(empty_region_mask.astype(bool), plane_mask.astype(bool))

                plane["empty_region"] = plane_empty_mask

                if plane_empty_mask.sum() < MIN_EMPTY_AREA:
                    dbg_str = f"Min empty area failure:: plane empty area < min empty area :: {plane_empty_mask.sum()} < {MIN_EMPTY_AREA}:"
                    process_next_steps = False

                ##################################################################
                # Plane must be have minimum number of keypoints
                ##################################################################
                if process_next_steps:
                    # Check if there are minimum number of kpts in this plane
                    mask_ = np.logical_and(plane_mask, 1-people_mask)
                    kpts_mask = mask_[kpts[:, 1], kpts[:, 0]] == 1
                    plane_kpts = kpts[kpts_mask]
                    plane["kpts"] = plane_kpts

                    if len(plane_kpts) < 6:
                        dbg_str = f"Min plane kps failure:: {len(plane_kpts)} < 10:"
                        process_next_steps = False


                ##################################################################
                # Plane must have minimum un-occluded area
                ##################################################################
                if process_next_steps:
                    plane_empty_mask = np.logical_and(empty_region_mask.astype(bool), plane_mask.astype(bool))
                    plane_mask_unoccl, empty_mask_unoccl = get_occlusions(img, 
                                                                    clips_paths, 
                                                                    plane_mask, 
                                                                    plane_empty_mask, 
                                                                    yolo_model, 
                                                                    plane_tracker,
                                                                    plane_idx, 
                                                                    wm_bbox_json_file,
                                                                    DBG_DIR)
                    plane["mask_unoccl"] = plane_mask_unoccl
                    plane["empty_region_unoccl"] = empty_mask_unoccl

                    if empty_mask_unoccl.sum() < MIN_EMPTY_AREA:
                        dbg_str = f"Empty unoccluded area criterion failed :: {empty_mask_unoccl.sum()} < {MIN_EMPTY_AREA}:"
                        process_next_steps = False


                ##################################################################
                # Plane must have minimum un-occluded keypoints
                ##################################################################
                if process_next_steps:
                    plane_kpts = []
                    for i in range(plane_mask_unoccl.shape[0]):
                        kpts_mask = plane_mask_unoccl[i][kpts[:, 1], kpts[:, 0]] == 1
                        clip_i_plane_kpts  = kpts[kpts_mask].reshape(-1,2)
                        plane_kpts.append(clip_i_plane_kpts)
                        if len(clip_i_plane_kpts) <10:
                            dbg_str = f"Min unoccluded kpts failure for clip{i} :: {len(plane_kpts)} < 10:"
                            process_next_steps = False
                    plane["kpts"] = plane_kpts

                ##################################################################
                # get roi regions
                ##################################################################
                if process_next_steps:
                    ad_rois_list = get_rois(empty_mask_unoccl, min_wh=[100, 75], itrs=100)
                    if len(ad_rois_list) == 0:
                        dbg_str = "No rois failure"
                        process_next_steps = False

                ##################################################################
                # Process roi regions
                ##################################################################
                if 0: #process_next_steps:
                    ad_rois = np.array(ad_rois_list).reshape(-1,2)
                    ad_roi_bbs = ad_rois.reshape(-1,4,2)
                    ad_rois_list = []
                    for ad_roi_ in ad_roi_bbs:
                        bb_roi_ = np.array([ad_roi_[0], ad_roi_[2]])
                        if check_if_bb_surrounded_by_kpts(bb_roi_, kpts, plane_mask):
                            ad_rois_list.append(ad_roi_)
                    if len(ad_rois_list) == 0:
                        dbg_str = "rois not surrounded kpts failure"
                        process_next_steps = False

                if process_next_steps:
                    ad_rois_list = sorted(ad_rois_list,
                                    key=(lambda roi: (roi[2][0] - roi[0][0]) * (roi[2][1] - roi[0][1])),
                                    reverse=True)
                    ad_rois = np.array(ad_rois_list).reshape(-1,2)

                    plane["ad_rois"] = ad_rois_list
                    plane["APO_present"] = True
                    dbg_str = f"{len(ad_rois_list)} ad rois found"
                    warped_rois, res_str = warp_bbox_using_depth_v1(img, depth_preds, pts_3d_img, plane)
                    plane["warped_rois"] = warped_rois


                    if len(warped_rois):
                        w_ad_rois = np.array(warped_rois).reshape(-1,2)
                    else:
                        w_ad_rois = np.array(ad_rois_list).reshape(-1,2)

                    process_next_steps = True

                if process_next_steps:
                    tracking_failed_for_all_clips = True
                    tracking_successul_for_clips = []
                    for clip_idx in range(len(clips_paths)):
                        tracked_rois, is_trackable, tracked_count, Hg_0to6_std = track_ad_rois(roi_tracker, 
                                                                            img, 
                                                                            clips_paths[clip_idx], 
                                                                            plane_mask_unoccl[clip_idx], 
                                                                            w_ad_rois,
                                                                            plane_idx=plane_idx)
                        if is_trackable:
                            # get information for this clip, 
                            clip_name = cluster[clip_idx]
                            assert clip_name in clips_info
                            clip_info = clips_info[clip_name]
                            apo_info_dict = {"filename": cluster[clip_idx],
                                            "start_frame": clip_info["start_frame"],
                                            "duration_frames": clip_info["duration_frames"],
                                            "roi": tracked_rois.tolist(),
                                            "tracked_count": tracked_count,
                                            "Hg_0to6_std": Hg_0to6_std.tolist()}
                            plane_apo_list.append(apo_info_dict)
                            APO_for_this_scene_found = True
                            print("{colorama.Fore.GREEN}\n+++++++++++++++APO FOUND+++++++++++++++++++++")
                            print("\n")
                            tracking_failed_for_all_clips = False
                            tracking_successul_for_clips.append(clip_idx)
                    if tracking_failed_for_all_clips:
                        plane["APO_present"] = False
                        dbg_str = f"Tracking failed for all clips"
                    else:
                        dbg_str = f"Tracking worked for {len(tracking_successul_for_clips)}/{len(clips_paths)}"
                        plane["APO_present"] = True


                    if plane_apo_list:
                        APO_info = {f"APO_plane_{plane_idx}": plane_apo_list}
                #print(scene_roi_info)


                ##################################################################
                # Add debug string to this plane
                ##################################################################
                plane["dbg_str"] = dbg_str

            debug_vis_all_planes(img, planes, kpts, img_filename, DBG_DIR)

            if APO_for_this_scene_found:
                json_record = json.dumps(APO_info, default=int, indent=1)
                if not first_line:
                    json_record = ",\n" + json_record
                else:
                    first_line = False
                op_apo_fh.write(json_record)
                op_apo_fh.flush()
    op_apo_fh.write("]")





def main_cleanup_apos():
    parser = argparse.ArgumentParser(description="Main application")
    parser.add_argument("--staging_dir", type=str, help="Path to stagind dir", default="staging_dir")
    args = parser.parse_args()

    if not os.path.exists(args.staging_dir):
        raise Exception(f"Error: Directory not present: {args.staging_dir}")

    apo_info_json_file = glob.glob(args.staging_dir+"/*_apo_info.json")
    if not apo_info_json_file:
        error_str = f"Error: apo info file _apo_info.json not found\n"
        error_str += "You forgot running the scene_analyzer script?"
        raise Exception(error_str)

    apo_info_json_file = apo_info_json_file[0]
    with open(apo_info_json_file, "r") as f:
        apo_infos = json.load(f)

    def sort_apos_by_start_frame_number(apo_infos):
        sorted_apos = []
        for i in range(len(apo_infos)):
            info_dict = apo_infos[i]

            # Skip unselected APOs (if selection info is present)
            if "selected" in info_dict and not info_dict["selected"]:
                continue

            # Get per-APO ad path if present
            apo_ad_path = info_dict.get("ad_path", None)

            # we might have a list of APO belonging to multiple planes.
            # Take the APO for the first plane key
            plane_key = None
            plane_data = None
            for key, val in info_dict.items():
                if key.startswith("APO_plane_"):
                    plane_key = key
                    plane_data = val
                    break
            if plane_data is None:
                continue

            for clip_api_info in plane_data:
                clip_api_info["apo_id"] = i
                if apo_ad_path:
                    clip_api_info["ad_path"] = apo_ad_path
                sorted_apos.append(clip_api_info)

        sorted_apos = sorted(sorted_apos, key=lambda item: item["start_frame"])
        return sorted_apos

    #sorted_apo_infos = sort_apos_by_start_frame_number(apo_infos)

    img_H = 720 
    img_W = 1280
    apo_infos_f = []
    for apo_info in apo_infos:
        for plane_name, plane_apo_info in apo_info.items():
            plane_apo_info_f = []
            for clip_apo_info in plane_apo_info:
                #apo_id = clip_apo_info["apo_id"]
                start_frame = clip_apo_info["start_frame"]
                duration_frames = clip_apo_info["duration_frames"]
                tracked_count = clip_apo_info["tracked_count"]
                clip_rois = np.array(clip_apo_info["roi"])
                ad_path = clip_apo_info.get("ad_path", None)
                #print(f"-- {plane_name} id:{apo_id} duration:{duration_frames} tracked_count:{tracked_count} rois:{clip_rois.shape}")
                res = check_if_rois_are_sane(clip_rois[:,:4,:], [img_W, img_H])
                if res:
                    plane_apo_info_f.append(clip_apo_info)
            if plane_apo_info_f:
                apo_info_f = {plane_name: plane_apo_info_f}
                apo_infos_f.append(apo_info_f)



    with open(apo_info_json_file, "w") as f:
        apo_infos = json.dump(apo_infos_f, f)


def main_debug_bb_warps():
    parser = argparse.ArgumentParser(description="Main application")
    parser.add_argument("--staging_dir", type=str, help="Path to staging dir", default="staging_dir")
    parser.add_argument("--clip_frame", type=str, required=True, help="Path to video clip")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_folder = os.path.join(args.staging_dir, "clip_frames")
    sn_img_folder = os.path.join(args.staging_dir, "sn")
    _, filename = os.path.split(args.clip_frame)

    sn_file = os.path.join(sn_img_folder, f"{filename}")

    img = cv2.imread(args.clip_frame)
    sn_img = cv2.imread(sn_file)

    kp_extractor, sam_mask_generator, lcnn_model, yolo_model, fe_model, da_model = (
        load_models(True, True, True, True, True, True)
    )

    # UniDepth inference
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
    with torch.no_grad():
        depth_preds = da_model.infer(rgb_tensor)
    depth_img = depth_preds["depth"].squeeze().cpu().numpy()
    pts_3d_img = depth_preds["points"].squeeze().permute(1, 2, 0).cpu().numpy()
    intrinsics = depth_preds["intrinsics"]
    print(img.shape, sn_img.shape, depth_img.shape, pts_3d_img.shape,intrinsics)

    ######################################################################
    # 2. detecting keypoints
    ######################################################################
    img_t = torch.tensor((img / 255.0).transpose(2, 0, 1), dtype=torch.float32).to(device)
    feats = kp_extractor(data={"image": img_t})
    kpts = feats["keypoints"].detach().cpu().numpy().astype(int)[0]

    ######################################################################
    # We will do plane processing only on the reference image
    ######################################################################
    vp_detector = None
    planes, img_segs, _ = process_clip(img, depth_img, pts_3d_img,
                                      sn_img, device, vp_detector,
                                      kp_extractor, sam_mask_generator, 
                                      lcnn_model, yolo_model,
                                      fe_model, filename, out_dir=DBG_DIR)

    ######################################################################
    # 1. check empty region using a resnet based feature extractor
    ######################################################################
    empty_region_mask, people_mask = get_empty_region(img, yolo_model, fe_model, device)
    all_lines = get_all_lines(img, depth_img, sn_img, kp_extractor, lcnn_model, device)

    for plane_idx, plane in enumerate(planes):
        plane_mask = plane["segmentation"].astype(bool)
        plane_lines = get_lines_belonging_to_this_plane(plane_mask, all_lines)
        plane["plane_lines"] = plane_lines


    update_X_and_Y_axes(img, depth_preds, pts_3d_img, planes, all_lines)

    for plane_idx, plane in enumerate(planes):
        process_next_steps = True
        plane_mask = plane["segmentation"].astype(bool)
        if plane_mask.sum() < MIN_EMPTY_AREA:
            process_next_steps = False

        plane_empty_mask = np.logical_and(empty_region_mask.astype(bool), plane_mask.astype(bool))
        plane["empty_region"] = plane_empty_mask
        plane["dbg_str"] = ""
        if plane_empty_mask.sum() < MIN_EMPTY_AREA:
            process_next_steps = False

        if process_next_steps:
            ad_rois_list = get_rois(plane_empty_mask, min_wh=[100, 75], itrs=100)
            if len(ad_rois_list) == 0:
                process_next_steps = False

        if process_next_steps:
            ad_rois_list = sorted(ad_rois_list,
                            key=(lambda roi: (roi[2][0] - roi[0][0]) * (roi[2][1] - roi[0][1])),
                            reverse=True)
            plane["ad_rois"] = ad_rois_list
            ad_rois = np.array(ad_rois_list).reshape(-1,2)
            warped_rois, res_str = warp_bbox_using_depth_v1(img, depth_preds, pts_3d_img, plane)
            warped_rois = np.array(warped_rois).reshape(-1,2)
            warped_roi0 = warped_rois[:4,:].astype(np.int32)
            img1 = np.copy(img)
            img2 = np.copy(img)
            img1 = cv2.fillPoly(img1, [ad_rois[:4]], (0,0,255))
            plane_lines = plane["plane_lines"]
            img2 = draw_lines(img2, all_lines, color=(255, 0, 0), th=2)
            if len(warped_rois):
                img2 = cv2.fillPoly(img2, [warped_roi0], (0,0,255))
            fig, ax = plt.subplots(nrows=2)
            ax[0].imshow(img1[:,:,::-1])
            ax[1].imshow(img2[:,:,::-1])
            filename, file_xtn = os.path.splitext(filename)
            save_file = os.path.join(DBG_DIR, f"{filename}_wroi{plane_idx}{file_xtn}")
            plt.savefig(save_file)
            print("--- done for plane")

    print("all done")


def process_clips():
    main()
    main_cleanup_apos()

if __name__ == "__main__":
    process_clips()
    #main_debug_bb_warps()
