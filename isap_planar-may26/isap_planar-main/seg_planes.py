import os
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from collections import deque
import sys
from torchvision import transforms
import matplotlib.pyplot as plt
import glob
from ultralytics import YOLO
from sklearn.cluster import DBSCAN, KMeans
from scipy import stats, ndimage
from skimage.segmentation import slic, mark_boundaries
import re
from tqdm import tqdm

from model_loader import load_models
from image_util import show_sam_anns, draw_matches, draw_keypoints


np.random.seed(10)

#sys.path.append("models/LightGlueGyrus/")
# sys.path.append("./tracker")
#sys.path.append("models")
#sys.path.append("models/third_party/TinySAM")

#tinysam_wt_path = "assets/checkpoints/tinysam.pth"

from tinysam import sam_model_registry, SamPredictor, SamHierarchicalMaskGenerator
from robustpoint.robustpoint_top import RobustPointTop
from lcnn.infer_lcnn import infer_single_image as lcnn_infer


DBG_DIR = "debug/seg_depth"
SN_PLANE_MAX_VAR = 0.05


def compute_sn_variance(mask, sn_img):
    # seg_mask = mask_dict['segmentation'].astype(bool)
    sn_norm = sn_img.astype(np.float32) / 255.0
    sn_norm = sn_norm * 2.0 - 1.0  # 0 to 1 to -1 to 1
    pixels = sn_norm[mask]
    variance_per_channel = np.var(pixels, axis=0)
    return float(np.mean(variance_per_channel))


def get_intersection(line1, line2):
    """
    Finds the intersection of two lines given in Hesse normal form (rho, theta) or
    by two points [[x1, y1], [x2, y2]]. The code below assumes two points.

    Returns closest integer pixel locations or None if lines are parallel.
    """
    # Assuming line1 and line2 are in the format [[x1, y1], [x2, y2]]
    s1 = np.array(line1[0])
    e1 = np.array(line1[1])
    s2 = np.array(line2[0])
    e2 = np.array(line2[1])

    A1 = e1[1] - s1[1]
    B1 = s1[0] - e1[0]
    C1 = A1 * s1[0] + B1 * s1[1]

    A2 = e2[1] - s2[1]
    B2 = s2[0] - e2[0]
    C2 = A2 * s2[0] + B2 * s2[1]

    determinant = A1 * B2 - A2 * B1

    if determinant == 0:
        # Lines are parallel
        return None
    else:
        x = (C1 * B2 - C2 * B1) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        # Return as integer coordinates for drawing with cv2
        return int(round(x)), int(round(y))


def sort_points_clockwise(pts):
    """
    Sorts a list of 2D points in clockwise order around their centroid.

    Args:
        pts (np.ndarray): A NumPy array of shape (N, 2) representing the points.

    Returns:
        np.ndarray: The pts sorted in clockwise order.
    """
    # 1. Calculate the centroid
    # np.mean(pts, axis=0) computes the mean for x and y coordinates separately
    centroid = np.mean(pts, axis=0)
    cx, cy = centroid

    # 2. Calculate angles
    # np.arctan2(y - cy, x - cx) gives the angle in radians (counter-clockwise)
    # The x and y coordinates need to be handled correctly with the centroid
    x, y = pts.T
    angles = np.arctan2(y - cy, x - cx)

    # 3. Sort by angle in descending order for clockwise direction
    # np.argsort returns the indices that would sort the array
    # Negating the angles array sorts in descending order (clockwise)
    indices = np.argsort(-angles)

    # Return the points using the sorted indices
    return pts[indices]


def draw_random_roi(img, roi, roi_h=250, roi_w=250, color=(200, 0, 200)):
    print_str = ""
    plane_mask = roi["mask"].astype(np.uint8)
    # plane_mask = roi["mask"]
    if 0:
        contours, _ = cv2.findContours(
            plane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, contours, 0, color, 5)
        M = cv2.moments(contours[0])
    else:
        M = cv2.moments(plane_mask)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    img = cv2.circle(img, (cx, cy), 10, color, -1)

    roi_cords = np.array([[0, 0], [roi_w, 0], [roi_w, roi_h], [0, roi_h]])

    vp_data = roi["vp_data"]  # {"vps": vps_2d, "valid": vps_valid}
    vps_valid = vp_data["valid"]
    # Shift the orgin, this makes the rest of the computations easy
    vps_2d = vp_data["vps"] - np.array([cx, cy]) + np.array([roi_w // 2, roi_h // 2])
    print_str += f"valid vps:{vps_valid} vps:{vps_2d}"

    if vps_valid[2]:
        if vps_valid[0]:  #
            # vp0 is to the Right of us and vp2 above or below
            line0 = [roi_cords[0], vps_2d[0]]
            line1 = [roi_cords[3], vps_2d[0]]
        if vps_valid[1]:  #
            # vp1 is to the left of us and vp2 above or below
            line0 = [roi_cords[1], vps_2d[1]]
            line1 = [roi_cords[2], vps_2d[1]]
        if vps_2d[2][1] < roi_h // 2:  # vp is above roi
            line2 = [roi_cords[3], vps_2d[2]]
            line3 = [roi_cords[2], vps_2d[2]]
            print_str += ", vp2 is above roi "
        else:  # vp is below us
            line2 = [roi_cords[0], vps_2d[2]]
            line3 = [roi_cords[1], vps_2d[2]]
            print_str += ", vp2 is below roi "
    else:
        # vp0 is to the right of us
        line0 = [roi_cords[0], vps_2d[0]]
        line1 = [roi_cords[3], vps_2d[0]]

        # vp1 is to the left of
        line2 = [roi_cords[1], vps_2d[1]]
        line3 = [roi_cords[2], vps_2d[1]]

    pt0 = get_intersection(line0, line2)
    pt1 = get_intersection(line0, line3)
    pt2 = get_intersection(line1, line3)
    pt3 = get_intersection(line1, line2)

    roi_pts = (
        np.array([pt0, pt1, pt2, pt3])
        + np.array([cx, cy])
        - np.array([roi_w // 2, roi_h // 2])
    )
    roi_pts = sort_points_clockwise(roi_pts)
    cv2.drawContours(img, [roi_pts.reshape(-1, 1, 2)], 0, color, 5)
    print(print_str)
    return img



def draw_bb_roi(img, bb, vp_data, color=(200, 0, 200)):
    print_str = ""
    x0, y0, x1, y1 = bb
    roi_h = y1 - y0
    roi_w = x1 - x0
    cx = int((x0 + x1) / 2)
    cy = int((y0 + y1) / 2)
    img = cv2.circle(img, (cx, cy), 10, color, -1)
    roi_cords = np.array([[0, 0], [roi_w, 0], [roi_w, roi_h], [0, roi_h]])

    vps_valid = vp_data["valid"]
    # Shift the orgin, this makes the rest of the computations easy
    vps_2d = vp_data["vps"] - np.array([cx, cy]) + np.array([roi_w // 2, roi_h // 2])
    print_str += f"valid vps:{vps_valid} vps:{vps_2d}"

    if vps_valid[2]:
        if vps_valid[0]:  #
            # vp0 is to the Right of us and vp2 above or below
            line0 = [roi_cords[0], vps_2d[0]]
            line1 = [roi_cords[3], vps_2d[0]]
        if vps_valid[1]:  #
            # vp1 is to the left of us and vp2 above or below
            line0 = [roi_cords[1], vps_2d[1]]
            line1 = [roi_cords[2], vps_2d[1]]
        if vps_2d[2][1] < roi_h // 2:  # vp is above roi
            line2 = [roi_cords[3], vps_2d[2]]
            line3 = [roi_cords[2], vps_2d[2]]
            print_str += ", vp2 is above roi "
        else:  # vp is below us
            line2 = [roi_cords[0], vps_2d[2]]
            line3 = [roi_cords[1], vps_2d[2]]
            print_str += ", vp2 is below roi "
    else:
        # vp0 is to the right of us
        line0 = [roi_cords[0], vps_2d[0]]
        line1 = [roi_cords[3], vps_2d[0]]

        # vp1 is to the left of
        line2 = [roi_cords[1], vps_2d[1]]
        line3 = [roi_cords[2], vps_2d[1]]

    pt0 = get_intersection(line0, line2)
    pt1 = get_intersection(line0, line3)
    pt2 = get_intersection(line1, line3)
    pt3 = get_intersection(line1, line2)

    roi_pts = (
        np.array([pt0, pt1, pt2, pt3])
        + np.array([cx, cy])
        - np.array([roi_w // 2, roi_h // 2])
    )
    roi_pts = sort_points_clockwise(roi_pts)
    cv2.drawContours(img, [roi_pts.reshape(-1, 1, 2)], 0, color, 3)
    print(print_str)
    return img


def show_rois(rois, ax):
    img_ = np.ones((rois[0]["mask"].shape[0], rois[0]["mask"].shape[1], 4))
    img_[:, :, 3] = 0
    for ann in rois:
        m = ann["mask"]
        color_mask = np.concatenate([np.random.random(3), [0.65]])
        img_[m] = color_mask
        lines = ann["lines"]
        kpts = ann["kpts"]
        img_ = draw_lines(img_, lines)
        img_ = draw_keypoints(img_, kpts, radius=3, color=(1, 1, 0))
    ax.imshow(img_)


def detect_humans(yolo_model, img_):
    results = yolo_model.predict(
        source=img_, classes=[0]
    )  # Class 0 is 'person' in the COCO dataset
    people_mask = np.zeros(img_.shape[:2], dtype=np.uint8)
    people_box_mask = np.zeros(img_.shape[:2], dtype=np.uint8)
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

        if result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates in xyxy format (top-left x, y, bottom-right x, y)
                xtl, ytl, xbr, ybr = box.xyxy[0].int().tolist()
                cv2.rectangle(people_box_mask, (xtl, ytl), (xbr, ybr), 1, -1)

    return people_mask, people_box_mask


def draw_lines(img, lines, color=1, th=1):
    for i in range(len(lines)):
        # LCNN outputs coordinates in (y, x) format, so we need to reverse them
        # pt1 = lines[i][0][::-1]  # [y1, x1] -> [x1, y1]
        # pt2 = lines[i][1][::-1]  # [y2, x2] -> [x2, y2]
        pt1, pt2 = lines[i]

        x1, y1 = pt1
        x2, y2 = pt2

        # Convert to integer coordinates
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw on output image
        # cv2.line(output_image, (x1, y1), (x2, y2), 1, 1, lineType=16)  # red
        cv2.line(img, (x1, y1), (x2, y2), color, th)
    return img


def filter_overlapping_segs(img, seg_masks):
    n_segs = len(seg_masks)

    del_idx = []
    for i in range(n_segs - 1):
        seg_i = seg_masks[i]
        mask_i = seg_i["segmentation"]
        for j in range(i + 1, n_segs, 1):
            seg_j = seg_masks[j]
            mask_j = seg_j["segmentation"]
            area_j = seg_j["area"]
            ints_mask = np.logical_and(mask_i, mask_j)
            if (np.sum(ints_mask) + 1000) >= area_j:
                del_idx.append(j)
    seg_masks_f = []
    for i in range(n_segs):
        if i not in del_idx:
            seg_masks_f.append(seg_masks[i])
    return seg_masks_f

def remove_overlapping_regions(img, seg_masks):
    n_segs = len(seg_masks)

    for i in range(n_segs - 1):
        seg_i = seg_masks[i]
        mask_i = seg_i["segmentation"]
        area_i = seg_i["area"]
        for j in range(i + 1, n_segs, 1):
            seg_j = seg_masks[j]
            mask_j = seg_j["segmentation"]
            area_j = seg_j["area"]
            ints_mask = np.logical_and(mask_i, mask_j)
            # We will reduce the bigger region
            if np.sum(ints_mask) >= 1000:
                mask_i = np.logical_and(mask_i, (1-ints_mask).astype(bool))
                seg_i["segmentation"] = mask_i
                area_i = area_i - np.sum(ints_mask)
                seg_i["area"] = area_i




def cleanup_segs_using_cca(img, seg_masks):

    for seg in seg_masks:
        mask = seg["segmentation"].astype(np.uint8)
        # Get connected components and their stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Set your minimum area threshold (e.g., 100 pixels)
        min_area = 4000
        output_mask = np.zeros_like(mask)

        # Loop through each component (skipping the background label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                # Keep this component by adding it to the mask
                output_mask[labels == i] = 1
        seg["segmentation"] = output_mask.astype(bool)
    return


def save_pointcloud(points, image, output_path):
    """
    Save 3D points as a PLY file with colors from the image.

    Args:
        points: (H, W, 3) array of 3D points.
        image: (H, W, 3) BGR image for color.
        output_path: path to save the PLY file.
    """
    H, W = points.shape[:2]
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {H * W}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(H):
            for j in range(W):
                x, y, z = points[i, j]
                b, g, r = image[i, j]
                f.write(f"{x} {y} {z} {r} {g} {b}\n")

def populate_surface_normal_for_segs(sam_segs, sn_img):

    for seg in sam_segs:
        mask = seg["segmentation"].astype(bool)
        plane_normals = sn_img[mask]

        plane_normals = plane_normals[:, ::-1]
        plane_normals = (plane_normals.astype(float) - 127.5) / 127.5

        plane_normals = np.round(plane_normals, 2)

        unique, counts = np.unique(plane_normals, axis=0, return_counts=True)
        surface_normal = unique[np.argmax(counts)]

        # unit normal
        norm = np.linalg.norm(surface_normal)
        if norm > 0:
            surface_normal = surface_normal / norm

        seg["surface_normal"] = surface_normal

    return sam_segs


def seg_depth(
    kp_extractor,
    sam_mask_generator,
    sam_model_bg,
    yolo_model,
    img,
    depth_img,
    pts_3d,
    sn_img,
    device,
    min_area=20000,
    filename=None,
    dbg_dir=None,
):

    PLANE_MIN_AREA = 20000
    # detect if image is blank
    if 1:
        min_ = np.min(img)
        max_ = np.max(img)
        if (max_ - min_) < 0.1:
            print("Image is blank")
            return [], [], []

    imgH, imgW = depth_img.shape[:2]
    people_mask, people_box_mask = detect_humans(yolo_model, img)

    sn_img = cv2.GaussianBlur(sn_img, (3, 3), 0).astype(np.uint8)

    #########################################################################
    # 1. Surface normal processing
    #########################################################################
    # Image has BGR since it is opened using opencv
    # Blue channel for surface normal coming out of image plane
    # Green channel for surface normal UP
    # Red channel for surface normal towards Left
    _, front_walls_mask = cv2.threshold(
        sn_img[:, :, 0], 220, maxval=255, type=cv2.THRESH_BINARY
    )
    _, up_surface_mask = cv2.threshold(
        sn_img[:, :, 1], 220, maxval=255, type=cv2.THRESH_BINARY
    )
    _, down_surface_mask = cv2.threshold(
        255 - sn_img[:, :, 1], 220, maxval=255, type=cv2.THRESH_BINARY
    )
    _, left_surface_mask = cv2.threshold(
        sn_img[:, :, 2], 220, maxval=255, type=cv2.THRESH_BINARY
    )
    _, right_surface_mask = cv2.threshold(
        255 - sn_img[:, :, 2], 220, maxval=255, type=cv2.THRESH_BINARY
    )

    floor_ceil_mask = np.logical_or(up_surface_mask, down_surface_mask)

    if False:
        # overlay floor ceil mask on the image for debugging
        overlay = img.copy()
        overlay[floor_ceil_mask] = (0, 255, 0)  # Green
        alpha = 0.5  # Transparency factor
        img_overlay = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        cv2.imshow("Floor and Ceiling Mask Overlay", img_overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    dg_x = cv2.Sobel(depth_img, cv2.CV_64F, 1, 0, ksize=3)
    dg_y = cv2.Sobel(depth_img, cv2.CV_64F, 0, 1, ksize=3)
    dg_xy = cv2.magnitude(dg_x, dg_y)
    knl = np.ones((3, 3), np.uint8)
    dg_xy = cv2.dilate(dg_xy, knl, iterations=1)
    _, depth_img_gxy = cv2.threshold(dg_xy, 5, maxval=127, type=cv2.THRESH_BINARY)

    g_x = cv2.Sobel(sn_img[:, :, 0], cv2.CV_64F, 1, 0, ksize=3)
    g_y = cv2.Sobel(sn_img[:, :, 0], cv2.CV_64F, 0, 1, ksize=3)
    g_xy = cv2.magnitude(g_x, g_y)
    knl = np.ones((3, 3), np.uint8)
    g_xy = cv2.dilate(g_xy, knl, iterations=1)
    _, sn_img_gxy = cv2.threshold(g_xy, 15, maxval=127, type=cv2.THRESH_BINARY)

    grad_img = (depth_img_gxy + sn_img_gxy)[:, :, np.newaxis].astype(np.uint8)


    img_t = torch.tensor((img / 255.0).transpose(2, 0, 1), dtype=torch.float32).to(
        device
    )
    feats = kp_extractor(data={"image": img_t})
    kpts = feats["keypoints"].detach().cpu().numpy().astype(int)[0]

    depth_img_ = np.repeat(depth_img[:,:,np.newaxis], 3, axis=2)
    
    #--------------------------------------------------------------------------
    ###########################################################################
    # 0. Depth based segmentation using TinySAM
    ###########################################################################
    try:
        depth_segs = sam_mask_generator.hierarchical_generate(depth_img_)
        depth_segs = sorted(depth_segs, key=(lambda x: x["area"]), reverse=True)
    except:
        return [], [], []


    #--------------------------------------------------------------------------
    ###########################################################################
    # 1. some basic cleanup of the segments
    ###########################################################################
    remove_overlapping_regions(img, depth_segs)
    depth_segs = sorted(depth_segs, key=(lambda x: x["area"]), reverse=True)

    if dbg_dir:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        fig.suptitle('Depth segs::Initial segments')
        ax[0, 0].set_title("Image")
        ax[0, 0].imshow(img[:,:,::-1])
        ax[0, 0].set_title("Depth Image")
        ax[0, 1].imshow(depth_img, cmap='grey', vmin=0, vmax=np.max(depth_img))
        ax[1, 0].set_title("SN Image")
        ax[1, 0].imshow(sn_img)
        ax[1, 1].set_title(f"depth segs {len(depth_segs)}")
        ax[1, 1].imshow(img[:,:,::-1])
        show_sam_anns(depth_segs, ax[1, 1], show_idx=True)
        for axis in ax.ravel():
            axis.axis("off")
        plt.tight_layout()
        dbg_filename = re.sub(r".png", "_depth_segs_s1.jpg", filename)
        save_file = os.path.join(dbg_dir, f"{dbg_filename}")
        plt.savefig(save_file)
        #plt.show()
        plt.close()



    #--------------------------------------------------------------------------
    ###########################################################################
    # 2. Remove people regions, floor and ceiling regions
    ###########################################################################
    #### CAUTION::: THIS PLOT CODE CONTINUES AFTER THE NEXT CODE BLOCK
    if dbg_dir:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        fig.suptitle('Depth segs::Removing people, floor, ceiling')
        ax[0, 0].set_title("Image")
        ax[0, 0].imshow(img[:,:,::-1])
        ax[0, 1].set_title(f"depth segs Before.. {len(depth_segs)}")
        ax[0, 1].imshow(img[:,:,::-1])
        show_sam_anns(depth_segs, ax[0, 1], show_idx=True)


    people_mask_dilated = cv2.dilate(people_mask.astype(np.uint8), knl, iterations=2).astype(bool)
    for i, seg in enumerate(depth_segs):
        seg_mask = seg["segmentation"].astype(bool)
        # If this mask was representing people, we need to just remove it
        seg_mask = np.logical_and(seg_mask, 1 - people_mask_dilated)

        # remove floor and ceiling regions based on surface normal
        seg_mask = np.logical_and(seg_mask, np.logical_not(floor_ceil_mask))
        seg["segmentation"] = seg_mask
        seg["area"] = np.sum(seg_mask)

    # sort the masks again
    depth_segs = sorted(depth_segs, key=(lambda x: x["area"]), reverse=True)

    #### CAUTION::: THIS PLOT CODE CONTINUED FROM THE PREVIOUS CODE BLOCK
    if dbg_dir:
        ax[1, 0].set_title("Floor ceil mask")
        ax[1, 0].imshow(floor_ceil_mask)
        ax[1, 1].set_title(f"depth segs After.. {len(depth_segs)}")
        ax[1, 1].imshow(img[:,:,::-1])
        show_sam_anns(depth_segs, ax[1, 1], show_idx=True)
        for axis in ax.ravel():
            axis.axis("off")
        plt.tight_layout()
        dbg_filename = re.sub(r".png", "_depth_segs_s2.jpg", filename)
        save_file = os.path.join(dbg_dir, f"{dbg_filename}")
        plt.savefig(save_file)
        plt.close()

    depth_segs_f = []
    for i, seg in enumerate(depth_segs):
        seg_mask = seg["segmentation"].astype(bool)
        area = np.sum(seg_mask)
        # we want to be a bit relaxed on the min area. After merge in the next
        # step the plane area might increase
        if np.sum(seg_mask) > PLANE_MIN_AREA//4:
            depth_segs_f.append(seg)

    depth_segs = depth_segs_f

    #--------------------------------------------------------------------------
    ###########################################################################
    # 3 Merge segments based on point cloud
    ###########################################################################
    #### CAUTION::: THIS PLOT CODE CONTINUES AFTER THE NEXT CODE BLOCK
    if dbg_dir:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        fig.suptitle('Depth segs::Merging planes based on point cloud')
        ax[0, 0].set_title("Image")
        ax[0, 0].imshow(img[:,:,::-1])
        ax[0, 1].set_title(f"depth segs Before.. {len(depth_segs)}")
        ax[0, 1].imshow(img[:,:,::-1])
        show_sam_anns(depth_segs, ax[0, 1], show_idx=True)

    depth_segs = populate_surface_normal_for_segs(depth_segs, sn_img)
    pts_3d_img = pts_3d.reshape(imgH, imgW, 3)
    depth_segs = update_plane_coefficients_ransac(depth_segs, pts_3d_img)
    #depth_segs = plane_mask_from_coefficients(depth_segs, pts_3d_img, sn_img)
    depth_segs =  merge_planes_using_plane_coefficients(depth_segs, False)

    #### CAUTION::: THIS PLOT CODE CONTINUED FROM THE PREVIOUS CODE BLOCK
    if dbg_dir:
        ax[1, 0].set_title("Depth Image")
        ax[1, 0].imshow(depth_img, cmap='grey', vmin=0, vmax=np.max(depth_img))
        ax[1, 1].set_title(f"depth segs After.. {len(depth_segs)}")
        ax[1, 1].imshow(img[:,:,::-1])
        show_sam_anns(depth_segs, ax[1, 1], show_idx=True)
        for axis in ax.ravel():
            axis.axis("off")
        plt.tight_layout()
        dbg_filename = re.sub(r".png", "_depth_segs_s3.jpg", filename)
        save_file = os.path.join(dbg_dir, f"{dbg_filename}")
        plt.savefig(save_file)
        plt.close()



    #--------------------------------------------------------------------------
    ###########################################################################
    # 4.remove segments which are not plane
    # We will use Surface Normal, curved surface will have too much variance
    # in SN
    ###########################################################################
    #### CAUTION::: THIS PLOT CODE CONTINUES AFTER THE NEXT CODE BLOCK
    if dbg_dir:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        fig.suptitle('Depth segs::Removing Non planar regions')
        ax[0, 0].set_title("Image")
        ax[0, 0].imshow(img[:,:,::-1])
        ax[0, 1].set_title(f"depth segs Before.. {len(depth_segs)}")
        ax[0, 1].imshow(img[:,:,::-1])
        show_sam_anns(depth_segs, ax[0, 1], show_idx=True)
    depth_segs_f = []
    # we will erode the masks slightly so that we don't take the edges. 
    # sn will vay near the edges and we don't want to consider that
    for i, seg in enumerate(depth_segs):
        seg_mask = seg["segmentation"].astype(bool)
        seg_mask = cv2.erode(seg_mask.astype(np.uint8), knl, iterations=1).astype(bool)
        sn_variance = compute_sn_variance(seg_mask, sn_img)
        if sn_variance < SN_PLANE_MAX_VAR:
            depth_segs_f.append(seg)
        else:
            #print(f" Removing segment {i} with sn var:{sn_variance}")
            pass

    # some more cleanup
    cleanup_segs_using_cca(img, depth_segs_f)

    depth_segs = depth_segs_f
    # sort the masks again
    depth_segs = sorted(depth_segs, key=(lambda x: x["area"]), reverse=True)

    #### CAUTION::: THIS PLOT CODE CONTINUED FROM THE PREVIOUS CODE BLOCK
    if dbg_dir:
        ax[1, 0].set_title("sn img")
        ax[1, 0].imshow(sn_img)
        ax[1, 1].set_title(f"depth segs After.. {len(depth_segs)}")
        ax[1, 1].imshow(img[:,:,::-1])
        show_sam_anns(depth_segs, ax[1, 1], show_idx=True)
        for axis in ax.ravel():
            axis.axis("off")
        plt.tight_layout()
        dbg_filename = re.sub(r".png", "_depth_segs_s4.jpg", filename)
        save_file = os.path.join(dbg_dir, f"{dbg_filename}")
        plt.savefig(save_file)
        plt.close()


    
    #####################################################################

    if 0:
        pts_3d, colors = depths_to_world_points_with_colors(
            depth_results.depth,
            depth_results.intrinsics,
            depth_results.extrinsics,
            depth_results.processed_images,
            None,
            None,
        )
        _, img_H, img_W = depth_results.processed_images.shape[:3]
        pts_3d_img = pts_3d.reshape(img_H, img_W, 3)
    else:
        pts_3d_img = None

    # save point cloud for debugging
    if False:
        pointcloud_filename = re.sub(r'.png', f"_pointcloud.ply", filename)
        pc_save_path = os.path.join(dbg_dir, pointcloud_filename)
        save_pointcloud(pts_3d_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), pc_save_path)

    #--------------------------------------------------------------------------
    ###########################################################################
    # 5. remove segments which < some minimum area
    ###########################################################################
    depth_segs = depth_segs_f
    depth_segs_f = []
    for i, seg in enumerate(depth_segs):
        seg_mask = seg["segmentation"].astype(bool)
        area = np.sum(seg_mask)
        if np.sum(seg_mask) > PLANE_MIN_AREA:
            # also erode the segments a bit
            seg_mask = cv2.erode(seg_mask.astype(np.uint8), knl, iterations=2).astype(bool)
            seg["segmentation"] = seg_mask
            seg["area"] = np.sum(seg_mask)
            depth_segs_f.append(seg)


    if dbg_dir:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        fig.suptitle('Depth segs::Cleanup, remove segs < area')
        ax[0, 0].set_title("Image")
        ax[0, 0].imshow(img[:,:,::-1])
        ax[0, 1].set_title("SN Image")
        ax[0, 1].imshow(sn_img)
        ax[1, 0].set_title(f"depth img")
        ax[1, 0].imshow(depth_img, cmap='grey', vmin=0, vmax=np.max(depth_img))
        ax[1, 1].set_title(f"depth segs {len(depth_segs_f)}")
        ax[1, 1].imshow(img[:,:,::-1])
        show_sam_anns(depth_segs_f, ax[1, 1], show_idx=True)
        for axis in ax.ravel():
            axis.axis("off")
        plt.tight_layout()
        dbg_filename = re.sub(r".png", "_depth_segs_s5.jpg", filename)
        save_file = os.path.join(dbg_dir, f"{dbg_filename}")
        plt.savefig(save_file)
        plt.close()


    depth_segs = depth_segs_f
    depth_segs = sorted(depth_segs, key=(lambda x: x["area"]), reverse=True)



    return depth_segs, None, pts_3d_img


#def clean_plane_segs(plane_segs):
#
#    knl = np.ones((7,7), np.uint8)
#    for seg in plane_segs:
#        seg_mask = seg['segmentation'].astype(np.uint8)
#        seg_mask = cv2.dilate(seg_mask, knl, iterations=2)
#        seg_mask = cv2.erode(seg_mask, knl, iterations=2)
#        #_, contours, hierarchy = cv.findContours(seg_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#        #contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        contours, _ = cv2.findContours(seg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#        biggest_contour = max(contours, key=cv2.contourArea) #
#        seg['segmentation'][:,:] = 0
#        #seg_mask = cv2.drawContours(seg['segmentation'].astype(np.uint8), [biggest_contour], -1, 1, 2)
#        seg_mask = cv2.fillConvexPoly(seg['segmentation'].astype(np.uint8), biggest_contour, 1)
#        seg['segmentation'] = seg_mask.astype(bool)
#
#    return plane_segs

def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


def depths_to_world_points_with_colors(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    images_u8: np.ndarray,
    conf: np.ndarray | None,
    conf_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each frame, transform (u,v,1) through K^{-1} to get rays,
    multiply by depth to camera frame, then use (w2c)^{-1} to transform to world frame.
    Simultaneously extract colors.
    """
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

    pts_all, col_all = [], []

    for i in range(N):
        d = depth[i]  # (H,W)
        valid = np.isfinite(d) & (d > 0)
        if conf is not None:
            valid &= conf[i] >= conf_thr
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K[i])  # (3,3)
        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c[i]))  # (4,4)

        rays = K_inv @ pix[vidx].T  # (3,M)
        Xc = rays * d_flat[vidx][None, :]  # (3,M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)

        ### knr
        invalid = (1 - valid).reshape(-1)
        Xw[invalid] = 0

        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M,3)

        pts_all.append(Xw)
        col_all.append(cols)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)


def find_plane_coefficients(p1, p2, p3):
    """
    Calculates the coefficients (a, b, c, d) for the plane equation ax + by + cz + d = 0
    given three non-collinear points p1, p2, and p3.
    """
    # Calculate two vectors that lie in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # The cross product of the two vectors is a vector normal to the plane
    normal = np.cross(v1, v2)
    unit_normal = normal / np.linalg.norm(normal)

    # Use one of the points (p1) and the normal vector to find d
    # The equation can be written as normal . (point - p1) = 0, which simplifies to
    # normal . point = normal . p1
    # So, d = - (normal . p1)
    d = -np.dot(unit_normal, p1)

    return unit_normal, d


def check_non_collinear(p1, p2, p3):

    if np.allclose(p1, 0) and np.allclose(p2, 0) and np.allclose(p3, 0):
        print("Depth not proper")
        return False
    v1 = p3 - p1
    v2 = p2 - p1

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("got zero length vector")
        return False

    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)

    angle = np.degrees(np.arccos(np.dot(unit_v1, unit_v2)))

    if np.isclose(angle, 0, atol=5) or np.isclose(angle, 180, atol=5):
        return False
    else:
        return True


def fit_plane_ransac(points, plane_normal, threshold=0.03, iterations=1000):
    n_points = len(points)
    if n_points < 1 or plane_normal is None:
        return None, None

    normal = np.array(plane_normal)
    normal /= np.linalg.norm(normal)
    a, b, c = normal

    best_d = None
    max_inliers = -1
    plane_point = None

    if n_points > 10000:
        idxs = np.random.choice(n_points, 10000, replace=False)
        pts_c = points[idxs]
    else:
        pts_c = points

    for _ in range(iterations):
        idx = np.random.randint(len(pts_c))
        p = pts_c[idx]
        d_candidate = -np.dot(normal, p)
        dists = np.abs(np.dot(pts_c, normal) + d_candidate)
        inliers = dists < threshold
        n_in = np.sum(inliers)

        if n_in > max_inliers:
            max_inliers = n_in
            best_d = d_candidate
            plane_point = p

    if best_d is not None:
        return (a, b, c, best_d), plane_point

    return None, None

def update_plane_coefficients_ransac(planes_seg, pts_3d_img):
    if len(planes_seg) == 0:
        return planes_seg
    img_H, img_W = planes_seg[0]["segmentation"].shape[:2]
    xs, ys = np.meshgrid(np.arange(0, img_W, 16), np.arange(0, img_H, 16))
    grid_pts = np.stack([xs, ys], axis=-1).reshape(-1, 2)  # (h*w,2)

    for plane in planes_seg:
        mask = plane["segmentation"].astype(bool)
        pts_mask = mask[grid_pts[:, 1], grid_pts[:, 0]] > 0
        pts = grid_pts[pts_mask]
        pts_len = len(pts)
        if pts_len < 1:
            print("didn't get enough points for this plane, skipping")
            continue

        pts_3d = pts_3d_img[pts[:, 1], pts[:, 0]]
        
        surface_normal = plane.get("surface_normal")
        if surface_normal is None:
            continue

        best_eq, p1 = fit_plane_ransac(pts_3d, surface_normal)
        
        if best_eq is None:
            continue
            
        a, b, c, d = best_eq
        unormal = np.array([a, b, c])

        if np.isnan(unormal).any() or np.isnan(d):
            print("got nan values for plane coeffs, skipping")
            continue

        plane["plane_coeffs"] = [unormal, d, p1]
        #print(f"got plane coeffs {unormal}, {d}")

    return planes_seg

def filter_plane_mask(plane_mask, roi_mask):
    # filter keep componets which roi lies on, remove remaining components
    num_labels, labels = cv2.connectedComponents(plane_mask.astype(np.uint8))
    
    updated_mask = np.zeros_like(plane_mask, dtype=bool)
    
    for i in range(1, num_labels):
        comp_mask = (labels == i)
        if np.any(np.logical_and(comp_mask, roi_mask)):
            updated_mask = np.logical_or(updated_mask, comp_mask)
            
    return updated_mask

def plane_mask_from_coefficients(
    sam_segs,
    points,
    normal,
    offset_multiplier=4.0,
    angle_threshold=10.0,
):

    H, W = points.shape[:2]
    pts_flat = points.reshape(-1, 3)

    normal = normal.copy()
    normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
    # from 0-255 to -1 to 1
    normal = (normal.astype(float) / 127.5) - 1.0


    for info in sam_segs:
        if "plane_coeffs" not in info:
            continue

        unormal, d, p1 = info["plane_coeffs"]

        # --- distance mask: points within offset of the plane ---
        dists = np.abs(np.dot(pts_flat, unormal) + d)
        # divide by norm of normal to get actual distance
        dists /= np.linalg.norm(unormal)
        dist_map = dists.reshape(H, W)
        
        # calculate offset of plane pointcloud
        # taking from plane equation to 98 percentile distance as threshold
        seg_mask = info.get("segmentation").astype(bool)
        seg_dists = dist_map[seg_mask]
        offset_threshold = np.percentile(seg_dists, 98) if len(seg_dists) > 0 else 0.1

        offset_threshold *= offset_multiplier
            
        dist_mask = dist_map < offset_threshold

        # exclude invalid (zero-depth) points
        valid_pts = np.abs(points).sum(axis=2) > 1e-6

        dist_mask = np.logical_and(dist_mask, valid_pts)

        # --- normal mask: pixel normals within angle_threshold of plane normal ---
        if unormal is not None:
            normal_flat = normal.reshape(-1, 3)
            # normalize pixel normals
            norms = np.linalg.norm(normal_flat, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-8, None)
            normal_flat_unit = normal_flat / norms

            # normalize plane normal
            plane_normal_unit = unormal / np.linalg.norm(unormal)

            # angle between each pixel normal and the plane normal
            dots = np.clip(np.dot(normal_flat_unit, plane_normal_unit), -1.0, 1.0)
            angles = np.degrees(np.arccos(np.abs(dots)))  # abs handles flipped normals
            angle_map = angles.reshape(H, W)
            normal_mask = angle_map < angle_threshold
        else:
            normal_mask = np.ones((H, W), dtype=bool)

        # --- combine both masks ---
        complete_mask = np.logical_and(dist_mask, normal_mask)

        complete_mask = filter_plane_mask(complete_mask, seg_mask.astype(bool))

        complete_mask = np.logical_or(complete_mask, seg_mask.astype(bool))

        info["plane_mask"] = complete_mask.astype(bool)

    return sam_segs

def merge_planes_using_plane_coefficients(depth_segs, debug=False):
    merged_segs = []
    n_planes = len(depth_segs)
    
    if debug:
        for i in range(n_planes):
            if "plane_coeffs" in depth_segs[i]:
                unorm_i, d_i, p_i = depth_segs[i]["plane_coeffs"]
                print(i, unorm_i, d_i, p_i)

    merge_maps = []
    for i in range(n_planes - 1):
        if not "plane_coeffs" in depth_segs[i]:
            continue
        unorm_i, d_i, p_i = depth_segs[i]["plane_coeffs"]
        for j in range(i + 1, n_planes, 1):
            if not "plane_coeffs" in depth_segs[j]:
                continue
            unorm_j, d_j, p_j = depth_segs[j]["plane_coeffs"]
            # check if the coefficients have same ratios
            # Dot product of unit vectors should be close to 1 or -1 for parallel/anti-parallel
            dot_product = np.dot(unorm_i, unorm_j)
            if np.isclose(abs(dot_product), 1.0, atol=1e-1):
                if debug:
                    print("--- dot product is close")
                # check if a point on one plane lies on the othr plane
                # for a point xi,yi,zi lying on plane j, the plane equation is
                # ajx + bjy + cjz + dj = 0
                val = np.dot(unorm_j, p_i) + d_j
                if np.isclose(val, 0.0, atol=1e-1):
                    if debug:
                        print("--- planes lie on the same plane")
                    merge_maps.append([i, j])

    merge_maps_ar = np.array(merge_maps)
    # print(merge_maps_ar)
    # remove circular maps a->b b->c is same as a->b, a->c

    adj = {i: set() for i in range(n_planes)}
    for src, dst in merge_maps_ar:
        adj[src].add(dst)
        adj[dst].add(src)

    # Find connected components
    visited = [False] * n_planes
    final_instructions = [] 

    for i in range(n_planes):
        if not visited[i] and adj[i]:
            # Start a BFS to find everything connected to plane i
            component = []
            queue = deque([i])
            visited[i] = True
            
            while queue:
                curr = queue.popleft()
                component.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            src = component[0]
            for dst in component[1:]:
                final_instructions.append([src, dst])

    merge_maps_ar = np.array(final_instructions)

    merge_indices = np.unique(merge_maps_ar.reshape(-1))
    for i in range(len(depth_segs)):
        seg_i = depth_segs[i]["segmentation"].astype(bool)
        area_i = depth_segs[i]["area"]
        if i not in merge_indices:
            if debug:
                print(f"adding {i} without comb")
            merged_segs.append(depth_segs[i])
        else:
            add_seg = False
            for merge_map in merge_maps_ar:
                if merge_map[0] == i:
                    add_seg = True
                    j = merge_map[1]
                    if debug:
                        print(f"merging {i},{j}")
                    seg_j = depth_segs[j]["segmentation"].astype(bool)
                    area_j = depth_segs[j]["area"]
                    seg_i = np.logical_or(seg_i, seg_j)
                    depth_segs[i]["segmentation"] = seg_i
                    depth_segs[i]["area"] = area_i + area_j
            if add_seg:
                if debug:
                    print(f"adding {i} after combining")
                merged_segs.append(depth_segs[i])

    return merged_segs

def draw_planes_on_3d_pts(sam_segs, pts_3d_img, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    points = pts_3d_img.copy()

    valid_segs = [seg for seg in sam_segs if "plane_coeffs" in seg]
    num_segs = len(valid_segs)

    if num_segs > 0:
        colors = np.random.randint(0, 255, size=(num_segs, 3), dtype=np.uint8)

    for idx, seg in enumerate(valid_segs):
        unormal, d, p1 = seg["plane_coeffs"]
        mask = seg["segmentation"].astype(bool)
        color = colors[idx]
        
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            p = points[y, x]
            # get new z value
            a, b, c = unormal
            x3d, y3d, z3d = p
            z_new = (-d - a * x3d - b * y3d) / c
            points[y, x, 2] = z_new
            img[y, x] = color

    return points, img

def main_single_img():
    parser = argparse.ArgumentParser(description="Tracker test")
    parser.add_argument("--img", required=True, help="Path to image file")
    parser.add_argument("--sn", required=True, help="Path to surface normal folder")
    args = parser.parse_args()

    if not os.path.exists(args.img):
        raise Exception(f"Error: img file not found: {args.img}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    os.makedirs(f"{DBG_DIR}", exist_ok=True)

    kp_extractor, sam_mask_generator, lcnn_model, yolo_model, fe_model, da_model = (
        load_models(True, True, True, True, True, True)
    )

    img_bgr = cv2.imread(args.img)

    img_H, img_W = img_bgr.shape[:2]
    assert img_H == 720 and img_W == 1280, f"Video resolution not proper, expected (1290x480), got ({img_W},{img_H})"

    _, filename = os.path.split(args.img)

    sn_file = os.path.join(args.sn, filename)
    sn_img = cv2.imread(sn_file)

    # UniDepth inference
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device)
    with torch.no_grad():
        depth_predictions = da_model.infer(rgb_tensor)
    depth_img = depth_predictions["depth"].squeeze().cpu().numpy()
    pts_3d = depth_predictions["points"].squeeze().permute(1, 2, 0).cpu().numpy()

    depth_results = None

    plane_segs, _, _ = seg_depth(
        kp_extractor,
        sam_mask_generator,
        None,
        yolo_model,
        img_bgr,
        depth_img,
        pts_3d,
        sn_img,
        device,
        filename=filename,
        dbg_dir=DBG_DIR,
    )

    if True:
        fig, ax = plt.subplots(nrows=2)
        ax[0].imshow(img_rgb)
        ax[1].set_title(f"plane segments. {len(plane_segs)}")
        ax[1].imshow(img_rgb)
        show_sam_anns(plane_segs, ax[1], show_idx=True)
        plt.show()


def detect_all_lines(img, sn_img, depth_img, lcnn_model, device):
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

    all_lines = np.vstack(
        [im_lines.reshape(-1, 4), sn_lines.reshape(-1, 4), depth_lines.reshape(-1, 4)]
    )
    all_lines = all_lines.reshape(-1, 2, 2)
    return all_lines


def main_folder():
    import glob
    parser = argparse.ArgumentParser(description="Tracker test")
    parser.add_argument("--img_folder", required=True, help="folder containing images")
    parser.add_argument("--sn", required=True, help="Path to surface normal folder")
    args = parser.parse_args()

    kp_extractor, sam_mask_generator, lcnn_model, yolo_model, fe_model, da_model = (
        load_models(True, True, True, True, True, True)
    )

    img_files = sorted(glob.glob(args.img_folder+"/*.png"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(f"{DBG_DIR}", exist_ok=True)

    for img_file in img_files:
        _, filename = os.path.split(img_file)
        sn_file = os.path.join(args.sn, filename)

        img_bgr = cv2.imread(img_file)
        sn_img = cv2.imread(sn_file)

        # UniDepth inference
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device)
        with torch.no_grad():
            depth_predictions = da_model.infer(rgb_tensor)
        depth_img = depth_predictions["depth"].squeeze().cpu().numpy()
        pts_3d = depth_predictions["points"].squeeze().permute(1, 2, 0).cpu().numpy()


        plane_segs, _, _ = seg_depth(kp_extractor,
                                     sam_mask_generator,
                                     None,
                                     yolo_model,
                                     img_bgr,
                                     depth_img,
                                     pts_3d,
                                     sn_img,
                                     device,
                                     filename=filename,
                                     dbg_dir=DBG_DIR)



if __name__ == "__main__":
    #main_single_img()
    main_folder()
