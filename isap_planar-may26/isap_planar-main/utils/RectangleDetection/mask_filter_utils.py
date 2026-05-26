"""
Upward-facing normal detection using DSINE surface normals.
Functions for filtering masks based on upward surface orientation.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_upward_facing_mask(dsine_image, normal_threshold=0.5):

    dsine_norm = dsine_image.astype(np.float32) / 255.0
    z_component =  dsine_norm[:, :, 0]
    upward_mask = z_component > normal_threshold
    
    return upward_mask.astype(np.uint8)



def create_upward_masks_from_filtered(filtered_masks, dsine_image, normal_threshold=0.5):

    upward_masks = []
    
    global_upward_mask = get_upward_facing_mask(dsine_image, normal_threshold)
    
    for mask_dict in filtered_masks:
        seg_mask = mask_dict['segmentation'].astype(np.uint8)
        
        combined_mask = (seg_mask > 0) & (global_upward_mask > 0)
        
        new_area = np.sum(combined_mask)
        
        if new_area > 0:
            new_mask_dict = mask_dict.copy()
            new_mask_dict['segmentation'] = combined_mask
            new_mask_dict['area'] = new_area
            upward_masks.append(new_mask_dict)
    
    return upward_masks



def create_parent_upward_mask(filtered_masks, dsine_image, normal_threshold=0.5):

    parent_mask = np.zeros((dsine_image.shape[0], dsine_image.shape[1]), dtype=np.uint8)
    
    global_upward_mask = get_upward_facing_mask(dsine_image, normal_threshold).astype(bool)
    
    for mask_dict in filtered_masks:
        seg_mask = mask_dict['segmentation'].astype(bool)
        
        parent_mask[seg_mask & global_upward_mask] = 255
    
    
    return parent_mask


def get_downward_facing_mask(dsine_image, normal_threshold=0.5):

    dsine_norm = dsine_image.astype(np.float32) / 255.0

    z_component =  dsine_norm[:, :, 0]
    
    downward_mask = z_component < -normal_threshold
    
    return downward_mask.astype(np.uint8)


def create_parent_downward_mask(filtered_masks, dsine_image, normal_threshold=0.5):

    parent_mask = np.zeros((dsine_image.shape[0], dsine_image.shape[1]), dtype=np.uint8)
    
    global_downward_mask = get_downward_facing_mask(dsine_image, normal_threshold).astype(bool)
    
    for mask_dict in filtered_masks:
        seg_mask = mask_dict['segmentation'].astype(bool)
        
        parent_mask[seg_mask & global_downward_mask] = 255
    
    downward_pixel_count = np.sum(parent_mask > 0)
    
    return parent_mask


def compute_mask_dsine_variance(mask_dict, dsine_image):

    seg_mask = mask_dict['segmentation'].astype(bool)
    dsine_norm = dsine_image.astype(np.float32) / 255.0
    dsine_norm = dsine_norm * 2.0 - 1.0 # 0 to 1 to -1 to 1
    pixels = dsine_norm[seg_mask]   
    variance_per_channel = np.var(pixels, axis=0)  
    return float(np.mean(variance_per_channel))


def filter_masks_by_dsine_variance(masks, dsine_image, variance_threshold=0.05):

    low_variance_masks = []
    high_variance_masks = []

    for mask_dict in masks:
        var = compute_mask_dsine_variance(mask_dict, dsine_image)
        mask_with_var = mask_dict.copy()
        mask_with_var['dsine_variance'] = var

        if var <= variance_threshold:
            low_variance_masks.append(mask_with_var)
        else:
            high_variance_masks.append(mask_with_var)


    return low_variance_masks, high_variance_masks




def get_person_mask_from_yolo(image_rgb, yolo_model):

    results = yolo_model.predict(image_rgb, conf=0.3, save=False, verbose=False, classes=[0])
    
    if len(results) > 0 and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
        
        combined_mask = (combined_mask * 255).astype(np.uint8)
        combined_mask = cv2.resize(combined_mask, (image_rgb.shape[1], image_rgb.shape[0]))
        return combined_mask
    else:
        # No person detected - return zero mask matching image dimensions
        return np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)



def clean_mask_largest_component(mask_dict):
    seg = mask_dict['segmentation'].astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg, connectivity=8)

    # stats[0] is background; find largest foreground component
    foreground_stats = stats[1:]  # skip background
    largest_idx = int(np.argmax(foreground_stats[:, cv2.CC_STAT_AREA])) + 1  # +1 for background offset

    clean_seg = (labels == largest_idx).astype(np.uint8)

    new_area = int(np.sum(clean_seg))

    ys, xs = np.where(clean_seg > 0)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    new_dict = mask_dict.copy()
    new_dict['segmentation'] = clean_seg.astype(bool)
    new_dict['area'] = new_area
    new_dict['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    return new_dict


def remove_outliers(masks, min_component_ratio=0.01):

    cleaned_masks = []
    dirty_indices = []
    original_dirty = []

    for i, mask_dict in enumerate(masks):
        seg = mask_dict['segmentation'].astype(np.uint8)
        total_pixels = int(np.sum(seg))
        if total_pixels == 0:
            continue

        cleaned = clean_mask_largest_component(mask_dict)
        if cleaned is None:
            continue

        kept_pixels = int(np.sum(cleaned['segmentation'].astype(np.uint8)))
        removed_ratio = (total_pixels - kept_pixels) / total_pixels

        if removed_ratio > min_component_ratio:
            dirty_indices.append(i)
            original_dirty.append(mask_dict)

        cleaned_masks.append(cleaned)
        
    return cleaned_masks, dirty_indices, original_dirty
