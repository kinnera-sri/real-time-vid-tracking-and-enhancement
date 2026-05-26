import matplotlib.pyplot as plt
from mask_filter_utils import compute_mask_dsine_variance
import numpy as np
import matplotlib as mpl


def visualize_parent_upward_mask(parent_mask, image, dsine_image):
    """
    Visualize the parent upward mask alongside the original image and DSINE.
    
    Args:
        parent_mask: Parent mask created from upward-facing regions
        image: Original RGB image
        dsine_image: DSINE depth/normal image
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(parent_mask, cmap='gray')
    axes[1].set_title('Parent Upward Mask (values: 0 or 255)', fontsize=12)
    axes[1].axis('off')
    
    if len(dsine_image.shape) == 3 and dsine_image.shape[2] >= 3:
        axes[2].imshow(dsine_image[:, :, :3])
    else:
        axes[2].imshow(dsine_image, cmap='viridis')
    axes[2].set_title('DSINE', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_parent_downward_mask(image, parent_downward_mask):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(parent_downward_mask, cmap='gray')
    axes[1].set_title('Parent Downward Mask', fontsize=12) 
    axes[1].axis('off')
    
    overlay_img = image.copy()
    overlay_img[parent_downward_mask > 0] = [255, 0, 0]
    axes[2].imshow(overlay_img)
    axes[2].set_title('Downward Mask Overlay', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()





def visualize_masks_with_dsine_variance(masks, dsine_image, image, title_prefix="Mask"):
    """
    Iteratively show each mask alongside its DSINE crop and variance value.

    Args:
        masks: List of mask dicts (should already have 'dsine_variance' key, or it is computed here)
        dsine_image: DSINE output image (H, W, 3)
        image: Original RGB image
        title_prefix: Prefix for subplot titles
    """
    if len(masks) == 0:
        print("No masks to visualise.")
        return

    dsine_norm = dsine_image.astype(np.float32)
    if dsine_norm.max() > 1.0:
        dsine_norm = dsine_norm / 255.0

    for i, mask_dict in enumerate(masks):
        seg_mask = mask_dict['segmentation'].astype(np.uint8)
        bbox = mask_dict['bbox']
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        var = mask_dict.get('dsine_variance',
                            compute_mask_dsine_variance(mask_dict, dsine_image))

        # Crop to bbox
        dsine_crop = dsine_image[y1:y2, x1:x2]
        mask_crop  = seg_mask[y1:y2, x1:x2]
        img_crop   = image[y1:y2, x1:x2]

        # Masked DSINE (zero out background)
        dsine_masked = dsine_crop.copy()
        dsine_masked[mask_crop == 0] = 0

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_crop)
        axes[0].set_title(f"{title_prefix} {i} — Image crop", fontsize=11)
        axes[0].axis('off')

        axes[1].imshow(mask_crop, cmap='gray')
        axes[1].set_title(f"{title_prefix} {i} — Mask  |  area={mask_dict['area']:.0f}",
                          fontsize=11)
        axes[1].axis('off')

        axes[2].imshow(dsine_masked[:, :, :3] if dsine_masked.ndim == 3 else dsine_masked,
                       cmap='viridis' if dsine_masked.ndim < 3 else None)
        axes[2].set_title(f"{title_prefix} {i} — DSINE  |  variance={var:.4f}", fontsize=11)
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()




def crop_dsine_by_mask_bbox(mask_dict, dsine_image):


    """
    Crop DSINE image region based on mask bounding box.
    Extract only the pixels where mask > 0, rest set to 0.
    
    Args:
        mask_dict: Single mask dict with 'segmentation' and 'bbox' keys
        dsine_image: DSINE depth/normal image
        
    Returns:
        Tuple of (cropped_dsine, mask_bbbox, mask_region)
        - cropped_dsine: DSINE values cropped to bbox region with mask applied
        - bbox: Bounding box coordinates [x1, y1, x2, y2]
        - mask_region: Extracted mask region
    """
    # Get mask and bbox
    mask = mask_dict['segmentation'].astype(np.uint8)
    bbox = mask_dict['bbox']  # [x, y, width, height]
    
    # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # Clip to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(dsine_image.shape[1], x2)
    y2 = min(dsine_image.shape[0], y2)
    
    # Crop DSINE image
    dsine_crop = dsine_image[y1:y2, x1:x2].copy()
    
    # Crop mask to same region
    mask_crop = mask[y1:y2, x1:x2]
    
    # Apply mask: keep only pixels where mask > 0
    dsine_masked = dsine_crop.copy()
    dsine_masked[mask_crop == 0] = 0
    
    bbox_coords = [x1, y1, x2, y2]
    
    return dsine_masked, bbox_coords, mask_crop



def plot_all_filtered_masks_with_dsine(filtered_masks, dsine_image, top_k=None):

    # Sort by area and select top K if specified
    if top_k is not None:
        sorted_masks = sorted(filtered_masks, key=lambda x: x['area'], reverse=True)
        masks_to_plot = sorted_masks[:top_k]
    else:
        masks_to_plot = filtered_masks
    
    num_to_plot = len(masks_to_plot)
    
    # Create grid layout
    cols = 2  # Two columns: mask and DSINE
    rows = num_to_plot
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * num_to_plot))
    
    # Handle single mask case
    if num_to_plot == 1:
        axes = axes.reshape(1, -1)
    
    for idx, mask_dict in enumerate(masks_to_plot):
        # Crop and extract
        dsine_masked, bbox, mask_crop = crop_dsine_by_mask_bbox(mask_dict, dsine_image)

        # Plot mask
        axes[idx, 0].imshow(mask_crop, cmap='gray')
        axes[idx, 0].set_title(f'Mask {idx} - Area: {mask_dict["area"]:.0f}', fontsize=10)
        axes[idx, 0].axis('off')
        
        # Plot DSINE
        if len(dsine_masked.shape) == 3 and dsine_masked.shape[2] >= 3:
            axes[idx, 1].imshow(dsine_masked[:, :, :3])
        else:
            axes[idx, 1].imshow(dsine_masked, cmap='viridis')
        
        axes[idx, 1].set_title(f'Mask {idx} - DSINE (bbox: {bbox})', fontsize=10)
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.show()



def visualize_all_detected_lines(image, all_lines, all_scores, score_threshold=0.85):
    """
    Visualize all detected lines on the full image and return as numpy array.
    Better version that handles visualization properly.
    
    Args:
        image: Original RGB image
        all_lines: All detected lines from LCNN (full image resolution) in [y, x] format
        all_scores: Confidence scores for all lines
        score_threshold: Minimum confidence to display
        
    Returns:
        numpy array of visualization (for saving)
    """
    if len(all_lines) == 0:
        print("No lines to visualize!")
        return None
    
    # Create colormap for confidence scores
    cmap = plt.get_cmap("jet")
    norm = mpl.colors.Normalize(vmin=max(all_scores.min(), 0.7), vmax=1.0)
    
    # Count lines by score
    high_conf = np.sum(all_scores >= 0.9)
    med_conf = np.sum((all_scores >= score_threshold) & (all_scores < 0.9))
    low_conf = np.sum(all_scores < score_threshold)
    
    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.imshow(image)
    
    # Draw all lines using matplotlib (same style as visualize_lines_comparison)
    for line, score in zip(all_lines, all_scores):
        if score >= score_threshold:
            p1, p2 = line[0], line[1]
            color = cmap(norm(score))
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color=color, linewidth=2, alpha=0.8)
            ax.scatter([p1[1], p2[1]], [p1[0], p2[0]], color=color, s=20, alpha=0.8)
    
    ax.set_title(f'All Detected Lines by LCNN | Total: {len(all_lines)} | '
                f'High-conf (≥0.9): {high_conf} | '
                f'Med-conf (≥{score_threshold}): {med_conf} | '
                f'Low-conf (<{score_threshold}): {low_conf}',
                fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Confidence Score', fraction=0.046, pad=0.04)
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig




def visualize_lcnn_masks(lcnn_masks, image, kpts, dsine_image=None, top_k=5):
    """
    Comprehensive visualization of masks detected as rectangles by LCNN.
    Shows: mask, lines on image, keypoints, DSINE
    
    Args:
        lcnn_masks: List of masks with LCNN line detection results
        image: Original RGB image
        kpts: Keypoints array
        dsine_image: DSINE depth/normal image (optional)
        top_k: Number of masks to visualize
    """
    if len(lcnn_masks) == 0:
        print("No LCNN-detected rectangular masks found!")
        return
    
    num_to_show = min(top_k, len(lcnn_masks))
    
    cmap = plt.get_cmap("jet")
    norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    def c(x):
        return sm.to_rgba(x)
    
    for mask_idx, mask_dict in enumerate(lcnn_masks[:num_to_show]):
        bbox = mask_dict['bbox']
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        # Get mask and side lines
        mask = mask_dict['segmentation'].astype(np.uint8)
        side_lines = mask_dict['side_lines']  # Dict with 'left', 'right', 'up', 'down'
        side_scores = mask_dict['side_scores']
        all_lines = mask_dict['lcnn_all_lines']
        all_scores = mask_dict['lcnn_all_scores']
        
        # Get keypoints in bbox
        mask_kpts = kpts[(kpts[:, 0] >= x1) & (kpts[:, 0] <= x2) & 
                         (kpts[:, 1] >= y1) & (kpts[:, 1] <= y2)]
        
        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Panel 1: Original image with detected lines
        ax = axes[0, 0]
        ax.imshow(image)
        
        # Draw the 4 side lines with different colors
        colors_per_side = {'left': 'red', 'right': 'blue', 'up': 'green', 'down': 'orange'}
        
        for side_name, line in side_lines.items():
            if line is not None:
                p1, p2 = line[0], line[1]
                y_p1, x_p1 = p1[0], p1[1]
                y_p2, x_p2 = p2[0], p2[1]
                score = side_scores[side_name]
                
                # Plot line directly (no offset - lines are already in full image coords)
                ax.plot([x_p1, x_p2], [y_p1, y_p2], 
                       color=colors_per_side[side_name], linewidth=3, label=f'{side_name}', zorder=10)
                # Mark endpoints
                ax.scatter([x_p1, x_p2], [y_p1, y_p2], s=30, c=colors_per_side[side_name], 
                          edgecolors='white', linewidth=1.5, zorder=11)
        
        # Draw bbox
        rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='cyan', fill=False, linestyle='--', alpha=0.7)
        ax.add_patch(rect)
        
        ax.set_title(f'LCNN Lines (4 sides detected)', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.axis('off')
        
        # Panel 2: Mask segmentation
        ax = axes[0, 1]
        ax.imshow(mask, cmap='gray')
        ax.set_title('Mask Segmentation', fontsize=12)
        ax.axis('off')
        
        # Panel 3: Image with keypoints
        ax = axes[1, 0]
        ax.imshow(image)
        
        # Draw bbox and dividing lines
        # rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='cyan', fill=False)
        # ax.add_patch(rect)
        
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        ax.axvline(x=x_mid, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=y_mid, color='purple', linestyle='--', linewidth=1, alpha=0.5)
        
        if len(mask_kpts) > 0:
            ax.scatter(mask_kpts[:, 0], mask_kpts[:, 1], s=10, c='red', alpha=0.7,
                      edgecolors='yellow', linewidth=0.5)
        
        kpt_dist = mask_dict['keypoint_dist']
        ax.set_title(f'Keypoints | Total: {len(mask_kpts)} | L:{kpt_dist["vert_num_of_point_left"]} '
                    f'R:{kpt_dist["vert_num_of_point_right"]}', fontsize=12)
        ax.axis('off')
        
        # Panel 4: DSINE or regular image
        ax = axes[1, 1]
        if dsine_image is not None:
            dsine_crop = dsine_image[y1:y2, x1:x2].copy()
            if len(dsine_crop.shape) == 3 and dsine_crop.shape[2] >= 3:
                ax.imshow(dsine_crop[:, :, :3])
            else:
                ax.imshow(dsine_crop, cmap='viridis')
            ax.set_title('DSINE (Cropped)', fontsize=12)
        else:
            img_crop = image[y1:y2, x1:x2]
            ax.imshow(img_crop)
            ax.set_title('Image (Cropped)', fontsize=12)
        ax.axis('off')
        
        # Add overall title
        num_sides = sum(1 for line in side_lines.values() if line is not None)
        fig.suptitle(f'LCNN Rectangle Mask {mask_idx} | {num_sides} sides detected',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
