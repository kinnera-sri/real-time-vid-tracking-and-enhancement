import sys
sys.path.append("../TinySAM")

from tinysam import sam_model_registry, SamHierarchicalMaskGenerator
from demo_hierachical_everything import visualize_filtered_points,  show_anns

from mask_filter_utils import (
    create_parent_upward_mask,
    create_parent_downward_mask,
    filter_masks_by_dsine_variance,
    get_person_mask_from_yolo,
    remove_outliers
)

from vis_utils import (visualize_parent_upward_mask, visualize_parent_downward_mask,
                        visualize_masks_with_dsine_variance, plot_all_filtered_masks_with_dsine, 
                        visualize_all_detected_lines, visualize_lcnn_masks)

from rectangle_detection import (
    setup_lcnn_model,
    filter_masks_with_lcnn,
)
import torch, cv2
import numpy as np
import matplotlib.pyplot as plt
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from ultralytics import YOLO

def calculate_iou(mask1, mask2):

    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(
            mask2.astype(np.uint8),
            (mask1.shape[1], mask1.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    valid_index = mask1 > 0
    mask2_at_valid_index = mask2[valid_index]
    overlap_count = np.sum(mask2_at_valid_index > 0)
    total_mask1_pixels = np.sum(valid_index)

    if total_mask1_pixels == 0:
        return 0.0
    
    overlap_ratio = overlap_count / total_mask1_pixels
    
    return overlap_ratio

def filter_masks_by_iou(masks, input_mask, iou_threshold=0.6):
    
    filtered_masks = []
    removed_masks = []
    iou_scores = {}
    
    for idx, mask_dict in enumerate(masks):
        mask = mask_dict['segmentation']
        
        iou = calculate_iou(mask, input_mask)
        iou_scores[idx] = iou
        
        if iou <= iou_threshold:
            filtered_masks.append(mask_dict)
        else:
            removed_masks.append((idx, mask_dict, iou))
    
    return filtered_masks, iou_scores, removed_masks


############################ images

name = "vlcsnap-2026-02-05-11h41m01s236"

# orig image
image_path = f'sitcom_images_RGB/{name}.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# DSINE image
dsine_image = cv2.imread(f"sitcom_images_DSINE/{name}.png")
dsine_image = cv2.cvtColor(dsine_image, cv2.COLOR_BGR2RGB)

# image = cv2.resize(image, (512, 512))
# dsine_image = cv2.resize(dsine_image, (512, 512))



######################## SAM, YOLO init #################

model_type = "vit_t"
sam = sam_model_registry[model_type](checkpoint="../TinySAM/weights/tinysam.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device=device)
sam.eval()
mask_generator = SamHierarchicalMaskGenerator(sam, points_per_side=48)


yolo_model = YOLO("yolov8l-seg.pt")


# person mask from YOLOv8 segmentation
kernel = np.ones((10,10), np.uint8)  # kept big dilation to remove motion blur on people.
person_mask = get_person_mask_from_yolo(image, yolo_model)

person_mask = cv2.dilate(person_mask, kernel, iterations=2)
person_mask = person_mask.astype(bool)




#
masks = mask_generator.hierarchical_generate(image, valid_masks=None)

del yolo_model , mask_generator
# remove non connected masks
masks, _, _ = remove_outliers(
    masks, min_component_ratio=0.01
)


# floor mask
parent_upward_mask = create_parent_upward_mask(masks, dsine_image, normal_threshold=0.6)

# ceil mask
parent_downward_mask = create_parent_downward_mask(masks, dsine_image, normal_threshold=0.6)


print(person_mask.shape, parent_upward_mask.shape)

# add upward and downward masks to people mask
input_mask = person_mask.astype(bool) | parent_upward_mask.astype(bool) | parent_downward_mask.astype(bool)


filtered_masks, iou_scores, removed_masks = filter_masks_by_iou(
    masks, 
    input_mask, 
    iou_threshold=0.6
)


if 0:
    plot_all_filtered_masks_with_dsine(filtered_masks, dsine_image, top_k=10)


if 0:

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    axes[0, 0].imshow(input_mask, cmap='gray')
    axes[0, 0].set_title(f'people mask', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].set_title(f'All Masks ({len(masks)})', fontsize=12)
    axes[0, 1].axis('off')
    show_anns(image, masks, ax=axes[0, 1])

    axes[1, 0].set_title(f'Filtered Masks: {len(filtered_masks)}', fontsize=12)
    axes[1, 0].axis('off')
    if len(filtered_masks) > 0:
        show_anns(image, filtered_masks, ax=axes[1, 0])
    else:
        axes[1, 0].imshow(image)

    axes[1, 1].set_title(f'Removed Masks: {len(removed_masks)}', fontsize=12)
    axes[1, 1].axis('off')
    removed_masks_only = [m[1] for m in removed_masks]
    if len(removed_masks_only) > 0:
        show_anns(image, removed_masks_only, ax=axes[1, 1])
    else:
        axes[1, 1].imshow(image)
    
    plt.tight_layout()
    plt.show()

print('before ', len(masks))
print('after ', len(filtered_masks))

# remove masks with high DSINE variance and keep ones which has like uniform normals
low_variance_masks, high_variance_masks = filter_masks_by_dsine_variance(
    filtered_masks, dsine_image, variance_threshold=0.03
)
filtered_masks = low_variance_masks  


if 0:  
    print("\n--- Low-variance masks (kept) ---")
    visualize_masks_with_dsine_variance(low_variance_masks, dsine_image, image,
                                        title_prefix="LowVar")
    print("\n--- High-variance masks (discarded) ---")
    visualize_masks_with_dsine_variance(high_variance_masks, dsine_image, image,
                                        title_prefix="HighVar")


##############################################################
# Lightglue init
#############################################################

extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor

feats0 = extractor.extract(torch.tensor(image).permute(2,0,1).float().cuda()/255.0)

# Extract keypoints for analysis
kpts = feats0['keypoints'].cpu().numpy()[0]  # x, y order here

if 0:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(image)
    ax.scatter(kpts[:, 0], kpts[:, 1], s=10, c='red', alpha=0.6, edgecolors='yellow', linewidth=0.5)
    ax.set_title(f'kps ({len(kpts)} points)', fontsize=14)
    plt.tight_layout()
    plt.show()


point_distribution_masks = []
threshold_num_points = 2  # min n.o  points in each side (left, right, up, down) of the mask bbox

for mask_dict in filtered_masks:
    bbox = mask_dict['bbox']  # [x, y, W, H]
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    mask_keypoints = kpts[(kpts[:, 0] >= x1) & (kpts[:, 0] <= x2) & 
                          (kpts[:, 1] >= y1) & (kpts[:, 1] <= y2)]
    
    if len(mask_keypoints) == 0:
        continue  
    
    # left right
    x_mid = (x1 + x2) / 2
    vert_num_of_point_left = np.sum(mask_keypoints[:, 0] < x_mid)
    vert_num_of_point_right = np.sum(mask_keypoints[:, 0] >= x_mid)
    
    # up down
    y_mid = (y1 + y2) / 2
    hori_num_of_point_up = np.sum(mask_keypoints[:, 1] < y_mid)
    hori_num_of_point_down = np.sum(mask_keypoints[:, 1] >= y_mid)
    
    if (vert_num_of_point_left >= threshold_num_points and 
        vert_num_of_point_right >= threshold_num_points and
        hori_num_of_point_up >= threshold_num_points and 
        hori_num_of_point_down >= threshold_num_points):
        
        mask_with_kpt_dist = mask_dict.copy()
        mask_with_kpt_dist['keypoint_dist'] = {
            'vert_num_of_point_left': vert_num_of_point_left,
            'vert_num_of_point_right': vert_num_of_point_right,
            'hori_num_of_point_up': hori_num_of_point_up,
            'hori_num_of_point_down': hori_num_of_point_down,
            'total_keypoints_in_bbox': len(mask_keypoints)
        }
        point_distribution_masks.append(mask_with_kpt_dist)

print(f"\n after kpy based fitlering: {len(point_distribution_masks)} / {len(filtered_masks)}")

################## lcnn model init ########
if True:
    lcnn_model_name  = checkpoint_file="../lcnn/lcnn_custom.pth"
else:
    lcnn_model_name  = checkpoint_file="../lcnn/190418-201834-f8934c6-lr4d10-312k.pth"

lcnn_model, lcnn_device = setup_lcnn_model(
    config_file="../lcnn/config/wireframe.yaml",
    checkpoint_file=lcnn_model_name
)

score_threshold = 0.95

print(person_mask.shape, person_mask.sum(), type(person_mask), person_mask.max())

print(f"mask before processing {len(point_distribution_masks)} ")
lcnn_masks, all_lines, all_scores = filter_masks_with_lcnn(
    lcnn_model, 
    lcnn_device, 
    point_distribution_masks,
    image,
    person_mask,
    score_threshold=score_threshold,
    apply_postprocess=False,
    show_postprocess_viz=True,
    dsine_image=dsine_image
)

print(f"final masks {len(lcnn_masks)} ")

# Visualize all detected lines from LCNN
if len(all_lines) > 0:
    visualize_all_detected_lines(image, all_lines, all_scores, score_threshold=score_threshold)


# final masks with pred rects
if len(lcnn_masks) > 0:
    visualize_lcnn_masks(lcnn_masks, image, kpts, dsine_image=dsine_image, top_k=10)



