import numpy as np
import cv2
import os
from collections import Counter
import trimesh
import matplotlib.pyplot as plt


def rotation_matrix_from_vectors(a, b):
    """
    Returns rotation matrix that rotates vector a to vector b
    """
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.linalg.norm(v) < 1e-8:
        # vectors are parallel
        if c > 0:
            return np.eye(3)  # same direction
        else:
            # opposite direction → rotate 180° around any perpendicular axis
            # find arbitrary orthogonal vector
            axis = np.array([1, 0, 0])
            if abs(a[0]) > 0.9:
                axis = np.array([0, 1, 0])
            v = np.cross(a, axis)
            v /= np.linalg.norm(v) + 1e-8
            return rotation_matrix_axis_angle(v, np.pi)

    s = np.linalg.norm(v)
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return R


def rotation_matrix_axis_angle(axis, angle):
    axis /= np.linalg.norm(axis) + 1e-8
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]
    ])

def get_hori_vert_reference_vectors(plane_normal):
    plane_normal = np.array(plane_normal, dtype=np.float32)
    plane_normal /= np.linalg.norm(plane_normal) + 1e-8

    # Reference front plane
    front_normal = np.array([0, 0, 1])
    ref_vertical = np.array([0, 1, 0])
    ref_horizontal = np.array([-1, 0, 0])

    # Compute rotation
    R = rotation_matrix_from_vectors(front_normal, plane_normal)

    # Rotate reference vectors
    vertical = R @ ref_vertical
    horizontal = R @ ref_horizontal

    horizontal /= np.linalg.norm(horizontal) + 1e-8
    vertical /= np.linalg.norm(vertical) + 1e-8

    return horizontal, vertical

def get_angle_for_line(line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def sort_lines_by_length(lines):
    def line_length(line):
        x1, y1 = line[0]
        x2, y2 = line[1]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return sorted(lines, key=line_length, reverse=True)


def get_line_3d_vector(line, points, H, W):
    """Get normalized 3D direction vector from line endpoints."""
    x1, y1 = int(line[0][0]), int(line[0][1])
    x2, y2 = int(line[1][0]), int(line[1][1])
    y1c, x1c = np.clip(y1, 0, H - 1), np.clip(x1, 0, W - 1)
    y2c, x2c = np.clip(y2, 0, H - 1), np.clip(x2, 0, W - 1)
    p1 = points[y1c, x1c]
    p2 = points[y2c, x2c]
    vec = p2 - p1
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return None
    return vec / norm


def angle_between_normals(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle = np.arccos(dot_product)
    return np.rad2deg(angle)


def line_on_mask_fraction(line, mask, pts3d_img=None):
    """Return fraction of line pixels that lie on mask using bitwise intersection."""
    line_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    pt1 = tuple(map(int, line[0]))
    pt2 = tuple(map(int, line[1]))

    cv2.line(line_mask, pt1, pt2, color=255, thickness=1)
    plane_line_mask = np.logical_and(mask.astype(bool), line_mask.astype(bool))

    if False:
        fig, ax = plt.subplots(nrows=2)
        ax[0].imshow(line_mask)
        ax[1].imshow(mask)
        plt.show()
    
    # Total pixels in the line / Total pixels in the intersection
    num_line_pixels = np.sum(line_mask.astype(bool))
    if num_line_pixels == 0:
        return 0.0
    
    num_intersection_pixels = np.sum(plane_line_mask)
    
    score = 1.0*num_intersection_pixels / num_line_pixels

    return score


def line_on_mask_fraction_v1(plane_coeffs, line, mask, pts_3d_img):
    """Return fraction of line pixels that lie on mask using bitwise intersection."""

    line_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    pt1 = tuple(map(int, line[0]))
    pt2 = tuple(map(int, line[1]))

    cv2.line(line_mask, pt1, pt2, color=1, thickness=1)
    plane_line_mask = np.logical_and(mask.astype(bool), line_mask.astype(bool))

    if False:
        fig, ax = plt.subplots(nrows=2)
        ax[0].imshow(line_mask)
        ax[1].imshow(mask)
        plt.show()
    
    #intersection = cv2.bitwise_and(line_mask, (mask > 0).astype(np.uint8) * 255)
    intersection = np.logical_and(line_mask.astype(bool), mask.astype(bool))
    
    # Total pixels in the line / Total pixels in the intersection
    num_line_pixels = np.sum(line_mask.astype(bool))
    
    num_intersection_pixels = np.sum(intersection)
    
    score = num_intersection_pixels / num_line_pixels

    # Sometimes major lines are ommitted by the segments mask
    # in such cases we use the point cloud and see if the points in the line
    # belong to the plane
    #print(f"score {score} ")
    if score == 0 :
        line_pts = np.column_stack(np.where(line_mask > 0))
        if len(line_pts):
            line_pts3d = pts_3d_img[line_pts[:,0], line_pts[:,1]]
            unormal, d, p1 = plane_coeffs
            dist = trimesh.points.point_plane_distance(points=line_pts3d, 
                                                       plane_normal=unormal, 
                                                       plane_origin=p1)
            num_intersection_pixels = np.sum(np.abs(dist) < 1e-1)

            score = 1.0*num_intersection_pixels / num_line_pixels
            #print(f"new score {score} using pont cloud method")

    return score


def find_dominant_vector(line_list, plane_normal=None, label="", points=None, H=1080, W=1920):
    """Sort the list in descending order of line length, then get 3d vectors, then return the first line which makes between 75 and 105 degrees with plane normal. return None if no line satisfies the condition. """
    
    # Ensure consistent direction
    hori_ref, vert_ref = get_hori_vert_reference_vectors(plane_normal)
    
    # line_list = sort_lines_by_length(line_list)
    for line in line_list:
        vec = get_line_3d_vector(line, points, H, W)
        if vec is not None and plane_normal is not None:
            angle = angle_between_normals(vec, plane_normal)
            if (abs(90 - angle) <= 7):
                print(f"    {label}: selected line {line} with angle {angle:.1f} to plane normal")
                if label == "Vertical":
                    reference = vert_ref
                elif label == "Horizontal":
                    reference = hori_ref

                dots = np.dot(vec, reference)

                if dots < 0:
                    vec = -vec
                return vec, np.array(line_list), np.array(line)
            else:
                print(f"    {label}: skipping line {line} with angle {angle:.1f} to plane normal")

    return None, None, None


def find_dominant_vector_ransac(line_list, plane_normal=None, label="", points=None, H=1080, W=1920):
    """Get dominant direction vector from a list of lines."""
    vectors = []
    filtered_lines = []
    
    # Pre-computation check
    if plane_normal is not None:
         # Ensure plane_normal is unit vector
        pn_norm = np.linalg.norm(plane_normal)
        if pn_norm > 1e-8:
            plane_normal = plane_normal / pn_norm
    
    for line in line_list:
        vec = get_line_3d_vector(line, points, H, W)
        
        if vec is not None:
            # if vec makes less than 75 degrees  or less than 105 degrees with plane normal, don't append it to the list
            if plane_normal is not None:
                angle = angle_between_normals(vec, plane_normal)
                if (abs(90 - angle) > 10):
                    print(f"    {label}: skipping line {line} with angle {angle:.1f} to plane normal")
                    continue


            vectors.append(vec)
            filtered_lines.append(line)

    if len(vectors) == 0:
        print(f"    {label}: no valid 3D vectors")
        return None, None, None

    vectors = np.array(vectors)
    filtered_lines = np.array(filtered_lines)

    # Ensure consistent direction
    hori_ref, vert_ref = get_hori_vert_reference_vectors(plane_normal)
    if len(vectors) > 0:
        if label == "vertical":
            reference = vert_ref
        elif label == "horizontal":
            reference = hori_ref
        else:
            reference = vectors[0]
        dots = np.dot(vectors, reference)
        vectors[dots < 0] *= -1

    if len(vectors) > 2:
        inlier_mask = ransac_direction(vectors, angle_threshold_deg=10.0)
        inlier_vectors = vectors[inlier_mask]
        inlier_lines = filtered_lines[inlier_mask]
    else:
        inlier_vectors = vectors
        inlier_lines = filtered_lines

    if len(inlier_vectors) == 0:
        return None, None, None

    # Round to 2 decimals to group essentially identical ones initially
    rounded = np.round(inlier_vectors, 2)
    # Convert each vector to unique rows and counting
    common_vecs, initial_counts = np.unique(rounded, axis=0, return_counts=True)

    if plane_normal is not None:
        # sort common_vecs, initial_counts based on angle to plane normal
        angles = np.array([angle_between_normals(vec, plane_normal) for vec in common_vecs])
        sorted_indices = np.argsort(np.abs(90 - angles))
        common_vecs = common_vecs[sorted_indices]
        initial_counts = initial_counts[sorted_indices]

    # merge similar normals that are within angle threshold
    common_vecs, common_counts = merge_similar_normals(common_vecs, initial_counts, angle_threshold=5)


    # --- Selection Logic ---
    max_count = np.max(common_counts)
    candidate_indices = np.where(common_counts == max_count)[0]
    
    selected_idx = -1
    
    if len(candidate_indices) == 1:
        # one candidate with max count
        selected_idx = candidate_indices[0]
    else:
        # multiple candidates with same max count
        print(f"    {label}: max count {max_count} among {len(candidate_indices)} vectors")
        if plane_normal is not None:
            best_diff = float('inf')
            best_idx = -1
            
            for idx in candidate_indices:
                vec = common_vecs[idx]
                n_vec = np.linalg.norm(vec)
                if n_vec > 1e-8:
                    vec = vec / n_vec
                
                angle = angle_between_normals(vec, plane_normal)
                diff = abs(90 - angle)
                
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
            
            selected_idx = best_idx
        else:
            # No plane normal, pick first
            selected_idx = candidate_indices[0]

    dominant_vec = common_vecs[selected_idx]

    # find index of dominant vec in original inlier_vectors to get corresponding lines
    best_idx = -1
    smallest_angle = float("inf")

    for i, vec in enumerate(inlier_vectors):
        v = vec / (np.linalg.norm(vec) + 1e-12)
        angle = angle_between_normals(v, dominant_vec)

        if angle < smallest_angle:
            smallest_angle = angle
            best_idx = i

    dominant_line = inlier_lines[best_idx]

    norm = np.linalg.norm(dominant_vec)
    if norm < 1e-8:
        return None, None, None
    dominant_vec = dominant_vec / norm

    return dominant_vec, inlier_lines, dominant_line


def ransac_direction(vectors, angle_threshold_deg=10.0, n_iterations=100):
    """RANSAC to find dominant direction from unit vectors."""
    if len(vectors) < 3:
        return np.ones(len(vectors), dtype=bool)

    best_inliers = None
    best_count = 0
    angle_th_rad = np.deg2rad(angle_threshold_deg)
    
    HAS_MORE_LINES = False
    if len(vectors) < n_iterations:
        n_iterations = len(vectors)
        HAS_MORE_LINES = True

    for idx in range(n_iterations):
        if not HAS_MORE_LINES:
            idx = np.random.randint(len(vectors))
        model = vectors[idx]
        dots = np.clip(np.abs(np.dot(vectors, model)), 0, 1)
        angles = np.arccos(dots)
        inlier_mask = angles < angle_th_rad
        count = int(np.sum(inlier_mask))
        if count > best_count:
            best_count = count
            best_inliers = inlier_mask

    return best_inliers if best_inliers is not None else np.ones(len(vectors), dtype=bool)


def merge_similar_normals(normals, counts, angle_threshold=10):
    merged_normals = []
    merged_counts = []

    for i in range(len(normals)):
        normal = normals[i]
        count = counts[i]

        found_similar = False
        for j in range(len(merged_normals)):
            angle = angle_between_normals(normal, merged_normals[j])
            if angle < angle_threshold:
                # high count gets priority
                merged_normals[j] = (
                    merged_normals[j] if merged_counts[j] >= count else normal
                )
                # add counts
                merged_counts[j] += count
                found_similar = True
                break

        if not found_similar:
            merged_normals.append(normal)
            merged_counts.append(count)

    return np.array(merged_normals), np.array(merged_counts)

def _project_bbox_3d_to_2d(
    point_3d, normal_2, normal_3, fx, fy, cx, cy, height=0.5, width=0.5
):
    """
    Takes a 3D point and two direction vectors, computes 3D bbox corners using
    height (along normal_2) and width (along normal_3), and projects them to 2D.

    Args:
        normal_2: vertical direction vector (3,)
        normal_3: horizontal direction vector (3,)
        fx, fy, cx, cy: camera intrinsics
        height: 3D length along normal_2 (vertical)
        width: 3D length along normal_3 (horizontal)

    Returns:
        dict of 2D projected points
    """
    n2 = np.array(normal_2, dtype=np.float32)
    n3 = np.array(normal_3, dtype=np.float32)

    n2 /= np.linalg.norm(n2) + 1e-8
    n3 /= np.linalg.norm(n3) + 1e-8

    half_h = abs(height / 2.0)
    half_w = abs(width / 2.0)

    def project(pt):
        x, y, z = pt
        if z == 0:
            z = 1e-6
        u = int(fx * x / z + cx)
        v = int(fy * y / z + cy)
        return [u, v]

    pts_3d = [
        point_3d - n2 * half_h + n3 * half_w, # TL
        point_3d - n2 * half_h - n3 * half_w, # TR
        point_3d + n2 * half_h - n3 * half_w, # BR
        point_3d + n2 * half_h + n3 * half_w, # BL
    ]

    pts_2d = [project(v) for v in pts_3d]
    return pts_2d

def draw_adjusted_bbox(image, pts_2d, color=(0, 255, 0), thickness=2):
    """
    Draw an adjusted bounding box on image from projected 2D points.
    """
    corners = [
        pts_2d[0],
        pts_2d[1],
        pts_2d[2],
        pts_2d[3],
    ]
    for i in range(4):
        cv2.line(image, corners[i], corners[(i + 1) % 4], color, thickness)
    return image



def project_length_to_3d(length_2d, depth, cam_K):
    """
    Projects a 2D length in the image to a 3D length in the world using the depth and focal length.
    """
    if depth == 0:
        return 0
    f = (cam_K[0, 0] + cam_K[1, 1]) / 2  # average focal length
    length_3d = (length_2d * depth) / f
    return length_3d

def line_correction(image, line, label, plane_mask, depth_map, surface_normal, points, line_count):
    """move line endpoints towards the inside of the plane mask to correct line, if they lie on the edge"""
    line_og = line.copy()
    x1, y1 = line[0]
    x2, y2 = line[1]

    # Check if line is on the edge
    kernel_size = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded_mask = cv2.erode(plane_mask, kernel, iterations=10)

    edge_mask = cv2.bitwise_xor(plane_mask, eroded_mask)

    # line_ann = image.copy()
    # cv2.line(line_ann, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    # image_with_edge = cv2.addWeighted(line_ann, 0.8, cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR), 0.2, 0)
    # cv2.imshow(f"edge_mask_{label}", edge_mask)
    # cv2.imshow(f"line_correction_{label}_checking", image_with_edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if line_on_mask_fraction(line, edge_mask) < 0.3:
        return line
    print(f"    {label}: line {line} is on the edge, applying correction")

    # Get line endpoints in 3d
    p1 = points[int(y1), int(x1)]
    p2 = points[int(y2), int(x2)]

    # Get vector from p1 to p2
    line_vec = p2 - p1
    line_length = np.linalg.norm(line_vec)
    if line_length < 1e-8:
        return line
    line_vec = line_vec / line_length

    # Check angle
    if surface_normal is not None:
        angle = angle_between_normals(line_vec, surface_normal)
        if abs(angle - 90) <= 10:
            return line
    print(f"    {label}: line {line} is not parallel to surface, applying correction")

    # Push line inside mask
    dist_transform = cv2.distanceTransform(plane_mask, cv2.DIST_L2, 5)

    shift_amount = 10
    H, W = plane_mask.shape

    if label == "Horizontal":
        y_up1, y_up2 = max(0, int(y1) - shift_amount), max(0, int(y2) - shift_amount)
        y_dn1, y_dn2 = min(H - 1, int(y1) + shift_amount), min(H - 1, int(y2) + shift_amount)

        dist_up1 = dist_transform[y_up1, int(x1)]
        dist_dn1 = dist_transform[y_dn1, int(x1)]

        dist_up2 = dist_transform[y_up2, int(x2)]
        dist_dn2 = dist_transform[y_dn2, int(x2)]

        dir1 = -1 if dist_up1 > dist_dn1 else (1 if dist_dn1 > dist_up1 else 0)
        dir2 = -1 if dist_up2 > dist_dn2 else (1 if dist_dn2 > dist_up2 else 0)

        # Decide final direction
        if dir1 == dir2:
            direction = dir1
        elif dir1 == 0:
            direction = dir2
        elif dir2 == 0:
            direction = dir1
        else:
            return line_og

        if direction == -1:
            y1 = max(0, y1 - shift_amount)
            y2 = max(0, y2 - shift_amount)
        elif direction == 1:
            y1 = min(H - 1, y1 + shift_amount)
            y2 = min(H - 1, y2 + shift_amount)
    elif label == "Vertical":
        x_lf1, x_lf2 = max(0, int(x1) - shift_amount), max(0, int(x2) - shift_amount)
        x_rt1, x_rt2 = min(W - 1, int(x1) + shift_amount), min(W - 1, int(x2) + shift_amount)

        dist_lf1 = dist_transform[int(y1), x_lf1]
        dist_rt1 = dist_transform[int(y1), x_rt1]

        dist_lf2 = dist_transform[int(y2), x_lf2]
        dist_rt2 = dist_transform[int(y2), x_rt2]

        dir1 = -1 if dist_lf1 > dist_rt1 else (1 if dist_rt1 > dist_lf1 else 0)
        dir2 = -1 if dist_lf2 > dist_rt2 else (1 if dist_rt2 > dist_lf2 else 0)

        if dir1 == dir2:
            direction = dir1
        elif dir1 == 0:
            direction = dir2
        elif dir2 == 0:
            direction = dir1
        else:
            return line_og

        if direction == -1:
            x1 = int(max(0, x1 - shift_amount))
            x2 = int(max(0, x2 - shift_amount))
        elif direction == 1:
            x1 = int(min(W - 1, x1 + shift_amount))
            x2 = int(min(W - 1, x2 + shift_amount))

    new_line = [(int(x1), int(y1)), (int(x2), int(y2))]

    return new_line

def warp_bbox_using_depth(img, depth_results, pts_3d_img, plane, lines):
    """
    Warp bounding boxes using depth-based 3D projection.
    Adds 'warp_ad_roi' key to each APO in plane['APO_list'].
    Returns the plane dict.
    """
    dbg_str = ""
    warped_rois = []

    H, W = img.shape[:2]

    # Get camera intrinsics
    cam_intrinsics = depth_results["intrinsics"].squeeze().cpu().numpy()

    fx = cam_intrinsics[0, 0]
    fy = cam_intrinsics[1, 1]
    cx_cam = cam_intrinsics[0, 2]
    cy_cam = cam_intrinsics[1, 2]

    # Get depth map
    depth = depth_results["depth"].squeeze().cpu().numpy()

    # Get plane normal from plane_coeffs
    plane_normal = None
    if 'plane_coeffs' in plane:
        plane_normal = np.array(plane['plane_coeffs'][0], dtype=np.float64)
        pn_norm = np.linalg.norm(plane_normal)
        if pn_norm > 1e-8:
            plane_normal = plane_normal / pn_norm

    if plane_normal is None:
        print("    No plane normal found, skipping warping.")
        dbg_str = "no plane normal found"
        return warped_rois, dbg_str

    # Get plane mask
    plane_mask = plane.get('segmentation', None)
    # plane_seg_mask = plane.get('plane_mask', None) # NOTE: kept it for experimentation, uncomment to use
    # if plane_seg_mask is not None:
    #     plane_mask = np.logical_or(plane_mask, plane_seg_mask)

    assert plane_mask is not None
    plane_mask = (plane_mask.astype(np.uint8))


    # Classify lines into horizontal and vertical
    h_th = 0.07 * W
    v_th = 0.07 * H

    lines = sort_lines_by_length(lines)

    # Sort lines into horizontal and vertical 
    h_lines = []
    v_lines = []
    for line in lines:
        angle = get_angle_for_line(line)
        angle = angle % 360 ## convert angles to be strictly positive
        line_len = np.linalg.norm(np.array(line[1]) - np.array(line[0]))
        if (angle >= 330 and angle <= 30) or (angle >= 150 and angle <= 210):
            if (line_len >= h_th):
                h_lines.append(line)
        elif (angle >= 60 and angle <= 120) or (angle >= 240 and angle <= 300):
            if (line_len >= v_th):
                v_lines.append(line)

    # Filter lines by plane mask (70% of line should lie on plane)
    if plane_mask is not None:
        plane_coeffs = plane["plane_coeffs"]
        plane_h_lines = [ l for l in h_lines if
                         line_on_mask_fraction_v1(plane_coeffs, l, plane_mask, pts_3d_img) >= 0.7 ]
        plane_v_lines = [ l for l in v_lines if
                         line_on_mask_fraction_v1(plane_coeffs, l, plane_mask, pts_3d_img) >= 0.7 ]

    if len(plane_h_lines) == 0 and len(plane_v_lines) == 0:
        print("    No lines found on plane mask.")
        dbg_str = "No lines found"
        return warped_rois, dbg_str

    # line correction
    plane_h_lines = [
        line_correction(
            img, l, "Horizontal", plane_mask, depth, plane_normal, pts_3d_img, idx
        )
        for idx, l in enumerate(plane_h_lines)
    ]
    plane_v_lines = [
        line_correction(
            img, l, "Vertical", plane_mask, depth, plane_normal, pts_3d_img, idx
        )
        for idx, l in enumerate(plane_v_lines)
    ]

    # Find dominant horizontal vector
    if len(plane_h_lines) > 5:
        h_res = find_dominant_vector_ransac(
            plane_h_lines, plane_normal, label="horizontal",
            points=pts_3d_img, H=H, W=W,
        )
    else:
        h_res = find_dominant_vector(
            plane_h_lines, plane_normal, label="Horizontal",
            points=pts_3d_img, H=H, W=W,
        )

    if h_res is not None:
        h_vector, h_inlier_lines, h_line = h_res
    else:
        h_vector, h_inlier_lines, h_line = None, None, None

    # Find dominant vertical vector
    if len(plane_v_lines) > 5:
        v_res = find_dominant_vector_ransac(
            plane_v_lines, plane_normal, label="vertical",
            points=pts_3d_img, H=H, W=W,
        )
    else:
        v_res = find_dominant_vector(
            plane_v_lines, plane_normal, label="Vertical",
            points=pts_3d_img, H=H, W=W,
        )

    if v_res is not None:
        v_vector, v_inlier_lines, v_line = v_res
    else:
        v_vector, v_inlier_lines, v_line = None, None, None

    # Cross-check perpendicularity and fallback
    if h_vector is not None and v_vector is not None:
        dot = np.clip(np.dot(h_vector, v_vector), -1.0, 1.0)
        angle = np.degrees(np.arccos(np.abs(dot)))
        if abs(angle - 90) > 10:
            h_line_length = np.linalg.norm(np.array(h_line[1]) - np.array(h_line[0])) if h_line is not None else 0
            v_line_length = np.linalg.norm(np.array(v_line[1]) - np.array(v_line[0])) if v_line is not None else 0
            if h_line_length >= v_line_length:
                v_vector = None
            else:
                h_vector = None

    if h_vector is not None and v_vector is None:
        if plane_normal is not None:
            v_vector = np.cross(plane_normal, h_vector)
            v_norm = np.linalg.norm(v_vector)
            if v_norm > 1e-8:
                v_vector = v_vector / v_norm
            else:
                v_vector = None

    if h_vector is None and v_vector is not None:
        if plane_normal is not None:
            h_vector = np.cross(v_vector, plane_normal)
            h_norm = np.linalg.norm(h_vector)
            if h_norm > 1e-8:
                h_vector = h_vector / h_norm
            else:
                h_vector = None

    ad_rois = plane.get('ad_rois', None)
    assert ad_rois is not None

    ad_rois = np.array(ad_rois).reshape(-1, 4, 2).tolist()  # reshape to (N, 4, 2)

    for ad_roi in ad_rois:

        tl_x, tl_y = ad_roi[0]
        br_x, br_y = ad_roi[2]

        sq_w_px = br_x - tl_x
        sq_h_px = br_y - tl_y
        cx_px = int((tl_x + br_x) / 2)
        cy_px = int((tl_y + br_y) / 2)

        cx_px = np.clip(cx_px, 0, W - 1)
        cy_px = np.clip(cy_px, 0, H - 1)

        # Get depth at center
        center_depth = depth[cy_px, cx_px]

        # If depth is noisy, take mean of 5x5 region around center
        if isinstance(center_depth, np.ndarray):
            center_depth = float(center_depth.mean())
        if center_depth == 0:
            y_s, y_e = max(0, cy_px - 5), min(H, cy_px + 5)
            x_s, x_e = max(0, cx_px - 5), min(W, cx_px + 5)
            region_depth = depth[y_s:y_e, x_s:x_e]
            nz = region_depth[region_depth > 0]
            center_depth = float(np.mean(nz)) if len(nz) > 0 else 0.0

        # Project 2D dimensions to 3D
        sq_w_3d = project_length_to_3d(sq_w_px, center_depth, cam_intrinsics)
        sq_h_3d = project_length_to_3d(sq_h_px, center_depth, cam_intrinsics)

        # Get 3D center point
        center_3d = pts_3d_img[cy_px, cx_px] if pts_3d_img is not None else None

        # NOTE: REMOVE IF YOU DONT WANT TO FALLBACK TO REFERENCE VECTORS BASED ON PLANE NORMAL
        if h_vector is None or v_vector is None:
            print("    No dominant vector found, falling back to reference vectors based on plane normal")
            h_vector, v_vector = get_hori_vert_reference_vectors(plane_normal)

        should_warp = True
        if h_vector is None or v_vector is None or center_3d is None:
            should_warp = False
        elif sq_w_3d is None or sq_h_3d is None or sq_w_3d == 0 or sq_h_3d == 0:
            should_warp = False
        elif np.linalg.norm(center_3d) < 1e-8:
            should_warp = False

        if should_warp:
            h_ref, v_ref = get_hori_vert_reference_vectors(plane_normal)
            h_dot = np.dot(h_vector, h_ref)
            v_dot = np.dot(v_vector, v_ref)
            if h_dot < 0:
                h_vector = -h_vector
            if v_dot < 0:
                v_vector = -v_vector
            pts_2d = _project_bbox_3d_to_2d(
                center_3d, v_vector, h_vector, fx, fy, cx_cam, cy_cam,
                height=sq_h_3d, width=sq_w_3d,
            )
            warped_rois.append(pts_2d)
        else:
            warped_rois.append(ad_roi)

    dbg_str = "warping roi successful!"
    warped_rois = np.array(warped_rois).reshape(-1, 2).tolist()  # reshape back to original format

    #return warped_rois, dbg_str
    return warped_rois, dbg_str, h_lines, v_lines



def warp_bbox_using_depth_v1(img, depth_results, pts_3d_img, plane):
    """
    Warp bounding boxes using depth-based 3D projection.
    Adds 'warp_ad_roi' key to each APO in plane['APO_list'].
    Returns the plane dict.
    """
    dbg_str = ""
    warped_rois = []
    if plane["X_axis"] is None or plane["Y_axis"] is None:
        return warped_rois, "Unable to determing X_axis or Y_axis"

    H, W = img.shape[:2]


    # Get depth map
    depth = depth_results["depth"].squeeze().cpu().numpy()
    cam_K = depth_results["intrinsics"].cpu().numpy()[0]
    X_axis = plane["X_axis"]
    Y_axis = plane["Y_axis"]

    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    cx_cam = cam_K[0, 2]
    cy_cam = cam_K[1, 2]

    plane_mask = plane.get('segmentation', None)
    assert plane_mask is not None

    ad_rois = plane.get('ad_rois', None)
    assert ad_rois is not None

    ad_rois = np.array(ad_rois).reshape(-1, 4, 2).tolist()  # reshape to (N, 4, 2)

    def get_3d_box_dims_at_Z(w_2d, h_2d, Z, cam_K):
        if Z == 0:
            return 0
        fx = cam_K[0, 0]
        fy = cam_K[1, 1]
        W_3d = (w_2d * Z) / fx
        H_3d = (h_2d * Z) / fy
        return W_3d, H_3d

    for ad_roi in ad_rois:

        tl_x, tl_y = ad_roi[0]
        br_x, br_y = ad_roi[2]

        w_2d = br_x - tl_x
        h_2d = br_y - tl_y

        cx_2d = int((tl_x + br_x) / 2)
        cy_2d = int((tl_y + br_y) / 2)

        # Get depth at center
        c_Z = depth[cy_2d, cx_2d]

        # depth is usually noisy, take mean of 5x5 region around center
        y_s, y_e = max(0, cy_2d - 5), min(H, cy_2d + 5)
        x_s, x_e = max(0, cx_2d - 5), min(W, cx_2d + 5)
        region_depth = depth[y_s:y_e, x_s:x_e]
        nz = region_depth[region_depth > 0]
        c_Z = np.mean(nz) if len(nz) > 0 else 0.0

        W_3d, H_3d = get_3d_box_dims_at_Z(w_2d, h_2d, c_Z, cam_K)
        # Project 2D dimensions to 3D
        #sq_w_3d = project_length_to_3d(sq_w_px, center_depth, cam_K)
        #sq_h_3d = project_length_to_3d(sq_h_px, center_depth, cam_K)

        # Get 3D center point
        center_3d = pts_3d_img[cy_2d, cx_2d] if pts_3d_img is not None else None

        should_warp = c_Z > 0
        if should_warp:
            pts_2d = _project_bbox_3d_to_2d(center_3d, Y_axis, X_axis, fx, fy, cx_cam, cy_cam, height=H_3d, width=W_3d)
            warped_rois.append(pts_2d)
        else:
            warped_rois.append(ad_roi)

    dbg_str = "warping roi successful!"
    warped_rois = np.array(warped_rois).reshape(-1, 2).tolist()  # reshape back to original format

    #return warped_rois, dbg_str
    return warped_rois, dbg_str

def get_surface_normal(plane, pts_3d_img):
    mask = plane["segmentation"].astype(bool)
    img_H, img_W = mask.shape[:2]
    xs, ys = np.meshgrid(np.arange(0, img_W, 16), np.arange(0, img_H, 16))
    grid_pts = np.stack([xs, ys], axis=-1).reshape(-1, 2)  # (h*w,2)
    pts_mask = mask[grid_pts[:, 1], grid_pts[:, 0]] > 0
    pts = grid_pts[pts_mask]
    assert len(pts) > 10
    pts_3d = pts_3d_img[pts[:, 1], pts[:, 0]]
    centroid, plane_normal = trimesh.points.plane_fit(pts_3d)
    return plane_normal
    






def update_X_and_Y_axes(img, depth_results, pts_3d_img, planes, all_lines):
    """
    Gets the Y axis from all vertical 2d lines
    The vertical lines will usually be same for all planes in the scene
    """
    img_H, img_W = img.shape[:2]
    #all_lines = sort_lines_by_length(all_lines)

    v_lines = []
    v_dirs = []
    v_th = 0.1 * img_H

    def fit_3d_line_from_pts(pts_3d):
        centroid = np.mean(pts_3d, axis=0)
        pts = pts_3d - centroid
        # Use SVD to find the direction vector of the line
        _, _, vh = np.linalg.svd(pts)
        line_dir = vh[0]  # First principal component 
        return line_dir

    def populate_axes(planes, Y_axis):
        """
        Gets the X axis for each plane
        The vertical lines will usually be same for all planes in the scene
        The X axis is found by cross product of plane normal and Y axis
        """
        for plane_idx, plane in enumerate(planes):
            surface_normal = plane["surface_normal"] 
            # Should we care about direction?
            #X_axis = np.cross(surface_normal, Y_axis)
            assert Y_axis is not None
            assert surface_normal is not None
            print(f"Y_axis:{Y_axis} sn:{surface_normal}")
            X_axis = np.cross(surface_normal, Y_axis) # NOTE: use get_hori_vert_reference_vectors if outputs are inconsistent
            plane["X_axis"] = X_axis
            plane["Y_axis"] = Y_axis
        return

    ##########################################################################
    # Compute Y axis
    ##########################################################################
    for plane_idx, plane in enumerate(planes):
        plane_mask = plane['segmentation'].astype(bool)
        plane_lines = plane["plane_lines"] 
        plane["X_axis"] = None
        plane["Y_axis"] = None
        for line in plane_lines:
            angle = get_angle_for_line(line)
            angle = angle % 360 ## convert angles to be strictly positive
            line_len = np.linalg.norm(np.array(line[1]) - np.array(line[0]))
            if (angle >= 60 and angle <= 120) or (angle >= 240 and angle <= 300):
                if (line_len >= v_th):
                    ##########################################################
                    # Get the corresponding 3d line and check if this line
                    # vector is normal to the plane normal
                    line_mask = np.zeros(plane_mask.shape, dtype=np.uint8)
                    pt1 = tuple(map(int, line[0]))
                    pt2 = tuple(map(int, line[1]))
                    cv2.line(line_mask, pt1, pt2, color=1, thickness=1)
                    line_pts = np.column_stack(np.where(line_mask > 0))
                    line_pts_3d = pts_3d_img[line_pts[:,0], line_pts[:,1]]
                    line_dir = fit_3d_line_from_pts(line_pts_3d)
                    if line_dir[1] < 0 :
                        line_dir = -line_dir
                    ### knr:TODO make sure that plane_normal is sane
                    plane_normal = plane['plane_coeffs'][0]

                    dot = np.clip(np.dot(line_dir, plane_normal), -1.0, 1.0)
                    angle = np.degrees(np.arccos(np.abs(dot)))
                    if abs(angle - 90) < 10:
                        v_lines.append(line)
                        v_dirs.append(line_dir)

    if len(v_lines) > 0: 
        mean_vector = np.mean(np.array(v_dirs), axis=0)
        Y_axis = mean_vector/np.linalg.norm(mean_vector)
        populate_axes(planes, Y_axis)
    else:
        plane["X_axis"] = None
        plane["Y_axis"] = None



    if 0:
        # project two points (line end point) at starting at (1, 1, 2)

        cam_K = depth_results["intrinsics"].cpu().numpy()[0]

        pt0 = np.array([-0.8,-1,3], dtype=np.float32)
        pt1 = pt0 + 3*Y_axis
        img = project_and_draw_3d_point_onto_img(img, pt0, pt1, cam_K)

        pt0 = np.array([0,-1,3], dtype=np.float32)
        pt1 = pt0 + 3*Y_axis
        img = project_and_draw_3d_point_onto_img(img, pt0, pt1, cam_K)

        pt0 = np.array([0.8,-1,3], dtype=np.float32)
        pt1 = pt0 + 3*Y_axis
        img = project_and_draw_3d_point_onto_img(img, pt0, pt1, cam_K)


        plt.imshow(img[:,:,::-1])
        plt.show()
    return 




def project_and_draw_3d_point_onto_img(img, pt0, pt1, cam_K):
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))

    pts = np.stack([pt0, pt1])
    pts_2d, _ = cv2.projectPoints(pts, rvec, tvec, cam_K, distCoeffs=None)

    pt0 = tuple(pts_2d[0].astype(int).reshape(-1))
    pt1 = tuple((pts_2d[1]).astype(int).reshape(-1))
    img = cv2.line(img, pt0, pt1, (0,0,255), 1)

    return img
