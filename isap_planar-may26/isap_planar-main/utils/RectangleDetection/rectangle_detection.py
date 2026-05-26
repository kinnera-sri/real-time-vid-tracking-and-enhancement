"""
Rectangle/Square detection using LCNN (Line CNN).
Detects well-formed rectangular or square masks using line detection.
"""

import os
import os.path as osp
import numpy as np
import cv2
import torch
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
import skimage.transform

# LCNN imports are done lazily inside setup_lcnn_model / filter_masks_with_lcnn
# to support two different sys.path conventions:
#   - "models/lcnn" in path  →  from lcnn.config import C, M
#   - "models" in path       →  from lcnn.lcnn.config import C, M


def _lcnn_imports():
    """Return (C, M) using whichever lcnn import path is available.

    Supports two sys.path conventions:
      - 'models/lcnn' in path  ->  from lcnn.config import C, M
      - 'models' in path       ->  from lcnn.lcnn.config import C, M
    """
    try:
        from lcnn.config import C, M
    except (ImportError, AttributeError):
        from lcnn.lcnn.config import C, M
    return C, M


def pline(x1, y1, x2, y2, x, y):
    """Point to line distance calculation for postprocessing"""
    px = x2 - x1
    py = y2 - y1
    dd = px * px + py * py
    u = ((x - x1) * px + (y - y1) * py) / max(1e-9, float(dd))
    dx = x1 + u * px - x
    dy = y1 + u * py - y
    return dx * dx + dy * dy


def psegment(x1, y1, x2, y2, x, y):
    """Point to segment distance calculation for postprocessing"""
    px = x2 - x1
    py = y2 - y1
    dd = px * px + py * py
    u = max(min(((x - x1) * px + (y - y1) * py) / float(dd), 1), 0)
    dx = x1 + u * px - x
    dy = y1 + u * py - y
    return dx * dx + dy * dy


def plambda(x1, y1, x2, y2, x, y):
    """Lambda calculation for postprocessing"""
    px = x2 - x1
    py = y2 - y1
    dd = px * px + py * py
    return ((x - x1) * px + (y - y1) * py) / max(1e-9, float(dd))


def postprocess_lines(lines, scores, threshold=0.01, tol=1e9, do_clip=False):
    # taken from from LCNN demo.py
    nlines, nscores = [], []
    for (p, q), score in zip(lines, scores):
        start, end = 0, 1
        for a, b in nlines:
            if (
                min(
                    max(pline(*p, *q, *a), pline(*p, *q, *b)),
                    max(pline(*a, *b, *p), pline(*a, *b, *q)),
                )
                > threshold ** 2
            ):
                continue
            lambda_a = plambda(*p, *q, *a)
            lambda_b = plambda(*p, *q, *b)
            if lambda_a > lambda_b:
                lambda_a, lambda_b = lambda_b, lambda_a
            lambda_a -= tol
            lambda_b += tol

            # case 1: skip (if not do_clip)
            if start < lambda_a and lambda_b < end:
                continue

            # not intersect
            if lambda_b < start or lambda_a > end:
                continue

            # cover
            if lambda_a <= start and end <= lambda_b:
                start = 10
                break

            # case 2 & 3:
            if lambda_a <= start and start <= lambda_b:
                start = lambda_b
            if lambda_a <= end and end <= lambda_b:
                end = lambda_a

            if start >= end:
                break

        if start >= end:
            continue
        nlines.append(np.array([p + (q - p) * start, p + (q - p) * end]))
        nscores.append(score)
    return np.array(nlines), np.array(nscores)


def setup_lcnn_model(config_file="../lcnn/config/wireframe.yaml", 
                     checkpoint_file="../lcnn/190418-201834-f8934c6-lr4d10-312k.pth"):
    """
    Initialize LCNN model for line detection.
    
    Args:
        config_file: Path to LCNN config yaml
        checkpoint_file: Path to LCNN checkpoint
        
    Returns:
        Loaded model on GPU/CPU, device
    """
    C, M = _lcnn_imports()
    try:
        import lcnn as _lcnn_pkg
        from lcnn.models.line_vectorizer import LineVectorizer
        from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
    except (ImportError, AttributeError):
        import lcnn as _lcnn_pkg
        from lcnn.lcnn.models.line_vectorizer import LineVectorizer
        from lcnn.lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner

    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Load model
    model = _lcnn_pkg.models.hg(
        depth=M.depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=sum(sum(M.head_size, [])),
    )
    model = MultitaskLearner(model)
    model = LineVectorizer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"LCNN model loaded on {device_name}")
    return model, device


def calculate_line_angle(line):
    if len(line) == 2:
        p1, p2 = line[0], line[1]
    else:
        x_p1, y_p1, x_p2, y_p2 = line
        p1 = np.array([y_p1, x_p1])
        p2 = np.array([y_p2, x_p2])

    dy = p2[0] - p1[0]
    dx = p2[1] - p1[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    angle_deg = abs(angle_deg)
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    return angle_deg


def check_parallelism(left_line, right_line, up_line, down_line, angle_tolerance=5.0):

    angle_tolerance=5.0

    left_angle = calculate_line_angle(left_line)
    right_angle = calculate_line_angle(right_line)
    up_angle = calculate_line_angle(up_line)
    down_angle = calculate_line_angle(down_line)
    

    vertical_angle_diff = abs(left_angle - right_angle)
    
    horizontal_angle_diff = abs(up_angle - down_angle)
    
    is_parallel = (vertical_angle_diff <= angle_tolerance and 
                   horizontal_angle_diff <= angle_tolerance)

    return is_parallel, vertical_angle_diff, horizontal_angle_diff


def corners_match(left, right, up, down, threshold, angle_tolerance=5.0):
    # checking if 4 lines form rect and the opps are parellel
    left_line, left_score, left_start, left_end = left
    right_line, right_score, right_start, right_end = right
    up_line, up_score, up_start, up_end = up
    down_line, down_score, down_start, down_end = down

    distances = [
        np.linalg.norm(left_end - down_start),
        np.linalg.norm(down_end - right_end),
        np.linalg.norm(right_start - up_end),
        np.linalg.norm(up_start - left_start),
    ]
    
    corners_valid = all(d < threshold for d in distances)
    
    if not corners_valid:
        return False, distances, None, None, None
    
    #  vertical lines and horizontal lines parallel to each other
    is_parallel, vert_diff, horiz_diff = check_parallelism(
        left_line, right_line, up_line, down_line, angle_tolerance
    )
    
    return corners_valid and is_parallel, distances, vert_diff, horiz_diff, is_parallel


def find_rectangle(working_left_lines,
                   working_right_lines,
                   working_up_lines,
                   working_down_lines,
                   corner_distance_threshold,
                   iteration,
                   angle_tolerance=5.0):
    


    for (left_idx, left), (right_idx, right), (up_idx, up), (down_idx, down) in product(
        enumerate(working_left_lines),
        enumerate(working_right_lines),
        enumerate(working_up_lines),
        enumerate(working_down_lines)
    ):

        valid, distances, vert_diff, horiz_diff, is_parallel = corners_match(
            left, right, up, down, corner_distance_threshold, angle_tolerance
        )

        if valid:
            left_line, left_score, *_ = left
            right_line, right_score, *_ = right
            up_line, up_score, *_ = up
            down_line, down_score, *_ = down

            total_score = left_score + right_score + up_score + down_score


            return {
                'rectangle_found': {
                    'left': left_line,
                    'right': right_line,
                    'up': up_line,
                    'down': down_line,
                    'scores': {
                        'left': left_score,
                        'right': right_score,
                        'up': up_score,
                        'down': down_score
                    },
                    'corner_distances': distances,
                    'parallelism': {
                        'vertical_angle_diff': vert_diff,
                        'horizontal_angle_diff': horiz_diff,
                        'is_parallel': is_parallel
                    },
                    'total_score': total_score,
                    'iteration': iteration
                },
                'lines_to_remove': {
                    'left': left_idx,
                    'right': right_idx,
                    'up': up_idx,
                    'down': down_idx
                }
            }

    print(f"no rec found")
    return None




def detect_lines_at_mask_corners(lines, scores, mask_dict, score_threshold=0.85, dilation=15, dsine_image=None):

    bbox = mask_dict['bbox']
    mask = mask_dict['segmentation'].astype(np.uint8)
    
    x, y, w, h = bbox
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation*2+1, dilation*2+1))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    
    dilated_coords = np.argwhere(dilated_mask > 0)
    if len(dilated_coords) == 0:
        return False, {}, {}
    
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    
    # Filter lines by score
    filtered_indices = np.where(scores >= score_threshold)[0]
    
    if len(filtered_indices) == 0:
        return False, {}, {}
    
    # Collect ALL candidate lines per side (not just one)
    left_lines = []    # List of (line, score, start_point, end_point)
    right_lines = []
    up_lines = []
    down_lines = []
    
    for idx in filtered_indices:
        line = lines[idx]

        score = scores[idx]
        
        p1, p2 = line[0], line[1]
        y_p1, x_p1 = p1[0], p1[1]
        y_p2, x_p2 = p2[0], p2[1]
        
        # Check if BOTH endpoints are in dilated mask because if there's even samll noice line might be start at the mask and end at some other place and it'll be bad
        p1_in_mask = dilated_mask[int(y_p1), int(x_p1)] > 0 if (0 <= int(y_p1) < dilated_mask.shape[0] and 0 <= int(x_p1) < dilated_mask.shape[1]) else False
        p2_in_mask = dilated_mask[int(y_p2), int(x_p2)] > 0 if (0 <= int(y_p2) < dilated_mask.shape[0] and 0 <= int(x_p2) < dilated_mask.shape[1]) else False
        
        if not (p1_in_mask and p2_in_mask):
            continue
        
        angle_deg = calculate_line_angle(line)


        # Calculate midpoint
        mid_x = (x_p1 + x_p2) / 2
        mid_y = (y_p1 + y_p2) / 2
 
        # Vertical line: angle close to 90 degrees (after normalization to [0,90]: 60-90)
        if 50 <= angle_deg <= 90:  
            if y_p1 < y_p2:
                start_pt = np.array([y_p1, x_p1])
                end_pt = np.array([y_p2, x_p2])
            else:
                start_pt = np.array([y_p2, x_p2])
                end_pt = np.array([y_p1, x_p1])
            
            # Classify as left or right based on midpoint
            if mid_x < x_center and x_p1 < x_center and x_p2 < x_center:
                left_lines.append((line, score, start_pt, end_pt))
            elif mid_x > x_center and x_p1 > x_center and x_p2 > x_center:
                right_lines.append((line, score, start_pt, end_pt))
            else:
                pass
                
        else:  # Horizontal line: angle close to 0 degrees 
            # Apply angle filter only if wall is facing camera
            # if is_wall_facing_camera:
            #     # For front-facing walls, use strict angle threshold
            #     if not (0 <= angle_deg <= 40):
            #         continue

            if x_p1 < x_p2:
                start_pt = np.array([y_p1, x_p1])
                end_pt = np.array([y_p2, x_p2])
            else:
                start_pt = np.array([y_p2, x_p2])
                end_pt = np.array([y_p1, x_p1])
            
            # up or down based on midpoint
            if mid_y < y_center and y_p1 < y_center and y_p2 < y_center:
                up_lines.append((line, score, start_pt, end_pt))
            elif mid_y > y_center and y_p1 > y_center and y_p2 > y_center:
                down_lines.append((line, score, start_pt, end_pt))

    print("***"*10)

    # min 1 line needed 
    if len(left_lines) == 0 or len(right_lines) == 0 or len(up_lines) == 0 or len(down_lines) == 0:
        return False, {}, {}
    
    found_rectangles = []
    corner_distance_threshold = 7  

    # Create working copies of line lists that we can modify
    working_left_lines = left_lines.copy()
    working_right_lines = right_lines.copy()
    working_up_lines = up_lines.copy()
    working_down_lines = down_lines.copy()
    
    # Keep searching for rectangles until no more can be formed. because of photo frames we are doin iterations 
    iteration = 0
    while (len(working_left_lines) > 0 and len(working_right_lines) > 0 and 
           len(working_up_lines) > 0 and len(working_down_lines) > 0):
        
        iteration += 1

        result = find_rectangle(
            working_left_lines,
            working_right_lines, 
            working_up_lines,
            working_down_lines,
            corner_distance_threshold,
            iteration,
            angle_tolerance=8.0
        )
        
        # if a set of lines formed rectangle remove them : 
        #  TODO: need to improve it as some times thr can be better lines
        if result is not None:
            rectangle_found = result['rectangle_found']
            lines_to_remove = result['lines_to_remove']
            
            found_rectangles.append(rectangle_found)

            indices_to_remove = sorted([
                ('down', lines_to_remove['down'], working_down_lines),
                ('up', lines_to_remove['up'], working_up_lines),
                ('right', lines_to_remove['right'], working_right_lines),
                ('left', lines_to_remove['left'], working_left_lines)
            ], key=lambda x: x[1], reverse=True)
            
            for side_name, idx, line_list in indices_to_remove:
                if 0 <= idx < len(line_list):
                    removed_line = line_list.pop(idx)
            
        else:

            print('no rec. search done')
            break
    

    if found_rectangles:
        print(f"Found {len(found_rectangles)} total rectangles in this mask")
        
        found_rectangles.sort(key=lambda x: x['total_score'], reverse=True)
        
        all_side_lines = []
        all_side_scores = []
        
        for i, rectangle in enumerate(found_rectangles):
            
            side_lines = {
                'left': rectangle['left'],
                'right': rectangle['right'],
                'up': rectangle['up'],
                'down': rectangle['down']
            }
            side_scores = rectangle['scores']
            
            all_side_lines.append(side_lines)
            all_side_scores.append(side_scores)
        
        return True, all_side_lines, all_side_scores
    else:
        print("No valid rectangles found")
        return False, [], []

def refine_corners_by_detection(side_lines, image, patch_radius=15):

    H, W = image.shape[:2]

    left  = side_lines['left']    # [[y_top,  x_top ], [y_bot, x_bot ]]
    right = side_lines['right']   # [[y_top,  x_top ], [y_bot, x_bot ]]
    up    = side_lines['up']      # [[y_left, x_left], [y_rgt, x_rgt ]]
    down  = side_lines['down']    # [[y_left, x_left], [y_rgt, x_rgt ]]

    def _top(ln):  return ln[0] if ln[0][0] <= ln[1][0] else ln[1]
    def _bot(ln):  return ln[0] if ln[0][0] >  ln[1][0] else ln[1]
    def _lft(ln):  return ln[0] if ln[0][1] <= ln[1][1] else ln[1]
    def _rgt(ln):  return ln[0] if ln[0][1] >  ln[1][1] else ln[1]

    coarse_corners = {
        'TL': (_top(left)  + _lft(up))   / 2.0,   # top of left  ↔ left  of up
        'TR': (_rgt(up)    + _top(right)) / 2.0,   # right of up  ↔ top   of right
        'BR': (_bot(right) + _rgt(down))  / 2.0,   # bot of right ↔ right of down
        'BL': (_lft(down)  + _bot(left))  / 2.0,   # left of down ↔ bot   of left
    }

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()

    if gray.dtype != np.uint8:
        gray = np.clip(gray * 255, 0, 255).astype(np.uint8)

    refined_corners = {}
    found_corners_all = []   # Collect all Shi-Tomasi corners from all 4 patches
    candidates_per_corner = {}  # name -> list of np.array([y, x]) full-image candidates


    if 0:
        _LF_CROP_R   = patch_radius    # ±px around each coarse corner
        _LF_TOP_N    = 2     # how many best rectangles to show
        _LF_ANG_TOL  = 4.0   # ° parallelism of opposite sides
        _LF_PERP_TOL = 7.0  # ° perpendicularity of adjacent sides
        _LF_LEN_TOL  = 0.25  # fraction: opposite side length ratio

        def _lf_qfilt(cands_yx, cy, cx, label, tol=5):
            """Keep (y,x) candidates in the geometrically correct quadrant."""
            def ok(ry, rx):
                dr, dc = ry - cy, rx - cx
                if label == 'TL': return dc <= tol  and dr <= tol
                if label == 'TR': return dc >= -tol and dr <= tol
                if label == 'BR': return dc >= -tol and dr >= -tol
                if label == 'BL': return dc <= tol  and dr >= -tol
                return True
            f = [(ry, rx) for (ry, rx) in cands_yx if ok(ry, rx)]
            return f if f else cands_yx

        def _lf_edge_angle(p1, p2):
            # p in (y,x)=(row,col): atan2(row_diff, col_diff)
            return float(np.degrees(np.arctan2(p2[0] - p1[0], p2[1] - p1[1])))

        def _lf_adiff(a, b):
            d = abs(a - b) % 180
            return min(d, 180 - d)

        def _lf_score_rect(tl, tr, br, bl):
            segs    = [(tl, tr), (tr, br), (br, bl), (bl, tl)]
            angles  = [_lf_edge_angle(a, b) for a, b in segs]
            lengths = [float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in segs]
            adiff_tb = _lf_adiff(angles[0], angles[2])
            adiff_rl = _lf_adiff(angles[1], angles[3])
            perp_01  = abs(_lf_adiff(angles[0], angles[1]) - 90)
            perp_12  = abs(_lf_adiff(angles[1], angles[2]) - 90)
            eps      = 1e-6
            ldiff_tb = abs(lengths[0] - lengths[2]) / (max(lengths[0], lengths[2]) + eps)
            ldiff_rl = abs(lengths[1] - lengths[3]) / (max(lengths[1], lengths[3]) + eps)
            if adiff_tb > _LF_ANG_TOL  or adiff_rl > _LF_ANG_TOL:  return None
            if perp_01  > _LF_PERP_TOL or perp_12  > _LF_PERP_TOL: return None
            if ldiff_tb > _LF_LEN_TOL  or ldiff_rl > _LF_LEN_TOL:  return None
            return (adiff_tb + adiff_rl + perp_01 + perp_12
                    + 100.0 * (ldiff_tb + ldiff_rl))

        _lf_gray = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    if len(image.shape) == 3 else image.copy())
        if _lf_gray.dtype != np.uint8:
            _lf_gray = np.clip(_lf_gray * 255, 0, 255).astype(np.uint8)

        _lf_cands    = {lbl: [] for lbl in ['TL', 'TR', 'BR', 'BL']}
        _lf_crop_vis = {}   # label -> BGR crop for row-0 grid

        for _lbl in ['TL', 'TR', 'BR', 'BL']:
            _cy  = float(coarse_corners[_lbl][0])
            _cx  = float(coarse_corners[_lbl][1])
            _py0 = max(0, int(round(_cy)) - _LF_CROP_R)
            _py1 = min(H, int(round(_cy)) + _LF_CROP_R + 1)
            _px0 = max(0, int(round(_cx)) - _LF_CROP_R)
            _px1 = min(W, int(round(_cx)) + _LF_CROP_R + 1)

            _crop_gray = _lf_gray[_py0:_py1, _px0:_px1]

            _cf = cv2.goodFeaturesToTrack(
                _crop_gray, maxCorners=30, qualityLevel=0.05,
                minDistance=2, blockSize=patch_radius//2)

            _raw_yx = []
            if _cf is not None:
                for _pt in _cf.reshape(-1, 2):
                    _col_f, _row_f = float(_pt[0]), float(_pt[1])
                    _ry = float(np.clip(_py0 + _row_f, 0, H - 1))
                    _rx = float(np.clip(_px0 + _col_f, 0, W - 1))
                    _raw_yx.append((_ry, _rx))

            _raw_yx.append((_cy, _cx))   # fallback: coarse corner
            _filtered = _lf_qfilt(_raw_yx, _cy, _cx, _lbl, tol=5)
            _lf_cands[_lbl] = _filtered
            print(f"  [lf] [{_lbl}] raw={len(_raw_yx)-1}  filtered={len(_filtered)}")

            
        # ── exhaustive combo search ───────────────────────────────────────────
        _lf_total = (len(_lf_cands['TL']) * len(_lf_cands['TR']) *
                     len(_lf_cands['BR']) * len(_lf_cands['BL']))
        print(f"  [lf] Searching {_lf_total} combos …")

        _lf_best = []
        for _tl in _lf_cands['TL']:
            for _tr in _lf_cands['TR']:
                for _br in _lf_cands['BR']:
                    for _bl in _lf_cands['BL']:
                        _sc = _lf_score_rect(_tl, _tr, _br, _bl)
                        if _sc is not None:
                            _lf_best.append((_sc, _tl, _tr, _br, _bl))
        _lf_best.sort(key=lambda x: x[0])
        _lf_best = _lf_best[:_LF_TOP_N]
        print(f"  [lf] {len(_lf_best)} valid rectangle(s) found")

        # ── override refined_corners with best rectangle ──────────────────────
        if _lf_best:
            _, _tl, _tr, _br, _bl = _lf_best[0]
            refined_corners['TL'] = np.asarray(_tl, dtype=float)
            refined_corners['TR'] = np.asarray(_tr, dtype=float)
            refined_corners['BR'] = np.asarray(_br, dtype=float)
            refined_corners['BL'] = np.asarray(_bl, dtype=float)
            print("  [lf] refined_corners set from rank-1 rectangle")
        else:
            print("  [lf] no valid rectangle – falling back to coarse corners")
            for _l in ['TL', 'TR', 'BR', 'BL']:
                refined_corners[_l] = np.asarray(coarse_corners[_l], dtype=float)

        # Populate found_corners_all and candidates_per_corner
        for _l in ['TL', 'TR', 'BR', 'BL']:
            found_corners_all.extend(
                [(float(c[0]), float(c[1])) for c in _lf_cands[_l]])
            candidates_per_corner[_l] = [
                np.asarray(c, dtype=float) for c in _lf_cands[_l]]

    if 1:
        for name, corner in coarse_corners.items():
            if name in refined_corners:   # already populated by line_fitter block above
                continue
            cy, cx = float(corner[0]), float(corner[1])

            # Clamp patch to image bounds
            py0 = max(0, int(round(cy)) - patch_radius)
            py1 = min(H, int(round(cy)) + patch_radius + 1)
            px0 = max(0, int(round(cx)) - patch_radius)
            px1 = min(W, int(round(cx)) + patch_radius + 1)
            patch = gray[py0:py1, px0:px1]

            if patch.size == 0:
                refined_corners[name] = np.array([cy, cx])
                candidates_per_corner[name] = [np.array([cy, cx])]
                continue

            # Shi-Tomasi: up to 5 candidates, pick the one closest to patch centre
            # corners_found = cv2.goodFeaturesToTrack(
            #     patch, maxCorners=5, qualityLevel=0.01, minDistance=3
            # )
            if 0:
                plt.imshow(patch, cmap='gray')
                plt.title(f"Patch for corner {patch.shape} ")
                plt.axis('off')
                plt.show()
            corners_found = cv2.goodFeaturesToTrack(
                    patch,
                    maxCorners=3,
                    qualityLevel=0.02,
                    minDistance=2,
                    blockSize=3
                )

            # plot all corners found in the patch
            candidates_per_corner[name] = []
            if corners_found is not None and len(corners_found) > 0:

                ph, pw = patch.shape
                patch_cy, patch_cx = ph / 2.0, pw / 2.0

                # Track every found corner in full-image coordinates (y, x)
                for pt in corners_found:
                    col_f, row_f = float(pt[0][0]), float(pt[0][1])
                    found_corners_all.append((float(py0 + row_f), float(px0 + col_f)))
                    # Store as full-image (y, x) for combination search
                    candidates_per_corner[name].append(np.array([
                        float(np.clip(py0 + row_f, 0, H - 1)),
                        float(np.clip(px0 + col_f, 0, W - 1)),
                    ]))

                best_row, best_col, best_dist = None, None, float('inf')
                for pt in corners_found:
                    # goodFeaturesToTrack returns (x, y) == (col, row)
                    col_f, row_f = float(pt[0][0]), float(pt[0][1])
                    dist = np.hypot(row_f - patch_cy, col_f - patch_cx)
                    if dist < best_dist:
                        best_dist = dist
                        best_row, best_col = row_f, col_f

                sp_half = 3   
                # Crop bounds (clamped to patch)
                sr0 = max(0, int(round(best_row)) - sp_half)
                sr1 = min(ph, int(round(best_row)) + sp_half + 1)
                sc0 = max(0, int(round(best_col)) - sp_half)
                sc1 = min(pw, int(round(best_col)) + sp_half + 1)


                # Map back to full-image coordinates
                ry = np.clip(py0 + best_row, 0, H - 1)
                rx = np.clip(px0 + best_col, 0, W - 1)
                refined_corners[name] = np.array([ry, rx])

            else:
                print(f"  Corner {name}: no Shi-Tomasi hit, keeping coarse ({cy:.1f},{cx:.1f})")
                refined_corners[name] = np.array([cy, cx])
                candidates_per_corner[name] = [np.array([cy, cx])]


    TL = refined_corners['TL']
    TR = refined_corners['TR']
    BR = refined_corners['BR']
    BL = refined_corners['BL']

    refined_side_lines = {
        'up':    np.array([TL, TR]),   # start=left-end, end=right-end
        'down':  np.array([BL, BR]),
        'left':  np.array([TL, BL]),   # start=top-end,  end=bottom-end
        'right': np.array([TR, BR]),
    }

    return refined_side_lines, coarse_corners, refined_corners, found_corners_all


def filter_lines_by_person_mask(lines, scores, person_mask):


    filtered_lines = []
    filtered_scores = []
    removed_count = 0
    
    for idx, (line, score) in enumerate(zip(lines, scores)):
        
        p1, p2 = line
        y1, x1 = p1[0], p1[1]
        y2, x2 = p2[0], p2[1]
        
        keep_line = True
        
        # Check start point
        y1_int, x1_int = int(np.round(y1)), int(np.round(x1))
        if 0 <= y1_int < person_mask.shape[0] and 0 <= x1_int < person_mask.shape[1]:
            if person_mask[y1_int, x1_int]:
                keep_line = False
        
        # Check end point
        if keep_line:
            y2_int, x2_int = int(np.round(y2)), int(np.round(x2))
            if 0 <= y2_int < person_mask.shape[0] and 0 <= x2_int < person_mask.shape[1]:
                if person_mask[y2_int, x2_int]:
                    keep_line = False
        
        # Check midpoint
        if keep_line:
            mid_y, mid_x = (y1 + y2) / 2, (x1 + x2) / 2
            mid_y_int, mid_x_int = int(np.round(mid_y)), int(np.round(mid_x))
            if 0 <= mid_y_int < person_mask.shape[0] and 0 <= mid_x_int < person_mask.shape[1]:
                if person_mask[mid_y_int, mid_x_int]:
                    keep_line = False
        
        if keep_line:
            filtered_lines.append(line)
            filtered_scores.append(score)
        else:
            removed_count += 1
        
    
    return np.array(filtered_lines), np.array(filtered_scores)


def filter_masks_with_lcnn(model, device, point_dist_masks, image, person_mask,
                           score_threshold=0.85, apply_postprocess=True,
                           show_postprocess_viz=False, dsine_image=None,
                           save_dir=None):

    # Use lazy imports to resolve the correct M config regardless of sys.path convention
    _, M = _lcnn_imports()

    # taken from demo.py of lcnn 
    im_resized = skimage.transform.resize(image, (512, 512)) * 255
    image_normalized = (im_resized - M.image.mean) / M.image.stddev
    image_tensor = torch.from_numpy(np.rollaxis(image_normalized, 2)[None].copy()).float()
    
    with torch.no_grad():
        input_dict = {
            "image": image_tensor.to(device),
            "meta": [
                {
                    "junc": torch.zeros(1, 2).to(device),
                    "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                    "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                    "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                }
            ],
            "target": {
                "jmap": torch.zeros([1, 1, 128, 128]).to(device),
                "joff": torch.zeros([1, 1, 2, 128, 128]).to(device),
            },
            "mode": "testing",
        }
        H = model(input_dict)["preds"]
    
    all_lines = H["lines"][0].cpu().numpy() / 128 * np.array(image.shape[:2])
    all_scores = H["score"][0].cpu().numpy()
    
    # # Remove duplicate/invalid lines
    for i in range(1, len(all_lines)):
        if (all_lines[i] == all_lines[0]).all():
            all_lines = all_lines[:i]
            all_scores = all_scores[:i]
            break
    

    # Filter out lines that have start/end/midpoint on person mask
    all_lines, all_scores = filter_lines_by_person_mask(all_lines, all_scores, person_mask)
    
    # NOTE: Postprocessing also taken from demo.py but its threhsolding is too strict ig. removing some imp lines so false for now
    if apply_postprocess and len(all_lines) > 0:
        
        original_lines = all_lines.copy()
        original_scores = all_scores.copy()
        
        # Calculate diagonal for threshold
        diag = (image.shape[0] ** 2 + image.shape[1] ** 2) ** 0.5
        all_lines, all_scores = postprocess_lines(all_lines, all_scores, diag * 0.01, 0, False)
        
        print(f"After postprocessing: {len(all_lines)} lines remaining")
        

    lcnn_masks = []
    
    for idx, mask_dict in enumerate(point_dist_masks):
        # check if lines exist on all 4 sides after dialation
        has_all_sides, all_side_lines, all_side_scores = detect_lines_at_mask_corners(
            all_lines, all_scores, mask_dict, score_threshold=score_threshold, dilation=7, dsine_image=dsine_image
        )
        
        if has_all_sides:
            # thr might be multiple rectangles per mask like phot oframes so we're iterating thru them
            for rect_idx, (side_lines, side_scores) in enumerate(zip(all_side_lines, all_side_scores)):

                if True:

                    side_lines, coarse_corners, refined_corners, tracked_corners = refine_corners_by_detection(
                        side_lines, image, patch_radius=3
                    )

                mask_with_lines = mask_dict.copy()
                mask_with_lines['side_lines'] = side_lines  # dict with 'left', 'right', 'up', 'down'
                mask_with_lines['side_scores'] = side_scores
                mask_with_lines['lcnn_all_lines'] = all_lines
                mask_with_lines['lcnn_all_scores'] = all_scores
                mask_with_lines['rectangle_index'] = rect_idx  # Track which rectangle this is from the mask
                mask_with_lines['total_rectangles_in_mask'] = len(all_side_lines)  # Track total rectangles found
                lcnn_masks.append(mask_with_lines)

    return lcnn_masks, all_lines, all_scores

