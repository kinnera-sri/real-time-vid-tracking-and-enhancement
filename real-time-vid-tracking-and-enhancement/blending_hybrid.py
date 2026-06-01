import os
import sys
from typing import Dict, Tuple, Union, Optional

# Ensure OpenCV Qt windows can find system fonts in Linux environments.
for candidate in [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
]:
    if os.path.isdir(candidate):
        os.environ.setdefault("QT_QPA_FONTDIR", candidate)
        break

import cv2
import numpy as np

# ============================================================
# SRGB <-> LINEAR RGB
# ============================================================

def srgb_to_linear(img):

    img = img.astype(np.float32) / 255.0

    return np.power(img, 2.2)


def linear_to_srgb(img):

    img = np.clip(img, 0, 1)

    img = np.power(img, 1.0 / 2.2)

    return (img * 255).astype(np.uint8)

class PlanarARTracker:
    def __init__(self, ad_image_path: str, feature_limit: int = 2000):
        """Initializes the tracker with SIFT, FLANN matching, and the target graphic."""
        self.sift = cv2.SIFT_create(nfeatures=feature_limit) # type: ignore
        
        # Configure FLANN matcher with explicit typing for strict environments
        FLANN_INDEX_KDTREE = 1
        IndexParamType = Dict[str, Union[bool, int, float, str]]
        index_params: IndexParamType = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5} # type: ignore
        search_params: IndexParamType = {"checks": 50} # type: ignore
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Load the advertising/logo graphics asset
        self.ad_img = cv2.imread(ad_image_path, cv2.IMREAD_UNCHANGED)
        if self.ad_img is None:
            raise FileNotFoundError(f"Could not load ad image from '{ad_image_path}'")
            
        # Tracking states
        self.kp0: tuple = ()
        self.des0: Optional[np.ndarray] = None
        self.ad_roi_0: Optional[np.ndarray] = None
        self.ad_roi_prev: Optional[np.ndarray] = None
        
        # Named windows configuration tracking layout
        self.win_roi = "1. Select Target ROI"
        self.win_plane = "2. Select Surface Plane"
        self.win_live = "Live Planar Tracking (Press 'q' to Quit)"
        
        # Temporal smoothing for Retinex illumination (rolling average over 8 frames)
        self.illum_history: list = []
        self.illum_history_max_len = 8

    def _create_warped_ad_and_mask(self, frame_shape: Tuple[int, ...], dst_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates perspective homography mapping and warps the graphic asset and its alpha mask."""
        fh, fw = frame_shape[:2]
        ah, aw = self.ad_img.shape[:2] # type: ignore
        
        src_pts = np.array([[0, 0], [aw - 1, 0], [aw - 1, ah - 1], [0, ah - 1]], dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)
        
        H, _ = cv2.findHomography(src_pts, dst_pts)
        if H is None:
            return np.zeros((fh, fw, 3), dtype=np.uint8), np.zeros((fh, fw), dtype=np.uint8)
        
        # Handle RGBA transparent channel split seamlessly
        if self.ad_img.shape[2] == 4: # type: ignore
            bgr = self.ad_img[:, :, :3] # type: ignore
            alpha = self.ad_img[:, :, 3] # type: ignore
        else:
            bgr = self.ad_img
            alpha = np.ones((ah, aw), dtype=np.uint8) * 255

        warped_ad = cv2.warpPerspective(bgr, H, (fw, fh)) # type: ignore
        warped_mask = cv2.warpPerspective(alpha, H, (fw, fh))
        return warped_ad, warped_mask

    def _apply_multiscale_retinex(self, warped_ad: np.ndarray, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Multi-scale Retinex: blends 3 illumination scales (15, 45, 120) for accurate lighting capture."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Compute illumination at 3 different scales
        illum_15 = cv2.GaussianBlur(gray_frame, (15, 15), 0).astype(np.float32)
        illum_45 = cv2.GaussianBlur(gray_frame, (45, 45), 0).astype(np.float32)
        illum_120 = cv2.GaussianBlur(gray_frame, (121, 121), 0).astype(np.float32)  # Must be odd
        
        # Average the three scales for balanced micro and macro lighting
        illumination = (illum_15 + illum_45 + illum_120) / 3.0
        
        # Safeguard against dividing by zero
        mask_pixels = illumination[mask > 0]
        avg_local_illum = np.mean(mask_pixels) if mask_pixels.size > 0 else 128.0
        if avg_local_illum < 1.0:
            avg_local_illum = 1.0
        
        # Temporal smoothing: rolling average to prevent flicker
        self.illum_history.append(avg_local_illum)
        if len(self.illum_history) > self.illum_history_max_len:
            self.illum_history.pop(0)
        smoothed_illum = np.mean(self.illum_history)
        
        # Create lighting modifier map
        illum_multiplier = illumination / smoothed_illum
        
        # Apply illumination to asset
        ad_retinex = warped_ad.astype(np.float32)
        for c in range(3):
            ad_retinex[:, :, c] *= illum_multiplier
            
        return np.clip(ad_retinex, 0, 255).astype(np.uint8)
    
    def _detect_shadow_regions(self, frame: np.ndarray, threshold_lum: float = 100.0, threshold_sat: float = 30.0) -> np.ndarray:
        """Detects shadow regions using luminance and saturation thresholds."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # V channel = brightness/luminance
        v = hsv[:, :, 2]
        
        # S channel = saturation
        s = hsv[:, :, 1]
        
        # Shadows: low brightness AND low saturation (or vice versa)
        shadow_mask = ((v < threshold_lum) | (s < threshold_sat)).astype(np.uint8) * 255
        
        # Morphological smoothing to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        
        return shadow_mask
    
    def _apply_retinex_lighting(self, warped_ad, frame, mask):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Multi-scale for accuracy
        scales = [15, 45, 120]
        illum_maps = [cv2.GaussianBlur(gray_frame, (s if s%2==1 else s+1, s if s%2==1 else s+1), 0) for s in scales]
        illumination = np.mean(illum_maps, axis=0)
    
        mask_pixels = illumination[mask > 0]
        avg_local_illum = float(np.mean(mask_pixels)) if mask_pixels.size > 0 else 128.0
        avg_local_illum = max(avg_local_illum, 1.0)
    
        illum_multiplier = illumination / avg_local_illum
    
        # --- CLAMP THE MULTIPLIER ---
        # Don't let it brighten or darken the ad more than ±25%
        illum_multiplier = np.clip(illum_multiplier, 0.75, 1.25)
    
        ad_retinex = warped_ad.astype(np.float32)
        for c in range(3):
            ad_retinex[:, :, c] *= illum_multiplier
    
        return np.clip(ad_retinex, 0, 255).astype(np.uint8)
    
    def _apply_shadow_aware_blending(self, warped_ad: np.ndarray, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply stronger darkening in shadow regions while preserving bright areas."""
        shadow_mask = self._detect_shadow_regions(frame)
        shadow_regions = (shadow_mask > 0) & (mask > 0)
        
        ad_shadow_aware = warped_ad.astype(np.float32)
        
        # Darken more aggressively in shadows (scale multiplier)
        if np.count_nonzero(shadow_regions) > 0:
            ad_shadow_aware[shadow_regions] *= 0.75  # Reduce to 75% brightness in shadows
        
        return np.clip(ad_shadow_aware, 0, 255).astype(np.uint8)
    
    def _guided_filter_feather(self, mask: np.ndarray, guide: np.ndarray, radius: int = 15, eps: float = 0.01) -> np.ndarray:
        """Edge-aware feathering using guided filter (uses background frame as guide)."""
        # Convert guide to grayscale if not already
        if len(guide.shape) == 3:
            guide = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            guide = guide.astype(np.float32) / 255.0
        
        mask_float = mask.astype(np.float32) / 255.0
        
        # Simplified guided filter using edge-aware smoothing
        # Step 1: Create a coarse approximation using bilateral filter on the guide
        guide_edges = cv2.bilateralFilter((guide * 255).astype(np.uint8), 9, 15, 15).astype(np.float32) / 255.0 
        
        # Step 2: Compute mean of mask and guide in local windows
        mean_mask = cv2.boxFilter(mask_float, -1, (radius, radius))
        mean_guide = cv2.boxFilter(guide_edges, -1, (radius, radius))
        
        mean_mask_guide = cv2.boxFilter(mask_float * guide_edges, -1, (radius, radius))
        var_guide = cv2.boxFilter(guide_edges * guide_edges, -1, (radius, radius)) - mean_guide * mean_guide
        
        # Step 3: Compute linear coefficients
        cov_mask_guide = mean_mask_guide - mean_mask * mean_guide
        a = cov_mask_guide / (var_guide + eps)
        b = mean_mask - a * mean_guide
        
        # Step 4: Apply to get filtered mask
        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))
        
        filtered = mean_a * guide_edges + mean_b
        filtered = np.clip(filtered, 0, 1) * 255
        
        return filtered.astype(np.uint8)
    
    def _detect_and_transfer_specular_highlights(self, blended: np.ndarray, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Detect bright spots on background plane and composite them on top using screen blending."""
        # Convert to grayscale for highlight detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Detect bright regions (specular highlights): very high luminance
        highlight_threshold = np.percentile(gray_frame[mask > 0], 85) if np.count_nonzero(mask) > 0 else 0.8
        highlight_mask = (gray_frame > highlight_threshold).astype(np.uint8) * 255
        highlight_mask = cv2.bitwise_and(highlight_mask, highlight_mask, mask=mask)
        
        # Dilate highlights slightly for visibility
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        highlight_mask = cv2.dilate(highlight_mask, kernel, iterations=1)
        
        # Extract highlight regions from original frame
        highlight_regions = frame.copy()
        
        # Apply screen blending for highlights: result = 1 - (1-A) * (1-B)
        result = blended.astype(np.float32) / 255.0
        highlights = highlight_regions.astype(np.float32) / 255.0
        highlight_alpha = highlight_mask.astype(np.float32) / 255.0
        
        for c in range(3):
            # Screen blend: additive blending that brightens
            screen_blended = 1.0 - (1.0 - result[:, :, c]) * (1.0 - highlights[:, :, c])
            result[:, :, c] = (
                highlight_alpha * screen_blended +
                (1.0 - highlight_alpha) * result[:, :, c]
            )
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    def _apply_lab_harmonization(self, blended_img, frame, mask, strength=0.3):
        blend_lab = cv2.cvtColor(blended_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        mask_region = mask > 0
        if np.count_nonzero(mask_region) == 0:
            return blended_img

        blend_pixels = blend_lab[mask_region]
        frame_pixels = frame_lab[mask_region]

        blend_mean = np.mean(blend_pixels, axis=0)
        blend_std  = np.std(blend_pixels, axis=0)
        frame_mean = np.mean(frame_pixels, axis=0)
        frame_std  = np.std(frame_pixels, axis=0)

        adjusted = blend_lab.copy()
        region = adjusted[mask_region]

        # Standard distribution match
        matched = (region - blend_mean) * (frame_std / (blend_std + 1e-6)) + frame_mean

        # --- SATURATION PROTECTION ---
        # Chroma = sqrt(a^2 + b^2) in LAB space (channels 1 and 2)
        chroma = np.sqrt(region[:, 1] ** 2 + region[:, 2] ** 2)
        # Normalize to [0, 1] — pixels with chroma > ~30 are visibly saturated
        sat_weight = np.clip(chroma / 40.0, 0, 1)  # 1 = fully saturated, 0 = gray
        # Reduce harmonization strength proportionally to saturation
        # Saturated pixels (logo red) get near-zero harmonization
        effective_strength = strength * (1.0 - sat_weight)  # shape: (N,)
        effective_strength = effective_strength[:, np.newaxis]  # broadcast over channels

        # Only harmonize L channel at full strength (lighting), protect a* b* (color)
        blended = np.zeros_like(region)
        blended[:, 0] = strength * matched[:, 0] + (1.0 - strength) * region[:, 0]       # L: always adjust
        blended[:, 1] = effective_strength[:, 0] * matched[:, 1] + (1.0 - effective_strength[:, 0]) * region[:, 1]  # a*: protected
        blended[:, 2] = effective_strength[:, 0] * matched[:, 2] + (1.0 - effective_strength[:, 0]) * region[:, 2]  # b*: protected

        adjusted[mask_region] = blended
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)

    def blend_ad_into_roi(self, frame: np.ndarray, dst_pts: np.ndarray, mode: str = 'retinex', feather_radius: int = 15) -> np.ndarray:
        """
        Advanced blending pipeline:
        1. Multi-scale Retinex for accurate lighting
        2. Shadow-aware darkening
        3. Edge-aware guided filter feathering
        4. Linear RGB compositing with proper gamma
        5. Selective LAB chrominance harmonization
        6. Specular highlight transfer
        """
        warped_ad, warped_mask = self._create_warped_ad_and_mask(frame.shape, dst_pts)
        if np.count_nonzero(warped_mask) == 0:
            return frame

        blended_base = frame.copy()
        mask_for_clone = cv2.threshold(warped_mask, 1, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        center = tuple(np.mean(dst_pts, axis=0).astype(int))

        # Apply advanced Retinex illumination matching
        if mode == 'retinex':
            warped_ad = self._apply_multiscale_retinex(warped_ad, frame, mask_for_clone)
            warped_ad = self._apply_shadow_aware_blending(warped_ad, frame, mask_for_clone)
            mode = 'feather'  # Hand over to feather processing stage

        if mode in ('seamless', 'seamless_feather'):
            try:
                blended_base = cv2.seamlessClone(warped_ad, frame, mask_for_clone, center, cv2.MIXED_CLONE)
            except cv2.error:
                blended_base = frame.copy()

        if mode in ('feather', 'seamless_feather'):
            # ============================================================
            # EDGE-AWARE FEATHERING (Guided Filter)
            # ============================================================
            
            soft_mask = self._guided_filter_feather(
                warped_mask,
                frame,
                radius=feather_radius
            )

            alpha = soft_mask.astype(np.float32) / 255.0

            alpha_3c = cv2.merge([
                alpha,
                alpha,
                alpha
            ])

            # ============================================================
            # SELECT SOURCE
            # ============================================================

            src_img = (
                warped_ad
                if mode == 'feather'
                else blended_base
            )

            # ============================================================
            # CONVERT TO LINEAR RGB
            # ============================================================

            src_linear = srgb_to_linear(src_img)
            dst_linear = srgb_to_linear(frame)

            # ============================================================
            # LINEAR RGB BLENDING
            # ============================================================

            composite_linear = (
                alpha_3c * src_linear +
                (1.0 - alpha_3c) * dst_linear
            )

            # ============================================================
            # CONVERT BACK TO SRGB
            # ============================================================

            composite_srgb = linear_to_srgb(composite_linear)

            # ============================================================
            # SELECTIVE LAB HARMONIZATION (Chrominance only)
            # ============================================================

            harmonized = self._apply_lab_harmonization(
                composite_srgb,
                frame,
                mask_for_clone,
                strength=0.3
            )

            # ============================================================
            # SPECULAR HIGHLIGHT TRANSFER
            # ============================================================

            final_result = self._detect_and_transfer_specular_highlights(
                harmonized,
                frame,
                mask_for_clone
            )

            return final_result

        return blended_base

    def initialize_from_first_frame(self, img0: np.ndarray, target_width: int = 1280) -> bool:
        """Handles interactive ROI / Plane selection and resizes the base frame for speed."""
        h, w = img0.shape[:2]
        scale = target_width / float(w)
        target_height = int(h * scale)
        
        img0_resized = cv2.resize(img0, (target_width, target_height), interpolation=cv2.INTER_AREA)
        img_H, img_W = img0_resized.shape[:2]

        # 1. Select Graphic Targeting Region
        cv2.namedWindow(self.win_roi, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_roi, 960, 540)
        print("--> Select the TARGET ROI you want to track, then press ENTER.")
        x, y, w, h = cv2.selectROI(self.win_roi, img0_resized, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(self.win_roi)
        
        if w == 0 or h == 0:
            print("Error: Invalid target ROI bounding setup parameters.")
            return False
        self.ad_roi_0 = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        self.ad_roi_prev = self.ad_roi_0.copy()

        # 2. Select Homography Ground Plane Surface Region
        cv2.namedWindow(self.win_plane, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_plane, 960, 540)
        print("--> Select the FLAT SURFACE PLANE containing the target, then press ENTER.")
        x_p, y_p, w_p, h_p = cv2.selectROI(self.win_plane, img0_resized, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(self.win_plane)
        
        if w_p == 0 or h_p == 0:
            print("Error: Invalid plane ROI surface bounds configuration context.")
            return False

        plane_mask0 = np.zeros((img_H, img_W), dtype=np.uint8)
        cv2.rectangle(plane_mask0, (x_p, y_p), (x_p + w_p, y_p + h_p), 255, -1)

        self.kp0, self.des0 = self.sift.detectAndCompute(img0_resized, mask=plane_mask0)
        if self.des0 is None or len(self.kp0) < 4:
            print("Error: Plain surface texture missing critical features configurations.")
            return False

        print(f"Tracking setup completed. Base plane anchors loaded: {len(self.kp0)}")
        return True

    def validate_polygon(self, pts: np.ndarray) -> bool:
        """Validates structural integrity by comparing edge proportions directly to Frame 0."""
        render_pts = np.round(pts).astype(np.int32)
        
        # 1. Basic Convexity Guard
        if not cv2.isContourConvex(render_pts):
            return False

        # 2. Compute current edge lengths
        # pts order: 0=TL, 1=TR, 2=BR, 3=BL
        len_top = np.linalg.norm(pts[0] - pts[1])
        len_right = np.linalg.norm(pts[1] - pts[2])
        len_bottom = np.linalg.norm(pts[2] - pts[3])
        len_left = np.linalg.norm(pts[3] - pts[0])
        
        # 3. Compute original edge lengths from Frame 0
        orig_pts = self.ad_roi_0
        orig_top = np.linalg.norm(orig_pts[0] - orig_pts[1]) # type: ignore
        orig_right = np.linalg.norm(orig_pts[1] - orig_pts[2]) # type: ignore
        orig_bottom = np.linalg.norm(orig_pts[2] - orig_pts[3]) # type: ignore
        orig_left = np.linalg.norm(orig_pts[3] - orig_pts[0]) # type: ignore

        # Avoid division by zero bugs
        if any(v < 1.0 for v in [orig_top, orig_right, orig_bottom, orig_left, len_top, len_right, len_bottom, len_left]):
            return False

        # 4. Calculate individual scaling factors per edge relative to baseline
        scale_top = len_top / orig_top
        scale_right = len_right / orig_right
        scale_bottom = len_bottom / orig_bottom
        scale_left = len_left / orig_left

        # 5. RIGID GUARD: Edge Scale Variance Check
        # In a real perspective warp, parallel edges scale relatively closely.
        # If the top edge expands by 2x but the bottom shrinks to 0.5x, it's a skewed calculation.
        scales = [scale_top, scale_right, scale_bottom, scale_left]
        max_scale = max(scales)
        min_scale = min(scales)
        
        # Rejects the frame if the distortion variance between any two edges exceeds a strict threshold
        if (max_scale / min_scale) > 1.45:
            return False

        # 6. Absolute Size Boundaries
        current_area = cv2.contourArea(render_pts)
        base_area = cv2.contourArea(np.round(self.ad_roi_0).astype(np.int32)) # type: ignore
        if current_area < (base_area * 0.3) or current_area > (base_area * 3.0):
            return False

        return True

    def track_and_render_frame(self, img1: np.ndarray, frame_idx: int, log_file_handle, blend_mode: str = 'retinex') -> np.ndarray:
        """Processes keypoint translation mapping updates dynamically tracking the asset frame changes."""
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        ad_roi_i = self.ad_roi_prev.copy() # type: ignore
        tracking_status = "LOST (Using Last Known)"

        if des1 is not None and len(self.des0) >= 4 and len(des1) >= 4: # type: ignore
            matches = self.flann.knnMatch(self.des0, des1, k=2) # type: ignore
            
            good_matches = []
            for m, n in matches:
                # Restored to 0.70 to allow enough structural points for MAGSAC to process
                if m.distance < 0.70 * n.distance:
                    if 0 <= m.queryIdx < len(self.kp0) and 0 <= m.trainIdx < len(kp1):
                        good_matches.append(m)

            if len(good_matches) >= 4:
                src_pts = np.array([self.kp0[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
                dst_pts = np.array([kp1[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)

                try:
                    # Upgrade to USAC_MAGSAC for significantly advanced noise filtering performance
                    Hg, _ = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 3.0, maxIters=2000, confidence=0.99)
                    if Hg is not None:
                        candidate_roi = cv2.perspectiveTransform(self.ad_roi_0.reshape(-1, 1, 2), Hg).reshape(-1, 2) # type: ignore
                        
                        # Validate structural constraints directly against Frame 0 blueprints
                        if self.validate_polygon(candidate_roi):
                            ad_roi_i = candidate_roi
                            self.ad_roi_prev = ad_roi_i  # Safe to update now
                            tracking_status = "TRACKING OK"
                        else:
                            tracking_status = "FAILED VALIDATION (Skewed Geometry)"
                except cv2.error:
                    tracking_status = "HOMOGRAPHY ERROR"

        # Format and log string coordinates data
        coords_list = np.round(ad_roi_i, 1).tolist()
        log_line = f"[{tracking_status}] Frame {frame_idx:04d} Coords -> TL: {coords_list[0]}, TR: {coords_list[1]}, BR: {coords_list[2]}, BL: {coords_list[3]}"
        
        print(log_line)
        log_file_handle.write(log_line + "\n")

        # Blend graphic overlays onto the background image
        img1 = self.blend_ad_into_roi(img1, ad_roi_i, mode=blend_mode)
        # render_pts = np.round(ad_roi_i).astype(np.int32)
        # cv2.polylines(img1, [render_pts], isClosed=True, color=(0, 0, 255), thickness=3)
        return img1
    
    def start_pipeline(self, video_path: str, blend_mode: str = 'retinex', target_width: int = 1280):
        """Main execution tracking wrapper tracking frames over the video lifecycle loop stream."""
        cap = cv2.VideoCapture(video_path)
        ret, img0 = cap.read()
        if not ret:
            print("Error: Unable to load properties target source video container file stream context.")
            cap.release()
            return

        if not self.initialize_from_first_frame(img0, target_width=target_width):
            cap.release()
            return

        # Setup save targets directories
        output_dir = "./real-time-vid-tracking-and-enhancement/tracked_frames_hybrid"
        os.makedirs(output_dir, exist_ok=True)
        log_filepath = "./real-time-vid-tracking-and-enhancement/tracking_log_hybrid.txt"

        cv2.namedWindow(self.win_live, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_live, 960, 540)

        h, w = img0.shape[:2]
        scale = target_width / float(w)
        target_height = int(h * scale)

        frame_idx = 0
        # Open text file using context manager handle structure to ensure logs are saved safely
        with open(log_filepath, "w", encoding="utf-8") as log_file:
            log_file.write("--- PLANAR TRACKING RUN LOG STARTED ---\n")
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Processing completed. End of video stream successfully reached.")
                        break

                    frame_idx += 1
                    frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

                    # Track components and pipe file streams safely
                    output_frame = self.track_and_render_frame(frame_resized, frame_idx, log_file, blend_mode=blend_mode)
                    
                    # Save files on a per-frame sequential matrix array basis
                    frame_filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
                    cv2.imwrite(frame_filename, output_frame)
                    
                    cv2.imshow(self.win_live, output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    TARGET_VIDEO = "6337008-uhd_3840_2160_25fps.mp4"
    # LOGO_ASSET = "pepsi_logo.png"
    LOGO_ASSET = "lego.png"


    try:
        # Default blending is configured to 'retinex' for natural lighting changes
        tracker = PlanarARTracker(ad_image_path=LOGO_ASSET)
        tracker.start_pipeline(video_path=TARGET_VIDEO, blend_mode='retinex')
    except Exception as e:
        print(f"Application crash exception context logged: {e}", file=sys.stderr)