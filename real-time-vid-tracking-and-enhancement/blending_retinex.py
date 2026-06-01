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
# RETINEX BLENDING ONLY
# ============================================================

class RetinexBlender:
    def __init__(self, ad_image_path: str, feature_limit: int = 2000):
        """Initializes the tracker with SIFT, FLANN matching, and Retinex blending."""
        self.sift = cv2.SIFT_create(nfeatures=feature_limit) # type: ignore
        
        # Configure FLANN matcher
        FLANN_INDEX_KDTREE = 1
        IndexParamType = Dict[str, Union[bool, int, float, str]]
        index_params: IndexParamType = {"algorithm": FLANN_INDEX_KDTREE, "trees": 9} # type: ignore
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
        
        # Named windows configuration
        self.win_roi = "1. Select Target ROI"
        self.win_plane = "2. Select Surface Plane"
        self.win_live = "Live Retinex Tracking (Press 'q' to Quit)"
        
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
        illum_120 = cv2.GaussianBlur(gray_frame, (95, 95), 0).astype(np.float32)  # Must be odd
        
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
    
    def _apply_shadow_aware_blending(self, warped_ad: np.ndarray, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply stronger darkening in shadow regions while preserving bright areas."""
        # Convert to grayscale for shadow detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Detect dark regions (shadows)
        threshold_lum = 100.0
        shadow_mask = (gray_frame < threshold_lum).astype(np.uint8) * 255
        shadow_mask = cv2.bitwise_and(shadow_mask, shadow_mask, mask=mask)
        
        ad_shadow_aware = warped_ad.astype(np.float32)
        
        # Darken more aggressively in shadows
        shadow_regions = (shadow_mask > 0) & (mask > 0)
        if np.count_nonzero(shadow_regions) > 0:
            ad_shadow_aware[shadow_regions] *= 0.75  # Reduce to 75% brightness in shadows
        
        return np.clip(ad_shadow_aware, 0, 255).astype(np.uint8)

    def blend_ad_into_roi(self, frame: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """
        Retinex blending pipeline:
        1. Multi-scale Retinex for accurate lighting
        2. Shadow-aware darkening
        3. Alpha blending with mask
        """
        warped_ad, warped_mask = self._create_warped_ad_and_mask(frame.shape, dst_pts)
        if np.count_nonzero(warped_mask) == 0:
            return frame

        mask_for_clone = cv2.threshold(warped_mask, 1, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

        # Apply Retinex illumination matching
        warped_ad = self._apply_multiscale_retinex(warped_ad, frame, mask_for_clone)
        warped_ad = self._apply_shadow_aware_blending(warped_ad, frame, mask_for_clone)

        # Simple alpha blending
        alpha = warped_mask.astype(np.float32) / 255.0
        alpha_3c = cv2.merge([alpha, alpha, alpha])
        
        blended = (
            alpha_3c * warped_ad.astype(np.float32) +
            (1.0 - alpha_3c) * frame.astype(np.float32)
        )
        
        return np.clip(blended, 0, 255).astype(np.uint8)

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
        scales = [scale_top, scale_right, scale_bottom, scale_left]
        max_scale = max(scales)
        min_scale = min(scales)
        
        if (max_scale / min_scale) > 1.45:
            return False

        # 6. Absolute Size Boundaries
        current_area = cv2.contourArea(render_pts)
        base_area = cv2.contourArea(np.round(self.ad_roi_0).astype(np.int32)) # type: ignore
        if current_area < (base_area * 0.3) or current_area > (base_area * 3.0):
            return False

        return True

    def track_and_render_frame(self, img1: np.ndarray, frame_idx: int, log_file_handle) -> np.ndarray:
        """Processes keypoint translation mapping updates dynamically tracking the asset frame changes."""
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        ad_roi_i = self.ad_roi_prev.copy() # type: ignore
        tracking_status = "LOST (Using Last Known)"

        if des1 is not None and len(self.des0) >= 4 and len(des1) >= 4: # type: ignore
            matches = self.flann.knnMatch(self.des0, des1, k=2) # type: ignore
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.70 * n.distance:
                    if 0 <= m.queryIdx < len(self.kp0) and 0 <= m.trainIdx < len(kp1):
                        good_matches.append(m)

            if len(good_matches) >= 4:
                src_pts = np.array([self.kp0[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
                dst_pts = np.array([kp1[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)

                try:
                    Hg, _ = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 3.0, maxIters=2000, confidence=0.99)
                    if Hg is not None:
                        candidate_roi = cv2.perspectiveTransform(self.ad_roi_0.reshape(-1, 1, 2), Hg).reshape(-1, 2) # type: ignore
                        
                        if self.validate_polygon(candidate_roi):
                            ad_roi_i = candidate_roi
                            self.ad_roi_prev = ad_roi_i
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

        # Blend graphic overlays onto the background image using Retinex
        img1 = self.blend_ad_into_roi(img1, ad_roi_i)
        return img1
    
    def start_pipeline(self, video_path: str, target_width: int = 1280):
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
        output_dir = "./retinex_tracked_frames"
        os.makedirs(output_dir, exist_ok=True)
        log_filepath = "./retinex_tracking_log.txt"

        cv2.namedWindow(self.win_live, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_live, 960, 540)

        h, w = img0.shape[:2]
        scale = target_width / float(w)
        target_height = int(h * scale)

        frame_idx = 0
        # Open text file using context manager handle structure to ensure logs are saved safely
        with open(log_filepath, "w", encoding="utf-8") as log_file:
            log_file.write("--- RETINEX BLENDING TRACKING RUN LOG STARTED ---\n")
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Processing completed. End of video stream successfully reached.")
                        break

                    frame_idx += 1
                    frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

                    # Track components and pipe file streams safely
                    output_frame = self.track_and_render_frame(frame_resized, frame_idx, log_file)
                    
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
    LOGO_ASSET = "lego.png"

    try:
        # Pure Retinex blending
        blender = RetinexBlender(ad_image_path=LOGO_ASSET)
        blender.start_pipeline(video_path=TARGET_VIDEO)
    except Exception as e:
        print(f"Application crash exception context logged: {e}", file=sys.stderr)
