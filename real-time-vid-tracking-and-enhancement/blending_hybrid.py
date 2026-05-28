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

    def _apply_retinex_lighting(self, warped_ad: np.ndarray, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extracts ambient lighting map from background frame (Retinex) and casts it onto the asset."""
        # Convert background frame to grayscale to calculate illumination intensity
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Large Gaussian kernel acts as a low-pass filter estimating the ambient illumination map (L)
        illumination = cv2.GaussianBlur(gray_frame, (45, 45), 0)
        
        # Safeguard against dividing by zero if the mask region is completely empty
        mask_pixels = illumination[mask > 0]
        avg_local_illum = np.mean(mask_pixels) if mask_pixels.size > 0 else 128.0
        if avg_local_illum < 1.0: 
            avg_local_illum = 1.0
            
        # Create a localized lighting modifier map
        illum_multiplier = illumination / avg_local_illum
        
        # Multiply the lighting variation map across the target asset channels
        ad_retinex = warped_ad.astype(np.float32)
        for c in range(3):
            ad_retinex[:, :, c] *= illum_multiplier
            
        return np.clip(ad_retinex, 0, 255).astype(np.uint8)
    
    def _apply_lab_harmonization(
        self,
        blended_img,
        frame,
        mask,
        strength=0.6
    ):

        # --------------------------------------------------------
        # CONVERT TO LAB
        # --------------------------------------------------------

        blend_lab = cv2.cvtColor(
            blended_img,
            cv2.COLOR_BGR2LAB
        ).astype(np.float32)

        frame_lab = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2LAB
        ).astype(np.float32)

        mask_region = mask > 0

        if np.count_nonzero(mask_region) == 0:
            return blended_img

        # --------------------------------------------------------
        # REGION STATS
        # --------------------------------------------------------

        blend_pixels = blend_lab[mask_region]

        frame_pixels = frame_lab[mask_region]

        blend_mean = np.mean(
            blend_pixels,
            axis=0
        )

        blend_std = np.std(
            blend_pixels,
            axis=0
        )

        frame_mean = np.mean(
            frame_pixels,
            axis=0
        )

        frame_std = np.std(
            frame_pixels,
            axis=0
        )

        # --------------------------------------------------------
        # MATCH DISTRIBUTION
        # --------------------------------------------------------

        adjusted = blend_lab.copy()

        region = adjusted[mask_region]

        region = (

            (region - blend_mean)

            *

            (frame_std / (blend_std + 1e-6))

            +

            frame_mean
        )

        # partial harmonization
        region = (

            strength * region

            +

            (1.0 - strength)

            * blend_lab[mask_region]
        )

        adjusted[mask_region] = region

        adjusted = np.clip(
            adjusted,
            0,
            255
        ).astype(np.uint8)

        return cv2.cvtColor(
            adjusted,
            cv2.COLOR_LAB2BGR
        )

    def blend_ad_into_roi(self, frame: np.ndarray, dst_pts: np.ndarray, mode: str = 'retinex', feather_radius: int = 5) -> np.ndarray:
        """Blends the warped asset onto the video frame backdrop using feathered alpha or Retinex mapping."""
        warped_ad, warped_mask = self._create_warped_ad_and_mask(frame.shape, dst_pts)
        if np.count_nonzero(warped_mask) == 0:
            return frame

        blended_base = frame.copy()
        mask_for_clone = cv2.threshold(warped_mask, 1, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        center = tuple(np.mean(dst_pts, axis=0).astype(int))

        # Apply Retinex illumination matching to the asset before standard compositing
        if mode == 'retinex':
            warped_ad = self._apply_retinex_lighting(warped_ad, frame, mask_for_clone)
            mode = 'feather' # Hand over execution down to the feather processing stage

        if mode in ('seamless', 'seamless_feather'):
            try:
                blended_base = cv2.seamlessClone(warped_ad, frame, mask_for_clone, center, cv2.MIXED_CLONE)
            except cv2.error:
                blended_base = frame.copy()

        if mode in ('feather', 'seamless_feather'):
            if feather_radius % 2 == 0:
                feather_radius += 1
                
            # ============================================================
            # FEATHER MASK
            # ============================================================

            soft_mask = cv2.GaussianBlur(
                warped_mask,
                (feather_radius, feather_radius),
                0
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

            composite_srgb = linear_to_srgb(
                composite_linear
            )

            # ============================================================
            # LAB HARMONIZATION
            # ============================================================

            final_result = self._apply_lab_harmonization(
                composite_srgb,
                frame,
                mask_for_clone,
                strength=0.5
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
        render_pts = np.round(ad_roi_i).astype(np.int32)
        cv2.polylines(img1, [render_pts], isClosed=True, color=(0, 0, 255), thickness=3)
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
        output_dir = "./real-time-vid-tracking-and-enhancement/tracked_frames_lego"
        os.makedirs(output_dir, exist_ok=True)
        log_filepath = "./real-time-vid-tracking-and-enhancement/tracking_log_lego.txt"

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