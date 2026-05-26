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

class PlanarARTracker:
    def __init__(self, ad_image_path: str, feature_limit: int = 2000):
        """Initializes the tracker with SIFT, FLANN matching, and the target graphic."""
        self.sift = cv2.SIFT_create(nfeatures=feature_limit)
        
        # Configure FLANN matcher with explicit typing for strict environments
        FLANN_INDEX_KDTREE = 1
        IndexParamType = Dict[str, Union[bool, int, float, str]]
        index_params: IndexParamType = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
        search_params: IndexParamType = {"checks": 50}
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
        ah, aw = self.ad_img.shape[:2]
        
        src_pts = np.array([[0, 0], [aw - 1, 0], [aw - 1, ah - 1], [0, ah - 1]], dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)
        
        H, _ = cv2.findHomography(src_pts, dst_pts)
        if H is None:
            return np.zeros((fh, fw, 3), dtype=np.uint8), np.zeros((fh, fw), dtype=np.uint8)
        
        # Handle RGBA transparent channel split seamlessly
        if self.ad_img.shape[2] == 4:
            bgr = self.ad_img[:, :, :3]
            alpha = self.ad_img[:, :, 3]
        else:
            bgr = self.ad_img
            alpha = np.ones((ah, aw), dtype=np.uint8) * 255

        warped_ad = cv2.warpPerspective(bgr, H, (fw, fh))
        warped_mask = cv2.warpPerspective(alpha, H, (fw, fh))
        return warped_ad, warped_mask

    def blend_ad_into_roi(self, frame: np.ndarray, dst_pts: np.ndarray, mode: str = 'feather', feather_radius: int = 5) -> np.ndarray:
        """Blends the warped asset onto the video frame backdrop."""
        warped_ad, warped_mask = self._create_warped_ad_and_mask(frame.shape, dst_pts)
        if np.count_nonzero(warped_mask) == 0:
            return frame

        blended_base = frame.copy()
        mask_for_clone = cv2.threshold(warped_mask, 1, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        center = tuple(np.mean(dst_pts, axis=0).astype(int))

        if mode in ('seamless', 'seamless_feather'):
            try:
                blended_base = cv2.seamlessClone(warped_ad, frame, mask_for_clone, center, cv2.MIXED_CLONE)
            except cv2.error:
                blended_base = frame.copy()

        if mode in ('feather', 'seamless_feather'):
            if feather_radius % 2 == 0:
                feather_radius += 1
                
            soft_mask = cv2.GaussianBlur(warped_mask, (feather_radius, feather_radius), 0)
            alpha = soft_mask.astype(np.float32) / 255.0
            alpha_3c = cv2.merge([alpha, alpha, alpha])

            src_f = warped_ad.astype(np.float32) if mode == 'feather' else blended_base.astype(np.float32)
            dst_f = frame.astype(np.float32)

            composite = (alpha_3c * src_f) + ((1.0 - alpha_3c) * dst_f)
            return np.clip(composite, 0, 255).astype(np.uint8)

        return blended_base

    def initialize_from_first_frame(self, img0: np.ndarray) -> bool:
        """Handles interactive ROI / Plane selection initialization logic metrics."""
        img_H, img_W = img0.shape[:2]

        # 1. Select Graphic Targeting Region
        cv2.namedWindow(self.win_roi, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_roi, 960, 540)
        print("--> Select the TARGET ROI you want to track, then press ENTER.")
        x, y, w, h = cv2.selectROI(self.win_roi, img0, fromCenter=False, showCrosshair=True)
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
        x_p, y_p, w_p, h_p = cv2.selectROI(self.win_plane, img0, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(self.win_plane)
        
        if w_p == 0 or h_p == 0:
            print("Error: Invalid plane ROI surface bounds configuration context.")
            return False

        # Isolate features exclusively inside key tracking plane bounds
        plane_mask0 = np.zeros((img_H, img_W), dtype=np.uint8)
        cv2.rectangle(plane_mask0, (x_p, y_p), (x_p + w_p, y_p + h_p), 255, -1)

        self.kp0, self.des0 = self.sift.detectAndCompute(img0, mask=plane_mask0)
        if self.des0 is None or len(self.kp0) < 4:
            print("Error: Plain surface texture missing critical features configurations.")
            return False

        print(f"Tracking setup completed. Base plane anchors loaded: {len(self.kp0)}")
        return True

    def validate_polygon(self, pts: np.ndarray) -> bool:
        """Verifies if the tracked polygon shape remains convex and mathematically realistic."""
        render_pts = np.round(pts).astype(np.int32)
        # Prevents self-intersecting anomalies or flipped geometric projections
        if not cv2.isContourConvex(render_pts):
            return False
        # Prevent degenerate near-zero scaling errors
        area = cv2.contourArea(render_pts)
        if area < 100:
            return False
        return True

    def track_and_render_frame(self, img1: np.ndarray, blend_mode: str = 'feather') -> np.ndarray:
        """Processes keypoint translation mapping updates dynamically tracking the asset frame changes."""
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        ad_roi_i = self.ad_roi_prev.copy()

        if des1 is not None and len(des1) >= 4:
            matches = self.flann.knnMatch(self.des0, des1, k=2)
            
            # Filter matches using standard Lowe's Ratio Testing metrics
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    if 0 <= m.queryIdx < len(self.kp0) and 0 <= m.trainIdx < len(kp1):
                        good_matches.append(m)

            if len(good_matches) >= 4:
                src_pts = np.array([self.kp0[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
                dst_pts = np.array([kp1[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)

                try:
                    Hg, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if Hg is not None:
                        candidate_roi = cv2.perspectiveTransform(self.ad_roi_0.reshape(-1, 1, 2), Hg).reshape(-1, 2)
                        # Validate the transformed polygon output structure before updating frame space
                        if self.validate_polygon(candidate_roi):
                            ad_roi_i = candidate_roi
                            self.ad_roi_prev = ad_roi_i
                except cv2.error:
                    pass

        # Apply compositing asset transformations onto output canvas array layers
        img1 = self.blend_ad_into_roi(img1, ad_roi_i, mode=blend_mode)
        
        # Border box rendering alignment check boundaries
        render_pts = np.round(ad_roi_i).astype(np.int32)
        cv2.polylines(img1, [render_pts], isClosed=True, color=(0, 0, 255), thickness=3)
        return img1

    def start_pipeline(self, video_path: str, blend_mode: str = 'feather'):
        """Main execution tracking wrapper tracking frames over the video lifecycle loop stream."""
        cap = cv2.VideoCapture(video_path)
        ret, img0 = cap.read()
        if not ret:
            print("Error: Unable to load properties target source video container file stream context.")
            cap.release()
            return

        if not self.initialize_from_first_frame(img0):
            cap.release()
            return

        cv2.namedWindow(self.win_live, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_live, 960, 540)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Processing completed. End of video stream successfully reached.")
                    break

                output_frame = self.track_and_render_frame(frame, blend_mode=blend_mode)
                
                cv2.imshow(self.win_live, output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Execution entry configuration
    TARGET_VIDEO = "6337008-uhd_3840_2160_25fps.mp4"
    LOGO_ASSET = "pepsi_logo.png"

    try:
        tracker = PlanarARTracker(ad_image_path=LOGO_ASSET)
        tracker.start_pipeline(video_path=TARGET_VIDEO, blend_mode='feather')
    except Exception as e:
        print(f"Application crash exception context logged: {e}", file=sys.stderr)