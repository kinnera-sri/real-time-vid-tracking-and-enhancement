import os

# Ensure OpenCV Qt windows can find system fonts in Linux environments.
# This should be set before importing cv2 to avoid Qt font warnings.
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
from typing import Dict, Union

def create_warped_ad_and_mask(ad_img, frame_shape, dst_pts):
    """Warp the ad image and generate a precise alpha mask based on image channels."""
    fh, fw = frame_shape[:2]
    ah, aw = ad_img.shape[:2]
    
    # Coordinates of the source ad image corners
    src_pts = np.array([[0, 0], [aw - 1, 0], [aw - 1, ah - 1], [0, ah - 1]], dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    
    # Calculate Homography Matrix
    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    # Separate color and alpha channel if image is RGBA
    if ad_img.shape[2] == 4:
        bgr = ad_img[:, :, :3]
        alpha = ad_img[:, :, 3]
    else:
        bgr = ad_img
        alpha = np.ones((ah, aw), dtype=np.uint8) * 255

    # Warp both the BGR image and the Alpha mask accurately
    warped_ad = cv2.warpPerspective(bgr, H, (fw, fh))
    warped_mask = cv2.warpPerspective(alpha, H, (fw, fh))
    
    return warped_ad, warped_mask


def blend_ad_into_roi(frame, ad_img, dst_pts, mode='feather', feather_radius=5):
    """Blend `ad_img` into `frame` at polygon `dst_pts` matching pixel specifications.

    mode: one of 'seamless', 'feather', 'seamless_feather'
    - 'feather' (RECOMMENDED FOR LOGOS): Pure alpha blending with a soft boundary transition.
    - 'seamless': Matches background illumination (use only for solid, non-branding textures).
    - 'seamless_feather': Color matches the interior via seamlessClone and softens the outer perimeter.
    """
    # 1. Warp the image and isolate its true alpha mask
    warped_ad, warped_mask = create_warped_ad_and_mask(ad_img, frame.shape, dst_pts)
    if np.count_nonzero(warped_mask) == 0:
        return frame

    # 2. Setup baseline canvas layers
    blended_base = frame.copy()
    
    # Create clean binary mask required by seamlessClone operations
    mask_for_clone = cv2.threshold(warped_mask, 1, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    center = tuple(np.mean(dst_pts, axis=0).astype(int))

    # --- SEAMLESS CLONING STAGE ---
    if mode in ('seamless', 'seamless_feather'):
        try:
            # Use MIXED_CLONE so background textures can show through transparent areas if needed
            blended_base = cv2.seamlessClone(warped_ad, frame, mask_for_clone, center, cv2.MIXED_CLONE)
        except cv2.error:
            blended_base = frame.copy()

    # --- FEATHER (ALPHA BLENDING) STAGE ---
    if mode in ('feather', 'seamless_feather'):
        # Ensure feather radius configuration is an odd integer
        if feather_radius % 2 == 0:
            feather_radius += 1
            
        # Create a smooth alpha gradient map at the boundaries
        soft_mask = cv2.GaussianBlur(warped_mask, (feather_radius, feather_radius), 0)
        alpha = soft_mask.astype(np.float32) / 255.0
        alpha_3c = cv2.merge([alpha, alpha, alpha])

        # Source layer is determined by whether seamless execution occurred first
        src_f = warped_ad.astype(np.float32) if mode == 'feather' else blended_base.astype(np.float32)
        dst_f = frame.astype(np.float32)

        # Mathematical Linear Interpolation Blend Formula
        composite = (alpha_3c * src_f) + ((1.0 - alpha_3c) * dst_f)
        final = np.clip(composite, 0, 255).astype(np.uint8)
        return final

    return blended_base

def run_pure_opencv_tracker(video_path):
    # 1. Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=2000)
    
    # Configure FLANN matcher for speed
    FLANN_INDEX_KDTREE = 1
    # Use explicit value union to satisfy strict type checkers
    IndexParamType = Dict[str, Union[bool, int, float, str]]
    index_params: IndexParamType = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
    search_params: IndexParamType = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 2. Load the ad image and open the video stream
    ad_image_path = "pepsi_logo.png"
    # ad_image_path = "lego.png"
    ad_img = cv2.imread(ad_image_path, cv2.IMREAD_UNCHANGED)
    if ad_img is None:
        print(f"Error: Could not load ad image from '{ad_image_path}'.")
        return

    cap = cv2.VideoCapture(video_path)
    ret, img0 = cap.read()
    if not ret:
        print("Error: Could not read the video file.")
        return

    img_H, img_W = img0.shape[:2]


    # ########################################################################
    # Get the ad and plane ROIs
    # ########################################################################
    
    # Create and resize the ROI selection windows before calling selectROI
    cv2.namedWindow("1. Select Target ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1. Select Target ROI", 960, 540)

    print("--> Select the TARGET ROI you want to track, then press ENTER.")
    x, y, w, h = cv2.selectROI("1. Select Target ROI", img0, fromCenter=False, showCrosshair=True)
    if w == 0 or h == 0:
        print("Error: Invalid target ROI selection. Please select a non-zero target area.")
        return
    ad_roi_0 = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    cv2.destroyWindow("1. Select Target ROI")

    cv2.namedWindow("2. Select Surface Plane", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("2. Select Surface Plane", 960, 540)

    print("--> Select the FLAT SURFACE PLANE containing the target, then press ENTER.")
    x_p, y_p, w_p, h_p = cv2.selectROI("2. Select Surface Plane", img0, fromCenter=False, showCrosshair=True)
    if w_p == 0 or h_p == 0:
        print("Error: Invalid plane ROI selection. Please select a non-zero plane area.")
        return
    cv2.destroyWindow("2. Select Surface Plane")

    # Create plane mask
    plane_mask0 = np.zeros((img_H, img_W), dtype=np.uint8)
    cv2.rectangle(plane_mask0, (x_p, y_p), (x_p + w_p, y_p + h_p), 255, -1)

    # 4. Extract Base Features from Frame 0
    kp0, des0 = sift.detectAndCompute(img0, mask=plane_mask0)
    if des0 is None or len(kp0) < 4:
        print("Error: Not enough texture/features found in your selected plane region.")
        return

    print(f"Tracking initiated with {len(kp0)} features on the base plane.")
    ad_roi_prev = ad_roi_0

    cv2.namedWindow("Live Planar Tracking (Press 'q' to Quit)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Planar Tracking (Press 'q' to Quit)", 960, 540)

    # 5. Process Video Loop
    while True:
        ret, img1 = cap.read()
        if not ret:
            print("End of video stream reached.")
            break

        # Detect features on current frame
        kp1, des1 = sift.detectAndCompute(img1, None)
        
        if des1 is not None and len(des1) >= 4:
            # Match features between frames
            matches = flann.knnMatch(des0, des1, k=2)
            
            # Apply Lowe's Ratio Test to filter good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            if len(good_matches) >= 4:
                # Validate matches and guard against out-of-range indices
                valid_matches = []
                for m in good_matches:
                    # ensure the match object has expected attributes and indices are valid
                    if not hasattr(m, 'queryIdx') or not hasattr(m, 'trainIdx'):
                        continue
                    if m.queryIdx < 0 or m.trainIdx < 0:
                        continue
                    if m.queryIdx >= len(kp0) or m.trainIdx >= len(kp1):
                        continue
                    valid_matches.append(m)

                if len(valid_matches) >= 4:
                    # Extract coordinates of matched pairs
                    src_pts = np.array([kp0[m.queryIdx].pt for m in valid_matches], dtype=np.float32).reshape(-1, 1, 2)
                    dst_pts = np.array([kp1[m.trainIdx].pt for m in valid_matches], dtype=np.float32).reshape(-1, 1, 2)

                    # Compute Homography using RANSAC
                    try:
                        Hg, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    except cv2.error:
                        Hg = None
                        mask = None

                    if Hg is not None:
                        # Warp the 4 ROI points into the new frame space
                        ad_roi_i = cv2.perspectiveTransform(ad_roi_0.reshape(-1, 1, 2), Hg).reshape(-1, 2)
                        ad_roi_prev = ad_roi_i
                    else:
                        ad_roi_i = ad_roi_prev
                else:
                    # Not enough valid matches after filtering
                    ad_roi_i = ad_roi_prev
            else:
                ad_roi_i = ad_roi_prev
        else:
            ad_roi_i = ad_roi_prev

        # 6. Render the ad into the tracked ROI and draw the ROI polygon
        render_pts = np.round(ad_roi_i).astype(np.int32)
        img1 = blend_ad_into_roi(img1, ad_img, ad_roi_i, mode='seamless')
        cv2.polylines(img1, [render_pts], isClosed=True, color=(0, 0, 255), thickness=3)

        tracked_roi_coords = ad_roi_i.reshape(-1, 2).tolist()
        # Uncomment the next line to print tracked ROI coordinates each frame:
        # print("Tracked ROI coords:", tracked_roi_coords)

        cv2.imshow("Live Planar Tracking (Press 'q' to Quit)", img1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change this path to point directly to your video file
    TARGET_VIDEO = "6337008-uhd_3840_2160_25fps.mp4" 
    run_pure_opencv_tracker(TARGET_VIDEO)