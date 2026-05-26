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

def warp_ad_into_roi(frame, ad_img, dst_pts):
    if ad_img.ndim == 2:
        ad_img = cv2.cvtColor(ad_img, cv2.COLOR_GRAY2BGR)

    src_h, src_w = ad_img.shape[:2]
    src_pts = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], dtype=np.float32)
    dst_pts = dst_pts.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_ad = cv2.warpPerspective(ad_img, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)

    if ad_img.shape[2] == 4:
        alpha = warped_ad[:, :, 3]
        mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)[1]
        warped_ad = warped_ad[:, :, :3]
    else:
        gray = cv2.cvtColor(warped_ad, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]

    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(warped_ad, warped_ad, mask=mask)
    return cv2.add(bg, fg)

def run_pure_opencv_tracker(video_path):
    # 1. Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=2000)
    
    # Configure FLANN matcher for speed
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
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
                # Extract coordinates of matched pairs
                src_pts = np.float32([kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Compute Homography using RANSAC
                Hg, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if Hg is not None:
                    # Warp the 4 ROI points into the new frame space
                    ad_roi_i = cv2.perspectiveTransform(ad_roi_0.reshape(-1, 1, 2), Hg).reshape(-1, 2)
                    ad_roi_prev = ad_roi_i
                else:
                    ad_roi_i = ad_roi_prev
            else:
                ad_roi_i = ad_roi_prev
        else:
            ad_roi_i = ad_roi_prev

        # 6. Render the ad into the tracked ROI and draw the ROI polygon
        render_pts = np.round(ad_roi_i).astype(np.int32)
        img1 = warp_ad_into_roi(img1, ad_img, ad_roi_i)
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