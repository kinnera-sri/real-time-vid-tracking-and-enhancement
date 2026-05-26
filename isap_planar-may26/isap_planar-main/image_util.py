import os
import cv2
import numpy as np


def draw_matches(img1, img2, ref_points, dst_points):
    """
    Draw matches between two images using keypoints and a homography transformation.
    Args:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.
        ref_points (numpy.ndarray): Keypoints in the first image (shape: Nx2).
        dst_points (numpy.ndarray): Keypoints in the second image (shape: Nx2).
    """
    H, mask = cv2.findHomography(
        ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999
    )
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    ).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i - 1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(
            img2_with_corners, start_point, end_point, (0, 255, 0), 4
        )  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    if 0:
        img_matches = cv2.drawMatches(
            img1,
            keypoints1,
            img2_with_corners,
            keypoints2,
            matches,
            None,
            matchColor=(0, 255, 0),
            flags=2,
        )
    else:
        img_matches = cv2.hconcat([img1, img2_with_corners])

    cv2.imshow("matches", img_matches)
    cv2.waitKey(0)


def change_png_background(img_bgra, new_background_color_bgr):
    """
    Changes the background of a PNG image.
    Transparent areas (defined by the alpha channel) are filled with the new_background_color_bgr.

    Args:
        image_path (str): Path to the input PNG image.
        new_background_color_bgr (tuple): The new background color in BGR format, e.g., (255, 0, 0) for blue.

    Returns:
        numpy.ndarray: The image with the new background, or None if an error occurs.
    """
    # Check if the image has an alpha channel
    if img_bgra.shape[2] != 4:
        print(
            "Error: Image does not have an alpha channel. This function replaces transparent backgrounds."
        )
        # Optionally, you could return the original image or handle it differently
        # For now, we'll return None as the operation isn't applicable as intended.
        return img_bgra  # Or return None if you prefer to signal failure

    # Extract the BGR channels and the Alpha channel
    b, g, r, a = cv2.split(img_bgra)

    # Create a 3-channel BGR version of the foreground
    foreground_bgr = cv2.merge((b, g, r))

    # Normalize the alpha channel to be used as a mask (0.0 - 1.0)
    alpha_normalized = a.astype(float) / 255.0

    # Create a 3-channel version of the alpha mask by repeating the alpha channel
    # This allows element-wise multiplication with the 3-channel foreground and background
    alpha_3channel = cv2.merge((alpha_normalized, alpha_normalized, alpha_normalized))

    # Create a solid background image with the new color
    # It should have the same dimensions as the foreground
    background_bgr_layer = np.full(
        foreground_bgr.shape, new_background_color_bgr, dtype=foreground_bgr.dtype
    )

    # Alpha blending:
    # Output = Foreground * Alpha + Background * (1 - Alpha)

    # Multiply the foreground with the alpha matte
    foreground_contribution = cv2.multiply(alpha_3channel, foreground_bgr.astype(float))

    # Multiply the background with (1 - alpha)
    background_contribution = cv2.multiply(
        1.0 - alpha_3channel, background_bgr_layer.astype(float)
    )

    # Add the foreground and background contributions
    result_bgr = cv2.add(foreground_contribution, background_contribution)

    # Convert back to uint8
    result_bgr = result_bgr.astype(np.uint8)

    return result_bgr


# write a function to select ROI points
points = []
drawing = False


def select_points_callback(event, x, y, flags, param):
    """
    Mouse callback function to select 4 points for homography.
    Args:
        event (int): Mouse event type.
        x (int): X coordinate of the mouse event.
        y (int): Y coordinate of the mouse event.
        flags (int): Flags for the mouse event.
        param (dict): Additional parameters, including the frame to draw on.
    """
    global points, drawing
    frame_display = param["frame_display"]

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(frame_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(
                "Select 4 Points on Court (TL, TR, BR, BL), then press 'c'",
                frame_display,
            )
        if len(points) == 4:
            print("4 points selected. Press 'c' to continue.")


def select_roi(first_frame):
    """
    Displays the first frame and lets the user select 4 points.
    The points should be selected in the order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    Returns:
        list: A list of points in the order they were selected.
    """
    global points
    points = []  # Reset points for each call if needed

    frame_display = first_frame.copy()
    cv2.namedWindow("Select 4 Points on Court (TL, TR, BR, BL), then press 'c'")
    cv2.setMouseCallback(
        "Select 4 Points on Court (TL, TR, BR, BL), then press 'c'",
        select_points_callback,
        {"frame_display": frame_display},
    )

    print(
        "Please select 4 points on the court in this order: Top-Left, Top-Right, Bottom-Right, Bottom-Left."
    )
    print(
        "Click on the image to select points. Press 'c' to confirm selection once 4 points are chosen."
    )

    while True:
        cv2.imshow(
            "Select 4 Points on Court (TL, TR, BR, BL), then press 'c'", frame_display
        )
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and len(points) == 4:
            break
        elif key == ord("r"):  # Allow resetting points
            points = []
            frame_display = first_frame.copy()
            print("Points reset. Please select 4 points again.")
        elif key == 27:  # ESC to quit selection
            points = []  # Indicate failure or cancellation
            break

    cv2.destroyWindow("Select 4 Points on Court (TL, TR, BR, BL), then press 'c'")
    return points


def select_roi_scalable(
    image, window_name="ROI Selection", max_display_width=1920, max_display_height=1080
):
    """
    Select ROI points on a high-resolution image with automatic scaling for display

    Args:
        image: Input image (can be 4K or any resolution)
        window_name: Name of the OpenCV window
        max_display_width: Maximum width for display (default: 1920 for FHD)
        max_display_height: Maximum height for display (default: 1080 for FHD)

    Returns:
        roi_points: List of 4 corner points in original image coordinates
    """

    # Get original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate scaling factor to fit within display limits
    scale_width = max_display_width / original_width
    scale_height = max_display_height / original_height
    scale_factor = min(scale_width, scale_height, 1.0)  # Don't upscale

    # Calculate display dimensions
    display_width = int(original_width * scale_factor)
    display_height = int(original_height * scale_factor)

    # Resize image for display
    display_image = cv2.resize(image, (display_width, display_height))

    print(f"Original image size: {original_width}x{original_height}")
    print(f"Display size: {display_width}x{display_height}")
    print(f"Scale factor: {scale_factor:.3f}")
    print("\nInstructions:")
    print("- Click 4 corner points in clockwise order")
    print("- Drag points to adjust their position")
    print("- Press 'r' to reset selection")
    print("- Press 'q' or ESC to quit")
    print("- Press ENTER or 'c' to confirm selection")

    # Initialize variables
    roi_points = []
    dragging_point_idx = None
    drag_threshold = 10  # pixels threshold for detecting point proximity

    def redraw_roi():
        """Redraw the ROI visualization"""
        nonlocal temp_image
        temp_image = display_image.copy()
        
        # Draw lines between points
        if len(roi_points) > 1:
            for i in range(len(roi_points)):
                next_i = (i + 1) % len(roi_points) if len(roi_points) == 4 else i + 1
                if next_i < len(roi_points):
                    cv2.line(
                        temp_image,
                        tuple(roi_points[i]),
                        tuple(roi_points[next_i]),
                        (0, 255, 0),
                        2,
                    )
        
        # Fill polygon if 4 points selected
        if len(roi_points) == 4:
            overlay = temp_image.copy()
            cv2.fillPoly(
                overlay, [np.array(roi_points, dtype=np.int32)], (0, 255, 0)
            )
            cv2.addWeighted(temp_image, 0.7, overlay, 0.3, 0, temp_image)
        
        # Draw points and labels
        for idx, point in enumerate(roi_points):
            cv2.circle(temp_image, tuple(point), 5, (0, 255, 0), -1)
            cv2.putText(
                temp_image,
                f"{idx + 1}",
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    def find_nearest_point(x, y):
        """Find the index of the nearest point within threshold"""
        for idx, point in enumerate(roi_points):
            dist = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if dist <= drag_threshold:
                return idx
        return None

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_points, dragging_point_idx

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near an existing point
            nearest_idx = find_nearest_point(x, y)
            if nearest_idx is not None:
                # Start dragging this point
                dragging_point_idx = nearest_idx
            elif len(roi_points) < 4:
                # Add new point
                roi_points.append([x, y])
                redraw_roi()
                
                if len(roi_points) == 4:
                    print(f"Selected 4 points. Press 'c' or ENTER to confirm or 'r' to reset.")

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_point_idx is not None:
                # Update the dragging point's position
                roi_points[dragging_point_idx] = [x, y]
                redraw_roi()

        elif event == cv2.EVENT_LBUTTONUP:
            # Stop dragging
            if dragging_point_idx is not None:
                dragging_point_idx = None
                redraw_roi()

    # Initialize temp_image
    temp_image = display_image.copy()
    
    # Create window and set mouse callback
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, temp_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):  # Reset selection
            roi_points = []
            dragging_point_idx = None
            temp_image = display_image.copy()
            print("Selection reset. Click 4 corner points.")

        elif key == ord("q") or key == 27:  # Quit (ESC)
            print("Selection cancelled.")
            cv2.destroyAllWindows()
            return []

        elif key == 13 or key == ord("c"):  # ENTER or 'c'
            if len(roi_points) == 4:
                break
            else:
                print(f"Please select {4 - len(roi_points)} more points.")

    cv2.destroyAllWindows()

    # Convert display coordinates back to original image coordinates
    original_roi_points = []
    for point in roi_points:
        original_x = int(point[0] / scale_factor)
        original_y = int(point[1] / scale_factor)
        original_roi_points.append([original_x, original_y])

    print(f"Selected ROI points (original coordinates): {original_roi_points}")

    return original_roi_points


#def extract_subject_region(image):
#    """
#    Extracts the subject region from a PNG image with an alpha channel and saves it to a new file.
#    Args:
#        image_path (str): Path to the input PNG image.
#        output_path (str): Path to save the cropped image.
#    """
#    if image.shape[2] == 4:
#        alpha = image[:, :, 3]
#        coords = cv2.findNonZero(alpha)
#        x, y, w, h = cv2.boundingRect(coords)
#        # cropped = image[y:y+h, x:x+w]
#        return [x, y, w, h]
#    else:
#        print("Image does not have an alpha channel.")
#        return [
#            0,
#            0,
#            image.shape[1],
#            image.shape[0],
#        ]  # Return full image dimensions if no alpha channel


def draw_keypoints(img, keypoints, color=(0, 255, 0), radius=3):
    """
    Draws keypoints on the image.

    Args:
        img (numpy.ndarray): Input image.
        keypoints (list): List of keypoints as (x, y) tuples.
        color (tuple): Color for the keypoints in BGR format.
        radius (int): Radius of the keypoint circles.

    Returns:
        numpy.ndarray: Image with keypoints drawn.
    """
    for x, y in keypoints:
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
    return img


def show_sam_anns(anns, ax, show_idx=True, valid_mask=None):
    if len(anns) == 0:
        return

    img_ = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    img_[:,:,3] = 0
    for i, ann in enumerate(anns):
        mask = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.85]])
        img_[mask] = color_mask
        if show_idx:
            M = cv2.moments(mask.astype(np.uint8))
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cx = np.clip(cx, 10, img_.shape[1]-10)
                cy = np.clip(cy, 10, img_.shape[0]-10)
                cv2.putText(img_, f"{i}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 1, 0, 1), 2)
            else:
                cv2.putText(img_, "-1", (i*10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 1, 0, 1), 2)
    if type(valid_mask) is not type(None):
        img_ = img_*(valid_mask)[:,:,np.newaxis]
    ax.imshow(img_) 
    return
