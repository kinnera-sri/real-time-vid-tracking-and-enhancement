import cv2
import numpy as np
import time
import logging
from .keypoint_util import roi_to_rect

"""
self.mask: Ad logo mask
self.frame: Current frame
self.current_mask: Current mask of the frame without the ad roi
self.I_ins_warped: Warped logo in the ad roi shape

"""


def alpha_blend_feathered(
    frame,
    ad_mask,
    I_ins_warped,
    alpha_mask,
    feather_blend_px=21,
    ROI_empty=None,
    roi_points=None,
):
    """
    Blends the images using alpha blending with a feathered mask.
    Parameters:
        frame: np.ndarray
            The current frame
        ad_mask: np.ndarray
            The mask of the ad roi
        I_ins_warped: np.ndarray
            The warped logo in the ad roi shape
        feather_blend_px: int
            The amount of feathering to apply to the mask
        roi_points: list
            Optional ROI points to process only a specific region
    Returns:
        blended_image: np.ndarray
            The blended image
    """

    # Store original frame for patching later if roi_points provided
    if roi_points is not None:
        original_frame = frame.copy()
        original_height, original_width = frame.shape[:2]

        # Get bounding box of roi points
        x_coords = [p[0] for p in roi_points]
        y_coords = [p[1] for p in roi_points]
        x_min, x_max = int(max(min(x_coords) - feather_blend_px * 2, 0)), int(
            min(max(x_coords) + feather_blend_px * 2, original_width)
        )
        y_min, y_max = int(max(min(y_coords) - feather_blend_px * 2, 0)), int(
            min(max(y_coords) + feather_blend_px * 2, original_height)
        )

        # Slice the region of interest
        frame = frame[y_min:y_max, x_min:x_max]
        ad_mask = ad_mask[y_min:y_max, x_min:x_max]
        I_ins_warped = I_ins_warped[y_min:y_max, x_min:x_max]
        if alpha_mask is not None:
            alpha_mask = alpha_mask[y_min:y_max, x_min:x_max]

    if ROI_empty:
        # Convert frame and I_ins_warped to hsv color space
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        I_ins_warped_hsv = cv2.cvtColor(I_ins_warped, cv2.COLOR_BGR2HSV)

        # FASTEST METHOD - Use cv2.addWeighted with mask
        frame_v = frame_hsv[:, :, 2]
        warped_v = I_ins_warped_hsv[:, :, 2]

        # Blend V channels using optimized cv2 function
        blended_v = cv2.addWeighted(frame_v, 0.50, warped_v, 0.50, 0, dtype=cv2.CV_8U)

        # Apply only where warped_v > 0
        mask_nonzero = warped_v > 0
        I_ins_warped_hsv[:, :, 2] = np.where(mask_nonzero, blended_v, warped_v)
        
        I_ins_warped = cv2.cvtColor(I_ins_warped_hsv, cv2.COLOR_HSV2BGR)

    # Ensure feather_blend_px is odd for GaussianBlur
    if feather_blend_px % 2 == 0:
        feather_blend_px += 1

    feathered_mask = cv2.GaussianBlur(ad_mask, (feather_blend_px, feather_blend_px), 0)
    feathered_mask = cv2.GaussianBlur(
        feathered_mask, (feather_blend_px, feather_blend_px), 0
    )

    if alpha_mask is not None:
        # If an alpha mask is provided, use it to feather the ad_mask
        feathered_mask = cv2.bitwise_and(feathered_mask, alpha_mask)

    # Normalize the feathered mask to range [0, 1] to act as alpha
    alpha_mask_float = feathered_mask.astype(np.float32) / 255.0

    # Expand to 3 channels to multiply with color images
    alpha_mask_3channel = cv2.cvtColor(alpha_mask_float, cv2.COLOR_GRAY2BGR)

    # # Alpha Blend
    # # Convert images to float for blending to avoid clipping issues
    inv_alpha_mask_3channel = 1.0 - alpha_mask_3channel
    frame_weighted = cv2.multiply(frame, inv_alpha_mask_3channel, dtype=cv2.CV_8U)
    warped_weighted = cv2.multiply(I_ins_warped, alpha_mask_3channel, dtype=cv2.CV_8U)
    blended_image_uint8 = cv2.add(frame_weighted, warped_weighted)

    # Patch back to original frame if roi_points provided
    if roi_points is not None:
        original_frame[y_min:y_max, x_min:x_max] = blended_image_uint8
        return original_frame

    return blended_image_uint8


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1 / 2.4)) - 0.055)


def retinex_blend_feathered(
    frame,
    ad_mask,
    I_ins_warped,
    alpha_mask,
    lighting_ref,
    feather_blend_px=21,
    roi_points=None,
    S_prev=None,
):
    """
    Blends the images using retinex blending with a feathered mask.
    Parameters:
        frame: np.ndarray
            The current frame
        ad_mask: np.ndarray
            The mask of the ad roi
        I_ins_warped: np.ndarray
            The warped logo in the ad roi shape
        alpha_mask: np.ndarray
            The alpha mask of the warped logo
        lighting_ref: np.ndarray
            The reference lighting image
        feather_blend_px: int
            The amount of feathering to apply to the mask
        ROI_empty: bool
            Whether the ROI is empty or not
        roi_points: list
            Optional ROI points to process only a specific region
    Returns:
        blended_image: np.ndarray
            The blended image
    """

    # Store original frame for patching later if roi_points provided
    if roi_points is not None:
        original_frame = frame.copy()
        original_height, original_width = frame.shape[:2]

        # Get bounding box of roi points
        x_coords = [p[0] for p in roi_points]
        y_coords = [p[1] for p in roi_points]
        x_min, x_max = int(max(min(x_coords) - feather_blend_px * 2, 0)), int(
            min(max(x_coords) + feather_blend_px * 2, original_width)
        )
        y_min, y_max = int(max(min(y_coords) - feather_blend_px * 2, 0)), int(
            min(max(y_coords) + feather_blend_px * 2, original_height)
        )

        # Slice the region of interest
        frame = frame[y_min:y_max, x_min:x_max]
        ad_mask = ad_mask[y_min:y_max, x_min:x_max]
        I_ins_warped = I_ins_warped[y_min:y_max, x_min:x_max]
        if alpha_mask is not None:
            alpha_mask = alpha_mask[y_min:y_max, x_min:x_max]

    ad_rgb = cv2.cvtColor(I_ins_warped, cv2.COLOR_BGR2RGB)
    ad_rgb = ad_rgb.astype(np.float32) / 255.0
    ad_lin = srgb_to_linear(ad_rgb)
    # I_ad = np.linalg.norm(ad_lin, axis=2)
    I_ad = I_ad = (
        0.2126 * ad_lin[..., 0] + 0.7152 * ad_lin[..., 1] + 0.0722 * ad_lin[..., 2]
    )
    I_ad = I_ad / (np.mean(I_ad) + 1e-6)

    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV).astype(np.float32)
    Y = frame_yuv[:, :, 0] / 255.0

    S = Y / (lighting_ref + 1e-6)
    S = np.clip(S, 0.5, 1.2)

    # p_low, p_high = np.percentile(S, [5, 95])
    # S = np.clip(S, p_low, p_high)

    # S = cv2.GaussianBlur(S, (0, 0), 5)  # 7
    S = cv2.GaussianBlur(S, (7, 7), 0)

    if S_prev is not None:
        S = 0.9 * S_prev + 0.1 * S

    # relight ad in **linear RGB intensity** for current gif frame
    I_final = I_ad * lighting_ref * S
    I_final = np.clip(I_final, 0.0, None)

    scale = I_final / (I_ad + 1e-6)
    ad_lin_lit = ad_lin * scale[..., None]

    ad_rgb_out = linear_to_srgb(np.clip(ad_lin_lit, 0, 1))
    ad_bgr_out = (ad_rgb_out[..., ::-1] * 255.0).astype(np.uint8)

    # Ensure feather_blend_px is odd for GaussianBlur
    if feather_blend_px % 2 == 0:
        feather_blend_px += 1

    feathered_mask = cv2.GaussianBlur(ad_mask, (feather_blend_px, feather_blend_px), 0)
    feathered_mask = cv2.GaussianBlur(
        feathered_mask, (feather_blend_px, feather_blend_px), 0
    )

    if alpha_mask is not None:
        # If an alpha mask is provided, use it to feather the ad_mask
        feathered_mask = cv2.bitwise_and(feathered_mask, alpha_mask)

    # Normalize the feathered mask to range [0, 1] to act as alpha
    alpha_mask_float = feathered_mask.astype(np.float32) / 255.0

    # Expand to 3 channels to multiply with color images
    alpha_mask_3channel = cv2.cvtColor(alpha_mask_float, cv2.COLOR_GRAY2BGR)

    # # Alpha Blend
    # # Convert images to float for blending to avoid clipping issues
    inv_alpha_mask_3channel = 1.0 - alpha_mask_3channel
    frame_weighted = cv2.multiply(frame, inv_alpha_mask_3channel, dtype=cv2.CV_8U)
    warped_weighted = cv2.multiply(ad_bgr_out, alpha_mask_3channel, dtype=cv2.CV_8U)
    blended_image_uint8 = cv2.add(frame_weighted, warped_weighted)

    # Patch back to original frame if roi_points provided
    if roi_points is not None:
        original_frame[y_min:y_max, x_min:x_max] = blended_image_uint8
        return original_frame, S

    return blended_image_uint8, S


def estimate_lighting(cap, roi, feather_blend_px, max_frames=50):
    """
    Estimate the reference lighting map from video frames.
    This function samples frames from a video capture object, extracts the luminance (Y channel)
    from a specified region of interest (ROI), and computes a reference lighting map by taking
    the 90th percentile across frames and applying Gaussian smoothing.
    Args:
        cap: OpenCV VideoCapture object containing the video to analyze.
        roi: Region of interest as a list of points defining the ROI polygon. If None,
             the entire frame is used.
        feather_blend_px (int): Number of pixels to extend the ROI bounding box on each side
                               for feathering/blending purposes.
        max_frames (int, optional): Maximum number of frames to sample for lighting estimation.
                                   Defaults to 50.
    Returns:
        numpy.ndarray: A 2D array representing the reference lighting map (L_ref) with values
                      in the range [0, 1]. The map has been smoothed with a Gaussian filter
                      (sigma=15).
    Note:
        - The function assumes roi_points, original_width, and original_height are defined
          in the parent scope if roi is not None.
        - The function releases the video capture object after processing.
        - The Y channel is extracted from YUV color space to represent luminance.
        - The 90th percentile is used to capture bright reference values while being robust
          to outliers.
    """

    Y_frames = []

    while len(Y_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if roi is not None:
            original_height, original_width = frame.shape[:2]

            # Get bounding box of roi points
            x_coords = [p[0] for p in roi]
            y_coords = [p[1] for p in roi]
            x_min, x_max = int(max(min(x_coords) - feather_blend_px * 2, 0)), int(
                min(max(x_coords) + feather_blend_px * 2, original_width)
            )
            y_min, y_max = int(max(min(y_coords) - feather_blend_px * 2, 0)), int(
                min(max(y_coords) + feather_blend_px * 2, original_height)
            )
            frame_roi = frame[y_min:y_max, x_min:x_max]
        else:
            frame_roi = frame

        # Convert to YUV color space and extract Y channel
        Y = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2YUV)[:, :, 0]
        Y = Y.astype(np.float32) / 255.0
        Y_frames.append(Y)

    # cap.release()
    Y_stack = np.stack(Y_frames, axis=0)
    L_ref = np.percentile(Y_stack, 90, axis=0)
    L_ref = cv2.GaussianBlur(L_ref, (0, 0), 3)
    # L_ref = cv2.GaussianBlur(L_ref, (5, 5), 0)

    return L_ref


def estimate_lighting_knr(frames, roi, feather_blend_px):
    """
    Estimate the reference lighting map from video frames.
    This function samples frames from a video capture object, extracts the luminance (Y channel)
    from a specified region of interest (ROI), and computes a reference lighting map by taking
    the 90th percentile across frames and applying Gaussian smoothing.
    Args:
        cap: OpenCV VideoCapture object containing the video to analyze.
        roi: Region of interest as a list of points defining the ROI polygon. If None,
             the entire frame is used.
        feather_blend_px (int): Number of pixels to extend the ROI bounding box on each side
                               for feathering/blending purposes.
        max_frames (int, optional): Maximum number of frames to sample for lighting estimation.
                                   Defaults to 50.
    Returns:
        numpy.ndarray: A 2D array representing the reference lighting map (L_ref) with values
                      in the range [0, 1]. The map has been smoothed with a Gaussian filter
                      (sigma=15).
    Note:
        - The function assumes roi_points, original_width, and original_height are defined
          in the parent scope if roi is not None.
        - The function releases the video capture object after processing.
        - The Y channel is extracted from YUV color space to represent luminance.
        - The 90th percentile is used to capture bright reference values while being robust
          to outliers.
    """

    Y_frames = []

    for frame in frames:
        if roi is not None:
            original_height, original_width = frame.shape[:2]

            # Get bounding box of roi points
            x_coords = [p[0] for p in roi]
            y_coords = [p[1] for p in roi]
            x_min, x_max = int(max(min(x_coords) - feather_blend_px * 2, 0)), int(
                min(max(x_coords) + feather_blend_px * 2, original_width)
            )
            y_min, y_max = int(max(min(y_coords) - feather_blend_px * 2, 0)), int(
                min(max(y_coords) + feather_blend_px * 2, original_height)
            )
            frame_roi = frame[y_min:y_max, x_min:x_max]
        else:
            frame_roi = frame

        # Convert to YUV color space and extract Y channel
        Y = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2YUV)[:, :, 0]
        Y = Y.astype(np.float32) / 255.0
        Y_frames.append(Y)

    # cap.release()
    Y_stack = np.stack(Y_frames, axis=0)
    L_ref = np.percentile(Y_stack, 90, axis=0)
    L_ref = cv2.GaussianBlur(L_ref, (0, 0), 3)
    # L_ref = cv2.GaussianBlur(L_ref, (5, 5), 0)

    return L_ref