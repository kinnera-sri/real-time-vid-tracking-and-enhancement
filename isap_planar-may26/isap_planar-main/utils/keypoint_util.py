import numpy as np

def is_roi_out_of_bounds(roi, img_shape):
    """
    Check if the ROI is out of the bounds of the image.

    Parameters:
    roi (list): List of coordinates in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    img_shape (tuple): Shape of the image in the format (H, W).

    Returns:
    bool: True if ROI is out of bounds, False otherwise.
    """
    H, W = img_shape
    for x, y in roi:
        if 0 <= x < W and 0 <= y < H:
            return False
    return True


def roi_to_rect(roi_pts):
    xs = roi_pts[:, 0]
    ys = roi_pts[:, 1]

    x_min = int(np.floor(xs.min()))
    y_min = int(np.floor(ys.min()))
    x_max = int(np.ceil(xs.max()))
    y_max = int(np.ceil(ys.max()))

    w = x_max - x_min
    h = y_max - y_min

    return x_min, y_min, w, h