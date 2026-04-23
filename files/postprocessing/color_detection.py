"""
HSV-based color detection for the file domain.

Identifies blue and yellow regions within an HSV region-of-interest.

HSV ranges used:
    Blue   — hue [100, 130]
    Yellow — hue [20,  40]
Both require saturation and value ≥ 50.
"""

from typing import Tuple

import cv2
import numpy as np

# HSV bounds (lower, upper) for each file colour
_BLUE_L   = np.array([100, 50, 50])
_BLUE_U   = np.array([130, 255, 255])
_YELLOW_L = np.array([20,  50, 50])
_YELLOW_U = np.array([40,  255, 255])


def detect_file_colors(
    roi_hsv: np.ndarray,
    threshold: float = 0.08,
    morph_kernel_size: int = 3,
) -> Tuple[bool, bool, float, float, np.ndarray, np.ndarray]:
    """
    Detect blue and yellow file colours in an HSV region-of-interest.

    Parameters
    ----------
    roi_hsv           : Cropped HSV image for one detection bbox.
    threshold         : Minimum pixel ratio for a colour to be considered present.
    morph_kernel_size : Side length of the morphological open kernel used to
                        reduce isolated noise before counting.

    Returns
    -------
    has_blue   : True when blue pixel ratio ≥ *threshold*.
    has_yellow : True when yellow pixel ratio ≥ *threshold*.
    blue_ratio : Fraction of ROI pixels that are blue.
    yellow_ratio: Fraction of ROI pixels that are yellow.
    blue_mask  : uint8 binary mask of blue pixels (same size as roi_hsv[:2]).
    yellow_mask: uint8 binary mask of yellow pixels.
    """
    total = float(roi_hsv.shape[0] * roi_hsv.shape[1])
    zeros = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)

    if total <= 0:
        return False, False, 0.0, 0.0, zeros, zeros

    blue_mask   = cv2.inRange(roi_hsv, _BLUE_L,   _BLUE_U)
    yellow_mask = cv2.inRange(roi_hsv, _YELLOW_L, _YELLOW_U)

    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    blue_mask   = cv2.morphologyEx(blue_mask,   cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    blue_ratio   = float(cv2.countNonZero(blue_mask))   / total
    yellow_ratio = float(cv2.countNonZero(yellow_mask)) / total

    has_blue   = blue_ratio   >= threshold
    has_yellow = yellow_ratio >= threshold

    return has_blue, has_yellow, blue_ratio, yellow_ratio, blue_mask, yellow_mask
