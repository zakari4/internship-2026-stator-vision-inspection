"""
Bounding-box splitting for dual-colour file detections.

When a single detection bbox contains both a blue and a yellow file region,
this module splits it into two smaller boxes — one per colour — derived from
the largest contour of each colour mask within that box.
"""

from typing import Dict, List

import cv2
import numpy as np


def split_dual_color_bbox(
    det: Dict,
    x1: int,
    y1: int,
    blue_mask: np.ndarray,
    yellow_mask: np.ndarray,
    blue_ratio: float,
    yellow_ratio: float,
    min_area: float = 40.0,
) -> List[Dict]:
    """
    Split a single detection into two colour-labelled detections.

    The input masks are ROI-local (cropped to the original bbox). Returned
    detections use absolute image coordinates.

    Parameters
    ----------
    det          : Original detection dict to clone for each colour.
    x1, y1       : Top-left corner of the original bbox in image coordinates.
    blue_mask    : Binary mask of blue pixels inside the ROI.
    yellow_mask  : Binary mask of yellow pixels inside the ROI.
    blue_ratio   : Fraction of ROI that is blue (stored in colour_scores).
    yellow_ratio : Fraction of ROI that is yellow (stored in colour_scores).
    min_area     : Minimum contour area to accept a colour region.

    Returns
    -------
    List of split detection dicts (0, 1 or 2 entries).
    """
    split_dets: List[Dict] = []

    for color_name, color_mask, ratio in (
        ("blue",   blue_mask,   blue_ratio),
        ("yellow", yellow_mask, yellow_ratio),
    ):
        cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        cnt  = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        sx, sy, sw, sh = cv2.boundingRect(cnt)
        abs_x1 = x1 + sx
        abs_y1 = y1 + sy
        abs_x2 = abs_x1 + sw
        abs_y2 = abs_y1 + sh

        split_det = dict(det)
        split_det["bbox"]         = [float(abs_x1), float(abs_y1), float(abs_x2), float(abs_y2)]
        split_det["file_color"]   = color_name
        split_det["color_ratio"]  = round(float(ratio), 3)
        split_det["color_scores"] = {
            "blue":   round(float(blue_ratio),   3),
            "yellow": round(float(yellow_ratio), 3),
        }
        split_dets.append(split_det)

    return split_dets
