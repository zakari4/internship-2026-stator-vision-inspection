"""
Contour extraction for chignon segmentation masks.

Converts a binary mask into a list of detection dicts with bounding boxes.
"""

from typing import Dict, List

import cv2
import numpy as np


def extract_contour_bboxes(
    binary_mask: np.ndarray,
    class_id: int,
    class_name: str,
    min_area: float = 50.0,
) -> List[Dict]:
    """
    Find external contours in *binary_mask* and return one detection dict per contour.

    Parameters
    ----------
    binary_mask : uint8 mask (values 0 or 255).
    class_id    : Integer class identifier to embed in each detection.
    class_name  : Human-readable class label.
    min_area    : Contours smaller than this pixel area are ignored.

    Returns
    -------
    List of detection dicts with keys:
        class_id, class_name, confidence, bbox, measurements
    """
    cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Dict] = []

    for cnt in cnts:
        if cv2.contourArea(cnt) <= min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        detections.append({
            "class_id":    class_id,
            "class_name":  class_name,
            "confidence":  0.99,
            "bbox":        [float(x), float(y), float(x + bw), float(y + bh)],
            "measurements": [],
        })

    return detections
