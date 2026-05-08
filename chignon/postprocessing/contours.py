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
    px_to_mm: float = 1.0,
) -> List[Dict]:
    """
    Find external contours in *binary_mask* and return one detection dict per contour.

    Parameters
    ----------
    binary_mask : uint8 mask (values 0 or 255).
    class_id    : Integer class identifier to embed in each detection.
    class_name  : Human-readable class label.
    min_area    : Contours smaller than this pixel area are ignored.
    px_to_mm    : Scale factor (mm per pixel). Area is multiplied by px_to_mm².

    Returns
    -------
    List of detection dicts with keys:
        class_id, class_name, confidence, bbox, measurements
    """
    cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Dict] = []

    for cnt in cnts:
        area_px = cv2.contourArea(cnt)
        if area_px <= min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        area_mm2 = round(area_px * (px_to_mm ** 2), 2)
        detections.append({
            "class_id":    class_id,
            "class_name":  class_name,
            "confidence":  0.99,
            "bbox":        [float(x), float(y), float(x + bw), float(y + bh)],
            "measurements": [{"type": "area", "value_px": round(area_px, 1), "value_mm": area_mm2}],
        })

    return detections
