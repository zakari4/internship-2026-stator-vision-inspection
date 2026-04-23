"""
Detection filtering utilities.

- apply_top_n_filtering     : keep only the top-N detections per class
- apply_spatial_heuristic_correction : reassign magnet/mechanical_part labels
                                       based on angular position relative to circle
"""

import math
from typing import Dict, List


def apply_top_n_filtering(detections: List[Dict]) -> List[Dict]:
    """
    Limit detections to a fixed count per class, keeping the highest-confidence ones.

    Limits applied:
        circle          → 1
        magnet          → 2
        mechanical_part → 4
    """
    limits = {
        "circle": 1,
        "magnet": 2,
        "mechanical_part": 4,
    }

    grouped: Dict[str, List[Dict]] = {}
    for det in detections:
        cname = det.get("class_name", "").lower()
        grouped.setdefault(cname, []).append(det)

    filtered: List[Dict] = []
    for cname, items in grouped.items():
        sorted_items = sorted(items, key=lambda x: x.get("confidence", 0.0), reverse=True)
        limit = limits.get(cname, len(sorted_items))
        filtered.extend(sorted_items[:limit])

    return filtered


def apply_spatial_heuristic_correction(detections: List[Dict]) -> List[Dict]:
    """
    Correct misclassified magnets and mechanical parts for PyTorch models using
    their angular position relative to the main stator circle.

    Objects near cardinal angles (0°, 90°, 180°, 270° ± 25°) are re-labelled
    as magnets; all others become mechanical parts.
    """
    circle_det = None
    for det in detections:
        if det.get("class_name", "").lower() == "circle":
            if circle_det is None or det.get("confidence", 0.0) > circle_det.get("confidence", 0.0):
                circle_det = det

    if not circle_det:
        return detections

    cx1, cy1, cx2, cy2 = circle_det["bbox"]
    circle_cx = (cx1 + cx2) / 2.0
    circle_cy = (cy1 + cy2) / 2.0

    for det in detections:
        cname = det.get("class_name", "").lower()
        if cname not in {"magnet", "mechanical_part"}:
            continue

        x1, y1, x2, y2 = det["bbox"]
        det_cx = (x1 + x2) / 2.0
        det_cy = (y1 + y2) / 2.0

        angle = math.degrees(math.atan2(det_cy - circle_cy, det_cx - circle_cx)) % 360
        is_cardinal = any(abs(angle - target) <= 25 for target in [0, 90, 180, 270, 360])

        if is_cardinal:
            det["class_name"] = "magnet"
            det["class_id"] = 2
        else:
            det["class_name"] = "mechanical_part"
            det["class_id"] = 1

    return detections
