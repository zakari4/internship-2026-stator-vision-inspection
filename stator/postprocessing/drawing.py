"""
Measurement annotation rendering — draws measurement lines and labels onto images.
"""

from typing import List

import cv2
import numpy as np


def draw_measurements_on_image(
    image: np.ndarray,
    measurements: List[dict],
    px_to_mm: float,
) -> np.ndarray:
    """Draw measurement annotations (lines, circles, text) on *image* and return it."""
    overlay = image.copy()
    color_line   = (255, 200, 0)   # Cyan-ish
    color_text   = (255, 255, 255) # White
    color_circle = (0, 200, 255)   # Orange

    for m in measurements:
        unit_val = m["value_mm"] if px_to_mm != 1.0 else m["value_px"]
        unit_str = "mm" if px_to_mm != 1.0 else "px"

        # Area / perimeter — text only, no line
        if m["type"] in ("area", "perimeter"):
            continue

        if m["type"] in ("outer_diameter", "diameter") and "center" in m:
            cx, cy = m["center"]
            r = m["radius"]
            cv2.circle(overlay, (cx, cy), r, color_circle, 1, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), 3, color_circle, -1)
            cv2.putText(overlay, f"{unit_val:.1f}{unit_str}", (cx + r + 5, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_text, 1, cv2.LINE_AA)
            continue

        pt1 = m.get("pt1")
        pt2 = m.get("pt2")
        if not pt1 or not pt2:
            continue

        is_cross   = m["type"] == "cross_diametric_opposite_distance"
        label_str  = m.get("label", "")

        if is_cross:
            if "magnet" in label_str.lower():
                line_color = (0, 255, 180)   # Mint green for magnets
            else:
                line_color = (255, 80, 200)  # Magenta for mechanical parts
            line_thickness = 2
            dot_radius = 5
            font_scale = 0.5
        else:
            line_color     = color_line
            line_thickness = 1
            dot_radius     = 3
            font_scale     = 0.4

        pt1t = tuple(int(v) for v in pt1)
        pt2t = tuple(int(v) for v in pt2)
        cv2.line(overlay, pt1t, pt2t, line_color, line_thickness, cv2.LINE_AA)
        cv2.circle(overlay, pt1t, dot_radius, line_color, -1)
        cv2.circle(overlay, pt2t, dot_radius, line_color, -1)

        mid_x = (pt1t[0] + pt2t[0]) // 2
        mid_y = (pt1t[1] + pt2t[1]) // 2
        text  = f"{unit_val:.1f}{unit_str}"
        # Shadow for readability
        cv2.putText(overlay, text, (mid_x + 6, mid_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (mid_x + 6, mid_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_color, 1, cv2.LINE_AA)

    return overlay
