"""
File position validation.

Checks the spatial arrangement of colour-labelled file detections:
the correct configuration is **blue on the left, yellow on the right**.

Returns structured result data; drawing is handled by the caller.
"""

import time
from typing import Dict, List, Optional, Tuple


def validate_file_positions(
    detections: List[Dict],
) -> Tuple[bool, Optional[Dict], Optional[Dict], str, str]:
    """
    Validate that the leftmost file is blue and the rightmost is yellow.

    Parameters
    ----------
    detections : All colour-labelled detections for the current frame.
                 Each must have a ``file_color`` key and a ``bbox`` key.

    Returns
    -------
    is_valid   : True when arrangement is correct.
    left_det   : Detection dict for the leftmost file (or None if < 2 found).
    right_det  : Detection dict for the rightmost file (or None if < 2 found).
    message    : Human-readable validation result.
    level      : Alert severity string — ``"info"`` or ``"warn"``.
    """
    colored = [d for d in detections if d.get("file_color") in {"blue", "yellow"}]

    if len(colored) < 2:
        msg = "Incorrect: need at least 2 color-identified files (blue/yellow)."
        return False, None, None, msg, "warn"

    def _center_x(det: Dict) -> float:
        x1, _, x2, _ = det["bbox"]
        return (x1 + x2) * 0.5

    ordered   = sorted(colored, key=_center_x)
    left_det  = ordered[0]
    right_det = ordered[-1]

    left_color  = left_det.get("file_color", "unknown")
    right_color = right_det.get("file_color", "unknown")
    is_valid    = (left_color == "blue") and (right_color == "yellow")

    if is_valid:
        msg   = "Correct position: blue file is on the left and yellow file is on the right."
        level = "info"
    else:
        msg = (
            f"Incorrect position: left is {left_color}, right is {right_color}. "
            "Expected left blue and right yellow."
        )
        level = "warn"

    return is_valid, left_det, right_det, msg, level


def build_alert(message: str, level: str) -> Dict:
    """Create a timestamped alert dict from a validation result."""
    return {"msg": message, "level": level, "ts": time.time()}
