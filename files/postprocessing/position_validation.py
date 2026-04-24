"""
File position validation.

Automatically determines the dominant axis (X or Y) from the spatial
spread between the blue and yellow file centroids, then validates the
colour order along that axis:

  X-axis — blue on the left,  yellow on the right
  Y-axis — blue on the top,   yellow on the bottom

No manual axis configuration is needed; the system adapts at runtime
as the physical arrangement changes.
"""

import time
from typing import Dict, List, Optional, Tuple


def _centroid(det: Dict) -> Tuple[float, float]:
    x1, y1, x2, y2 = det["bbox"]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def validate_file_positions(
    detections: List[Dict],
) -> Tuple[bool, Optional[Dict], Optional[Dict], str, str]:
    """
    Validate file colour-order, auto-detecting whether the arrangement
    is horizontal (X-axis) or vertical (Y-axis).

    The axis is chosen by comparing the absolute horizontal spread to
    the absolute vertical spread between the blue and yellow group
    centroids. Whichever is larger determines the active axis.

    Parameters
    ----------
    detections : All colour-labelled detections for the current frame.
                 Each must have a ``file_color`` key and a ``bbox`` key.

    Returns
    -------
    is_valid   : True when the arrangement matches the expected order.
    first_det  : Detection closest to the "start" of the detected axis.
    second_det : Detection at the "end" of the detected axis.
    message    : Human-readable result including the detected axis.
    level      : ``"info"`` (correct) or ``"warn"`` (incorrect / ambiguous).
    """
    colored = [d for d in detections if d.get("file_color") in {"blue", "yellow"}]

    if len(colored) < 2:
        msg = "Incorrect: need at least 2 color-identified files (blue/yellow)."
        return False, None, None, msg, "warn"

    # Aggregate centroids per colour group
    blues   = [d for d in colored if d.get("file_color") == "blue"]
    yellows = [d for d in colored if d.get("file_color") == "yellow"]

    if not blues or not yellows:
        msg = "Incorrect: need at least one blue and one yellow file."
        return False, None, None, msg, "warn"

    def _mean_centroid(dets: List[Dict]) -> Tuple[float, float]:
        xs = [_centroid(d)[0] for d in dets]
        ys = [_centroid(d)[1] for d in dets]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    bx, by = _mean_centroid(blues)
    yx, yy = _mean_centroid(yellows)

    dx = abs(bx - yx)   # horizontal spread between colour groups
    dy = abs(by - yy)   # vertical spread between colour groups

    # Pick the axis with the larger spread
    use_y_axis = dy > dx

    if use_y_axis:
        # Sort all coloured detections by vertical centroid
        ordered    = sorted(colored, key=lambda d: _centroid(d)[1])
        first_det  = ordered[0]    # topmost
        second_det = ordered[-1]   # bottommost
        top_color  = first_det.get("file_color", "unknown")
        bot_color  = second_det.get("file_color", "unknown")
        is_valid   = (top_color == "blue") and (bot_color == "yellow")

        axis_label = "Y-axis"
        if is_valid:
            msg   = f"Correct position [{axis_label}]: blue on top, yellow on bottom."
            level = "info"
        else:
            msg = (
                f"Incorrect position [{axis_label}]: top is {top_color}, "
                f"bottom is {bot_color}. Expected blue on top, yellow on bottom."
            )
            level = "warn"
    else:
        # Sort all coloured detections by horizontal centroid
        ordered    = sorted(colored, key=lambda d: _centroid(d)[0])
        first_det  = ordered[0]    # leftmost
        second_det = ordered[-1]   # rightmost
        left_color  = first_det.get("file_color", "unknown")
        right_color = second_det.get("file_color", "unknown")
        is_valid    = (left_color == "blue") and (right_color == "yellow")

        axis_label = "X-axis"
        if is_valid:
            msg   = f"Correct position [{axis_label}]: blue on left, yellow on right."
            level = "info"
        else:
            msg = (
                f"Incorrect position [{axis_label}]: left is {left_color}, "
                f"right is {right_color}. Expected blue on left, yellow on right."
            )
            level = "warn"

    return is_valid, first_det, second_det, msg, level


def build_alert(message: str, level: str) -> Dict:
    """Create a timestamped alert dict from a validation result."""
    return {"msg": message, "level": level, "ts": time.time()}
