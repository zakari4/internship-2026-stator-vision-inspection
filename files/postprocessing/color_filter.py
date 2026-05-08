"""
One-per-colour file filter.

Ensures that at most one blue and one yellow detection are retained after
colour labelling.  When the model produces multiple candidates of the same
colour (e.g. two blue bboxes), only the one with the largest bounding-box
area survives.  All detections whose colour is "unknown" are also discarded.

This filter is enabled by default on FileModelManager and can be toggled via
``manager.enable_one_per_color``.
"""

from typing import Dict, List


def keep_one_per_color(detections: List[Dict]) -> List[Dict]:
    """Return at most one blue and one yellow detection.

    Selection criterion: largest bounding-box area (proxy for the most
    prominent / least-fragmented detection of that colour).  Detections
    without a recognised colour ("unknown" or missing) are dropped.

    Parameters
    ----------
    detections : All colour-labelled detections for the current frame.
                 Each must have a ``file_color`` key and a ``bbox`` key.

    Returns
    -------
    Filtered list containing 0–1 blue and 0–1 yellow detection.
    """
    def _area(det: Dict) -> float:
        b = det.get("bbox", [0, 0, 0, 0])
        return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))

    blues   = [d for d in detections if d.get("file_color") == "blue"]
    yellows = [d for d in detections if d.get("file_color") == "yellow"]

    result: List[Dict] = []
    if blues:
        result.append(max(blues, key=_area))
    if yellows:
        result.append(max(yellows, key=_area))

    return result
