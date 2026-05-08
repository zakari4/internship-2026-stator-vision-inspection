"""
Files domain postprocessing subpackage.

Modules
-------
color_detection     : detect_file_colors — HSV-based blue/yellow detection
bbox_splitting      : split_dual_color_bbox — split dual-colour bboxes
position_validation : validate_file_positions, build_alert — arrangement check
color_filter        : keep_one_per_color — enforce exactly one blue + one yellow
"""

from .color_detection import detect_file_colors
from .bbox_splitting import split_dual_color_bbox
from .position_validation import validate_file_positions, build_alert
from .color_filter import keep_one_per_color

__all__ = [
    "detect_file_colors",
    "split_dual_color_bbox",
    "validate_file_positions",
    "build_alert",
    "keep_one_per_color",
]
