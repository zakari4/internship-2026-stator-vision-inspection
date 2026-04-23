"""
Chignon postprocessing subpackage.

Modules
-------
mask_overlay : apply_mask_overlay — blend a coloured mask onto a frame
contours     : extract_contour_bboxes — convert a mask to detection dicts
"""

from .mask_overlay import apply_mask_overlay
from .contours import extract_contour_bboxes

__all__ = ["apply_mask_overlay", "extract_contour_bboxes"]
