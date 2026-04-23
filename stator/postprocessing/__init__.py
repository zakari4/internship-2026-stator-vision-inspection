"""
Stator postprocessing subpackage.

Modules
-------
logging        : InferenceLogger — per-frame JSONL session logging
calibration    : CameraSettings  — pixel-to-mm calibration
filtering      : apply_top_n_filtering, apply_spatial_heuristic_correction
measurements   : all geometric measurement functions
drawing        : draw_measurements_on_image
"""

from .logging import InferenceLogger
from .calibration import CameraSettings
from .filtering import apply_top_n_filtering, apply_spatial_heuristic_correction
from .measurements import (
    normalize_measurement_family,
    compute_contour_measurements,
    compute_single_contour_measurements,
    compute_edge_center_points,
    compute_edge_center_distances,
    compute_aligned_same_class_distances,
    compute_cross_diametric_opposite_distances,
)
from .drawing import draw_measurements_on_image

__all__ = [
    "InferenceLogger",
    "CameraSettings",
    "apply_top_n_filtering",
    "apply_spatial_heuristic_correction",
    "normalize_measurement_family",
    "compute_contour_measurements",
    "compute_single_contour_measurements",
    "compute_edge_center_points",
    "compute_edge_center_distances",
    "compute_aligned_same_class_distances",
    "compute_cross_diametric_opposite_distances",
    "draw_measurements_on_image",
]
