"""
Backward-compatibility shim.

The stator inference engine has moved to:
    stator/inference.py          — ModelManager
    stator/postprocessing/       — one file per postprocessing concern

All public symbols are re-exported from the new location so that any existing
code importing from this module continues to work without modification.
"""

from stator.inference import ModelManager  # noqa: F401
from stator.postprocessing import (        # noqa: F401
    InferenceLogger,
    CameraSettings,
    apply_top_n_filtering,
    apply_spatial_heuristic_correction,
    normalize_measurement_family,
    compute_contour_measurements,
    compute_single_contour_measurements,
    compute_edge_center_points,
    compute_edge_center_distances,
    compute_aligned_same_class_distances,
    compute_cross_diametric_opposite_distances,
    draw_measurements_on_image,
)
