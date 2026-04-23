"""
Camera calibration settings and pixel-to-mm conversion helpers.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraSettings:
    """
    Calibration parameters for pixel-to-mm conversion.

    Supported methods:
        - camera_intrinsics : uses sensor_width, focal_length, object_distance
        - reference_label   : uses a detected label with known real-world size
        - manual            : user provides a fixed px→mm factor
        - ml_depth_midas    : uses MiDaS depth estimation to infer distance
    """

    VALID_METHODS = ("camera_intrinsics", "reference_label", "manual", "ml_depth_midas")
    VALID_DIM_TYPES = ("diameter", "width", "height")

    def __init__(self):
        # General
        self.enabled: bool = False
        self.method: str = "camera_intrinsics"

        # Measurement overlays
        self.show_edge_distances: bool = True
        self.show_center_distances: bool = True
        self.show_aligned_pair_distances: bool = False
        self.show_opposite_distances: bool = False

        # Camera-intrinsics parameters
        self.sensor_width_mm: float = 6.17
        self.focal_length_mm: float = 4.0
        self.object_distance_mm: float = 300.0

        # Reference-label parameters
        self.reference_label_name: str = ""
        self.reference_known_dimension_mm: float = 10.0
        self.reference_dimension_type: str = "diameter"  # diameter | width | height

        # Manual factor
        self.manual_px_to_mm: float = 0.1

        # Visualization
        self.show_depth_map: bool = False

    # -- px→mm helpers ---------------------------------------------------

    def pixel_to_mm_intrinsics(self, image_width_px: int) -> float:
        """Compute mm-per-pixel from camera intrinsics."""
        if self.focal_length_mm < 1e-6 or image_width_px < 1:
            return 1.0
        return (self.object_distance_mm * self.sensor_width_mm) / (
            self.focal_length_mm * image_width_px
        )

    def pixel_to_mm(self, image_width_px: int) -> float:
        """Shortcut used when method == camera_intrinsics."""
        return self.pixel_to_mm_intrinsics(image_width_px)

    @staticmethod
    def pixel_to_mm_from_reference(
        contour: np.ndarray,
        known_dimension_mm: float,
        dimension_type: str = "diameter",
    ) -> float:
        """Compute mm-per-pixel from a reference contour with a known size."""
        if contour is None or len(contour) < 5:
            return 1.0

        if dimension_type == "diameter":
            (_, _), radius = cv2.minEnclosingCircle(contour)
            pixel_dim = 2.0 * radius
        elif dimension_type == "width":
            _, _, w, _ = cv2.boundingRect(contour)
            pixel_dim = float(w)
        elif dimension_type == "height":
            _, _, _, h = cv2.boundingRect(contour)
            pixel_dim = float(h)
        else:
            pixel_dim = 0.0

        if pixel_dim < 1e-3:
            return 1.0
        return known_dimension_mm / pixel_dim

    # -- Serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "show_edge_distances": self.show_edge_distances,
            "show_center_distances": self.show_center_distances,
            "show_aligned_pair_distances": self.show_aligned_pair_distances,
            "show_opposite_distances": self.show_opposite_distances,
            "sensor_width_mm": self.sensor_width_mm,
            "focal_length_mm": self.focal_length_mm,
            "object_distance_mm": self.object_distance_mm,
            "reference_label_name": self.reference_label_name,
            "reference_known_dimension_mm": self.reference_known_dimension_mm,
            "reference_dimension_type": self.reference_dimension_type,
            "manual_px_to_mm": self.manual_px_to_mm,
            "show_depth_map": self.show_depth_map,
        }

    def update(self, data: dict):
        if "enabled" in data:
            self.enabled = bool(data["enabled"])
        if "method" in data and data["method"] in self.VALID_METHODS:
            self.method = data["method"]
        if "show_edge_distances" in data:
            self.show_edge_distances = bool(data["show_edge_distances"])
        if "show_center_distances" in data:
            self.show_center_distances = bool(data["show_center_distances"])
        if "show_aligned_pair_distances" in data:
            self.show_aligned_pair_distances = bool(data["show_aligned_pair_distances"])
        if "show_opposite_distances" in data:
            self.show_opposite_distances = bool(data["show_opposite_distances"])
        if "sensor_width_mm" in data:
            self.sensor_width_mm = float(data["sensor_width_mm"])
        if "focal_length_mm" in data:
            self.focal_length_mm = float(data["focal_length_mm"])
        if "object_distance_mm" in data:
            self.object_distance_mm = float(data["object_distance_mm"])
        if "reference_label_name" in data:
            self.reference_label_name = str(data["reference_label_name"]).strip()
        if "reference_known_dimension_mm" in data:
            self.reference_known_dimension_mm = float(data["reference_known_dimension_mm"])
        if "reference_dimension_type" in data:
            dt = data["reference_dimension_type"]
            if dt in self.VALID_DIM_TYPES:
                self.reference_dimension_type = dt
        if "manual_px_to_mm" in data:
            self.manual_px_to_mm = float(data["manual_px_to_mm"])
        if "show_depth_map" in data:
            self.show_depth_map = bool(data["show_depth_map"])
