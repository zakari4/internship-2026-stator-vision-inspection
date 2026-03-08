"""
Geometric measurement methods for industrial inspection.
Includes distance computation, diameter estimation, and calibration.
"""

import time
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt

from src.config import config
from src.utils.contour import GeometryFitter, FittedGeometry


@dataclass
class MeasurementResult:
    """
    Result of a geometric measurement operation.

    Encapsulates the raw pixel value, the converted metric value (if calibrated),
    estimated confidence, and any algorithm-specific metadata.

    :param measurement_type: String identifier of the measurement performed.
    :param value_pixels: The calculated dimension in pixels.
    :param value_mm: The dimension converted to millimeters (None if not calibrated).
    :param confidence: Metric indicating measurement reliability (0.0 to 1.0).
    :param additional_info: Dictionary containing metadata or intermediate results.
    """
    measurement_type: str
    value_pixels: float
    value_mm: Optional[float]
    confidence: float
    additional_info: Dict


class MeasurementComputer:
    """
    Computes geometric measurements on contours and masks.

    This class provides high-level methods for calculating distances,
    diameters, and thicknesses, supporting both pixel-level and metric-level
    outputs based on the current calibration factor.
    """
    
    def __init__(
        self,
        pixel_to_mm: float = None,
        use_subpixel: bool = None
    ):
        """
        Initialize the measurement computer.

        :param pixel_to_mm: Conversion factor (default: from config).
        :param use_subpixel: Enable sub-pixel refinement (default: from config).
        """
        cfg = config.measurement
        
        self.pixel_to_mm = pixel_to_mm or cfg.pixel_to_mm
        self.use_subpixel = use_subpixel if use_subpixel is not None else cfg.use_subpixel
        self.geometry_fitter = GeometryFitter()
    
    def to_mm(self, value_pixels: float) -> float:
        """
        Convert pixel measurement to millimeters.

        :param value_pixels: Dimension in pixels.
        :return: Dimension in millimeters.
        """
        return value_pixels * self.pixel_to_mm
    
    # =========================================================================
    # Distance Measurements
    # =========================================================================
    
    def min_distance_between_contours(
        self,
        contour1: np.ndarray,
        contour2: np.ndarray
    ) -> MeasurementResult:
        """
        Compute minimum distance between two contours.
        
        :param contour1: First contour array.
        :param contour2: Second contour array.
        :return: MeasurementResult with minimum distance and closest point metadata.
        """
        # Reshape contours if needed
        pts1 = contour1.reshape(-1, 2).astype(np.float32)
        pts2 = contour2.reshape(-1, 2).astype(np.float32)
        
        # Compute pairwise distances
        distances = cdist(pts1, pts2, 'euclidean')
        
        # Find minimum
        min_dist = np.min(distances)
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        
        # Get closest points
        closest_pt1 = pts1[min_idx[0]]
        closest_pt2 = pts2[min_idx[1]]
        
        return MeasurementResult(
            measurement_type='min_distance_contours',
            value_pixels=float(min_dist),
            value_mm=self.to_mm(min_dist),
            confidence=1.0,
            additional_info={
                'closest_point_1': closest_pt1.tolist(),
                'closest_point_2': closest_pt2.tolist()
            }
        )
    
    def max_distance_between_contours(
        self,
        contour1: np.ndarray,
        contour2: np.ndarray
    ) -> MeasurementResult:
        """
        Compute maximum distance between two contours (Hausdorff-like).
        
        :param contour1: First contour array.
        :param contour2: Second contour array.
        :return: MeasurementResult with maximum distance.
        """
        pts1 = contour1.reshape(-1, 2).astype(np.float32)
        pts2 = contour2.reshape(-1, 2).astype(np.float32)
        
        distances = cdist(pts1, pts2, 'euclidean')
        max_dist = np.max(distances)
        
        return MeasurementResult(
            measurement_type='max_distance_contours',
            value_pixels=float(max_dist),
            value_mm=self.to_mm(max_dist),
            confidence=1.0,
            additional_info={}
        )
    
    def distance_between_parallel_lines(
        self,
        line1: FittedGeometry,
        line2: FittedGeometry
    ) -> MeasurementResult:
        """
        Compute distance between two approximately parallel lines.
        
        :param line1: First fitted line geometry.
        :param line2: Second fitted line geometry.
        :return: MeasurementResult with perpendicular distance.
        """
        if line1 is None or line2 is None:
            return None
        
        if line1.geometry_type != 'line' or line2.geometry_type != 'line':
            return None
        
        # Get line parameters
        pt1 = line1.parameters['point']
        dir1 = line1.parameters['direction']
        pt2 = line2.parameters['point']
        dir2 = line2.parameters['direction']
        
        # Check if approximately parallel (dot product close to 1 or -1)
        dot = abs(np.dot(dir1, dir2))
        if dot < 0.95:
            confidence = dot
        else:
            confidence = 1.0
        
        # Compute perpendicular distance
        # Vector from pt1 to pt2
        v = pt2 - pt1
        
        # Project onto perpendicular
        perp = np.array([-dir1[1], dir1[0]])
        distance = abs(np.dot(v, perp))
        
        return MeasurementResult(
            measurement_type='parallel_line_distance',
            value_pixels=float(distance),
            value_mm=self.to_mm(distance),
            confidence=confidence,
            additional_info={
                'parallelism': dot
            }
        )
    
    def gap_measurement(
        self,
        mask: np.ndarray,
        direction: str = 'horizontal'
    ) -> MeasurementResult:
        """
        Measure the largest gap/clearance in a binary mask.
        
        :param mask: Binary mask array.
        :param direction: 'horizontal' or 'vertical' (experimental).
        :return: MeasurementResult with maximum inscribed gap diameter.
        """
        # Invert mask to measure gaps
        inverted = 255 - mask
        
        # Distance transform
        dist_transform = distance_transform_edt(inverted)
        
        # Find maximum gap
        max_gap = np.max(dist_transform) * 2  # Diameter of inscribed circle
        
        # Find location of maximum
        max_loc = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
        
        return MeasurementResult(
            measurement_type='gap_measurement',
            value_pixels=float(max_gap),
            value_mm=self.to_mm(max_gap),
            confidence=0.9,
            additional_info={
                'max_location': max_loc,
                'direction': direction
            }
        )
    
    # =========================================================================
    # Diameter Measurements
    # =========================================================================
    
    def inner_diameter(
        self,
        mask: np.ndarray
    ) -> MeasurementResult:
        """
        Estimate the inner diameter of a ring-like object using distance transform.
        
        :param mask: Binary mask of the object.
        :return: MeasurementResult with inner diameter.
        """
        # Distance transform to find maximum inscribed circle
        dist_transform = distance_transform_edt(mask > 0)
        
        # Find center (maximum distance)
        max_dist = np.max(dist_transform)
        max_loc = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
        
        # Inner diameter is 2x max distance
        inner_diam = 2 * max_dist
        
        return MeasurementResult(
            measurement_type='inner_diameter',
            value_pixels=float(inner_diam),
            value_mm=self.to_mm(inner_diam),
            confidence=0.85,
            additional_info={
                'center': max_loc,
                'inscribed_radius': max_dist
            }
        )
    
    def outer_diameter(
        self,
        contour: np.ndarray
    ) -> MeasurementResult:
        """
        Compute the outer diameter using a minimum enclosing circle.
        
        :param contour: Object contour array.
        :return: MeasurementResult with outer diameter and center metadata.
        """
        (x, y), radius = cv2.minEnclosingCircle(contour)
        outer_diam = 2 * radius
        
        return MeasurementResult(
            measurement_type='outer_diameter',
            value_pixels=float(outer_diam),
            value_mm=self.to_mm(outer_diam),
            confidence=0.9,
            additional_info={
                'center': (x, y),
                'radius': radius
            }
        )
    
    def fitted_circle_diameter(
        self,
        contour: np.ndarray
    ) -> MeasurementResult:
        """
        Compute the diameter using a best-fit circle algorithm (least squares).
        
        :param contour: Object contour array.
        :return: MeasurementResult with fitted diameter and residual error.
        """
        fitted = self.geometry_fitter.fit_circle(contour)
        
        if fitted is None:
            return None
        
        diameter = 2 * fitted.parameters['radius']
        
        return MeasurementResult(
            measurement_type='fitted_circle_diameter',
            value_pixels=float(diameter),
            value_mm=self.to_mm(diameter),
            confidence=max(0, 1 - fitted.residual_error),
            additional_info={
                'center': fitted.parameters['center'],
                'radius': fitted.parameters['radius'],
                'fit_residual': fitted.residual_error
            }
        )
    
    # =========================================================================
    # Thickness Measurements
    # =========================================================================
    
    def thickness_along_normal(
        self,
        contour: np.ndarray,
        mask: np.ndarray,
        num_samples: int = 20
    ) -> MeasurementResult:
        """
        Measure the average thickness at points along the contour normal.
        
        :param contour: External contour array.
        :param mask: Binary mask for ray intersection tests.
        :param num_samples: Number of points along the contour to sample.
        :return: MeasurementResult with average thickness and statistical distribution.
        """
        contour_pts = contour.reshape(-1, 2)
        n_pts = len(contour_pts)
        
        if n_pts < 3:
            return None
        
        # Sample points along contour
        indices = np.linspace(0, n_pts - 1, num_samples, dtype=int)
        thicknesses = []
        
        for i in indices:
            pt = contour_pts[i]
            
            # Estimate normal direction
            prev_pt = contour_pts[(i - 1) % n_pts]
            next_pt = contour_pts[(i + 1) % n_pts]
            tangent = next_pt - prev_pt
            normal = np.array([-tangent[1], tangent[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            
            # Ray cast in normal direction
            thickness = self._ray_cast_thickness(pt, normal, mask)
            if thickness > 0:
                thicknesses.append(thickness)
        
        if len(thicknesses) == 0:
            return None
        
        avg_thickness = np.mean(thicknesses)
        std_thickness = np.std(thicknesses)
        
        return MeasurementResult(
            measurement_type='thickness_normal',
            value_pixels=float(avg_thickness),
            value_mm=self.to_mm(avg_thickness),
            confidence=max(0, 1 - std_thickness / (avg_thickness + 1e-8)),
            additional_info={
                'std': std_thickness,
                'min': min(thicknesses),
                'max': max(thicknesses),
                'samples': len(thicknesses)
            }
        )
    
    def _ray_cast_thickness(
        self,
        start_point: np.ndarray,
        direction: np.ndarray,
        mask: np.ndarray,
        max_distance: int = 500
    ) -> float:
        """
        Perform a ray cast to find the nearest mask boundary.

        :param start_point: (x, y) starting coordinate.
        :param direction: Normalized (dx, dy) direction vector.
        :param mask: Binary mask array.
        :param max_distance: Maximum search range in pixels.
        :return: Distance to boundary, or 0 if not found.
        """
        h, w = mask.shape
        
        # Cast ray in direction
        for dist in range(1, max_distance):
            pt = start_point + direction * dist
            x, y = int(pt[0]), int(pt[1])
            
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            
            if mask[y, x] == 0:
                return dist
        
        return 0
    
    # =========================================================================
    # Sub-pixel Refinement
    # =========================================================================
    
    def subpixel_edge_location(
        self,
        image: np.ndarray,
        contour: np.ndarray
    ) -> np.ndarray:
        """
        Refine contour points to sub-pixel accuracy using OpenCV corner refinement.
        
        :param image: Grayscale source image.
        :param contour: Initial integer contour.
        :return: Refined floating-point contour points.
        """
        if not self.use_subpixel:
            return contour
        
        contour_pts = contour.reshape(-1, 1, 2).astype(np.float32)
        
        # Use OpenCV's cornerSubPix for refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        try:
            refined = cv2.cornerSubPix(
                image,
                contour_pts,
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=criteria
            )
            return refined.reshape(-1, 2)
        except Exception:
            return contour.reshape(-1, 2)
    
    # =========================================================================
    # Comprehensive Measurement
    # =========================================================================
    
    def compute_all_measurements(
        self,
        contours: List[np.ndarray],
        mask: np.ndarray = None,
        image: np.ndarray = None
    ) -> Dict[str, List[MeasurementResult]]:
        """
        Execute all applicable measurements on a set of contours.
        
        :param contours: List of contours to measure.
        :param mask: Binary mask (optional, for thickness/inner diam).
        :param image: Source image (optional, for sub-pixel refinement).
        :return: Dictionary categorized by measurement type.
        """
        results = {
            'diameters': [],
            'distances': [],
            'thicknesses': [],
            'shapes': []
        }
        
        for i, contour in enumerate(contours):
            # Diameter measurements
            outer = self.outer_diameter(contour)
            if outer:
                outer.additional_info['contour_idx'] = i
                results['diameters'].append(outer)
            
            fitted_circle = self.fitted_circle_diameter(contour)
            if fitted_circle:
                fitted_circle.additional_info['contour_idx'] = i
                results['diameters'].append(fitted_circle)
            
            # Thickness if mask available
            if mask is not None:
                thickness = self.thickness_along_normal(contour, mask)
                if thickness:
                    thickness.additional_info['contour_idx'] = i
                    results['thicknesses'].append(thickness)
        
        # Distance between contour pairs
        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                dist = self.min_distance_between_contours(contours[i], contours[j])
                dist.additional_info['contour_pair'] = (i, j)
                results['distances'].append(dist)
        
        # Inner diameter if mask available
        if mask is not None:
            inner = self.inner_diameter(mask)
            if inner:
                results['diameters'].append(inner)
            
            gap = self.gap_measurement(mask)
            if gap:
                results['distances'].append(gap)
        
        return results


class CalibrationManager:
    """
    Manage pixel-to-metric calibration.
    
    Supports static calibration from config, manual calibration from a known
    distance or reference object, and dynamic post-prediction calibration
    using a named reference label among predicted contours.
    """
    
    def __init__(self):
        """Initialize with static calibration from config."""
        self.pixel_to_mm = config.measurement.pixel_to_mm
        self.default_ppm = 1.0 / self.pixel_to_mm if self.pixel_to_mm > 0 else 1.0
        self.calibration_method = 'static_config'
    
    def get_ppm(self) -> float:
        """Get current Pixels Per Millimeter (PPM)."""
        return 1.0 / self.pixel_to_mm if self.pixel_to_mm > 0 else 0.0
    
    def set_ppm(self, ppm: float):
        """Set conversion factor from PPM."""
        if ppm > 0:
            self.pixel_to_mm = 1.0 / ppm
            self.calibration_method = 'manual_ppm'
            
    def reset(self):
        """Reset to configuration default."""
        self.pixel_to_mm = config.measurement.pixel_to_mm
        self.calibration_method = 'static_config'
    
    def calibrate_from_known_distance(
        self,
        pixel_distance: float,
        real_distance_mm: float
    ) -> float:
        """
        Calibrate using a known reference distance.
        
        :param pixel_distance: Distance in pixels.
        :param real_distance_mm: Known real distance in mm.
        :return: Calibration factor (mm per pixel).
        """
        self.pixel_to_mm = real_distance_mm / pixel_distance
        self.calibration_method = 'known_distance'
        return self.pixel_to_mm
    
    def calibrate_from_reference_object(
        self,
        contour: np.ndarray,
        real_diameter_mm: float
    ) -> float:
        """
        Calibrate using a reference circular object.
        
        :param contour: Contour of reference object.
        :param real_diameter_mm: Known diameter in mm.
        :return: Calibration factor (mm per pixel).
        """
        (_, _), radius = cv2.minEnclosingCircle(contour)
        pixel_diameter = 2 * radius
        
        self.pixel_to_mm = real_diameter_mm / pixel_diameter
        self.calibration_method = 'reference_object'
        return self.pixel_to_mm
    
    def _measure_contour_dimension(
        self,
        contour: np.ndarray,
        dimension_type: str
    ) -> float:
        """
        Measure a geometric property of a contour in pixels.
        
        :param contour: Contour points array.
        :param dimension_type: 'diameter', 'width', or 'height'.
        :return: The measured dimension in pixels.
        :raises ValueError: If dimension_type is not recognized.
        """
        if dimension_type == "diameter":
            (_, _), radius = cv2.minEnclosingCircle(contour)
            return 2.0 * radius
        elif dimension_type == "width":
            x, y, w, h = cv2.boundingRect(contour)
            return float(w)
        elif dimension_type == "height":
            x, y, w, h = cv2.boundingRect(contour)
            return float(h)
        else:
            raise ValueError(
                f"Unknown dimension_type '{dimension_type}'. "
                "Use 'diameter', 'width', or 'height'."
            )
    
    def calibrate_from_contour(
        self,
        contour: np.ndarray,
        known_dimension_mm: float,
        dimension_type: str = "diameter"
    ) -> float:
        """
        Derive px→mm from a single reference contour whose real size is known.
        
        :param contour: The contour of the reference object.
        :param known_dimension_mm: Real-world dimension in mm.
        :param dimension_type: 'diameter', 'width', or 'height'.
        :return: The new pixel_to_mm factor.
        """
        pixel_dim = self._measure_contour_dimension(contour, dimension_type)
        
        if pixel_dim < 1e-6:
            print(f"  [WARNING] Reference contour has near-zero {dimension_type} "
                  f"({pixel_dim:.2f} px). Skipping calibration.")
            return self.pixel_to_mm
        
        self.pixel_to_mm = known_dimension_mm / pixel_dim
        self.calibration_method = 'post_prediction_reference'
        print(f"  [CALIBRATION] {dimension_type}={pixel_dim:.1f}px → "
              f"{known_dimension_mm:.1f}mm  ⇒  {self.pixel_to_mm:.4f} mm/px")
        return self.pixel_to_mm
    
    def calibrate_from_predictions(
        self,
        prediction: Any,
        reference_label: str = None,
        known_dimension_mm: float = None,
        dimension_type: str = None
    ) -> Optional[float]:
        """
        Search predictions for the reference label and calibrate dynamically.
        
        Supports both a dictionary of contours {label: [contours]} and a 
        raw prediction mask (integer labeled).

        :param prediction: Mask (np.ndarray) or dict of contours.
        :param reference_label: Label name to use as reference (default: from config).
        :param known_dimension_mm: Known dimension in mm (default: from config).
        :param dimension_type: 'diameter', 'width', or 'height' (default: from config).
        :return: Updated pixel_to_mm factor, or None if reference was not found.
        """
        cfg = config.measurement
        ref_label = reference_label or cfg.reference_label_name
        ref_mm = known_dimension_mm if known_dimension_mm is not None else cfg.reference_known_dimension_mm
        ref_dim = dimension_type or cfg.reference_dimension_type
        
        if not ref_label:
            return None
        
        # 1. Handle case where prediction is a dictionary of contours
        if isinstance(prediction, dict):
            if ref_label not in prediction or len(prediction[ref_label]) == 0:
                return None
            ref_contours = prediction[ref_label]
            largest = max(ref_contours, key=cv2.contourArea)
            return self.calibrate_from_contour(largest, ref_mm, ref_dim)
            
        # 2. Handle case where prediction is an integer mask
        if isinstance(prediction, np.ndarray):
            # Try to find class index for ref_label
            try:
                # Assuming labels are available in config or dataset
                from src.data.dataset import LABELS
                if ref_label in LABELS:
                    label_idx = LABELS.index(ref_label)
                    # Extract contours for this label
                    label_mask = (prediction == label_idx).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        return self.calibrate_from_contour(largest, ref_mm, ref_dim)
            except (ImportError, ValueError):
                pass
                
        return None
    
    def get_conversion_factor(self) -> float:
        """Get current conversion factor."""
        return self.pixel_to_mm
    
    def convert_to_mm(self, value_pixels: float) -> float:
        """Convert pixel value to millimeters."""
        return value_pixels * self.pixel_to_mm

    # ==================================================================
    # Method A: ArUco / Reference Object Calibration
    # ==================================================================

    def calibrate_from_aruco(
        self,
        image: np.ndarray,
        marker_real_size_mm: float = None
    ) -> Optional[float]:
        """
        Detect an ArUco marker in the image and calibrate using its known size.

        Computes PPM = marker_width_px / marker_real_size_mm, then sets
        pixel_to_mm = 1 / PPM.

        :param image: BGR or grayscale input image.
        :param marker_real_size_mm: Physical side length of the marker in mm.
            Falls back to config.measurement.aruco_marker_size_mm if None.
        :return: Updated pixel_to_mm factor, or None if no marker detected.
        """
        if marker_real_size_mm is None:
            marker_real_size_mm = config.measurement.aruco_marker_size_mm

        corners, ids = detect_aruco_markers(image)
        if corners is None or len(corners) == 0:
            print("  [ARUCO] No markers detected.")
            return None

        # Use the first detected marker
        marker_corners = corners[0].reshape(4, 2)
        marker_id = int(ids[0]) if ids is not None else -1

        # Compute average side length in pixels
        side_lengths = []
        for i in range(4):
            p1 = marker_corners[i]
            p2 = marker_corners[(i + 1) % 4]
            side_lengths.append(np.linalg.norm(p2 - p1))
        avg_side_px = np.mean(side_lengths)

        if avg_side_px < 1e-6:
            print("  [ARUCO] Marker too small to calibrate.")
            return None

        ppm = avg_side_px / marker_real_size_mm
        self.pixel_to_mm = 1.0 / ppm
        self.calibration_method = 'aruco_marker'

        print(f"  [ARUCO] Marker ID={marker_id}, side={avg_side_px:.1f}px, "
              f"real={marker_real_size_mm:.1f}mm => PPM={ppm:.3f}, "
              f"pixel_to_mm={self.pixel_to_mm:.5f}")
        return self.pixel_to_mm

    # ==================================================================
    # Method B: Geometric Camera Calibration
    # ==================================================================

    def calibrate_from_camera_intrinsics(
        self,
        sensor_width_mm: float = None,
        focal_length_mm: float = None,
        object_distance_mm: float = None,
        image_width_px: int = None,
        image: np.ndarray = None
    ) -> float:
        """
        Calibrate using the camera's physical parameters.

        Computes pixel_size = (D * W_s) / (f * image_width_px) where:
        - D = object distance from lens
        - W_s = physical sensor width
        - f = focal length

        :param sensor_width_mm: Physical sensor width (default: from config).
        :param focal_length_mm: Lens focal length (default: from config).
        :param object_distance_mm: Distance from lens to object (default: from config).
        :param image_width_px: Image width in pixels. Inferred from image if provided.
        :param image: Optional image to infer width from.
        :return: Computed pixel_to_mm factor.
        """
        cfg = config.measurement
        W_s = sensor_width_mm if sensor_width_mm is not None else cfg.sensor_width_mm
        f = focal_length_mm if focal_length_mm is not None else cfg.focal_length_mm
        D = object_distance_mm if object_distance_mm is not None else cfg.object_distance_mm

        if image_width_px is None and image is not None:
            image_width_px = image.shape[1]
        elif image_width_px is None:
            raise ValueError("Either image_width_px or image must be provided.")

        if f < 1e-6 or image_width_px < 1:
            raise ValueError(f"Invalid parameters: focal={f}, width={image_width_px}")

        pixel_size_mm = (D * W_s) / (f * image_width_px)
        self.pixel_to_mm = pixel_size_mm
        self.calibration_method = 'camera_intrinsics'

        print(f"  [CAMERA] sensor={W_s:.2f}mm, focal={f:.2f}mm, "
              f"distance={D:.1f}mm, img_width={image_width_px}px")
        print(f"  [CAMERA] pixel_size={pixel_size_mm:.5f} mm/px")
        return self.pixel_to_mm

    # ==================================================================
    # Method C: ML Depth Estimation
    # ==================================================================

    def calibrate_from_depth_estimation(
        self,
        image: np.ndarray,
        sensor_width_mm: float = None,
        focal_length_mm: float = None,
        mask: np.ndarray = None,
        model_type: str = "MiDaS_small"
    ) -> Optional[float]:
        """
        Estimate object distance using a MiDaS/DPT monocular depth model,
        then calibrate using camera intrinsics.

        :param image: BGR input image.
        :param sensor_width_mm: Sensor width (default: from config).
        :param focal_length_mm: Focal length (default: from config).
        :param mask: Optional binary mask to focus depth estimation on the
            object region. If None, uses the center 50% of the image.
        :param model_type: MiDaS model type ('DPT_Small', 'DPT_Hybrid', 'MiDaS_small').
        :return: Computed pixel_to_mm factor, or None on failure.
        """
        try:
            import torch
        except ImportError:
            print("  [DEPTH] PyTorch not available.")
            return None

        cfg = config.measurement
        W_s = sensor_width_mm if sensor_width_mm is not None else cfg.sensor_width_mm
        f = focal_length_mm if focal_length_mm is not None else cfg.focal_length_mm

        print(f"  [DEPTH] Loading MiDaS model: {model_type}...")
        try:
            midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)

            if model_type in ("DPT_Large", "DPT_Hybrid"):
                transform = midas_transforms.dpt_transform
            else:
                transform = midas_transforms.small_transform

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            midas.to(device).eval()
        except Exception as e:
            print(f"  [DEPTH] Failed to load MiDaS: {e}")
            return None

        # Run inference
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)

        with torch.no_grad():
            depth_prediction = midas(input_batch)
            depth_prediction = torch.nn.functional.interpolate(
                depth_prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        # Extract depth for region of interest
        if mask is not None and mask.shape == depth_prediction.shape:
            roi_mask = mask > 0
        else:
            # Use center 50% of image
            h, w = depth_prediction.shape
            roi_mask = np.zeros_like(depth_prediction, dtype=bool)
            roi_mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True

        roi_depths = depth_prediction[roi_mask]
        if len(roi_depths) == 0:
            print("  [DEPTH] No valid depth values in ROI.")
            return None

        # MiDaS outputs inverse depth (relative). Convert to approximate mm.
        # The absolute scale requires calibration; we use median inverse depth
        # as a relative distance proxy. For uncalibrated setups, we normalize
        # against the focal length to get an approximate working distance.
        median_inv_depth = np.median(roi_depths)
        if median_inv_depth < 1e-6:
            print("  [DEPTH] Estimated depth is too small.")
            return None

        # Approximate distance: D ≈ f * scale / median_inverse_depth
        # Using a heuristic scale factor based on typical MiDaS output ranges
        estimated_distance_mm = f * 1000.0 / median_inv_depth

        print(f"  [DEPTH] Median inverse depth: {median_inv_depth:.4f}")
        print(f"  [DEPTH] Estimated distance: {estimated_distance_mm:.1f} mm")

        # Now use camera intrinsics formula
        pixel_size_mm = (estimated_distance_mm * W_s) / (f * image.shape[1])
        self.pixel_to_mm = pixel_size_mm
        self.calibration_method = 'depth_estimation'
        self._last_depth_map = depth_prediction
        self._estimated_distance_mm = estimated_distance_mm

        print(f"  [DEPTH] pixel_to_mm = {pixel_size_mm:.5f}")
        return self.pixel_to_mm

    def get_last_depth_map(self) -> Optional[np.ndarray]:
        """Return the last depth map computed by calibrate_from_depth_estimation."""
        return getattr(self, '_last_depth_map', None)


# ==============================================================================
# Standalone Utility Functions
# ==============================================================================

def detect_aruco_markers(
    image: np.ndarray,
    dictionary_name: str = None
) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
    """
    Detect ArUco markers in an image.

    Tries the requested dictionary first, then falls back to scanning all
    common ArUco dictionaries.

    :param image: BGR or grayscale input image.
    :param dictionary_name: ArUco dictionary name (e.g. 'DICT_4X4_50').
        Falls back to config default, then scans all.
    :return: Tuple of (corners_list, ids_array), or (None, None) if none found.
    """
    if dictionary_name is None:
        dictionary_name = config.measurement.aruco_dictionary

    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try to use cv2.aruco (OpenCV contrib)
    try:
        aruco = cv2.aruco
    except AttributeError:
        print("  [ARUCO] cv2.aruco not available. Install opencv-contrib-python.")
        return None, None

    detector_params = aruco.DetectorParameters()

    # Dictionaries to try
    dict_names = [dictionary_name] if dictionary_name else []
    dict_names += [
        "DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_7X7_50",
        "DICT_4X4_100", "DICT_5X5_100", "DICT_ARUCO_ORIGINAL",
    ]

    for name in dict_names:
        dict_id = getattr(aruco, name, None)
        if dict_id is None:
            continue

        dictionary = aruco.getPredefinedDictionary(dict_id)

        # Try new API first (OpenCV 4.7+), fall back to legacy
        try:
            detector = aruco.ArucoDetector(dictionary, detector_params)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=detector_params)

        if ids is not None and len(ids) > 0:
            return corners, ids

    return None, None


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    new_camera_matrix: np.ndarray = None
) -> np.ndarray:
    """
    Remove lens distortion from an image using known camera parameters.

    :param image: Distorted input image.
    :param camera_matrix: 3x3 intrinsic camera matrix.
    :param dist_coeffs: Distortion coefficients (k1, k2, p1, p2[, k3...]).
    :param new_camera_matrix: Optional new camera matrix for undistortion.
        If None, uses the optimal matrix retaining all pixels.
    :return: Undistorted image.
    """
    h, w = image.shape[:2]

    if new_camera_matrix is None:
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha=1.0, newImgSize=(w, h)
        )

    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted


def calibrate_camera_from_checkerboard(
    images: List[np.ndarray],
    board_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute camera matrix and distortion coefficients from checkerboard images.

    :param images: List of images containing the checkerboard pattern.
    :param board_size: Internal corner count (columns, rows).
    :param square_size_mm: Physical size of each square in mm.
    :return: Tuple of (camera_matrix, dist_coeffs, reprojection_error).
    :raises ValueError: If no checkerboard corners are found in any image.
    """
    # Prepare object points (0,0,0), (1,0,0), ... * square_size
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = []  # 3D world coordinates
    img_points = []  # 2D image coordinates

    for img in images:
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size, None)

        if found:
            # Refine to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)

    if not obj_points:
        raise ValueError("No checkerboard corners found in any of the provided images.")

    h, w = images[0].shape[:2]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )

    print(f"  [CHECKERBOARD] Calibrated from {len(obj_points)} images")
    print(f"  [CHECKERBOARD] Reprojection error: {ret:.4f} px")
    print(f"  [CHECKERBOARD] Camera Matrix:\n{camera_matrix}")
    print(f"  [CHECKERBOARD] Distortion: {dist_coeffs.ravel()}")

    return camera_matrix, dist_coeffs, ret

