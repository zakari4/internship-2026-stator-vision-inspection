"""
Contour extraction and geometric fitting methods for industrial inspection.
Includes various contour extraction techniques and geometric primitive fitting.
"""

import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize, thin
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

from src.config import config


@dataclass
class ContourResult:
    """Result of contour extraction"""
    contours: List[np.ndarray]
    hierarchy: Optional[np.ndarray]
    extraction_time_ms: float
    method_name: str
    num_contours: int


@dataclass
class FittedGeometry:
    """Fitted geometric primitive"""
    geometry_type: str  # 'line', 'circle', 'ellipse', 'polygon'
    parameters: Dict
    points: np.ndarray
    residual_error: float


class ContourExtractor:
    """
    Unified contour extraction with multiple methods.
    """
    
    def __init__(
        self,
        min_area: int = None,
        max_area: int = None,
        min_length: int = None,
        epsilon_factor: float = None
    ):
        cfg = config.contour
        
        self.min_area = min_area or cfg.min_area
        self.max_area = max_area or cfg.max_area
        self.min_length = min_length or cfg.min_length
        self.epsilon_factor = epsilon_factor or cfg.epsilon_factor
    
    def extract_opencv(
        self,
        mask: np.ndarray,
        mode: int = cv2.RETR_EXTERNAL,
        method: int = cv2.CHAIN_APPROX_SIMPLE
    ) -> ContourResult:
        """
        Extract contours using OpenCV findContours.
        
        Args:
            mask: Binary mask (0 or 255)
            mode: Contour retrieval mode
            method: Contour approximation method
            
        Returns:
            ContourResult with extracted contours
        """
        start = time.perf_counter()
        
        # Ensure binary mask
        if mask.max() == 1:
            mask = mask * 255
        mask = mask.astype(np.uint8)
        
        # Find contours
        contours, hierarchy = cv2.findContours(mask, mode, method)
        
        # Filter contours
        filtered_contours = self._filter_contours(list(contours))
        
        extraction_time = (time.perf_counter() - start) * 1000
        
        return ContourResult(
            contours=filtered_contours,
            hierarchy=hierarchy,
            extraction_time_ms=extraction_time,
            method_name="OpenCV_findContours",
            num_contours=len(filtered_contours)
        )
    
    def extract_morphological(self, mask: np.ndarray) -> ContourResult:
        """
        Extract contours using morphological gradient.
        
        Args:
            mask: Binary mask
            
        Returns:
            ContourResult with extracted contours
        """
        start = time.perf_counter()
        
        # Ensure binary mask
        if mask.max() == 1:
            mask = mask * 255
        mask = mask.astype(np.uint8)
        
        # Compute morphological gradient (contour)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        
        # Find contours on gradient
        contours, hierarchy = cv2.findContours(
            gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        filtered_contours = self._filter_contours(list(contours))
        
        extraction_time = (time.perf_counter() - start) * 1000
        
        return ContourResult(
            contours=filtered_contours,
            hierarchy=hierarchy,
            extraction_time_ms=extraction_time,
            method_name="Morphological_Gradient",
            num_contours=len(filtered_contours)
        )
    
    def extract_skeleton(self, mask: np.ndarray) -> ContourResult:
        """
        Extract contours using skeletonization.
        
        Args:
            mask: Binary mask
            
        Returns:
            ContourResult with extracted contours
        """
        start = time.perf_counter()
        
        # Normalize mask to [0, 1]
        mask_bool = mask > 127 if mask.max() > 1 else mask > 0
        
        # Skeletonize
        skeleton = skeletonize(mask_bool).astype(np.uint8) * 255
        
        # Find contours on skeleton
        contours, hierarchy = cv2.findContours(
            skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        
        filtered_contours = self._filter_contours(list(contours))
        
        extraction_time = (time.perf_counter() - start) * 1000
        
        return ContourResult(
            contours=filtered_contours,
            hierarchy=hierarchy,
            extraction_time_ms=extraction_time,
            method_name="Skeletonization",
            num_contours=len(filtered_contours)
        )
    
    def extract_thinning(self, mask: np.ndarray) -> ContourResult:
        """
        Extract contours using morphological thinning.
        
        Args:
            mask: Binary mask
            
        Returns:
            ContourResult with extracted contours
        """
        start = time.perf_counter()
        
        # Normalize mask
        mask_bool = mask > 127 if mask.max() > 1 else mask > 0
        
        # Apply thinning
        thinned = thin(mask_bool).astype(np.uint8) * 255
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            thinned, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        
        filtered_contours = self._filter_contours(list(contours))
        
        extraction_time = (time.perf_counter() - start) * 1000
        
        return ContourResult(
            contours=filtered_contours,
            hierarchy=hierarchy,
            extraction_time_ms=extraction_time,
            method_name="Thinning",
            num_contours=len(filtered_contours)
        )
    
    def _filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Filter contours by area and length."""
        filtered = []
        
        for cnt in contours:
            # Check minimum points
            if len(cnt) < 3:
                continue
            
            # Check area
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue
            
            # Check length
            length = cv2.arcLength(cnt, closed=True)
            if length < self.min_length:
                continue
            
            filtered.append(cnt)
        
        return filtered
    
    def approximate_contour(
        self,
        contour: np.ndarray,
        epsilon_factor: float = None
    ) -> np.ndarray:
        """
        Approximate contour with fewer points.
        
        Args:
            contour: Input contour
            epsilon_factor: Approximation factor
            
        Returns:
            Approximated contour
        """
        if epsilon_factor is None:
            epsilon_factor = self.epsilon_factor
        
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        return approx
    
    def compute_shape_descriptors(self, contour: np.ndarray) -> Dict:
        """
        Compute shape descriptors for a contour.
        
        Args:
            contour: Input contour
            
        Returns:
            Dictionary of shape descriptors
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity: 4*pi*area / perimeter^2
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / (h + 1e-8)
        extent = area / (w * h + 1e-8)
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-8)
        
        # Equivalent diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        
        # Moments
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
        else:
            cx, cy = 0, 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'solidity': solidity,
            'equivalent_diameter': equivalent_diameter,
            'centroid': (cx, cy),
            'bounding_box': (x, y, w, h)
        }


class GeometryFitter:
    """
    Fit geometric primitives to contours and points.
    """
    
    def __init__(
        self,
        line_method: str = None,
        ransac_threshold: float = None
    ):
        cfg = config.measurement
        
        self.line_method = line_method or cfg.line_fitting_method
        self.ransac_threshold = ransac_threshold or cfg.ransac_threshold
    
    def fit_line_least_squares(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fit line using least squares.
        
        Args:
            points: Array of points (N, 2)
            
        Returns:
            Tuple of (line_point, direction_vector, residual_error)
        """
        if len(points) < 2:
            return None, None, float('inf')
        
        # Reshape if needed
        if points.ndim == 3:
            points = points.squeeze()
        
        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1]
        
        # Fit line
        model = LinearRegression()
        model.fit(x, y)
        
        # Get line parameters
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Line point and direction
        point = np.array([0, intercept])
        direction = np.array([1, slope])
        direction = direction / np.linalg.norm(direction)
        
        # Compute residual
        y_pred = model.predict(x)
        residual = np.sqrt(np.mean((y - y_pred) ** 2))
        
        return point, direction, residual
    
    def fit_line_ransac(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Fit line using RANSAC for robustness.
        
        Args:
            points: Array of points (N, 2)
            
        Returns:
            Tuple of (line_point, direction_vector, residual_error, inlier_mask)
        """
        if len(points) < 2:
            return None, None, float('inf'), None
        
        # Reshape if needed
        if points.ndim == 3:
            points = points.squeeze()
        
        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1]
        
        # RANSAC fitting
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=self.ransac_threshold,
            random_state=42
        )
        
        try:
            ransac.fit(x, y)
            
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
            
            point = np.array([0, intercept])
            direction = np.array([1, slope])
            direction = direction / np.linalg.norm(direction)
            
            inlier_mask = ransac.inlier_mask_
            y_pred = ransac.predict(x)
            residual = np.sqrt(np.mean((y[inlier_mask] - y_pred[inlier_mask]) ** 2))
            
            return point, direction, residual, inlier_mask
            
        except Exception as e:
            return None, None, float('inf'), None
    
    def fit_line(
        self,
        points: np.ndarray,
        method: str = None
    ) -> FittedGeometry:
        """
        Fit line to points using specified method.
        
        Args:
            points: Array of points
            method: 'least_squares' or 'ransac'
            
        Returns:
            FittedGeometry object
        """
        if method is None:
            method = self.line_method
        
        if method == 'ransac':
            point, direction, residual, inliers = self.fit_line_ransac(points)
        else:
            point, direction, residual = self.fit_line_least_squares(points)
            inliers = None
        
        if point is None:
            return None
        
        return FittedGeometry(
            geometry_type='line',
            parameters={
                'point': point,
                'direction': direction,
                'inliers': inliers
            },
            points=points,
            residual_error=residual
        )
    
    def fit_circle(self, points: np.ndarray) -> FittedGeometry:
        """
        Fit circle to points using algebraic distance.
        
        Args:
            points: Array of points (N, 2)
            
        Returns:
            FittedGeometry with circle parameters
        """
        if len(points) < 3:
            return None
        
        # Reshape if needed
        if points.ndim == 3:
            points = points.squeeze()
        
        x = points[:, 0]
        y = points[:, 1]
        
        # Build design matrix for algebraic circle fit
        # x^2 + y^2 + Dx + Ey + F = 0
        A = np.column_stack([x, y, np.ones(len(x))])
        b = -(x**2 + y**2)
        
        # Solve least squares
        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            D, E, F = result
            
            # Convert to center and radius
            cx = -D / 2
            cy = -E / 2
            radius = np.sqrt(cx**2 + cy**2 - F)
            
            # Compute residual
            distances = np.sqrt((x - cx)**2 + (y - cy)**2)
            residual = np.sqrt(np.mean((distances - radius)**2))
            
            return FittedGeometry(
                geometry_type='circle',
                parameters={
                    'center': (cx, cy),
                    'radius': radius
                },
                points=points,
                residual_error=residual
            )
        except Exception:
            return None
    
    def fit_ellipse(self, contour: np.ndarray) -> FittedGeometry:
        """
        Fit ellipse to contour using OpenCV.
        
        Args:
            contour: Contour points
            
        Returns:
            FittedGeometry with ellipse parameters
        """
        if len(contour) < 5:
            return None
        
        try:
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            # Compute residual (distance from contour to ellipse)
            # Simplified: use bounding rect comparison
            rect_area = axes[0] * axes[1] * np.pi / 4
            contour_area = cv2.contourArea(contour)
            residual = abs(rect_area - contour_area) / (contour_area + 1e-8)
            
            return FittedGeometry(
                geometry_type='ellipse',
                parameters={
                    'center': center,
                    'axes': axes,
                    'angle': angle
                },
                points=contour,
                residual_error=residual
            )
        except Exception:
            return None
    
    def fit_min_rect(self, contour: np.ndarray) -> FittedGeometry:
        """
        Fit minimum area rotated rectangle.
        
        Args:
            contour: Contour points
            
        Returns:
            FittedGeometry with rectangle parameters
        """
        if len(contour) < 4:
            return None
        
        try:
            rect = cv2.minAreaRect(contour)
            center, size, angle = rect
            box = cv2.boxPoints(rect)
            
            # Compute residual
            rect_area = size[0] * size[1]
            contour_area = cv2.contourArea(contour)
            residual = abs(rect_area - contour_area) / (contour_area + 1e-8)
            
            return FittedGeometry(
                geometry_type='rectangle',
                parameters={
                    'center': center,
                    'size': size,
                    'angle': angle,
                    'box_points': box
                },
                points=contour,
                residual_error=residual
            )
        except Exception:
            return None


def extract_all_contours(
    mask: np.ndarray,
    methods: List[str] = None
) -> Dict[str, ContourResult]:
    """
    Extract contours using all available methods.
    
    Args:
        mask: Binary mask
        methods: List of methods to use (default: all)
        
    Returns:
        Dictionary of method name to ContourResult
    """
    extractor = ContourExtractor()
    
    if methods is None:
        methods = ['opencv', 'morphological', 'skeleton', 'thinning']
    
    results = {}
    
    if 'opencv' in methods:
        results['OpenCV'] = extractor.extract_opencv(mask)
    
    if 'morphological' in methods:
        results['Morphological'] = extractor.extract_morphological(mask)
    
    if 'skeleton' in methods:
        results['Skeleton'] = extractor.extract_skeleton(mask)
    
    if 'thinning' in methods:
        results['Thinning'] = extractor.extract_thinning(mask)
    
    return results
