"""
Benchmarking metrics for segmentation and measurement accuracy.
Includes IoU, Dice, F1-score, MAE, RMSE, and hardware profiling.
"""

import time
import psutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    PYNVML_AVAILABLE = False


@dataclass
class SegmentationMetrics:
    """
    Accuracy metrics for pixel-wise segmentation.
    
    :param iou: Intersection over Union (Jaccard Index).
    :param dice: Dice coefficient (F1-score for masks).
    :param precision: Positive Predictive Value.
    :param recall: Sensitivity or True Positive Rate.
    :param accuracy: Ratio of correctly classified pixels.
    :param f1_score: Harmonic mean of precision and recall.
    :param boundary_f1: F1-score specifically for object boundaries.
    :param hausdorff_distance: Max distance between predicted and GT contours.
    """
    iou: float
    dice: float
    precision: float
    recall: float
    accuracy: float
    f1_score: float
    boundary_f1: float = 0.0
    hausdorff_distance: float = 0.0


@dataclass
class ContourMetrics:
    """
    Quality metrics for object contours/boundaries.
    
    :param boundary_f1: F1-score emphasizing boundary pixel alignment.
    :param avg_boundary_distance: Mean distance between boundary points (pixels).
    :param average_contour_distance: Alias for avg_boundary_distance.
    :param hausdorff_distance: Max outlier distance for the contour.
    :param contour_completeness: Ratio of GT boundary points successfully matched.
    :param smoothness_score: Metric of contour curvature smoothness.
    """
    boundary_f1: float = 0.0
    avg_boundary_distance: float = 0.0
    average_contour_distance: float = 0.0
    hausdorff_distance: float = 0.0
    contour_completeness: float = 0.0
    smoothness_score: float = 0.0


@dataclass
class MeasurementMetrics:
    """
    Accuracy metrics for physical dimension measurements.
    
    :param mae: Mean Absolute Error (in mm).
    :param rmse: Root Mean Square Error (in mm).
    :param relative_error: Error as a percentage of the ground truth value.
    :param max_error: Worst-case absolute error found in the batch.
    """
    mae: float
    rmse: float
    relative_error: float
    max_error: float


@dataclass
class LatencyMetrics:
    """
    Timing and throughput metrics for inference pipelines.
    
    :param preprocessing_time_ms: Time spent in image resizing/normalization.
    :param inference_time_ms: Pure model execution time.
    :param postprocessing_time_ms: Time spent in mask generation/refinement.
    :param total_time_ms: Sum of pre, inference, and post-processing.
    :param fps: Frames per second (1000 / total_time_ms).
    :param std_dev_ms: Latency jitter/standard deviation.
    """
    preprocessing_time_ms: float
    inference_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float
    fps: float
    std_dev_ms: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    data_load_time_ms: float = 0.0


@dataclass
class HardwareMetrics:
    """Hardware resource usage metrics"""
    cpu_percent: float
    cpu_percent_peak: float = 0.0
    ram_used_mb: float = 0.0
    ram_peak_mb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_utilization_peak: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    gpu_memory_used_mb: float = 0.0  # Alias for compatibility


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single model/sample"""
    model_name: str
    model_type: str = 'unknown'
    sample_id: str = ''
    segmentation: Optional[SegmentationMetrics] = None
    contour: Optional[ContourMetrics] = None
    measurement: Optional[MeasurementMetrics] = None
    latency: Optional[LatencyMetrics] = None
    hardware: Optional[HardwareMetrics] = None


class MetricsComputer:
    """
    Utility class for computing all benchmark-related metrics.
    
    Provides specialized methods for calculating segmentation accuracy,
    contour quality (using boundary distance and Hausdorff), and physical
    measurement errors (MAE/RMSE).
    """
    
    def __init__(self, boundary_tolerance: int = 2):
        """
        Initialize metrics computer.
        
        Args:
            boundary_tolerance: Tolerance in pixels for boundary metrics
        """
        self.boundary_tolerance = boundary_tolerance
    
    # =========================================================================
    # Segmentation Metrics
    # =========================================================================
    
    def compute_segmentation_metrics(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> SegmentationMetrics:
        """
        Compute pixel-wise segmentation accuracy metrics between two masks.
        
        Calculates IoU, Dice, Precision, Recall, and Accuracy by comparing 
        the predicted binary mask with the ground truth.

        :param pred_mask: Predicted binary mask (0 or 255).
        :param gt_mask: Ground truth binary mask (0 or 255).
        :return: SegmentationMetrics containing the calculated scores.
        """
        # Ensure binary masks
        pred = (pred_mask > 127).astype(np.float32) if pred_mask.max() > 1 else pred_mask.astype(np.float32)
        gt = (gt_mask > 127).astype(np.float32) if gt_mask.max() > 1 else gt_mask.astype(np.float32)
        
        # Flatten
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        
        # True/False Positives/Negatives
        tp = np.sum(pred_flat * gt_flat)
        fp = np.sum(pred_flat * (1 - gt_flat))
        fn = np.sum((1 - pred_flat) * gt_flat)
        tn = np.sum((1 - pred_flat) * (1 - gt_flat))
        
        # Metrics
        eps = 1e-7
        
        # IoU (Intersection over Union / Jaccard Index)
        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + eps)
        
        # Dice coefficient (F1 for segmentation)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        
        # Precision
        precision = tp / (tp + fp + eps)
        
        # Recall
        recall = tp / (tp + fn + eps)
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        
        # F1-score (same as Dice for binary)
        f1_score = 2 * precision * recall / (precision + recall + eps)
        
        return SegmentationMetrics(
            iou=float(iou),
            dice=float(dice),
            precision=float(precision),
            recall=float(recall),
            accuracy=float(accuracy),
            f1_score=float(f1_score)
        )
    
    def compute_batch_segmentation_metrics(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray]
    ) -> SegmentationMetrics:
        """Compute average metrics over a batch."""
        metrics_list = []
        for pred, gt in zip(pred_masks, gt_masks):
            metrics_list.append(self.compute_segmentation_metrics(pred, gt))
        
        return SegmentationMetrics(
            iou=np.mean([m.iou for m in metrics_list]),
            dice=np.mean([m.dice for m in metrics_list]),
            precision=np.mean([m.precision for m in metrics_list]),
            recall=np.mean([m.recall for m in metrics_list]),
            accuracy=np.mean([m.accuracy for m in metrics_list]),
            f1_score=np.mean([m.f1_score for m in metrics_list])
        )
    
    # =========================================================================
    # Contour/Boundary Metrics
    # =========================================================================
    
    def compute_contour_metrics(
        self,
        pred_contours: List[np.ndarray],
        gt_contours: List[np.ndarray],
        image_shape: Tuple[int, int]
    ) -> ContourMetrics:
        """
        Compute quality metrics based on object boundaries and contours.
        
        Calculates the Boundary F1-score (with pixel tolerance), mean distance
        to the nearest boundary point, and the Hausdorff distance.

        :param pred_contours: List of contours from the predicted mask.
        :param gt_contours: List of contours from the ground truth.
        :param image_shape: Dimensions of the image (H, W).
        :return: ContourMetrics containing boundary-specific scores.
        """
        if len(pred_contours) == 0 or len(gt_contours) == 0:
            return ContourMetrics(
                boundary_f1=0.0,
                avg_boundary_distance=float('inf'),
                hausdorff_distance=float('inf'),
                contour_completeness=0.0
            )
        
        # Create boundary masks
        pred_boundary = np.zeros(image_shape, dtype=np.uint8)
        gt_boundary = np.zeros(image_shape, dtype=np.uint8)
        
        for cnt in pred_contours:
            cv2.drawContours(pred_boundary, [cnt], -1, 255, 1)
        
        for cnt in gt_contours:
            cv2.drawContours(gt_boundary, [cnt], -1, 255, 1)
        
        # Boundary F1-score with tolerance
        pred_dilated = cv2.dilate(
            pred_boundary,
            np.ones((2*self.boundary_tolerance+1, 2*self.boundary_tolerance+1), np.uint8)
        )
        gt_dilated = cv2.dilate(
            gt_boundary,
            np.ones((2*self.boundary_tolerance+1, 2*self.boundary_tolerance+1), np.uint8)
        )
        
        # Precision: predicted boundary within tolerance of GT
        pred_pts = np.argwhere(pred_boundary > 0)
        gt_mask_dilated = gt_dilated > 0
        if len(pred_pts) > 0:
            precision = np.mean([gt_mask_dilated[p[0], p[1]] for p in pred_pts])
        else:
            precision = 0
        
        # Recall: GT boundary within tolerance of predicted
        gt_pts = np.argwhere(gt_boundary > 0)
        pred_mask_dilated = pred_dilated > 0
        if len(gt_pts) > 0:
            recall = np.mean([pred_mask_dilated[p[0], p[1]] for p in gt_pts])
        else:
            recall = 0
        
        # F1
        boundary_f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        # Average boundary distance
        if len(pred_pts) > 0 and len(gt_pts) > 0:
            from scipy.spatial.distance import cdist
            distances = cdist(pred_pts, gt_pts, 'euclidean')
            avg_dist = np.mean(np.min(distances, axis=1))
        else:
            avg_dist = float('inf')
        
        # Hausdorff distance
        if len(pred_pts) > 0 and len(gt_pts) > 0:
            hausdorff_1 = directed_hausdorff(pred_pts, gt_pts)[0]
            hausdorff_2 = directed_hausdorff(gt_pts, pred_pts)[0]
            hausdorff = max(hausdorff_1, hausdorff_2)
        else:
            hausdorff = float('inf')
        
        # Contour completeness: ratio of GT contour points matched
        completeness = recall
        
        return ContourMetrics(
            boundary_f1=float(boundary_f1),
            avg_boundary_distance=float(avg_dist),
            hausdorff_distance=float(hausdorff),
            contour_completeness=float(completeness)
        )
    
    # =========================================================================
    # Measurement Metrics
    # =========================================================================
    
    def compute_measurement_metrics(
        self,
        pred_measurements: List[float],
        gt_measurements: List[float]
    ) -> MeasurementMetrics:
        """
        Compute measurement accuracy metrics.
        
        Args:
            pred_measurements: Predicted measurement values
            gt_measurements: Ground truth measurement values
            
        Returns:
            MeasurementMetrics object
        """
        if len(pred_measurements) == 0 or len(gt_measurements) == 0:
            return MeasurementMetrics(
                mae=float('inf'),
                rmse=float('inf'),
                relative_error=float('inf'),
                max_error=float('inf')
            )
        
        pred = np.array(pred_measurements)
        gt = np.array(gt_measurements)
        
        # Ensure same length
        min_len = min(len(pred), len(gt))
        pred = pred[:min_len]
        gt = gt[:min_len]
        
        # Absolute errors
        abs_errors = np.abs(pred - gt)
        
        # MAE - Mean Absolute Error
        mae = np.mean(abs_errors)
        
        # RMSE - Root Mean Square Error
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        
        # Relative error
        relative_errors = abs_errors / (np.abs(gt) + 1e-7)
        relative_error = np.mean(relative_errors)
        
        # Maximum error
        max_error = np.max(abs_errors)
        
        return MeasurementMetrics(
            mae=float(mae),
            rmse=float(rmse),
            relative_error=float(relative_error),
            max_error=float(max_error)
        )


class HardwareProfiler:
    """
    Monitor for hardware resource utilization (CPU, RAM, GPU).
    
    Uses psutil and pynvml to capture snapshots of system load during
    model execution, allowing for detailed efficiency benchmarking.
    """
    
    def __init__(self):
        self.cpu_samples = []
        self.gpu_memory_samples = []
        self.gpu_util_samples = []
        self.start_time = None
        self._process = psutil.Process()
    
    def start(self):
        """Start profiling."""
        self.cpu_samples = []
        self.gpu_memory_samples = []
        self.gpu_util_samples = []
        self.start_time = time.perf_counter()
        
        # Initial sample
        self._sample()
    
    def _sample(self):
        """Take a hardware sample."""
        # CPU
        self.cpu_samples.append(self._process.cpu_percent())
        
        # GPU (if available)
        if torch.cuda.is_available():
            self.gpu_memory_samples.append(
                torch.cuda.memory_allocated() / 1024 / 1024
            )
            
            if PYNVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_util_samples.append(util.gpu)
                except:
                    self.gpu_util_samples.append(0)
    
    def stop(self) -> HardwareMetrics:
        """Stop profiling and return metrics."""
        self._sample()
        
        return HardwareMetrics(
            cpu_percent=np.mean(self.cpu_samples) if self.cpu_samples else 0,
            cpu_percent_peak=max(self.cpu_samples) if self.cpu_samples else 0,
            ram_used_mb=self._process.memory_info().rss / 1024 / 1024,
            gpu_utilization=np.mean(self.gpu_util_samples) if self.gpu_util_samples else 0,
            gpu_memory_mb=np.mean(self.gpu_memory_samples) if self.gpu_memory_samples else 0,
            gpu_memory_peak_mb=max(self.gpu_memory_samples) if self.gpu_memory_samples else 0
        )
    
    @staticmethod
    def get_gpu_info() -> Dict:
        """Get GPU information."""
        info = {
            'available': torch.cuda.is_available(),
            'device_count': 0,
            'devices': []
        }
        
        if torch.cuda.is_available():
            info['device_count'] = torch.cuda.device_count()
            for i in range(info['device_count']):
                device_props = torch.cuda.get_device_properties(i)
                info['devices'].append({
                    'name': device_props.name,
                    'total_memory_mb': device_props.total_memory / 1024 / 1024,
                    'major': device_props.major,
                    'minor': device_props.minor
                })
        
        return info


class LatencyProfiler:
    """Profile inference latency."""
    
    def __init__(self):
        self.times = {
            'preprocessing': [],
            'inference': [],
            'postprocessing': []
        }
    
    def reset(self):
        """Reset all timings."""
        self.times = {
            'preprocessing': [],
            'inference': [],
            'postprocessing': []
        }
    
    def add_timing(self, stage: str, time_ms: float):
        """Add a timing sample."""
        if stage in self.times:
            self.times[stage].append(time_ms)
    
    def get_metrics(self) -> LatencyMetrics:
        """Get aggregated latency metrics."""
        preproc = np.mean(self.times['preprocessing']) if self.times['preprocessing'] else 0
        inference = np.mean(self.times['inference']) if self.times['inference'] else 0
        postproc = np.mean(self.times['postprocessing']) if self.times['postprocessing'] else 0
        
        total = preproc + inference + postproc
        fps = 1000 / total if total > 0 else 0
        
        return LatencyMetrics(
            preprocessing_time_ms=preproc,
            inference_time_ms=inference,
            postprocessing_time_ms=postproc,
            total_time_ms=total,
            fps=fps
        )
    
    def get_variance(self) -> Dict[str, float]:
        """Get timing variance for stability analysis."""
        return {
            stage: np.var(times) if times else 0
            for stage, times in self.times.items()
        }


import cv2  # Added for contour drawing
