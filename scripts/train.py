#!/usr/bin/env python3
"""
Main benchmark runner for Industrial Chignon Detection.
Orchestrates complete benchmark pipeline: data loading, model training,
model evaluation, metrics computation, and report generation.

Usage:
    # Inference benchmarking only
    python main.py --mode benchmark --models all
    python main.py --mode benchmark --models classical
    python main.py --mode benchmark --models deep_learning
    
    # Training with comprehensive data collection
    python main.py --mode train --models unet_lightweight unet_resnet18 --epochs 50
    
    # Full pipeline (train + benchmark)
    python main.py --mode full --models deep_learning --epochs 30
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

# Ensure project root is on sys.path so `src` package is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Import project modules
from src.config import config
from src.data.dataset import IndustrialDataset, create_dataloaders
from src.data.yolo_prep import prepare_yolo_dataset
from src.utils.preprocessing import PreprocessingPipeline
from src.utils.contour import ContourExtractor, GeometryFitter
from src.utils.measurements import MeasurementComputer, CalibrationManager
from src.evaluation.metrics import (
    MetricsComputer, HardwareProfiler, LatencyProfiler,
    BenchmarkResult, SegmentationMetrics, ContourMetrics,
    MeasurementMetrics, LatencyMetrics, HardwareMetrics
)
from src.evaluation.visualization import BenchmarkPlotter, VisualizationUtils
from src.training.trainer import ComprehensiveTrainer, train_all_models, compare_training_results, TrainingHistory, YOLOTrainer, train_yolo_models
from src.training.training_viz import TrainingVisualizer

from src.models import (
    # Deep learning models
    UNetLightweight,
    UNetResNet18,
    DeepLabV3MobileNet,
    SegFormerB0Simple,
    MaskRCNNModel,
    # YOLO Segmentation models
    YOLOv8SegNano,
    YOLOv8SegSmall,
    YOLOv8SegMedium,
    YOLOv11SegNano,
    YOLOv11SegSmall,
    YOLOv11SegMedium,
    # YOLOv12 is not available in official Ultralytics releases
    YOLOv26SegNano,
    YOLOv26SegSmall,
    YOLOv26SegMedium,
    # RT-DETR models
    RTDETRLarge,
    RTDETRXLarge,
    # DETR Variants
    CoDETRModel,
    RFDETRModel,
    # ScyllaNet
    ScyllaNetModel,
    # SAM 2 model
    SAM2LoRAModel,
    # Foundation / SOTA Models
    InternImageModel,
    SETRModel,
    SegNeXtModel,
    OneFormerModel,
    Mask2FormerModel,
    # SOTA Edge Detection models
    HEDModel,
    RCFModel,
    BDCNModel,
    PiDiNetModel,
    TEEDModel,
    LDCModel
)

warnings.filterwarnings('ignore')

# Model registry

DEEP_LEARNING_MODELS = {
    'unet_lightweight': UNetLightweight,
    'unet_resnet18': UNetResNet18,
    'deeplabv3_mobilenet': DeepLabV3MobileNet,
    'segformer_b0': SegFormerB0Simple,
    'mask_rcnn': MaskRCNNModel,
    # YOLOv8 Segmentation
    'yolov8n_seg': YOLOv8SegNano,
    'yolov8s_seg': YOLOv8SegSmall,
    'yolov8m_seg': YOLOv8SegMedium,
    # YOLOv11 Segmentation
    'yolov11n_seg': YOLOv11SegNano,
    'yolov11s_seg': YOLOv11SegSmall,
    'yolov11m_seg': YOLOv11SegMedium,
    # YOLOv26 Segmentation
    'yolov26n_seg': YOLOv26SegNano,
    'yolov26s_seg': YOLOv26SegSmall,
    'yolov26m_seg': YOLOv26SegMedium,
    # RT-DETR (Real-Time Detection Transformer)
    'rtdetr_l': RTDETRLarge,
    'rtdetr_x': RTDETRXLarge,
    # SAM 2 with LoRA
    'sam2_lora': SAM2LoRAModel,
    # New SOTA variants
    'codetr': CoDETRModel,
    'rf_detr': RFDETRModel,
    'scyllanet': ScyllaNetModel,
    'internimage': InternImageModel,
    'setr': SETRModel,
    'segnext': SegNeXtModel,
    'oneformer': OneFormerModel,
    'mask2former': Mask2FormerModel,
    # SOTA Edge Detection
    'hed': HEDModel,
    'rcf': RCFModel,
    'bdcn': BDCNModel,
    'pidinet': PiDiNetModel,
    'teed': TEEDModel,
    'ldc': LDCModel
}

ALL_MODELS = DEEP_LEARNING_MODELS


class BenchmarkRunner:
    """Main benchmark runner class."""
    
    def __init__(
        self,
        device: str = None,
        output_dir: str = None,
        num_warmup: int = 5,
        num_runs: int = 20,
        label_filter: list = None
    ):
        """
        Initialize benchmark runner.
        
        Args:
            device: Compute device ('cuda' or 'cpu')
            output_dir: Output directory for results
            num_warmup: Number of warmup iterations
            num_runs: Number of benchmark runs per sample
            label_filter: Optional list of label names to include
        """
        self.device = device or config.benchmark.device
        self.output_dir = output_dir or str(config.benchmark.results_dir)
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        self.label_filter = label_filter
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(config.benchmark.plots_dir, exist_ok=True)
        os.makedirs(config.benchmark.visual_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = PreprocessingPipeline()
        self.contour_extractor = ContourExtractor()
        self.geometry_fitter = GeometryFitter()
        self.measurement_computer = MeasurementComputer()
        self.calibration = CalibrationManager()
        
        self.metrics_computer = MetricsComputer()
        self.hardware_profiler = HardwareProfiler()
        self.latency_profiler = LatencyProfiler()
        
        self.visualizer = VisualizationUtils()
        self.plotter = BenchmarkPlotter()
        
        # Results storage
        self.results: Dict[str, BenchmarkResult] = {}
        
        print(f"Benchmark Runner initialized")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}")
        print(f"  Warmup runs: {num_warmup}")
        print(f"  Benchmark runs: {num_runs}")
    
    def load_data(self) -> Tuple:
        """Load dataset and create dataloaders."""
        print("\n" + "="*60)
        print("Loading Dataset")
        print("="*60)
        
        train_loader, val_loader, test_loader = create_dataloaders(
            root_path=str(config.dataset.root_path),
            batch_size=config.benchmark.batch_size,
            label_filter=self.label_filter
        )
        
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def benchmark_deep_learning_model(
        self,
        model,
        test_loader,
        model_name: str
    ) -> BenchmarkResult:
        """
        Benchmark a PyTorch-based deep learning model.
        
        Handles GPU memory profiling, warmup cycles, and automated hardware
        synchronization to ensure precise latency measurements for neural
        networks.

        :param model: The PyTorch model wrapper.
        :param test_loader: DataLoader for the test dataset.
        :param model_name: Label for result reporting.
        :return: BenchmarkResult including GPU resource utilization.
        """
        """
        Benchmark a deep learning segmentation model.
        
        Args:
            model: Model instance
            test_loader: Test data loader
            model_name: Model name for reporting
            
        Returns:
            BenchmarkResult
        """
        print(f"\n  Benchmarking {model_name}...")
        
        model.to(self.device)
        model.eval()
        
        all_predictions = []
        all_ground_truths = []
        all_latencies = []
        all_measurement_metrics = []
        gpu_memory_peaks = []
        
        # Clear GPU cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Warmup
        for i, batch in enumerate(test_loader):
            if i >= self.num_warmup:
                break
            
            # Extract from dict batch
            images = batch['image'] if isinstance(batch, dict) else batch[0]
            
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
                
            with torch.no_grad():
                if hasattr(model, 'segment'):
                    img_np = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    _ = model.segment(img_np)
                else:
                    _ = model(images.to(self.device))
        
        # Benchmark
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"    {model_name}", leave=False):
                # Extract from dict batch
                images = batch['image'] if isinstance(batch, dict) else batch[0]
                masks = batch['mask'] if isinstance(batch, dict) else batch[1]
                
                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
                if isinstance(masks, np.ndarray):
                    masks = torch.from_numpy(masks).float()
                
                images = images.to(self.device)
                
                for i in range(len(images)):
                    img = images[i:i+1]
                    gt_mask = masks[i].numpy() if torch.is_tensor(masks) else masks[i]
                    
                    # Clear cache before measurement
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    
                    # Multiple runs for timing
                    timings = []
                    for _ in range(self.num_runs):
                        if self.device == 'cuda':
                            torch.cuda.synchronize()
                        if hasattr(model, 'segment'):
                            # Use specialized segment method that handles post-processing
                            # Convert tensor to numpy for the wrapper
                            img_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            res = model.segment(img_np)
                            # Store result but we'll re-calculate pred_mask below for consistency
                            # if needed, or we can just use the mask from res
                            output = res
                            elapsed = res.inference_time_ms
                        else:
                            start = time.perf_counter()
                            output = model(img)
                            if self.device == 'cuda':
                                torch.cuda.synchronize()
                            elapsed = (time.perf_counter() - start) * 1000
                        
                        timings.append(elapsed)
                    
                    avg_time = np.mean(timings)
                    all_latencies.append(avg_time)
                    
                    # Get GPU memory
                    if self.device == 'cuda':
                        gpu_memory_peaks.append(
                            torch.cuda.max_memory_allocated() / 1024 / 1024
                        )
                    
                    # Convert output to mask
                    if model_name == 'mask_rcnn':
                        # Mask R-CNN returns dict with 'masks'
                        if len(output) > 0 and 'masks' in output[0] and len(output[0]['masks']) > 0:
                            pred_mask = output[0]['masks'][0, 0].cpu().numpy()
                            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                        else:
                            pred_mask = np.zeros_like(gt_mask)
                    else:
                        if hasattr(output, 'mask'):
                            pred_mask = output.mask
                        else:
                            pred_mask = output.squeeze().cpu().numpy()
                            if pred_mask.ndim == 3:
                                pred_mask = pred_mask.argmax(axis=0)
                            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                    
                    # Resize if needed
                    if pred_mask.shape != gt_mask.shape:
                        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
                    
                    all_predictions.append(pred_mask)
                    all_ground_truths.append(gt_mask)
                    
                    # Dynamic calibration and measurement
                    meas_metrics = self._post_prediction_calibrate(
                        pred_mask=pred_mask,
                        gt_mask=gt_mask,
                        shapes=batch.get('shapes', [[]])[i] if isinstance(batch, dict) else None
                    )
                    if meas_metrics:
                        all_measurement_metrics.append(meas_metrics)
        
        # Compute metrics
        seg_metrics = self._compute_segmentation_metrics(all_predictions, all_ground_truths)
        latency_metrics = self._compute_latency_metrics(all_latencies)
        contour_metrics = self._compute_contour_metrics(all_predictions, all_ground_truths)
        
        # Average measurement metrics
        measurement_metrics = self._average_measurement_metrics(all_measurement_metrics)
        
        # Hardware metrics
        hardware_metrics = HardwareMetrics(
            cpu_percent=0.0,
            gpu_memory_used_mb=np.mean(gpu_memory_peaks) if gpu_memory_peaks else 0.0,
            gpu_memory_peak_mb=np.max(gpu_memory_peaks) if gpu_memory_peaks else 0.0,
            gpu_utilization=0.0
        )
        
        # Clean up
        model.cpu()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return BenchmarkResult(
            model_name=model_name,
            model_type='deep_learning',
            segmentation=seg_metrics,
            contour=contour_metrics,
            measurement=None,
            latency=latency_metrics,
            hardware=hardware_metrics
        )
    
    def benchmark_yolo_model(
        self,
        model,
        test_loader,
        model_name: str
    ) -> BenchmarkResult:
        """
        Benchmark a YOLO-variant segmentation model.
        
        Specialized benchmark loop that interfaces with the Ultralytics 
        YOLO API, ensuring consistent preprocessing and post-processing 
        comparisons with other models.

        :param model: The YOLO model wrapper.
        :param test_loader: DataLoader for the test dataset.
        :param model_name: Label for result reporting.
        :return: BenchmarkResult with YOLO-specific timing metrics.
        """
        """
        Benchmark a YOLO segmentation model using its segment() API.
        
        YOLO models use the Ultralytics API and cannot be called with
        standard PyTorch forward passes. This method uses model.segment()
        which handles preprocessing, inference, and postprocessing internally.
        
        Args:
            model: YOLO model instance (BaseYOLOSegModel subclass)
            test_loader: Test data loader
            model_name: Model name for reporting
            
        Returns:
            BenchmarkResult
        """
        print(f"\n  Benchmarking {model_name} (YOLO)...")
        
        all_predictions = []
        all_ground_truths = []
        all_latencies = []
        all_measurement_metrics = []
        gpu_memory_usage = []
        
        # Clear GPU cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        # Warmup - run a few inferences to warm up the model
        for i, batch in enumerate(test_loader):
            if i >= self.num_warmup:
                break
            
            # Extract from dict batch
            images = batch['image'] if isinstance(batch, dict) else batch[0]
            
            # Dataset returns numpy arrays in (B, H, W, C) format already
            # For tensors in (C, H, W) format, transpose to (H, W, C)
            if torch.is_tensor(images):
                img = images[0].numpy().transpose(1, 2, 0)
            else:
                # Already in (H, W, C) format from numpy batch
                img = images[0]
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            _ = model.segment(img)
        
        # Benchmark
        for batch in tqdm(test_loader, desc=f"    {model_name}", leave=False):
            images = batch['image'] if isinstance(batch, dict) else batch[0]
            masks = batch['mask'] if isinstance(batch, dict) else batch[1]
            
            for i in range(len(images)):
                # Convert to numpy RGB image in (H, W, C) format
                if torch.is_tensor(images):
                    img = images[i].numpy().transpose(1, 2, 0)
                else:
                    # Already in (H, W, C) format from numpy batch
                    img = images[i]
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                gt_mask = masks[i].numpy() if torch.is_tensor(masks) else masks[i]
                
                # Multiple runs for timing
                timings = []
                pred_mask = None
                for run_idx in range(self.num_runs):
                    result = model.segment(img)
                    timings.append(result.inference_time_ms)
                    if run_idx == 0:
                        pred_mask = result.mask
                        gpu_memory_usage.append(result.gpu_memory_mb)
                
                avg_time = np.mean(timings)
                all_latencies.append(avg_time)
                
                # Resize predicted mask if needed
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
                
                all_predictions.append(pred_mask)
                all_ground_truths.append(gt_mask)
                
                # Dynamic calibration and measurement
                meas_metrics = self._post_prediction_calibrate(
                    pred_mask=pred_mask,
                    gt_mask=gt_mask,
                    shapes=batch.get('shapes', [[]])[i] if isinstance(batch, dict) else None
                )
                if meas_metrics:
                    all_measurement_metrics.append(meas_metrics)
        
        # Compute metrics
        seg_metrics = self._compute_segmentation_metrics(all_predictions, all_ground_truths)
        latency_metrics = self._compute_latency_metrics(all_latencies)
        contour_metrics = self._compute_contour_metrics(all_predictions, all_ground_truths)
        
        # Average measurement metrics
        measurement_metrics = self._average_measurement_metrics(all_measurement_metrics)
        
        # Hardware metrics
        hardware_metrics = HardwareMetrics(
            cpu_percent=0.0,
            gpu_memory_used_mb=np.mean(gpu_memory_usage) if gpu_memory_usage else 0.0,
            gpu_memory_peak_mb=np.max(gpu_memory_usage) if gpu_memory_usage else 0.0,
            gpu_utilization=0.0
        )
        
        # Clean up
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return BenchmarkResult(
            model_name=model_name,
            model_type='deep_learning',
            segmentation=seg_metrics,
            contour=contour_metrics,
            measurement=measurement_metrics,
            latency=latency_metrics,
            hardware=hardware_metrics
        )
    
    def _post_prediction_calibrate(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        shapes: Optional[List] = None
    ) -> Optional[MeasurementMetrics]:
        """
        Perform dynamic calibration and compute measurement metrics.
        
        Attempts to calibrate using reference objects in the prediction.
        If calibration succeeds, it computes measurement metrics by 
        comparing predicted dimensions against ground truth.

        :param pred_mask: Predicted segmentation mask.
        :param gt_mask: Ground truth segmentation mask.
        :param shapes: Original LabelMe shapes for GT measurements.
        :return: MeasurementMetrics if calibration was possible, else None.
        """
        try:
            # 1. Update calibration dynamically if possible
            # We use the prediction to find the calibration reference
            success = self.calibration.calibrate_from_predictions(pred_mask)
            
            # 2. Extract contours
            pred_contours = self.contour_extractor.extract_opencv(pred_mask).contours
            gt_contours = self.contour_extractor.extract_opencv(gt_mask).contours
            
            if not pred_contours or not gt_contours:
                return None
                
            # 3. Compute measurements
            pred_meas = self.measurement_computer.compute_all_measurements(pred_contours)
            gt_meas = self.measurement_computer.compute_all_measurements(gt_contours)
            
            # 4. Compute measurement error metrics
            return self.metrics_computer.compute_measurement_metrics(pred_meas, gt_meas)
            
        except Exception as e:
            # Silently fail for individual images to keep benchmarking running
            return None

    def _average_measurement_metrics(
        self,
        metrics_list: List[MeasurementMetrics]
    ) -> Optional[MeasurementMetrics]:
        """Aggregate measurement metrics across all samples."""
        if not metrics_list:
            return None
            
        return MeasurementMetrics(
            mae_mm=np.mean([m.mae_mm for m in metrics_list]),
            rmse_mm=np.mean([m.rmse_mm for m in metrics_list]),
            max_error_mm=np.max([m.max_error_mm for m in metrics_list]),
            mean_relative_error=np.mean([m.mean_relative_error for m in metrics_list])
        )
    
    def _compute_segmentation_metrics(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray]
    ) -> SegmentationMetrics:
        """Compute segmentation metrics."""
        all_metrics = []
        
        for pred, gt in zip(predictions, ground_truths):
            # Normalize masks to binary
            pred_bin = (pred > 127) if pred.max() > 1 else (pred > 0)
            gt_bin = (gt > 127) if gt.max() > 1 else (gt > 0)
            
            # Convert to uint8 for metrics computation
            pred_mask = pred_bin.astype(np.uint8) * 255
            gt_mask = gt_bin.astype(np.uint8) * 255
            
            # Compute metrics using existing method
            metrics = self.metrics_computer.compute_segmentation_metrics(pred_mask, gt_mask)
            all_metrics.append(metrics)
        
        # Average all metrics
        return SegmentationMetrics(
            iou=np.mean([m.iou for m in all_metrics]),
            dice=np.mean([m.dice for m in all_metrics]),
            precision=np.mean([m.precision for m in all_metrics]),
            recall=np.mean([m.recall for m in all_metrics]),
            accuracy=np.mean([m.accuracy for m in all_metrics]),
            f1_score=np.mean([m.f1_score for m in all_metrics]),
            boundary_f1=np.mean([getattr(m, 'boundary_f1', 0.0) for m in all_metrics]),
            hausdorff_distance=np.mean([getattr(m, 'hausdorff_distance', 0.0) for m in all_metrics])
        )
    
    def _compute_latency_metrics(
        self,
        latencies: List[float]
    ) -> LatencyMetrics:
        """Compute latency metrics."""
        mean_latency = np.mean(latencies)
        
        return LatencyMetrics(
            preprocessing_time_ms=0.0,  # Included in inference
            inference_time_ms=mean_latency,
            postprocessing_time_ms=0.0,
            total_time_ms=mean_latency,
            fps=1000.0 / mean_latency if mean_latency > 0 else 0.0,
            std_dev_ms=np.std(latencies)
        )
    
    def _compute_contour_metrics(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray]
    ) -> ContourMetrics:
        """Compute contour metrics."""
        avg_contour_dists = []
        
        for pred, gt in zip(predictions, ground_truths):
            pred_bin = ((pred > 127) * 255).astype(np.uint8)
            gt_bin = ((gt > 127) * 255).astype(np.uint8)
            
            pred_result = self.contour_extractor.extract_opencv(pred_bin)
            gt_result = self.contour_extractor.extract_opencv(gt_bin)
            
            # Extract contour lists from ContourResult objects
            pred_contours = pred_result.contours
            gt_contours = gt_result.contours
            
            if pred_contours and gt_contours:
                # Compute average contour distance
                try:
                    pred_pts = np.vstack([c.reshape(-1, 2) for c in pred_contours])
                    gt_pts = np.vstack([c.reshape(-1, 2) for c in gt_contours])
                    
                    # Nearest neighbor distances
                    from scipy.spatial.distance import cdist
                    if len(pred_pts) > 0 and len(gt_pts) > 0:
                        dists = cdist(pred_pts, gt_pts)
                        avg_dist = np.mean(np.min(dists, axis=1))
                        avg_contour_dists.append(avg_dist)
                except Exception:
                    pass  # Skip if contours can't be processed
        
        return ContourMetrics(
            average_contour_distance=np.mean(avg_contour_dists) if avg_contour_dists else 0.0,
            contour_completeness=0.0,
            smoothness_score=0.0
        )
    
    def run_benchmark(
        self,
        models_to_run: List[str] = None,
        run_deep_learning: bool = True
    ) -> Dict[str, BenchmarkResult]:
        """
        Run complete benchmark.
        
        Args:
            models_to_run: Specific models to benchmark (None = all)
            run_deep_learning: Whether to run deep learning models
            
        Returns:
            Dict mapping model name to BenchmarkResult
        """
        # Load data
        train_loader, val_loader, test_loader = self.load_data()
        
        # Determine which models to run
        if models_to_run is None:
            models = {}
            if run_deep_learning:
                models.update(DEEP_LEARNING_MODELS)
        else:
            models = {m: ALL_MODELS[m] for m in models_to_run if m in ALL_MODELS}
        
        print("\n" + "="*60)
        print(f"Running Benchmark: {len(models)} models")
        print("="*60)
        
        # Benchmark each model
        for model_name, model_class in models.items():
            try:
                if 'yolo' in model_name.lower():
                    # YOLO models use Ultralytics API, not standard PyTorch
                    model = model_class()
                    result = self.benchmark_yolo_model(
                        model, test_loader, model_name
                    )
                else:
                    # All wrapper model classes use default init (no args)
                    model = model_class()
                    result = self.benchmark_deep_learning_model(
                        model, test_loader, model_name
                    )
                
                self.results[model_name] = result
                self._print_result_summary(result)
                
            except Exception as e:
                print(f"  Error benchmarking {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return self.results
    
    def _print_result_summary(self, result: BenchmarkResult):
        """Print summary of benchmark result."""
        print(f"\n    Results for {result.model_name}:")
        if result.segmentation:
            print(f"      IoU: {result.segmentation.iou:.4f}")
            print(f"      Dice: {result.segmentation.dice:.4f}")
            print(f"      F1: {result.segmentation.f1_score:.4f}")
        if result.latency:
            print(f"      Latency: {result.latency.total_time_ms:.2f} ms")
            print(f"      FPS: {result.latency.fps:.1f}")
        if result.hardware and result.hardware.gpu_memory_peak_mb > 0:
            print(f"      GPU Memory: {result.hardware.gpu_memory_peak_mb:.1f} MB")
    
    def generate_report(self):
        """Generate comprehensive benchmark report."""
        print("\n" + "="*60)
        print("Generating Report")
        print("="*60)
        
        # Generate plots
        seg_metrics = {m: r.segmentation for m, r in self.results.items() if r.segmentation}
        latency_metrics = {m: r.latency for m, r in self.results.items() if r.latency}
        hardware_metrics = {m: r.hardware for m, r in self.results.items() if r.hardware}
        
        if seg_metrics:
            self.plotter.plot_accuracy_comparison(seg_metrics)
            print("  Created: accuracy_comparison.png")
        
        if latency_metrics:
            self.plotter.plot_latency_comparison(latency_metrics)
            print("  Created: latency_comparison.png")
        
        if seg_metrics and latency_metrics:
            self.plotter.plot_accuracy_vs_latency(seg_metrics, latency_metrics)
            print("  Created: accuracy_vs_latency.png")
        
        if seg_metrics and hardware_metrics:
            self.plotter.plot_accuracy_vs_gpu_memory(seg_metrics, hardware_metrics)
            print("  Created: accuracy_vs_gpu_memory.png")
        
        # Create ranking table
        self._create_ranking_table()
        
        # Save detailed results as JSON
        self._save_json_results()
        
        print(f"\n  Reports saved to: {self.output_dir}")
    
    def _create_ranking_table(self):
        """Create ranked model comparison table."""
        results_list = []
        
        for model_name, result in self.results.items():
            row = {
                'model': model_name,
                'type': result.model_type,
                'iou': result.segmentation.iou if result.segmentation else 0,
                'dice': result.segmentation.dice if result.segmentation else 0,
                'f1': result.segmentation.f1_score if result.segmentation else 0,
                'latency_ms': result.latency.total_time_ms if result.latency else 0,
                'fps': result.latency.fps if result.latency else 0,
                'gpu_mb': result.hardware.gpu_memory_peak_mb if result.hardware else 0
            }
            results_list.append(row)
        
        # Sort by IoU
        results_list.sort(key=lambda x: x['iou'], reverse=True)
        
        # Generate markdown table
        report = "# Benchmark Results\n\n"
        report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Model Ranking (by IoU)\n\n"
        report += "| Rank | Model | Type | IoU | Dice | F1 | Latency (ms) | FPS | GPU (MB) |\n"
        report += "|------|-------|------|-----|------|----|--------------|----|----------|\n"
        
        for i, row in enumerate(results_list, 1):
            report += f"| {i} | {row['model']} | {row['type']} | "
            report += f"{row['iou']:.4f} | {row['dice']:.4f} | {row['f1']:.4f} | "
            report += f"{row['latency_ms']:.2f} | {row['fps']:.1f} | {row['gpu_mb']:.1f} |\n"
        
        report += "\n## Deployment Recommendation\n\n"
        
        # Find best models
        if results_list:
            best_accuracy = results_list[0]
            report += f"- **Best Accuracy**: {best_accuracy['model']} (IoU: {best_accuracy['iou']:.4f})\n"
            
            best_speed = min(results_list, key=lambda x: x['latency_ms'] if x['latency_ms'] > 0 else float('inf'))
            report += f"- **Fastest**: {best_speed['model']} (Latency: {best_speed['latency_ms']:.2f} ms, FPS: {best_speed['fps']:.1f})\n"
            
            # Best tradeoff - simple scoring
            for r in results_list:
                r['score'] = r['iou'] * 0.6 + (1 / (r['latency_ms'] + 1)) * 0.3 + (1 - r['gpu_mb']/4000) * 0.1
            
            best_tradeoff = max(results_list, key=lambda x: x.get('score', 0))
            report += f"- **Best Tradeoff**: {best_tradeoff['model']} (Score: balanced accuracy/speed/memory)\n"
        
        # Save report
        report_path = os.path.join(self.output_dir, "benchmark_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"  Created: benchmark_report.md")
    
    def _save_json_results(self):
        """Save detailed results as JSON."""
        json_results = {}
        
        for model_name, result in self.results.items():
            json_results[model_name] = {
                'model_type': result.model_type,
                'segmentation': {
                    'iou': result.segmentation.iou,
                    'dice': result.segmentation.dice,
                    'precision': result.segmentation.precision,
                    'recall': result.segmentation.recall,
                    'f1_score': result.segmentation.f1_score,
                    'boundary_f1': result.segmentation.boundary_f1,
                    'hausdorff_distance': result.segmentation.hausdorff_distance
                } if result.segmentation else None,
                'latency': {
                    'inference_time_ms': result.latency.inference_time_ms,
                    'total_time_ms': result.latency.total_time_ms,
                    'fps': result.latency.fps,
                    'std_dev_ms': result.latency.std_dev_ms
                } if result.latency else None,
                'hardware': {
                    'gpu_memory_used_mb': result.hardware.gpu_memory_used_mb,
                    'gpu_memory_peak_mb': result.hardware.gpu_memory_peak_mb,
                    'gpu_utilization': result.hardware.gpu_utilization
                } if result.hardware else None,
                'measurement': {
                    'mae_mm': result.measurement.mae_mm,
                    'rmse_mm': result.measurement.rmse_mm,
                    'max_error_mm': result.measurement.max_error_mm,
                    'mean_relative_error': result.measurement.mean_relative_error
                } if result.measurement else None
            }
        
        json_path = os.path.join(self.output_dir, "benchmark_results.json")
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"  Created: benchmark_results.json")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Industrial Chignon Detection Benchmark"
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'benchmark', 'full'],
        default='benchmark',
        help='Mode: train (training only), benchmark (inference only), full (both)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        help='Models to use: all, classical, deep_learning, or specific model names'
    )
    
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Compute device (cuda/cpu)'
    )
    
    parser.add_argument(
        '--warmup',
        type=int,
        default=5,
        help='Number of warmup iterations for benchmarking'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=20,
        help='Number of benchmark runs per sample'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='Weight decay for optimizer'
    )
    
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=10,
        help='Early stopping patience (epochs)'
    )
    
    parser.add_argument(
        '--optimizer',
        default='adamw',
        choices=['adamw', 'adam', 'sgd', 'rmsprop', 'auto'],
        help='Optimizer for training (auto uses model default)'
    )
    
    parser.add_argument(
        '--scheduler',
        default='cosine',
        choices=['cosine', 'step', 'plateau'],
        help='LR scheduler for PyTorch model training'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training (especially YOLO models)'
    )
    
    parser.add_argument(
        '--output',
        default=None,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--labels',
        nargs='+',
        default=None,
        help='Label names to include in training (e.g. --labels mecparts chignon). '
             'If not specified, all labels are used.'
    )
    
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='Enable automatic mixed precision training (FP16). Disabled by default for stability.'
    )
    
    parser.add_argument(
        '--no-amp',
        action='store_true',
        default=False,
        help='Disable automatic mixed precision training'
    )
    
    parser.add_argument(
        '--loss',
        default='bce_dice',
        choices=['combined', 'bce_dice'],
        help='Loss function: "bce_dice" (BCE+Dice, default) or "combined" (BCE+Dice+Boundary+Lovász)'
    )
    
    parser.add_argument(
        '--augment',
        action='store_true',
        default=False,
        help='Run offline data augmentation before training/benchmark. '
             'Generates augmented data from data/ into outputs/augmented_data/.'
    )
    
    return parser.parse_args()


def run_training(
    args,
    train_loader,
    val_loader,
    models_to_train: List[str]
) -> Dict[str, TrainingHistory]:
    """
    Run training with comprehensive data collection.
    
    Args:
        args: Command line arguments
        train_loader: Training data loader
        val_loader: Validation data loader
        models_to_train: List of model names to train
        
    Returns:
        Dict of model_name -> TrainingHistory
    """
    print("\n" + "="*60)
    print("Training Mode - Comprehensive Data Collection")
    print("="*60)
    print(f"Models to train: {models_to_train}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Early stopping: {args.early_stopping} epochs")
    print()
    
    histories = {}
    
    # Separate YOLO and non-YOLO models
    yolo_models = [m for m in models_to_train if 'yolo' in m.lower()]
    pytorch_models = [m for m in models_to_train if 'yolo' not in m.lower()]
    
    # Train PyTorch models (UNet, DeepLab, SegFormer, Mask R-CNN)
    for model_name in pytorch_models:
        if model_name not in DEEP_LEARNING_MODELS:
            print(f"Skipping {model_name} - only DL models can be trained")
            continue
        
        # Skip Mask R-CNN - requires specialized training loop with
        # target annotations (bounding boxes + labels) in the forward pass
        if model_name == 'mask_rcnn':
            print(f"\nSkipping {model_name} - requires specialized training loop (detection targets)")
            continue
        
        print(f"\n{'#'*60}")
        print(f"# Training: {model_name}")
        print(f"{'#'*60}")
        
        try:
            # Instantiate model
            model_class = DEEP_LEARNING_MODELS[model_name]
            if model_name == 'mask_rcnn':
                model = model_class(pretrained=True)
            else:
                model = model_class()
            
            # Create trainer with comprehensive data collection
            trainer = ComprehensiveTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                model_name=model_name,
                device=args.device,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                num_epochs=args.epochs,
                early_stopping_patience=args.early_stopping,
                optimizer_name=args.optimizer if args.optimizer != 'auto' else 'adamw',
                scheduler_name=args.scheduler,
                output_dir=config.benchmark.output_dir,
                save_every_epoch=True,
                hardware_sample_interval=0.1,
                use_amp=args.amp and not args.no_amp,
                loss_mode=args.loss
            )
            
            # Run training
            history = trainer.train()
            histories[model_name] = history
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Train YOLO models (requires data.yaml)
    if yolo_models:
        # If labels are specified, we must re-prepare the YOLO dataset
        # to ensure the classes match the requested filter
        if getattr(args, 'labels', None):
            print(f"\nRe-preparing YOLO dataset with label filter: {args.labels}")
            yolo_source = str(config.PROJECT_ROOT / "data")
            yolo_output = config.dataset.yolo_dataset_path
            data_yaml = prepare_yolo_dataset(
                source_dir=yolo_source,
                output_dir=yolo_output,
                label_filter=args.labels
            )
        else:
            data_yaml = config.dataset.yolo_data_yaml
        
        if not os.path.exists(data_yaml):
            print(f"\nWarning: YOLO data.yaml not found at {data_yaml}")
            print("YOLO models require a properly prepared dataset.")
            print("Please run: python prepare_yolo_dataset.py")
            print("This will convert LabelMe annotations to YOLO format.")
        
        else:
            print(f"\nUsing YOLO dataset: {data_yaml}")
            # Map model names to variants
            yolo_variant_map = {
                'yolov8n_seg': 'yolov8n-seg',
                'yolov8s_seg': 'yolov8s-seg',
                'yolov8m_seg': 'yolov8m-seg',
                'yolov11n_seg': 'yolo11n-seg',
                'yolov11s_seg': 'yolo11s-seg',
                'yolov11m_seg': 'yolo11m-seg',
                'yolov26n_seg': 'yolo26n-seg',
                'yolov26s_seg': 'yolo26s-seg',
                'yolov26m_seg': 'yolo26m-seg',
            }
            
            for model_name in yolo_models:
                if model_name not in DEEP_LEARNING_MODELS:
                    print(f"Skipping {model_name} - not in model registry")
                    continue
                
                variant = yolo_variant_map.get(model_name, model_name)
                
                print(f"\n{'#'*60}")
                print(f"# Training YOLO: {model_name} ({variant})")
                print(f"{'#'*60}")
                
                try:
                    # Map optimizer name for YOLO (Ultralytics format)
                    yolo_optimizer = args.optimizer.upper() if args.optimizer != 'auto' else 'auto'
                    if yolo_optimizer == 'ADAMW':
                        yolo_optimizer = 'AdamW'
                    elif yolo_optimizer == 'RMSPROP':
                        yolo_optimizer = 'RMSProp'
                    
                    yolo_trainer = YOLOTrainer(
                        model_variant=variant,
                        data_yaml=data_yaml,
                        model_name=model_name,
                        device=0 if args.device == 'cuda' else 'cpu',
                        epochs=args.epochs,
                        imgsz=512,  # Match preprocessing size
                        batch=args.batch_size if hasattr(args, 'batch_size') else 8,
                        patience=args.early_stopping,
                        optimizer=yolo_optimizer,
                        lr0=args.lr if args.lr != 1e-4 else 0.01,
                        weight_decay=args.weight_decay if args.weight_decay != 1e-5 else 0.0005
                    )
                    
                    history = yolo_trainer.train()
                    histories[model_name] = history
                    
                    # Clear GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Generate training comparison visualizations
    if histories:
        print("\n" + "="*60)
        print("Generating Training Comparison Plots")
        print("="*60)
        
        visualizer = TrainingVisualizer()
        visualizer.generate_all_training_plots(histories)
        
        # Generate comparison summary
        comparison = compare_training_results(histories)
        
        # Save comparison
        comparison_path = os.path.join(
            config.benchmark.output_dir,
            'training_comparison.json'
        )
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nTraining Comparison Summary:")
        print(f"  Best model by IoU: {comparison['best_model_by_iou']}")
        print(f"  Fastest model: {comparison['fastest_model']}")
        print(f"  Most efficient: {comparison['most_efficient_model']}")
    
    return histories


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("Industrial Chignon Detection Benchmark")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {args.device}")
    print(f"  Models: {args.models}")
    if args.mode in ['train', 'full']:
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Early stopping: {args.early_stopping}")
    if args.mode in ['benchmark', 'full']:
        print(f"  Warmup runs: {args.warmup}")
        print(f"  Benchmark runs: {args.runs}")
    
    # Set random seeds
    np.random.seed(config.benchmark.seed)
    torch.manual_seed(config.benchmark.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.benchmark.seed)
    
    # Determine which models to use
    models_to_run = None
    run_deep_learning = True
    
    if args.models != ['all']:
        if args.models == ['deep_learning']:
            models_to_run = list(DEEP_LEARNING_MODELS.keys())
        elif args.models == ['yolo']:
            models_to_run = [m for m in DEEP_LEARNING_MODELS.keys() if 'yolo' in m.lower()]
        else:
            models_to_run = args.models
            run_deep_learning = any(m in DEEP_LEARNING_MODELS for m in models_to_run)
    else:
        models_to_run = list(ALL_MODELS.keys())
    
    # Load data
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    
    label_filter = getattr(args, 'labels', None)
    
    # Run augmentation if requested
    if args.augment:
        from src.data.augmentation import run_augmentation
        augment_input = str(config.dataset.root_path)
        augment_output = os.path.join(_PROJECT_ROOT, "outputs", "augmented_data")
        print(f"\nRunning data augmentation...")
        print(f"  Input:  {augment_input}")
        print(f"  Output: {augment_output}")
        run_augmentation(augment_input, augment_output)
        # Override data root to use augmented data
        config.dataset.root_path = augment_output
        print(f"  Data root overridden to: {augment_output}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        root_path=str(config.dataset.root_path),
        batch_size=config.benchmark.batch_size,
        label_filter=label_filter
    )
    
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    training_histories = None
    
    # Training mode
    if args.mode in ['train', 'full']:
        dl_models = [m for m in models_to_run if m in DEEP_LEARNING_MODELS]
        if dl_models:
            training_histories = run_training(
                args, train_loader, val_loader, dl_models
            )
        else:
            print("No deep learning models specified for training")
    
    # Benchmark mode
    if args.mode in ['benchmark', 'full']:
        runner = BenchmarkRunner(
            device=args.device,
            output_dir=args.output,
            num_warmup=args.warmup,
            num_runs=args.runs,
            label_filter=label_filter
        )
        
        # Run benchmark
        results = runner.run_benchmark(
            models_to_run=models_to_run if models_to_run else None,
            run_deep_learning=run_deep_learning
        )
        
        # Generate report
        runner.generate_report()
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    
    if training_histories:
        print("\nTraining Results Summary:")
        for name, hist in training_histories.items():
            print(f"  {name}: Best IoU={hist.best_val_iou:.4f} @ epoch {hist.best_epoch}")
    
    print(f"\nResults saved to: {config.benchmark.output_dir}")


if __name__ == "__main__":
    main()
