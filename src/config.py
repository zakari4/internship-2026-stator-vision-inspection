"""
Configuration file for the industrial segmentation benchmark.
Contains all hyperparameters, paths, and settings.

All paths are resolved dynamically relative to PROJECT_ROOT
so the project can run from any location.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import torch
import random

# ==============================================================================
# Project Root — resolved dynamically (src/ is one level below root)
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==============================================================================
# Reproducibility — Fix random seeds
# ==============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str = "Custom_Chignon_Industrial_Dataset"
    root_path: str = str(PROJECT_ROOT / "data")
    yolo_dataset_path: str = str(PROJECT_ROOT / "outputs" / "yolo_dataset")
    
    # File patterns
    image_pattern: str = "*.jpg"
    annotation_pattern: str = "*.json"
    
    # Dataset versions
    versions: List[str] = field(default_factory=lambda: ["run_*", "v2_run_*", "v3_run_*"])
    
    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Classes
    classes: List[str] = field(default_factory=lambda: [
        "background", "michanical_part", "magnet", "circle"
    ])
    num_classes: int = 2  # Binary: background vs all foreground labels
    
    # Label filter (select specific labels for training/benchmark)
    # If None, all labels in self.classes are used.
    label_filter: Optional[List[str]] = None
    
    @property
    def data_dir(self) -> str:
        """Alias for root_path."""
        return self.root_path
    
    @property
    def yolo_data_yaml(self) -> str:
        """Path to YOLO data.yaml file."""
        return f"{self.yolo_dataset_path}/data.yaml"


@dataclass  
class PreprocessingConfig:
    """Preprocessing pipeline configuration"""
    # Image size normalization
    target_size: tuple = (512, 512)  # (height, width)
    
    # Noise reduction
    use_bilateral_filter: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75
    bilateral_sigma_space: float = 75
    
    # Alternative: Gaussian blur
    use_gaussian_blur: bool = False
    gaussian_kernel_size: int = 5
    
    # Intensity normalization
    normalize_intensity: bool = True
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)
    
    # Morphological cleanup
    use_morphological_cleanup: bool = False
    morph_kernel_size: int = 3


@dataclass
class BenchmarkConfig:
    """Benchmark execution configuration"""
    # Reproducibility
    seed: int = 42
    
    # Timing
    warmup_runs: int = 3
    benchmark_runs: int = 10
    
    # Batching (minimum 2 for models with BatchNorm like DeepLabV3)
    batch_size: int = 4
    
    # GPU settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False  # Disabled as per requirements
    
    # Output paths — dynamic, relative to PROJECT_ROOT
    output_dir: str = str(PROJECT_ROOT / "outputs" / "results")
    logs_dir: str = str(PROJECT_ROOT / "outputs" / "results" / "logs")
    plots_dir: str = str(PROJECT_ROOT / "outputs" / "results" / "plots")
    visual_dir: str = str(PROJECT_ROOT / "outputs" / "results" / "visualizations")
    results_dir: str = str(PROJECT_ROOT / "outputs" / "results")


@dataclass
class DeepLearningModelConfig:
    """Deep learning model configuration"""
    # UNet Lightweight
    unet_lightweight_enabled: bool = True
    unet_light_enabled: bool = True  # Alias for backward compat
    unet_light_base_filters: int = 32
    
    # UNet with ResNet18 encoder
    unet_resnet18_enabled: bool = True
    
    # DeepLabV3 with MobileNet
    deeplabv3_enabled: bool = True
    deeplabv3_mobilenet_enabled: bool = True  # Alias
    
    # SegFormer B0
    segformer_enabled: bool = True
    segformer_b0_enabled: bool = True  # Alias
    
    # Mask R-CNN with ResNet50
    mask_rcnn_enabled: bool = True
    
    # YOLO Segmentation Models
    yolov8_seg_enabled: bool = True
    yolov11_seg_enabled: bool = True
    yolov26_seg_enabled: bool = True
    
    # RT-DETR and variants
    rtdetr_enabled: bool = True
    rf_detr_enabled: bool = True
    codetr_enabled: bool = True
    
    # ScyllaNet
    scyllanet_enabled: bool = True
    
    # Foundation Models for Segmentation
    sam2_enabled: bool = True
    internimage_enabled: bool = True
    setr_enabled: bool = True
    segnext_enabled: bool = True
    oneformer_enabled: bool = True
    mask2former_enabled: bool = True
    
    # SOTA Edge Detection Models
    hed_enabled: bool = True  # Holistically-Nested Edge Detection
    rcf_enabled: bool = True  # Richer Convolutional Features
    bdcn_enabled: bool = True  # Bi-Directional Cascade Network
    pidinet_enabled: bool = True  # Pixel Difference Networks
    teed_enabled: bool = True  # Tiny and Efficient Edge Detector
    ldc_enabled: bool = True  # Lightweight Dense CNN
    
    # Weights directory
    weights_dir: str = str(PROJECT_ROOT / "weights")
    
    # Training settings (if needed)
    optimizer: str = 'adamw'  # adamw, adam, sgd, rmsprop
    scheduler: str = 'cosine'  # cosine, step, plateau
    early_stopping_patience: int = 10
    learning_rate: float = 1e-4
    epochs: int = 50
    weight_decay: float = 1e-4


@dataclass
class ContourExtractionConfig:
    """Contour extraction configuration"""
    # Area filtering
    min_area: int = 100
    max_area: Optional[int] = None
    
    # Length filtering
    min_length: int = 50
    
    # Shape descriptors thresholds
    min_circularity: float = 0.0
    max_circularity: float = 1.0
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float = 10.0
    
    # Contour approximation
    epsilon_factor: float = 0.02  # Factor for cv2.approxPolyDP


@dataclass
class MeasurementConfig:
    """Geometric measurement configuration"""
    # Pixel-to-metric conversion (mm per pixel)
    # Set to 1.0 if no calibration available
    pixel_to_mm: float = 1.0
    
    # Distance computation methods
    use_euclidean: bool = True
    use_model_based: bool = True
    
    # Sub-pixel refinement
    use_subpixel: bool = True
    
    # Line fitting
    line_fitting_method: str = "ransac"  # or "least_squares"
    ransac_threshold: float = 1.0
    
    # Circle fitting
    circle_fitting_enabled: bool = True
    
    # Dynamic calibration settings
    reference_label_name: str = "michanical_part"
    reference_known_dimension_mm: float = 52.0
    reference_dimension_type: str = "diameter"
    
    # --- Method A: ArUco / Reference Object ---
    aruco_marker_size_mm: float = 50.0
    aruco_dictionary: str = "DICT_4X4_50"
    
    # --- Method B: Geometric Camera Calibration ---
    sensor_width_mm: float = 6.17       # 1/2.3" sensor
    focal_length_mm: float = 4.0
    object_distance_mm: float = 300.0


# ==============================================================================
# Create combined configuration
# ==============================================================================
@dataclass
class Config:
    """Master configuration combining all sub-configs"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    deep_learning: DeepLearningModelConfig = field(default_factory=DeepLearningModelConfig)
    # Aliases for backward compatibility
    models: DeepLearningModelConfig = field(default_factory=DeepLearningModelConfig)
    contour: ContourExtractionConfig = field(default_factory=ContourExtractionConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    
    def __post_init__(self):
        """Create output directories and sync aliases."""
        # Sync models alias with deep_learning
        self.models = self.deep_learning
        
        os.makedirs(self.benchmark.output_dir, exist_ok=True)
        os.makedirs(self.benchmark.logs_dir, exist_ok=True)
        os.makedirs(self.benchmark.plots_dir, exist_ok=True)
        os.makedirs(self.benchmark.visual_dir, exist_ok=True)
        os.makedirs(self.deep_learning.weights_dir, exist_ok=True)


# Global config instance
config = Config()
