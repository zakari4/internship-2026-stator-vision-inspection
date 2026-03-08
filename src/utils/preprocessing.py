"""
Preprocessing pipeline for industrial segmentation benchmark.
Handles image loading, normalization, filtering, and enhancement.
"""

import time
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import config


@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline"""
    image: np.ndarray
    gray: np.ndarray
    original_size: Tuple[int, int]
    preprocessing_time_ms: float
    steps_time: Dict[str, float]


class PreprocessingPipeline:
    """
    Unified preprocessing pipeline for all segmentation methods.
    Ensures consistent preprocessing across classical and deep learning models.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = None,
        use_bilateral: bool = None,
        use_gaussian: bool = None,
        normalize: bool = None,
        use_clahe: bool = None,
        use_morph_cleanup: bool = None
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            target_size: Target size for resizing (height, width)
            use_bilateral: Whether to use bilateral filter
            use_gaussian: Whether to use Gaussian blur
            normalize: Whether to normalize intensity
            use_clahe: Whether to apply CLAHE
            use_morph_cleanup: Whether to apply morphological cleanup
        """
        cfg = config.preprocessing
        
        self.target_size = target_size or cfg.target_size
        self.use_bilateral = use_bilateral if use_bilateral is not None else cfg.use_bilateral_filter
        self.use_gaussian = use_gaussian if use_gaussian is not None else cfg.use_gaussian_blur
        self.normalize = normalize if normalize is not None else cfg.normalize_intensity
        self.use_clahe = use_clahe if use_clahe is not None else cfg.use_clahe
        self.use_morph_cleanup = use_morph_cleanup if use_morph_cleanup is not None else cfg.use_morphological_cleanup
        
        # Bilateral filter params
        self.bilateral_d = cfg.bilateral_d
        self.bilateral_sigma_color = cfg.bilateral_sigma_color
        self.bilateral_sigma_space = cfg.bilateral_sigma_space
        
        # Gaussian blur params
        self.gaussian_kernel_size = cfg.gaussian_kernel_size
        
        # CLAHE params
        self.clahe_clip_limit = cfg.clahe_clip_limit
        self.clahe_tile_grid_size = cfg.clahe_tile_grid_size
        
        # Morphological params
        self.morph_kernel_size = cfg.morph_kernel_size
        
        # Initialize CLAHE
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size
            )
    
    def resize(self, image: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Resize image (and optionally mask) to target size.
        
        Args:
            image: Input image
            mask: Optional mask to resize
            
        Returns:
            Tuple of (resized_image, resized_mask)
        """
        if self.target_size is None:
            return image, mask
        
        resized_image = cv2.resize(
            image, 
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        resized_mask = None
        if mask is not None:
            resized_mask = cv2.resize(
                mask,
                (self.target_size[1], self.target_size[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        return resized_image, resized_mask
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        return image
    
    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter for edge-preserving noise reduction."""
        return cv2.bilateralFilter(
            image,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur for noise reduction."""
        kernel_size = (self.gaussian_kernel_size, self.gaussian_kernel_size)
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    def normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity to [0, 255] range."""
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            image = (image * 255).astype(np.uint8)
        return image
    
    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement."""
        return self.clahe.apply(gray)
    
    def apply_morphological_cleanup(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological opening-closing for cleanup."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        # Opening (erosion followed by dilation) - removes small noise
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        # Closing (dilation followed by erosion) - fills small holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed
    
    def process(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
        return_timings: bool = True
    ) -> PreprocessingResult:
        """
        Apply full preprocessing pipeline to an image.
        
        Args:
            image: Input image (BGR or RGB)
            mask: Optional ground truth mask
            return_timings: Whether to measure and return timings
            
        Returns:
            PreprocessingResult with processed image and timings
        """
        steps_time = {}
        total_start = time.perf_counter()
        original_size = image.shape[:2]
        
        # Step 1: Resize
        start = time.perf_counter()
        image, mask = self.resize(image, mask)
        steps_time["resize"] = (time.perf_counter() - start) * 1000
        
        # Step 2: Convert to grayscale
        start = time.perf_counter()
        gray = self.to_grayscale(image)
        steps_time["grayscale"] = (time.perf_counter() - start) * 1000
        
        # Step 3: Noise reduction
        start = time.perf_counter()
        if self.use_bilateral:
            gray = self.apply_bilateral_filter(gray)
        elif self.use_gaussian:
            gray = self.apply_gaussian_blur(gray)
        steps_time["noise_reduction"] = (time.perf_counter() - start) * 1000
        
        # Step 4: Intensity normalization
        start = time.perf_counter()
        if self.normalize:
            gray = self.normalize_intensity(gray)
        steps_time["normalization"] = (time.perf_counter() - start) * 1000
        
        # Step 5: CLAHE
        start = time.perf_counter()
        if self.use_clahe:
            gray = self.apply_clahe(gray)
        steps_time["clahe"] = (time.perf_counter() - start) * 1000
        
        # Step 6: Morphological cleanup
        start = time.perf_counter()
        if self.use_morph_cleanup:
            gray = self.apply_morphological_cleanup(gray)
        steps_time["morph_cleanup"] = (time.perf_counter() - start) * 1000
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        return PreprocessingResult(
            image=image,
            gray=gray,
            original_size=original_size,
            preprocessing_time_ms=total_time,
            steps_time=steps_time
        )
    
    def process_for_deep_learning(
        self,
        image: np.ndarray,
        normalize_for_model: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Process image for deep learning models.
        
        Args:
            image: Input image (BGR or RGB)
            normalize_for_model: Whether to normalize to [0, 1]
            
        Returns:
            Tuple of (processed_image, preprocessing_time_ms)
        """
        start = time.perf_counter()
        
        # Resize
        image, _ = self.resize(image, None)
        
        # Convert to float and normalize
        if normalize_for_model:
            image = image.astype(np.float32) / 255.0
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
        
        # Convert to CHW format for PyTorch
        image = np.transpose(image, (2, 0, 1))
        
        preprocessing_time = (time.perf_counter() - start) * 1000
        
        return image, preprocessing_time


def preprocess_batch(
    images: np.ndarray,
    pipeline: PreprocessingPipeline = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Preprocess a batch of images.
    
    Args:
        images: Batch of images (N, H, W, C)
        pipeline: Preprocessing pipeline to use
        
    Returns:
        Tuple of (processed_images, grayscale_images, total_time_ms)
    """
    if pipeline is None:
        pipeline = PreprocessingPipeline()
    
    processed_images = []
    gray_images = []
    total_time = 0
    
    for image in images:
        result = pipeline.process(image)
        processed_images.append(result.image)
        gray_images.append(result.gray)
        total_time += result.preprocessing_time_ms
    
    return np.stack(processed_images), np.stack(gray_images), total_time


# Create default pipeline instance
default_pipeline = PreprocessingPipeline()
