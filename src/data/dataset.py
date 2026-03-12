"""
Dataset loader and utilities for the industrial segmentation benchmark.
Handles LabelMe format annotations and creates ground truth masks.
"""

import os
import json
import glob
import random
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import config, SEED


# Global registry of all possible labels in the dataset
LABELS = [
    "background",
    "michanical_part",
    "magnet",
    "circle"
]


@dataclass
class Sample:
    """
    Representation of a single image and its corresponding annotations.

    Stores the file paths and lazy-loaded data for a sample in the dataset,
    serving as a container for images, masks, and metadata.

    :param image_path: Absolute path to the image file.
    :param annotation_path: Absolute path to the LabelMe JSON file.
    :param image: Optional cached numpy array of the image.
    :param mask: Optional cached numpy array of the ground truth mask.
    :param contours: Optional list of pre-extracted ground truth contours.
    :param metadata: Dictionary of auxiliary image metadata.
    """
    image_path: str
    annotation_path: str
    image: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    contours: Optional[List[np.ndarray]] = None
    metadata: Optional[Dict] = None


def load_labelme_annotation(json_path: str) -> Dict:
    """
    Load raw LabelMe format annotation from a JSON file.
    
    :param json_path: Path to the JSON annotation file.
    :return: Dictionary containing the full annotation structure.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def create_mask_from_shapes(
    shapes: List[Dict],
    image_height: int,
    image_width: int,
    class_mapping: Optional[Dict[str, int]] = None,
    label_filter: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Convert LabelMe shapes into a binary mask and extract object contours.

    Parses multiple shape types (polygons, linestrips, lines) and applies
    them to a zero-initialized mask. For linestrips, the path is automatically
    closed to form an enclosed area.

    :param shapes: List of shape dictionaries from raw JSON.
    :param image_height: Targeted height of the output mask.
    :param image_width: Targeted width of the output mask.
    :param class_mapping: Mapping of labels to integer indices (0=background).
    :param label_filter: List of labels to include (others are ignored).
    :return: A tuple of (binary mask array, list of contour arrays).
    """
    if class_mapping is None:
        # Multi-class mapping: each label gets its own index
        # background=0, michanical_part=1, magnet=2, circle=3
        class_mapping = {label: i for i, label in enumerate(LABELS)}
    
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    contours = []
    
    for shape in shapes:
        shape_type = shape.get("shape_type", "")
        label = shape.get("label", "unknown")
        points = np.array(shape.get("points", []), dtype=np.float32)
        
        if len(points) < 2:
            continue
        
        # If label_filter is set, skip labels not in the filter
        if label_filter is not None and label not in label_filter:
            continue
        
        # Get class index (default to foreground)
        class_idx = class_mapping.get(label, 1)
        if class_idx == 0:
            continue  # Skip background labels
        
        points_int = points.astype(np.int32)
        
        if shape_type == "polygon":
            # Standard closed polygon - fill directly
            cv2.fillPoly(mask, [points_int], class_idx)
            contours.append(points_int)
            
        elif shape_type == "linestrip":
            # Open polyline - close it by connecting last point to first,
            # then fill the enclosed area
            if len(points_int) >= 3:
                # Close the polyline to form a polygon
                closed_points = np.vstack([points_int, points_int[0:1]])
                cv2.fillPoly(mask, [closed_points], class_idx)
                contours.append(closed_points)
            else:
                # Only 2 points - draw as a thick line
                cv2.line(mask, tuple(points_int[0]), tuple(points_int[1]),
                         class_idx, thickness=3)
                contours.append(points_int)
                
        elif shape_type == "line":
            # Two-point line segment - draw as thick line
            if len(points_int) >= 2:
                cv2.line(mask, tuple(points_int[0]), tuple(points_int[1]),
                         class_idx, thickness=3)
                contours.append(points_int)
    
    return mask, contours


# Backward-compatible alias
create_mask_from_polygons = create_mask_from_shapes


def get_all_samples(
    root_path: str,
    version_pattern: Optional[str] = None
) -> List[Sample]:
    """
    Retrieve all valid image-annotation pairs from the filesystem.
    
    :param root_path: The base directory to scan.
    :param version_pattern: Optional prefix pattern (e.g. 'run_001_').
    :return: A list of consolidated Sample objects.
    """
    samples = []
    
    # Get all JSON files
    if version_pattern:
        json_pattern = os.path.join(root_path, version_pattern.replace("*", "*.json"))
    else:
        json_pattern = os.path.join(root_path, "*.json")
    
    json_files = glob.glob(json_pattern)
    
    for json_path in sorted(json_files):
        # Get corresponding image path (support both .jpg and .png)
        base_name = os.path.splitext(json_path)[0]
        image_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = base_name + ext
            if os.path.exists(candidate):
                image_path = candidate
                break
        
        if image_path is not None:
            samples.append(Sample(
                image_path=image_path,
                annotation_path=json_path
            ))
    
    return samples


def split_dataset(
    samples: List[Sample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Partition samples into training, validation, and testing sets.

    Discovers all image-annotation pairs in the root directory that match
    the specified patterns and optional version prefixes, then shuffles and
    partitions them.

    :param samples: List of all samples to split.
    :param train_ratio: Fraction of samples for training.
    :param val_ratio: Fraction of samples for validation.
    :param test_ratio: Fraction of samples for testing.
    :param seed: Random seed for reproducible shuffling.
    :return: A tuple of (train_samples, val_samples, test_samples).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Shuffle with seed
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_samples = shuffled[:train_end]
    val_samples = shuffled[train_end:val_end]
    test_samples = shuffled[val_end:]
    
    return train_samples, val_samples, test_samples


class IndustrialDataset(Dataset):
    """
    PyTorch Dataset for industrial chignon segmentation.

    Provides access to image-mask pairs with support for on-the-fly resizing,
    label filtering, and Albumentations transformations.

    :param samples: List of Sample objects to include in the dataset.
    :param target_size: Optional (H, W) tuple for image resizing.
    :param transform: Optional Albumentations transform pipeline.
    :param label_filter: Optional list of labels to include in mask generation.
    :param return_contours: Whether to return extracted contours in the sample dict.
    """
    
    def __init__(
        self,
        samples: List[Sample],
        transform: Optional[Any] = None,
        target_size: Optional[Tuple[int, int]] = (512, 512),
        return_contours: bool = True,
        label_filter: Optional[List[str]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            samples: List of Sample objects
            transform: Optional preprocessing transform
            target_size: Target size for resizing (height, width)
            return_contours: Whether to return contour information
            label_filter: Optional list of label names to include.
                If None, all labels are used.
        """
        self.samples = samples
        self.transform = transform
        self.target_size = target_size
        self.return_contours = return_contours
        self.label_filter = label_filter
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieve a single sample with processed image and mask.

        Loads the raw data from disk, generates the occupancy mask, applies
        transforms, and resizes both image and mask to the target dimensions.
        Includes raw annotation shapes for dynamic calibration.

        :param idx: Index of the sample.
        :return: Dict containing tensors and original metadata.
        """
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # Load annotation
        annotation = load_labelme_annotation(sample.annotation_path)
        
        # Create mask and get contours
        mask, contours = create_mask_from_polygons(
            annotation["shapes"],
            annotation["imageHeight"],
            annotation["imageWidth"],
            label_filter=self.label_filter
        )
        
        # Resize if needed
        if self.target_size is not None:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), 
                            interpolation=cv2.INTER_NEAREST)
            
            # Scale contours
            scale_x = self.target_size[1] / original_size[1]
            scale_y = self.target_size[0] / original_size[0]
            scaled_contours = []
            for cnt in contours:
                scaled_cnt = cnt.astype(np.float32)
                scaled_cnt[:, 0] *= scale_x
                scaled_cnt[:, 1] *= scale_y
                scaled_contours.append(scaled_cnt.astype(np.int32))
            contours = scaled_contours
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        result = {
            "image": image,
            "mask": mask,
            "image_path": sample.image_path,
            "original_size": original_size,
            "shapes": annotation.get("shapes", []),
        }
        
        if self.return_contours:
            result["contours"] = contours
        
        return result


def create_dataloaders(
    root_path: str = None,
    batch_size: int = 1,
    target_size: Tuple[int, int] = (512, 512),
    num_workers: int = 0,
    label_filter: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create training, validation, and testing DataLoaders.

    Orchestrates the full data pipeline: splitting the raw data, instantiating
    Dataset objects with appropriate augmentations, and wrapping them in
    PyTorch DataLoaders.

    :param root_path: Path to the root directory of the augmented data.
    :param batch_size: Number of samples per batch.
    :param target_size: (H, W) tuple to which all images will be resized.
    :param label_filter: Optional list of labels to isolate.
    :param num_workers: Number of subprocesses for data loading.
    :return: A tuple containing (train_loader, val_loader, test_loader).
    """
    if root_path is None:
        root_path = config.dataset.root_path
    
    if label_filter:
        print(f"Label filter active: {label_filter}")
    
    # Get all samples
    samples = get_all_samples(root_path)
    
    # Split dataset
    train_samples, val_samples, test_samples = split_dataset(
        samples,
        config.dataset.train_ratio,
        config.dataset.val_ratio,
        config.dataset.test_ratio
    )
    
    print(f"Dataset split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    # Create datasets
    train_dataset = IndustrialDataset(train_samples, target_size=target_size, label_filter=label_filter)
    val_dataset = IndustrialDataset(val_samples, target_size=target_size, label_filter=label_filter)
    test_dataset = IndustrialDataset(test_samples, target_size=target_size, label_filter=label_filter)
    
    # Custom collate function to handle contours and shapes
    def collate_fn(batch):
        """Standardizes batch output for DataLoader."""
        result = {
            "image": [],
            "mask": [],
            "image_path": [],
            "original_size": [],
            "contours": [],
            "shapes": []
        }
        
        for item in batch:
            result["image"].append(item["image"])
            result["mask"].append(item["mask"])
            result["image_path"].append(item["image_path"])
            result["original_size"].append(item["original_size"])
            result["contours"].append(item.get("contours", []))
            result["shapes"].append(item.get("shapes", []))
        
        result["image"] = np.stack(result["image"])
        result["mask"] = np.stack(result["mask"])
        
        return result
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn,
        drop_last=True  # Avoid batch_size=1 which crashes BatchNorm (e.g. DeepLabV3 ASPP)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def get_samples_by_version(
    root_path: str = None,
    version: str = "run_001"
) -> List[Sample]:
    """
    Retrieve all samples matching a specific version prefix.
    
    :param root_path: The base directory of the dataset.
    :param version: The version identification string (e.g. 'run_001').
    :return: A list of filtered Sample objects.
    """
    if root_path is None:
        root_path = config.dataset.root_path
    
    return get_all_samples(root_path, f"{version}_*")
