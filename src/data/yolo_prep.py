#!/usr/bin/env python3
"""
Prepare YOLO format dataset from LabelMe JSON annotations.
Converts LabelMe polygon annotations to YOLO segmentation format
and splits data into train/val/test sets.
"""

import json
import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Any
import numpy as np

# Import global project configuration
from src.config import config
from src.data.dataset import LABELS

# Configuration
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def get_image_dimensions(json_path: str) -> Tuple[int, int]:
    """Get image dimensions from LabelMe JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('imageWidth', 640), data.get('imageHeight', 480)


def labelme_to_yolo_seg(json_path: str, class_names: List[str] = None) -> List[str]:
    """
    Convert LabelMe JSON to YOLO segmentation format.
    
    YOLO segmentation format: class_id x1 y1 x2 y2 ... xn yn
    where coordinates are normalized [0, 1].
    
    Supports polygon, linestrip, and line shape types.
    """
    if class_names is None:
        # Use global LABELS registry (excluding background)
        class_names = [label for label in LABELS if label != 'background']
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img_width = data.get('imageWidth', 640)
    img_height = data.get('imageHeight', 480)
    
    yolo_lines = []
    
    for shape in data.get('shapes', []):
        shape_type = shape.get('shape_type', '')
        label = shape.get('label', '')
        
        # Skip unknown labels
        if label not in class_names:
            continue
        
        points = shape.get('points', [])
        
        if shape_type == 'polygon':
            if len(points) < 3:
                continue
        elif shape_type == 'linestrip':
            if len(points) < 3:
                continue
            # Close the linestrip to form a polygon
            points = points + [points[0]]
        elif shape_type == 'line':
            if len(points) < 2:
                continue
            # Convert 2-point line to thin rectangle for YOLO compatibility
            p1, p2 = np.array(points[0]), np.array(points[1])
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            perp = np.array([-direction[1], direction[0]]) / length * 2  # 2px width
            points = [
                (p1 + perp).tolist(),
                (p2 + perp).tolist(),
                (p2 - perp).tolist(),
                (p1 - perp).tolist(),
            ]
        else:
            continue
        
        # Class ID mapping
        try:
            class_id = class_names.index(label)
        except ValueError:
            continue
        
        # Normalize polygon points
        normalized_points = []
        for x, y in points:
            nx = x / img_width
            ny = y / img_height
            # Clamp to [0, 1]
            nx = max(0, min(1, nx))
            ny = max(0, min(1, ny))
            normalized_points.extend([nx, ny])
        
        # Format: class_id x1 y1 x2 y2 ... xn yn
        line = f"{class_id} " + " ".join(f"{p:.6f}" for p in normalized_points)
        yolo_lines.append(line)
    
    return yolo_lines


def prepare_yolo_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    label_filter: list = None
):
    """
    Prepare YOLO format dataset from LabelMe annotations.
    
    Args:
        source_dir: Directory containing .jpg and .json pairs
        output_dir: Output directory for YOLO dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Find all image-json pairs
    json_files = sorted(source_path.glob('*.json'))
    pairs = []
    
    for json_file in json_files:
        # Find corresponding image
        img_path = json_file.with_suffix('.jpg')
        if not img_path.exists():
            img_path = json_file.with_suffix('.png')
        if not img_path.exists():
            img_path = json_file.with_suffix('.jpeg')
        
        if img_path.exists():
            pairs.append((img_path, json_file))
    
    print(f"Found {len(pairs)} image-annotation pairs")
    
    if len(pairs) == 0:
        print("ERROR: No valid image-annotation pairs found!")
        return
    
    # Shuffle and split
    random.shuffle(pairs)
    
    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    print(f"Split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    
    # Create output directories
    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    
    for split_name, split_pairs in splits.items():
        images_dir = output_path / split_name / 'images'
        labels_dir = output_path / split_name / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, json_path in split_pairs:
            # Copy image
            dst_img = images_dir / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Convert and save label
            yolo_lines = labelme_to_yolo_seg(str(json_path), class_names=label_filter)
            
            label_name = img_path.stem + '.txt'
            dst_label = labels_dir / label_name
            
            with open(dst_label, 'w') as f:
                f.write('\n'.join(yolo_lines))
        
        print(f"  {split_name}: {len(split_pairs)} samples")
    
    # Create data.yaml
    if label_filter is None:
        # Use global LABELS registry (excluding background)
        label_filter = [label for label in LABELS if label != 'background']
        
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(label_filter),
        'names': label_filter
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        for key, value in data_yaml.items():
            if isinstance(value, list):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"\nDataset prepared at: {output_path}")
    print(f"Data YAML: {yaml_path}")
    
    return str(yaml_path)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare YOLO dataset from LabelMe annotations')
    parser.add_argument('--source', type=str, 
                        default=str(_PROJECT_ROOT / 'data'),
                        help='Source directory with LabelMe annotations')
    parser.add_argument('--output', type=str,
                        default=str(_PROJECT_ROOT / 'outputs' / 'yolo_dataset'),
                        help='Output directory for YOLO dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Label names to include (e.g. --labels mecparts chignon)')
    
    args = parser.parse_args()
    
    yaml_path = prepare_yolo_dataset(
        args.source,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        label_filter=args.labels
    )
    
    print(f"\nTo use with YOLO training, set data path to:\n  {yaml_path}")
