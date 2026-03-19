#!/usr/bin/env python3
"""
Offline data augmentation for industrial chignon detection.

Generates 35x augmented dataset from original images + LabelMe JSON annotations.
Geometric transforms (flips + 90°/180°/270° rotations) properly update annotation
coordinates.  Photometric transforms (brightness, luminosity, contrast, blur,
CLAHE, gamma) are industrial-appropriate.

Strategy: 5 geometric variants × 9 photometric variants = 45 versions per image

Usage:
    python benchmark/augment_data.py --input data --output augmented_data
"""

import os
import sys
import json
import copy
import glob
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Callable

import cv2
import numpy as np

# ============================================================================
# Seed for reproducibility
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ============================================================================
# Coordinate Transformation Functions
# ============================================================================

def transform_points(
    points: List[List[float]],
    transform_fn: Callable,
    img_width: int,
    img_height: int
) -> List[List[float]]:
    """Apply a coordinate transform function to all annotation points."""
    return [transform_fn(p, img_width, img_height) for p in points]


def identity_point(p, w, h):
    return p


def rotate90_point(p, w, h):
    """Rotate 90° clockwise: (x, y) -> (h - y - 1, x)"""
    return [h - p[1] - 1, p[0]]


def rotate180_point(p, w, h):
    """Rotate 180°: (x, y) -> (w - x - 1, h - y - 1)"""
    return [w - p[0] - 1, h - p[1] - 1]


def rotate270_point(p, w, h):
    """Rotate 270° clockwise: (x, y) -> (y, w - x - 1)"""
    return [p[1], w - p[0] - 1]


def flip_horizontal_point(p, w, h):
    """Horizontal flip: (x, y) -> (w - x - 1, y)"""
    return [w - p[0] - 1, p[1]]


def flip_vertical_point(p, w, h):
    """Vertical flip: (x, y) -> (x, h - y - 1)"""
    return [p[0], h - p[1] - 1]


def flip_h_rotate90_point(p, w, h):
    """Flip horizontal then rotate 90° CW."""
    p2 = flip_horizontal_point(p, w, h)
    return rotate90_point(p2, w, h)


def flip_v_rotate90_point(p, w, h):
    """Flip vertical then rotate 90° CW."""
    p2 = flip_vertical_point(p, w, h)
    return rotate90_point(p2, w, h)


def rotate_arbitrary_point(p, w, h, angle_deg, M):
    """Apply an arbitrary affine rotation matrix to a point."""
    pt = np.array([p[0], p[1], 1.0])
    result = M @ pt
    return [float(result[0]), float(result[1])]


# ============================================================================
# Image Transformation Functions
# ============================================================================

def identity_image(img):
    return img.copy()


def rotate90_image(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def rotate180_image(img):
    return cv2.rotate(img, cv2.ROTATE_180)


def rotate270_image(img):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def flip_horizontal_image(img):
    return cv2.flip(img, 1)


def flip_vertical_image(img):
    return cv2.flip(img, 0)


def flip_h_rotate90_image(img):
    return rotate90_image(flip_horizontal_image(img))


def flip_v_rotate90_image(img):
    return rotate90_image(flip_vertical_image(img))


# ============================================================================
# Photometric Augmentation Functions (industrial-appropriate)
# ============================================================================

def photo_identity(img):
    """No photometric change."""
    return img.copy()


def photo_brightness_up(img):
    """Simulate brighter industrial lighting (+30)."""
    return cv2.convertScaleAbs(img, alpha=1.0, beta=35)


def photo_brightness_down(img):
    """Simulate dimmer industrial lighting (-30)."""
    return cv2.convertScaleAbs(img, alpha=1.0, beta=-35)


def photo_contrast_up(img):
    """Higher contrast (common in well-lit industrial settings)."""
    return cv2.convertScaleAbs(img, alpha=1.4, beta=-40)


def photo_gaussian_noise(img):
    """Simulate sensor noise from industrial cameras."""
    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def photo_gaussian_blur(img):
    """Simulate slight defocus or vibration blur."""
    return cv2.GaussianBlur(img, (5, 5), 1.2)


def photo_motion_blur(img):
    """Simulate horizontal motion blur from conveyor belt movement."""
    kernel_size = 7
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(img, -1, kernel)


def photo_clahe(img):
    """CLAHE - very common preprocessing for industrial vision."""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(img)


def photo_gamma_dark(img):
    """Gamma correction (darken) - simulates underexposure."""
    gamma = 0.6
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def photo_gamma_bright(img):
    """Gamma correction (brighten) - simulates overexposure / strong backlight."""
    gamma = 1.8
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def photo_luminosity_shift(img):
    """HSV value-channel shift — uniform luminosity boost without hue change."""
    if len(img.shape) == 2:
        return cv2.convertScaleAbs(img, alpha=1.0, beta=20)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + 20, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ============================================================================
# Augmentation Pipeline Definitions
# ============================================================================

# Geometric transforms: flips + 90 / 180 / 270° rotations → 5x geometric multiplier
GEOMETRIC_TRANSFORMS = [
    ("orig",    identity_image,        identity_point),
    ("flipH",   flip_horizontal_image, flip_horizontal_point),
    ("rot90",   rotate90_image,        rotate90_point),
    ("rot180",  rotate180_image,       rotate180_point),
    ("rot270",  rotate270_image,       rotate270_point),
]

# Photometric transforms: 7 variants covering luminosity, contrast, blur, gamma
# Chosen to simulate real-world industrial lighting conditions
PHOTOMETRIC_TRANSFORMS = [
    ("clean",       photo_identity),
    ("bright_up",   photo_brightness_up),
    ("bright_down", photo_brightness_down),
    ("contrast_up", photo_contrast_up),
    ("gauss_blur",  photo_gaussian_blur),
    ("clahe",       photo_clahe),
    ("gamma_dark",  photo_gamma_dark),
    ("gamma_bright",photo_gamma_bright),
    ("lum_shift",   photo_luminosity_shift),
]

# Total: 5 × 9 = 45 versions per image (45x augmentation)


# ============================================================================
# Core Augmentation Logic
# ============================================================================

def transform_annotation(
    annotation: Dict,
    point_fn: Callable,
    new_width: int,
    new_height: int,
    original_width: int,
    original_height: int
) -> Dict:
    """
    Apply a coordinate transform to all shapes in a LabelMe annotation.
    Returns a new annotation dict with updated coordinates and dimensions.
    """
    new_ann = copy.deepcopy(annotation)
    new_ann["imageWidth"] = new_width
    new_ann["imageHeight"] = new_height

    for shape in new_ann.get("shapes", []):
        shape["points"] = transform_points(
            shape["points"], point_fn, original_width, original_height
        )

    return new_ann


def augment_single_image(
    image_path: str,
    json_path: str,
    output_dir: str,
    base_name: str
) -> int:
    """
    Generate all augmented versions of a single image + annotation pair.

    Returns:
        Number of augmented samples generated.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"  WARNING: Could not read image {image_path}, skipping")
        return 0

    # Load annotation
    with open(json_path, 'r') as f:
        annotation = json.load(f)

    orig_w = annotation.get("imageWidth", img.shape[1])
    orig_h = annotation.get("imageHeight", img.shape[0])

    count = 0

    for geo_name, geo_img_fn, geo_pt_fn in GEOMETRIC_TRANSFORMS:
        # Apply geometric transform to image
        geo_img = geo_img_fn(img)
        new_h, new_w = geo_img.shape[:2]

        # Apply geometric transform to annotation coordinates
        geo_ann = transform_annotation(
            annotation, geo_pt_fn, new_w, new_h, orig_w, orig_h
        )

        for photo_name, photo_fn in PHOTOMETRIC_TRANSFORMS:
            # Apply photometric transform to image (no coordinate change)
            aug_img = photo_fn(geo_img)

            # Generate output filename
            aug_name = f"{base_name}_{geo_name}_{photo_name}"
            img_out = os.path.join(output_dir, f"{aug_name}.png")
            json_out = os.path.join(output_dir, f"{aug_name}.json")

            # Save augmented image
            cv2.imwrite(img_out, aug_img)

            # Save updated annotation (remove imageData to save space)
            out_ann = copy.deepcopy(geo_ann)
            out_ann["imagePath"] = f"{aug_name}.png"
            out_ann["imageData"] = None  # Don't embed base64 image
            out_ann["imageWidth"] = new_w
            out_ann["imageHeight"] = new_h

            with open(json_out, 'w') as f:
                json.dump(out_ann, f, indent=2)

            count += 1

    return count


def run_augmentation(
    input_dir: str,
    output_dir: str,
    enable_copy_paste: bool = False,
    allowed_splits: List[str] | None = None,
    sync_yolo_dataset: bool = True,
):
    """
    Run the full augmentation pipeline on all images in input_dir.
    Saves augmented images + updated JSON annotations to output_dir.

    If ``input_dir`` contains split sub-directories (``train/``, ``val/``,
    ``test/``), only the splits listed in *allowed_splits* are augmented
    (default: ``["train", "val"]``).  A flat directory (no split folders)
    is augmented entirely.

    Args:
        input_dir: Input directory with images and LabelMe JSON annotations
        output_dir: Output directory for augmented dataset
        enable_copy_paste: Whether to apply Copy-Paste augmentation after standard pipeline
        allowed_splits: Which split folders to augment (ignored for flat dirs).
                        Defaults to ``["train", "val"]``.
        sync_yolo_dataset: If True, regenerate ``outputs/yolo_dataset`` from
                           the augmented output automatically.
    """
    if allowed_splits is None:
        allowed_splits = ["train", "val"]

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Detect split sub-directories
    split_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name in ("train", "val", "test")]

    if split_dirs:
        # ------ YOLO / split-based layout ------
        skipped = sorted(d.name for d in split_dirs if d.name not in allowed_splits)
        active  = sorted(d.name for d in split_dirs if d.name in allowed_splits)
        if skipped:
            print(f"Skipping splits: {', '.join(skipped)}")
        if not active:
            print("ERROR: No allowed splits found to augment.")
            return

        for split_name in active:
            split_path = input_path / split_name
            # Look for images sub-folder (YOLO layout) or flat JSON files
            images_dir = split_path / "images" if (split_path / "images").is_dir() else split_path
            json_files = sorted(images_dir.glob("*.json"))
            # Also try the split root for LabelMe-style flat layout
            if not json_files:
                json_files = sorted(split_path.glob("*.json"))

            pairs = _find_pairs(json_files)
            if not pairs:
                print(f"[{split_name}] No image-annotation pairs found — skipping.")
                continue

            split_output = output_path / split_name
            split_output.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"Augmenting split: {split_name}  ({len(pairs)} pairs)")
            print(f"{'='*60}")
            _augment_pairs(pairs, str(split_output), enable_copy_paste)
    else:
        # ------ Flat LabelMe layout (no split folders) ------
        json_files = sorted(input_path.glob("*.json"))
        pairs = _find_pairs(json_files)
        if not pairs:
            print(f"ERROR: No image-annotation pairs found in {input_dir}")
            return
        _augment_pairs(pairs, str(output_path), enable_copy_paste)

    if sync_yolo_dataset:
        _sync_augmented_to_yolo_dataset(output_path)


def _sync_augmented_to_yolo_dataset(augmented_output_path: Path) -> None:
    """Regenerate outputs/yolo_dataset from the augmented output folder."""
    project_root = Path(__file__).resolve().parent.parent.parent
    yolo_output = project_root / "outputs" / "yolo_dataset"

    # prepare_yolo_dataset expects a flat folder with image/json pairs at root.
    # If not found, keep augmentation result and skip sync with a warning.
    has_root_json = any(augmented_output_path.glob("*.json"))
    if not has_root_json:
        print(
            "\n[WARN] Augmentation output has no root-level JSON files; "
            "skipping automatic YOLO dataset sync."
        )
        return

    try:
        from src.data.yolo_prep import prepare_yolo_dataset

        print("\n" + "=" * 60)
        print("Syncing augmented data to YOLO dataset")
        print("=" * 60)
        prepare_yolo_dataset(
            source_dir=str(augmented_output_path),
            output_dir=str(yolo_output),
        )
        print(f"YOLO dataset updated at: {yolo_output}")
    except Exception as exc:
        print(f"[WARN] Failed to sync YOLO dataset automatically: {exc}")


def _find_pairs(json_files: list) -> List[Tuple[str, str]]:
    """Return (image_path, json_path) pairs from a list of JSON files."""
    pairs = []
    for json_file in json_files:
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = json_file.with_suffix(ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is not None:
            pairs.append((str(img_path), str(json_file)))
    return pairs


def _augment_pairs(
    pairs: List[Tuple[str, str]],
    output_dir: str,
    enable_copy_paste: bool = False,
):
    """Core loop: augment a list of image-annotation pairs into *output_dir*."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pairs)} image-annotation pairs")
    print(f"Augmentation: {len(GEOMETRIC_TRANSFORMS)} geometric × "
          f"{len(PHOTOMETRIC_TRANSFORMS)} photometric = "
          f"{len(GEOMETRIC_TRANSFORMS) * len(PHOTOMETRIC_TRANSFORMS)} versions per image")
    print(f"Expected total: {len(pairs) * len(GEOMETRIC_TRANSFORMS) * len(PHOTOMETRIC_TRANSFORMS)} images")
    if enable_copy_paste:
        print(f"Copy-Paste augmentation: ENABLED (will add ~{len(pairs)} extra samples)")
    print(f"Output directory: {output_path}")
    print()

    total_generated = 0

    for i, (img_path, json_path) in enumerate(pairs):
        base_name = Path(img_path).stem
        short_name = base_name[:30] + "..." if len(base_name) > 30 else base_name
        print(f"[{i+1}/{len(pairs)}] Processing: {short_name}")

        n = augment_single_image(img_path, json_path, str(output_path), base_name)
        total_generated += n
        print(f"         Generated {n} augmented samples")

    # Copy-Paste augmentation: paste annotated objects from one image onto another
    if enable_copy_paste and len(pairs) >= 2:
        print(f"\n{'='*60}")
        print("Running Copy-Paste Augmentation")
        print(f"{'='*60}")
        cp_count = run_copy_paste(pairs, str(output_path))
        total_generated += cp_count
        print(f"Copy-Paste generated {cp_count} additional samples")

    print(f"\nAugmentation complete!")
    print(f"  Original images: {len(pairs)}")
    print(f"  Augmented images: {total_generated}")
    print(f"  Multiplier: {total_generated / max(len(pairs), 1):.0f}x")
    print(f"  Output: {output_path}")


# ============================================================================
# Copy-Paste Augmentation (SOTA for instance segmentation)
# Reference: Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation
#            Method for Instance Segmentation", CVPR 2021.
# ============================================================================

def extract_object_mask(shape: Dict, img_height: int, img_width: int) -> np.ndarray:
    """Extract a binary mask for a single annotation shape."""
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    points = np.array(shape["points"], dtype=np.int32)
    
    if shape.get("shape_type") == "polygon" and len(points) >= 3:
        cv2.fillPoly(mask, [points], 255)
    elif shape.get("shape_type") in ("linestrip", "line") and len(points) >= 2:
        # Convert line to thin polygon
        cv2.polylines(mask, [points], isClosed=False, color=255, thickness=4)
    
    return mask


def copy_paste_single(
    target_img: np.ndarray,
    target_ann: Dict,
    source_img: np.ndarray,
    source_ann: Dict,
    max_objects: int = 3,
    scale_range: Tuple = (0.5, 1.5)
) -> Tuple[np.ndarray, Dict]:
    """
    Paste annotated objects from source image onto target image.
    
    Args:
        target_img: Target image to paste objects onto
        target_ann: Target annotation dict
        source_img: Source image to extract objects from
        source_ann: Source annotation dict
        max_objects: Maximum number of objects to paste
        scale_range: Scale range for pasted objects
    
    Returns:
        Tuple of (augmented_image, updated_annotation)
    """
    result_img = target_img.copy()
    result_ann = copy.deepcopy(target_ann)
    
    source_shapes = source_ann.get("shapes", [])
    if not source_shapes:
        return result_img, result_ann
    
    src_h, src_w = source_img.shape[:2]
    tgt_h, tgt_w = target_img.shape[:2]
    
    # Randomly choose how many objects to paste
    n_objects = min(random.randint(1, max_objects), len(source_shapes))
    chosen_shapes = random.sample(source_shapes, n_objects)
    
    for shape in chosen_shapes:
        # Extract object mask from source
        obj_mask = extract_object_mask(shape, src_h, src_w)
        
        # Find bounding box of the object
        coords = np.where(obj_mask > 0)
        if len(coords[0]) == 0:
            continue
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        obj_h = y_max - y_min + 1
        obj_w = x_max - x_min + 1
        
        if obj_h < 5 or obj_w < 5:
            continue
        
        # Crop object region
        obj_crop = source_img[y_min:y_max+1, x_min:x_max+1].copy()
        mask_crop = obj_mask[y_min:y_max+1, x_min:x_max+1].copy()
        
        # Random scale
        scale = random.uniform(*scale_range)
        new_h = max(5, int(obj_h * scale))
        new_w = max(5, int(obj_w * scale))
        
        # Ensure it fits in target
        new_h = min(new_h, tgt_h - 10)
        new_w = min(new_w, tgt_w - 10)
        
        if new_h < 5 or new_w < 5:
            continue
        
        obj_crop = cv2.resize(obj_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_crop = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Random position in target
        paste_y = random.randint(0, tgt_h - new_h)
        paste_x = random.randint(0, tgt_w - new_w)
        
        # Paste using mask (alpha blending)
        mask_3ch = np.stack([mask_crop > 0] * 3, axis=-1).astype(np.float32)
        roi = result_img[paste_y:paste_y+new_h, paste_x:paste_x+new_w]
        result_img[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = (
            mask_3ch * obj_crop + (1 - mask_3ch) * roi
        ).astype(np.uint8)
        
        # Update annotation: shift and scale source points to paste position
        new_shape = copy.deepcopy(shape)
        new_points = []
        for pt in shape["points"]:
            # Normalize to [0,1] within bounding box, then to paste position
            nx = (pt[0] - x_min) / obj_w * new_w + paste_x
            ny = (pt[1] - y_min) / obj_h * new_h + paste_y
            new_points.append([float(nx), float(ny)])
        new_shape["points"] = new_points
        result_ann["shapes"].append(new_shape)
    
    return result_img, result_ann


def run_copy_paste(
    pairs: List[Tuple[str, str]],
    output_dir: str,
    num_augments_per_image: int = 1
) -> int:
    """
    Run Copy-Paste augmentation across all image pairs.
    For each image, paste objects from a randomly chosen other image.
    
    Args:
        pairs: List of (image_path, json_path) tuples
        output_dir: Output directory
        num_augments_per_image: Number of copy-paste variants per image
    
    Returns:
        Number of augmented samples generated
    """
    count = 0
    
    for i, (tgt_img_path, tgt_json_path) in enumerate(pairs):
        tgt_img = cv2.imread(tgt_img_path)
        if tgt_img is None:
            continue
        
        with open(tgt_json_path, 'r') as f:
            tgt_ann = json.load(f)
        
        for aug_idx in range(num_augments_per_image):
            # Pick a random source (different from target)
            src_idx = random.choice([j for j in range(len(pairs)) if j != i])
            src_img_path, src_json_path = pairs[src_idx]
            
            src_img = cv2.imread(src_img_path)
            if src_img is None:
                continue
            
            with open(src_json_path, 'r') as f:
                src_ann = json.load(f)
            
            # Apply copy-paste
            result_img, result_ann = copy_paste_single(
                tgt_img, tgt_ann, src_img, src_ann,
                max_objects=3, scale_range=(0.5, 1.5)
            )
            
            # Save
            base_name = Path(tgt_img_path).stem
            aug_name = f"{base_name}_copypaste_{aug_idx}"
            img_out = os.path.join(output_dir, f"{aug_name}.png")
            json_out = os.path.join(output_dir, f"{aug_name}.json")
            
            cv2.imwrite(img_out, result_img)
            
            result_ann["imagePath"] = f"{aug_name}.png"
            result_ann["imageData"] = None
            with open(json_out, 'w') as f:
                json.dump(result_ann, f, indent=2)
            
            count += 1
        
        if (i + 1) % 5 == 0:
            print(f"  Copy-Paste: [{i+1}/{len(pairs)}] processed")
    
    return count


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="10x industrial data augmentation with coordinate-aware transforms"
    )
    parser.add_argument(
        "--input", type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "data"),
        help="Input directory with images and LabelMe JSON annotations"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "outputs" / "augmented_data"),
        help="Output directory for augmented dataset"
    )
    parser.add_argument(
        "--copy-paste", action="store_true", default=False,
        help="Enable Copy-Paste augmentation (pastes objects between images)"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Which split folders to augment when input contains train/val/test dirs (default: train val)"
    )
    parser.add_argument(
        "--no-sync-yolo", action="store_true", default=False,
        help="Disable automatic regeneration of outputs/yolo_dataset after augmentation"
    )

    args = parser.parse_args()
    run_augmentation(
        args.input,
        args.output,
        enable_copy_paste=args.copy_paste,
        allowed_splits=args.splits,
        sync_yolo_dataset=not args.no_sync_yolo,
    )

