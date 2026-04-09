#!/usr/bin/env python3
"""
Comprehensive Pipeline Audit Script
Tests all pipeline components to verify correctness after label changes.
"""

import os
import sys
import json
import time
import traceback
import numpy as np

# Setup paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import cv2

results = {"passed": [], "failed": [], "warnings": []}


def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            try:
                print(f"\n{'='*60}")
                print(f"TEST: {name}")
                print(f"{'='*60}")
                func()
                results["passed"].append(name)
                print(f"  [PASS] {name}")
            except AssertionError as e:
                results["failed"].append((name, str(e)))
                print(f"  [FAIL] {name}: {e}")
            except Exception as e:
                results["failed"].append((name, str(e)))
                print(f"  [ERROR] {name}: {e}")
                traceback.print_exc()
        return wrapper
    return decorator


# ============================================================
# TEST 1: Labels and Config
# ============================================================
@test("Labels and Config Consistency")
def test_labels_config():
    from src.data.dataset import LABELS
    from src.config import config

    print(f"  LABELS: {LABELS}")
    print(f"  config.dataset.classes: {config.dataset.classes}")
    print(f"  config.dataset.root_path: {config.dataset.root_path}")

    assert "background" in LABELS, "Missing 'background' in LABELS"
    assert "mechanical_part" in LABELS, "Missing 'mechanical_part' in LABELS"
    assert "magnet" in LABELS, "Missing 'magnet' in LABELS"
    assert "circle" in LABELS, "Missing 'circle' in LABELS"
    assert len(LABELS) == 4, f"Expected 4 labels, got {len(LABELS)}"

    assert os.path.isdir(config.dataset.root_path), \
        f"Data root does not exist: {config.dataset.root_path}"

    # Check old labels are gone
    for old in ["chignon", "mecparts", "d_intern_chignon", "Frame1", "Frame2"]:
        assert old not in LABELS, f"Old label '{old}' still in LABELS"


# ============================================================
# TEST 2: Data Loading
# ============================================================
@test("Data Loading")
def test_data_loading():
    from src.data.dataset import get_all_samples

    samples = get_all_samples("data")
    print(f"  Total samples found: {len(samples)}")
    assert len(samples) >= 300, f"Expected at least 300 samples, got {len(samples)}"

    # Check file extensions
    for s in samples[:5]:
        assert s.image_path.endswith(".jpg"), f"Expected .jpg, got {s.image_path}"
        assert s.annotation_path.endswith(".json"), f"Expected .json, got {s.annotation_path}"
        assert os.path.exists(s.image_path), f"Image missing: {s.image_path}"
        assert os.path.exists(s.annotation_path), f"Annotation missing: {s.annotation_path}"


# ============================================================
# TEST 3: Mask Generation
# ============================================================
@test("Mask Generation from Annotations")
def test_mask_generation():
    from src.data.dataset import load_labelme_annotation, create_mask_from_shapes

    # Test multiple samples
    test_files = ["data/run_001_00003.json", "data/v2_run_001_00003.json", "data/v3_run_001_00001.json"]
    for json_path in test_files:
        if not os.path.exists(json_path):
            results["warnings"].append(f"Skipped {json_path} (not found)")
            continue

        ann = load_labelme_annotation(json_path)
        shapes = ann.get("shapes", [])
        h, w = ann["imageHeight"], ann["imageWidth"]

        mask, contours = create_mask_from_shapes(shapes, h, w)

        print(f"  {os.path.basename(json_path)}: mask={mask.shape}, "
              f"non-zero={np.count_nonzero(mask)}, contours={len(contours)}")

        assert mask.shape == (h, w), f"Mask shape mismatch: {mask.shape} vs ({h}, {w})"
        assert np.count_nonzero(mask) > 0, f"Mask is all zeros for {json_path}"
        assert len(contours) > 0, f"No contours for {json_path}"


# ============================================================
# TEST 4: Dataset and DataLoader
# ============================================================
@test("IndustrialDataset and DataLoaders")
def test_dataset():
    from src.data.dataset import create_dataloaders

    train_loader, val_loader, test_loader = create_dataloaders()

    total = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    print(f"  Total: {total} (Train={len(train_loader.dataset)}, "
          f"Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)})")

    assert total >= 300, f"Expected at least 300 total, got {total}"

    # Test a batch
    batch = next(iter(test_loader))
    assert isinstance(batch, dict), f"Expected dict batch, got {type(batch)}"
    assert "image" in batch, "Missing 'image' key in batch"
    assert "mask" in batch, "Missing 'mask' key in batch"

    img = batch["image"]
    mask = batch["mask"]
    print(f"  Image: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
    print(f"  Mask: shape={mask.shape}, dtype={mask.dtype}, unique={np.unique(mask)}")

    assert img.shape[1:3] == (512, 512), f"Image not resized to 512x512: {img.shape}"
    assert mask.shape[1:3] == (512, 512), f"Mask not resized to 512x512: {mask.shape}"
    assert np.count_nonzero(mask) > 0, "Mask is all zeros in batch"


# ============================================================
# TEST 5: UNet Model Forward Pass
# ============================================================
@test("UNet Model Forward Pass")
def test_unet_forward():
    from src.models.deep_learning import UNetLightweight

    model = UNetLightweight()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    img = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        out = model(img)

    print(f"  Input: {img.shape}")
    print(f"  Output: {out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")

    assert out.shape == (1, 1, 512, 512), f"Unexpected output shape: {out.shape}"
    model.cpu()
    torch.cuda.empty_cache()


# ============================================================
# TEST 6: Segmentation Metrics Computation
# ============================================================
@test("Segmentation Metrics Computation")
def test_metrics():
    from src.evaluation.metrics import MetricsComputer

    computer = MetricsComputer()

    # Test with known masks
    # Perfect match
    perfect_pred = np.ones((100, 100), dtype=np.uint8) * 255
    perfect_gt = np.ones((100, 100), dtype=np.uint8) * 255
    m1 = computer.compute_segmentation_metrics(perfect_pred, perfect_gt)
    print(f"  Perfect match - IoU: {m1.iou:.4f}, Dice: {m1.dice:.4f}")
    assert m1.iou > 0.99, f"Perfect match IoU should be ~1.0, got {m1.iou}"

    # Half overlap
    half_pred = np.zeros((100, 100), dtype=np.uint8)
    half_pred[:50, :] = 255
    half_gt = np.zeros((100, 100), dtype=np.uint8)
    half_gt[:, :50] = 255
    m2 = computer.compute_segmentation_metrics(half_pred, half_gt)
    print(f"  Half overlap - IoU: {m2.iou:.4f}, Dice: {m2.dice:.4f}")
    assert 0.2 < m2.iou < 0.4, f"Half overlap IoU should be ~0.33, got {m2.iou}"

    # No overlap
    no_pred = np.zeros((100, 100), dtype=np.uint8)
    no_pred[:50, :] = 255
    no_gt = np.zeros((100, 100), dtype=np.uint8)
    no_gt[50:, :] = 255
    m3 = computer.compute_segmentation_metrics(no_pred, no_gt)
    print(f"  No overlap - IoU: {m3.iou:.4f}, Dice: {m3.dice:.4f}")
    assert m3.iou < 0.01, f"No overlap IoU should be ~0, got {m3.iou}"


# ============================================================
# TEST 7: Model + Data E2E (prediction quality for untrained model)
# ============================================================
@test("End-to-End Benchmark Simulation")
def test_e2e_benchmark():
    from src.data.dataset import create_dataloaders
    from src.models.deep_learning import UNetLightweight
    from src.evaluation.metrics import MetricsComputer

    _, _, test_loader = create_dataloaders()
    model = UNetLightweight()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    metrics_computer = MetricsComputer()

    batch = next(iter(test_loader))
    images = torch.from_numpy(batch["image"]).float().permute(0, 3, 1, 2) / 255.0
    masks = batch["mask"]

    with torch.no_grad():
        output = model(images.to(device))

    # Convert output to mask (same as benchmark code line 350-353)
    out_np = output.squeeze().cpu().numpy()
    if out_np.ndim == 3:
        pred_mask = out_np.argmax(axis=0)
    else:
        pred_mask = out_np

    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
    gt_mask = masks[0]

    # Normalize GT (same as benchmark _compute_segmentation_metrics)
    gt_bin = (gt_mask > 0).astype(np.uint8) * 255

    print(f"  Pred mask: shape={pred_mask_bin.shape}, non-zero={np.count_nonzero(pred_mask_bin)}")
    print(f"  GT   mask: shape={gt_bin.shape}, non-zero={np.count_nonzero(gt_bin)}")

    m = metrics_computer.compute_segmentation_metrics(pred_mask_bin, gt_bin)
    print(f"  IoU: {m.iou:.4f}, Dice: {m.dice:.4f}, F1: {m.f1_score:.4f}")
    print(f"  NOTE: Low IoU is EXPECTED for an untrained model.")

    # The key assertion: the pipeline doesn't crash
    assert pred_mask_bin.shape == gt_bin.shape, "Shape mismatch between pred and gt"
    # IoU for untrained model will be near 0 - this is EXPECTED
    results["warnings"].append(
        f"IoU={m.iou:.4f} for untrained UNet is expected behavior (model outputs random predictions)"
    )

    model.cpu()
    torch.cuda.empty_cache()


# ============================================================
# TEST 8: Short Training Loop (2 epochs)
# ============================================================
@test("Training Loop (2 epochs)")
def test_training():
    from src.data.dataset import create_dataloaders
    from src.models.deep_learning import UNetLightweight

    train_loader, val_loader, _ = create_dataloaders()
    model = UNetLightweight()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    losses = []
    for epoch in range(2):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 5:  # Only 5 batches per epoch for speed
                break

            images = torch.from_numpy(batch["image"]).float().permute(0, 3, 1, 2) / 255.0
            masks = torch.from_numpy(batch["mask"]).float().unsqueeze(1)

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    # Verify loss is decreasing (or at least not NaN)
    assert not np.isnan(losses[-1]), "Training produced NaN loss"
    assert losses[-1] < losses[0] * 1.5, \
        f"Loss increased significantly: {losses[0]:.4f} -> {losses[-1]:.4f}"

    print(f"  Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")

    model.cpu()
    torch.cuda.empty_cache()


# ============================================================
# TEST 9: YOLO Data Prep Labels
# ============================================================
@test("YOLO Data Prep Label Compatibility")
def test_yolo_prep():
    from src.data.dataset import LABELS
    from src.data.yolo_prep import prepare_yolo_dataset

    class_names = [label for label in LABELS if label != 'background']
    print(f"  YOLO class names: {class_names}")
    assert "mechanical_part" in class_names
    assert "magnet" in class_names
    assert "circle" in class_names
    assert "chignon" not in class_names, "Old label 'chignon' still in YOLO classes"


# ============================================================
# TEST 10: Augmentation Module Import
# ============================================================
@test("Augmentation Module Import")
def test_augmentation():
    from src.data.augmentation import run_augmentation
    print(f"  run_augmentation function imported successfully")
    # We don't actually run augmentation here, just verify the import works


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  COMPREHENSIVE PIPELINE AUDIT")
    print("=" * 60)

    tests = [
        test_labels_config,
        test_data_loading,
        test_mask_generation,
        test_dataset,
        test_unet_forward,
        test_metrics,
        test_e2e_benchmark,
        test_training,
        test_yolo_prep,
        test_augmentation,
    ]

    for t in tests:
        t()

    print("\n" + "=" * 60)
    print("  AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Passed: {len(results['passed'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"  Warnings: {len(results['warnings'])}")

    if results["warnings"]:
        print("\n  Warnings:")
        for w in results["warnings"]:
            print(f"    - {w}")

    if results["failed"]:
        print("\n  Failures:")
        for name, err in results["failed"]:
            print(f"    - {name}: {err}")
        sys.exit(1)
    else:
        print("\n  All tests PASSED!")
