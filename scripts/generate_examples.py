#!/usr/bin/env python3
"""
Generate detection example images for all 4 trained models.

Produces:
  - Segmentation-only overlay
  - Segmentation with measurements (1px = 1mm)
  - Saves to docs/ for embedding in reports
"""

import os
import sys
import json
import base64

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "server"))

from inference import ModelManager, compute_single_contour_measurements, draw_measurements_on_image

DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "images", "examples")
os.makedirs(DOCS_DIR, exist_ok=True)

# Use two different test images for variety
TEST_JSONS = [
    os.path.join(PROJECT_ROOT, "data", "run_001_00003.json"),
    os.path.join(PROJECT_ROOT, "data", "run_002_00005.json"),
]


def load_image_from_json(json_path):
    """Load image from LabelMe JSON (base64 encoded)."""
    with open(json_path) as f:
        data = json.load(f)
    img_data = base64.b64decode(data["imageData"])
    buf = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img, data


def generate_for_model(manager, model_name, frame, out_prefix):
    """Generate segmentation-only and measurement images for one model."""
    ok = manager.load_model(model_name)
    if not ok:
        print(f"  SKIP: Could not load {model_name}")
        return None

    # 1. Segmentation only (no measurements)
    manager.camera_settings.enabled = False
    overlay_seg, dets_seg = manager.predict(frame.copy())
    seg_path = os.path.join(DOCS_DIR, f"{out_prefix}_segmentation.png")
    cv2.imwrite(seg_path, overlay_seg)
    print(f"    Saved: {seg_path}")

    # 2. With measurements (1px = 1mm manual calibration)
    manager.camera_settings.enabled = True
    manager.camera_settings.method = "manual"
    manager.camera_settings.manual_px_to_mm = 1.0
    overlay_meas, dets_meas = manager.predict(frame.copy())
    meas_path = os.path.join(DOCS_DIR, f"{out_prefix}_measurements.png")
    cv2.imwrite(meas_path, overlay_meas)
    print(f"    Saved: {meas_path}")

    # Reset
    manager.camera_settings.enabled = False

    return {
        "model_name": model_name,
        "seg_path": seg_path,
        "meas_path": meas_path,
        "detections_seg": dets_seg,
        "detections_meas": dets_meas,
    }


def main():
    print("Initializing ModelManager...")
    manager = ModelManager()
    print(f"Available models: {list(manager.available_models.keys())}")

    # Models we want to showcase
    target_models = ["yolov8m_seg", "yolov11m_seg", "unet_resnet18", "segformer_b0"]

    # Load the first test image (run_001_00003 — good variety of objects)
    json_path = TEST_JSONS[0]
    img, data = load_image_from_json(json_path)
    img_name = os.path.splitext(os.path.basename(json_path))[0]
    print(f"\nTest image: {img_name} ({img.shape[1]}x{img.shape[0]})")
    print(f"  Labels: {[s['label'] for s in data['shapes']]}")

    all_results = {}
    for model_name in target_models:
        print(f"\n  Processing: {model_name}")
        prefix = f"example_{model_name}"
        result = generate_for_model(manager, model_name, img, prefix)
        if result:
            all_results[model_name] = result

    # Also generate a combined comparison image
    print("\nGenerating combined comparison grid...")
    imgs = []
    labels = []
    for model_name in target_models:
        seg_path = os.path.join(DOCS_DIR, f"example_{model_name}_segmentation.png")
        if os.path.exists(seg_path):
            im = cv2.imread(seg_path)
            # Add label at top
            h, w = im.shape[:2]
            labeled = np.zeros((h + 40, w, 3), dtype=np.uint8)
            labeled[40:] = im
            display_name = {
                "yolov8m_seg": "YOLOv8m-seg",
                "yolov11m_seg": "YOLOv11m-seg",
                "unet_resnet18": "UNet ResNet18",
                "segformer_b0": "SegFormer B0",
            }.get(model_name, model_name)
            cv2.putText(labeled, display_name, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            imgs.append(labeled)

    if len(imgs) >= 2:
        # Make a 2x2 grid
        h, w = imgs[0].shape[:2]
        # Resize all to same size
        imgs_resized = [cv2.resize(im, (w, h)) for im in imgs]
        while len(imgs_resized) < 4:
            imgs_resized.append(np.zeros((h, w, 3), dtype=np.uint8))
        top = np.hstack(imgs_resized[:2])
        bottom = np.hstack(imgs_resized[2:4])
        grid = np.vstack([top, bottom])
        grid_path = os.path.join(DOCS_DIR, "example_all_models_comparison.png")
        cv2.imwrite(grid_path, grid)
        print(f"  Saved: {grid_path}")

    # Also generate measurement comparison grid
    meas_imgs = []
    for model_name in target_models:
        meas_path = os.path.join(DOCS_DIR, f"example_{model_name}_measurements.png")
        if os.path.exists(meas_path):
            im = cv2.imread(meas_path)
            h, w = im.shape[:2]
            labeled = np.zeros((h + 40, w, 3), dtype=np.uint8)
            labeled[40:] = im
            display_name = {
                "yolov8m_seg": "YOLOv8m-seg",
                "yolov11m_seg": "YOLOv11m-seg",
                "unet_resnet18": "UNet ResNet18",
                "segformer_b0": "SegFormer B0",
            }.get(model_name, model_name)
            cv2.putText(labeled, display_name + " + Measurements", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            meas_imgs.append(labeled)

    if len(meas_imgs) >= 2:
        h, w = meas_imgs[0].shape[:2]
        meas_resized = [cv2.resize(im, (w, h)) for im in meas_imgs]
        while len(meas_resized) < 4:
            meas_resized.append(np.zeros((h, w, 3), dtype=np.uint8))
        top = np.hstack(meas_resized[:2])
        bottom = np.hstack(meas_resized[2:4])
        grid = np.vstack([top, bottom])
        grid_path = os.path.join(DOCS_DIR, "example_all_models_measurements.png")
        cv2.imwrite(grid_path, grid)
        print(f"  Saved: {grid_path}")

    # Save detection details JSON
    det_summary = {}
    for model_name, r in all_results.items():
        det_summary[model_name] = {
            "num_detections": len(r["detections_seg"]),
            "detections": r["detections_seg"],
        }
    with open(os.path.join(DOCS_DIR, "example_detections.json"), "w") as f:
        json.dump(det_summary, f, indent=2)
    print(f"\n  Saved: docs/example_detections.json")

    print("\nDone! All example images saved to docs/")


if __name__ == "__main__":
    main()
