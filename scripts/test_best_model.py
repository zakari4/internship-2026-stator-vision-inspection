#!/usr/bin/env python3
"""
Test the best YOLO model (YOLOv26n-seg) on a test image.
Uses the trained weights from the benchmark results.
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Paths — dynamic, relative to project root
WEIGHTS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "results", "yolo_training")
IMAGES_DIR = os.path.join(_PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "results", "test_predictions")

# Best YOLO models by benchmark IoU
BEST_MODELS = {
    "yolov26n_seg": {"weights": f"{WEIGHTS_DIR}/yolov26n_seg/weights/best.pt", "iou": 0.9486},
    "yolov8n_seg":  {"weights": f"{WEIGHTS_DIR}/yolov8n_seg/weights/best.pt",  "iou": 0.9480},
}

def run_inference(model_name, weights_path, image_path, output_dir):
    """Run segmentation inference and save results."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Weights: {weights_path}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    # Load model
    model = YOLO(weights_path)

    # Run inference
    results = model.predict(
        source=image_path,
        imgsz=512,
        conf=0.25,
        save=False,
        verbose=True
    )

    # Load original image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Create output visualization
    overlay = img.copy()
    mask_combined = np.zeros((h, w), dtype=np.uint8)

    result = results[0]
    if result.masks is not None:
        print(f"Detected {len(result.masks)} object(s)")
        for i, mask in enumerate(result.masks.data):
            # Resize mask to original image size
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (mask_resized > 0.5).astype(np.uint8)
            mask_combined = np.maximum(mask_combined, binary_mask)

            # Get confidence and class
            conf = result.boxes.conf[i].item() if result.boxes.conf is not None else 0
            cls = int(result.boxes.cls[i].item()) if result.boxes.cls is not None else 0
            print(f"  Object {i+1}: class={cls}, confidence={conf:.4f}")

        # Create colored overlay (green mask)
        colored_mask = np.zeros_like(img)
        colored_mask[mask_combined == 1] = [0, 255, 0]
        overlay = cv2.addWeighted(img, 0.6, colored_mask, 0.4, 0)

        # Draw contours in red
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        # Add text
        cv2.putText(overlay, f"{model_name} | Conf: {conf:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        print("No objects detected!")
        cv2.putText(overlay, f"{model_name} | No detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Save results
    basename = os.path.splitext(os.path.basename(image_path))[0]
    overlay_path = os.path.join(output_dir, f"{basename}_{model_name}_overlay.jpg")
    mask_path = os.path.join(output_dir, f"{basename}_{model_name}_mask.png")

    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(mask_path, mask_combined * 255)
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved mask: {mask_path}")

    # Also create a side-by-side comparison
    comparison = np.hstack([
        cv2.resize(img, (400, 400)),
        cv2.resize(overlay, (400, 400)),
        cv2.resize(cv2.cvtColor(mask_combined * 255, cv2.COLOR_GRAY2BGR), (400, 400))
    ])
    cv2.putText(comparison, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, "Prediction", (410, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, "Mask", (810, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    comparison_path = os.path.join(output_dir, f"{basename}_{model_name}_comparison.jpg")
    cv2.imwrite(comparison_path, comparison)
    print(f"Saved comparison: {comparison_path}")

    return overlay_path, mask_path, comparison_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pick a test image (use one from the dataset)
    images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if not images:
        print("No images found!")
        sys.exit(1)

    # Pick a representative test image (from the middle of the dataset)
    test_image_name = images[len(images) // 2]
    test_image_path = os.path.join(IMAGES_DIR, test_image_name)
    print(f"Test image: {test_image_path}")

    # Run inference with both best models
    for model_name, info in BEST_MODELS.items():
        if os.path.exists(info["weights"]):
            run_inference(model_name, info["weights"], test_image_path, OUTPUT_DIR)
        else:
            print(f"Weights not found for {model_name}: {info['weights']}")

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
