# Project State Report — Stator Vision Inspection

**Date**: 2026-03-12  
**Branch**: `dev`  
**Repository**: `github.com:zakari4/internship-2026-stator-vision-inspection.git`

---

## 1. Project Overview

This project is an **industrial stator case vision inspection system** that uses segmentation and object detection models to identify and measure components on stator assemblies. It combines:

- **Multi-class segmentation** (PyTorch: UNet ResNet18, SegFormer B0) — 4 classes: `background`, `michanical_part`, `magnet`, `circle`
- **Instance segmentation** (Ultralytics YOLO: YOLOv8m-seg, YOLOv11m-seg) — 3 object classes + bounding boxes
- **Measurement pipeline** — Converts pixel-based detections to real-world mm measurements via camera calibration
- **Web application** — Flask + WebRTC server with real-time detection, model switching, and measurement overlay

### Hardware

- NVIDIA GTX 1650 (4 GB VRAM)
- Linux (Python 3.12.9)

---

## 2. Project Structure

```
chignon_detection/
├── src/                          # Core Python package (10,294 lines)
│   ├── config.py                 # Central configuration (4 classes, 512×512)
│   ├── data/
│   │   ├── dataset.py            # LabelMe dataset loader, multi-class mask generation
│   │   ├── augmentation.py       # 10× data augmentation pipeline
│   │   └── yolo_prep.py          # LabelMe → YOLO format conversion
│   ├── models/
│   │   └── deep_learning.py      # UNet, SegFormer, DeepLab, edge detectors (23 models)
│   ├── training/
│   │   ├── trainer.py            # ComprehensiveTrainer (multi-class CE+Dice loss)
│   │   └── training_viz.py       # Training comparison plots
│   └── evaluation/               # Metrics (IoU, Dice, F1, Hausdorff), analysis, visualisation
│
├── server/                       # Flask + WebRTC server (1,668 lines)
│   ├── server.py                 # API endpoints, WebRTC signaling
│   └── inference.py              # Model loading, inference, measurement post-processing
│
├── client/                       # Web UI (1,799 lines)
│   ├── index.html / app.js / style.css
│   └── nginx.conf                # Production deployment config
│
├── scripts/                      # CLI tools
│   ├── train.py                  # Main training & benchmark runner (1,368 lines)
│   ├── regenerate_plots.py       # Cross-model comparison plot generator
│   └── generate_examples.py      # Detection example image generator
│
├── data/                         # 346 original LabelMe JSON annotations
├── outputs/
│   ├── augmented_data/           # 3,460 augmented images (10× pipeline)
│   ├── yolo_dataset/             # YOLO-format dataset (images + labels)
│   └── results/                  # Training outputs
│       ├── checkpoints/          # PyTorch model weights (.pth)
│       ├── yolo_training/        # YOLO model weights + metrics
│       ├── training_logs/        # Per-epoch JSON histories
│       └── plots/                # Comparison visualizations
│
├── weights/                      # Pretrained YOLO base weights
├── docker-compose.yml            # Docker deployment
└── docs/                         # Reports, example images
```

---

## 3. Dataset

| Property | Value |
|----------|-------|
| **Annotation format** | LabelMe JSON (base64-encoded images + polygon shapes) |
| **Original images** | 346 annotated images (640×480) |
| **Augmented images** | 3,460 (10× augmentation: 2 geometric × 5 photometric) |
| **Classes** | `michanical_part` (1), `magnet` (2), `circle` (3), + `background` (0) |
| **Typical image** | 7 objects: 1 circle, 2 magnets, 4 mechanical parts |
| **Input resolution** | 512×512 (resized during training) |

---

## 4. Trained Models — Summary

All models are trained for **20 epochs** on **512×512** images.

| Model | Framework | Type | Dataset | Best Val IoU | Train Time | Peak GPU | Checkpoint |
|-------|-----------|------|---------|-------------|------------|----------|------------|
| **UNet ResNet18** | PyTorch | Multi-class seg. | Original (346) | **0.8763** | 12.4 min | 3,585 MB | `outputs/results/checkpoints/unet_resnet18/best_model.pth` |
| **YOLOv8m-seg** | Ultralytics | Instance seg. | Augmented (3,460) | 0.8569 | 1h 52min | 4,086 MB | `outputs/results/yolo_training/yolov8m_seg/weights/best.pt` |
| **YOLOv11m-seg** | Ultralytics | Instance seg. | Augmented (3,460) | 0.8476 | 2h 32min | 3,975 MB | `outputs/results/yolo_training/yolov11m_seg/weights/best.pt` |
| **SegFormer B0** | PyTorch | Multi-class seg. | Original (346) | 0.5954 | 3.0 min | 1,962 MB | `outputs/results/checkpoints/segformer_b0/best_model.pth` |

> **Note**: YOLO models were trained on the augmented dataset (3,460 images) while PyTorch models used the original dataset (346 images). This is due to **hardware constraints** — the GTX 1650 (4 GB VRAM) makes training PyTorch models on the full augmented set prohibitively slow, with each epoch taking significantly longer and limited batch sizes.

### 4.1 UNet ResNet18 — Best Overall (IoU: 0.8763)

| Metric | Value |
|--------|-------|
| Best Val IoU | 0.8763 (epoch 19) |
| Val Dice | 0.9324 |
| Val Precision | 0.9545 |
| Val Recall | 0.9835 |
| Val Accuracy | 98.64% |
| Batch Size | 4 |
| Optimizer | AdamW + Cosine LR |
| Loss | Multi-class (CrossEntropy + per-class Dice) |

### 4.2 SegFormer B0 — Fastest Training (3 min)

| Metric | Value |
|--------|-------|
| Best Val IoU | 0.5954 (epoch 20, still improving) |
| Val Dice | 0.7169 |
| Val Precision | 0.8562 |
| Val Recall | 0.9051 |
| Val Accuracy | 95.57% |
| Throughput | ~27 samples/sec (4× faster than UNet) |

### 4.3 YOLOv8m-seg — Best Instance Detection

| Metric | Value |
|--------|-------|
| mAP50-95 (Mask) | 0.8569 |
| mAP50 (Mask) | 0.9916 |
| Precision (Mask) | 0.9922 |
| Recall (Mask) | 0.9956 |
| Batch Size | 8 |

### 4.4 YOLOv11m-seg — Highest Recall

| Metric | Value |
|--------|-------|
| mAP50-95 (Mask) | 0.8476 |
| mAP50 (Mask) | 0.9932 |
| Precision (Mask) | 0.9902 |
| Recall (Mask) | 0.9970 |

---

## 5. Detection Examples

All examples use test image `run_001_00003` (640×480) containing **7 objects**: 1 circle, 2 magnets, 4 mechanical parts.

### 5.1 All Models — Segmentation Comparison

![All Models Comparison](../images/examples/example_all_models_comparison.png)

*Side-by-side comparison of all 4 models. YOLOv8m and YOLOv11m produce instance masks with bounding boxes; UNet ResNet18 produces pixel-level multi-class masks; SegFormer B0 produces multi-class masks with lower resolution.*

### 5.2 All Models — With Measurements (1 px = 1 mm)

![All Models Measurements](../images/examples/example_all_models_measurements.png)

*Same detections with measurement overlay enabled (manual calibration: 1 px = 1 mm). Width, height, diameter, area, and perimeter are computed for each detected object.*

---

### 5.3 Per-Model Detections

#### YOLOv8m-seg (7 detections)

| # | Class | Confidence |
|---|-------|-----------|
| 1 | circle | 96.85% |
| 2 | magnet | 93.05% |
| 3 | magnet | 92.54% |
| 4 | michanical_part | 89.02% |
| 5 | michanical_part | 88.19% |
| 6 | michanical_part | 87.67% |
| 7 | michanical_part | 85.63% |

| Segmentation | With Measurements |
|---|---|
| ![YOLOv8m seg](../images/examples/example_yolov8m_seg_segmentation.png) | ![YOLOv8m meas](../images/examples/example_yolov8m_seg_measurements.png) |

#### YOLOv11m-seg (7 detections)

| # | Class | Confidence |
|---|-------|-----------|
| 1 | circle | 95.87% |
| 2 | magnet | 91.37% |
| 3 | magnet | 90.39% |
| 4 | michanical_part | 87.02% |
| 5 | michanical_part | 85.30% |
| 6 | michanical_part | 82.33% |
| 7 | michanical_part | 82.05% |

| Segmentation | With Measurements |
|---|---|
| ![YOLOv11m seg](../images/examples/example_yolov11m_seg_segmentation.png) | ![YOLOv11m meas](../images/examples/example_yolov11m_seg_measurements.png) |

#### UNet ResNet18 (7 detections)

| # | Class | Confidence |
|---|-------|-----------|
| 1 | michanical_part | 93.39% |
| 2 | michanical_part | 89.88% |
| 3 | michanical_part | 93.33% |
| 4 | michanical_part | 91.41% |
| 5 | magnet | 92.29% |
| 6 | magnet | 93.33% |
| 7 | circle | 86.02% |

| Segmentation | With Measurements |
|---|---|
| ![UNet seg](../images/examples/example_unet_resnet18_segmentation.png) | ![UNet meas](../images/examples/example_unet_resnet18_measurements.png) |

#### SegFormer B0 (8 detections)

| # | Class | Confidence |
|---|-------|-----------|
| 1 | michanical_part | 79.44% |
| 2 | michanical_part | 77.15% |
| 3 | michanical_part | 87.51% |
| 4 | michanical_part | 78.17% |
| 5 | magnet | 44.55% |
| 6 | magnet | 76.33% |
| 7 | magnet | 84.96% |
| 8 | circle | 93.61% |

| Segmentation | With Measurements |
|---|---|
| ![SegFormer seg](../images/examples/example_segformer_b0_segmentation.png) | ![SegFormer meas](../images/examples/example_segformer_b0_measurements.png) |

> SegFormer detects 8 objects (1 extra spurious magnet region), with generally lower confidence, consistent with its lower IoU.

---

## 6. Measurement Pipeline

The system supports 3 calibration methods for converting pixel measurements to real-world dimensions:

| Method | How It Works | Use Case |
|--------|-------------|----------|
| **Camera Intrinsics** | Uses sensor width, focal length, and object distance | Fixed camera setups |
| **Reference Label** | Uses a detected object with known real-world size | When a reference object is visible |
| **Manual** | User provides a fixed px→mm factor | Quick testing, calibration |

For the examples above, **manual calibration** is used with **1 px = 1 mm**.

**Measurements per detection**:
- **Width** / **Height** (bounding rectangle)
- **Diameter** (minimum enclosing circle)
- **Area** (contour area)
- **Perimeter** (contour arc length)
- **Inter-detection distances** (minimum contour-to-contour distance between pairs)

---

## 7. Training & Comparison Plots

All plots integrate all 4 models and are stored in `outputs/results/plots/`.

### 7.1 Best Validation IoU

![Best IoU](../../outputs/results/plots/best_iou_comparison.png)

### 7.2 IoU Curves (20 epochs)

![IoU Curves](../../outputs/results/plots/iou_curves.png)

### 7.3 Loss Curves

![Loss Curves](../../outputs/results/plots/loss_curves.png)

### 7.4 Training Time

![Training Time](../../outputs/results/plots/training_time.png)

### 7.5 Accuracy vs Speed

![Accuracy vs Speed](../../outputs/results/plots/accuracy_vs_speed.png)

### 7.6 Comprehensive Dashboard

![Comprehensive](../../outputs/results/plots/comprehensive_comparison.png)

### 7.7 Final Comparison Table

![Final Table](../../outputs/results/plots/final_comparison_table.png)

---

## 8. Web Application

The project includes a full-stack web application for real-time inspection:

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Server** | Flask + aiortc | REST API + WebRTC peer connections |
| **Client** | Vanilla JS + HTML/CSS | Camera feed, model selector, measurement controls |
| **Deployment** | Docker + Nginx | `docker-compose.yml` for containerised deployment |

**Features**:
- Real-time WebRTC video processing (camera or file upload)
- Model switching at runtime (all 4 trained models + pretrained weights)
- 3 calibration methods for measurement overlay
- Per-detection measurement data (JSON over DataChannel)
- Detection results panel with class, confidence, and measurements

**Endpoints**:
- `GET /api/models` — List available models
- `POST /api/select-model` — Switch active model
- `POST /offer` — WebRTC SDP signaling
- `POST /api/camera-settings` — Configure measurement calibration

---

## 9. Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Multi-class segmentation** (4 classes) | Enables per-class detection labels (vs old binary foreground/background) |
| **CrossEntropy + Dice loss** | CE handles class imbalance; Dice optimises IoU directly |
| **No ImageNet normalisation** | Training data differs significantly from ImageNet distribution |
| **BGR→RGB conversion** | Matches training preprocessing; corrects colour channel mismatch |
| **Original data for PyTorch** | Hardware constraint: GTX 1650 cannot train augmented (3,460) efficiently |
| **Augmented data for YOLO** | YOLO's efficient pipeline handles 10× data within 2h |
| **1 px = 1 mm demo calibration** | Demonstrates measurement pipeline; real deployments use camera intrinsics |

---

## 10. Current State & Next Steps

### Completed
- [x] Multi-class segmentation pipeline (4 classes)
- [x] Data augmentation (10× pipeline: 346 → 3,460 images)
- [x] 4 models trained: UNet ResNet18, SegFormer B0, YOLOv8m-seg, YOLOv11m-seg
- [x] Measurement pipeline (3 calibration methods)
- [x] Web application (Flask + WebRTC + client UI)
- [x] Docker deployment configuration
- [x] Benchmark reports and comparison plots
- [x] Detection example images for all models

### Potential Improvements
- [ ] Train PyTorch models on augmented data (needs better GPU or cloud compute)
- [ ] Add more training epochs for SegFormer B0 (still improving at epoch 20)
- [ ] Experiment with larger SegFormer variants (B2, B4)
- [ ] Add confidence thresholding controls in the web UI
- [ ] Implement ArUco marker-based automatic calibration
- [ ] Add production logging and model performance monitoring
