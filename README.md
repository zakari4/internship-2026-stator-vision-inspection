# Edge Detection Benchmark for Stator Case 

A comprehensive benchmarking framework for evaluating **30 segmentation and edge detection models** on industrial stator case images. Combines classical computer vision, deep learning architectures, YOLO instance segmentation, transformer-based detection, and state-of-the-art edge detectors — with automated data augmentation, configurable training, inference benchmarking, and rich visualization.

---

## Table of Contents

- [Project Architecture](#project-architecture)
  - [Directory Structure](#directory-structure)
  - [Package Overview](#package-overview)
  - [Data Flow](#data-flow)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Automated Setup](#automated-setup)
  - [Manual Setup](#manual-setup)
- [Dataset & Annotations](#dataset--annotations)
  - [LabelMe Format](#labelme-format)
  - [Mask Generation](#mask-generation)
  - [Label Filtering](#label-filtering)
  - [YOLO Format Conversion](#yolo-format-conversion)
- [Data Augmentation](#data-augmentation)
  - [Geometric Transforms](#geometric-transforms)
  - [Photometric Transforms](#photometric-transforms)
  - [Running Augmentation](#running-augmentation)
- [Preprocessing Pipeline](#preprocessing-pipeline)
  - [Classical Model Preprocessing](#classical-model-preprocessing)
  - [Deep Learning Preprocessing](#deep-learning-preprocessing)
- [Training](#training)
  - [Quick Start](#quick-start)
  - [Training Architecture](#training-architecture)
  - [Loss Functions](#loss-functions)
  - [Optimizers & Schedulers](#optimizers--schedulers)
  - [Hardware Monitoring](#hardware-monitoring)
  - [All Training Options](#all-training-options)
- [Benchmarking](#benchmarking)
- [Metrics & Evaluation](#metrics--evaluation)
  - [Segmentation Metrics](#segmentation-metrics)
  - [Contour Metrics](#contour-metrics)
  - [Latency & Hardware Metrics](#latency--hardware-metrics)
- [Contour Extraction & Geometry Fitting](#contour-extraction--geometry-fitting)
- [Measurement & Calibration Methodologies](#measurement--calibration-methodologies)
  - [Method A: Reference Object (ArUco)](#method-a-reference-object-aruco-marker)
  - [Method B: Camera Intrinsics](#method-b-geometric-camera-calibration)
  - [Method C: ML Depth Estimation](#method-c-ml-depth-estimation-midas)
  - [Lens Undistortion](#lens-undistortion)
- [Visualization & Reporting](#visualization--reporting)
- [Available Models](#available-models)
  - [Classical Models (7)](#classical-models-7)
  - [Deep Learning Models (23)](#deep-learning-models-23)
- [Configuration Reference](#configuration-reference)
- [Hardware Requirements](#hardware-requirements)

---

## Project Architecture

### Directory Structure

```
chignon_detection/
├── src/                          # Core Python package
│   ├── __init__.py
│   ├── config.py                 # Central configuration (dataclasses)
│   ├── data/                     # Data loading & augmentation
│   │   ├── dataset.py            #   LabelMe dataset loader, mask generation, DataLoader
│   │   ├── augmentation.py       #   Offline 72x data augmentation pipeline
│   │   └── yolo_prep.py          #   LabelMe → YOLO format conversion
│   ├── models/                   # Model definitions
│   │   ├── __init__.py           #   Model registry (all 30 models exported)
│   │   ├── classical.py          #   7 classical CV models (Otsu, Canny, etc.)
│   │   └── deep_learning.py      #   23 deep learning models (UNet → LDC)
│   ├── training/                 # Training engine
│   │   ├── trainer.py            #   ComprehensiveTrainer, loss functions, YOLOTrainer
│   │   └── training_viz.py       #   Training curves & comparison plots
│   ├── evaluation/               # Evaluation & benchmarking
│   │   ├── metrics.py            #   IoU, Dice, F1, Hausdorff, hardware profiling
│   │   ├── analyze.py            #   Statistical analysis & report generation
│   │   └── visualization.py      #   Benchmark visualizations (bar charts, heatmaps)
│   └── utils/                    # Shared utilities
│       ├── preprocessing.py      #   Image preprocessing pipeline
│       ├── contour.py            #   Contour extraction & geometry fitting
│       └── measurements.py       #   Geometric measurements & calibration
│
├── scripts/                      # Entry-point scripts
│   ├── train.py                  # Main training & benchmark runner (CLI)
│   ├── test_best_model.py        # Evaluate best model on test set
│   ├── test_measurements.py      # Test measurement methodologies (A/B/C)
│   ├── audit_pipeline.py         # Comprehensive data pipeline audit
│   ├── run_failed_models.py      # Re-run failed model training
│   └── generate_viz.py           # Generate standalone visualizations
│
├── data/                         # Raw dataset (LabelMe images + JSON)
├── weights/                      # Pretrained model weights (.pt files)
├── outputs/                      # All generated outputs
│   ├── augmented_data/           #   72x augmented images + annotations
│   ├── yolo_dataset/             #   YOLO-format dataset (images + labels + data.yaml)
│   └── results/                  #   Training logs, plots, visualizations
│       ├── training_logs/        #     Per-model training history (JSON)
│       ├── plots/                #     Loss curves, IoU charts, comparisons
│       ├── visualizations/       #     Prediction overlays, heatmaps
│       └── logs/                 #     Text logs
│
├── docs/                         # Documentation
│   └── benchmark_report.md       #   Generated benchmark report
│
├── setup.sh                      # One-command project setup
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore rules
```

### Package Overview

The project is organized as a standard Python package under `src/`:

| Package | Module | Purpose |
|---------|--------|---------|
| `src.config` | `config.py` | Central configuration with 8 dataclass configs (dataset, preprocessing, benchmark, classical models, DL models, contour, measurement, master) |
| `src.data` | `dataset.py` | `IndustrialDataset` (PyTorch Dataset), `create_dataloaders()`, LabelMe mask generation, train/val/test splitting |
| `src.data` | `augmentation.py` | Offline 72x augmentation (8 geometric × 9 photometric) with coordinate-aware transforms |
| `src.data` | `yolo_prep.py` | Converts LabelMe annotations to YOLO segmentation format |
| `src.models` | `classical.py` | 7 classical CV models with a common `BaseClassicalModel` interface |
| `src.models` | `deep_learning.py` | 23 DL models with `BaseDeepLearningModel` interface, including UNet, YOLO, RT-DETR, edge detectors |
| `src.training` | `trainer.py` | `ComprehensiveTrainer` (PyTorch training loop), `YOLOTrainer` (Ultralytics API), loss functions (BCE, Dice, Boundary, Lovász) |
| `src.training` | `training_viz.py` | Training curve generation, model comparison charts |
| `src.evaluation` | `metrics.py` | `MetricsComputer` for IoU/Dice/F1/Hausdorff, `HardwareProfiler` for GPU/CPU monitoring |
| `src.evaluation` | `analyze.py` | Statistical analysis and markdown report generation |
| `src.evaluation` | `visualization.py` | Benchmark bar charts, heatmaps, comparison tables |
| `src.utils` | `preprocessing.py` | `PreprocessingPipeline` with bilateral filtering, CLAHE, morphological cleanup |
| `src.utils` | `contour.py` | `ContourExtractor` and `GeometryFitter` for post-processing |
| `src.utils` | `measurements.py` | `MeasurementComputer`, `CalibrationManager` (ArUco PPM, camera intrinsics, ML depth estimation), lens undistortion, checkerboard calibration |

### Data Flow

```
               ┌─────────────────────────────────────────────────────────┐
               │                    SETUP PHASE                          │
               │                                                         │
    Raw Data   │   Augmentation (72x)     YOLO Format    Weight Download │
    data/*.png ──→ outputs/augmented/ ──→ outputs/yolo/   weights/*.pt  │
    data/*.json│                                                         │
               └────────────────────────────┬────────────────────────────┘
                                            │
               ┌────────────────────────────▼────────────────────────────┐
               │                   TRAINING PHASE                        │
               │                                                         │
               │  IndustrialDataset ──→ DataLoader ──→ ComprehensiveTrainer
               │       (masks)           (batches)       (train/val loop) │
               │                                              │          │
               │                              ┌───────────────┼──────┐   │
               │                              │  Loss (BCE+Dice)     │   │
               │                              │  Optimizer (AdamW)   │   │
               │                              │  Scheduler (Cosine)  │   │
               │                              │  GradScaler (AMP)    │   │
               │                              └──────────────────────┘   │
               └────────────────────────────┬────────────────────────────┘
                                            │
               ┌────────────────────────────▼────────────────────────────┐
               │                 BENCHMARK PHASE                         │
               │                                                         │
               │  BenchmarkRunner ──→ Metrics (IoU, Dice, F1, Latency)  │
               │       │          ──→ Hardware Profiling (GPU/CPU)       │
               │       │          ──→ Contour Extraction                 │
               │       ▼                                                 │
               │  Reports, Plots, Visualizations, JSON Results           │
               └─────────────────────────────────────────────────────────┘
```

**All paths are resolved dynamically** via `PROJECT_ROOT = Path(__file__).resolve().parent.parent` in `src/config.py`, so the project runs correctly from any working directory.

---

## Setup

### Prerequisites

- **Python** >= 3.9
- **CUDA-capable GPU** (recommended) — NVIDIA GPU with >= 4 GB VRAM
- **OS**: Linux (tested), macOS, Windows (WSL recommended)

### Automated Setup

A single script handles the entire environment setup:

```bash
chmod +x setup.sh
./setup.sh
```

The script performs **7 sequential steps**:

| Step | Action | Idempotent |
|------|--------|:----------:|
| 1/7 | Create Python virtual environment (`venv/`) | Yes |
| 2/7 | Upgrade pip | — |
| 3/7 | Install dependencies from `requirements.txt` + `scikit-learn` | — |
| 4/7 | Run 72x data augmentation (`data/` → `outputs/augmented_data/`) | Yes |
| 5/7 | Prepare YOLO-format dataset (`outputs/augmented_data/` → `outputs/yolo_dataset/`) | Yes |
| 6/7 | Clean up corrupt `.pt` weight files (< 5 MB) | Yes |
| 7/7 | Download YOLO-seg + RT-DETR pretrained weights (11 variants) | Yes |

Steps marked **idempotent** check for existing output before running. Re-running `setup.sh` is safe and skips completed steps.

After setup, activate the environment:

```bash
source venv/bin/activate
```

### Manual Setup

If you prefer manual installation:

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install scikit-learn

# 3. Run data augmentation
python src/data/augmentation.py --input data --output outputs/augmented_data

# 4. Prepare YOLO dataset
python src/data/yolo_prep.py --source outputs/augmented_data --output outputs/yolo_dataset

# 5. Download model weights (handled automatically on first use by Ultralytics)
```

### Dependencies

| Category | Packages |
|----------|----------|
| Core | `numpy`, `scipy` |
| Image Processing | `opencv-python`, `Pillow`, `scikit-image` |
| Deep Learning | `torch`, `torchvision` |
| YOLO/RT-DETR | `ultralytics` |
| Plotting | `matplotlib`, `seaborn` |
| Utilities | `tqdm`, `pyyaml`, `natsort`, `psutil`, `pynvml` |

---

## Dataset & Annotations

### LabelMe Format

The dataset uses **LabelMe JSON** annotations with paired files (`.png` + `.json`). Each JSON file contains shapes with:
- **Shape types**: `polygon`, `linestrip`, `line`
- **Labels**: `michanical_part`, `magnet`, `circle`

### Mask Generation

The `src/data/dataset.py` module converts annotations to binary segmentation masks:

1. **Label mapping**: All foreground labels → class index `1`; `background` → `0`
2. **Shape handling**:
   - `polygon` → Filled polygon via `cv2.fillPoly`
   - `linestrip` → Closed into polygon (last point → first), then filled
   - `line` → Drawn as 3px-wide line segment
3. **Output**: Binary `uint8` mask with values `{0, 1}`

### Label Filtering

The `--labels` flag allows training on specific object types only:

```bash
# Train on specific labels only
python scripts/train.py --mode train --labels michanical_part magnet

# Train on all labels (default)
python scripts/train.py --mode train
```

When label filtering is active, only shapes with matching labels appear in the generated masks.

### YOLO Format Conversion

`src/data/yolo_prep.py` converts LabelMe annotations to YOLO segmentation format:

- Normalized polygon coordinates `[0, 1]` for each shape
- All foreground classes mapped to class ID `0` (single-class)
- Outputs: images + `.txt` label files + `data.yaml`
- Default split: 70% train / 15% val / 15% test

```bash
python src/data/yolo_prep.py --source outputs/augmented_data --output outputs/yolo_dataset
```

---

## Data Augmentation

The project uses **offline 72x augmentation** via `src/data/augmentation.py`. Each original image generates **72 augmented versions** (8 geometric × 9 photometric).

### Geometric Transforms

All geometric transforms **update LabelMe JSON annotation coordinates** — polygons, linestrips, and lines are transformed using coordinate mapping functions.

| # | Transform | Coordinate Mapping |
|---|-----------|-------------------|
| 1 | Identity | `(x, y) → (x, y)` |
| 2 | Rotate 90° CW | `(x, y) → (h-y-1, x)` |
| 3 | Rotate 180° | `(x, y) → (w-x-1, h-y-1)` |
| 4 | Rotate 270° CW | `(x, y) → (y, w-x-1)` |
| 5 | Flip Horizontal | `(x, y) → (w-x-1, y)` |
| 6 | Flip Vertical | `(x, y) → (x, h-y-1)` |
| 7 | Flip H + Rotate 90° | Combined mapping |
| 8 | Flip V + Rotate 90° | Combined mapping |

### Photometric Transforms

Simulate real industrial imaging conditions. **Do not** affect annotation coordinates.

| # | Transform | Parameters | Purpose |
|---|-----------|------------|---------|
| 1 | Identity | — | Baseline |
| 2 | Brightness Up | `+35` | Brighter lighting |
| 3 | Brightness Down | `-35` | Dimmer lighting |
| 4 | Contrast Up | `alpha=1.4` | High-contrast scenes |
| 5 | Gaussian Noise | `sigma=15` | Sensor noise |
| 6 | Gaussian Blur | `kernel=5×5` | Defocus / vibration |
| 7 | Motion Blur | `7px horizontal` | Conveyor belt motion |
| 8 | CLAHE | `clip=2.0, tile=8×8` | Adaptive contrast |
| 9 | Gamma Dark | `gamma=0.6` | Underexposure |

### Running Augmentation

```bash
python src/data/augmentation.py --input data --output outputs/augmented_data
```

---

## Preprocessing Pipeline

### Classical Model Preprocessing

All images pass through `src/utils/preprocessing.py` before classical models:

```
Input Image
    │
    ▼
1. Resize ──────────── Target size (512×512), INTER_LINEAR
    │
    ▼
2. Grayscale ────────── Single-channel conversion
    │
    ▼
3. Noise Reduction ──── Bilateral filter (d=9, σ_color=75, σ_space=75)
    │
    ▼
4. Normalization ─────── Intensity normalization to [0, 255]
    │
    ▼
5. CLAHE ──────────────── clipLimit=2.0, tileGridSize=8×8
    │
    ▼
6. Morphological ──────── Opening + Closing (elliptical kernel 3×3)
    │
    ▼
Output (processed grayscale)
```

> **Design choice**: The bilateral filter is the default because it preserves edges while smoothing — critical for industrial inspection where edge boundaries define regions of interest.

### Deep Learning Preprocessing

For deep learning models, the data pipeline in `src/data/dataset.py` handles:

1. **Resize** to target size (default: 512×512)
2. **Image normalization**: `float32 / 255.0` in the training loop
3. **Mask**: Binary `{0, 1}` float tensor
4. **Batch collation**: Custom `collate_fn` stacks images and masks as numpy arrays, converted to tensors in the training loop

---

## Training

### Quick Start

```bash
# Activate environment
source venv/bin/activate

# Train all models (50 epochs, AdamW, cosine scheduler, early stopping)
python scripts/train.py --mode train --optimizer adamw --early-stopping 10

# Train a specific model
python scripts/train.py --mode train --models unet_lightweight --epochs 30

# Train with label filtering
python scripts/train.py --mode train --labels michanical_part magnet

# Run inference benchmark only
python scripts/train.py --mode benchmark

# Full pipeline (train + benchmark)
python scripts/train.py --mode full
```

### Training Architecture

The `ComprehensiveTrainer` class in `src/training/trainer.py` manages the complete training lifecycle:

```
ComprehensiveTrainer
├── Model Management
│   ├── Auto-detect PyTorch nn.Module vs Ultralytics model
│   ├── OrderedDict output handling (DeepLabV3)
│   └── Output shape interpolation to match mask size
│
├── Training Loop (train_epoch)
│   ├── Forward pass (with optional AMP autocast)
│   ├── Loss computation (always FP32)
│   ├── Backward pass with GradScaler
│   ├── Gradient clipping (max_norm=1.0)
│   └── Per-batch metrics collection
│
├── Validation Loop (validate_epoch)
│   ├── No-grad forward pass
│   └── Comprehensive metrics (IoU, Dice, F1, accuracy)
│
├── Training Orchestration (train)
│   ├── Epoch iteration with early stopping
│   ├── Learning rate scheduling
│   ├── Best model checkpointing
│   └── Training history JSON export
│
└── Hardware Monitoring
    ├── Background thread sampling (100ms interval)
    └── GPU memory, utilization, CPU, RAM tracking
```

**Key design decisions:**

- **FP32 loss computation**: Loss is always computed in FP32 (`outputs.float()`, `masks.float()`) even when AMP is enabled, preventing NaN from FP16 overflow
- **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(max_norm=1.0)` prevents gradient explosion
- **Kaiming initialization**: Custom models (UNetLightweight) use Kaiming normal initialization for stable training
- **YOLO models** are trained using a separate `YOLOTrainer` that wraps the Ultralytics `model.train()` API

### Loss Functions

Two loss modes are available via `--loss`:

#### `bce_dice` (default)

The standard combination for binary segmentation:

```
L = 0.5 × BCEWithLogitsLoss + 0.5 × DiceLoss
```

- **BCEWithLogitsLoss**: Pixel-wise binary cross-entropy with built-in sigmoid
- **DiceLoss**: `1 - (2 × |P ∩ G| + ε) / (|P| + |G| + ε)`, directly optimizes overlap

#### `combined`

Extended loss with boundary and region terms:

```
L = 0.3 × BCE + 0.3 × Dice + 0.2 × BoundaryLoss + 0.2 × LovaszHingeLoss
```

- **BoundaryLoss**: Uses signed distance transforms (scipy) to penalize predictions far from boundaries. Runs in FP32 to prevent overflow. *Note: significantly slower due to CPU-based distance transforms per batch.*
- **LovaszHingeLoss**: Directly optimizes the IoU metric via the Lovász extension of submodular functions.

### Optimizers & Schedulers

| Optimizer | CLI Flag | Description |
|-----------|----------|-------------|
| **AdamW** | `--optimizer adamw` | Adam with decoupled weight decay (default) |
| **Adam** | `--optimizer adam` | Standard Adam optimizer |
| **SGD** | `--optimizer sgd` | SGD with momentum (0.9) |
| **RMSprop** | `--optimizer rmsprop` | RMSprop optimizer |

| Scheduler | CLI Flag | Description |
|-----------|----------|-------------|
| **Cosine** | `--scheduler cosine` | Cosine annealing to 0 (default) |
| **Step** | `--scheduler step` | Step decay (×0.1 every 10 epochs) |
| **Plateau** | `--scheduler plateau` | Reduce on val loss plateau (factor=0.5, patience=5) |

### Hardware Monitoring

The `HardwareMonitor` runs in a background thread during training, sampling every 100ms:

| Metric | Description |
|--------|-------------|
| GPU Memory (MB) | VRAM usage per batch |
| GPU Utilization (%) | GPU compute utilization |
| CPU Usage (%) | System CPU utilization |
| RAM Usage (MB) | System memory usage |
| Forward Time (ms) | Model forward pass latency |
| Backward Time (ms) | Gradient computation latency |
| Throughput (samples/s) | Training throughput |

All metrics are logged per-batch and aggregated per-epoch in the training history JSON.

### All Training Options

```
python scripts/train.py [OPTIONS]

Mode:
  --mode {train,benchmark,full}    Pipeline mode (default: benchmark)

Model Selection:
  --models MODEL [MODEL ...]       Models to use: all, classical, deep_learning,
                                   or specific names (default: all)

Training:
  --epochs N                       Number of epochs (default: 50)
  --lr RATE                        Learning rate (default: 0.0001)
  --weight-decay WD                Weight decay (default: 1e-5)
  --early-stopping N               Early stopping patience in epochs (default: 10)
  --optimizer {adamw,adam,sgd,rmsprop,auto}
                                   Optimizer (default: adamw)
  --scheduler {cosine,step,plateau}
                                   LR scheduler (default: cosine)
  --batch-size N                   Batch size for YOLO models (default: 8)
  --loss {bce_dice,combined}       Loss function (default: bce_dice)
  --labels LABEL [LABEL ...]       Filter by label names (default: all)

Mixed Precision:
  --amp                            Enable FP16 mixed precision (disabled by default)
  --no-amp                         Explicitly disable mixed precision

Hardware:
  --device {cuda,cpu}              Compute device (default: cuda if available)

Benchmark:
  --warmup N                       Warmup iterations (default: 5)
  --runs N                         Benchmark runs per sample (default: 20)

Output:
  --output DIR                     Custom output directory
```

---

## Benchmarking

The benchmark mode evaluates all models on the test set:

```bash
# Run inference benchmark on all models
python scripts/train.py --mode benchmark

# Benchmark specific models
python scripts/train.py --mode benchmark --models unet_lightweight deeplabv3_mobilenet canny
```

The `BenchmarkRunner` class in `scripts/train.py` handles:

1. **Data Loading** — Loads test set with custom collation
2. **Model Evaluation** — Separate paths for classical, DL, and YOLO models
3. **Metrics Computation** — Segmentation, contour, latency, hardware metrics per model
4. **Report Generation** — Ranked comparison table, JSON results, visualization plots

Results are saved to `outputs/results/`.

---

## Metrics & Evaluation

All metrics are computed by `src/evaluation/metrics.py`:

### Segmentation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **IoU** | `TP / (TP + FP + FN)` | Intersection over Union |
| **Dice** | `2·TP / (2·TP + FP + FN)` | Dice coefficient / F1 |
| **Precision** | `TP / (TP + FP)` | Positive predictive value |
| **Recall** | `TP / (TP + FN)` | Sensitivity |
| **Accuracy** | `(TP + TN) / Total` | Overall pixel accuracy |
| **Boundary F1** | Boundary-aware F1 | Edge alignment score |
| **Hausdorff** | `max(h(A,B), h(B,A))` | Maximum boundary distance |

### Contour Metrics

| Metric | Description |
|--------|-------------|
| Boundary F1 | F1-score on boundary pixels within tolerance |
| Average Boundary Distance | Mean distance between predicted and GT boundaries |
| Hausdorff Distance | Worst-case boundary distance |
| Contour Completeness | Fraction of GT boundary covered by prediction |
| Smoothness Score | Contour smoothness measure |

### Latency & Hardware Metrics

| Metric | Description |
|--------|-------------|
| Preprocessing Time (ms) | Image preprocessing latency |
| Inference Time (ms) | Model forward pass time |
| Postprocessing Time (ms) | Mask extraction / NMS time |
| Total Time (ms) | End-to-end per-sample time |
| FPS | Frames per second |
| GPU Memory (MB) | Peak VRAM usage |
| GPU Utilization (%) | Average GPU compute usage |
| CPU Usage (%) | Average CPU usage |

---

## Contour Extraction & Geometry Fitting

The `src/utils/contour.py` module provides post-processing:

### ContourExtractor

Extracts contours from binary masks with configurable filtering:
- **Area filtering**: `min_area=100`, `max_area=None`
- **Length filtering**: `min_length=50`
- **Shape descriptors**: Circularity, aspect ratio thresholds
- **Approximation**: Douglas-Peucker with `epsilon_factor=0.02`

### GeometryFitter

Fits geometric primitives to extracted contours:
- **Line fitting**: RANSAC or least-squares
- **Circle fitting**: Algebraic circle fit
- **Bounding shapes**: Minimum enclosing circle, rotated rectangle

### MeasurementComputer

Computes geometric measurements from `src/utils/measurements.py`:
- Pixel-to-metric conversion (`pixel_to_mm`)
- Euclidean and model-based distance computation
- Sub-pixel refinement support

### CalibrationManager — Pixel-to-Metric Calibration

Three calibration methodologies are available in `src/utils/measurements.py`, each with different requirements and accuracy trade-offs:

#### Method A: Reference Object (Fiducial)

The most accurate approach. Uses a **detected label** of known real-world dimensions as the calibration reference — no external markers needed.

- **How it works**: The model detects labels in the image (e.g., `michanical_part`). Since you know the real-world size of that object, its contour is used to compute the Pixels Per Metric (PPM) ratio:

  ```
  PPM = object_dimension_px / object_dimension_mm
  pixel_to_mm = 1.0 / PPM
  ```

- **Configuration** (in `src/config.py` → `MeasurementConfig`):
  ```python
  reference_label_name = "michanical_part"        # Label to use as reference
  reference_known_dimension_mm = 52.0             # Its real-world size in mm
  reference_dimension_type = "diameter"            # "diameter", "width", or "height"
  ```

- **Usage — from predictions**:
  ```python
  from src.utils.measurements import CalibrationManager
  cal = CalibrationManager()

  # Automatically find the reference label in model predictions and calibrate
  cal.calibrate_from_predictions(
      prediction=predicted_contours,   # dict: {label: [contours]} or mask array
      reference_label="michanical_part",
      known_dimension_mm=52.0,
      dimension_type="diameter"
  )

  # Now convert any pixel measurement to mm
  distance_mm = cal.convert_to_mm(distance_px)
  ```

- **Usage — from a single contour**:
  ```python
  cal.calibrate_from_contour(contour, known_dimension_mm=52.0, dimension_type="diameter")
  ```

- **Alternative — ArUco markers**: If an ArUco marker is placed in the scene instead:
  ```python
  cal.calibrate_from_aruco(image, marker_real_size_mm=50.0)
  ```
  Supports `DICT_4X4_50`, `DICT_5X5_50`, `DICT_6X6_50`, and more (auto-scanned).

#### Method B: Geometric Camera Calibration

Uses camera intrinsic parameters when no reference object is available.

- **Required parameters**:
  - `sensor_width_mm`: Physical sensor width (e.g., 6.17mm for 1/2.3" sensors)
  - `focal_length_mm`: Lens focal length (from EXIF metadata)
  - `object_distance_mm`: Distance from lens to target

- **Formula**:
  ```
  pixel_size_mm = (distance × sensor_width) / (focal_length × image_width_px)
  ```

- **Usage**:
  ```python
  cal.calibrate_from_camera_intrinsics(
      sensor_width_mm=6.17, focal_length_mm=4.0,
      object_distance_mm=300.0, image=image
  )
  ```

#### Method C: ML Depth Estimation (MiDaS)

Automates distance estimation using monocular depth prediction, removing the need for manual measurement of object distance.

- **How it works**: Runs a MiDaS/DPT depth model on the image, estimates the object distance from the depth map, then applies the camera intrinsics formula.

- **Usage**:
  ```python
  cal.calibrate_from_depth_estimation(image, model_type="MiDaS_small")
  depth_map = cal.get_last_depth_map()  # Access the depth map
  ```

- **Available models**: `MiDaS_small` (81MB, fast), `DPT_Hybrid`, `DPT_Large` (more accurate)

#### Lens Undistortion

Corrects barrel/pincushion lens distortion using OpenCV:

```python
from src.utils.measurements import undistort_image, calibrate_camera_from_checkerboard

# Calibrate from checkerboard images
cam_matrix, dist_coeffs, error = calibrate_camera_from_checkerboard(
    checkerboard_images, board_size=(9, 6), square_size_mm=25.0
)

# Undistort any image
corrected = undistort_image(image, cam_matrix, dist_coeffs)
```

---

## Visualization & Reporting

### Training Visualizations (`src/training/training_viz.py`)

Generated automatically after training completes:

| Plot | Description |
|------|-------------|
| `loss_curves.png` | Training & validation loss per epoch for all models |
| `iou_curves.png` | IoU progression over epochs |
| `dice_curves.png` | Dice score progression |
| `gpu_usage.png` | GPU memory usage over epochs |
| `cpu_usage.png` | CPU utilization over epochs |
| `latency_breakdown.png` | Forward/backward pass latency breakdown |
| `throughput.png` | Training throughput (samples/s) |
| `comprehensive_comparison.png` | Multi-metric comparison across all models |
| `final_comparison_table.png` | Summary table with best metrics |

### Benchmark Visualizations (`src/evaluation/visualization.py`)

| Plot | Description |
|------|-------------|
| IoU comparison bar chart | Side-by-side IoU scores |
| Latency comparison | Inference time per model |
| Model ranking table | Ranked by IoU, with latency and hardware stats |
| Prediction overlays | Ground truth vs prediction side-by-side |

### Reports

- **Benchmark report**: `docs/benchmark_report.md` — Detailed markdown report with tables
- **JSON results**: Full metrics exported as JSON for programmatic analysis
- **Training history**: Per-model `training_history.json` in `outputs/results/training_logs/<model_name>/`

---

## Available Models

### Classical Models (7)

| Model | Class | Description |
|-------|-------|-------------|
| Otsu Thresholding | `OtsuThresholding` | Global automatic threshold |
| Adaptive Thresholding | `AdaptiveThresholding` | Local neighborhood thresholding |
| Canny Edge Detection | `CannyEdgeDetection` | Gradient-based edge detection with hysteresis |
| Morphological Gradient | `MorphologicalGradient` | Dilation − Erosion for contour detection |
| Active Contours (Snakes) | `ActiveContoursSnakes` | Energy-minimizing deformable contours |
| Chan-Vese Level Set | `ChanVeseLevelSet` | Region-based level set segmentation |
| Morphological Chan-Vese | `MorphologicalChanVese` | Morphological active contours without edges |

### Deep Learning Models (23)

#### Semantic Segmentation (4)

| Model | Class | Backbone | Params |
|-------|-------|----------|--------|
| UNet Lightweight | `UNetLightweight` | Custom (32 base filters) | ~1.9M |
| UNet ResNet18 | `UNetResNet18` | ResNet-18 (pretrained) | ~14M |
| DeepLabV3 MobileNet | `DeepLabV3MobileNet` | MobileNetV3-Large | ~11M |
| SegFormer B0 | `SegFormerB0Simple` | Custom MiT-B0 variant | ~3.7M |

#### Instance Segmentation (1)

| Model | Class | Backbone |
|-------|-------|----------|
| Mask R-CNN | `MaskRCNNModel` | ResNet-50 FPN (pretrained) |

#### YOLO Segmentation (9)

| Model | Class | Size | Weight File |
|-------|-------|------|-------------|
| YOLOv8n-seg | `YOLOv8SegNano` | Nano | `yolov8n-seg.pt` |
| YOLOv8s-seg | `YOLOv8SegSmall` | Small | `yolov8s-seg.pt` |
| YOLOv8m-seg | `YOLOv8SegMedium` | Medium | `yolov8m-seg.pt` |
| YOLOv11n-seg | `YOLOv11SegNano` | Nano | `yolo11n-seg.pt` |
| YOLOv11s-seg | `YOLOv11SegSmall` | Small | `yolo11s-seg.pt` |
| YOLOv11m-seg | `YOLOv11SegMedium` | Medium | `yolo11m-seg.pt` |
| YOLOv26n-seg | `YOLOv26SegNano` | Nano | `yolo26n-seg.pt` |
| YOLOv26s-seg | `YOLOv26SegSmall` | Small | `yolo26s-seg.pt` |
| YOLOv26m-seg | `YOLOv26SegMedium` | Medium | `yolo26m-seg.pt` |

#### Transformer-Based Detection (2)

| Model | Class | Weight File |
|-------|-------|-------------|
| RT-DETR Large | `RTDETRLarge` | `rtdetr-l.pt` |
| RT-DETR X-Large | `RTDETRXLarge` | `rtdetr-x.pt` |

#### Foundation Model (1)

| Model | Class | Description |
|-------|-------|-------------|
| SAM 2 + LoRA | `SAM2LoRAModel` | Segment Anything Model 2 with LoRA fine-tuning |

#### SOTA Edge Detection (6)

| Model | Class | Paper |
|-------|-------|-------|
| HED | `HEDModel` | Holistically-Nested Edge Detection (Xie & Tu, 2015) |
| RCF | `RCFModel` | Richer Convolutional Features (Liu et al., 2017) |
| BDCN | `BDCNModel` | Bi-Directional Cascade Network (He et al., 2019) |
| PiDiNet | `PiDiNetModel` | Pixel Difference Networks (Su et al., 2021) |
| TEED | `TEEDModel` | Tiny and Efficient Edge Detector (Soria et al., 2023) |
| LDC | `LDCModel` | Lightweight Dense CNN (Soria et al., 2022) |

---

## Configuration Reference

All configuration is centralized in `src/config.py` using Python dataclasses:

### DatasetConfig

```python
name = "Custom_Chignon_Industrial_Dataset"
root_path = "<PROJECT_ROOT>/data"
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15
num_classes = 2  # Binary: background vs foreground
```

### PreprocessingConfig

```python
target_size = (512, 512)
use_bilateral_filter = True     # Bilateral: d=9, σ_color=75, σ_space=75
use_clahe = True                # CLAHE: clip=2.0, tile=8×8
normalize_intensity = True
```

### BenchmarkConfig

```python
batch_size = 4
device = "cuda"                 # Auto-detected
warmup_runs = 3
benchmark_runs = 10
```

### DeepLearningModelConfig

```python
optimizer = "adamw"
scheduler = "cosine"
learning_rate = 1e-4
epochs = 50
weight_decay = 1e-4
early_stopping_patience = 10
```

### ContourExtractionConfig

```python
min_area = 100
min_length = 50
epsilon_factor = 0.02           # Douglas-Peucker approximation
```

### MeasurementConfig

```python
pixel_to_mm = 1.0               # Set calibration for real-world units
line_fitting_method = "ransac"
ransac_threshold = 1.0

# Method A: ArUco Reference Object
aruco_marker_size_mm = 50.0     # Default marker physical size
aruco_dictionary = "DICT_4X4_50"

# Method B: Camera Intrinsics
sensor_width_mm = 6.17          # 1/2.3" sensor
focal_length_mm = 4.0
object_distance_mm = 300.0

# Dynamic calibration
reference_label_name = "michanical_part"
reference_known_dimension_mm = 52.0
reference_dimension_type = "diameter"
```

---

## Hardware Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| CPU | 4 cores |
| RAM | 8 GB |
| GPU | NVIDIA GPU, 4 GB VRAM |
| Storage | 10 GB free |
| Python | >= 3.9 |

### Recommended

| Component | Recommendation |
|-----------|----------------|
| CPU | 8+ cores |
| RAM | 16 GB |
| GPU | NVIDIA GPU, ≥ 8 GB VRAM |
| Storage | 20 GB free (augmented data + weights + outputs) |
| CUDA | >= 11.7 |

### Notes

- **AMP (FP16)** is disabled by default. Custom models (UNet/edge detectors) can produce NaN under FP16 due to skip connection overflow. Enable with `--amp` only for pretrained models.
- **Batch size** is limited by GPU VRAM. The default `batch_size=4` works on 4 GB GPUs. Increase for larger GPUs.
- Training all 23 deep learning models with 50 epochs takes approximately **2–4 hours** on a modern GPU.
- Classical models run on CPU and require no GPU.

---

## License

This project is for research and educational purposes.
