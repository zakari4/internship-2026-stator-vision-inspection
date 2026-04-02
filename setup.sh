#!/bin/bash
# =============================================================================
# Stator Case - Chignon Detection Benchmark - Project Setup
# =============================================================================
# Initializes the project: creates venv, installs dependencies, downloads
# all model weights into weights/.
#
# Usage:
#   ./setup.sh              # Full setup (training + server + all weights) ~5.7GB
#   ./setup.sh --train      # Training-only (lean, no server deps)        ~3.5GB
#   ./setup.sh --server     # Server-only  (no training weights)          ~5.7GB
# =============================================================================

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
WEIGHTS_DIR="$PROJECT_DIR/weights"

# ── Parse arguments ──────────────────────────────────────────────
MODE="full"
for arg in "$@"; do
    case "$arg" in
        --train)   MODE="train"  ;;
        --server)  MODE="server" ;;
        --help|-h)
            echo "Usage: ./setup.sh [--train | --server]"
            echo ""
            echo "  (no flag)   Full setup: all deps + augmentation + YOLO weights + MiDaS"
            echo "  --train     Training-only: lean deps (~3.5GB), augmentation, YOLO dataset"
            echo "  --server    Server-only: full deps, skip augmentation & training data prep"
            exit 0
            ;;
    esac
done

echo "============================================================"
echo "  Stator Case - Chignon Detection - Setup  [MODE: $MODE]"
echo "============================================================"
echo ""
echo "Project directory: $PROJECT_DIR"
echo ""

# ------------------------------------------------------------------
# 1. Create virtual environment
# ------------------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
    echo "[1/8] Virtual environment already exists at: $VENV_DIR"
else
    echo "[1/8] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "      Created: $VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "      Activated venv: $(which python)"

# ------------------------------------------------------------------
# 2. Upgrade pip
# ------------------------------------------------------------------
echo ""
echo "[2/8] Upgrading pip..."
pip install --upgrade pip --quiet

# ------------------------------------------------------------------
# 3. Install requirements (mode-dependent)
# ------------------------------------------------------------------
echo ""
if [ "$MODE" = "train" ]; then
    echo "[3/8] Installing TRAINING dependencies (lean)..."
    pip install -r "$PROJECT_DIR/requirements-train.txt" --quiet
    # Explicitly skip heavy unused packages
    echo "      Skipping: scipy, scikit-learn, scikit-image, seaborn, pandas,"
    echo "                polars, flask, aiortc, av, sympy, cryptography, triton"
    # Uninstall triton if it got pulled in by torch (saves ~640MB)
    pip uninstall -y triton 2>/dev/null || true
else
    echo "[3/8] Installing FULL dependencies..."
    pip install -r "$PROJECT_DIR/requirements.txt" --quiet
    pip install scikit-learn --quiet
    pip install opencv-contrib-python --quiet
    pip install timm --quiet
    if [ "$MODE" = "server" ] || [ "$MODE" = "full" ]; then
        echo "      Installing server dependencies..."
        pip install flask flask-cors aiortc av --quiet
    fi
fi
echo "      Dependencies installed."

# ------------------------------------------------------------------
# 4. Augment data (skip for --server)
# ------------------------------------------------------------------
echo ""
AUG_DIR="$PROJECT_DIR/outputs/augmented_data"
if [ "$MODE" = "server" ]; then
    echo "[4/8] Skipping data augmentation (server mode)"
else
    if [ -d "$AUG_DIR" ] && [ "$(ls -A $AUG_DIR/*.png 2>/dev/null | head -1)" ]; then
        AUG_COUNT=$(ls -1 "$AUG_DIR"/*.png 2>/dev/null | wc -l)
        echo "[4/8] Augmented data already exists ($AUG_COUNT images in $AUG_DIR)"
    else
        echo "[4/8] Augmenting data (10x, no rotation)..."
        python -m src.data.augmentation \
            --input "$PROJECT_DIR/data" \
            --output "$AUG_DIR"
        echo "      Augmentation complete."
    fi
fi

# ------------------------------------------------------------------
# 5. Prepare YOLO format dataset (skip for --server)
# ------------------------------------------------------------------
echo ""
YOLO_DIR="$PROJECT_DIR/outputs/yolo_dataset"
if [ "$MODE" = "server" ]; then
    echo "[5/8] Skipping YOLO dataset preparation (server mode)"
else
    if [ -d "$YOLO_DIR" ] && [ -f "$YOLO_DIR/data.yaml" ]; then
        echo "[5/8] YOLO dataset already exists at: $YOLO_DIR"
    else
        echo "[5/8] Preparing YOLO dataset from augmented data..."
        python "$PROJECT_DIR/src/data/yolo_prep.py" \
            --source "$AUG_DIR" \
            --output "$YOLO_DIR" \
            --labels michanical_part magnet circle
        echo "      YOLO dataset ready (Labels: michanical_part, magnet, circle)."
    fi
fi

# ------------------------------------------------------------------
# 6. Clean up corrupt model files (from interrupted downloads)
# ------------------------------------------------------------------
echo ""
echo "[6/8] Cleaning up corrupt model files..."
mkdir -p "$WEIGHTS_DIR"
MIN_SIZE=5000000  # 5MB minimum for a valid .pt file

for pt_file in "$WEIGHTS_DIR"/*.pt; do
    [ -f "$pt_file" ] || continue
    size=$(wc -c < "$pt_file")
    if [ "$size" -lt "$MIN_SIZE" ]; then
        echo "      Removing corrupt: $(basename "$pt_file") (${size} bytes)"
        rm "$pt_file"
    fi
done
echo "      Cleanup complete."

# ------------------------------------------------------------------
# 7. Download YOLO-seg and RT-DETR model weights
# ------------------------------------------------------------------
echo ""
if [ "$MODE" = "train" ]; then
    echo "[7/8] Downloading training model weights only..."

    # Only download the 3 models used for training
    YOLO_VARIANTS=(
        "yolov8m-seg"
        "yolo11m-seg"
    )
    RTDETR_VARIANTS=()
else
    echo "[7/8] Downloading all model weights to $WEIGHTS_DIR..."

    YOLO_VARIANTS=(
        "yolov8n-seg"
        "yolov8s-seg"
        "yolov8m-seg"
        "yolo11n-seg"
        "yolo11s-seg"
        "yolo11m-seg"
        "yolo26n-seg"
        "yolo26s-seg"
        "yolo26m-seg"
    )
    RTDETR_VARIANTS=(
        "rtdetr-l"
        "rtdetr-x"
    )
fi

download_yolo_model() {
    local variant="$1"
    local pt_file="$WEIGHTS_DIR/${variant}.pt"

    if [ -f "$pt_file" ]; then
        echo "      [SKIP] $(basename $pt_file) already exists"
        return 0
    fi

    echo "      Downloading $(basename $pt_file) ..."
    python -c "
import os
os.chdir('$WEIGHTS_DIR')
from ultralytics import YOLO
try:
    model = YOLO('${variant}.pt')
    print('        OK: ${variant}.pt')
except Exception as e:
    print(f'        WARN: Could not download ${variant}.pt: {e}')
" 2>/dev/null || echo "        WARN: Failed to download ${variant}.pt"
}

download_rtdetr_model() {
    local variant="$1"
    local pt_file="$WEIGHTS_DIR/${variant}.pt"

    if [ -f "$pt_file" ]; then
        echo "      [SKIP] $(basename $pt_file) already exists"
        return 0
    fi

    echo "      Downloading $(basename $pt_file) ..."
    python -c "
import os
os.chdir('$WEIGHTS_DIR')
from ultralytics import RTDETR
try:
    model = RTDETR('${variant}.pt')
    print('        OK: ${variant}.pt')
except Exception as e:
    print(f'        WARN: Could not download ${variant}.pt: {e}')
" 2>/dev/null || echo "        WARN: Failed to download ${variant}.pt"
}

echo "      --- YOLO-seg weights ---"
for variant in "${YOLO_VARIANTS[@]}"; do
    download_yolo_model "$variant"
done

if [ ${#RTDETR_VARIANTS[@]} -gt 0 ]; then
    echo ""
    echo "      --- RT-DETR weights ---"
    for variant in "${RTDETR_VARIANTS[@]}"; do
        download_rtdetr_model "$variant"
    done
fi

# ------------------------------------------------------------------
# 8. Pre-download MiDaS depth estimation model (skip for --train)
# ------------------------------------------------------------------
echo ""
if [ "$MODE" = "train" ]; then
    echo "[8/8] Skipping MiDaS download (not needed for training)"
else
    echo "[8/8] Pre-downloading MiDaS depth estimation model..."
    MIDAS_CHECKPOINT="$HOME/.cache/torch/hub/checkpoints/midas_v21_small_256.pt"
    if [ -f "$MIDAS_CHECKPOINT" ]; then
        echo "      [SKIP] MiDaS_small already cached"
    else
        echo "      Downloading MiDaS_small (~82MB)..."
        python -c "
import torch
try:
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
    print('        OK: MiDaS_small downloaded and cached')
except Exception as e:
    print(f'        WARN: Could not download MiDaS: {e}')
" 2>/dev/null || echo "        WARN: MiDaS download failed (will retry on first use)"
    fi
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Setup Complete!  [MODE: $MODE]"
echo "============================================================"
echo ""
echo "  To activate the environment:"
echo "    source venv/bin/activate"
echo ""

if [ "$MODE" = "train" ]; then
    echo "  To run training (RTX 4070 Ti recommended settings):"
    echo "    python scripts/train.py --mode train \\"
    echo "      --models unet_resnet18 yolov8m_seg yolov11m_seg \\"
    echo "      --epochs 40 --early-stopping 25 --batch-size 16 --amp"
    echo ""
    echo "  Estimated venv size: ~3.5 GB"
elif [ "$MODE" = "server" ]; then
    echo "  To start the dashboard server:"
    echo "    python server/server.py --port 5001"
    echo ""
else
    echo "  To run training:"
    echo "    python scripts/train.py --mode train --optimizer adamw --early-stopping 10"
    echo ""
    echo "  To run benchmark:"
    echo "    python scripts/train.py --mode benchmark"
    echo ""
    echo "  To run full pipeline:"
    echo "    python scripts/train.py --mode full"
fi
echo ""
