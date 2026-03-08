#!/bin/bash
# =============================================================================
# Stator Case - Chignon Detection Benchmark - Project Setup
# =============================================================================
# Initializes the project: creates venv, installs dependencies, downloads
# all model weights into weights/.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================================

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
REQ_FILE="$PROJECT_DIR/requirements.txt"
WEIGHTS_DIR="$PROJECT_DIR/weights"

echo "============================================================"
echo "  Stator Case - Chignon Detection Benchmark - Setup"
echo "============================================================"
echo ""
echo "Project directory: $PROJECT_DIR"
echo ""

# ------------------------------------------------------------------
# 1. Create virtual environment
# ------------------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
    echo "[1/7] Virtual environment already exists at: $VENV_DIR"
else
    echo "[1/7] Creating virtual environment..."
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
echo "[2/7] Upgrading pip..."
pip install --upgrade pip --quiet

# ------------------------------------------------------------------
# 3. Install requirements
# ------------------------------------------------------------------
echo ""
echo "[3/7] Installing requirements..."
pip install -r "$REQ_FILE" --quiet
pip install scikit-learn --quiet  # needed by contour extraction
echo "      All dependencies installed."

# ------------------------------------------------------------------
# 4. Augment data (72x)
# ------------------------------------------------------------------
echo ""
AUG_DIR="$PROJECT_DIR/outputs/augmented_data"
if [ -d "$AUG_DIR" ] && [ "$(ls -A $AUG_DIR/*.png 2>/dev/null | head -1)" ]; then
    AUG_COUNT=$(ls -1 "$AUG_DIR"/*.png 2>/dev/null | wc -l)
    echo "[4/7] Augmented data already exists ($AUG_COUNT images in $AUG_DIR)"
else
    echo "[4/7] Augmenting data (72x)..."
    python "$PROJECT_DIR/src/data/augmentation.py" \
        --input "$PROJECT_DIR/data" \
        --output "$AUG_DIR"
    echo "      Augmentation complete."
fi

# ------------------------------------------------------------------
# 5. Prepare YOLO format dataset
# ------------------------------------------------------------------
echo ""
YOLO_DIR="$PROJECT_DIR/outputs/yolo_dataset"
if [ -d "$YOLO_DIR" ] && [ -f "$YOLO_DIR/data.yaml" ]; then
    echo "[5/7] YOLO dataset already exists at: $YOLO_DIR"
else
    echo "[5/7] Preparing YOLO dataset from augmented data..."
    python "$PROJECT_DIR/src/data/yolo_prep.py" \
        --source "$AUG_DIR" \
        --output "$YOLO_DIR" \
        --labels chignon mecparts
    echo "      YOLO dataset ready (Labels: chignon, mecparts)."
fi

# ------------------------------------------------------------------
# 6. Clean up corrupt model files (from interrupted downloads)
# ------------------------------------------------------------------
echo ""
echo "[6/7] Cleaning up corrupt model files..."
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
echo "[7/7] Downloading model weights to $WEIGHTS_DIR..."

# All YOLO-seg and RT-DETR variants used in the benchmark
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

echo ""
echo "      --- RT-DETR weights ---"
for variant in "${RTDETR_VARIANTS[@]}"; do
    download_rtdetr_model "$variant"
done

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "  To activate the environment:"
echo "    source venv/bin/activate"
echo ""
echo "  To run training:"
echo "    python scripts/train.py --mode train --optimizer adamw --early-stopping 10"
echo ""
echo "  To run benchmark:"
echo "    python scripts/train.py --mode benchmark"
echo ""
echo "  To run full pipeline:"
echo "    python scripts/train.py --mode full"
echo ""
