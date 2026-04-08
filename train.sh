#!/bin/bash
set -e

echo "Starting model training..."
python3 scripts/train.py --mode train \
  --models yolov8m_seg yolov11m_seg unet_resnet18 \
  --early-stopping 30 \
  --epochs 40 \
  --batch-size 8 \
  --lr 0.000001 \
  --optimizer adamw \
  --scheduler plateau \
  --output ./output

echo "Training completed successfully."
echo "Committing and pushing outputs to repository..."

git add ./output
git commit -m "update output by training"
git push origin dev

echo "All done!"
