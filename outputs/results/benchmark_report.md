# Benchmark Results

**Date**: 2026-03-12  
**Hardware**: NVIDIA GTX 1650 (4 GB VRAM), Linux  
**Training**: 20 epochs, AdamW + Cosine Annealing LR, 512×512  
**Labels**: `michanical_part`, `magnet`, `circle` (multi-class segmentation, 4 classes incl. background)

> **Important — Dataset Difference**:  
> - **YOLO models** (YOLOv8m-seg, YOLOv11m-seg) were trained on **augmented data** (3,460 images, 10× augmentation).  
> - **PyTorch models** (UNet ResNet18, SegFormer B0) were trained on the **original dataset** (346 images) due to **hardware constraints** — training these models on the full augmented set takes prohibitively long on a GTX 1650 with limited VRAM.  
> This means YOLO models had ~10× more training data, which should be considered when comparing results.

---

## Model Ranking (by Best Validation IoU)

| Rank | Model | Type | Dataset | Best IoU | Best Epoch | Train Time | Peak GPU (MB) |
|------|-------|------|---------|----------|------------|------------|---------------|
| 1 | UNet ResNet18 | Multi-class seg. | Original (346) | **0.8763** | 19 | 12.4 min | 3,585 |
| 2 | YOLOv8m-seg | Instance seg. | Augmented (3,460) | 0.8569 | 20 | 1 h 52 min | 4,086 |
| 3 | YOLOv11m-seg | Instance seg. | Augmented (3,460) | 0.8476 | 20 | 2 h 32 min | 3,975 |
| 4 | SegFormer B0 | Multi-class seg. | Original (346) | 0.5954 | 20 | 3.0 min | 1,962 |

---

## Detailed Training Results

### UNet ResNet18 (Original Data — 346 images)
- **Best Val IoU**: 0.8763 at epoch 19
- **Best Val Loss**: 0.3166
- **Val Dice**: 0.9324 | **Precision**: 0.9545 | **Recall**: 0.9835 | **Accuracy**: 98.64%
- **Training Time**: 12.4 min (802.9 s) | **Peak GPU**: 3,585 MB | **GPU Util**: 96.1%
- **Throughput**: ~6.5 samples/s

### SegFormer B0 (Original Data — 346 images)
- **Best Val IoU**: 0.5954 at epoch 20
- **Best Val Loss**: 0.2674
- **Val Dice**: 0.7169 | **Precision**: 0.8562 | **Recall**: 0.9051 | **Accuracy**: 95.57%
- **Training Time**: 3.0 min (177.8 s) | **Peak GPU**: 1,962 MB | **GPU Util**: 86.0%
- **Throughput**: ~27.0 samples/s

### YOLOv8m-seg (Augmented Data — 3,460 images)
- **Best mAP50-95 (Mask)**: 0.8569 at epoch 20
- **mAP50 (Mask)**: 0.9916 | **Precision (M)**: 0.9922 | **Recall (M)**: 0.9956
- **Training Time**: 1 h 52 min (6,746 s) | **Peak GPU**: 4,086 MB | **GPU Util**: 83.2%

### YOLOv11m-seg (Augmented Data — 3,460 images)
- **Best mAP50-95 (Mask)**: 0.8476 at epoch 20
- **mAP50 (Mask)**: 0.9932 | **Precision (M)**: 0.9902 | **Recall (M)**: 0.9970
- **Training Time**: 2 h 32 min (9,103 s) | **Peak GPU**: 3,975 MB | **GPU Util**: 81.1%

---

## Deployment Recommendation

- **Best Accuracy**: UNet ResNet18 (IoU: 0.8763) — top performer even on the smaller original dataset
- **Fastest Training**: SegFormer B0 (3.0 min) — extremely efficient, suitable for rapid prototyping
- **Best Instance Detection**: YOLOv8m-seg — highest mAP50 (0.9916) with per-class bounding boxes + masks
- **Best Tradeoff**: UNet ResNet18 — best IoU in just 12 min training

---

## Comparison Plots

All plots include all 4 models and are stored in `outputs/results/plots/`.

![Comprehensive Comparison](../outputs/results/plots/comprehensive_comparison.png)
![Best IoU](../outputs/results/plots/best_iou_comparison.png)
![IoU Curves](../outputs/results/plots/iou_curves.png)
![Loss Curves](../outputs/results/plots/loss_curves.png)
![Training Time](../outputs/results/plots/training_time.png)
![Accuracy vs Speed](../outputs/results/plots/accuracy_vs_speed.png)
![Final Table](../outputs/results/plots/final_comparison_table.png)
