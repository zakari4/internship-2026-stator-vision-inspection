#!/usr/bin/env python3
"""
Regenerate all benchmark plots integrating YOLO + PyTorch model results.

YOLO models were trained on augmented data (3,460 images).
PyTorch models (UNet ResNet18, SegFormer B0) were trained on original data (346 images)
due to hardware constraints (training on augmented data is prohibitively slow).
"""

import os
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global variables that will be set in main()
RESULTS_DIR = ""
PLOTS_DIR = ""
LOGS_DIR = ""
YOLO_DIR = ""

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'unet_resnet18': '#1f77b4',
    'segformer_b0': '#ff7f0e',
    'yolov8m_seg': '#2ca02c',
    'yolov11m_seg': '#d62728',
}
MODEL_LABELS = {
    'unet_resnet18': 'UNet ResNet18',
    'segformer_b0': 'SegFormer B0',
    'yolov8m_seg': 'YOLOv8m-seg *',
    'yolov11m_seg': 'YOLOv11m-seg *',
}


def load_pytorch_history(model_name):
    """Load per-epoch data from PyTorch training history JSON."""
    path = os.path.join(LOGS_DIR, model_name, "training_history.json")
    with open(path) as f:
        data = json.load(f)
    epochs_data = data.get("epochs", [])
    result = {
        "model_name": model_name,
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_dice": [],
        "val_dice": [],
        "gpu_mem": [],
        "gpu_util": [],
        "cpu_percent": [],
        "throughput": [],
        "forward_ms": [],
        "backward_ms": [],
        "best_val_iou": data.get("best_val_iou", 0),
        "best_epoch": data.get("best_epoch", 0),
        "total_train_time_sec": data.get("total_train_time_sec", 0),
        "peak_gpu_memory_mb": data.get("peak_gpu_memory_mb", 0),
        "avg_gpu_utilization": data.get("avg_gpu_utilization", 0),
    }
    for ep in epochs_data:
        result["epochs"].append(ep["epoch"])
        result["train_loss"].append(ep["train"]["loss"])
        result["val_loss"].append(ep["val"]["loss"])
        result["train_iou"].append(ep["train"]["iou"])
        result["val_iou"].append(ep["val"]["iou"])
        result["train_dice"].append(ep["train"]["dice"])
        result["val_dice"].append(ep["val"]["dice"])
        result["gpu_mem"].append(ep["train"].get("gpu_mem_max_mb", 0))
        result["gpu_util"].append(ep["train"].get("gpu_util_mean", 0))
        result["cpu_percent"].append(ep["train"].get("cpu_percent_mean", 0))
        result["throughput"].append(ep["train"].get("throughput", 0))
        result["forward_ms"].append(ep["train"].get("avg_forward_ms", 0))
        result["backward_ms"].append(ep["train"].get("avg_backward_ms", 0))
    return result


def load_yolo_history(model_name):
    """Load per-epoch data from YOLO results.csv + training_history.json."""
    csv_path = os.path.join(YOLO_DIR, model_name, "results.csv")
    json_path = os.path.join(LOGS_DIR, model_name, "training_history.json")

    with open(json_path) as f:
        meta = json.load(f)

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    result = {
        "model_name": model_name,
        "epochs": [],
        "train_loss": [],   # seg_loss as proxy
        "val_loss": [],      # val/seg_loss
        "train_iou": [],     # not directly available, use mAP50-95(M) as proxy
        "val_iou": [],       # metrics/mAP50-95(M)
        "train_dice": [],
        "val_dice": [],
        "gpu_mem": [],
        "gpu_util": [],
        "cpu_percent": [],
        "throughput": [],
        "forward_ms": [],
        "backward_ms": [],
        "best_val_iou": meta.get("best_val_iou", 0),
        "best_epoch": meta.get("best_epoch", 0),
        "total_train_time_sec": meta.get("total_train_time_sec", 0),
        "peak_gpu_memory_mb": meta.get("peak_gpu_memory_mb", 0),
        "avg_gpu_utilization": meta.get("avg_gpu_utilization", 0),
    }

    for row in rows:
        # Strip whitespace from keys
        row = {k.strip(): v for k, v in row.items()}
        ep = int(row["epoch"])
        result["epochs"].append(ep)

        # YOLO losses
        train_seg = float(row.get("train/seg_loss", 0))
        val_seg = float(row.get("val/seg_loss", 0))
        train_box = float(row.get("train/box_loss", 0))
        val_box = float(row.get("val/box_loss", 0))
        result["train_loss"].append(train_seg + train_box)
        result["val_loss"].append(val_seg + val_box)

        # YOLO mask metrics
        map50_95_m = float(row.get("metrics/mAP50-95(M)", 0))
        map50_m = float(row.get("metrics/mAP50(M)", 0))
        result["val_iou"].append(map50_95_m)
        result["train_iou"].append(map50_95_m)  # only val metric available

        # Use mAP50(M) as dice proxy
        result["val_dice"].append(map50_m)
        result["train_dice"].append(map50_m)

        # Estimate GPU/timing from metadata (constant for YOLO)
        time_sec = float(row.get("time", 0))
        result["gpu_mem"].append(meta.get("peak_gpu_memory_mb", 0))
        result["gpu_util"].append(meta.get("avg_gpu_utilization", 0))
        result["cpu_percent"].append(50.0)  # approximate
        # YOLO doesn't report throughput per epoch in CSV; estimate from time
        # Assume ~3460 images per epoch for YOLO (augmented data)
        n_images = 3460
        thr = n_images / time_sec if time_sec > 0 else 0
        result["throughput"].append(thr)
        result["forward_ms"].append(0)
        result["backward_ms"].append(0)

    return result


def plot_val_loss(models, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        ax.plot(m["epochs"], m["val_loss"], '-', color=COLORS[m["model_name"]],
                label=MODEL_LABELS[m["model_name"]], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Comparison (All Models)', fontsize=14)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    note = "* YOLO loss = seg_loss + box_loss (not directly comparable to CE+Dice)"
    ax.annotate(note, xy=(0.01, 0.01), xycoords='axes fraction', fontsize=7,
                fontstyle='italic', color='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_val_iou(models, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        ax.plot(m["epochs"], m["val_iou"], '-', color=COLORS[m["model_name"]],
                label=MODEL_LABELS[m["model_name"]], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation IoU', fontsize=12)
    ax.set_title('Validation IoU Comparison (All Models)', fontsize=14)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 1)
    note = "* YOLO IoU = mAP50-95 (Mask);  * trained on augmented data (3,460 images)"
    ax.annotate(note, xy=(0.01, 0.01), xycoords='axes fraction', fontsize=7,
                fontstyle='italic', color='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_val_dice(models, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        ax.plot(m["epochs"], m["val_dice"], '-', color=COLORS[m["model_name"]],
                label=MODEL_LABELS[m["model_name"]], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Dice / mAP50', fontsize=12)
    ax.set_title('Validation Dice / mAP50 Comparison (All Models)', fontsize=14)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 1)
    note = "* YOLO uses mAP50 (Mask) as Dice proxy"
    ax.annotate(note, xy=(0.01, 0.01), xycoords='axes fraction', fontsize=7,
                fontstyle='italic', color='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_gpu_usage(models, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m in models:
        c = COLORS[m["model_name"]]
        lbl = MODEL_LABELS[m["model_name"]]
        axes[0].plot(m["epochs"], m["gpu_mem"], '-', color=c, label=lbl, linewidth=2)
        axes[1].plot(m["epochs"], m["gpu_util"], '-', color=c, label=lbl, linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('GPU Memory (MB)')
    axes[0].set_title('Peak GPU Memory'); axes[0].legend(fontsize=8)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('GPU Utilization (%)')
    axes[1].set_title('GPU Utilization'); axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_throughput(models, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [MODEL_LABELS[m["model_name"]] for m in models]
    avgs = [np.mean(m["throughput"]) for m in models]
    colors = [COLORS[m["model_name"]] for m in models]
    bars = ax.bar(names, avgs, color=colors)
    for bar, val in zip(bars, avgs):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10)
    ax.set_ylabel('Avg Training Throughput (samples/sec)', fontsize=12)
    ax.set_title('Training Throughput Comparison', fontsize=14)
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_time(models, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [MODEL_LABELS[m["model_name"]] for m in models]
    times_min = [m["total_train_time_sec"] / 60 for m in models]
    colors = [COLORS[m["model_name"]] for m in models]
    bars = ax.bar(names, times_min, color=colors)
    for bar, val in zip(bars, times_min):
        if val >= 60:
            label = f'{val/60:.1f} h'
        else:
            label = f'{val:.1f} min'
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10)
    ax.set_ylabel('Total Training Time (minutes)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14)
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_best_iou_bar(models, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [MODEL_LABELS[m["model_name"]] for m in models]
    ious = [m["best_val_iou"] for m in models]
    colors = [COLORS[m["model_name"]] for m in models]
    bars = ax.bar(names, ious, color=colors)
    for bar, val in zip(bars, ious):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10,
                    fontweight='bold')
    ax.set_ylabel('Best Validation IoU', fontsize=12)
    ax.set_title('Best Validation IoU Comparison', fontsize=14)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=15)
    note = "* YOLO models trained on augmented data (3,460 images); PyTorch models on original data (346 images)"
    ax.annotate(note, xy=(0.01, 0.01), xycoords='axes fraction', fontsize=7,
                fontstyle='italic', color='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_accuracy_vs_speed(models, save_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    for m in models:
        name = MODEL_LABELS[m["model_name"]]
        c = COLORS[m["model_name"]]
        iou = m["best_val_iou"]
        time_min = m["total_train_time_sec"] / 60
        gpu = m["peak_gpu_memory_mb"]
        ax.scatter(time_min, iou, s=gpu/5, c=c, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.annotate(name, xy=(time_min, iou), xytext=(8, 8),
                    textcoords='offset points', fontsize=9, color=c, fontweight='bold')
    ax.set_xlabel('Total Training Time (minutes)', fontsize=12)
    ax.set_ylabel('Best Validation IoU', fontsize=12)
    ax.set_title('Accuracy vs Training Time (bubble size = GPU memory)', fontsize=14)
    ax.set_ylim(0.4, 1.0)
    note = "* YOLO models on augmented data; PyTorch models on original data"
    ax.annotate(note, xy=(0.01, 0.01), xycoords='axes fraction', fontsize=7,
                fontstyle='italic', color='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comprehensive(models, save_path):
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Val Loss
    ax1 = fig.add_subplot(gs[0, 0])
    for m in models:
        ax1.plot(m["epochs"], m["val_loss"], color=COLORS[m["model_name"]],
                 label=MODEL_LABELS[m["model_name"]])
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Val Loss')
    ax1.set_title('Validation Loss'); ax1.legend(fontsize=7)

    # 2. Val IoU
    ax2 = fig.add_subplot(gs[0, 1])
    for m in models:
        ax2.plot(m["epochs"], m["val_iou"], color=COLORS[m["model_name"]],
                 label=MODEL_LABELS[m["model_name"]])
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU'); ax2.legend(fontsize=7); ax2.set_ylim(0, 1)

    # 3. Val Dice
    ax3 = fig.add_subplot(gs[0, 2])
    for m in models:
        ax3.plot(m["epochs"], m["val_dice"], color=COLORS[m["model_name"]],
                 label=MODEL_LABELS[m["model_name"]])
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Dice / mAP50')
    ax3.set_title('Validation Dice / mAP50'); ax3.legend(fontsize=7); ax3.set_ylim(0, 1)

    # 4. GPU Memory
    ax4 = fig.add_subplot(gs[1, 0])
    for m in models:
        ax4.plot(m["epochs"], m["gpu_mem"], color=COLORS[m["model_name"]],
                 label=MODEL_LABELS[m["model_name"]])
    ax4.set_xlabel('Epoch'); ax4.set_ylabel('GPU Memory (MB)')
    ax4.set_title('Peak GPU Memory'); ax4.legend(fontsize=7)

    # 5. GPU Utilization
    ax5 = fig.add_subplot(gs[1, 1])
    for m in models:
        ax5.plot(m["epochs"], m["gpu_util"], color=COLORS[m["model_name"]],
                 label=MODEL_LABELS[m["model_name"]])
    ax5.set_xlabel('Epoch'); ax5.set_ylabel('GPU Util (%)')
    ax5.set_title('GPU Utilization'); ax5.legend(fontsize=7); ax5.set_ylim(0, 100)

    # 6. Training Time bar
    ax6 = fig.add_subplot(gs[1, 2])
    names = [MODEL_LABELS[m["model_name"]] for m in models]
    times = [m["total_train_time_sec"] / 60 for m in models]
    colors = [COLORS[m["model_name"]] for m in models]
    bars6 = ax6.bar(names, times, color=colors)
    for bar, val in zip(bars6, times):
        lbl = f'{val/60:.1f}h' if val >= 60 else f'{val:.0f}m'
        ax6.annotate(lbl, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    ax6.set_ylabel('Time (min)'); ax6.set_title('Total Training Time')
    ax6.tick_params(axis='x', rotation=30, labelsize=7)

    # 7. Throughput bar
    ax7 = fig.add_subplot(gs[2, 0])
    avgs = [np.mean(m["throughput"]) for m in models]
    bars7 = ax7.bar(names, avgs, color=colors)
    for bar, val in zip(bars7, avgs):
        ax7.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    ax7.set_ylabel('Samples/sec'); ax7.set_title('Avg Throughput')
    ax7.tick_params(axis='x', rotation=30, labelsize=7)

    # 8. Peak GPU bar
    ax8 = fig.add_subplot(gs[2, 1])
    gpus = [m["peak_gpu_memory_mb"] for m in models]
    bars8 = ax8.bar(names, gpus, color=colors)
    for bar, val in zip(bars8, gpus):
        ax8.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    ax8.set_ylabel('GPU Memory (MB)'); ax8.set_title('Peak GPU Memory')
    ax8.tick_params(axis='x', rotation=30, labelsize=7)

    # 9. Best IoU bar
    ax9 = fig.add_subplot(gs[2, 2])
    ious = [m["best_val_iou"] for m in models]
    bars9 = ax9.bar(names, ious, color=colors)
    for bar, val in zip(bars9, ious):
        ax9.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8,
                     fontweight='bold')
    ax9.set_ylabel('Best Val IoU'); ax9.set_title('Best Validation IoU')
    ax9.set_ylim(0, 1)
    ax9.tick_params(axis='x', rotation=30, labelsize=7)

    fig.suptitle('Comprehensive Training Comparison — All Models\n'
                 '(* YOLO trained on augmented data; PyTorch on original data)',
                 fontsize=13, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_final_table(models, save_path):
    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.axis('off')

    columns = ['Model', 'Dataset', 'Best Epoch', 'Val IoU', 'Val Dice/mAP50',
               'Train Time', 'Peak GPU (MB)', 'GPU Util (%)']
    data = []
    for m in models:
        is_yolo = 'yolo' in m["model_name"]
        dataset = "Augmented (3,460)" if is_yolo else "Original (346)"
        tt = m["total_train_time_sec"]
        if tt >= 3600:
            time_str = f'{tt/3600:.1f} h'
        else:
            time_str = f'{tt/60:.1f} min'
        dice_val = m["val_dice"][-1] if m["val_dice"] else 0
        data.append([
            MODEL_LABELS[m["model_name"]].replace(' *', ''),
            dataset,
            str(m["best_epoch"]),
            f'{m["best_val_iou"]:.4f}',
            f'{dice_val:.4f}',
            time_str,
            f'{m["peak_gpu_memory_mb"]:.0f}',
            f'{m["avg_gpu_utilization"]:.1f}',
        ])

    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(data) + 1):
        color = '#E2EFDA' if i % 2 == 0 else 'white'
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)

    plt.title('Training Results Comparison — All Models', fontsize=14, pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Regenerate training plots.")
    parser.add_argument("--results-dir", type=str, default="outputs/results",
                        help="Path to results directory (relative to project root or absolute)")
    args = parser.parse_args()

    global RESULTS_DIR, PLOTS_DIR, LOGS_DIR, YOLO_DIR
    
    if os.path.isabs(args.results_dir):
        RESULTS_DIR = args.results_dir
    else:
        RESULTS_DIR = os.path.join(PROJECT_ROOT, args.results_dir)
        
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    LOGS_DIR = os.path.join(RESULTS_DIR, "training_logs")
    YOLO_DIR = os.path.join(RESULTS_DIR, "yolo_training")

    print(f"Targeting Results Directory: {RESULTS_DIR}")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading model histories...")

    all_models = []
    
    try:
        unet = load_pytorch_history("unet_resnet18")
        all_models.append(unet)
        print(f"  UNet ResNet18:  best IoU={unet['best_val_iou']:.4f}, {unet['total_train_time_sec']/60:.1f} min")
    except FileNotFoundError: pass

    try:
        segformer = load_pytorch_history("segformer_b0")
        all_models.append(segformer)
        print(f"  SegFormer B0:   best IoU={segformer['best_val_iou']:.4f}, {segformer['total_train_time_sec']/60:.1f} min")
    except FileNotFoundError: pass

    try:
        yolov8 = load_yolo_history("yolov8m_seg")
        all_models.append(yolov8)
        print(f"  YOLOv8m-seg:    best IoU={yolov8['best_val_iou']:.4f}, {yolov8['total_train_time_sec']/60:.1f} min")
    except FileNotFoundError: pass

    try:
        yolov11 = load_yolo_history("yolov11m_seg")
        all_models.append(yolov11)
        print(f"  YOLOv11m-seg:   best IoU={yolov11['best_val_iou']:.4f}, {yolov11['total_train_time_sec']/60:.1f} min")
    except FileNotFoundError: pass


    print("\nGenerating plots...")

    plot_val_loss(all_models, os.path.join(PLOTS_DIR, "loss_curves.png"))
    print("  Created: loss_curves.png")

    plot_val_iou(all_models, os.path.join(PLOTS_DIR, "iou_curves.png"))
    print("  Created: iou_curves.png")

    plot_val_dice(all_models, os.path.join(PLOTS_DIR, "dice_curves.png"))
    print("  Created: dice_curves.png")

    plot_gpu_usage(all_models, os.path.join(PLOTS_DIR, "gpu_usage.png"))
    print("  Created: gpu_usage.png")

    plot_throughput(all_models, os.path.join(PLOTS_DIR, "throughput.png"))
    print("  Created: throughput.png")

    plot_training_time(all_models, os.path.join(PLOTS_DIR, "training_time.png"))
    print("  Created: training_time.png")

    plot_best_iou_bar(all_models, os.path.join(PLOTS_DIR, "best_iou_comparison.png"))
    print("  Created: best_iou_comparison.png")

    plot_accuracy_vs_speed(all_models, os.path.join(PLOTS_DIR, "accuracy_vs_speed.png"))
    print("  Created: accuracy_vs_speed.png")

    plot_comprehensive(all_models, os.path.join(PLOTS_DIR, "comprehensive_comparison.png"))
    print("  Created: comprehensive_comparison.png")

    plot_final_table(all_models, os.path.join(PLOTS_DIR, "final_comparison_table.png"))
    print("  Created: final_comparison_table.png")

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
