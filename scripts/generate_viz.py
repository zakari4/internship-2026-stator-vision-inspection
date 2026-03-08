#!/usr/bin/env python3
"""Generate comprehensive visualization comparing all benchmark results."""

import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np


def load_all_results():
    """Load results from all models."""
    results = {}
    base_dir = Path(_PROJECT_ROOT) / 'outputs' / 'results' / 'training_logs'
    
    # Load training results
    if base_dir.exists():
        for model_dir in base_dir.iterdir():
            if model_dir.is_dir():
                history_file = model_dir / 'training_history.json'
                if history_file.exists():
                    with open(history_file) as f:
                        data = json.load(f)
                        model_name = model_dir.name
                        results[model_name] = {
                            'type': 'trained',
                            'best_val_iou': data.get('best_val_iou', 0),
                            'best_val_dice': data.get('best_val_dice', 0),
                            'best_epoch': data.get('best_epoch', 0),
                            'total_time': data.get('total_time_seconds', 0),
                            'val_iou_history': data.get('val_iou', []),
                            'val_loss_history': data.get('val_loss', [])
                        }
    
    # Load SOTA edge detection benchmark results
    sota_file = Path(_PROJECT_ROOT) / 'outputs' / 'results' / 'sota_edge_detection_results.json'
    if sota_file.exists():
        with open(sota_file) as f:
            sota_results = json.load(f)
            for name, data in sota_results.items():
                if name not in results:
                    results[name] = {
                        'type': 'benchmark_only',
                        'best_val_iou': data.get('iou_mean', 0),
                        'best_val_dice': data.get('dice_mean', 0),
                        'fps': data.get('fps', 0),
                        'latency_ms': data.get('latency_mean_ms', 0)
                    }
    
    return results


def create_comparison_plots(results):
    """Create comparison visualizations."""
    output_dir = Path(_PROJECT_ROOT) / 'outputs' / 'results' / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter and prepare data
    trained_models = {k: v for k, v in results.items() if v.get('type') == 'trained' and v['best_val_iou'] > 0.01}
    
    # Sort by IoU
    sorted_models = sorted(trained_models.items(), key=lambda x: x[1]['best_val_iou'], reverse=True)
    
    # 1. Bar chart: IoU comparison
    plt.figure(figsize=(14, 8))
    names = [m[0] for m in sorted_models]
    ious = [m[1]['best_val_iou'] for m in sorted_models]
    
    colors = []
    for name in names:
        if 'yolo' in name.lower():
            colors.append('#FF6B6B')  # Red for YOLO
        elif 'unet' in name.lower():
            colors.append('#4ECDC4')  # Teal for UNet
        elif any(x in name.lower() for x in ['hed', 'rcf', 'bdcn', 'pidinet', 'teed', 'ldc']):
            colors.append('#45B7D1')  # Blue for edge detection
        else:
            colors.append('#96CEB4')  # Green for others
    
    bars = plt.bar(range(len(names)), ious, color=colors)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('Validation IoU')
    plt.title('Model Comparison: Best Validation IoU')
    plt.ylim(0, 1.05)
    
    # Add value labels
    for bar, iou in zip(bars, ious):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{iou:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='YOLO Seg'),
        Patch(facecolor='#4ECDC4', label='UNet'),
        Patch(facecolor='#45B7D1', label='Edge Detection'),
        Patch(facecolor='#96CEB4', label='Other')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_iou.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'model_comparison_iou.png'}")
    
    # 2. Training curves (for models with history)
    plt.figure(figsize=(14, 8))
    
    for name, data in sorted_models[:10]:  # Top 10 models
        if 'val_iou_history' in data and data['val_iou_history']:
            epochs = range(1, len(data['val_iou_history']) + 1)
            plt.plot(epochs, data['val_iou_history'], label=f"{name} ({data['best_val_iou']:.3f})", linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation IoU')
    plt.title('Training Progress: Validation IoU over Epochs')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'training_curves.png'}")
    
    # 3. Summary table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Best Val IoU':>12} {'Best Val Dice':>14} {'Best Epoch':>12}")
    print("-"*80)
    
    for name, data in sorted_models:
        iou = data.get('best_val_iou', 0)
        dice = data.get('best_val_dice', 0)
        epoch = data.get('best_epoch', 'N/A')
        print(f"{name:<25} {iou:>12.4f} {dice:>14.4f} {str(epoch):>12}")
    
    print("="*80)
    
    # 4. Create grouped comparison by model type
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group by type
    yolo_models = [(k, v) for k, v in sorted_models if 'yolo' in k.lower()]
    dl_models = [(k, v) for k, v in sorted_models if 'yolo' not in k.lower()]
    
    # YOLO comparison
    if yolo_models:
        ax = axes[0]
        names = [m[0] for m in yolo_models]
        ious = [m[1]['best_val_iou'] for m in yolo_models]
        bars = ax.barh(range(len(names)), ious, color='#FF6B6B')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Validation IoU')
        ax.set_title('YOLO Segmentation Models')
        ax.set_xlim(0, 1.0)
        for i, (bar, iou) in enumerate(zip(bars, ious)):
            ax.text(iou + 0.02, i, f'{iou:.3f}', va='center', fontsize=9)
    
    # DL comparison
    if dl_models:
        ax = axes[1]
        names = [m[0] for m in dl_models]
        ious = [m[1]['best_val_iou'] for m in dl_models]
        colors = []
        for name in names:
            if 'unet' in name.lower():
                colors.append('#4ECDC4')
            elif any(x in name.lower() for x in ['hed', 'rcf', 'bdcn', 'pidinet', 'teed', 'ldc']):
                colors.append('#45B7D1')
            else:
                colors.append('#96CEB4')
        bars = ax.barh(range(len(names)), ious, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Validation IoU')
        ax.set_title('Deep Learning Models')
        ax.set_xlim(0, 1.0)
        for i, (bar, iou) in enumerate(zip(bars, ious)):
            ax.text(iou + 0.02, i, f'{iou:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_grouped.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'model_comparison_grouped.png'}")
    
    # 5. Top 5 models comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top5 = sorted_models[:5]
    names = [m[0] for m in top5]
    ious = [m[1]['best_val_iou'] for m in top5]
    dices = [m[1].get('best_val_dice', 0) for m in top5]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ious, width, label='IoU', color='#4ECDC4')
    bars2 = ax.bar(x + width/2, dices, width, label='Dice', color='#FF6B6B')
    
    ax.set_ylabel('Score')
    ax.set_title('Top 5 Models: IoU vs Dice Score')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top5_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'top5_comparison.png'}")
    
    return sorted_models


def main():
    print("="*60)
    print("Generating Benchmark Visualizations")
    print("="*60)
    
    results = load_all_results()
    print(f"\nLoaded results for {len(results)} models")
    
    sorted_results = create_comparison_plots(results)
    
    print("\n" + "="*60)
    print("TOP 5 MODELS")
    print("="*60)
    for i, (name, data) in enumerate(sorted_results[:5], 1):
        print(f"{i}. {name}: IoU = {data['best_val_iou']:.4f}")
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
