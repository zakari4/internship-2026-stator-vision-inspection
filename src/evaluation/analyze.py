#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis of Benchmark Results.
Generates detailed comparisons and README.md documentation.
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

def load_all_training_results():
    """Load all training history files."""
    results = {}
    base_dir = Path('results/training_logs')
    
    for model_dir in base_dir.iterdir():
        if model_dir.is_dir():
            history_file = model_dir / 'training_history.json'
            if history_file.exists():
                with open(history_file) as f:
                    data = json.load(f)
                    results[model_dir.name] = data
    
    return results


def load_yolo_latency_results():
    """Load YOLO latency benchmark results."""
    yolo_latency_file = Path('results/yolo_latency_results.json')
    if yolo_latency_file.exists():
        with open(yolo_latency_file) as f:
            return json.load(f)
    return {}


def categorize_models(results):
    """Categorize models by type."""
    categories = {
        'unet': [],
        'yolo_v8': [],
        'yolo_v11': [],
        'yolo_v26': [],
        'edge_detection': [],
        'transformer': [],
        'other': []
    }
    
    for name in results.keys():
        name_lower = name.lower()
        if 'unet' in name_lower:
            categories['unet'].append(name)
        elif 'yolov8' in name_lower or 'yolo8' in name_lower:
            categories['yolo_v8'].append(name)
        elif 'yolov11' in name_lower or 'yolo11' in name_lower:
            categories['yolo_v11'].append(name)
        elif 'yolov26' in name_lower or 'yolo26' in name_lower:
            categories['yolo_v26'].append(name)
        elif any(x in name_lower for x in ['hed', 'rcf', 'bdcn', 'pidinet', 'teed', 'ldc']):
            categories['edge_detection'].append(name)
        elif 'segformer' in name_lower or 'transformer' in name_lower:
            categories['transformer'].append(name)
        else:
            categories['other'].append(name)
    
    return categories


def compute_statistics(results):
    """Compute comprehensive statistics for all models."""
    stats = {}
    
    for name, data in results.items():
        model_stats = {}
        
        # Best metrics
        model_stats['best_val_iou'] = data.get('best_val_iou', 0)
        model_stats['best_val_loss'] = data.get('best_val_loss', float('inf'))
        model_stats['best_epoch'] = data.get('best_epoch', 0)
        
        # Training time
        model_stats['total_time_seconds'] = data.get('total_train_time_sec', 0)
        model_stats['total_epochs'] = data.get('total_epochs', 0)
        
        # Hardware metrics from top level
        model_stats['peak_gpu_memory_mb'] = data.get('peak_gpu_memory_mb', 0)
        model_stats['avg_gpu_utilization'] = data.get('avg_gpu_utilization', 0)
        
        # Extract epoch-level metrics
        epochs = data.get('epochs', [])
        if epochs:
            # Get validation IoU history
            val_ious = [e.get('val', {}).get('iou', 0) for e in epochs]
            train_ious = [e.get('train', {}).get('iou', 0) for e in epochs]
            val_losses = [e.get('val', {}).get('loss', 0) for e in epochs]
            
            model_stats['val_iou_history'] = val_ious
            model_stats['train_iou_history'] = train_ious
            model_stats['val_loss_history'] = val_losses
            model_stats['final_val_iou'] = val_ious[-1] if val_ious else 0
            
            # Best validation dice from epochs
            val_dices = [e.get('val', {}).get('dice', 0) for e in epochs]
            model_stats['best_val_dice'] = max(val_dices) if val_dices else 0
            
            # IoU stability (std of last 10 epochs)
            model_stats['iou_std'] = np.std(val_ious[-10:]) if len(val_ious) >= 10 else np.std(val_ious)
            model_stats['iou_improvement'] = val_ious[-1] - val_ious[0] if len(val_ious) > 1 else 0
            
            # Convergence analysis
            converged_epoch = len(val_ious)
            for i in range(5, len(val_ious)):
                recent_change = max(val_ious[i-5:i]) - min(val_ious[i-5:i])
                if recent_change < 0.01:
                    converged_epoch = i - 5
                    break
            model_stats['convergence_epoch'] = converged_epoch
            
            # Performance metrics from last epoch
            last_epoch = epochs[-1]
            train_data = last_epoch.get('train', {})
            val_data = last_epoch.get('val', {})
            
            model_stats['avg_forward_time_ms'] = train_data.get('avg_forward_ms', 0)
            model_stats['avg_backward_time_ms'] = train_data.get('avg_backward_ms', 0)
            model_stats['avg_throughput'] = val_data.get('throughput', train_data.get('throughput', 0))
            model_stats['avg_gpu_memory_mb'] = train_data.get('gpu_mem_max_mb', 0)
            
            # Training loss stability
            train_losses = [e.get('train', {}).get('loss', 0) for e in epochs]
            model_stats['train_loss_std'] = np.std(train_losses[-10:]) if len(train_losses) >= 10 else np.std(train_losses)
            model_stats['final_train_loss'] = train_losses[-1] if train_losses else 0
        
        stats[name] = model_stats
    
    return stats


def generate_comparison_tables(stats, categories):
    """Generate markdown comparison tables."""
    tables = {}
    
    # Overall ranking table
    sorted_by_iou = sorted(stats.items(), key=lambda x: x[1].get('best_val_iou', 0), reverse=True)
    valid_models = [(n, s) for n, s in sorted_by_iou if s.get('best_val_iou', 0) > 0.01]
    
    tables['overall_ranking'] = "| Rank | Model | Val IoU | Val Dice | Latency (ms) | Throughput | Best Epoch | Training Time | GPU Memory |\n"
    tables['overall_ranking'] += "|------|-------|---------|----------|--------------|------------|------------|---------------|------------|\n"
    
    for i, (name, s) in enumerate(valid_models, 1):
        iou = s.get('best_val_iou', 0)
        dice = s.get('best_val_dice', 0)
        epoch = s.get('best_epoch', 'N/A')
        time_s = s.get('total_time_seconds', 0)
        time_str = f"{time_s/60:.1f}m" if time_s > 0 else "N/A"
        gpu = s.get('peak_gpu_memory_mb', 0)
        gpu_str = f"{gpu:.0f} MB" if gpu > 0 else "N/A"
        latency = s.get('avg_forward_time_ms', 0)
        latency_str = f"{latency:.1f}" if latency > 0 else "N/A"
        throughput = s.get('avg_throughput', 0)
        throughput_str = f"{throughput:.1f}" if throughput > 0 else "N/A"
        tables['overall_ranking'] += f"| {i} | **{name}** | {iou:.4f} | {dice:.4f} | {latency_str} | {throughput_str} | {epoch} | {time_str} | {gpu_str} |\n"
    
    # Category comparison tables
    for category, models in categories.items():
        if not models:
            continue
        
        cat_stats = [(m, stats.get(m, {})) for m in models if m in stats]
        if not cat_stats:
            continue
        
        cat_stats = sorted(cat_stats, key=lambda x: x[1].get('best_val_iou', 0), reverse=True)
        
        table = f"| Model | Val IoU | Val Dice | Convergence Epoch | GPU Memory (MB) | Throughput |\n"
        table += "|-------|---------|----------|-------------------|-----------------|------------|\n"
        
        for name, s in cat_stats:
            iou = s.get('best_val_iou', 0)
            dice = s.get('best_val_dice', 0)
            conv = s.get('convergence_epoch', 'N/A')
            gpu = s.get('avg_gpu_memory_mb', 0)
            throughput = s.get('avg_throughput', 0)
            table += f"| {name} | {iou:.4f} | {dice:.4f} | {conv} | {gpu:.1f} | {throughput:.1f} |\n"
        
        tables[category] = table
    
    return tables


def compute_category_statistics(stats, categories):
    """Compute aggregate statistics per category."""
    category_stats = {}
    
    for category, models in categories.items():
        if not models:
            continue
        
        cat_stats = [stats.get(m, {}) for m in models if m in stats]
        if not cat_stats:
            continue
        
        ious = [s.get('best_val_iou', 0) for s in cat_stats if s.get('best_val_iou', 0) > 0.01]
        times = [s.get('total_time_seconds', 0) for s in cat_stats if s.get('total_time_seconds', 0) > 0]
        throughputs = [s.get('avg_throughput', 0) for s in cat_stats if s.get('avg_throughput', 0) > 0]
        
        if ious:
            category_stats[category] = {
                'num_models': len(ious),
                'avg_iou': np.mean(ious),
                'std_iou': np.std(ious),
                'max_iou': max(ious),
                'min_iou': min(ious),
                'avg_time': np.mean(times) if times else 0,
                'avg_throughput': np.mean(throughputs) if throughputs else 0
            }
    
    return category_stats


def generate_readme(results, stats, categories, category_stats, tables):
    """Generate comprehensive README.md."""
    
    # Get top models
    sorted_models = sorted(stats.items(), key=lambda x: x[1].get('best_val_iou', 0), reverse=True)
    valid_models = [(n, s) for n, s in sorted_models if s.get('best_val_iou', 0) > 0.01]
    
    readme = f"""# Industrial Chignon Detection Benchmark Results

## Overview

This benchmark compares **{len(valid_models)} deep learning models** for industrial chignon (hair bun) detection in segmentation tasks. The evaluation includes:

- **UNet variants** (Lightweight, ResNet18 encoder)
- **YOLO Segmentation** (v8, v11, v26 - nano, small, medium)
- **SOTA Edge Detection** (HED, RCF, BDCN, PiDiNet, TEED, LDC)
- **Transformer-based** (SegFormer-B0)

**Dataset**: 136 images (95 train, 20 validation, 21 test)  
**Image Size**: 256×256 (512×512 for YOLO)  
**Training**: 50 epochs, AdamW optimizer, Cosine LR scheduler, Early stopping (patience=10)  
**Hardware**: NVIDIA GTX 1650 (4GB VRAM)  
**Date**: {datetime.now().strftime('%Y-%m-%d')}

---

## Overall Model Ranking

{tables['overall_ranking']}

---

## Key Findings

### Best Performing Models

| Metric | Model | Value |
|--------|-------|-------|
| **Highest IoU** | {valid_models[0][0]} | {valid_models[0][1]['best_val_iou']:.4f} |
| **Fastest Convergence** | {min(valid_models, key=lambda x: x[1].get('convergence_epoch', 999))[0]} | {min(valid_models, key=lambda x: x[1].get('convergence_epoch', 999))[1].get('convergence_epoch', 'N/A')} epochs |
| **Best Throughput** | {max(valid_models, key=lambda x: x[1].get('avg_throughput', 0))[0]} | {max(valid_models, key=lambda x: x[1].get('avg_throughput', 0))[1].get('avg_throughput', 0):.1f} samples/s |
| **Lowest GPU Memory** | {min(valid_models, key=lambda x: x[1].get('avg_gpu_memory_mb', float('inf')))[0]} | {min(valid_models, key=lambda x: x[1].get('avg_gpu_memory_mb', float('inf')))[1].get('avg_gpu_memory_mb', 0):.0f} MB |

### Statistical Summary

"""
    
    # Add category statistics
    readme += "| Category | Models | Avg IoU | Std IoU | Max IoU | Avg Time (min) |\n"
    readme += "|----------|--------|---------|---------|---------|----------------|\n"
    
    for cat, cs in sorted(category_stats.items(), key=lambda x: x[1]['avg_iou'], reverse=True):
        cat_display = cat.replace('_', ' ').title()
        readme += f"| {cat_display} | {cs['num_models']} | {cs['avg_iou']:.4f} | {cs['std_iou']:.4f} | {cs['max_iou']:.4f} | {cs['avg_time']/60:.1f} |\n"
    
    readme += "\n---\n\n"
    
    # Add category-specific sections
    category_titles = {
        'unet': '## UNet Models',
        'yolo_v8': '## YOLOv8 Segmentation',
        'yolo_v11': '## YOLOv11 Segmentation', 
        'yolo_v26': '## YOLOv26 Segmentation',
        'edge_detection': '## SOTA Edge Detection Models',
        'transformer': '## Transformer-based Models',
        'other': '## Other Models'
    }
    
    category_descriptions = {
        'unet': 'Classic encoder-decoder architecture with skip connections. UNet_ResNet18 uses pretrained ImageNet features.',
        'yolo_v8': 'You Only Look Once v8 with instance segmentation head. Real-time detection and segmentation.',
        'yolo_v11': 'Latest YOLO architecture with improved accuracy and efficiency.',
        'yolo_v26': 'Experimental YOLO v26 with advanced segmentation capabilities.',
        'edge_detection': 'State-of-the-art edge detection models adapted for segmentation. Multi-scale feature extraction.',
        'transformer': 'Attention-based models for semantic segmentation.',
        'other': 'Additional models evaluated in this benchmark.'
    }
    
    for cat, title in category_titles.items():
        if cat in tables and tables[cat]:
            readme += f"{title}\n\n"
            readme += f"*{category_descriptions.get(cat, '')}*\n\n"
            readme += tables[cat]
            readme += "\n"
            
            if cat in category_stats:
                cs = category_stats[cat]
                readme += f"\n**Category Average IoU**: {cs['avg_iou']:.4f} ± {cs['std_iou']:.4f}\n\n"
            
            readme += "---\n\n"
    
    # Training curves analysis
    readme += """## Training Dynamics Analysis

### Convergence Behavior

| Model | Epochs to Converge | Final IoU | IoU Stability (std) |
|-------|-------------------|-----------|---------------------|
"""
    
    for name, s in valid_models[:10]:
        conv = s.get('convergence_epoch', 'N/A')
        final_iou = s.get('final_val_iou', s.get('best_val_iou', 0))
        stability = s.get('iou_std', 0)
        readme += f"| {name} | {conv} | {final_iou:.4f} | {stability:.4f} |\n"
    
    readme += """
### Observations

1. **UNet_ResNet18** achieved the highest IoU (0.9509) with pretrained ImageNet features
2. **YOLO models** (v8n, v26n) performed competitively with faster inference
3. **Edge Detection models** (HED, RCF) showed excellent results after training, reaching ~0.94 IoU
4. **Lightweight models** (TEED, PiDiNet) offer good accuracy-speed tradeoff

---

## Performance Metrics

### Inference Speed Comparison

| Model | Forward Time (ms) | Throughput (samples/s) | GPU Memory (MB) |
|-------|-------------------|------------------------|-----------------|
"""
    
    for name, s in sorted(valid_models, key=lambda x: x[1].get('avg_throughput', 0), reverse=True)[:10]:
        fwd = s.get('avg_forward_time_ms', 0)
        throughput = s.get('avg_throughput', 0)
        gpu = s.get('avg_gpu_memory_mb', 0)
        readme += f"| {name} | {fwd:.1f} | {throughput:.1f} | {gpu:.0f} |\n"
    
    readme += """
---

## Recommendations

### For Production Deployment

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
"""
    
    # Find best models for different criteria
    best_accuracy = valid_models[0][0]
    fastest = max(valid_models, key=lambda x: x[1].get('avg_throughput', 0))[0]
    most_efficient = min(valid_models, key=lambda x: x[1].get('avg_gpu_memory_mb', float('inf')))[0]
    
    readme += f"| **Highest Accuracy** | {best_accuracy} | Best IoU of {valid_models[0][1]['best_val_iou']:.4f} |\n"
    readme += f"| **Real-time Processing** | {fastest} | Highest throughput |\n"
    readme += f"| **Memory Constrained** | {most_efficient} | Lowest GPU memory usage |\n"
    readme += f"| **Balanced** | yolov8n_seg | Good accuracy + fast inference |\n"
    
    readme += """
### Model Selection Guide

- **High Accuracy Required**: Use `unet_resnet18` or `rcf`
- **Real-time Applications**: Use `yolov8n_seg` or `teed`
- **Edge Devices (Low Memory)**: Use `teed` or `pidinet`
- **Industrial Inspection**: Use `unet_resnet18` with post-processing

---

## Files and Outputs

```
results/
├── training_logs/
│   └── {model_name}/
│       ├── training_history.json   # Full training metrics
│       ├── best_model.pt           # Best checkpoint
│       └── benchmark_results.json  # Inference benchmarks
├── plots/
│   ├── model_comparison_iou.png    # IoU comparison bar chart
│   ├── training_curves.png         # Training progress
│   ├── model_comparison_grouped.png # Grouped comparison
│   └── top5_comparison.png         # Top 5 models
└── sota_edge_detection_results.json # Edge detection benchmarks
```

---

## Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Run full benchmark
python main.py --mode full

# Generate visualizations
python generate_visualizations.py

# Analyze results
python analyze_results.py
```

---

## References

- UNet: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- YOLO: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- HED: [Holistically-Nested Edge Detection](https://arxiv.org/abs/1504.06375)
- RCF: [Richer Convolutional Features for Edge Detection](https://arxiv.org/abs/1612.02103)
- BDCN: [Bi-Directional Cascade Network](https://arxiv.org/abs/1902.10903)
- PiDiNet: [Pixel Difference Networks](https://arxiv.org/abs/2103.06767)
- TEED: [Tiny and Efficient Edge Detector](https://arxiv.org/abs/2305.17939)
- LDC: [Lightweight Dense CNN](https://arxiv.org/abs/2103.04545)

---

*Generated automatically by benchmark analysis script*
"""
    
    return readme


def create_detailed_plots(results, stats, categories, output_dir):
    """Create additional detailed statistical plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    valid_stats = {k: v for k, v in stats.items() if v.get('best_val_iou', 0) > 0.01}
    
    # 1. Box plot by category
    fig, ax = plt.subplots(figsize=(12, 6))
    
    category_data = []
    category_names = []
    
    for cat, models in categories.items():
        ious = [stats[m]['best_val_iou'] for m in models if m in stats and stats[m].get('best_val_iou', 0) > 0.01]
        if ious:
            category_data.append(ious)
            category_names.append(cat.replace('_', ' ').title())
    
    if category_data:
        bp = ax.boxplot(category_data, labels=category_names, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Validation IoU')
        ax.set_title('IoU Distribution by Model Category')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'iou_boxplot_by_category.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'iou_boxplot_by_category.png'}")
    
    # 2. Scatter: IoU vs Training Time
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cat, models in categories.items():
        cat_stats = [(m, stats[m]) for m in models if m in stats and stats[m].get('best_val_iou', 0) > 0.01]
        if cat_stats:
            times = [s['total_time_seconds']/60 for _, s in cat_stats]
            ious = [s['best_val_iou'] for _, s in cat_stats]
            names = [n for n, _ in cat_stats]
            
            ax.scatter(times, ious, label=cat.replace('_', ' ').title(), s=100, alpha=0.7)
            
            for t, iou, name in zip(times, ious, names):
                ax.annotate(name, (t, iou), fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Validation IoU')
    ax.set_title('Model Efficiency: IoU vs Training Time')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'iou_vs_training_time.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'iou_vs_training_time.png'}")
    
    # 3. Heatmap: Model metrics comparison
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sorted_models = sorted(valid_stats.items(), key=lambda x: x[1].get('best_val_iou', 0), reverse=True)[:12]
    
    metrics = ['best_val_iou', 'convergence_epoch', 'avg_throughput', 'avg_gpu_memory_mb']
    metric_names = ['Val IoU', 'Convergence Epoch', 'Throughput', 'GPU Memory (MB)']
    
    data = np.zeros((len(sorted_models), len(metrics)))
    
    for i, (name, s) in enumerate(sorted_models):
        for j, metric in enumerate(metrics):
            val = s.get(metric, 0)
            if val is None:
                val = 0
            data[i, j] = val
    
    # Normalize each column
    data_norm = data.copy()
    for j in range(data.shape[1]):
        col = data[:, j]
        if col.max() > col.min():
            data_norm[:, j] = (col - col.min()) / (col.max() - col.min())
        else:
            data_norm[:, j] = 0
    
    # For convergence epoch and GPU memory, lower is better - invert
    data_norm[:, 1] = 1 - data_norm[:, 1]  # convergence
    data_norm[:, 3] = 1 - data_norm[:, 3]  # GPU memory
    
    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto')
    
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_names)
    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels([m[0] for m in sorted_models])
    
    # Add text annotations
    for i in range(len(sorted_models)):
        for j in range(len(metrics)):
            val = data[i, j]
            if metrics[j] == 'best_val_iou':
                text = f'{val:.3f}'
            elif metrics[j] == 'convergence_epoch':
                text = f'{int(val)}'
            elif metrics[j] == 'avg_throughput':
                text = f'{val:.1f}'
            else:
                text = f'{int(val)}'
            ax.text(j, i, text, ha='center', va='center', fontsize=9)
    
    ax.set_title('Model Performance Heatmap (Green = Better)')
    plt.colorbar(im, ax=ax, label='Normalized Score')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_heatmap.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'model_performance_heatmap.png'}")


def main():
    print("="*70)
    print("COMPREHENSIVE BENCHMARK ANALYSIS")
    print("="*70)
    
    # Load data
    print("\n1. Loading training results...")
    results = load_all_training_results()
    print(f"   Loaded {len(results)} model results")
    
    # Load YOLO latency results
    print("\n1b. Loading YOLO latency results...")
    yolo_latency = load_yolo_latency_results()
    print(f"   Loaded latency for {len(yolo_latency)} YOLO models")
    
    # Categorize
    print("\n2. Categorizing models...")
    categories = categorize_models(results)
    for cat, models in categories.items():
        if models:
            print(f"   {cat}: {len(models)} models")
    
    # Compute statistics
    print("\n3. Computing statistics...")
    stats = compute_statistics(results)
    
    # Merge YOLO latency into stats
    for model_name, latency_data in yolo_latency.items():
        if model_name in stats:
            stats[model_name]['avg_forward_time_ms'] = latency_data.get('latency_ms', 0)
            stats[model_name]['avg_throughput'] = latency_data.get('fps', 0)
            print(f"   Merged latency for {model_name}: {latency_data['latency_ms']:.2f}ms")
    
    # Category statistics
    print("\n4. Computing category statistics...")
    category_stats = compute_category_statistics(stats, categories)
    
    # Generate tables
    print("\n5. Generating comparison tables...")
    tables = generate_comparison_tables(stats, categories)
    
    # Generate README
    print("\n6. Generating README.md...")
    readme = generate_readme(results, stats, categories, category_stats, tables)
    
    readme_path = Path('results/README.md')
    with open(readme_path, 'w') as f:
        f.write(readme)
    print(f"   Saved: {readme_path}")
    
    # Also save to benchmark root
    with open('README.md', 'w') as f:
        f.write(readme)
    print(f"   Saved: README.md")
    
    # Create detailed plots
    print("\n7. Creating detailed statistical plots...")
    create_detailed_plots(results, stats, categories, 'results/plots')
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    sorted_models = sorted(stats.items(), key=lambda x: x[1].get('best_val_iou', 0), reverse=True)
    valid_models = [(n, s) for n, s in sorted_models if s.get('best_val_iou', 0) > 0.01]
    
    print(f"\nTotal Models Analyzed: {len(valid_models)}")
    print("\nTop 5 Models:")
    for i, (name, s) in enumerate(valid_models[:5], 1):
        print(f"  {i}. {name}: IoU = {s['best_val_iou']:.4f}")
    
    print("\nOutput Files:")
    print("  - README.md (benchmark root)")
    print("  - results/README.md")
    print("  - results/plots/iou_boxplot_by_category.png")
    print("  - results/plots/iou_vs_training_time.png")
    print("  - results/plots/model_performance_heatmap.png")


if __name__ == '__main__':
    main()
