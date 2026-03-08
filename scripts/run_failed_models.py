#!/usr/bin/env python3
"""Run benchmarks for models that failed due to missing to() method."""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import config
from src.models.deep_learning import (
    HEDModel, RCFModel, BDCNModel, PiDiNetModel, TEEDModel, LDCModel
)


def benchmark_model(model, test_dataset, model_name):
    """Benchmark a single model."""
    print(f"\nBenchmarking {model_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_ious = []
    all_dice = []
    all_latencies = []
    
    import time
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        image = sample['image']
        gt_mask = sample['mask']
        
        # Ensure image is in right format (H, W, C)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if image.shape[0] == 3:  # C, H, W -> H, W, C
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Run inference
        start = time.perf_counter()
        result = model.segment(image)
        latency = (time.perf_counter() - start) * 1000
        all_latencies.append(latency)
        
        # Get prediction mask
        pred_mask = result.mask
        
        # Ensure masks are binary
        pred_binary = (pred_mask > 127).astype(np.float32)
        gt_binary = (gt_mask > 0.5).astype(np.float32) if gt_mask.max() <= 1.0 else (gt_mask > 127).astype(np.float32)
        
        # Resize if needed
        if pred_binary.shape != gt_binary.shape:
            import cv2
            pred_binary = cv2.resize(pred_binary, (gt_binary.shape[1], gt_binary.shape[0]))
        
        # Calculate metrics
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
        iou = intersection / (union + 1e-8)
        dice = 2 * intersection / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-8)
        
        all_ious.append(iou)
        all_dice.append(dice)
    
    results = {
        'model_name': model_name,
        'iou_mean': float(np.mean(all_ious)),
        'iou_std': float(np.std(all_ious)),
        'dice_mean': float(np.mean(all_dice)),
        'dice_std': float(np.std(all_dice)),
        'latency_mean_ms': float(np.mean(all_latencies)),
        'latency_std_ms': float(np.std(all_latencies)),
        'fps': float(1000 / np.mean(all_latencies)),
        'num_samples': len(test_dataset)
    }
    
    print(f"  IoU: {results['iou_mean']:.4f} +/- {results['iou_std']:.4f}")
    print(f"  Dice: {results['dice_mean']:.4f} +/- {results['dice_std']:.4f}")
    print(f"  Latency: {results['latency_mean_ms']:.2f} ms, FPS: {results['fps']:.1f}")
    
    return results


def main():
    print("=" * 60)
    print("Running Failed SOTA Edge Detection Models")
    print("=" * 60)
    
    # Load dataset
    from src.data.dataset import create_dataloaders
    
    train_loader, val_loader, test_loader = create_dataloaders(
        root_path=str(config.dataset.root_path),
        batch_size=1
    )
    
    test_dataset = test_loader.dataset
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Models to benchmark
    failed_models = {
        'hed': HEDModel,
        'rcf': RCFModel,
        'bdcn': BDCNModel,
        'pidinet': PiDiNetModel,
        'teed': TEEDModel,
        'ldc': LDCModel
    }
    
    output_base = Path(_PROJECT_ROOT) / "outputs" / "results"
    all_results = {}
    
    for model_name, model_class in failed_models.items():
        try:
            model = model_class()
            results = benchmark_model(model, test_dataset, model_name)
            all_results[model_name] = results
            
            # Save individual result
            output_dir = output_base / 'training_logs' / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / 'benchmark_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Clear GPU memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    combined_file = output_base / 'sota_edge_detection_results.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, res in all_results.items():
        print(f"{name:12s}: IoU={res['iou_mean']:.4f}, Dice={res['dice_mean']:.4f}, FPS={res['fps']:.1f}")
    
    print(f"\nResults saved to {combined_file}")


if __name__ == '__main__':
    main()
