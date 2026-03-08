"""
Visualization utilities for benchmark results.
Includes mask overlays, contour plots, measurement annotations, and comparison charts.
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from src.config import config
from src.evaluation.metrics import (
    SegmentationMetrics, ContourMetrics, MeasurementMetrics,
    LatencyMetrics, HardwareMetrics, BenchmarkResult
)


# Color scheme
COLORS = {
    'gt': (0, 255, 0),       # Green for ground truth
    'pred': (255, 0, 0),     # Red for prediction
    'overlap': (255, 255, 0), # Yellow for overlap
    'contour': (0, 255, 255), # Cyan for contours
    'measurement': (255, 165, 0)  # Orange for measurements
}


class VisualizationUtils:
    """Utilities for visualizing segmentation and measurement results."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.benchmark.visual_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def overlay_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay binary mask on image.
        
        Args:
            image: RGB image
            mask: Binary mask
            color: Overlay color
            alpha: Transparency
            
        Returns:
            Image with mask overlay
        """
        overlay = image.copy()
        
        # Ensure mask is binary
        binary_mask = (mask > 127) if mask.max() > 1 else (mask > 0)
        
        # Apply color to mask region
        overlay[binary_mask] = overlay[binary_mask] * (1 - alpha) + np.array(color) * alpha
        
        return overlay.astype(np.uint8)
    
    def compare_masks(
        self,
        image: np.ndarray,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> np.ndarray:
        """
        Create comparison visualization of predicted vs ground truth masks.
        
        Args:
            image: RGB image
            pred_mask: Predicted mask
            gt_mask: Ground truth mask
            
        Returns:
            Comparison visualization
        """
        # Normalize masks
        pred = (pred_mask > 127) if pred_mask.max() > 1 else (pred_mask > 0)
        gt = (gt_mask > 127) if gt_mask.max() > 1 else (gt_mask > 0)
        
        overlay = image.copy().astype(np.float32)
        
        # True positive (overlap) - Yellow
        tp = pred & gt
        overlay[tp] = overlay[tp] * 0.5 + np.array(COLORS['overlap']) * 0.5
        
        # False positive (pred only) - Red
        fp = pred & ~gt
        overlay[fp] = overlay[fp] * 0.5 + np.array(COLORS['pred']) * 0.5
        
        # False negative (gt only) - Green
        fn = ~pred & gt
        overlay[fn] = overlay[fn] * 0.5 + np.array(COLORS['gt']) * 0.5
        
        return overlay.astype(np.uint8)
    
    def draw_contours(
        self,
        image: np.ndarray,
        contours: List[np.ndarray],
        color: Tuple[int, int, int] = None,
        thickness: int = 2,
        filled: bool = False
    ) -> np.ndarray:
        """
        Draw contours on image.
        
        Args:
            image: RGB image
            contours: List of contours
            color: Contour color
            thickness: Line thickness
            filled: Whether to fill contours
            
        Returns:
            Image with contours
        """
        if color is None:
            color = COLORS['contour']
        
        result = image.copy()
        
        if filled:
            cv2.drawContours(result, contours, -1, color, -1)
        else:
            cv2.drawContours(result, contours, -1, color, thickness)
        
        return result
    
    def annotate_measurement(
        self,
        image: np.ndarray,
        pt1: Tuple[float, float],
        pt2: Tuple[float, float],
        value: float,
        unit: str = "px",
        color: Tuple[int, int, int] = None
    ) -> np.ndarray:
        """
        Annotate measurement on image.
        
        Args:
            image: RGB image
            pt1: Start point
            pt2: End point
            value: Measurement value
            unit: Unit label
            color: Annotation color
            
        Returns:
            Image with measurement annotation
        """
        if color is None:
            color = COLORS['measurement']
        
        result = image.copy()
        
        # Draw line
        cv2.line(result, 
                (int(pt1[0]), int(pt1[1])), 
                (int(pt2[0]), int(pt2[1])), 
                color, 2)
        
        # Draw endpoints
        cv2.circle(result, (int(pt1[0]), int(pt1[1])), 5, color, -1)
        cv2.circle(result, (int(pt2[0]), int(pt2[1])), 5, color, -1)
        
        # Add text
        mid_pt = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
        text = f"{value:.2f} {unit}"
        cv2.putText(result, text, 
                   (int(mid_pt[0]) + 10, int(mid_pt[1]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
    
    def create_summary_grid(
        self,
        images: List[np.ndarray],
        titles: List[str],
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Create a grid of images for comparison.
        
        Args:
            images: List of images
            titles: List of titles
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n = len(images)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).flatten()
        
        for i, (img, title) in enumerate(zip(images, titles)):
            ax = axes[i]
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(n, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_visualization(
        self,
        image: np.ndarray,
        filename: str,
        subdir: str = None
    ):
        """Save visualization image."""
        if subdir:
            save_dir = os.path.join(self.output_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.output_dir
        
        filepath = os.path.join(save_dir, filename)
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(filepath, image)


class BenchmarkPlotter:
    """Create benchmark comparison plots."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.benchmark.plots_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_accuracy_comparison(
        self,
        results: Dict[str, SegmentationMetrics],
        title: str = "Segmentation Accuracy Comparison",
        save_name: str = "accuracy_comparison.png"
    ) -> plt.Figure:
        """
        Bar chart comparing accuracy metrics across models.
        
        Args:
            results: Dict mapping model name to SegmentationMetrics
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        models = list(results.keys())
        metrics = ['iou', 'dice', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(models))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, metric in enumerate(metrics):
            values = [getattr(results[m], metric) for m in models]
            ax.bar(x + i * width, values, width, label=metric.upper())
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=150)
        
        return fig
    
    def plot_latency_comparison(
        self,
        results: Dict[str, LatencyMetrics],
        title: str = "Latency Comparison",
        save_name: str = "latency_comparison.png"
    ) -> plt.Figure:
        """
        Stacked bar chart showing latency breakdown.
        
        Args:
            results: Dict mapping model name to LatencyMetrics
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        models = list(results.keys())
        
        preproc = [results[m].preprocessing_time_ms for m in models]
        inference = [results[m].inference_time_ms for m in models]
        postproc = [results[m].postprocessing_time_ms for m in models]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        
        ax.bar(x, preproc, label='Preprocessing', color='#3498db')
        ax.bar(x, inference, bottom=preproc, label='Inference', color='#e74c3c')
        ax.bar(x, postproc, bottom=np.array(preproc) + np.array(inference), 
               label='Postprocessing', color='#2ecc71')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Time (ms)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        
        # Add FPS annotation
        for i, m in enumerate(models):
            total = preproc[i] + inference[i] + postproc[i]
            ax.annotate(f'{results[m].fps:.1f} FPS',
                       xy=(i, total), xytext=(0, 5),
                       textcoords='offset points',
                       ha='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=150)
        
        return fig
    
    def plot_accuracy_vs_latency(
        self,
        accuracy_results: Dict[str, SegmentationMetrics],
        latency_results: Dict[str, LatencyMetrics],
        title: str = "Accuracy vs Latency Trade-off",
        save_name: str = "accuracy_vs_latency.png"
    ) -> plt.Figure:
        """
        Scatter plot showing accuracy vs latency trade-off.
        
        Args:
            accuracy_results: Dict mapping model name to SegmentationMetrics
            latency_results: Dict mapping model name to LatencyMetrics
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        models = list(accuracy_results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            if model in latency_results:
                iou = accuracy_results[model].iou
                latency = latency_results[model].total_time_ms
                
                ax.scatter(latency, iou, s=100, c=[colors[i]], label=model, alpha=0.8)
                ax.annotate(model, (latency, iou), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('IoU Score')
        ax.set_title(title)
        ax.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=150)
        
        return fig
    
    def plot_accuracy_vs_gpu_memory(
        self,
        accuracy_results: Dict[str, SegmentationMetrics],
        hardware_results: Dict[str, HardwareMetrics],
        title: str = "Accuracy vs GPU Memory Trade-off",
        save_name: str = "accuracy_vs_gpu_memory.png"
    ) -> plt.Figure:
        """
        Scatter plot showing accuracy vs GPU memory trade-off.
        
        Args:
            accuracy_results: Dict mapping model name to SegmentationMetrics
            hardware_results: Dict mapping model name to HardwareMetrics
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        models = list(accuracy_results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            if model in hardware_results:
                iou = accuracy_results[model].iou
                mem = hardware_results[model].gpu_memory_peak_mb
                
                ax.scatter(mem, iou, s=100, c=[colors[i]], label=model, alpha=0.8)
                ax.annotate(model, (mem, iou), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Peak GPU Memory (MB)')
        ax.set_ylabel('IoU Score')
        ax.set_title(title)
        ax.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=150)
        
        return fig
    
    def plot_latency_vs_gpu_utilization(
        self,
        latency_results: Dict[str, LatencyMetrics],
        hardware_results: Dict[str, HardwareMetrics],
        title: str = "Latency vs GPU Utilization",
        save_name: str = "latency_vs_gpu_util.png"
    ) -> plt.Figure:
        """
        Scatter plot showing latency vs GPU utilization.
        
        Args:
            latency_results: Dict mapping model name to LatencyMetrics
            hardware_results: Dict mapping model name to HardwareMetrics
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        models = list(latency_results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            if model in hardware_results:
                latency = latency_results[model].total_time_ms
                util = hardware_results[model].gpu_utilization
                
                ax.scatter(latency, util, s=100, c=[colors[i]], label=model, alpha=0.8)
                ax.annotate(model, (latency, util), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=150)
        
        return fig
    
    def plot_model_ranking(
        self,
        results: List[BenchmarkResult],
        title: str = "Model Ranking",
        save_name: str = "model_ranking.png"
    ) -> plt.Figure:
        """
        Create radar chart for model comparison.
        
        Args:
            results: List of BenchmarkResult objects
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        metrics = ['IoU', 'Dice', 'FPS', 'Memory Eff.']
        num_metrics = len(metrics)
        
        # Prepare data
        model_data = {}
        for result in results:
            if result.segmentation and result.latency:
                model_data[result.model_name] = [
                    result.segmentation.iou,
                    result.segmentation.dice,
                    min(result.latency.fps / 100, 1.0),  # Normalize FPS
                    1 - min((result.hardware.gpu_memory_peak_mb if result.hardware else 1000) / 4000, 1.0)  # Memory efficiency
                ]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))
        
        for i, (model, values) in enumerate(model_data.items()):
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=150,
                       bbox_inches='tight')
        
        return fig


def generate_all_visualizations(
    results: Dict[str, BenchmarkResult],
    images: Dict[str, np.ndarray],
    pred_masks: Dict[str, np.ndarray],
    gt_masks: Dict[str, np.ndarray]
):
    """
    Generate all visualization outputs.
    
    Args:
        results: Dict of model name to BenchmarkResult
        images: Dict of sample_id to image
        pred_masks: Dict of (model, sample) to predicted mask
        gt_masks: Dict of sample_id to ground truth mask
    """
    viz = VisualizationUtils()
    plotter = BenchmarkPlotter()
    
    # Extract metrics by model
    seg_metrics = {}
    latency_metrics = {}
    hardware_metrics = {}
    
    for model_name, result in results.items():
        if result.segmentation:
            seg_metrics[model_name] = result.segmentation
        if result.latency:
            latency_metrics[model_name] = result.latency
        if result.hardware:
            hardware_metrics[model_name] = result.hardware
    
    # Generate plots
    if seg_metrics:
        plotter.plot_accuracy_comparison(seg_metrics)
    
    if latency_metrics:
        plotter.plot_latency_comparison(latency_metrics)
    
    if seg_metrics and latency_metrics:
        plotter.plot_accuracy_vs_latency(seg_metrics, latency_metrics)
    
    if seg_metrics and hardware_metrics:
        plotter.plot_accuracy_vs_gpu_memory(seg_metrics, hardware_metrics)
    
    print("Visualizations saved to:", plotter.output_dir)
