"""
Training visualization utilities.
Creates comprehensive plots for comparing training runs across models.
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from src.config import config
from src.training.trainer import TrainingHistory


class TrainingVisualizer:
    """Visualizer for training metrics comparison."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.benchmark.plots_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Style settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = plt.cm.tab10.colors
        self.figsize = (12, 6)
    
    def plot_loss_curves(
        self,
        histories: Dict[str, TrainingHistory],
        save_name: str = "loss_curves.png"
    ) -> plt.Figure:
        """
        Plot training and validation loss curves for all models.
        
        Args:
            histories: Dict of model_name -> TrainingHistory
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, (model_name, history) in enumerate(histories.items()):
            color = self.colors[i % len(self.colors)]
            epochs = [e.epoch for e in history.train_epochs]
            train_losses = [e.loss_mean for e in history.train_epochs]
            val_losses = [e.loss_mean for e in history.val_epochs]
            
            axes[0].plot(epochs, train_losses, '-', color=color, 
                        label=model_name, linewidth=2)
            axes[1].plot(epochs, val_losses, '-', color=color,
                        label=model_name, linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss')
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=150)
        
        return fig
    
    def plot_accuracy_curves(
        self,
        histories: Dict[str, TrainingHistory],
        metric: str = 'iou',
        save_name: str = "accuracy_curves.png"
    ) -> plt.Figure:
        """
        Plot accuracy metric curves for all models.
        
        Args:
            histories: Dict of model_name -> TrainingHistory
            metric: Metric to plot ('iou', 'dice', 'f1', 'accuracy')
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        metric_attr = f'{metric}_mean'
        
        for i, (model_name, history) in enumerate(histories.items()):
            color = self.colors[i % len(self.colors)]
            epochs = [e.epoch for e in history.train_epochs]
            train_vals = [getattr(e, metric_attr) for e in history.train_epochs]
            val_vals = [getattr(e, metric_attr) for e in history.val_epochs]
            
            axes[0].plot(epochs, train_vals, '-', color=color,
                        label=model_name, linewidth=2)
            axes[1].plot(epochs, val_vals, '-', color=color,
                        label=model_name, linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel(metric.upper())
        axes[0].set_title(f'Training {metric.upper()}')
        axes[0].legend(loc='lower right', fontsize=8)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].set_ylim(0, 1)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric.upper())
        axes[1].set_title(f'Validation {metric.upper()}')
        axes[1].legend(loc='lower right', fontsize=8)
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=150)
        
        return fig
    
    def plot_gpu_usage(
        self,
        histories: Dict[str, TrainingHistory],
        save_name: str = "gpu_usage.png"
    ) -> plt.Figure:
        """
        Plot GPU memory and utilization over epochs.
        
        Args:
            histories: Dict of model_name -> TrainingHistory
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, (model_name, history) in enumerate(histories.items()):
            color = self.colors[i % len(self.colors)]
            epochs = [e.epoch for e in history.train_epochs]
            gpu_mem = [e.gpu_memory_mb_max for e in history.train_epochs]
            gpu_util = [e.gpu_utilization_mean for e in history.train_epochs]
            
            axes[0].plot(epochs, gpu_mem, '-o', color=color,
                        label=model_name, linewidth=2, markersize=3)
            axes[1].plot(epochs, gpu_util, '-o', color=color,
                        label=model_name, linewidth=2, markersize=3)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('GPU Memory (MB)')
        axes[0].set_title('Peak GPU Memory Usage')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('GPU Utilization (%)')
        axes[1].set_title('Average GPU Utilization')
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].set_ylim(0, 100)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=150)
        
        return fig
    
    def plot_cpu_usage(
        self,
        histories: Dict[str, TrainingHistory],
        save_name: str = "cpu_usage.png"
    ) -> plt.Figure:
        """
        Plot CPU and RAM usage over epochs.
        
        Args:
            histories: Dict of model_name -> TrainingHistory
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, (model_name, history) in enumerate(histories.items()):
            color = self.colors[i % len(self.colors)]
            epochs = [e.epoch for e in history.train_epochs]
            cpu_percent = [e.cpu_percent_mean for e in history.train_epochs]
            ram_mb = [e.ram_used_mb_mean for e in history.train_epochs]
            
            axes[0].plot(epochs, cpu_percent, '-o', color=color,
                        label=model_name, linewidth=2, markersize=3)
            axes[1].plot(epochs, ram_mb, '-o', color=color,
                        label=model_name, linewidth=2, markersize=3)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('CPU Usage (%)')
        axes[0].set_title('Average CPU Usage')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].set_ylim(0, 100)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('RAM (MB)')
        axes[1].set_title('Average RAM Usage')
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=150)
        
        return fig
    
    def plot_latency_breakdown(
        self,
        histories: Dict[str, TrainingHistory],
        save_name: str = "latency_breakdown.png"
    ) -> plt.Figure:
        """
        Plot latency breakdown (forward, backward, data loading).
        
        Args:
            histories: Dict of model_name -> TrainingHistory
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Line plot over epochs
        for i, (model_name, history) in enumerate(histories.items()):
            color = self.colors[i % len(self.colors)]
            epochs = [e.epoch for e in history.train_epochs]
            forward = [e.avg_forward_time_ms for e in history.train_epochs]
            backward = [e.avg_backward_time_ms for e in history.train_epochs]
            
            axes[0].plot(epochs, forward, '-', color=color,
                        label=f'{model_name} (fwd)', linewidth=2)
            axes[0].plot(epochs, backward, '--', color=color,
                        label=f'{model_name} (bwd)', linewidth=2, alpha=0.7)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Time (ms)')
        axes[0].set_title('Forward/Backward Pass Latency')
        axes[0].legend(loc='upper right', fontsize=7)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Bar chart of average latencies
        models = list(histories.keys())
        x = np.arange(len(models))
        width = 0.25
        
        avg_forward = []
        avg_backward = []
        avg_data = []
        
        for model_name in models:
            history = histories[model_name]
            avg_forward.append(np.mean([e.avg_forward_time_ms for e in history.train_epochs]))
            avg_backward.append(np.mean([e.avg_backward_time_ms for e in history.train_epochs]))
            avg_data.append(np.mean([e.avg_data_load_time_ms for e in history.train_epochs]))
        
        axes[1].bar(x - width, avg_forward, width, label='Forward', color='#3498db')
        axes[1].bar(x, avg_backward, width, label='Backward', color='#e74c3c')
        axes[1].bar(x + width, avg_data, width, label='Data Load', color='#2ecc71')
        
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Time (ms)')
        axes[1].set_title('Average Latency Breakdown')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=150)
        
        return fig
    
    def plot_throughput(
        self,
        histories: Dict[str, TrainingHistory],
        save_name: str = "throughput.png"
    ) -> plt.Figure:
        """
        Plot training throughput (samples/sec).
        
        Args:
            histories: Dict of model_name -> TrainingHistory
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, (model_name, history) in enumerate(histories.items()):
            color = self.colors[i % len(self.colors)]
            epochs = [e.epoch for e in history.train_epochs]
            throughput = [e.throughput_samples_per_sec for e in history.train_epochs]
            
            axes[0].plot(epochs, throughput, '-o', color=color,
                        label=model_name, linewidth=2, markersize=3)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Samples/sec')
        axes[0].set_title('Training Throughput')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Bar chart of average throughput
        models = list(histories.keys())
        avg_throughput = []
        for model_name in models:
            history = histories[model_name]
            avg_throughput.append(
                np.mean([e.throughput_samples_per_sec for e in history.train_epochs])
            )
        
        colors = [self.colors[i % len(self.colors)] for i in range(len(models))]
        bars = axes[1].bar(models, avg_throughput, color=colors)
        
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Samples/sec')
        axes[1].set_title('Average Training Throughput')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, avg_throughput):
            axes[1].annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=150)
        
        return fig
    
    def plot_comprehensive_comparison(
        self,
        histories: Dict[str, TrainingHistory],
        save_name: str = "comprehensive_comparison.png"
    ) -> plt.Figure:
        """
        Create a comprehensive comparison dashboard.
        
        Args:
            histories: Dict of model_name -> TrainingHistory
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Train/Val Loss
        ax1 = fig.add_subplot(gs[0, 0])
        for i, (model_name, history) in enumerate(histories.items()):
            epochs = [e.epoch for e in history.train_epochs]
            val_loss = [e.loss_mean for e in history.val_epochs]
            ax1.plot(epochs, val_loss, color=self.colors[i % len(self.colors)], label=model_name)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Val Loss')
        ax1.set_title('Validation Loss')
        ax1.legend(fontsize=7)
        
        # 2. Val IoU
        ax2 = fig.add_subplot(gs[0, 1])
        for i, (model_name, history) in enumerate(histories.items()):
            epochs = [e.epoch for e in history.val_epochs]
            val_iou = [e.iou_mean for e in history.val_epochs]
            ax2.plot(epochs, val_iou, color=self.colors[i % len(self.colors)], label=model_name)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.set_title('Validation IoU')
        ax2.legend(fontsize=7)
        ax2.set_ylim(0, 1)
        
        # 3. Val Dice
        ax3 = fig.add_subplot(gs[0, 2])
        for i, (model_name, history) in enumerate(histories.items()):
            epochs = [e.epoch for e in history.val_epochs]
            val_dice = [e.dice_mean for e in history.val_epochs]
            ax3.plot(epochs, val_dice, color=self.colors[i % len(self.colors)], label=model_name)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Dice')
        ax3.set_title('Validation Dice')
        ax3.legend(fontsize=7)
        ax3.set_ylim(0, 1)
        
        # 4. GPU Memory
        ax4 = fig.add_subplot(gs[1, 0])
        for i, (model_name, history) in enumerate(histories.items()):
            epochs = [e.epoch for e in history.train_epochs]
            gpu_mem = [e.gpu_memory_mb_max for e in history.train_epochs]
            ax4.plot(epochs, gpu_mem, color=self.colors[i % len(self.colors)], label=model_name)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('GPU Memory (MB)')
        ax4.set_title('Peak GPU Memory')
        ax4.legend(fontsize=7)
        
        # 5. GPU Utilization
        ax5 = fig.add_subplot(gs[1, 1])
        for i, (model_name, history) in enumerate(histories.items()):
            epochs = [e.epoch for e in history.train_epochs]
            gpu_util = [e.gpu_utilization_mean for e in history.train_epochs]
            ax5.plot(epochs, gpu_util, color=self.colors[i % len(self.colors)], label=model_name)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('GPU Util (%)')
        ax5.set_title('GPU Utilization')
        ax5.legend(fontsize=7)
        ax5.set_ylim(0, 100)
        
        # 6. CPU Usage
        ax6 = fig.add_subplot(gs[1, 2])
        for i, (model_name, history) in enumerate(histories.items()):
            epochs = [e.epoch for e in history.train_epochs]
            cpu = [e.cpu_percent_mean for e in history.train_epochs]
            ax6.plot(epochs, cpu, color=self.colors[i % len(self.colors)], label=model_name)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('CPU (%)')
        ax6.set_title('CPU Usage')
        ax6.legend(fontsize=7)
        ax6.set_ylim(0, 100)
        
        # 7. Throughput
        ax7 = fig.add_subplot(gs[2, 0])
        for i, (model_name, history) in enumerate(histories.items()):
            epochs = [e.epoch for e in history.train_epochs]
            throughput = [e.throughput_samples_per_sec for e in history.train_epochs]
            ax7.plot(epochs, throughput, color=self.colors[i % len(self.colors)], label=model_name)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Samples/sec')
        ax7.set_title('Training Throughput')
        ax7.legend(fontsize=7)
        
        # 8. Forward Latency
        ax8 = fig.add_subplot(gs[2, 1])
        for i, (model_name, history) in enumerate(histories.items()):
            epochs = [e.epoch for e in history.train_epochs]
            lat = [e.avg_forward_time_ms for e in history.train_epochs]
            ax8.plot(epochs, lat, color=self.colors[i % len(self.colors)], label=model_name)
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Time (ms)')
        ax8.set_title('Forward Pass Latency')
        ax8.legend(fontsize=7)
        
        # 9. Summary bar chart - Best IoU
        ax9 = fig.add_subplot(gs[2, 2])
        models = list(histories.keys())
        best_ious = [h.best_val_iou for h in histories.values()]
        colors = [self.colors[i % len(self.colors)] for i in range(len(models))]
        bars = ax9.bar(models, best_ious, color=colors)
        ax9.set_ylabel('Best Val IoU')
        ax9.set_title('Best Validation IoU')
        ax9.tick_params(axis='x', rotation=45)
        ax9.set_ylim(0, 1)
        
        for bar, val in zip(bars, best_ious):
            ax9.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=8)
        
        plt.suptitle('Comprehensive Training Comparison', fontsize=14, y=1.02)
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), 
                       dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_final_comparison_table(
        self,
        histories: Dict[str, TrainingHistory],
        save_name: str = "final_comparison_table.png"
    ) -> plt.Figure:
        """
        Create a visual table comparing final metrics.
        
        Args:
            histories: Dict of model_name -> TrainingHistory
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, len(histories) * 0.8 + 2))
        ax.axis('off')
        
        # Prepare data
        columns = ['Model', 'Best Epoch', 'Val Loss', 'Val IoU', 'Val Dice', 
                  'GPU Peak (MB)', 'GPU Util (%)', 'CPU (%)', 'Throughput (smp/s)']
        
        data = []
        for model_name, history in histories.items():
            best_epoch_idx = history.best_epoch - 1
            if best_epoch_idx < len(history.val_epochs):
                val_epoch = history.val_epochs[best_epoch_idx]
                train_epoch = history.train_epochs[best_epoch_idx]
                
                row = [
                    model_name,
                    str(history.best_epoch),
                    f"{history.best_val_loss:.4f}",
                    f"{history.best_val_iou:.4f}",
                    f"{val_epoch.dice_mean:.4f}",
                    f"{history.peak_gpu_memory_mb:.0f}",
                    f"{history.avg_gpu_utilization:.1f}",
                    f"{train_epoch.cpu_percent_mean:.1f}",
                    f"{train_epoch.throughput_samples_per_sec:.1f}"
                ]
            else:
                row = [model_name] + ['N/A'] * 8
            data.append(row)
        
        # Create table
        table = ax.table(
            cellText=data,
            colLabels=columns,
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # Alternate row colors
        for i in range(1, len(data) + 1):
            color = '#E2EFDA' if i % 2 == 0 else 'white'
            for j in range(len(columns)):
                table[(i, j)].set_facecolor(color)
        
        plt.title('Training Results Comparison', fontsize=14, pad=20)
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), 
                       dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_all_training_plots(
        self,
        histories: Dict[str, TrainingHistory]
    ):
        """Generate all training visualization plots."""
        print("Generating training visualizations...")
        
        self.plot_loss_curves(histories)
        print("  Created: loss_curves.png")
        
        self.plot_accuracy_curves(histories, metric='iou', save_name='iou_curves.png')
        print("  Created: iou_curves.png")
        
        self.plot_accuracy_curves(histories, metric='dice', save_name='dice_curves.png')
        print("  Created: dice_curves.png")
        
        self.plot_gpu_usage(histories)
        print("  Created: gpu_usage.png")
        
        self.plot_cpu_usage(histories)
        print("  Created: cpu_usage.png")
        
        self.plot_latency_breakdown(histories)
        print("  Created: latency_breakdown.png")
        
        self.plot_throughput(histories)
        print("  Created: throughput.png")
        
        self.plot_comprehensive_comparison(histories)
        print("  Created: comprehensive_comparison.png")
        
        self.plot_final_comparison_table(histories)
        print("  Created: final_comparison_table.png")
        
        print(f"\nAll plots saved to: {self.output_dir}")


def load_training_history(logs_dir: str) -> TrainingHistory:
    """
    Load training history from saved JSON files.
    
    Args:
        logs_dir: Directory containing training logs
        
    Returns:
        TrainingHistory object
    """
    history_path = os.path.join(logs_dir, 'training_history.json')
    
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Training history not found: {history_path}")
    
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    # Create TrainingHistory from loaded data
    history = TrainingHistory(
        model_name=data['model_name'],
        started_at=data['started_at'],
        ended_at=data.get('ended_at'),
        total_epochs=data['total_epochs'],
        best_val_loss=data['best_val_loss'],
        best_val_accuracy=data['best_val_accuracy'],
        best_val_iou=data['best_val_iou'],
        best_epoch=data['best_epoch'],
        total_train_time_sec=data['total_train_time_sec'],
        total_val_time_sec=data['total_val_time_sec'],
        peak_gpu_memory_mb=data['peak_gpu_memory_mb'],
        peak_cpu_percent=data['peak_cpu_percent'],
        peak_ram_mb=data['peak_ram_mb'],
        avg_gpu_utilization=data['avg_gpu_utilization']
    )
    
    # Note: Epoch-level metrics would need separate loading from epoch files
    # This is a simplified version for comparison
    
    return history
