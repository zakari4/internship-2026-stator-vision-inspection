"""
Comprehensive training module with maximum data collection.
Collects GPU usage, CPU usage, training/val loss, accuracy, latency per epoch/batch.
"""

import os
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field, asdict
from functools import partial
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import psutil

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    PYNVML_AVAILABLE = False

from src.config import config


# =============================================================================
# Data Classes for Comprehensive Metrics Collection
# =============================================================================

@dataclass
class BatchMetrics:
    """
    Detailed metrics collected for a single training or validation batch.
    
    Captures exact timings, resource utilization, and loss/accuracy scores to
    enable bottleneck detection and hardware efficiency analysis.

    :param batch_idx: Sequential index of the batch within the epoch.
    :param timestamp: Unix timestamp when the batch was completed.
    :param loss: Calculated loss value for the batch.
    :param forward_time_ms: Duration of the model forward pass.
    :param backward_time_ms: Duration of backpropagation (train only).
    :param optimizer_time_ms: Duration of weight update (train only).
    :param data_load_time_ms: Time spent waiting for data from the DataLoader.
    :param total_time_ms: Sum of all timings for this batch.
    :param cpu_percent: Instantaneous CPU usage at batch end.
    :param ram_used_mb: RSS memory usage in megabytes.
    :param gpu_memory_mb: VRAM allocated by the model/tensors (MB).
    :param gpu_utilization: GPU core activity percentage.
    """
    batch_idx: int
    timestamp: float
    loss: float
    
    # Timing
    forward_time_ms: float
    backward_time_ms: float
    optimizer_time_ms: float
    data_load_time_ms: float
    total_time_ms: float
    
    # Hardware at batch end
    cpu_percent: float
    ram_used_mb: float
    gpu_memory_mb: float
    gpu_utilization: float
    
    # Optional accuracy metrics
    accuracy: Optional[float] = None
    iou: Optional[float] = None
    dice: Optional[float] = None


@dataclass
class EpochMetrics:
    """
    Aggregated performance data for a full training or validation epoch.
    
    Summarizes batch-level data into statistical distributions (mean, std, 
    max) to provide a high-level view of model convergence and throughput.

    :param epoch: Epoch number (1-indexed).
    :param phase: Training stage ('train' or 'val').
    :param duration_sec: Total wall-clock time for the epoch.
    :param loss_mean: Average loss across all batches.
    :param accuracy_mean: Mean pixel accuracy.
    :param iou_mean: Mean Intersection over Union.
    :param throughput_samples_per_sec: Processing speed (samples/second).
    :param learning_rate: Optimizer learning rate at the end of the epoch.
    :param batch_metrics: Optional list of raw metrics for every batch.
    """
    epoch: int
    phase: str  # 'train' or 'val'
    timestamp_start: float
    timestamp_end: float
    duration_sec: float
    
    # Loss
    loss_mean: float
    loss_std: float
    loss_min: float
    loss_max: float
    
    # Accuracy metrics
    accuracy_mean: float
    iou_mean: float
    dice_mean: float
    precision_mean: float
    recall_mean: float
    f1_mean: float
    
    # Latency
    avg_batch_time_ms: float
    avg_forward_time_ms: float
    avg_backward_time_ms: float
    avg_data_load_time_ms: float
    throughput_samples_per_sec: float
    
    # Hardware - aggregated
    cpu_percent_mean: float
    cpu_percent_max: float
    ram_used_mb_mean: float
    ram_used_mb_max: float
    gpu_memory_mb_mean: float
    gpu_memory_mb_max: float
    gpu_utilization_mean: float
    gpu_utilization_max: float
    
    # Learning rate
    learning_rate: float
    
    # Batch-level data
    batch_metrics: List[BatchMetrics] = field(default_factory=list)


@dataclass
class TrainingHistory:
    """Complete training history with all collected data."""
    model_name: str
    started_at: str
    ended_at: Optional[str] = None
    total_epochs: int = 0
    best_val_loss: float = float('inf')
    best_val_accuracy: float = 0.0
    best_val_iou: float = 0.0
    best_epoch: int = 0
    
    # Per-epoch data
    train_epochs: List[EpochMetrics] = field(default_factory=list)
    val_epochs: List[EpochMetrics] = field(default_factory=list)
    
    # Summary statistics
    total_train_time_sec: float = 0.0
    total_val_time_sec: float = 0.0
    
    # Hardware summary
    peak_gpu_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_ram_mb: float = 0.0
    avg_gpu_utilization: float = 0.0


# =============================================================================
# Hardware Monitor (Background Thread)
# =============================================================================

class HardwareMonitor:
    """
    Background-threaded monitor for continuous resource tracking.
    
    Samples CPU, RAM, and GPU state at high frequency independently of the 
    training loop to capture transient spikes and background system load.
    """
    
    def __init__(self, sample_interval: float = 0.1):
        """
        Initialize hardware monitor.
        
        Args:
            sample_interval: Sampling interval in seconds
        """
        self.sample_interval = sample_interval
        self.running = False
        self.thread = None
        
        # Collected samples
        self.cpu_samples: List[float] = []
        self.ram_samples: List[float] = []
        self.gpu_util_samples: List[float] = []
        self.gpu_mem_samples: List[float] = []
        self.timestamps: List[float] = []
        
        # GPU handle
        self.gpu_handle = None
        if PYNVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                pass
    
    def start(self):
        """Start monitoring."""
        self.running = True
        self.clear()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return statistics."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        return self.get_stats()
    
    def clear(self):
        """Clear collected samples."""
        self.cpu_samples = []
        self.ram_samples = []
        self.gpu_util_samples = []
        self.gpu_mem_samples = []
        self.timestamps = []
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                timestamp = time.time()
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_samples.append(cpu_percent)
                
                # RAM usage
                ram = psutil.virtual_memory()
                self.ram_samples.append(ram.used / 1024 / 1024)
                
                # GPU metrics
                if self.gpu_handle and PYNVML_AVAILABLE:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        self.gpu_util_samples.append(util.gpu)
                        self.gpu_mem_samples.append(mem_info.used / 1024 / 1024)
                    except:
                        self.gpu_util_samples.append(0.0)
                        self.gpu_mem_samples.append(0.0)
                elif torch.cuda.is_available():
                    self.gpu_mem_samples.append(
                        torch.cuda.memory_allocated() / 1024 / 1024
                    )
                    self.gpu_util_samples.append(0.0)
                else:
                    self.gpu_util_samples.append(0.0)
                    self.gpu_mem_samples.append(0.0)
                
                self.timestamps.append(timestamp)
                
                time.sleep(self.sample_interval)
            except Exception as e:
                pass
    
    def get_current(self) -> Dict[str, float]:
        """Get current hardware stats."""
        cpu_percent = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        ram_mb = ram.used / 1024 / 1024
        
        gpu_util = 0.0
        gpu_mem = 0.0
        
        if self.gpu_handle and PYNVML_AVAILABLE:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_util = util.gpu
                gpu_mem = mem_info.used / 1024 / 1024
            except:
                pass
        elif torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
        
        return {
            'cpu_percent': cpu_percent,
            'ram_used_mb': ram_mb,
            'gpu_utilization': gpu_util,
            'gpu_memory_mb': gpu_mem
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        def safe_stat(samples):
            if not samples:
                return {'mean': 0, 'max': 0, 'min': 0, 'std': 0}
            return {
                'mean': float(np.mean(samples)),
                'max': float(np.max(samples)),
                'min': float(np.min(samples)),
                'std': float(np.std(samples))
            }
        
        return {
            'cpu': safe_stat(self.cpu_samples),
            'ram': safe_stat(self.ram_samples),
            'gpu_util': safe_stat(self.gpu_util_samples),
            'gpu_mem': safe_stat(self.gpu_mem_samples),
            'num_samples': len(self.timestamps),
            'duration_sec': (self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0
        }


# =============================================================================
# Loss Functions
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice coefficient loss for binary segmentation.
    
    Optimizes the overlap between predicted and ground truth masks, which is
    more robust to class imbalance than standard Cross Entropy.

    :param smooth: Laplacian smoothing factor to prevent division by zero.
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate 1 - Dice coefficient."""
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class BoundaryLoss(nn.Module):
    """
    Boundary loss based on distance transform maps.
    
    Penalizes pixels that are predicted with high confidence but are far from
    the actual object boundary. Highly effective for fine-tuning edge details.
    Reference: Kervadec et al., "Boundary loss for highly unbalanced segmentation", 2019.
    """
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def _compute_distance_map(mask: torch.Tensor) -> torch.Tensor:
        """Compute signed distance transform for a batch of masks."""
        from scipy.ndimage import distance_transform_edt
        
        dist_maps = []
        mask_np = mask.detach().cpu().numpy()
        
        for i in range(mask_np.shape[0]):
            for j in range(mask_np.shape[1]):
                m = mask_np[i, j]
                # Signed distance: negative inside, positive outside
                pos_dist = distance_transform_edt(1 - m)
                neg_dist = distance_transform_edt(m)
                dist_map = pos_dist - neg_dist
                dist_maps.append(dist_map)
        
        dist_arr = np.array(dist_maps).reshape(mask_np.shape)
        # Normalize to [-1, 1] to prevent FP16 overflow under AMP
        max_abs = np.abs(dist_arr).max()
        if max_abs > 0:
            dist_arr = dist_arr / max_abs
        
        dist_tensor = torch.tensor(
            dist_arr,
            dtype=torch.float32,
            device=mask.device
        )
        return dist_tensor
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        dist_map = self._compute_distance_map(target)
        # Boundary loss = mean of (predicted probability * distance map)
        loss = (pred_prob * dist_map).mean()
        return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovász-Hinge loss for binary segmentation.
    
    A tractable surrogate for the Jaccard index (IoU). It directly optimizes
    the IoU using the Lovász extension of submodular set functions.
    Reference: Berman et al., "The Lovász-Softmax loss", 2018.
    """
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of the Lovász extension w.r.t sorted errors."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / (union + 1e-7)  # prevent 0/0 -> NaN
        if p > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard
    
    def _lovasz_hinge_flat(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Binary Lovász hinge loss (flat tensors)."""
        if len(labels) == 0:
            return logits.sum() * 0.0
        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(torch.nn.functional.relu(errors_sorted), grad)
        return loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Average over batch
        losses = []
        for i in range(pred.shape[0]):
            loss = self._lovasz_hinge_flat(
                pred[i].view(-1),
                target[i].view(-1)
            )
            losses.append(loss)
        return torch.stack(losses).mean()


class CombinedLoss(nn.Module):
    """
    Combined loss with 4 components: BCE + Dice + Boundary + Lovász.
    Default weights: BCE=0.3, Dice=0.3, Boundary=0.2, Lovász=0.2.
    Use mode='bce_dice' for legacy 2-component loss.
    """
    
    def __init__(
        self,
        bce_weight: float = 0.3,
        dice_weight: float = 0.3,
        boundary_weight: float = 0.2,
        lovasz_weight: float = 0.2,
        mode: str = 'combined'
    ):
        super().__init__()
        self.mode = mode
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
        if mode == 'bce_dice':
            # Legacy mode: only BCE + Dice
            total = bce_weight + dice_weight
            self.bce_weight = bce_weight / total
            self.dice_weight = dice_weight / total
            self.boundary_weight = 0.0
            self.lovasz_weight = 0.0
        else:
            self.bce_weight = bce_weight
            self.dice_weight = dice_weight
            self.boundary_weight = boundary_weight
            self.lovasz_weight = lovasz_weight
            self.boundary_loss = BoundaryLoss()
            self.lovasz_loss = LovaszHingeLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)
        
        if self.mode != 'bce_dice':
            # Run boundary loss in float32 to avoid FP16 overflow
            with torch.cuda.amp.autocast(enabled=False):
                pred_f32 = pred.float()
                target_f32 = target.float()
                boundary = self.boundary_loss(pred_f32, target_f32)
                lovasz = self.lovasz_loss(pred_f32, target_f32)
            loss = loss + self.boundary_weight * boundary
            loss = loss + self.lovasz_weight * lovasz
        
        # Clamp to prevent extreme values that could cause NaN gradients
        loss = torch.clamp(loss, min=0.0, max=100.0)
        
        return loss


# =============================================================================
# Trainer with Comprehensive Data Collection
# =============================================================================

class ComprehensiveTrainer:
    """
    State-of-the-art training orchestration engine.
    
    Handles the entire training lifecycle including data loading, mixed 
    precision (AMP), resource monitoring, advanced loss computation, 
    periodic checkpointing, and comprehensive metrics logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str,
        device: str = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        optimizer_name: str = 'adamw',
        scheduler_name: str = 'cosine',
        output_dir: str = None,
        save_every_epoch: bool = True,
        hardware_sample_interval: float = 0.1,
        use_amp: bool = True,
        loss_mode: str = 'bce_dice'
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name for logging
            device: Compute device
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            num_epochs: Maximum epochs
            early_stopping_patience: Stop if no improvement
            optimizer_name: Optimizer choice ('adamw', 'adam', 'sgd', 'rmsprop')
            scheduler_name: LR scheduler choice ('cosine', 'step', 'plateau')
            output_dir: Output directory for checkpoints and logs
            save_every_epoch: Whether to save metrics every epoch
            hardware_sample_interval: Hardware sampling interval in seconds
            use_amp: Whether to use automatic mixed precision (FP16)
            loss_mode: Loss function mode ('combined' or 'bce_dice')
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.device = device or config.benchmark.device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.optimizer_name = optimizer_name.lower()
        self.scheduler_name = scheduler_name.lower()
        self.save_every_epoch = save_every_epoch
        
        # Output directory
        self.output_dir = output_dir or config.benchmark.output_dir
        self.checkpoints_dir = os.path.join(self.output_dir, 'checkpoints', model_name)
        self.logs_dir = os.path.join(self.output_dir, 'training_logs', model_name)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Move model to device
        if hasattr(self.model, 'model') and self.model.model is not None:
            # Wrapper model (e.g., DeepLabV3MobileNet) - use inner model
            self.model.model = self.model.model.to(self.device)
        elif hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.loss_mode = loss_mode
        self.criterion = CombinedLoss(mode=loss_mode)
        
        # Mixed precision (AMP)
        self.use_amp = use_amp and (self.device == 'cuda')
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print(f"  [AMP] Mixed precision training enabled (FP16)")
        
        # Hardware monitor
        self.hardware_monitor = HardwareMonitor(sample_interval=hardware_sample_interval)
        
        # Training history
        self.history = TrainingHistory(
            model_name=model_name,
            started_at=datetime.now().isoformat(),
            total_epochs=num_epochs
        )
        
        # Early stopping tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def _get_nn_model(self):
        """Get the underlying nn.Module from wrapper models."""
        if isinstance(self.model, nn.Module):
            return self.model
        elif hasattr(self.model, 'model') and isinstance(self.model.model, nn.Module):
            return self.model.model
        return self.model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """
        Create optimizer based on optimizer_name.
        
        Supports: adamw, adam, sgd, rmsprop
        """
        if hasattr(self.model, 'parameters') and callable(self.model.parameters):
            params = self.model.parameters()
        elif hasattr(self.model, 'model') and self.model.model is not None:
            params = self.model.model.parameters()
        else:
            raise AttributeError(f"Cannot get parameters from model: {type(self.model)}")
        
        if self.optimizer_name == 'adamw':
            return optim.AdamW(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'adam':
            return optim.Adam(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'sgd':
            return optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True
            )
        elif self.optimizer_name == 'rmsprop':
            return optim.RMSprop(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {self.optimizer_name}. "
                f"Supported: adamw, adam, sgd, rmsprop"
            )
    
    def _create_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        """
        Create LR scheduler based on scheduler_name.
        
        Supports: cosine, step, plateau
        """
        if self.scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=self.learning_rate * 0.01
            )
        elif self.scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(1, self.num_epochs // 3),
                gamma=0.1
            )
        elif self.scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.learning_rate * 0.001
            )
        else:
            raise ValueError(
                f"Unknown scheduler: {self.scheduler_name}. "
                f"Supported: cosine, step, plateau"
            )
    
    def compute_batch_accuracy(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Compute accuracy metrics for a batch."""
        with torch.no_grad():
            # Sigmoid + threshold
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            target_binary = target.float()
            
            # Flatten
            pred_flat = pred_binary.view(-1)
            target_flat = target_binary.view(-1)
            
            # Compute metrics
            eps = 1e-7
            tp = (pred_flat * target_flat).sum()
            fp = (pred_flat * (1 - target_flat)).sum()
            fn = ((1 - pred_flat) * target_flat).sum()
            tn = ((1 - pred_flat) * (1 - target_flat)).sum()
            
            # Accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
            
            # IoU
            intersection = tp
            union = tp + fp + fn
            iou = intersection / (union + eps)
            
            # Dice
            dice = 2 * tp / (2 * tp + fp + fn + eps)
            
            # Precision, Recall
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            
            return {
                'accuracy': float(accuracy.cpu()),
                'iou': float(iou.cpu()),
                'dice': float(dice.cpu()),
                'precision': float(precision.cpu()),
                'recall': float(recall.cpu()),
                'f1': float(f1.cpu())
            }
    
    def train_epoch(self, epoch: int) -> EpochMetrics:
        """Train for one epoch with comprehensive data collection."""
        self._get_nn_model().train()
        timestamp_start = time.time()
        
        batch_metrics_list = []
        losses = []
        accuracies = []
        ious = []
        dices = []
        precisions = []
        recalls = []
        f1s = []
        
        forward_times = []
        backward_times = []
        optimizer_times = []
        data_load_times = []
        batch_times = []
        
        cpu_percents = []
        ram_usages = []
        gpu_mems = []
        gpu_utils = []
        
        data_start = time.time()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            data_load_time = (time.time() - data_start) * 1000
            data_load_times.append(data_load_time)
            
            batch_start = time.time()
            
            # Extract from dict batch and convert numpy to tensor
            images = batch['image'] if isinstance(batch, dict) else batch[0]
            masks = batch['mask'] if isinstance(batch, dict) else batch[1]
            
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks).float()
            
            # Move to device
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            # Ensure masks have correct shape
            if masks.dim() == 3:
                masks = masks.unsqueeze(1).float()
            else:
                masks = masks.float()
            
            # Forward pass timing
            if self.device == 'cuda':
                torch.cuda.synchronize()
            forward_start = time.time()
            
            # AMP: autocast forward pass + loss computation
            with autocast(enabled=self.use_amp):
                outputs = self._get_nn_model()(images)
                
                # Handle models that return OrderedDict (e.g., DeepLabV3)
                if isinstance(outputs, dict):
                    outputs = outputs['out'] if 'out' in outputs else list(outputs.values())[0]
                
                # Handle different output shapes
                if outputs.shape != masks.shape:
                    outputs = nn.functional.interpolate(
                        outputs, size=masks.shape[2:], mode='bilinear', align_corners=False
                    )
            
            # Compute loss in FP32 to prevent NaN from FP16 overflow
            outputs_f32 = outputs.float()
            masks_f32 = masks.float()
            loss = self.criterion(outputs_f32, masks_f32)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            forward_time = (time.time() - forward_start) * 1000
            forward_times.append(forward_time)
            
            # Backward pass timing
            backward_start = time.time()
            
            self.optimizer.zero_grad()
            # AMP: scale loss and backward
            self.scaler.scale(loss).backward()
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            backward_time = (time.time() - backward_start) * 1000
            backward_times.append(backward_time)
            
            # Optimizer step timing
            optimizer_start = time.time()
            
            # AMP: unscale, clip gradients, step, update
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self._get_nn_model().parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            optimizer_time = (time.time() - optimizer_start) * 1000
            optimizer_times.append(optimizer_time)
            
            # Total batch time
            batch_time = (time.time() - batch_start) * 1000
            batch_times.append(batch_time)
            
            # Compute accuracy metrics
            acc_metrics = self.compute_batch_accuracy(outputs, masks)
            
            # Store metrics
            losses.append(loss.item())
            accuracies.append(acc_metrics['accuracy'])
            ious.append(acc_metrics['iou'])
            dices.append(acc_metrics['dice'])
            precisions.append(acc_metrics['precision'])
            recalls.append(acc_metrics['recall'])
            f1s.append(acc_metrics['f1'])
            
            # Hardware metrics
            hw = self.hardware_monitor.get_current()
            cpu_percents.append(hw['cpu_percent'])
            ram_usages.append(hw['ram_used_mb'])
            gpu_mems.append(hw['gpu_memory_mb'])
            gpu_utils.append(hw['gpu_utilization'])
            
            # Create batch metrics
            batch_metric = BatchMetrics(
                batch_idx=batch_idx,
                timestamp=time.time(),
                loss=loss.item(),
                forward_time_ms=forward_time,
                backward_time_ms=backward_time,
                optimizer_time_ms=optimizer_time,
                data_load_time_ms=data_load_time,
                total_time_ms=batch_time,
                cpu_percent=hw['cpu_percent'],
                ram_used_mb=hw['ram_used_mb'],
                gpu_memory_mb=hw['gpu_memory_mb'],
                gpu_utilization=hw['gpu_utilization'],
                accuracy=acc_metrics['accuracy'],
                iou=acc_metrics['iou'],
                dice=acc_metrics['dice']
            )
            batch_metrics_list.append(batch_metric)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{acc_metrics['iou']:.4f}",
                'gpu': f"{hw['gpu_memory_mb']:.0f}MB"
            })
            
            data_start = time.time()
        
        timestamp_end = time.time()
        duration = timestamp_end - timestamp_start
        
        # Calculate throughput
        num_samples = len(self.train_loader.dataset)
        throughput = num_samples / duration if duration > 0 else 0
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Create epoch metrics
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            phase='train',
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            duration_sec=duration,
            loss_mean=float(np.mean(losses)),
            loss_std=float(np.std(losses)),
            loss_min=float(np.min(losses)),
            loss_max=float(np.max(losses)),
            accuracy_mean=float(np.mean(accuracies)),
            iou_mean=float(np.mean(ious)),
            dice_mean=float(np.mean(dices)),
            precision_mean=float(np.mean(precisions)),
            recall_mean=float(np.mean(recalls)),
            f1_mean=float(np.mean(f1s)),
            avg_batch_time_ms=float(np.mean(batch_times)),
            avg_forward_time_ms=float(np.mean(forward_times)),
            avg_backward_time_ms=float(np.mean(backward_times)),
            avg_data_load_time_ms=float(np.mean(data_load_times)),
            throughput_samples_per_sec=throughput,
            cpu_percent_mean=float(np.mean(cpu_percents)),
            cpu_percent_max=float(np.max(cpu_percents)),
            ram_used_mb_mean=float(np.mean(ram_usages)),
            ram_used_mb_max=float(np.max(ram_usages)),
            gpu_memory_mb_mean=float(np.mean(gpu_mems)),
            gpu_memory_mb_max=float(np.max(gpu_mems)),
            gpu_utilization_mean=float(np.mean(gpu_utils)),
            gpu_utilization_max=float(np.max(gpu_utils)),
            learning_rate=current_lr,
            batch_metrics=batch_metrics_list
        )
        
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> EpochMetrics:
        """Validate for one epoch with comprehensive data collection."""
        self._get_nn_model().eval()
        timestamp_start = time.time()
        
        batch_metrics_list = []
        losses = []
        accuracies = []
        ious = []
        dices = []
        precisions = []
        recalls = []
        f1s = []
        
        forward_times = []
        data_load_times = []
        batch_times = []
        
        cpu_percents = []
        ram_usages = []
        gpu_mems = []
        gpu_utils = []
        
        data_start = time.time()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                data_load_time = (time.time() - data_start) * 1000
                data_load_times.append(data_load_time)
                
                batch_start = time.time()
                
                # Extract from dict batch and convert numpy to tensor
                images = batch['image'] if isinstance(batch, dict) else batch[0]
                masks = batch['mask'] if isinstance(batch, dict) else batch[1]
                
                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
                if isinstance(masks, np.ndarray):
                    masks = torch.from_numpy(masks).float()
                
                # Move to device
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1).float()
                else:
                    masks = masks.float()
                
                # Forward pass timing
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                forward_start = time.time()
                
                outputs = self._get_nn_model()(images)
                
                # Handle models that return OrderedDict (e.g., DeepLabV3)
                if isinstance(outputs, dict):
                    outputs = outputs['out'] if 'out' in outputs else list(outputs.values())[0]
                
                if outputs.shape != masks.shape:
                    outputs = nn.functional.interpolate(
                        outputs, size=masks.shape[2:], mode='bilinear', align_corners=False
                    )
                
                loss = self.criterion(outputs.float(), masks.float())
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                forward_time = (time.time() - forward_start) * 1000
                forward_times.append(forward_time)
                
                batch_time = (time.time() - batch_start) * 1000
                batch_times.append(batch_time)
                
                # Compute accuracy metrics
                acc_metrics = self.compute_batch_accuracy(outputs, masks)
                
                # Store metrics
                losses.append(loss.item())
                accuracies.append(acc_metrics['accuracy'])
                ious.append(acc_metrics['iou'])
                dices.append(acc_metrics['dice'])
                precisions.append(acc_metrics['precision'])
                recalls.append(acc_metrics['recall'])
                f1s.append(acc_metrics['f1'])
                
                # Hardware metrics
                hw = self.hardware_monitor.get_current()
                cpu_percents.append(hw['cpu_percent'])
                ram_usages.append(hw['ram_used_mb'])
                gpu_mems.append(hw['gpu_memory_mb'])
                gpu_utils.append(hw['gpu_utilization'])
                
                # Create batch metrics
                batch_metric = BatchMetrics(
                    batch_idx=batch_idx,
                    timestamp=time.time(),
                    loss=loss.item(),
                    forward_time_ms=forward_time,
                    backward_time_ms=0.0,  # No backward in validation
                    optimizer_time_ms=0.0,
                    data_load_time_ms=data_load_time,
                    total_time_ms=batch_time,
                    cpu_percent=hw['cpu_percent'],
                    ram_used_mb=hw['ram_used_mb'],
                    gpu_memory_mb=hw['gpu_memory_mb'],
                    gpu_utilization=hw['gpu_utilization'],
                    accuracy=acc_metrics['accuracy'],
                    iou=acc_metrics['iou'],
                    dice=acc_metrics['dice']
                )
                batch_metrics_list.append(batch_metric)
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'iou': f"{acc_metrics['iou']:.4f}"
                })
                
                data_start = time.time()
        
        timestamp_end = time.time()
        duration = timestamp_end - timestamp_start
        
        num_samples = len(self.val_loader.dataset)
        throughput = num_samples / duration if duration > 0 else 0
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            phase='val',
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            duration_sec=duration,
            loss_mean=float(np.mean(losses)),
            loss_std=float(np.std(losses)),
            loss_min=float(np.min(losses)),
            loss_max=float(np.max(losses)),
            accuracy_mean=float(np.mean(accuracies)),
            iou_mean=float(np.mean(ious)),
            dice_mean=float(np.mean(dices)),
            precision_mean=float(np.mean(precisions)),
            recall_mean=float(np.mean(recalls)),
            f1_mean=float(np.mean(f1s)),
            avg_batch_time_ms=float(np.mean(batch_times)),
            avg_forward_time_ms=float(np.mean(forward_times)),
            avg_backward_time_ms=0.0,
            avg_data_load_time_ms=float(np.mean(data_load_times)),
            throughput_samples_per_sec=throughput,
            cpu_percent_mean=float(np.mean(cpu_percents)),
            cpu_percent_max=float(np.max(cpu_percents)),
            ram_used_mb_mean=float(np.mean(ram_usages)),
            ram_used_mb_max=float(np.max(ram_usages)),
            gpu_memory_mb_mean=float(np.mean(gpu_mems)),
            gpu_memory_mb_max=float(np.max(gpu_mems)),
            gpu_utilization_mean=float(np.mean(gpu_utils)),
            gpu_utilization_max=float(np.max(gpu_utils)),
            learning_rate=current_lr,
            batch_metrics=batch_metrics_list
        )
        
        return epoch_metrics
    
    def train(self) -> TrainingHistory:
        """
        Run full training loop with comprehensive data collection.
        
        Returns:
            TrainingHistory with all collected data
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Optimizer: {self.optimizer_name}")
        print(f"Scheduler: {self.scheduler_name}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Output: {self.logs_dir}")
        print()
        
        # Start hardware monitor
        self.hardware_monitor.start()
        
        training_start = time.time()
        
        try:
            for epoch in range(1, self.num_epochs + 1):
                print(f"\nEpoch {epoch}/{self.num_epochs}")
                print("-" * 40)
                
                # Training
                train_metrics = self.train_epoch(epoch)
                self.history.train_epochs.append(train_metrics)
                
                # Validation
                val_metrics = self.validate_epoch(epoch)
                self.history.val_epochs.append(val_metrics)
                
                # Learning rate step
                if self.scheduler_name == 'plateau':
                    self.scheduler.step(val_metrics.loss_mean)
                else:
                    self.scheduler.step()
                
                # Print epoch summary
                print(f"  Train - Loss: {train_metrics.loss_mean:.4f}, "
                      f"IoU: {train_metrics.iou_mean:.4f}, "
                      f"Dice: {train_metrics.dice_mean:.4f}")
                print(f"  Val   - Loss: {val_metrics.loss_mean:.4f}, "
                      f"IoU: {val_metrics.iou_mean:.4f}, "
                      f"Dice: {val_metrics.dice_mean:.4f}")
                print(f"  Hardware - GPU: {train_metrics.gpu_memory_mb_max:.0f}MB, "
                      f"GPU Util: {train_metrics.gpu_utilization_mean:.1f}%, "
                      f"CPU: {train_metrics.cpu_percent_mean:.1f}%")
                print(f"  Latency - Forward: {train_metrics.avg_forward_time_ms:.1f}ms, "
                      f"Backward: {train_metrics.avg_backward_time_ms:.1f}ms, "
                      f"Throughput: {train_metrics.throughput_samples_per_sec:.1f} samples/s")
                
                # Update best metrics
                if val_metrics.loss_mean < self.best_val_loss:
                    self.best_val_loss = val_metrics.loss_mean
                    self.history.best_val_loss = val_metrics.loss_mean
                    self.history.best_val_accuracy = val_metrics.accuracy_mean
                    self.history.best_val_iou = val_metrics.iou_mean
                    self.history.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.epochs_without_improvement += 1
                
                # Update peak hardware metrics
                self.history.peak_gpu_memory_mb = max(
                    self.history.peak_gpu_memory_mb,
                    train_metrics.gpu_memory_mb_max
                )
                self.history.peak_cpu_percent = max(
                    self.history.peak_cpu_percent,
                    train_metrics.cpu_percent_max
                )
                self.history.peak_ram_mb = max(
                    self.history.peak_ram_mb,
                    train_metrics.ram_used_mb_max
                )
                
                # Save epoch metrics
                if self.save_every_epoch:
                    self._save_epoch_metrics(epoch, train_metrics, val_metrics)
                
                # Early stopping check
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
        
        finally:
            # Stop hardware monitor
            hw_stats = self.hardware_monitor.stop()
            
            # Finalize history
            training_end = time.time()
            self.history.ended_at = datetime.now().isoformat()
            self.history.total_train_time_sec = sum(
                e.duration_sec for e in self.history.train_epochs
            )
            self.history.total_val_time_sec = sum(
                e.duration_sec for e in self.history.val_epochs
            )
            self.history.avg_gpu_utilization = hw_stats['gpu_util']['mean']
            
            # Save final history
            self._save_training_history()
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Best epoch: {self.history.best_epoch}")
        print(f"Best val loss: {self.history.best_val_loss:.4f}")
        print(f"Best val IoU: {self.history.best_val_iou:.4f}")
        print(f"Total time: {training_end - training_start:.1f}s")
        print(f"Peak GPU memory: {self.history.peak_gpu_memory_mb:.0f}MB")
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self._get_nn_model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        if is_best:
            path = os.path.join(self.checkpoints_dir, 'best_model.pth')
        else:
            path = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
    
    def _save_epoch_metrics(
        self,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics
    ):
        """Save epoch metrics to JSON."""
        # Convert to dict without batch_metrics for summary
        train_summary = {k: v for k, v in asdict(train_metrics).items() 
                        if k != 'batch_metrics'}
        val_summary = {k: v for k, v in asdict(val_metrics).items() 
                      if k != 'batch_metrics'}
        
        epoch_data = {
            'epoch': epoch,
            'train': train_summary,
            'val': val_summary
        }
        
        path = os.path.join(self.logs_dir, f'epoch_{epoch:03d}.json')
        with open(path, 'w') as f:
            json.dump(epoch_data, f, indent=2)
        
        # Also save batch-level data (more detailed)
        batch_data = {
            'epoch': epoch,
            'train_batches': [asdict(b) for b in train_metrics.batch_metrics],
            'val_batches': [asdict(b) for b in val_metrics.batch_metrics]
        }
        
        batch_path = os.path.join(self.logs_dir, f'epoch_{epoch:03d}_batches.json')
        with open(batch_path, 'w') as f:
            json.dump(batch_data, f, indent=2)
    
    def _save_training_history(self):
        """Save complete training history."""
        # Create summary without batch-level data
        history_summary = {
            'model_name': self.history.model_name,
            'started_at': self.history.started_at,
            'ended_at': self.history.ended_at,
            'total_epochs': len(self.history.train_epochs),
            'best_val_loss': self.history.best_val_loss,
            'best_val_accuracy': self.history.best_val_accuracy,
            'best_val_iou': self.history.best_val_iou,
            'best_epoch': self.history.best_epoch,
            'total_train_time_sec': self.history.total_train_time_sec,
            'total_val_time_sec': self.history.total_val_time_sec,
            'peak_gpu_memory_mb': self.history.peak_gpu_memory_mb,
            'peak_cpu_percent': self.history.peak_cpu_percent,
            'peak_ram_mb': self.history.peak_ram_mb,
            'avg_gpu_utilization': self.history.avg_gpu_utilization,
            'epochs': []
        }
        
        for train_epoch, val_epoch in zip(self.history.train_epochs, self.history.val_epochs):
            epoch_summary = {
                'epoch': train_epoch.epoch,
                'train': {
                    'loss': train_epoch.loss_mean,
                    'accuracy': train_epoch.accuracy_mean,
                    'iou': train_epoch.iou_mean,
                    'dice': train_epoch.dice_mean,
                    'precision': train_epoch.precision_mean,
                    'recall': train_epoch.recall_mean,
                    'f1': train_epoch.f1_mean,
                    'duration_sec': train_epoch.duration_sec,
                    'throughput': train_epoch.throughput_samples_per_sec,
                    'avg_forward_ms': train_epoch.avg_forward_time_ms,
                    'avg_backward_ms': train_epoch.avg_backward_time_ms,
                    'gpu_mem_max_mb': train_epoch.gpu_memory_mb_max,
                    'gpu_util_mean': train_epoch.gpu_utilization_mean,
                    'cpu_percent_mean': train_epoch.cpu_percent_mean,
                    'learning_rate': train_epoch.learning_rate
                },
                'val': {
                    'loss': val_epoch.loss_mean,
                    'accuracy': val_epoch.accuracy_mean,
                    'iou': val_epoch.iou_mean,
                    'dice': val_epoch.dice_mean,
                    'precision': val_epoch.precision_mean,
                    'recall': val_epoch.recall_mean,
                    'f1': val_epoch.f1_mean,
                    'duration_sec': val_epoch.duration_sec,
                    'throughput': val_epoch.throughput_samples_per_sec,
                    'avg_forward_ms': val_epoch.avg_forward_time_ms,
                    'gpu_mem_max_mb': val_epoch.gpu_memory_mb_max,
                    'gpu_util_mean': val_epoch.gpu_utilization_mean,
                    'cpu_percent_mean': val_epoch.cpu_percent_mean
                }
            }
            history_summary['epochs'].append(epoch_summary)
        
        path = os.path.join(self.logs_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(history_summary, f, indent=2)
        
        print(f"Training history saved to: {path}")


# =============================================================================
# Utility Functions
# =============================================================================

def train_all_models(
    train_loader: DataLoader,
    val_loader: DataLoader,
    models_dict: Dict[str, nn.Module],
    num_epochs: int = 50,
    device: str = None
) -> Dict[str, TrainingHistory]:
    """
    Benchmark multiple models by training them sequentially on the same data.
    
    Provides a high-level entry point for comparative studies, collecting
    detailed performance histories for each model in the batch.

    :param train_loader: DataLoader for the training set.
    :param val_loader: DataLoader for the validation set.
    :param models_dict: Mapping of model names to their PyTorch implementations.
    :param num_epochs: Number of training epochs per model.
    :param device: Target compute device ('cuda' or 'cpu').
    :return: Dictionary of performance histories indexed by model name.
    """
    histories = {}
    
    for model_name, model in models_dict.items():
        print(f"\n{'#'*60}")
        print(f"# Training: {model_name}")
        print(f"{'#'*60}")
        
        trainer = ComprehensiveTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name,
            device=device,
            num_epochs=num_epochs
        )
        
        history = trainer.train()
        histories[model_name] = history
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return histories


def compare_training_results(
    histories: Dict[str, TrainingHistory]
) -> Dict[str, Any]:
    """
    Compare training results across models.
    
    Args:
        histories: Dict of model_name -> TrainingHistory
        
    Returns:
        Comparison summary
    """
    comparison = {
        'models': [],
        'best_model_by_iou': None,
        'fastest_model': None,
        'most_efficient_model': None
    }
    
    best_iou = 0
    fastest_time = float('inf')
    best_efficiency = 0
    
    for model_name, history in histories.items():
        model_summary = {
            'name': model_name,
            'best_val_iou': history.best_val_iou,
            'best_val_loss': history.best_val_loss,
            'best_epoch': history.best_epoch,
            'total_train_time_sec': history.total_train_time_sec,
            'peak_gpu_memory_mb': history.peak_gpu_memory_mb,
            'avg_gpu_utilization': history.avg_gpu_utilization
        }
        comparison['models'].append(model_summary)
        
        # Track best model
        if history.best_val_iou > best_iou:
            best_iou = history.best_val_iou
            comparison['best_model_by_iou'] = model_name
        
        if history.total_train_time_sec < fastest_time:
            fastest_time = history.total_train_time_sec
            comparison['fastest_model'] = model_name
        
        # Efficiency = IoU / GPU memory
        efficiency = history.best_val_iou / (history.peak_gpu_memory_mb + 1)
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            comparison['most_efficient_model'] = model_name
    
    return comparison


# =============================================================================
# YOLO Segmentation Trainer with Hardware Monitoring
# =============================================================================

class YOLOTrainer:
    """
    Specialized trainer for YOLO segmentation models (v8, v11, v26).
    
    Wraps the Ultralytics training API and provides continuous hardware 
    monitoring in a background thread to capture resource fingerprints
    specific to YOLO architectures.
    """
    
    def __init__(
        self,
        model_variant: str,
        data_yaml: str,
        model_name: str = None,
        device: str = None,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 8,
        patience: int = 10,
        optimizer: str = 'auto',
        lr0: float = 0.01,
        lrf: float = 0.01,
        weight_decay: float = 0.0005,
        output_dir: str = None,
        hardware_sample_interval: float = 0.5
    ):
        """
        Initialize YOLO trainer.
        
        Args:
            model_variant: YOLO model variant (e.g., 'yolov8n-seg', 'yolo11n-seg')
            data_yaml: Path to dataset YAML file
            model_name: Display name for logging
            device: Compute device (0 for GPU, 'cpu' for CPU)
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            patience: Early stopping patience (0 to disable)
            optimizer: Optimizer name ('auto', 'SGD', 'Adam', 'AdamW', 'RMSProp')
            lr0: Initial learning rate
            lrf: Final learning rate factor (lr0 * lrf)
            weight_decay: Weight decay
            output_dir: Output directory
            hardware_sample_interval: Hardware sampling interval in seconds
        """
        self.model_variant = model_variant
        self.data_yaml = data_yaml
        self.model_name = model_name or model_variant
        self.device = device or (0 if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.patience = patience
        self.optimizer_name = optimizer
        self.lr0 = lr0
        self.lrf = lrf
        self.weight_decay = weight_decay
        self.output_dir = output_dir or config.benchmark.output_dir
        
        # Output directories
        self.project_dir = os.path.join(self.output_dir, 'yolo_training')
        self.logs_dir = os.path.join(self.output_dir, 'training_logs', self.model_name)
        os.makedirs(self.project_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Hardware monitor
        self.hardware_monitor = HardwareMonitor(sample_interval=hardware_sample_interval)
        
        # Load model
        try:
            from ultralytics import YOLO
            self.model = YOLO(f"{model_variant}.pt")
        except ImportError:
            raise ImportError("ultralytics package required: pip install ultralytics")
        
        # Training history
        self.history = TrainingHistory(
            model_name=self.model_name,
            started_at=datetime.now().isoformat(),
            total_epochs=epochs
        )
    
    def train(self) -> TrainingHistory:
        """
        Execute the YOLO training process with integrated resource monitoring.
        
        Orchestrates weight initialization, data preparation, and the 
        Ultralytics training loop while sampling hardware state.

        :return: TrainingHistory containing YOLO-specific performance metrics.
        """
        print(f"\n{'='*60}")
        print(f"Training YOLO Model: {self.model_name}")
        print(f"{'='*60}")
        print(f"Variant: {self.model_variant}")
        print(f"Data: {self.data_yaml}")
        print(f"Epochs: {self.epochs}")
        print(f"Image size: {self.imgsz}")
        print(f"Batch size: {self.batch}")
        print(f"Optimizer: {self.optimizer_name}")
        print(f"Early stopping patience: {self.patience}")
        print(f"Initial LR: {self.lr0}")
        print(f"Device: {self.device}")
        print()
        
        # Start hardware monitoring
        self.hardware_monitor.start()
        training_start = time.time()
        
        try:
            # Run YOLO training with early stopping and optimizer
            results = self.model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                device=self.device,
                project=self.project_dir,
                name=self.model_name,
                exist_ok=True,
                verbose=True,
                save=True,
                plots=True,
                patience=self.patience,
                optimizer=self.optimizer_name,
                lr0=self.lr0,
                lrf=self.lrf,
                weight_decay=self.weight_decay
            )
            
            training_end = time.time()
            
            # Stop hardware monitoring
            hw_stats = self.hardware_monitor.stop()
            
            # Extract training metrics from YOLO results
            self._extract_yolo_metrics(results, hw_stats, training_end - training_start)
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            self.hardware_monitor.stop()
            raise
        
        # Finalize history
        self.history.ended_at = datetime.now().isoformat()
        self._save_training_history()
        
        print(f"\n{'='*60}")
        print("YOLO Training Complete!")
        print(f"{'='*60}")
        print(f"Best results saved to: {self.project_dir}/{self.model_name}")
        
        return self.history
    
    def _extract_yolo_metrics(
        self,
        results,
        hw_stats: Dict[str, Any],
        total_time: float
    ):
        """Extract metrics from YOLO training results."""
        try:
            # Try to read results.csv from YOLO output
            results_csv = os.path.join(
                self.project_dir, self.model_name, 'results.csv'
            )
            
            if os.path.exists(results_csv):
                import csv
                with open(results_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                
                best_epoch = 0
                best_mask_map = 0
                
                for i, row in enumerate(rows):
                    epoch = i + 1
                    
                    # Extract metrics (column names may vary)
                    train_loss = float(row.get('train/seg_loss', row.get('train/box_loss', 0)))
                    val_loss = float(row.get('val/seg_loss', row.get('metrics/mAP50(M)', 0)))
                    mask_map50 = float(row.get('metrics/mAP50(M)', 0))
                    mask_map = float(row.get('metrics/mAP50-95(M)', 0))
                    
                    # Create epoch metrics
                    train_epoch = EpochMetrics(
                        epoch=epoch,
                        phase='train',
                        timestamp_start=0,
                        timestamp_end=0,
                        duration_sec=total_time / len(rows),
                        loss_mean=train_loss,
                        loss_std=0,
                        loss_min=train_loss,
                        loss_max=train_loss,
                        accuracy_mean=mask_map50,
                        iou_mean=mask_map,
                        dice_mean=mask_map,
                        precision_mean=float(row.get('metrics/precision(M)', 0)),
                        recall_mean=float(row.get('metrics/recall(M)', 0)),
                        f1_mean=0,
                        avg_batch_time_ms=0,
                        avg_forward_time_ms=0,
                        avg_backward_time_ms=0,
                        avg_data_load_time_ms=0,
                        throughput_samples_per_sec=0,
                        cpu_percent_mean=hw_stats['cpu']['mean'],
                        cpu_percent_max=hw_stats['cpu']['max'],
                        ram_used_mb_mean=hw_stats['ram']['mean'],
                        ram_used_mb_max=hw_stats['ram']['max'],
                        gpu_memory_mb_mean=hw_stats['gpu_mem']['mean'],
                        gpu_memory_mb_max=hw_stats['gpu_mem']['max'],
                        gpu_utilization_mean=hw_stats['gpu_util']['mean'],
                        gpu_utilization_max=hw_stats['gpu_util']['max'],
                        learning_rate=float(row.get('lr/pg0', 0))
                    )
                    self.history.train_epochs.append(train_epoch)
                    
                    # Val epoch (using same loss for simplicity)
                    val_epoch = EpochMetrics(
                        epoch=epoch,
                        phase='val',
                        timestamp_start=0,
                        timestamp_end=0,
                        duration_sec=0,
                        loss_mean=val_loss,
                        loss_std=0,
                        loss_min=val_loss,
                        loss_max=val_loss,
                        accuracy_mean=mask_map50,
                        iou_mean=mask_map,
                        dice_mean=mask_map,
                        precision_mean=float(row.get('metrics/precision(M)', 0)),
                        recall_mean=float(row.get('metrics/recall(M)', 0)),
                        f1_mean=0,
                        avg_batch_time_ms=0,
                        avg_forward_time_ms=0,
                        avg_backward_time_ms=0,
                        avg_data_load_time_ms=0,
                        throughput_samples_per_sec=0,
                        cpu_percent_mean=hw_stats['cpu']['mean'],
                        cpu_percent_max=hw_stats['cpu']['max'],
                        ram_used_mb_mean=hw_stats['ram']['mean'],
                        ram_used_mb_max=hw_stats['ram']['max'],
                        gpu_memory_mb_mean=hw_stats['gpu_mem']['mean'],
                        gpu_memory_mb_max=hw_stats['gpu_mem']['max'],
                        gpu_utilization_mean=hw_stats['gpu_util']['mean'],
                        gpu_utilization_max=hw_stats['gpu_util']['max'],
                        learning_rate=float(row.get('lr/pg0', 0))
                    )
                    self.history.val_epochs.append(val_epoch)
                    
                    # Track best
                    if mask_map > best_mask_map:
                        best_mask_map = mask_map
                        best_epoch = epoch
                
                self.history.best_epoch = best_epoch
                self.history.best_val_iou = best_mask_map
                self.history.best_val_accuracy = best_mask_map
        
        except Exception as e:
            print(f"Warning: Could not extract detailed metrics: {e}")
        
        # Update hardware summary
        self.history.total_train_time_sec = total_time
        self.history.peak_gpu_memory_mb = hw_stats['gpu_mem']['max']
        self.history.peak_cpu_percent = hw_stats['cpu']['max']
        self.history.peak_ram_mb = hw_stats['ram']['max']
        self.history.avg_gpu_utilization = hw_stats['gpu_util']['mean']
    
    def _save_training_history(self):
        """Save training history to JSON."""
        history_data = {
            'model_name': self.history.model_name,
            'model_variant': self.model_variant,
            'started_at': self.history.started_at,
            'ended_at': self.history.ended_at,
            'total_epochs': len(self.history.train_epochs),
            'best_epoch': self.history.best_epoch,
            'best_val_iou': self.history.best_val_iou,
            'total_train_time_sec': self.history.total_train_time_sec,
            'peak_gpu_memory_mb': self.history.peak_gpu_memory_mb,
            'peak_cpu_percent': self.history.peak_cpu_percent,
            'avg_gpu_utilization': self.history.avg_gpu_utilization,
            'config': {
                'epochs': self.epochs,
                'imgsz': self.imgsz,
                'batch': self.batch,
                'data_yaml': self.data_yaml
            }
        }
        
        path = os.path.join(self.logs_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Training history saved to: {path}")
    
    def validate(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        results = self.model.val(data=self.data_yaml, imgsz=self.imgsz)
        
        return {
            'mAP50': results.seg.map50 if hasattr(results, 'seg') else results.box.map50,
            'mAP50-95': results.seg.map if hasattr(results, 'seg') else results.box.map,
            'precision': results.seg.mp if hasattr(results, 'seg') else results.box.mp,
            'recall': results.seg.mr if hasattr(results, 'seg') else results.box.mr
        }


def train_yolo_models(
    data_yaml: str,
    model_variants: List[str] = None,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    device: str = None
) -> Dict[str, TrainingHistory]:
    """
    Train multiple YOLO segmentation models.
    
    Args:
        data_yaml: Path to dataset YAML file
        model_variants: List of model variants to train
        epochs: Training epochs
        imgsz: Image size
        batch: Batch size
        device: Compute device
        
    Returns:
        Dict of model_name -> TrainingHistory
    """
    if model_variants is None:
        model_variants = [
            'yolov8n-seg', 'yolov8s-seg',
            'yolo11n-seg', 'yolo11s-seg',
            'yolo12n-seg', 'yolo12s-seg'
        ]
    
    histories = {}
    
    for variant in model_variants:
        print(f"\n{'#'*60}")
        print(f"# Training: {variant}")
        print(f"{'#'*60}")
        
        try:
            trainer = YOLOTrainer(
                model_variant=variant,
                data_yaml=data_yaml,
                device=device,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch
            )
            
            history = trainer.train()
            histories[variant] = history
            
            # Clear GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Failed to train {variant}: {e}")
            import traceback
            traceback.print_exc()
    
    return histories
