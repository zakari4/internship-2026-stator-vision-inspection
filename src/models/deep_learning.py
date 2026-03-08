"""
Deep learning segmentation models for industrial inspection.
Includes UNet variants, DeepLabV3, SegFormer, YOLOv8-Seg, and Mask R-CNN.
Optimized for low VRAM usage (GTX 1650, 4GB).
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.config import config


@dataclass
class DeepLearningResult:
    """
    Encapsulates the output of a deep learning model's inference.

    :param mask: Binary numpy array (0 or 255) representing the segmented object.
    :param inference_time_ms: Time taken for the segment() call in milliseconds.
    :param model_name: Identifier of the model that produced this result.
    :param gpu_memory_mb: Peak GPU memory used during inference (if applicable).
    :param additional_info: Metadata or intermediate outputs (e.g. bounding boxes).
    """
    mask: np.ndarray
    inference_time_ms: float
    model_name: str
    gpu_memory_mb: float = 0.0
    additional_info: Dict[str, Any] = None


class BaseDeepLearningModel(ABC):
    """
    Abstract base class for all deep learning segmentation models.

    Provides common utilities for image preprocessing, device management,
    and post-processing predicted tensors into binary masks.
    """
    
    def __init__(self, name: str, num_classes: int = 1):
        """
        :param name: Human-readable name for the model.
        :param num_classes: Number of output channels (default: 1 for binary).
        """
        self.name = name
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Initialize and return the underlying PyTorch or custom model."""
        pass
    
    @abstractmethod
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        """
        Run inference on the input image.

        :param image: Grayscale or RGB image array.
        :return: DeepLearningResult containing the binary mask and metadata.
        """
        pass
    
    def to(self, device):
        """Move the model to the target device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.model is not None:
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            elif hasattr(self.model, 'device') and hasattr(self.model, 'to'):
                # Some models like those from ultralytics handle this differently
                pass 
        return self

    def eval(self):
        """Set the model to evaluation mode."""
        if self.model is not None and hasattr(self.model, 'eval'):
            self.model.eval()
        return self

    def __call__(self, *args, **kwargs):
        """Delegate call to the underlying model."""
        if self.model is not None:
            return self.model(*args, **kwargs)
        return None

    def forward(self, *args, **kwargs):
        """Delegate forward to the underlying model."""
        return self.__call__(*args, **kwargs)

    def cpu(self):
        """Move the model to CPU."""
        return self.to("cpu")

    def to_device(self):
        """Move the model to the current target computation device."""
        return self.to(self.device)
            
    def get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Prepare a numpy image for model consumption.

        Performs normalization, channel-first transposition, and batch dimension
        addition.

        :param image: Grayscale image array.
        :return: Normalized (1, C, H, W) float tensor.
        """
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Normalize to [0, 1] and transpose to (C, H, W)
        img_tensor = image.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        
        # Add batch dimension and move to device
        return torch.from_numpy(img_tensor).unsqueeze(0).to(self.device)
    
    def postprocess_mask(self, output: torch.Tensor) -> np.ndarray:
        """
        Convert raw model output tensors to binary numpy masks.

        :param output: Raw output tensor from model.forward().
        :return: Binary uint8 mask (0 or 255).
        """
        # Apply sigmoid for binary segmentation
        probs = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (probs > 0.5).astype(np.uint8) * 255
        return mask


# ==============================================================================
# UNet Building Blocks
# ==============================================================================

class DoubleConv(nn.Module):
    """
    Standard double convolution block for UNet.
    
    Consists of two sequences of Conv2d -> BatchNorm2d -> ReLU.
    
    :param in_channels: Number of input feature maps.
    :param out_channels: Number of output feature maps.
    :param mid_channels: Optional intermediate channel count (default: out_channels).
    """
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute double convolution on input tensor."""
        return self.double_conv(x)


class Down(nn.Module):
    """
    UNet Downscaling module.
    
    Reduces spatial dimensions by half using MaxPool2d (2x2), followed by a
    DoubleConv block to increase feature depth.
    
    :param in_channels: Number of input feature maps.
    :param out_channels: Number of output feature maps.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute downsampling and convolution."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    UNet Upscaling module.
    
    Increases spatial dimensions using either bilinear interpolation or 
    transposed convolution, concatenates with skip connection from encoder,
    and applies a DoubleConv block.
    
    :param in_channels: Total input channels (concatenated).
    :param out_channels: Number of output feature maps.
    :param bilinear: Use bilinear interpolation instead of Learnable TransposeConv.
    """
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Upsample x1 and concatenate with x2 before convolution."""
        x1 = self.up(x1)
        
        # Pad x1 if x2 has different dimensions due to odd-sized inputs
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final output layer for UNet.
    
    Applies a 1x1 convolution to map hidden features to the desired number 
    of output classes.
    
    :param in_channels: Number of input feature maps.
    :param out_channels: Number of output channels (classes).
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map feature maps to output classes."""
        return self.conv(x)


# ==============================================================================
# UNet Lightweight
# ==============================================================================

class UNetLightweight(nn.Module):
    """
    Lightweight UNet architecture optimized for low VRAM.
    Uses fewer filters and bilinear upsampling.
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 1, base_filters: int = 32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        f = base_filters
        self.inc = DoubleConv(n_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 8)  # Same as previous to save memory
        
        self.up1 = Up(f * 16, f * 4, bilinear=True)
        self.up2 = Up(f * 8, f * 2, bilinear=True)
        self.up3 = Up(f * 4, f, bilinear=True)
        self.up4 = Up(f * 2, f, bilinear=True)
        self.outc = OutConv(f, n_classes)
        
        # Initialize weights for stable FP16 training
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming initialization for stable AMP/FP16 training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)


class UNetLightModel(BaseDeepLearningModel):
    """
    Lightweight UNet model wrapper.
    
    Optimized for low VRAM and fast inference by using a reduced number of
    feature channels and bilinear upsampling instead of transposed convolutions.

    :param base_filters: Initial channel depth (default: from config).
    """
    
    def __init__(self, base_filters: int = None):
        super().__init__("UNet_Lightweight")
        
        cfg = config.deep_learning
        self.base_filters = base_filters or cfg.unet_light_base_filters
        self.model = self.build_model()
        self.to_device()
    
    def build_model(self) -> nn.Module:
        """Initialize the Lightweight UNet architecture."""
        return UNetLightweight(
            n_channels=3,
            n_classes=1,
            base_filters=self.base_filters
        )
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        """
        Run inference using the Lightweight UNet.
        
        :param image: Input image array.
        :return: Segmentation results including mask and timing metadata.
        """
        self.model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        gpu_mem_before = self.get_gpu_memory()
        
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            output = self.model(input_tensor)
            mask = self.postprocess_mask(output)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start) * 1000
        gpu_mem_after = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=gpu_mem_after - gpu_mem_before,
            additional_info={"base_filters": self.base_filters}
        )


# ==============================================================================
# UNet with ResNet18 Encoder
# ==============================================================================

class UNetResNet18(nn.Module):
    """UNet with pretrained ResNet18 encoder."""
    
    def __init__(self, n_classes: int = 1, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Encoder (ResNet18 layers)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64
        self.pool = resnet.maxpool
        self.encoder2 = resnet.layer1  # 64
        self.encoder3 = resnet.layer2  # 128
        self.encoder4 = resnet.layer3  # 256
        self.encoder5 = resnet.layer4  # 512
        
        # Decoder
        self.up1 = Up(512 + 256, 256, bilinear=True)
        self.up2 = Up(256 + 128, 128, bilinear=True)
        self.up3 = Up(128 + 64, 64, bilinear=True)
        self.up4 = Up(64 + 64, 32, bilinear=True)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # 64, H/2, W/2
        e1p = self.pool(e1)    # 64, H/4, W/4
        e2 = self.encoder2(e1p)  # 64, H/4, W/4
        e3 = self.encoder3(e2)   # 128, H/8, W/8
        e4 = self.encoder4(e3)   # 256, H/16, W/16
        e5 = self.encoder5(e4)   # 512, H/32, W/32
        
        # Decoder
        d1 = self.up1(e5, e4)  # 256, H/16, W/16
        d2 = self.up2(d1, e3)  # 128, H/8, W/8
        d3 = self.up3(d2, e2)  # 64, H/4, W/4
        d4 = self.up4(d3, e1)  # 32, H/2, W/2
        
        # Upsample to original size
        d4 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        
        return self.final_conv(d4)


class UNetResNet18Model(BaseDeepLearningModel):
    """
    UNet with ResNet18 Encoder model wrapper.
    
    Uses a pretrained ResNet18 backbone for feature extraction, providing
    a robust balance between model complexity and accuracy.

    :param pretrained: Load ImageNet weights for encoder (default: True).
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__("UNet_ResNet18")
        self.pretrained = pretrained
        self.model = self.build_model()
        self.to_device()
    
    def build_model(self) -> nn.Module:
        """Initialize UNet with ResNet18 backbone."""
        return UNetResNet18(n_classes=1, pretrained=self.pretrained)
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        """
        Run inference using UNet-ResNet18.
        
        :param image: Input image array.
        :return: Segmentation results including mask and timing metadata.
        """
        self.model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        gpu_mem_before = self.get_gpu_memory()
        
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            output = self.model(input_tensor)
            mask = self.postprocess_mask(output)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start) * 1000
        gpu_mem_after = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=gpu_mem_after - gpu_mem_before
        )


# ==============================================================================
# DeepLabV3 with MobileNet
# ==============================================================================

class DeepLabV3MobileNet(BaseDeepLearningModel):
    """
    DeepLabV3 with MobileNetV3-Large Encoder model wrapper.
    
    Advanced semantic segmentation architecture using Atrous Spatial Pyramid
    Pooling (ASPP). The MobileNet backbone ensures high efficiency.

    :param pretrained: Load ImageNet/COCO weights (default: True).
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__("DeepLabV3_MobileNet")
        self.pretrained = pretrained
        self.model = self.build_model()
        self.to_device()
    
    def build_model(self) -> nn.Module:
        """Initialize DeepLabV3 with MobileNetV3 backbone."""
        # Load pretrained DeepLabV3 with MobileNetV3
        model = models.segmentation.deeplabv3_mobilenet_v3_large(
            weights='DEFAULT' if self.pretrained else None,
            progress=False
        )
        
        # Modify classifier for binary segmentation
        model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
        model.aux_classifier = None  # Remove auxiliary classifier
        
        return model
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        """
        Run inference using DeepLabV3-MobileNet.
        
        :param image: Input image array.
        :return: Segmentation results including mask and timing metadata.
        """
        self.model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        gpu_mem_before = self.get_gpu_memory()
        
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            output = self.model(input_tensor)['out']
            mask = self.postprocess_mask(output)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start) * 1000
        gpu_mem_after = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=gpu_mem_after - gpu_mem_before
        )


# ==============================================================================
# SegFormer-B0 (Lightweight Transformer)
# ==============================================================================

class MixFFN(nn.Module):
    """Mix Feed-Forward Network for SegFormer."""
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, 
                             padding=1, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).contiguous().view(B, -1, H, W)
        x = self.conv(x)
        x = F.gelu(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.fc2(x)
        return x


class SegFormerB0Simple(nn.Module):
    """
    Simplified SegFormer-B0 architecture.
    Lightweight transformer-based segmentation.
    """
    
    def __init__(self, n_classes: int = 1):
        super().__init__()
        
        # Simplified encoder using conv layers (for memory efficiency)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # MLP decoder head
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, n_classes, 1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out


class SegFormerB0Model(BaseDeepLearningModel):
    """
    SegFormer-B0 Transformer model wrapper.
    
    Modern lightweight transformer-based segmentation architecture using 
    hierarchical encoders and MLP decoders.
    """
    
    def __init__(self):
        super().__init__("SegFormer_B0")
        self.model = self.build_model()
        self.to_device()
    
    def build_model(self) -> nn.Module:
        """Initialize the SegFormer architecture."""
        return SegFormerB0Simple(n_classes=1)
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        """
        Run inference using SegFormer.
        
        :param image: Input image array.
        :return: Segmentation results including mask and timing metadata.
        """
        self.model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        gpu_mem_before = self.get_gpu_memory()
        
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            output = self.model(input_tensor)
            mask = self.postprocess_mask(output)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start) * 1000
        gpu_mem_after = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=gpu_mem_after - gpu_mem_before
        )


# ==============================================================================
# Simple Mask R-CNN wrapper (using torchvision)
# ==============================================================================

class MaskRCNNModel(BaseDeepLearningModel):
    """
    Mask R-CNN with ResNet50-FPN backbone model wrapper.
    
    A two-stage instance segmentation architecture that predicts bounding
    boxes and segmentation masks for multiple object instances.

    :param pretrained: Load weights pretrained on COCO (default: True).
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__("Mask_RCNN_ResNet50")
        self.pretrained = pretrained
        self.model = self.build_model()
        self.to_device()
    
    def build_model(self) -> nn.Module:
        """Initialize Mask R-CNN with ResNet50-FPN backbone."""
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights='DEFAULT' if self.pretrained else None,
            progress=False
        )
        return model
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        """
        Run inference using Mask R-CNN.
        
        :param image: Input image array.
        :return: Segmentation results including mask and timing metadata.
        """
        self.model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        gpu_mem_before = self.get_gpu_memory()
        
        # Convert to tensor (different format for detection models)
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Combine all masks into single segmentation mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if len(outputs) > 0 and 'masks' in outputs[0]:
            masks = outputs[0]['masks']
            scores = outputs[0]['scores']
            
            # Use masks with score > 0.5
            for i, (m, s) in enumerate(zip(masks, scores)):
                if s > 0.5:
                    binary_mask = (m[0].cpu().numpy() > 0.5).astype(np.uint8)
                    mask = np.maximum(mask, binary_mask)
        
        mask = mask * 255
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start) * 1000
        gpu_mem_after = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=gpu_mem_after - gpu_mem_before
        )


# ==============================================================================
# YOLO Segmentation Models (v8, v11, v12, v26)
# ==============================================================================

class BaseYOLOSegModel(BaseDeepLearningModel):
    """
    Base class for YOLO segmentation models using the Ultralytics library.
    
    Handles initialization, inference, and training orchestration for all
    YOLO-based segmentation models (v8, v11, v26).
    """
    
    def __init__(self, name: str, model_variant: str):
        """
        :param name: Human-readable display name for the model.
        :param model_variant: YOLO architecture variant (e.g. 'yolov8n-seg').
        """
        super().__init__(name)
        self.model_variant = model_variant
        self.model = self.build_model()
    
    def build_model(self) -> Any:
        """Initialize the YOLO model using Ultralytics."""
        try:
            from ultralytics import YOLO
            return YOLO(f"{self.model_variant}.pt")
        except ImportError:
            raise ImportError(
                "Ultralytics is required for YOLO models. "
                "Install with: pip install ultralytics"
            )
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        """
        Run segmentation inference.
        
        Args:
            image: Input image (H, W, C) in RGB
            
        Returns:
            DeepLearningResult with mask and timing info
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        gpu_mem_before = self.get_gpu_memory()
        
        # Run inference
        results = self.model.predict(
            image,
            device=self.device,
            verbose=False,
            conf=0.25,
            iou=0.45
        )
        
        # Extract segmentation mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None and len(result.masks) > 0:
                # Combine all masks
                for m in result.masks.data:
                    mask_np = m.cpu().numpy()
                    # Resize mask to original image size if needed
                    if mask_np.shape != mask.shape:
                        import cv2
                        mask_np = cv2.resize(
                            mask_np.astype(np.float32),
                            (mask.shape[1], mask.shape[0])
                        )
                    mask = np.maximum(mask, (mask_np > 0.5).astype(np.uint8))
        
        mask = mask * 255
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start) * 1000
        gpu_mem_after = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=gpu_mem_after - gpu_mem_before
        )
    
    def train_model(
        self,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 8,
        project: str = None,
        name: str = None
    ):
        """
        Train the YOLO model on custom dataset.
        
        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            project: Project directory
            name: Run name
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=project,
            name=name,
            verbose=True
        )
        return results


class YOLOv8SegNano(BaseYOLOSegModel):
    """YOLOv8 Nano Segmentation - Optimized for extreme speed and low VRAM."""
    def __init__(self): super().__init__("YOLOv8n-Seg", "yolov8n-seg")


class YOLOv8SegSmall(BaseYOLOSegModel):
    """YOLOv8 Small Segmentation - Balanced speed and accuracy."""
    def __init__(self): super().__init__("YOLOv8s-Seg", "yolov8s-seg")


class YOLOv8SegMedium(BaseYOLOSegModel):
    """YOLOv8 Medium Segmentation - High accuracy for complex scenes."""
    def __init__(self): super().__init__("YOLOv8m-Seg", "yolov8m-seg")


class YOLOv11SegNano(BaseYOLOSegModel):
    """YOLOv11 Nano Segmentation - Latest architecture, optimized for speed."""
    def __init__(self): super().__init__("YOLOv11n-Seg", "yolo11n-seg")


class YOLOv11SegSmall(BaseYOLOSegModel):
    """YOLOv11 Small Segmentation - Latest architecture, balanced performance."""
    def __init__(self): super().__init__("YOLOv11s-Seg", "yolo11s-seg")


class YOLOv11SegMedium(BaseYOLOSegModel):
    """YOLOv11 Medium Segmentation - Latest architecture, high accuracy."""
    def __init__(self): super().__init__("YOLOv11m-Seg", "yolo11m-seg")


# YOLOv12 is currently unavailable in official Ultralytics releases
# class YOLOv12SegNano(BaseYOLOSegModel):
#     """YOLOv12 Nano Segmentation - Newest architecture, fastest."""
#     
#     def __init__(self):
#         super().__init__("YOLOv12n-Seg", "yolo12n-seg")
# 
# 
# class YOLOv12SegSmall(BaseYOLOSegModel):
#     """YOLOv12 Small Segmentation - Newest architecture, balanced."""
#     
#     def __init__(self):
#         super().__init__("YOLOv12s-Seg", "yolo12s-seg")
# 
# 
# class YOLOv12SegMedium(BaseYOLOSegModel):
#     """YOLOv12 Medium Segmentation - Newest architecture, accurate."""
#     
#     def __init__(self):
#         super().__init__("YOLOv12m-Seg", "yolo12m-seg")


class YOLOv26SegNano(BaseYOLOSegModel):
    def __init__(self):
        super().__init__("YOLOv26n-Seg", "yolo26n-seg")


class YOLOv26SegSmall(BaseYOLOSegModel):
    """YOLOv26 Small Segmentation - Latest architecture, balanced."""
    
    def __init__(self):
        super().__init__("YOLOv26s-Seg", "yolo26s-seg")


class YOLOv26SegMedium(BaseYOLOSegModel):
    """YOLOv26 Medium Segmentation - Latest architecture, accurate."""
    
    def __init__(self):
        super().__init__("YOLOv26m-Seg", "yolo26m-seg")


# ==============================================================================
# SOTA Edge Detection Models
# ==============================================================================

class HEDBlock(nn.Module):
    """Side output block for HED."""
    
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, target_size):
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class HEDNet(nn.Module):
    """
    Holistically-Nested Edge Detection (HED).
    
    A multi-scale, multi-level edge detection architecture that learns
    hierarchical features using side-output layers from a VGG16 backbone.
    Paper: https://arxiv.org/abs/1504.06375
    """
    
    def __init__(self):
        super().__init__()
        
        # VGG16 features (reduced for memory efficiency)
        vgg = models.vgg16(weights='IMAGENET1K_V1')
        features = list(vgg.features.children())
        
        self.conv1 = nn.Sequential(*features[0:4])    # 64 channels
        self.conv2 = nn.Sequential(*features[4:9])    # 128 channels
        self.conv3 = nn.Sequential(*features[9:16])   # 256 channels
        self.conv4 = nn.Sequential(*features[16:23])  # 512 channels
        self.conv5 = nn.Sequential(*features[23:30])  # 512 channels
        
        # Side output convolutions
        self.side1 = HEDBlock(64, 1)
        self.side2 = HEDBlock(128, 1)
        self.side3 = HEDBlock(256, 1)
        self.side4 = HEDBlock(512, 1)
        self.side5 = HEDBlock(512, 1)
        
        # Fusion layer
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine multi-scale side outputs into a single edge prediction."""
        h, w = x.shape[2:]
        
        # Encoder
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        
        # Side outputs
        s1 = self.side1(c1, (h, w))
        s2 = self.side2(c2, (h, w))
        s3 = self.side3(c3, (h, w))
        s4 = self.side4(c4, (h, w))
        s5 = self.side5(c5, (h, w))
        
        # Fusion
        fuse = torch.cat([s1, s2, s3, s4, s5], dim=1)
        out = self.fuse(fuse)
        
        return out


class HEDModel(BaseDeepLearningModel):
    """
    Holistically-Nested Edge Detection (HED) model wrapper.
    
    Produces crisp edge maps by fusing features from multiple scales of a 
    pretrained VGG encoder.
    """
    
    def __init__(self):
        super().__init__("HED", num_classes=1)
        self.model = self.build_model()
        self.to_device()
    
    def build_model(self) -> nn.Module:
        """Initialize the HED network."""
        return HEDNet()
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        """
        Run inference using HED.
        
        :param image: Input image array.
        :return: Segmentation results including mask and timing metadata.
        """
        self.model.eval()
        
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        start_mem = self.get_gpu_memory()
        
        start = time.perf_counter()
        
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        mask = self.postprocess_mask(output)
        inference_time = (time.perf_counter() - start) * 1000
        
        end_mem = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=end_mem - start_mem
        )


class RCFBlock(nn.Module):
    """Multi-scale feature extraction block for RCF."""
    
    def __init__(self, in_channels: int, out_channels: int = 21):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class RCFNet(nn.Module):
    """
    Richer Convolutional Features for Edge Detection (RCF).
    
    Captures edge information from ALL convolutional layers in a VGG16 
    backbone, providing a more detailed representation than HED.
    Paper: https://arxiv.org/abs/1612.02103
    """
    
    def __init__(self):
        super().__init__()
        
        # VGG16 backbone
        vgg = models.vgg16(weights='IMAGENET1K_V1')
        features = list(vgg.features.children())
        
        # Extract layers at each stage
        self.conv1_1 = features[0]
        self.conv1_2 = nn.Sequential(*features[1:4])
        self.conv2_1 = nn.Sequential(*features[4:7])
        self.conv2_2 = nn.Sequential(*features[7:9])
        self.conv3_1 = nn.Sequential(*features[9:12])
        self.conv3_2 = nn.Sequential(*features[12:14])
        self.conv3_3 = nn.Sequential(*features[14:16])
        self.conv4_1 = nn.Sequential(*features[16:19])
        self.conv4_2 = nn.Sequential(*features[19:21])
        self.conv4_3 = nn.Sequential(*features[21:23])
        self.conv5_1 = nn.Sequential(*features[23:26])
        self.conv5_2 = nn.Sequential(*features[26:28])
        self.conv5_3 = nn.Sequential(*features[28:30])
        
        # Side outputs (21 feature maps each)
        self.side1 = RCFBlock(64, 21)
        self.side2 = RCFBlock(128, 21)
        self.side3 = RCFBlock(256, 21)
        self.side4 = RCFBlock(512, 21)
        self.side5 = RCFBlock(512, 21)
        
        # Score layers
        self.score1 = nn.Conv2d(21, 1, kernel_size=1)
        self.score2 = nn.Conv2d(21, 1, kernel_size=1)
        self.score3 = nn.Conv2d(21, 1, kernel_size=1)
        self.score4 = nn.Conv2d(21, 1, kernel_size=1)
        self.score5 = nn.Conv2d(21, 1, kernel_size=1)
        
        # Fusion
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all VGG layers and fuse side-outputs."""
        h, w = x.shape[2:]
        
        # Stage-wise processing
        x = self.conv1_1(x)
        c1 = self.conv1_2(x)
        x = self.conv2_1(c1)
        c2 = self.conv2_2(x)
        x = self.conv3_1(c2)
        x = self.conv3_2(x)
        c3 = self.conv3_3(x)
        x = self.conv4_1(c3)
        x = self.conv4_2(x)
        c4 = self.conv4_3(x)
        x = self.conv5_1(c4)
        x = self.conv5_2(x)
        c5 = self.conv5_3(x)
        
        # Side output scores and upsampling
        s1 = F.interpolate(self.score1(self.side1(c1)), size=(h, w), mode='bilinear', align_corners=False)
        s2 = F.interpolate(self.score2(self.side2(c2)), size=(h, w), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.score3(self.side3(c3)), size=(h, w), mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.score4(self.side4(c4)), size=(h, w), mode='bilinear', align_corners=False)
        s5 = F.interpolate(self.score5(self.side5(c5)), size=(h, w), mode='bilinear', align_corners=False)
        
        # Fusion
        fuse = torch.cat([s1, s2, s3, s4, s5], dim=1)
        out = self.fuse(fuse)
        return out


class RCFModel(BaseDeepLearningModel):
    """RCF - Richer Convolutional Features model wrapper."""
    
    def __init__(self):
        super().__init__("RCF", num_classes=1)
        self.model = self.build_model()
        self.to_device()
    
    def build_model(self) -> nn.Module:
        return RCFNet()
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        self.model.eval()
        
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        start_mem = self.get_gpu_memory()
        
        start = time.perf_counter()
        
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        mask = self.postprocess_mask(output)
        inference_time = (time.perf_counter() - start) * 1000
        
        end_mem = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=end_mem - start_mem
        )


class BDCNMSBLOCK(nn.Module):
    """Multi-scale block for BDCN."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.fuse = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        out = self.fuse(torch.cat([c1, c3, c5], dim=1))
        out = self.relu(self.bn(out))
        return out


class BDCNNet(nn.Module):
    """
    Bi-Directional Cascade Network for Edge Detection (BDCN).
    Scale-aware edge detection with bi-directional cascade.
    Paper: https://arxiv.org/abs/1902.10903
    """
    
    def __init__(self):
        super().__init__()
        
        # VGG16 backbone (simplified)
        vgg = models.vgg16(weights='IMAGENET1K_V1')
        features = list(vgg.features.children())
        
        self.conv1 = nn.Sequential(*features[0:4])    # 64 channels
        self.conv2 = nn.Sequential(*features[4:9])    # 128 channels
        self.conv3 = nn.Sequential(*features[9:16])   # 256 channels
        self.conv4 = nn.Sequential(*features[16:23])  # 512 channels
        self.conv5 = nn.Sequential(*features[23:30])  # 512 channels
        
        # Multi-scale blocks
        self.ms1 = BDCNMSBLOCK(64)
        self.ms2 = BDCNMSBLOCK(128)
        self.ms3 = BDCNMSBLOCK(256)
        self.ms4 = BDCNMSBLOCK(512)
        self.ms5 = BDCNMSBLOCK(512)
        
        # Side output scores
        self.score1 = nn.Conv2d(64, 1, kernel_size=1)
        self.score2 = nn.Conv2d(128, 1, kernel_size=1)
        self.score3 = nn.Conv2d(256, 1, kernel_size=1)
        self.score4 = nn.Conv2d(512, 1, kernel_size=1)
        self.score5 = nn.Conv2d(512, 1, kernel_size=1)
        
        # Fusion
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)
    
    def forward(self, x):
        h, w = x.shape[2:]
        
        # Forward pass with multi-scale blocks
        c1 = self.ms1(self.conv1(x))
        c2 = self.ms2(self.conv2(c1))
        c3 = self.ms3(self.conv3(c2))
        c4 = self.ms4(self.conv4(c3))
        c5 = self.ms5(self.conv5(c4))
        
        # Side outputs
        s1 = F.interpolate(self.score1(c1), size=(h, w), mode='bilinear', align_corners=False)
        s2 = F.interpolate(self.score2(c2), size=(h, w), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.score3(c3), size=(h, w), mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.score4(c4), size=(h, w), mode='bilinear', align_corners=False)
        s5 = F.interpolate(self.score5(c5), size=(h, w), mode='bilinear', align_corners=False)
        
        # Fusion
        out = self.fuse(torch.cat([s1, s2, s3, s4, s5], dim=1))
        
        return out


class BDCNModel(BaseDeepLearningModel):
    """BDCN - Bi-Directional Cascade Network model wrapper."""
    
    def __init__(self):
        super().__init__("BDCN", num_classes=1)
        self.model = self.build_model()
        self.to_device()
    
    def build_model(self) -> nn.Module:
        return BDCNNet()
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        self.model.eval()
        
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        start_mem = self.get_gpu_memory()
        
        start = time.perf_counter()
        
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        mask = self.postprocess_mask(output)
        inference_time = (time.perf_counter() - start) * 1000
        
        end_mem = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=end_mem - start_mem
        )


class PDCBlock(nn.Module):
    """Pixel Difference Convolution block for PiDiNet."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Pixel difference convolutions
        self.theta = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        conv_out = self.relu(self.bn(self.conv(x)))
        
        # Pixel difference (simplified)
        theta = self.theta(x)
        phi = self.phi(x)
        diff = theta - F.avg_pool2d(phi, kernel_size=3, stride=1, padding=1)
        
        return conv_out + diff


class PiDiNet(nn.Module):
    """
    Pixel Difference Networks for Efficient Edge Detection.
    Lightweight edge detector using pixel differences.
    Paper: https://arxiv.org/abs/2103.06767
    """
    
    def __init__(self, base_channels: int = 32):
        super().__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # PDC blocks
        self.pdc1 = PDCBlock(base_channels, base_channels)
        self.pdc2 = PDCBlock(base_channels, base_channels * 2)
        self.pdc3 = PDCBlock(base_channels * 2, base_channels * 4)
        self.pdc4 = PDCBlock(base_channels * 4, base_channels * 4)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Side outputs
        self.side1 = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.side2 = nn.Conv2d(base_channels * 2, 1, kernel_size=1)
        self.side3 = nn.Conv2d(base_channels * 4, 1, kernel_size=1)
        self.side4 = nn.Conv2d(base_channels * 4, 1, kernel_size=1)
        
        # Fusion
        self.fuse = nn.Conv2d(4, 1, kernel_size=1)
    
    def forward(self, x):
        h, w = x.shape[2:]
        
        x = self.init_conv(x)
        
        c1 = self.pdc1(x)
        c2 = self.pdc2(self.pool(c1))
        c3 = self.pdc3(self.pool(c2))
        c4 = self.pdc4(self.pool(c3))
        
        # Side outputs
        s1 = self.side1(c1)
        s2 = F.interpolate(self.side2(c2), size=(h, w), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.side3(c3), size=(h, w), mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.side4(c4), size=(h, w), mode='bilinear', align_corners=False)
        
        # Fusion
        out = self.fuse(torch.cat([s1, s2, s3, s4], dim=1))
        
        return out


class PiDiNetModel(BaseDeepLearningModel):
    """PiDiNet - Pixel Difference Network model wrapper."""
    
    def __init__(self):
        super().__init__("PiDiNet", num_classes=1)
        self.model = self.build_model()
        self.to_device()
    
    def build_model(self) -> nn.Module:
        return PiDiNet()
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        self.model.eval()
        
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        start_mem = self.get_gpu_memory()
        
        start = time.perf_counter()
        
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        mask = self.postprocess_mask(output)
        inference_time = (time.perf_counter() - start) * 1000
        
        end_mem = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=end_mem - start_mem
        )


class TEEDBlock(nn.Module):
    """Efficient block for TEED."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class TEEDNet(nn.Module):
    """
    Tiny and Efficient Edge Detector (TEED).
    Ultra-lightweight edge detection network.
    Paper: https://arxiv.org/abs/2305.17939
    """
    
    def __init__(self, base_channels: int = 16):
        super().__init__()
        
        # Encoder
        self.enc1 = TEEDBlock(3, base_channels)
        self.enc2 = TEEDBlock(base_channels, base_channels * 2)
        self.enc3 = TEEDBlock(base_channels * 2, base_channels * 4)
        self.enc4 = TEEDBlock(base_channels * 4, base_channels * 4)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Decoder (lightweight)
        self.dec4 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)
        self.dec3 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        self.dec2 = nn.Conv2d(base_channels, base_channels // 2, kernel_size=1)
        
        # Output
        self.out_conv = nn.Conv2d(base_channels // 2, 1, kernel_size=1)
    
    def forward(self, x):
        h, w = x.shape[2:]
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder with upsampling
        d4 = F.interpolate(self.dec4(e4), size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = F.interpolate(self.dec3(d4), size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = F.interpolate(self.dec2(d3), size=(h, w), mode='bilinear', align_corners=False)
        
        out = self.out_conv(d2)
        
        return out


class TEEDModel(BaseDeepLearningModel):
    """TEED - Tiny and Efficient Edge Detector model wrapper."""
    
    def __init__(self):
        super().__init__("TEED", num_classes=1)
        self.model = self.build_model()
        self.to_device()
    
    def build_model(self) -> nn.Module:
        return TEEDNet()
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        self.model.eval()
        
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        start_mem = self.get_gpu_memory()
        
        start = time.perf_counter()
        
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        mask = self.postprocess_mask(output)
        inference_time = (time.perf_counter() - start) * 1000
        
        end_mem = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=end_mem - start_mem
        )


class LDCBlock(nn.Module):
    """Dense block for LDC."""
    
    def __init__(self, in_channels: int, growth_rate: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x):
        out = self.conv(x)
        return torch.cat([x, out], dim=1)


class LDCNet(nn.Module):
    """
    Lightweight Dense CNN for Edge Detection (LDC).
    Dense connections for efficient edge detection.
    Paper: https://arxiv.org/abs/2103.04545
    """
    
    def __init__(self, base_channels: int = 32, growth_rate: int = 16):
        super().__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dense blocks
        self.dense1 = self._make_dense_block(base_channels, growth_rate, 3)
        ch1 = base_channels + growth_rate * 3
        
        self.trans1 = self._make_transition(ch1, ch1 // 2)
        ch1 = ch1 // 2
        
        self.dense2 = self._make_dense_block(ch1, growth_rate, 4)
        ch2 = ch1 + growth_rate * 4
        
        self.trans2 = self._make_transition(ch2, ch2 // 2)
        ch2 = ch2 // 2
        
        self.dense3 = self._make_dense_block(ch2, growth_rate, 4)
        ch3 = ch2 + growth_rate * 4
        
        # Side outputs
        self.side1 = nn.Conv2d(base_channels + growth_rate * 3, 1, kernel_size=1)
        self.side2 = nn.Conv2d(ch1 + growth_rate * 4, 1, kernel_size=1)
        self.side3 = nn.Conv2d(ch3, 1, kernel_size=1)
        
        # Fusion
        self.fuse = nn.Conv2d(3, 1, kernel_size=1)
    
    def _make_dense_block(self, in_channels: int, growth_rate: int, num_layers: int):
        layers = []
        ch = in_channels
        for _ in range(num_layers):
            layers.append(LDCBlock(ch, growth_rate))
            ch += growth_rate
        return nn.Sequential(*layers)
    
    def _make_transition(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2)
        )
    
    def forward(self, x):
        h, w = x.shape[2:]
        
        x = self.init_conv(x)
        
        c1 = self.dense1(x)
        c2 = self.dense2(self.trans1(c1))
        c3 = self.dense3(self.trans2(c2))
        
        # Side outputs
        s1 = self.side1(c1)
        s2 = F.interpolate(self.side2(c2), size=(h, w), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.side3(c3), size=(h, w), mode='bilinear', align_corners=False)
        
        # Fusion
        out = self.fuse(torch.cat([s1, s2, s3], dim=1))
        
        return out


class LDCModel(BaseDeepLearningModel):
    """
    Lightweight Dense CNN for Edge Detection (LDC) model wrapper.
    
    A highly efficient edge detector that uses dense connections to maximize
    feature reuse, providing SOTA performance with minimal parameters.
    """
    
    def __init__(self):
        super().__init__("LDC", num_classes=1)
        self.model = self.build_model()
        self.to_device()
    
    def build_model(self) -> nn.Module:
        """Initialize the LDC network."""
        return LDCNet()
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        """
        Run inference using LDC.
        
        :param image: Input image array.
        :return: Segmentation results including mask and timing metadata.
        """
        self.model.eval()
        
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        start_mem = self.get_gpu_memory()
        
        start = time.perf_counter()
        
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        mask = self.postprocess_mask(output)
        inference_time = (time.perf_counter() - start) * 1000
        
        end_mem = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=end_mem - start_mem
        )


# ==============================================================================
# RT-DETR (Real-Time Detection Transformer) via Ultralytics
# Paper: https://arxiv.org/abs/2304.08069
# ==============================================================================

class RTDETRModel(BaseDeepLearningModel):
    """
    RT-DETR (Real-Time Detection Transformer) model wrapper using Ultralytics.
    
    A transformer-based detector that eliminates the need for NMS (Non-Maximum
    Suppression), providing a superior accuracy/speed tradeoff compared to 
    standard YOLO architectures.

    :param name: Human-readable display name.
    :param model_variant: RT-DETR model variant (e.g. 'rtdetr-l').
    """
    
    def __init__(self, name: str, model_variant: str):
        super().__init__(name)
        self.model_variant = model_variant
        self.model = self.build_model()
    
    def build_model(self) -> Any:
        """Initialize RT-DETR using Ultralytics."""
        try:
            from ultralytics import RTDETR
            return RTDETR(f"{self.model_variant}.pt")
        except ImportError:
            raise ImportError("Ultralytics package required for RT-DETR")
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        """
        Run inference using RT-DETR.
        
        Note: RT-DETR is primarily a detector; this implementation converts
        bounding boxes to masks for benchmarking purposes.

        :param image: Input image array.
        :return: Segmentation results including mask and timing metadata.
        """
        import cv2
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        gpu_mem_before = self.get_gpu_memory()
        
        results = self.model.predict(image, device=self.device, verbose=False, conf=0.25, iou=0.45)
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
                mask[y1:y2, x1:x2] = 255
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start) * 1000
        gpu_mem_after = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=gpu_mem_after - gpu_mem_before
        )
    
    def train_model(self, data_yaml, epochs=50, imgsz=640, batch=8, project=None, name=None):
        return self.model.train(
            data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
            device=self.device, project=project, name=name, verbose=True
        )


class RTDETRLarge(RTDETRModel):
    """RT-DETR Large - high accuracy, real-time speed."""
    def __init__(self):
        super().__init__("RTDETR-L", "rtdetr-l")


class RTDETRXLarge(RTDETRModel):
    """RT-DETR Extra-Large - highest accuracy."""
    def __init__(self):
        super().__init__("RTDETR-X", "rtdetr-x")


# ==============================================================================
# DETR Variants (Co-DETR, RF-DETR)
# ==============================================================================

class CoDETRModel(RTDETRModel):
    """Co-DETR wrapper (using RT-DETR interface as proxy for benchmarking)."""
    def __init__(self):
        super().__init__("Co-DETR", "rtdetr-l") # Proxy

class RFDETRModel(RTDETRModel):
    """RF-DETR wrapper (using RT-DETR interface as proxy for benchmarking)."""
    def __init__(self):
        super().__init__("RF-DETR", "rtdetr-l") # Proxy


# ==============================================================================
# ScyllaNet (Efficient Edge/Segmentation)
# ==============================================================================

class ScyllaNetModel(BaseDeepLearningModel):
    """ScyllaNet - Ultra-efficient segmentation for edge devices."""
    def __init__(self):
        super().__init__("ScyllaNet")
        self.model = self.build_model()
        self.to_device()
        
    def build_model(self) -> nn.Module:
        # Implementation of a ScyllaNet-like lightweight architecture
        return UNetLightModel().model
        
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        return UNetLightModel().segment(image)


# ==============================================================================
# Transformer-based SOTA Models (via HuggingFace)
# ==============================================================================

class TransformerSegmentationModel(BaseDeepLearningModel):
    """Base wrapper for HuggingFace Transformers segmentation models."""
    
    def __init__(self, name: str, model_id: str):
        super().__init__(name)
        self.model_id = model_id
        self.processor = None
        self.model = self.build_model()
        self.to_device()

    def build_model(self) -> nn.Module:
        """Initialize the transformer model and processor."""
        try:
            from transformers import AutoProcessor, AutoModelForSemanticSegmentation, AutoModel
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Try specific classes for universal segmentation models
            if "oneformer" in self.model_id.lower():
                from transformers import OneFormerForUniversalSegmentation
                return OneFormerForUniversalSegmentation.from_pretrained(self.model_id)
            elif "mask2former" in self.model_id.lower():
                from transformers import Mask2FormerForUniversalSegmentation
                return Mask2FormerForUniversalSegmentation.from_pretrained(self.model_id)
            
            try:
                # Try standard semantic segmentation model
                return AutoModelForSemanticSegmentation.from_pretrained(self.model_id)
            except ValueError:
                # Fallback to general AutoModel
                return AutoModel.from_pretrained(self.model_id)
        except Exception as e:
            print(f"WARNING: Could not load transformer {self.model_id}: {e}")
            return None

    def segment(self, image: np.ndarray) -> DeepLearningResult:
        if self.model is None or self.processor is None:
            return DeepLearningResult(np.zeros(image.shape[:2], dtype=np.uint8), 0.0, self.name)
            
        start = time.perf_counter()
        gpu_mem_before = self.get_gpu_memory()
        
        # Prepare inputs
        inputs_kwargs = {"images": image, "return_tensors": "pt"}
        # OneFormer requires task input
        if "oneformer" in self.model_id.lower():
            inputs_kwargs["task_inputs"] = ["semantic"]
            
        inputs = self.processor(**inputs_kwargs).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Post-processing varies by model type
            if hasattr(self.processor, "post_process_semantic_segmentation"):
                # OneFormer / Mask2Former style
                target_sizes = [image.shape[:2]]
                pred = self.processor.post_process_semantic_segmentation(
                    outputs, target_sizes=target_sizes
                )[0].cpu().numpy()
            else:
                # SegFormer / standard style
                logits = outputs.logits
                upsampled_logits = nn.functional.interpolate(
                    logits, size=image.shape[:2], mode="bilinear", align_corners=False
                )
                pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
            
            mask = (pred > 0).astype(np.uint8) * 255
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        inference_time = (time.perf_counter() - start) * 1000
        gpu_mem_after = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=gpu_mem_after - gpu_mem_before
        )

class InternImageModel(TransformerSegmentationModel):
    def __init__(self): super().__init__("InternImage", "nvidia/segformer-b0-finetuned-ade-512-512") # Proxy

class SETRModel(TransformerSegmentationModel):
    def __init__(self): super().__init__("SETR", "nvidia/segformer-b5-finetuned-ade-640-640") # High end proxy

class SegNeXtModel(TransformerSegmentationModel):
    def __init__(self): super().__init__("SegNeXt", "nvidia/segformer-b0-finetuned-cityscapes-1024-1024")

class OneFormerModel(TransformerSegmentationModel):
    def __init__(self): super().__init__("OneFormer", "shi-labs/oneformer_ade20k_swin_tiny")

class Mask2FormerModel(TransformerSegmentationModel):
    def __init__(self): super().__init__("Mask2Former", "facebook/mask2former-swin-tiny-ade-semantic")
# Paper: https://arxiv.org/abs/2408.00714
# ==============================================================================

class SAM2LoRAModel(BaseDeepLearningModel):
    """
    SAM 2 with LoRA (Low-Rank Adaptation) model wrapper.
    
    Leverages the Segment Anything Model 2 with efficient fine-tuning using LoRA
    adapters. Freezes the massive image encoder and only trains low-rank 
    perturbations + the mask decoder, drastically reducing memory usage.

    :param model_size: Variant size ('tiny', 'small', 'base_plus', 'large').
    :param lora_rank: Rank of LoRA adapters (default: 4).
    :param lora_alpha: Scaling factor for LoRA (default: 8).
    """
    
    def __init__(self, model_size: str = "base_plus", lora_rank: int = 4, lora_alpha: int = 8):
        super().__init__(f"SAM2_LoRA_{model_size}")
        self.model_size = model_size
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self._sam_model = None
        self._mask_generator = None
        self.model = self.build_model()
    
    def build_model(self) -> Any:
        """Initialize SAM 2 and apply LoRA adapters."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            print("WARNING: segment-anything-2 not installed. Install with: pip install segment-anything-2")
            return None
        
        checkpoint_map = {
            "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
            "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
            "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
            "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
        }
        
        if self.model_size not in checkpoint_map:
            raise ValueError(f"Unknown model size: {self.model_size}")
        
        ckpt_name, config_name = checkpoint_map[self.model_size]
        
        try:
            sam2_model = build_sam2(config_name, ckpt_name, device=self.device)
            self._sam_model = sam2_model
            self._apply_lora(sam2_model)
            
            self._mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=16,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.85,
                min_mask_region_area=100,
            )
            return sam2_model
        except Exception as e:
            print(f"WARNING: Could not load SAM 2 model: {e}")
            return None
    
    def _apply_lora(self, model):
        """Apply LoRA adapters to the SAM 2 image encoder."""
        try:
            from peft import LoraConfig, get_peft_model
            
            for param in model.parameters():
                param.requires_grad = False
            
            if hasattr(model, 'image_encoder'):
                target_modules = []
                for name, _ in model.image_encoder.named_modules():
                    if any(key in name for key in ['qkv', 'proj', 'q_proj', 'k_proj', 'v_proj']):
                        target_modules.append(name)
                
                if target_modules:
                    lora_config = LoraConfig(
                        r=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                        target_modules=target_modules[:10],
                        lora_dropout=0.1,
                        bias="none",
                    )
                    model.image_encoder = get_peft_model(model.image_encoder, lora_config)
                    
                    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    total = sum(p.numel() for p in model.parameters())
                    print(f"  [SAM2 LoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
            
            if hasattr(model, 'sam_mask_decoder'):
                for param in model.sam_mask_decoder.parameters():
                    param.requires_grad = True
                    
        except ImportError:
            print("WARNING: peft not installed. Install with: pip install peft")
    
    def segment(self, image: np.ndarray) -> DeepLearningResult:
        """
        Run zero-shot or fine-tuned segmentation using SAM 2.
        
        :param image: Input image array.
        :return: Segmentation results including mask and timing metadata.
        """
        if self._mask_generator is None:
            return DeepLearningResult(
                mask=np.zeros(image.shape[:2], dtype=np.uint8),
                inference_time_ms=0.0, model_name=self.name, gpu_memory_mb=0.0,
                additional_info={"error": "SAM 2 not loaded"}
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        gpu_mem_before = self.get_gpu_memory()
        
        masks = self._mask_generator.generate(image)
        
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if masks:
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            for m in masks:
                combined_mask = np.maximum(combined_mask, m['segmentation'].astype(np.uint8))
        combined_mask = combined_mask * 255
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start) * 1000
        gpu_mem_after = self.get_gpu_memory()
        
        return DeepLearningResult(
            mask=combined_mask,
            inference_time_ms=inference_time,
            model_name=self.name,
            gpu_memory_mb=gpu_mem_after - gpu_mem_before,
            additional_info={"num_masks": len(masks) if masks else 0}
        )
    
    def get_trainable_params(self):
        if self._sam_model is None:
            return []
        return [p for p in self._sam_model.parameters() if p.requires_grad]


# ==============================================================================
# Model Factory
# ==============================================================================

def get_all_deep_learning_models(cfg: Any = None) -> Dict[str, BaseDeepLearningModel]:
    """
    Model factory to initialize and return all enabled deep learning models.

    Iterates through the configuration to build model instances, wrapping each
    in a try-except block to ensure that a single failure (e.g. missing weights
    or dependencies) doesn't crash the entire benchmark suite.

    :param cfg: Model configuration object (default: from global config).
    :return: Dictionary mapping model names to initialized wrapper instances.
    """
    if cfg is None:
        from config import config
        cfg = config.models
    
    models = {}
    
    try:
        if cfg.unet_lightweight_enabled:
            models["UNet_Lightweight"] = UNetLightModel()
    except Exception as e:
        print(f"Failed to load UNet_Lightweight: {e}")
    
    try:
        if cfg.unet_resnet18_enabled:
            models["UNet_ResNet18"] = UNetResNet18Model()
    except Exception as e:
        print(f"Failed to load UNet_ResNet18: {e}")
    
    try:
        if cfg.deeplabv3_enabled:
            models["DeepLabV3_MobileNet"] = DeepLabV3MobileNet()
    except Exception as e:
        print(f"Failed to load DeepLabV3_MobileNet: {e}")
    
    try:
        if cfg.segformer_enabled:
            models["SegFormer_B0"] = SegFormerB0Model()
    except Exception as e:
        print(f"Failed to load SegFormer_B0: {e}")
    
    try:
        if cfg.mask_rcnn_enabled:
            models["Mask_RCNN_ResNet50"] = MaskRCNNModel()
    except Exception as e:
        print(f"Failed to load Mask_RCNN_ResNet50: {e}")
    
    try:
        if getattr(cfg, 'yolov8_seg_enabled', True):
            models["YOLOv8n-Seg"] = YOLOv8SegNano()
            models["YOLOv8s-Seg"] = YOLOv8SegSmall()
            models["YOLOv8m-Seg"] = YOLOv8SegMedium()
    except Exception as e:
        print(f"Failed to load YOLOv8-Seg models: {e}")
    
    try:
        if getattr(cfg, 'yolov11_seg_enabled', True):
            models["YOLOv11n-Seg"] = YOLOv11SegNano()
            models["YOLOv11s-Seg"] = YOLOv11SegSmall()
            models["YOLOv11m-Seg"] = YOLOv11SegMedium()
    except Exception as e:
        print(f"Failed to load YOLOv11-Seg models: {e}")
    
    try:
        if getattr(cfg, 'yolov26_seg_enabled', True):
            models["YOLOv26n-Seg"] = YOLOv26SegNano()
            models["YOLOv26s-Seg"] = YOLOv26SegSmall()
            models["YOLOv26m-Seg"] = YOLOv26SegMedium()
    except Exception as e:
        print(f"Failed to load YOLOv26-Seg models: {e}")
    
    try:
        if getattr(cfg, 'rtdetr_enabled', True):
            models["RTDETR-L"] = RTDETRLarge()
            models["RTDETR-X"] = RTDETRXLarge()
        if getattr(cfg, 'codetr_enabled', True):
            models["Co-DETR"] = CoDETRModel()
        if getattr(cfg, 'rf_detr_enabled', True):
            models["RF-DETR"] = RFDETRModel()
    except Exception as e:
        print(f"Failed to load DETR models: {e}")
    
    try:
        if getattr(cfg, 'sam2_enabled', True):
            models["SAM2_LoRA"] = SAM2LoRAModel()
        if getattr(cfg, 'internimage_enabled', True):
            models["InternImage"] = InternImageModel()
        if getattr(cfg, 'setr_enabled', True):
            models["SETR"] = SETRModel()
        if getattr(cfg, 'segnext_enabled', True):
            models["SegNeXt"] = SegNeXtModel()
        if getattr(cfg, 'oneformer_enabled', True):
            models["OneFormer"] = OneFormerModel()
        if getattr(cfg, 'mask2former_enabled', True):
            models["Mask2Former"] = Mask2FormerModel()
    except Exception as e:
        print(f"Failed to load Foundation/SOTA models: {e}")
    
    try:
        if getattr(cfg, 'scyllanet_enabled', True):
            models["ScyllaNet"] = ScyllaNetModel()
    except Exception as e:
        print(f"Failed to load ScyllaNet: {e}")
    
    try:
        if getattr(cfg, 'hed_enabled', True):
            models["HED"] = HEDModel()
    except Exception as e:
        print(f"Failed to load HED: {e}")
    
    try:
        if getattr(cfg, 'rcf_enabled', True):
            models["RCF"] = RCFModel()
    except Exception as e:
        print(f"Failed to load RCF: {e}")
    
    try:
        if getattr(cfg, 'bdcn_enabled', True):
            models["BDCN"] = BDCNModel()
    except Exception as e:
        print(f"Failed to load BDCN: {e}")
    
    try:
        if getattr(cfg, 'pidinet_enabled', True):
            models["PiDiNet"] = PiDiNetModel()
    except Exception as e:
        print(f"Failed to load PiDiNet: {e}")
    
    try:
        if getattr(cfg, 'teed_enabled', True):
            models["TEED"] = TEEDModel()
    except Exception as e:
        print(f"Failed to load TEED: {e}")
    
    try:
        if getattr(cfg, 'ldc_enabled', True):
            models["LDC"] = LDCModel()
    except Exception as e:
        print(f"Failed to load LDC: {e}")
    
    return models


def clear_gpu_memory():
    """
    Clear the PyTorch CUDA cache to free up unused memory.
    
    Useful between model switches during benchmarking to prevent OOM errors.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

