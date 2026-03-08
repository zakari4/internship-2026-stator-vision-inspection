"""
Model definitions for classical and deep learning segmentation.
"""

from .deep_learning import (
    BaseDeepLearningModel,
    # UNet variants
    UNetLightweight,
    UNetLightModel,
    UNetResNet18,
    UNetResNet18Model,
    # DeepLab
    DeepLabV3MobileNet,
    # SegFormer
    SegFormerB0Simple,
    SegFormerB0Model,
    # Mask R-CNN
    MaskRCNNModel,
    # YOLO Segmentation v8
    YOLOv8SegNano,
    YOLOv8SegSmall,
    YOLOv8SegMedium,
    # YOLO Segmentation v11
    YOLOv11SegNano,
    YOLOv11SegSmall,
    YOLOv11SegMedium,
    # YOLO Segmentation v26
    YOLOv26SegNano,
    YOLOv26SegSmall,
    YOLOv26SegMedium,
    # RT-DETR (Real-Time Detection Transformer)
    RTDETRLarge,
    RTDETRXLarge,
    # DETR Variants
    CoDETRModel,
    RFDETRModel,
    # ScyllaNet
    ScyllaNetModel,
    # SAM 2 with LoRA
    SAM2LoRAModel,
    # Foundation / SOTA Models
    InternImageModel,
    SETRModel,
    SegNeXtModel,
    OneFormerModel,
    Mask2FormerModel,
    # SOTA Edge Detection Models
    HEDModel,
    RCFModel,
    BDCNModel,
    PiDiNetModel,
    TEEDModel,
    LDCModel,
    # Utilities
    get_all_deep_learning_models,
    clear_gpu_memory,
    DeepLearningResult
)


__all__ = [
    # Deep Learning Models
    'BaseDeepLearningModel',
    'UNetLightweight',
    'UNetLightModel',
    'UNetResNet18',
    'UNetResNet18Model',
    'DeepLabV3MobileNet',
    'SegFormerB0Simple',
    'SegFormerB0Model',
    'MaskRCNNModel',
    
    # YOLO Segmentation Models
    'YOLOv8SegNano',
    'YOLOv8SegSmall',
    'YOLOv8SegMedium',
    'YOLOv11SegNano',
    'YOLOv11SegSmall',
    'YOLOv11SegMedium',
    'YOLOv26SegNano',
    'YOLOv26SegSmall',
    'YOLOv26SegMedium',
    
    # RT-DETR Models
    'RTDETRLarge',
    'RTDETRXLarge',
    'CoDETRModel',
    'RFDETRModel',
    'ScyllaNetModel',
    
    # SAM 2 Model
    'SAM2LoRAModel',
    'InternImageModel',
    'SETRModel',
    'SegNeXtModel',
    'OneFormerModel',
    'Mask2FormerModel',
    
    # SOTA Edge Detection Models
    'HEDModel',
    'RCFModel',
    'BDCNModel',
    'PiDiNetModel',
    'TEEDModel',
    'LDCModel',
    
    'get_all_deep_learning_models',
    'clear_gpu_memory',
    'DeepLearningResult',
]
