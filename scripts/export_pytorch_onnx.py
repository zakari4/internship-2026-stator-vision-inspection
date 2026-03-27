import sys
import os
import argparse
import torch

# Resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import builders from models
from src.models.deep_learning import UNetResNet18, SegFormerB0Simple

def export_pytorch_to_onnx(model_name: str, checkpoint_path: str, output_path: str, imgsz: int=512):
    print(f"Building PyTorch architecture for: {model_name}")
    
    if model_name == "unet_resnet18":
        model = UNetResNet18(n_classes=4, pretrained=False)
    elif model_name == "segformer_b0":
        model = SegFormerB0Simple(n_classes=4)
    else:
        print(f"Unknown architecture: {model_name}")
        return False
        
    print(f"Loading weights from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = checkpoint
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return False
        
    model.eval()
    
    # Export using FP16 if possible
    dummy_input = torch.randn(1, 3, imgsz, imgsz, device="cpu")
    
    print(f"Exporting to ONNX at {output_path}...")
    try:
        # Dynamic axes for batch size if needed, but we keep it static for TensorRT optimization later
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None  # Static batch size = 1
        )
        print(f"Successfully exported to: {output_path}")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument("--arch", type=str, required=True, help="Architecture name (e.g., unet_resnet18)")
    parser.add_argument("--weights", type=str, required=True, help="Path to the PyTorch .pth weights")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX file path")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size for export")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"Error: Model weights {args.weights} do not exist.")
        return
        
    export_pytorch_to_onnx(args.arch, args.weights, args.output, args.imgsz)

if __name__ == "__main__":
    main()
