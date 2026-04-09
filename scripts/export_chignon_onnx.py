#!/usr/bin/env python3
"""Export chignon YOLO .pt weights to ONNX for FPS-optimized inference."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def export_model(pt_path: Path):
    """Export a single YOLO .pt model to ONNX format."""
    print(f"\n{'='*60}")
    print(f"Exporting: {pt_path}")
    print(f"{'='*60}")

    model = YOLO(str(pt_path), task="segment")

    # Export to ONNX with FP16 optimization (matching stator approach)
    onnx_path = model.export(
        format="onnx",
        half=False,        # ONNX doesn't natively support FP16 weights export on CPU
        simplify=True,     # Simplify the ONNX graph for faster inference
        dynamic=False,     # Fixed input shape for max speed
        imgsz=640,
        opset=17,
    )
    print(f"✅ Exported: {onnx_path}")
    return onnx_path


def main():
    chignon_yolo_dir = PROJECT_ROOT / "chignon" / "results" / "yolo_training"

    if not chignon_yolo_dir.exists():
        print(f"❌ Directory not found: {chignon_yolo_dir}")
        sys.exit(1)

    pt_files = list(chignon_yolo_dir.rglob("best.pt"))
    if not pt_files:
        print(f"❌ No best.pt files found under {chignon_yolo_dir}")
        sys.exit(1)

    print(f"Found {len(pt_files)} model(s) to export:")
    for p in pt_files:
        print(f"  - {p}")

    exported = []
    for pt_path in pt_files:
        try:
            result = export_model(pt_path)
            exported.append(result)
        except Exception as e:
            print(f"❌ Failed to export {pt_path}: {e}")

    print(f"\n{'='*60}")
    print(f"Export complete. {len(exported)}/{len(pt_files)} models converted.")
    for p in exported:
        print(f"  ✅ {p}")


if __name__ == "__main__":
    main()
