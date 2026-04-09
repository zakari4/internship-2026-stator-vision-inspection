#!/usr/bin/env python3
"""Export stator YOLO .pt weights to ONNX for FPS-optimized inference."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def export_model(pt_path: Path):
    """Export a single YOLO .pt model to ONNX format."""
    onnx_path = pt_path.with_suffix(".onnx")
    if onnx_path.exists():
        print(f"⏭️  Already exists: {onnx_path}")
        return str(onnx_path)

    print(f"\n{'='*60}")
    print(f"Exporting: {pt_path}")
    print(f"{'='*60}")

    model = YOLO(str(pt_path), task="segment")
    result = model.export(
        format="onnx",
        half=False,
        simplify=True,
        dynamic=False,
        imgsz=640,
        opset=17,
    )
    print(f"✅ Exported: {result}")
    return result


def main():
    stator_yolo_dir = PROJECT_ROOT / "outputs" / "results" / "yolo_training"

    if not stator_yolo_dir.exists():
        print(f"❌ Directory not found: {stator_yolo_dir}")
        sys.exit(1)

    pt_files = list(stator_yolo_dir.rglob("best.pt"))
    if not pt_files:
        print(f"❌ No best.pt files found under {stator_yolo_dir}")
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


if __name__ == "__main__":
    main()
