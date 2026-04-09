"""
Chignon detection inference engine.

Self-contained module that discovers chignon YOLO/UNet models from
chignon/results/ and runs segmentation inference on images.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHIGNON_RESULTS_DIR = PROJECT_ROOT / "chignon" / "results"

logger = logging.getLogger(__name__)


# ─── Color palette ────────────────────────────────────────────────────────
CLASS_COLORS = {
    "chignon": (0, 200, 255),   # Orange
    "background": (50, 50, 50),
}
DEFAULT_COLOR = (0, 255, 255)   # Yellow fallback


# ─── Display name mappings ────────────────────────────────────────────────
_YOLO_DISPLAY = {
    "yolov8m_seg":  "YOLOv8 Medium Seg  (Chignon)",
    "yolov11m_seg": "YOLOv11 Medium Seg (Chignon)",
}

_PYTORCH_META = {
    "unet_resnet18": {"display": "UNet ResNet18 (Chignon)", "arch": "unet_resnet18"},
}


class ChignonModelManager:
    """Lightweight model manager specifically for the chignon domain."""

    def __init__(self):
        self.current_model = None
        self.current_model_name: Optional[str] = None
        self.available_models: Dict[str, dict] = {}

        # Global Post-processing Settings
        self.enable_postprocessing = True
        self.enable_top_n = True
        self.draw_boxes = True
        self.draw_masks = True
        self.draw_labels = True
        self.conf_threshold = 0.05

        self._frame_fps = 0.0
        self._fps_frame_count = 0
        self._fps_window_start = time.monotonic()

        self._discover_models()
        self._autoload_best()

    # ──────────────────────────────────────────────────────────────────
    # Discovery
    # ──────────────────────────────────────────────────────────────────

    def _discover_models(self):
        """Scan chignon/results/ for trained weights."""
        self.available_models.clear()

        # 1. YOLO weights
        yolo_dir = CHIGNON_RESULTS_DIR / "yolo_training"
        if yolo_dir.exists():
            for model_dir in sorted(yolo_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                best_engine = model_dir / "weights" / "best.engine"
                best_pt     = model_dir / "weights" / "best.pt"

                wf = best_engine if best_engine.exists() else (
                    best_pt if best_pt.exists() else None
                )
                if wf is None:
                    continue

                folder = model_dir.name
                display = _YOLO_DISPLAY.get(folder, folder)
                if wf.suffix == ".engine":
                    display += " [TensorRT]"

                self.available_models[folder] = {
                    "path": str(wf),
                    "type": "yolo",
                    "source": "trained",
                    "display_name": display,
                }

        # 2. PyTorch checkpoints
        ckpt_dir = CHIGNON_RESULTS_DIR / "checkpoints"
        if ckpt_dir.exists():
            for model_dir in sorted(ckpt_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                best_pth = model_dir / "best_model.pth"
                best_pt  = model_dir / "best_model.pt"
                wf = best_pth if best_pth.exists() else (
                    best_pt if best_pt.exists() else None
                )
                if wf is None:
                    continue

                folder = model_dir.name
                meta = _PYTORCH_META.get(folder, {})
                self.available_models[folder] = {
                    "path": str(wf),
                    "type": "pytorch",
                    "source": "trained",
                    "display_name": meta.get("display", folder),
                    "arch": meta.get("arch", folder),
                }

        logger.info(
            "[Chignon] Discovered %d model(s): %s",
            len(self.available_models),
            list(self.available_models.keys()),
        )

    def _autoload_best(self):
        """Auto-load the first available trained model."""
        for name, info in self.available_models.items():
            if info["source"] == "trained":
                logger.info("[Chignon] Auto-loading: %s", name)
                if self.load_model(name):
                    return
        logger.warning("[Chignon] No model could be auto-loaded.")

    # ──────────────────────────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────────────────────────

    def load_model(self, model_name: str) -> bool:
        if model_name not in self.available_models:
            logger.error("[Chignon] Model '%s' not found.", model_name)
            return False

        info = self.available_models[model_name]
        model_path = info["path"]

        if model_name == self.current_model_name and self.current_model is not None:
            return True

        try:
            if info["type"] == "yolo":
                from ultralytics import YOLO
                self.current_model = YOLO(model_path, task="segment")
                logger.info("[Chignon] Loaded YOLO: %s", model_path)

            elif info["type"] == "pytorch":
                import torch
                arch = info.get("arch", model_name)
                model = self._build_pytorch(arch)
                if model is None:
                    return False

                ckpt = torch.load(model_path, map_location="cpu")
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                else:
                    model = ckpt

                model.eval()
                if torch.cuda.is_available():
                    model.to("cuda").half()
                self.current_model = model
                logger.info("[Chignon] Loaded PyTorch: %s", model_path)
            else:
                return False

            self.current_model_name = model_name
            return True
        except Exception:
            logger.exception("[Chignon] Failed to load %s", model_name)
            return False

    @staticmethod
    def _build_pytorch(arch: str):
        """Build a PyTorch model architecture by name."""
        if arch == "unet_resnet18":
            try:
                from src.models.unet import UNetResNet18
                return UNetResNet18(num_classes=2)
            except ImportError:
                logger.error("Cannot import UNetResNet18")
                return None
        return None

    # ──────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        if self.current_model is None:
            return frame, []

        info = self.available_models.get(self.current_model_name, {})

        # FPS tracking
        self._fps_frame_count += 1
        now = time.monotonic()
        if now - self._fps_window_start >= 1.0:
            self._frame_fps = self._fps_frame_count / (now - self._fps_window_start)
            self._fps_frame_count = 0
            self._fps_window_start = now

        try:
            if info.get("type") == "yolo":
                return self._predict_yolo(frame)
            elif info.get("type") == "pytorch":
                return self._predict_pytorch(frame)
        except Exception:
            logger.exception("[Chignon] Inference failed")
        return frame, []

    def _predict_yolo(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        try:
            import torch
            use_half = torch.cuda.is_available()
        except Exception:
            use_half = False

        results = self.current_model.predict(
            frame, imgsz=640, conf=self.conf_threshold, verbose=False, half=use_half,
        )
        result = results[0]
        annotated = frame.copy()
        h, w = frame.shape[:2]
        detections = []

        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                detections.append({
                    "class_id": int(box.cls.item()),
                    "class_name": result.names.get(int(box.cls.item()), "unknown"),
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
                    "has_mask": result.masks is not None and i < len(result.masks),
                    "measurements": [],
                    "idx": i,
                })

        # 4. Rendering
        for det_idx, det in enumerate(detections):
            cls_name = det.get("class_name", "unknown")
            color = CLASS_COLORS.get(cls_name.lower(), DEFAULT_COLOR)
            
            # Draw Box & Label
            x1, y1, x2, y2 = det["bbox"]
            if self.draw_boxes:
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            if self.draw_labels:
                label = f"{cls_name} {det['confidence']:.2f}"
                cv2.putText(annotated, label, (int(x1), max(15, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Mask Processing
            if result.masks is not None:
                orig_idx = det["idx"]
                if not det["has_mask"]:
                    continue
                mask_data = result.masks.data[orig_idx]
                m = mask_data.cpu().numpy()
                m_resized = cv2.resize(m, (w, h))
                m_bin = (m_resized > 0.5).astype(np.uint8) * 255
                
                if self.draw_masks:
                    color_layer = np.zeros_like(frame)
                    color_layer[m_bin > 127] = color
                    annotated = cv2.addWeighted(annotated, 1.0, color_layer, 0.35, 0)

        # Cleanup
        for det in detections:
            det.pop("idx", None)
            det.pop("has_mask", None)

        return annotated, detections

    def _predict_pytorch(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        import torch
        import torch.nn.functional as F

        h, w = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)

        if torch.cuda.is_available():
            tensor = tensor.to("cuda").half()

        with torch.no_grad():
            output = self.current_model(tensor)

        if isinstance(output, dict):
            output = output.get("out", list(output.values())[0])

        if output.shape[1] > 1:
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
            overlay = frame.copy()
            mask_resized = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

            detections = []
            for cls_id in range(1, output.shape[1]):
                cls_mask = (mask_resized == cls_id).astype(np.uint8) * 255
                if cls_mask.sum() < 100:
                    continue
                color = CLASS_COLORS.get("chignon", DEFAULT_COLOR)
                color_layer = np.zeros_like(frame)
                color_layer[cls_mask > 127] = color
                overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.35, 0)

                cnts, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    if cv2.contourArea(cnt) > 50:
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        detections.append({
                            "class_id": cls_id,
                            "class_name": "chignon",
                            "confidence": 0.99,
                            "bbox": [x, y, x + bw, y + bh],
                            "measurements": [],
                        })
                        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)
                        cv2.putText(overlay, "chignon", (x, max(15, y - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            return overlay, detections

        # Single-channel fallback
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask_resized = cv2.resize(mask, (w, h))
        m_bin = (mask_resized > 0.5).astype(np.uint8) * 255
        overlay = frame.copy()
        color = CLASS_COLORS.get("chignon", DEFAULT_COLOR)
        color_layer = np.zeros_like(frame)
        color_layer[m_bin > 127] = color
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.35, 0)

        detections = []
        cnts, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) > 50:
                x, y, bw, bh = cv2.boundingRect(cnt)
                detections.append({
                    "class_id": 1,
                    "class_name": "chignon",
                    "confidence": 0.99,
                    "bbox": [x, y, x + bw, y + bh],
                    "measurements": [],
                })
        return overlay, detections

    # ──────────────────────────────────────────────────────────────────
    # API helpers
    # ──────────────────────────────────────────────────────────────────

    def get_models_info(self) -> List[Dict]:
        models = []
        for name, info in self.available_models.items():
            models.append({
                "name": name,
                "display_name": info.get("display_name", name),
                "type": info["type"],
                "source": info["source"],
                "active": name == self.current_model_name,
            })
        return models

    @property
    def fps(self) -> float:
        return self._frame_fps
