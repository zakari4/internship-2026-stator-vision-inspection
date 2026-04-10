"""
File detection inference engine.

ResNet/UNet-only file-domain manager that discovers checkpoints from
files/results/checkpoints and runs segmentation inference on images.
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

FILE_RESULTS_DIR = PROJECT_ROOT / "files" / "results"

logger = logging.getLogger(__name__)


# ─── Color palette ────────────────────────────────────────────────────────
CLASS_COLORS = {
    "file": (0, 255, 127),      # Spring Green
    "background": (50, 50, 50),
}
DEFAULT_COLOR = (0, 255, 127)


# ─── Display name mappings ────────────────────────────────────────────────
_PYTORCH_META = {
    "unet_resnet18": {"display": "UNet ResNet18 (File)", "arch": "unet_resnet18"},
}

CLASS_ID_TO_NAME = {
    1: "file",
    2: "file",
}


class FileModelManager:
    """Lightweight model manager specifically for the file domain."""

    def __init__(self):
        self.current_model = None
        self.current_model_name: Optional[str] = None
        self.available_models: Dict[str, dict] = {}

        # Global Post-processing Settings (Domain Specific)
        self.enable_postprocessing = True
        self.enable_file_color_validation = True
        self.enable_top_n = True
        self.draw_boxes = True
        self.draw_masks = True
        self.draw_labels = True
        self.conf_threshold = 0.25
        self.min_component_area = 80
        self._latest_alerts: List[Dict] = []
        self._latest_position_message: str = ""

        self._frame_fps = 0.0
        self._fps_frame_count = 0
        self._fps_window_start = time.monotonic()

        self._discover_models()
        self._autoload_best()

    # ──────────────────────────────────────────────────────────────────
    # Discovery
    # ──────────────────────────────────────────────────────────────────

    def _discover_models(self):
        """Scan files/results/checkpoints for trained PyTorch weights only."""
        self.available_models.clear()

        # PyTorch checkpoints
        ckpt_dir = FILE_RESULTS_DIR / "checkpoints"
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
            "[File] Discovered %d model(s): %s",
            len(self.available_models),
            list(self.available_models.keys()),
        )

    def _autoload_best(self):
        """Auto-load the first available trained model."""
        for name, info in self.available_models.items():
            if info["source"] == "trained":
                logger.info("[File] Auto-loading: %s", name)
                if self.load_model(name):
                    return
        logger.warning("[File] No model could be auto-loaded.")

    # ──────────────────────────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────────────────────────

    def load_model(self, model_name: str) -> bool:
        if model_name not in self.available_models:
            logger.error("[File] Model '%s' not found.", model_name)
            return False

        info = self.available_models[model_name]
        model_path = info["path"]

        if model_name == self.current_model_name and self.current_model is not None:
            return True

        try:
            if info["type"] == "pytorch":
                import torch
                arch = info.get("arch", model_name)
                model = self._build_pytorch(arch)
                if model is None:
                    return False

                ckpt = torch.load(model_path, map_location="cpu")
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                else:
                    state_dict = ckpt

                if isinstance(state_dict, dict) and state_dict and all(
                    str(k).startswith("module.") for k in state_dict.keys()
                ):
                    state_dict = {str(k).replace("module.", "", 1): v for k, v in state_dict.items()}

                try:
                    model.load_state_dict(state_dict, strict=True)
                except Exception:
                    model.load_state_dict(state_dict, strict=False)

                model.eval()
                if torch.cuda.is_available():
                    model.to("cuda")
                self.current_model = model
                logger.info("[File] Loaded PyTorch: %s", model_path)
            else:
                return False

            self.current_model_name = model_name
            return True
        except Exception:
            logger.exception("[File] Failed to load %s", model_name)
            return False

    @staticmethod
    def _build_pytorch(arch: str):
        """Build a PyTorch model architecture by name."""
        if arch == "unet_resnet18":
            try:
                from src.models.deep_learning import UNetResNet18
                return UNetResNet18(n_classes=2, pretrained=False)
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
            if info.get("type") == "pytorch":
                return self._predict_pytorch(frame)
        except Exception:
            logger.exception("[File] Inference failed")
        return frame, []

    def _predict_pytorch(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        annotated = frame.copy()
        h, w = frame.shape[:2]

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(device)

        with torch.no_grad():
            output = self.current_model(tensor)

        if isinstance(output, (dict,)):
            output = output.get("out", list(output.values())[0])
        if isinstance(output, (tuple, list)):
            output = output[0]

        if output.shape[1] == 1:
            pred = (torch.sigmoid(output) > 0.5).long().squeeze(0).squeeze(0).cpu().numpy()
        else:
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

        pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        detections = []

        for cls_id in sorted(int(v) for v in np.unique(pred) if int(v) > 0):
            cls_mask = (pred == cls_id).astype(np.uint8) * 255

            if self.draw_masks:
                color = CLASS_COLORS.get(CLASS_ID_TO_NAME.get(cls_id, "file"), DEFAULT_COLOR)
                color_layer = np.zeros_like(frame)
                color_layer[cls_mask > 127] = color
                annotated = cv2.addWeighted(annotated, 1.0, color_layer, 0.35, 0)

            cnts, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < self.min_component_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                class_name = CLASS_ID_TO_NAME.get(cls_id, "file")
                color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

                if self.draw_boxes:
                    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
                if self.draw_labels:
                    cv2.putText(
                        annotated,
                        class_name,
                        (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": 0.99,
                    "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
                    "measurements": [],
                })

        if self.enable_postprocessing and self.enable_file_color_validation:
            annotated, detections = self._postprocess_file_layout(frame, annotated, detections)

        return annotated, detections

    def _postprocess_file_layout(
        self,
        original_frame: np.ndarray,
        annotated: np.ndarray,
        detections: List[Dict],
    ) -> Tuple[np.ndarray, List[Dict]]:
        self._latest_alerts = []
        self._latest_position_message = ""

        hsv = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV)

        # User-defined HSV strategy:
        # - Blue: hue in [100, 130]
        # - Yellow: hue in [20, 40]
        # The same bbox may contain both colors; when this happens,
        # split it into two smaller boxes from color-mask contours.
        blue_l, blue_u = np.array([100, 50, 50]), np.array([130, 255, 255])
        yellow_l, yellow_u = np.array([20, 50, 50]), np.array([40, 255, 255])

        color_pixel_ratio_threshold = 0.08
        min_color_component_area = max(40, int(self.min_component_area * 0.35))

        processed_detections: List[Dict] = []

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(hsv.shape[1], x2), min(hsv.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                det["file_color"] = "unknown"
                det["color_ratio"] = 0.0
                det["color_scores"] = {"blue": 0.0, "yellow": 0.0}
                processed_detections.append(det)
                continue

            roi_hsv = hsv[y1:y2, x1:x2]
            total = float(roi_hsv.shape[0] * roi_hsv.shape[1])
            if total <= 0:
                det["file_color"] = "unknown"
                det["color_ratio"] = 0.0
                det["color_scores"] = {"blue": 0.0, "yellow": 0.0}
                processed_detections.append(det)
                continue

            blue_mask = cv2.inRange(roi_hsv, blue_l, blue_u)
            yellow_mask = cv2.inRange(roi_hsv, yellow_l, yellow_u)

            # Reduce isolated noise before counting/contours.
            kernel = np.ones((3, 3), np.uint8)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

            blue_ratio = float(cv2.countNonZero(blue_mask)) / total
            yellow_ratio = float(cv2.countNonZero(yellow_mask)) / total

            has_blue = blue_ratio >= color_pixel_ratio_threshold
            has_yellow = yellow_ratio >= color_pixel_ratio_threshold

            # Fusion case: both colors inside one detection box.
            # Split by finding contours in each color mask and deriving
            # new smaller boxes in image coordinates.
            if has_blue and has_yellow:
                split_added = 0
                for color_name, color_mask in (("blue", blue_mask), ("yellow", yellow_mask)):
                    cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not cnts:
                        continue

                    cnt = max(cnts, key=cv2.contourArea)
                    area = cv2.contourArea(cnt)
                    if area < min_color_component_area:
                        continue

                    sx, sy, sw, sh = cv2.boundingRect(cnt)
                    abs_x1, abs_y1 = x1 + sx, y1 + sy
                    abs_x2, abs_y2 = abs_x1 + sw, abs_y1 + sh

                    split_det = dict(det)
                    split_det["bbox"] = [float(abs_x1), float(abs_y1), float(abs_x2), float(abs_y2)]
                    split_det["file_color"] = color_name
                    split_det["color_ratio"] = round(
                        float(blue_ratio if color_name == "blue" else yellow_ratio),
                        3,
                    )
                    split_det["color_scores"] = {
                        "blue": round(float(blue_ratio), 3),
                        "yellow": round(float(yellow_ratio), 3),
                    }
                    processed_detections.append(split_det)
                    split_added += 1

                    if self.draw_boxes:
                        box_color = (255, 0, 0) if color_name == "blue" else (0, 255, 255)
                        cv2.rectangle(annotated, (abs_x1, abs_y1), (abs_x2, abs_y2), box_color, 2)
                    if self.draw_labels:
                        ty = max(15, abs_y1 - 10)
                        cv2.putText(
                            annotated,
                            color_name,
                            (abs_x1, ty),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                if split_added > 0:
                    continue

            if has_blue and not has_yellow:
                color_name = "blue"
                ratio = blue_ratio
            elif has_yellow and not has_blue:
                color_name = "yellow"
                ratio = yellow_ratio
            elif has_blue and has_yellow:
                color_name = "blue" if blue_ratio >= yellow_ratio else "yellow"
                ratio = max(blue_ratio, yellow_ratio)
            else:
                color_name = "unknown"
                ratio = max(blue_ratio, yellow_ratio)

            det["file_color"] = color_name
            det["color_ratio"] = round(float(ratio), 3)
            det["color_scores"] = {
                "blue": round(float(blue_ratio), 3),
                "yellow": round(float(yellow_ratio), 3),
            }
            processed_detections.append(det)

            if self.draw_labels:
                tx, ty = int(det["bbox"][0]), int(max(15, det["bbox"][1] - 20))
                cv2.putText(
                    annotated,
                    color_name,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        detections = processed_detections

        colored = [d for d in detections if d.get("file_color") in {"blue", "yellow"}]
        if len(colored) < 2:
            msg = "Incorrect: need at least 2 color-identified files (blue/yellow)."
            self._latest_position_message = msg
            self._latest_alerts.append({"msg": msg, "level": "warn", "ts": time.time()})
            return annotated, detections

        # Evaluate arrangement using the leftmost and rightmost files.
        def center_x(det: Dict) -> float:
            x1, _, x2, _ = det["bbox"]
            return (x1 + x2) * 0.5

        ordered = sorted(colored, key=center_x)
        left_det = ordered[0]
        right_det = ordered[-1]

        left_color = left_det.get("file_color", "unknown")
        right_color = right_det.get("file_color", "unknown")

        left_ok = left_color == "blue"
        right_ok = right_color == "yellow"
        valid = left_ok and right_ok

        lx1, ly1, lx2, ly2 = left_det["bbox"]
        rx1, ry1, rx2, ry2 = right_det["bbox"]
        p1 = (int((lx1 + lx2) * 0.5), int((ly1 + ly2) * 0.5))
        p2 = (int((rx1 + rx2) * 0.5), int((ry1 + ry2) * 0.5))
        mid = (int((p1[0] + p2[0]) * 0.5), int((p1[1] + p2[1]) * 0.5))

        line_color = (0, 220, 0) if valid else (0, 0, 255)
        cv2.line(annotated, p1, p2, line_color, 2)
        cv2.circle(annotated, mid, 4, line_color, -1)

        if valid:
            msg = "Correct position: blue file is on the left and yellow file is on the right."
            level = "info"
        else:
            msg = (
                f"Incorrect position: left is {left_color}, right is {right_color}. "
                "Expected left blue and right yellow."
            )
            level = "warn"

        self._latest_position_message = msg
        self._latest_alerts.append({"msg": msg, "level": level, "ts": time.time()})
        return annotated, detections

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

    def get_latest_alerts(self) -> List[Dict]:
        alerts = list(self._latest_alerts)
        self._latest_alerts = []
        return alerts

    def get_latest_position_message(self) -> str:
        return self._latest_position_message

    @property
    def fps(self) -> float:
        return self._frame_fps
