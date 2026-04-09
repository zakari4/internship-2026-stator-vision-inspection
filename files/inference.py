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

        # HSV ranges for file-color discrimination.
        blue_l, blue_u = np.array([85, 50, 35]), np.array([140, 255, 255])
        yellow_l, yellow_u = np.array([12, 35, 45]), np.array([45, 255, 255])
        green_l, green_u = np.array([38, 40, 35]), np.array([92, 255, 255])

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(hsv.shape[1], x2), min(hsv.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                det["file_color"] = "unknown"
                det["color_ratio"] = 0.0
                continue

            roi_hsv = hsv[y1:y2, x1:x2]
            roi_bgr = original_frame[y1:y2, x1:x2]

            # Use inner crop to avoid background influence on bbox borders.
            rh, rw = roi_hsv.shape[:2]
            mx, my = int(rw * 0.1), int(rh * 0.1)
            if rw - 2 * mx > 10 and rh - 2 * my > 10:
                roi_hsv = roi_hsv[my:rh - my, mx:rw - mx]
                roi_bgr = roi_bgr[my:rh - my, mx:rw - mx]

            total = float(roi_hsv.shape[0] * roi_hsv.shape[1])
            if total <= 0:
                det["file_color"] = "unknown"
                det["color_ratio"] = 0.0
                continue

            h_ch, s_ch, v_ch = cv2.split(roi_hsv)
            valid_mask = ((s_ch > 45) & (v_ch > 40)).astype(np.uint8) * 255
            valid_total = float(cv2.countNonZero(valid_mask))
            if valid_total < 50:
                valid_mask = np.ones_like(valid_mask, dtype=np.uint8) * 255
                valid_total = total

            blue_mask = cv2.bitwise_and(cv2.inRange(roi_hsv, blue_l, blue_u), valid_mask)
            yellow_mask = cv2.bitwise_and(cv2.inRange(roi_hsv, yellow_l, yellow_u), valid_mask)
            green_mask = cv2.bitwise_and(cv2.inRange(roi_hsv, green_l, green_u), valid_mask)

            blue_hsv = float(cv2.countNonZero(blue_mask)) / max(valid_total, 1.0)
            yellow_hsv = float(cv2.countNonZero(yellow_mask)) / max(valid_total, 1.0)
            green_hsv = float(cv2.countNonZero(green_mask)) / max(valid_total, 1.0)

            b_ch, g_ch, r_ch = cv2.split(roi_bgr.astype(np.float32))
            vm = valid_mask > 0
            blue_dom = ((b_ch > 1.12 * g_ch) & (b_ch > 1.12 * r_ch) & vm)
            yellow_dom = (
                (r_ch > 70)
                & (g_ch > 70)
                & (np.abs(r_ch - g_ch) < 75)
                & (b_ch < 0.85 * np.minimum(r_ch, g_ch))
                & vm
            )
            green_dom = ((g_ch > 1.08 * r_ch) & (g_ch > 1.08 * b_ch) & vm)

            blue_bgr = float(np.count_nonzero(blue_dom)) / max(valid_total, 1.0)
            yellow_bgr = float(np.count_nonzero(yellow_dom)) / max(valid_total, 1.0)
            green_bgr = float(np.count_nonzero(green_dom)) / max(valid_total, 1.0)

            scores = {
                "blue": 0.65 * blue_hsv + 0.35 * blue_bgr,
                "yellow": 0.65 * yellow_hsv + 0.35 * yellow_bgr,
                "green": 0.65 * green_hsv + 0.35 * green_bgr,
            }
            color_name, ratio = max(scores.items(), key=lambda kv: kv[1])
            if ratio < 0.07:
                color_name = "unknown"

            det["file_color"] = color_name
            det["color_ratio"] = round(float(ratio), 3)
            det["color_scores"] = {
                "blue": round(float(scores["blue"]), 3),
                "yellow": round(float(scores["yellow"]), 3),
                "green": round(float(scores["green"]), 3),
            }

            if self.draw_labels:
                tx, ty = int(det["bbox"][0]), int(max(15, det["bbox"][1] - 20))
                cv2.putText(
                    annotated,
                    f"{color_name}",
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Pair-aware disambiguation: if both files are tagged blue but one has
        # strong yellow evidence, relabel that candidate as yellow.
        color_dets = [d for d in detections if d.get("file_color") in {"blue", "yellow"}]
        has_blue = any(d.get("file_color") == "blue" for d in color_dets)
        has_yellow = any(d.get("file_color") == "yellow" for d in color_dets)
        if len(color_dets) >= 2 and has_blue and not has_yellow:
            candidate = max(
                color_dets,
                key=lambda d: (d.get("color_scores", {}).get("yellow", 0.0), d.get("color_ratio", 0.0)),
            )
            y_score = float(candidate.get("color_scores", {}).get("yellow", 0.0))
            b_score = float(candidate.get("color_scores", {}).get("blue", 0.0))
            if y_score >= 0.20 and (b_score - y_score) <= 0.35:
                candidate["file_color"] = "yellow"
                candidate["color_ratio"] = round(y_score, 3)

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
