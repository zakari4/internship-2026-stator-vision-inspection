"""
Model inference engine for the WebRTC application.

Discovers trained and pretrained model weights from the project,
loads them on demand, and runs segmentation inference on video frames.
Optionally performs post-processing measurements using camera calibration.
"""

import os
import sys
import time
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Resolve project root — can be overridden via environment variable
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# In Docker, models are copied to /server/models/
MODEL_BASE_DIR = Path(os.environ.get("MODEL_DIR", str(PROJECT_ROOT)))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Camera Settings & Calibration (pixel ↔ mm conversion)
# ═══════════════════════════════════════════════════════════════════════════

class CameraSettings:
    """
    Calibration parameters for pixel-to-mm conversion.

    Supported methods:
        - camera_intrinsics : uses sensor_width, focal_length, object_distance
        - reference_label   : uses a detected label with known real-world size
        - manual            : user provides a fixed px→mm factor
    """

    VALID_METHODS = ("camera_intrinsics", "reference_label", "manual")
    VALID_DIM_TYPES = ("diameter", "width", "height")

    def __init__(self):
        # General
        self.enabled: bool = False
        self.method: str = "camera_intrinsics"

        # Camera-intrinsics parameters
        self.sensor_width_mm: float = 6.17
        self.focal_length_mm: float = 4.0
        self.object_distance_mm: float = 300.0

        # Reference-label parameters
        self.reference_label_name: str = ""
        self.reference_known_dimension_mm: float = 10.0
        self.reference_dimension_type: str = "diameter"  # diameter | width | height

        # Manual factor
        self.manual_px_to_mm: float = 0.1

    # -- px→mm helpers ---------------------------------------------------

    def pixel_to_mm_intrinsics(self, image_width_px: int) -> float:
        """Compute mm-per-pixel from camera intrinsics."""
        if self.focal_length_mm < 1e-6 or image_width_px < 1:
            return 1.0
        return (self.object_distance_mm * self.sensor_width_mm) / (
            self.focal_length_mm * image_width_px
        )

    def pixel_to_mm(self, image_width_px: int) -> float:
        """Shortcut used when method == camera_intrinsics."""
        return self.pixel_to_mm_intrinsics(image_width_px)

    @staticmethod
    def pixel_to_mm_from_reference(
        contour: np.ndarray,
        known_dimension_mm: float,
        dimension_type: str = "diameter",
    ) -> float:
        """Compute mm-per-pixel from a reference contour with a known size.

        ``dimension_type`` is one of ``diameter``, ``width``, ``height``.
        """
        if contour is None or len(contour) < 5:
            return 1.0

        if dimension_type == "diameter":
            (_, _), radius = cv2.minEnclosingCircle(contour)
            pixel_dim = 2.0 * radius
        elif dimension_type == "width":
            _, _, w, _ = cv2.boundingRect(contour)
            pixel_dim = float(w)
        elif dimension_type == "height":
            _, _, _, h = cv2.boundingRect(contour)
            pixel_dim = float(h)
        else:
            pixel_dim = 0.0

        if pixel_dim < 1e-3:
            return 1.0
        return known_dimension_mm / pixel_dim

    # -- Serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "method": self.method,
            # camera intrinsics
            "sensor_width_mm": self.sensor_width_mm,
            "focal_length_mm": self.focal_length_mm,
            "object_distance_mm": self.object_distance_mm,
            # reference label
            "reference_label_name": self.reference_label_name,
            "reference_known_dimension_mm": self.reference_known_dimension_mm,
            "reference_dimension_type": self.reference_dimension_type,
            # manual
            "manual_px_to_mm": self.manual_px_to_mm,
        }

    def update(self, data: dict):
        if "enabled" in data:
            self.enabled = bool(data["enabled"])
        if "method" in data and data["method"] in self.VALID_METHODS:
            self.method = data["method"]
        # camera intrinsics
        if "sensor_width_mm" in data:
            self.sensor_width_mm = float(data["sensor_width_mm"])
        if "focal_length_mm" in data:
            self.focal_length_mm = float(data["focal_length_mm"])
        if "object_distance_mm" in data:
            self.object_distance_mm = float(data["object_distance_mm"])
        # reference label
        if "reference_label_name" in data:
            self.reference_label_name = str(data["reference_label_name"]).strip()
        if "reference_known_dimension_mm" in data:
            self.reference_known_dimension_mm = float(data["reference_known_dimension_mm"])
        if "reference_dimension_type" in data:
            dt = data["reference_dimension_type"]
            if dt in self.VALID_DIM_TYPES:
                self.reference_dimension_type = dt
        # manual
        if "manual_px_to_mm" in data:
            self.manual_px_to_mm = float(data["manual_px_to_mm"])


# ═══════════════════════════════════════════════════════════════════════════
# Post-processing: measurements on segmentation masks
# ═══════════════════════════════════════════════════════════════════════════

def compute_contour_measurements(
    contours: List[np.ndarray],
    mask: np.ndarray,
    px_to_mm: float,
) -> List[dict]:
    """
    Compute geometric measurements on extracted contours.

    Returns a list of measurement dicts, each with:
      type, value_px, value_mm, unit, contour_idx, points (for drawing)
    """
    measurements = []

    for i, cnt in enumerate(contours):
        if len(cnt) < 5:
            continue

        # -- Bounding rect (width, height) --
        x, y, w, h = cv2.boundingRect(cnt)
        measurements.append({
            "type": "bounding_width",
            "label": f"Width #{i}",
            "value_px": round(float(w), 1),
            "value_mm": round(float(w) * px_to_mm, 2),
            "contour_idx": i,
            "pt1": [int(x), int(y + h // 2)],
            "pt2": [int(x + w), int(y + h // 2)],
        })
        measurements.append({
            "type": "bounding_height",
            "label": f"Height #{i}",
            "value_px": round(float(h), 1),
            "value_mm": round(float(h) * px_to_mm, 2),
            "contour_idx": i,
            "pt1": [int(x + w // 2), int(y)],
            "pt2": [int(x + w // 2), int(y + h)],
        })

        # -- Min enclosing circle (outer diameter) --
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        diameter_px = 2 * radius
        measurements.append({
            "type": "outer_diameter",
            "label": f"Outer Diam #{i}",
            "value_px": round(float(diameter_px), 1),
            "value_mm": round(float(diameter_px) * px_to_mm, 2),
            "contour_idx": i,
            "center": [int(cx), int(cy)],
            "radius": int(radius),
        })

        # -- Area --
        area_px = cv2.contourArea(cnt)
        area_mm2 = area_px * (px_to_mm ** 2)
        measurements.append({
            "type": "area",
            "label": f"Area #{i}",
            "value_px": round(float(area_px), 1),
            "value_mm": round(float(area_mm2), 2),
            "contour_idx": i,
        })

        # -- Perimeter --
        perimeter_px = cv2.arcLength(cnt, True)
        measurements.append({
            "type": "perimeter",
            "label": f"Perimeter #{i}",
            "value_px": round(float(perimeter_px), 1),
            "value_mm": round(float(perimeter_px) * px_to_mm, 2),
            "contour_idx": i,
        })

    # -- Distances between contour pairs --
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            if len(contours[i]) < 2 or len(contours[j]) < 2:
                continue
            pts_a = contours[i].reshape(-1, 2).astype(np.float64)
            pts_b = contours[j].reshape(-1, 2).astype(np.float64)
            # Brute-force min distance (fast enough for < ~5k points)
            from scipy.spatial.distance import cdist
            dists = cdist(pts_a, pts_b)
            min_dist = float(np.min(dists))
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            pt_a = pts_a[idx[0]].astype(int).tolist()
            pt_b = pts_b[idx[1]].astype(int).tolist()
            measurements.append({
                "type": "min_distance",
                "label": f"Dist #{i}↔#{j}",
                "value_px": round(min_dist, 1),
                "value_mm": round(min_dist * px_to_mm, 2),
                "contour_idx": [i, j],
                "pt1": pt_a,
                "pt2": pt_b,
            })

    return measurements


def draw_measurements_on_image(
    image: np.ndarray,
    measurements: List[dict],
    px_to_mm: float,
) -> np.ndarray:
    """Draw measurement annotations (lines, text) on the image."""
    overlay = image.copy()
    color_line = (255, 200, 0)     # Cyan-ish
    color_text = (255, 255, 255)   # White
    color_circle = (0, 200, 255)   # Orange

    for m in measurements:
        unit_val = m["value_mm"] if px_to_mm != 1.0 else m["value_px"]
        unit_str = "mm" if px_to_mm != 1.0 else "px"

        # Measurements with area/perimeter — just text, no line
        if m["type"] in ("area", "perimeter"):
            continue

        if m["type"] == "outer_diameter" and "center" in m:
            cx, cy = m["center"]
            r = m["radius"]
            cv2.circle(overlay, (cx, cy), r, color_circle, 1, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), 3, color_circle, -1)
            text = f"{unit_val:.1f}{unit_str}"
            cv2.putText(overlay, text, (cx + r + 5, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_text, 1, cv2.LINE_AA)
            continue

        # Line-based measurements
        pt1 = m.get("pt1")
        pt2 = m.get("pt2")
        if pt1 and pt2:
            cv2.line(overlay, tuple(pt1), tuple(pt2), color_line, 1, cv2.LINE_AA)
            cv2.circle(overlay, tuple(pt1), 3, color_line, -1)
            cv2.circle(overlay, tuple(pt2), 3, color_line, -1)
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            text = f"{unit_val:.1f}{unit_str}"
            cv2.putText(overlay, text, (mid_x + 5, mid_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_text, 1, cv2.LINE_AA)

    return overlay


class ModelManager:
    """
    Manages discovering, loading, and switching between
    detection/segmentation models for real-time inference.
    """

    def __init__(self):
        self.current_model = None
        self.current_model_name: Optional[str] = None
        self.available_models: Dict[str, dict] = {}
        self.camera_settings = CameraSettings()
        self._discover_models()

        # Auto-load the best model if available
        self._autoload_best()

    # ------------------------------------------------------------------
    # Model Discovery
    # ------------------------------------------------------------------

    # Map checkpoint folder names → human-readable display names + model arch id
    _PYTORCH_MODEL_MAP = {
        "segformer_b0":       {"display": "SegFormer B0",          "arch": "segformer_b0"},
        "deeplabv3_mobilenet": {"display": "DeepLabV3 MobileNet",  "arch": "deeplabv3_mobilenet"},
        "unet_lightweight":   {"display": "UNet Lightweight",      "arch": "unet_lightweight"},
        "unet_resnet18":      {"display": "UNet ResNet18",         "arch": "unet_resnet18"},
        "hed":                {"display": "HED",                   "arch": "hed"},
        "rcf":                {"display": "RCF",                   "arch": "rcf"},
        "pidinet":            {"display": "PiDiNet",               "arch": "pidinet"},
        "teed":               {"display": "TEED",                  "arch": "teed"},
    }

    # Map YOLO folder names → human-readable display names
    _YOLO_DISPLAY_NAMES = {
        "yolov8n_seg":  "YOLOv8 Nano Seg",
        "yolov8s_seg":  "YOLOv8 Small Seg",
        "yolov8m_seg":  "YOLOv8 Medium Seg",
        "yolov11n_seg": "YOLOv11 Nano Seg",
        "yolov11s_seg": "YOLOv11 Small Seg",
        "yolov11m_seg": "YOLOv11 Medium Seg",
        "yolov26n_seg": "YOLOv26 Nano Seg",
        "yolov26s_seg": "YOLOv26 Small Seg",
        "yolov26m_seg": "YOLOv26 Medium Seg",
        # Pretrained weight file stems → display names
        "yolov8n-seg":  "YOLOv8 Nano Seg (pretrained)",
        "yolov8s-seg":  "YOLOv8 Small Seg (pretrained)",
        "yolov8m-seg":  "YOLOv8 Medium Seg (pretrained)",
        "yolo11n-seg":  "YOLOv11 Nano Seg (pretrained)",
        "yolo11s-seg":  "YOLOv11 Small Seg (pretrained)",
        "yolo11m-seg":  "YOLOv11 Medium Seg (pretrained)",
        "yolo26n-seg":  "YOLOv26 Nano Seg (pretrained)",
        "yolo26s-seg":  "YOLOv26 Small Seg (pretrained)",
        "yolo26m-seg":  "YOLOv26 Medium Seg (pretrained)",
        "rtdetr-l":     "RT-DETR Large (pretrained)",
        "rtdetr-x":     "RT-DETR XLarge (pretrained)",
    }

    def _discover_models(self):
        """Auto-discover trained model weights from the outputs folder only."""

        # 1. Trained YOLO weights  (outputs/results/yolo_training/*/weights/best.pt)
        yolo_dir = MODEL_BASE_DIR / "outputs" / "results" / "yolo_training"
        if yolo_dir.exists():
            for model_dir in sorted(yolo_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                best_pt = model_dir / "weights" / "best.pt"
                if best_pt.exists():
                    folder = model_dir.name
                    display = self._YOLO_DISPLAY_NAMES.get(folder, folder)
                    self.available_models[folder] = {
                        "path": str(best_pt),
                        "type": "yolo",
                        "source": "trained",
                        "display_name": display,
                    }

        # 2. Pretrained YOLO weights in weights/ directory
        weights_dir = MODEL_BASE_DIR / "weights"
        if weights_dir.exists():
            for pt_file in sorted(weights_dir.glob("*.pt")):
                name = pt_file.stem          # e.g. "yolov8n-seg"
                display = self._YOLO_DISPLAY_NAMES.get(name, name)
                self.available_models[name] = {
                    "path": str(pt_file),
                    "type": "yolo",
                    "source": "pretrained",
                    "display_name": display,
                }

        # 3. Root-level .pt files (rtdetr-l.pt, rtdetr-x.pt)
        for pt_file in sorted(MODEL_BASE_DIR.glob("*.pt")):
            name = pt_file.stem
            if name not in self.available_models:
                display = self._YOLO_DISPLAY_NAMES.get(name, name)
                self.available_models[name] = {
                    "path": str(pt_file),
                    "type": "yolo",
                    "source": "pretrained",
                    "display_name": display,
                }

        # 4. Trained PyTorch checkpoints (outputs/results/checkpoints/*/best_model.pth)
        ckpt_dir = MODEL_BASE_DIR / "outputs" / "results" / "checkpoints"
        if ckpt_dir.exists():
            for model_dir in sorted(ckpt_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                # Support both .pth and .pt extensions
                best_pth = model_dir / "best_model.pth"
                best_pt = model_dir / "best_model.pt"
                weight_file = best_pth if best_pth.exists() else (best_pt if best_pt.exists() else None)
                if weight_file is None:
                    continue

                folder = model_dir.name
                meta = self._PYTORCH_MODEL_MAP.get(folder, {})
                display = meta.get("display", folder)
                arch = meta.get("arch", folder)

                self.available_models[folder] = {
                    "path": str(weight_file),
                    "type": "pytorch",
                    "source": "trained",
                    "display_name": display,
                    "arch": arch,
                }

        logger.info(
            "Discovered %d model(s): %s",
            len(self.available_models),
            list(self.available_models.keys()),
        )

    def _autoload_best(self):
        """Try to load the best trained model automatically."""
        # Priority order: yolov26n_seg > yolov8n_seg > first available trained
        priority = [
            "yolov26n_seg",
            "yolov8n_seg",
            "yolov26s_seg",
            "yolov8s_seg",
            "yolov11n_seg",
        ]

        for name in priority:
            if name in self.available_models:
                logger.info("Auto-loading best model: %s", name)
                if self.load_model(name):
                    return

        # Fallback: first trained model
        for name, info in self.available_models.items():
            if info["source"] == "trained":
                logger.info("Auto-loading first trained model: %s", name)
                if self.load_model(name):
                    return

        logger.warning("No model could be auto-loaded.")

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load_model(self, model_name: str) -> bool:
        """Load a model by name. Returns True on success."""
        if model_name not in self.available_models:
            logger.error("Model '%s' not found in available models.", model_name)
            return False

        info = self.available_models[model_name]
        model_path = info["path"]

        # Don't reload if already loaded
        if model_name == self.current_model_name and self.current_model is not None:
            return True

        try:
            if info["type"] == "yolo":
                from ultralytics import YOLO

                self.current_model = YOLO(model_path)
                logger.info("Loaded YOLO model: %s from %s", model_name, model_path)

            elif info["type"] == "pytorch":
                import torch

                arch = info.get("arch", model_name)
                model = self._build_pytorch_model(arch)

                if model is None:
                    logger.error(
                        "No architecture builder for '%s'. Cannot load checkpoint.", arch
                    )
                    return False

                checkpoint = torch.load(model_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    logger.info(
                        "Loaded state_dict (epoch %s, loss %.4f) for %s",
                        checkpoint.get("epoch", "?"),
                        checkpoint.get("best_val_loss", float("nan")),
                        model_name,
                    )
                else:
                    # Full model object
                    model = checkpoint

                model.eval()
                if torch.cuda.is_available():
                    model.to("cuda")

                self.current_model = model
                logger.info(
                    "Loaded PyTorch model: %s from %s", model_name, model_path
                )
            else:
                logger.error("Unknown model type: %s", info["type"])
                return False

            self.current_model_name = model_name
            return True

        except Exception:
            logger.exception("Failed to load model %s", model_name)
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Run inference on a BGR video frame.

        Returns:
            (annotated_frame, detections_list)
            Each detection may include a "measurements" key when camera settings are enabled.
        """
        if self.current_model is None:
            return frame, []

        info = self.available_models.get(self.current_model_name, {})

        try:
            if info.get("type") == "yolo":
                return self._predict_yolo(frame)
            elif info.get("type") == "pytorch":
                return self._predict_pytorch(frame)
            else:
                return frame, []
        except Exception:
            logger.exception("Inference failed")
            return frame, []

    # ------------------------------------------------------------------
    # Measurement helpers
    # ------------------------------------------------------------------

    def _get_px_to_mm(
        self,
        image_width: int,
        detection_masks: Optional[List[Tuple[str, np.ndarray]]] = None,
    ) -> float:
        """Compute mm-per-pixel factor based on the active method.

        Args:
            image_width: Width of the image in pixels.
            detection_masks: List of (class_name, binary_mask) tuples –
                only needed for the ``reference_label`` method.

        Returns:
            mm-per-pixel factor (1.0 when measurements are disabled).
        """
        cs = self.camera_settings
        if not cs.enabled:
            return 1.0

        if cs.method == "manual":
            return cs.manual_px_to_mm if cs.manual_px_to_mm > 0 else 1.0

        if cs.method == "reference_label":
            return self._px_to_mm_from_detections(detection_masks)

        # Default: camera_intrinsics
        return cs.pixel_to_mm_intrinsics(image_width)

    def _px_to_mm_from_detections(
        self,
        detection_masks: Optional[List[Tuple[str, np.ndarray]]] = None,
    ) -> float:
        """Find the reference label among detections and calibrate."""
        cs = self.camera_settings
        ref_name = cs.reference_label_name.lower()
        if not ref_name or detection_masks is None:
            logger.warning("Reference calibration: no reference label or masks provided – falling back to 1.0")
            return 1.0

        # Find the best (largest) contour for the reference label
        best_contour = None
        best_area = 0.0
        for class_name, mask in detection_masks:
            if class_name.lower() != ref_name:
                continue
            m = (mask > 127).astype(np.uint8) * 255 if mask.dtype != np.uint8 else mask
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a = cv2.contourArea(c)
                if a > best_area:
                    best_area = a
                    best_contour = c

        if best_contour is None:
            logger.warning("Reference calibration: label '%s' not found in detections", cs.reference_label_name)
            return 1.0

        return CameraSettings.pixel_to_mm_from_reference(
            best_contour,
            cs.reference_known_dimension_mm,
            cs.reference_dimension_type,
        )

    def _postprocess_measurements(
        self,
        annotated: np.ndarray,
        mask: np.ndarray,
        detection_masks: Optional[List[Tuple[str, np.ndarray]]] = None,
    ) -> Tuple[np.ndarray, List[dict]]:
        """Extract contours from mask, compute measurements, and draw them.

        Args:
            annotated: Image to draw on.
            mask: Combined binary mask (all classes merged).
            detection_masks: Per-detection (class_name, mask) list for
                reference-based calibration.
        """
        h, w = mask.shape[:2]
        px_to_mm = self._get_px_to_mm(w, detection_masks)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Filter tiny contours
        contours = [c for c in contours if cv2.contourArea(c) > 50]

        if not contours:
            return annotated, []

        # Draw bold contour borders on the annotated image
        cv2.drawContours(annotated, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

        measurements = []
        if self.camera_settings.enabled:
            measurements = compute_contour_measurements(contours, mask, px_to_mm)
            annotated = draw_measurements_on_image(annotated, measurements, px_to_mm)

        return annotated, measurements

    def _predict_yolo(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """YOLO-based inference with segmentation mask overlay + contour borders."""
        results = self.current_model.predict(
            frame, imgsz=512, conf=0.25, verbose=False
        )
        result = results[0]
        annotated = result.plot()
        h, w = frame.shape[:2]

        detections = []
        all_measurements = []

        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                det = {
                    "class_id": int(box.cls.item()),
                    "class_name": result.names.get(int(box.cls.item()), "unknown"),
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
                    "has_mask": (
                        result.masks is not None and i < len(result.masks)
                    ),
                }
                detections.append(det)

        # Draw contour borders on each mask + measure
        if result.masks is not None:
            combined_mask = np.zeros((h, w), dtype=np.uint8)

            # Build per-detection (class_name, mask) pairs for reference calibration
            detection_masks: List[Tuple[str, np.ndarray]] = []
            for i, mask_data in enumerate(result.masks.data):
                m = mask_data.cpu().numpy()
                m_resized = cv2.resize(m, (w, h))
                m_bin = (m_resized > 0.5).astype(np.uint8) * 255
                combined_mask[m_bin > 0] = 255
                # Map mask to its class name via matching box index
                cls_name = "unknown"
                if result.boxes is not None and i < len(result.boxes):
                    cls_id = int(result.boxes[i].cls.item())
                    cls_name = result.names.get(cls_id, "unknown")
                detection_masks.append((cls_name, m_bin))

            annotated, all_measurements = self._postprocess_measurements(
                annotated, combined_mask, detection_masks
            )

        # Attach measurements to response
        if all_measurements:
            for det in detections:
                det["measurements"] = all_measurements

        return annotated, detections

    def _predict_pytorch(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        PyTorch model inference (UNet, DeepLab, etc.).
        Produces a binary segmentation mask overlay.
        """
        import torch

        model = self.current_model
        if not hasattr(model, "forward"):
            return frame, []

        device = next(
            (p.device for p in model.parameters()), torch.device("cpu")
        )

        # Preprocess: resize, normalize, to tensor
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (512, 512))
        img_float = img.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std
        tensor = (
            torch.from_numpy(img_norm)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        with torch.no_grad():
            output = model(tensor)

        # Handle OrderedDict output (e.g. DeepLabV3)
        if isinstance(output, dict):
            output = output.get("out", list(output.values())[0])

        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]

        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (w, h))

        # Find contours for overlay
        contours, _ = cv2.findContours(
            mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create overlay
        overlay = frame.copy()
        green_mask = np.zeros_like(frame)
        green_mask[mask_resized > 127] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.6, green_mask, 0.4, 0)

        # Draw contour borders (always)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)

        # Optional measurements via post-processing
        overlay, measurements = self._postprocess_measurements(overlay, mask_resized)

        coverage = float(np.count_nonzero(mask_resized) / mask_resized.size)
        detections = [
            {
                "class_id": 1,
                "class_name": "foreground",
                "confidence": round(float(mask[mask > 0.5].mean()) if np.any(mask > 0.5) else 0, 4),
                "has_mask": True,
                "mask_coverage": round(coverage, 4),
            }
        ]
        if measurements:
            detections[0]["measurements"] = measurements

        return overlay, detections

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # PyTorch model architecture builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pytorch_model(arch: str):
        """
        Instantiate an empty PyTorch model by architecture name.

        The model classes live in ``src.models.deep_learning``.
        Returns ``None`` if the architecture is unknown.
        """
        try:
            from src.models.deep_learning import (
                SegFormerB0Simple,
                UNetLightweight,
                UNetResNet18,
                HEDNet,
                RCFNet,
                PiDiNet,
                TEEDNet,
            )
            from torchvision import models as tv_models
            import torch.nn as nn
        except ImportError as exc:
            logger.error("Cannot import model classes: %s", exc)
            return None

        builders = {
            "segformer_b0":       lambda: SegFormerB0Simple(n_classes=1),
            "unet_lightweight":   lambda: UNetLightweight(n_channels=3, n_classes=1, base_filters=32),
            "unet_resnet18":      lambda: UNetResNet18(n_classes=1, pretrained=False),
            "hed":                lambda: HEDNet(),
            "rcf":                lambda: RCFNet(),
            "pidinet":            lambda: PiDiNet(),
            "teed":               lambda: TEEDNet(),
        }

        # DeepLabV3 MobileNet needs special handling (torchvision + head swap)
        def _build_deeplabv3():
            model = tv_models.segmentation.deeplabv3_mobilenet_v3_large(
                weights=None, progress=False
            )
            model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
            model.aux_classifier = None
            return model

        builders["deeplabv3_mobilenet"] = _build_deeplabv3

        builder = builders.get(arch)
        if builder is None:
            return None
        return builder()

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_models_info(self) -> List[Dict]:
        """Return list of available models with metadata."""
        models = []
        for name, info in self.available_models.items():
            models.append(
                {
                    "name": name,
                    "display_name": info.get("display_name", name),
                    "type": info["type"],
                    "source": info["source"],
                    "active": name == self.current_model_name,
                }
            )
        return models

    def get_class_labels(self) -> List[str]:
        """Return class label names for the currently loaded model.

        For YOLO / RT-DETR models the label list comes from ``model.names``.
        For generic PyTorch models the only class is *foreground*.
        """
        if self.current_model is None:
            return []

        info = self.available_models.get(self.current_model_name, {})
        if info.get("type") == "yolo":
            names = getattr(self.current_model, "names", None)
            if isinstance(names, dict):
                return list(names.values())
            if isinstance(names, (list, tuple)):
                return list(names)
            return []

        # PyTorch single-class segmentation
        return ["foreground"]
