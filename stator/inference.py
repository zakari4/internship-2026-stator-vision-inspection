"""
Stator detection inference engine.

Discovers trained model weights, loads them on demand, and runs segmentation
inference on video frames. Optionally performs post-processing measurements
using camera calibration.

Post-processing utilities live in the sibling ``postprocessing/`` package:
    stator/postprocessing/logging.py       — InferenceLogger
    stator/postprocessing/calibration.py   — CameraSettings
    stator/postprocessing/filtering.py     — top-N & spatial heuristic filters
    stator/postprocessing/measurements.py  — all geometric measurement functions
    stator/postprocessing/drawing.py       — draw_measurements_on_image
"""

import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import cdist

# Resolve project root so that src.* and sibling packages are importable
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# In Docker, models may be copied to a different directory
MODEL_BASE_DIR = Path(os.environ.get("MODEL_DIR", str(PROJECT_ROOT)))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Post-processing imports (one file per concern)
# ---------------------------------------------------------------------------
from stator.postprocessing import (
    InferenceLogger,
    CameraSettings,
    apply_top_n_filtering,
    apply_spatial_heuristic_correction,
    normalize_measurement_family,
    compute_contour_measurements,
    compute_single_contour_measurements,
    compute_edge_center_points,
    compute_edge_center_distances,
    compute_aligned_same_class_distances,
    compute_cross_diametric_opposite_distances,
    draw_measurements_on_image,
)


# ═══════════════════════════════════════════════════════════════════════════
# Model Manager
# ═══════════════════════════════════════════════════════════════════════════

class ModelManager:
    """
    Manages discovering, loading, and switching between
    detection/segmentation models for real-time stator inference.
    """

    # Map checkpoint folder names → human-readable display names + model arch id
    _PYTORCH_MODEL_MAP = {
        "segformer_b0":        {"display": "SegFormer B0",         "arch": "segformer_b0"},
        "deeplabv3_mobilenet": {"display": "DeepLabV3 MobileNet",  "arch": "deeplabv3_mobilenet"},
        "unet_lightweight":    {"display": "UNet Lightweight",     "arch": "unet_lightweight"},
        "unet_resnet18":       {"display": "UNet ResNet18",        "arch": "unet_resnet18"},
        "hed":                 {"display": "HED",                  "arch": "hed"},
        "rcf":                 {"display": "RCF",                  "arch": "rcf"},
        "pidinet":             {"display": "PiDiNet",              "arch": "pidinet"},
        "teed":                {"display": "TEED",                 "arch": "teed"},
    }

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

    def __init__(self):
        self.current_model = None
        self.current_model_name: Optional[str] = None
        self.available_models: Dict[str, dict] = {}
        self.current_domain = "stator"

        # Post-processing flags
        self.enable_postprocessing = True
        self.enable_heuristic = True
        self.enable_top_n = True
        self.draw_boxes = True
        self.draw_masks = True
        self.draw_labels = True
        self.conf_threshold = 0.05

        self.camera_settings = CameraSettings()

        # SOTA pipeline options
        self.enable_tracking = False
        self.enable_edge_refinement = False
        self.enable_depth_viz = False

        # Logging & validation
        self._inference_logger = InferenceLogger()
        self._latest_alerts: deque = deque(maxlen=20)
        self._frame_fps = 0.0
        self._fps_frame_count = 0
        self._fps_window_start = time.monotonic()

        self._discover_models()
        self._autoload_best()

    def set_domain(self, domain: str):
        if domain != self.current_domain:
            self.current_domain = domain
            logger.info("Switched ModelManager domain to: %s", domain)

    # ------------------------------------------------------------------
    # Model Discovery
    # ------------------------------------------------------------------

    def _discover_models(self):
        """Auto-discover trained model weights from the outputs folder."""

        # 1. Trained YOLO weights
        yolo_dir = MODEL_BASE_DIR / "outputs" / "results" / "yolo_training"
        if yolo_dir.exists():
            for model_dir in sorted(yolo_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                best_engine = model_dir / "weights" / "best.engine"
                best_onnx   = model_dir / "weights" / "best.onnx"
                best_pt     = model_dir / "weights" / "best.pt"

                weight_file = (
                    best_engine if best_engine.exists() else
                    best_onnx   if best_onnx.exists()   else
                    best_pt     if best_pt.exists()      else None
                )
                if weight_file is None:
                    continue

                folder  = model_dir.name
                display = self._YOLO_DISPLAY_NAMES.get(folder, folder)
                if weight_file.suffix == ".engine":
                    display += " (TensorRT FP16)"
                elif weight_file.suffix == ".onnx":
                    display += " (ONNX FP16)"
                else:
                    display += " (PT FP16)"

                self.available_models[folder] = {
                    "path": str(weight_file),
                    "type": "yolo",
                    "source": "trained",
                    "display_name": display,
                }

        # 2. Trained PyTorch checkpoints
        ckpt_dir = MODEL_BASE_DIR / "outputs" / "results" / "checkpoints"
        if ckpt_dir.exists():
            for model_dir in sorted(ckpt_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                best_engine = model_dir / "best_model.engine"
                best_onnx   = model_dir / "best_model.onnx"
                best_pth    = model_dir / "best_model.pth"
                best_pt     = model_dir / "best_model.pt"

                weight_file = (
                    best_engine if best_engine.exists() else
                    best_onnx   if best_onnx.exists()   else
                    best_pth    if best_pth.exists()     else
                    best_pt     if best_pt.exists()      else None
                )
                if weight_file is None:
                    continue

                folder  = model_dir.name
                meta    = self._PYTORCH_MODEL_MAP.get(folder, {})
                display = meta.get("display", folder)
                arch    = meta.get("arch", folder)

                if weight_file.suffix == ".engine":
                    display += " (TensorRT FP16)"
                elif weight_file.suffix == ".onnx":
                    display += " (ONNX FP16)"
                else:
                    display += " (PyTorch FP16+Compile)"

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
        priority = [
            "yolov26n_seg", "yolov8n_seg",
            "yolov26s_seg", "yolov8s_seg",
            "yolov11n_seg",
        ]
        for name in priority:
            if name in self.available_models:
                logger.info("Auto-loading best model: %s", name)
                if self.load_model(name):
                    return

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

        if model_name == self.current_model_name and self.current_model is not None:
            return True

        try:
            if info["type"] == "yolo":
                from ultralytics import YOLO
                self.current_model = YOLO(model_path, task="segment")
                logger.info("Loaded YOLO model: %s from %s", model_name, model_path)

            elif info["type"] == "pytorch":
                if model_path.endswith(".onnx"):
                    import onnxruntime as ort
                    providers = (
                        ["CUDAExecutionProvider"]
                        if "CUDAExecutionProvider" in ort.get_available_providers()
                        else ["CPUExecutionProvider"]
                    )
                    self.current_model = ort.InferenceSession(model_path, providers=providers)
                    logger.info("Loaded ONNX model: %s via %s", model_name, providers[0])
                else:
                    import torch
                    arch  = info.get("arch", model_name)
                    model = self._build_pytorch_model(arch)
                    if model is None:
                        logger.error("No architecture builder for '%s'.", arch)
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
                        model = checkpoint

                    model.eval()
                    if torch.cuda.is_available():
                        model.to("cuda").half()
                        try:
                            model = torch.compile(model, mode="reduce-overhead")
                            logger.info("Applied torch.compile() for faster inference")
                        except Exception as e:
                            logger.warning("torch.compile() not available: %s", e)

                    self.current_model = model
                    logger.info("Loaded PyTorch model: %s from %s", model_name, model_path)
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

        Returns (annotated_frame, detections_list).
        Each detection may include a ``measurements`` key when calibration is on.
        """
        if self.current_model is None:
            return frame, []

        info = self.available_models.get(self.current_model_name, {})

        self._fps_frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._frame_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_window_start = now

        t0 = time.monotonic()
        try:
            if info.get("type") == "yolo":
                result = self._predict_yolo(frame)
            elif info.get("type") == "pytorch":
                result = self._predict_pytorch(frame)
            else:
                return frame, []
        except Exception:
            logger.exception("Inference failed")
            return frame, []

        latency_ms = (time.monotonic() - t0) * 1000
        annotated, detections = result

        avg_conf = 0.0
        if detections:
            confs = [d.get("confidence", 0.0) for d in detections if "confidence" in d]
            avg_conf = sum(confs) / len(confs) if confs else 0.0

        self._inference_logger.log_frame(
            model_name=self.current_model_name or "unknown",
            fps=self._frame_fps,
            latency_ms=latency_ms,
            num_detections=len(detections),
            avg_confidence=avg_conf,
        )

        alerts = self._validate_inference(detections, self._frame_fps, latency_ms)
        if alerts:
            self._latest_alerts.extend(alerts)

        return annotated, detections

    def _validate_inference(
        self,
        detections: List[Dict],
        fps: float,
        latency_ms: float,
    ) -> List[Dict]:
        """Quality checks on the latest inference frame."""
        alerts: List[Dict] = []
        ts = round(time.time(), 3)

        if fps > 0 and fps < 5.0:
            alerts.append({"level": "warning", "msg": f"FPS dropped to {fps:.1f}", "ts": ts})
        if latency_ms > 500:
            alerts.append({"level": "warning", "msg": f"High latency: {latency_ms:.0f} ms", "ts": ts})
        if len(detections) == 0:
            alerts.append({"level": "info", "msg": "No detections in frame", "ts": ts})

        for det in detections:
            conf = det.get("confidence", 1.0)
            if conf < 0.5:
                cls_name = det.get("class", "?")
                alerts.append({"level": "warning", "msg": f"Low confidence: {cls_name} ({conf:.0%})", "ts": ts})

        return alerts

    def get_latest_alerts(self) -> List[Dict]:
        return list(self._latest_alerts)

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------

    def _get_px_to_mm(
        self,
        image_width: int,
        detection_masks: Optional[List[Tuple[str, np.ndarray]]] = None,
        frame: Optional[np.ndarray] = None,
    ) -> float:
        """Compute mm-per-pixel factor based on the active calibration method."""
        cs = self.camera_settings
        if not cs.enabled:
            return 1.0

        if cs.method == "manual":
            return cs.manual_px_to_mm if cs.manual_px_to_mm > 0 else 1.0

        if cs.method == "reference_label":
            return self._px_to_mm_from_detections(detection_masks)

        if cs.method == "ml_depth_midas" and frame is not None:
            try:
                _, depth_map = self.predict_depth(frame)
                median_disp = float(np.median(depth_map))
                if median_disp > 0:
                    cs.object_distance_mm = 10000.0 / median_disp
                    return cs.pixel_to_mm_intrinsics(image_width)
            except Exception as e:
                logger.error("MiDaS failed: %s. Falling back to intrinsics.", e)

        return cs.pixel_to_mm_intrinsics(image_width)

    def _lazy_init_midas(self):
        """Initialise the MiDaS model once on first use."""
        import torch
        if not hasattr(self, "_midas_model"):
            logger.info("Initialising MiDaS…")
            try:
                self._midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
                self._midas_transform = torch.hub.load(
                    "intel-isl/MiDaS", "transforms", trust_repo=True
                ).small_transform
                self._midas_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._midas_model.to(self._midas_device).eval()
            except Exception as e:
                logger.error("Failed to load MiDaS: %s", e)
                raise

    def predict_depth(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run MiDaS depth estimation. Returns (colored_viz, raw_disparity_map)."""
        import torch
        self._lazy_init_midas()

        img_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._midas_transform(img_rgb).to(self._midas_device)

        with torch.no_grad():
            pred = self._midas_model(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        raw_disparity = pred.cpu().numpy()
        d_min, d_max  = raw_disparity.min(), raw_disparity.max()
        if d_max > d_min:
            depth_norm = (raw_disparity - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(raw_disparity)

        depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        return depth_colored, raw_disparity

    def _px_to_mm_from_detections(
        self,
        detection_masks: Optional[List[Tuple[str, np.ndarray]]] = None,
    ) -> float:
        """Find the reference label among detections and calibrate."""
        cs      = self.camera_settings
        ref_name = cs.reference_label_name.lower()
        if not ref_name or detection_masks is None:
            logger.warning("Reference calibration: no reference label or masks – falling back to 1.0")
            return 1.0

        best_contour = None
        best_area    = 0.0
        for class_name, mask in detection_masks:
            if class_name.lower() != ref_name:
                continue
            m = (mask > 127).astype(np.uint8) * 255 if mask.dtype != np.uint8 else mask
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a = cv2.contourArea(c)
                if a > best_area:
                    best_area    = a
                    best_contour = c

        if best_contour is None:
            logger.warning("Reference calibration: label '%s' not found", cs.reference_label_name)
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
        """Extract contours from mask, compute measurements, and draw them."""
        h, w    = mask.shape[:2]
        px_to_mm = self._get_px_to_mm(w, detection_masks, annotated)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours     = [c for c in contours if cv2.contourArea(c) > 50]

        if not contours:
            return annotated, []

        cv2.drawContours(annotated, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

        measurements = []
        if self.camera_settings.enabled:
            measurements = compute_contour_measurements(contours, mask, px_to_mm)
            annotated    = draw_measurements_on_image(annotated, measurements, px_to_mm)

        return annotated, measurements

    # ------------------------------------------------------------------
    # YOLO inference
    # ------------------------------------------------------------------

    def _predict_yolo(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """YOLO-based inference with per-detection measurements."""
        try:
            import torch
            use_half = torch.cuda.is_available()
        except Exception:
            use_half = False

        yolo_imgsz = 640
        yolo_conf  = self.conf_threshold

        if getattr(self, "enable_tracking", False) and hasattr(self.current_model, "track"):
            results = self.current_model.track(
                frame, imgsz=yolo_imgsz, conf=yolo_conf,
                verbose=False, persist=True, tracker="bytetrack.yaml",
            )
        else:
            try:
                results = self.current_model.predict(
                    frame, imgsz=yolo_imgsz, conf=yolo_conf, verbose=False, half=use_half,
                )
            except RuntimeError as e:
                if "binding" in str(e).lower() or "device" in str(e).lower():
                    logger.warning("YOLO GPU/ONNX failed, falling back to CPU: %s", e)
                    results = self.current_model.predict(
                        frame, imgsz=yolo_imgsz, conf=yolo_conf, verbose=False, device="cpu",
                    )
                else:
                    raise

        result   = results[0]
        annotated = frame.copy()
        h, w      = frame.shape[:2]
        detections: List[Dict] = []

        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                det = {
                    "class_id":   int(box.cls.item()),
                    "class_name": result.names.get(int(box.cls.item()), "unknown"),
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox":       [round(v, 1) for v in box.xyxy[0].tolist()],
                    "has_mask":   result.masks is not None and i < len(result.masks),
                    "measurements": [],
                    "idx": i,
                }
                if box.is_track and getattr(self, "enable_tracking", False):
                    det["track_id"] = int(box.id.item()) if box.id is not None else -1
                detections.append(det)

        detections = apply_top_n_filtering(detections)

        CLASS_COLORS_MAP = {
            "mechanical_part": (0, 255, 0),
            "magnet":          (255, 0, 255),
            "circle":          (255, 255, 0),
        }

        detection_masks   = []
        det_contours      = []
        all_measurements  = []
        edge_center_entries = []

        do_edge_refine = getattr(self, "enable_edge_refinement", False)
        guide_filter   = None
        if do_edge_refine:
            try:
                guide_img    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                guide_filter = cv2.ximgproc.createGuidedFilter(guide_img, radius=8, eps=50)
            except AttributeError:
                do_edge_refine = False

        for det_idx, det in enumerate(detections):
            cls_name  = det.get("class_name", "unknown")
            norm_name = cls_name.lower().replace("mechanical", "mechanical")
            color     = CLASS_COLORS_MAP.get(norm_name, (0, 255, 255))

            x1, y1, x2, y2 = det["bbox"]
            if self.draw_boxes:
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            if self.draw_labels:
                label_text = f"{cls_name.replace('_', ' ')} {det.get('confidence', 0.0):.2f}"
                if "track_id" in det:
                    label_text = f"#{det['track_id']} {label_text}"
                cv2.putText(annotated, label_text, (int(x1), max(15, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            if det.get("has_mask") and result.masks is not None:
                orig_idx = det.get("idx")
                if orig_idx is not None:
                    mask_data = result.masks.data[orig_idx]
                    m         = mask_data.cpu().numpy()
                    m_resized = cv2.resize(m, (w, h))

                    if do_edge_refine and guide_filter is not None:
                        m_refined = guide_filter.filter((m_resized * 255.0).astype(np.uint8))
                        m_bin     = (m_refined > 127).astype(np.uint8) * 255
                    else:
                        m_bin = (m_resized > 0.5).astype(np.uint8) * 255

                    detection_masks.append((cls_name, m_bin))

                    if self.draw_masks:
                        color_layer = np.zeros_like(frame)
                        color_layer[m_bin > 127] = color
                        annotated = cv2.addWeighted(annotated, 1.0, color_layer, 0.35, 0)

                    cnts, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        largest = max(cnts, key=cv2.contourArea)
                        if cv2.contourArea(largest) > 50:
                            det_contours.append((largest, cls_name, det_idx))

        # Measurements
        px_to_mm = self._get_px_to_mm(w, detection_masks, frame)
        if self.camera_settings.enabled and det_contours:
            target_names   = {"mechanical_part", "magnet"}
            cross_only_mode = bool(self.camera_settings.show_opposite_distances)

            for contour, cls_name, det_idx in det_contours:
                if not cross_only_mode:
                    measurements = compute_single_contour_measurements(contour, px_to_mm, cls_name)
                    detections[det_idx]["measurements"] = measurements
                    all_measurements.extend(measurements)

                if normalize_measurement_family(cls_name) in target_names:
                    edge_center_entries.append({
                        "name":    cls_name,
                        "points":  compute_edge_center_points(contour),
                        "det_idx": det_idx,
                    })

            if cross_only_mode:
                circle_center = None
                for c, name, _ in det_contours:
                    if name.lower() == "circle":
                        mo = cv2.moments(c)
                        if abs(mo.get("m00", 0.0)) > 1e-6:
                            circle_center = (int(mo["m10"] / mo["m00"]), int(mo["m01"] / mo["m00"]))
                        break

                cross_ms = compute_cross_diametric_opposite_distances(
                    edge_center_entries, px_to_mm, circle_center=circle_center
                )
                for m in cross_ms:
                    for d_idx in m.get("det_indices", []):
                        if d_idx is not None and 0 <= d_idx < len(detections):
                            detections[d_idx]["measurements"].append(m)
                all_measurements.extend(cross_ms)
            else:
                for i in range(len(det_contours)):
                    for j in range(i + 1, len(det_contours)):
                        cnt_a, name_a, idx_a = det_contours[i]
                        cnt_b, name_b, idx_b = det_contours[j]
                        pts_a = cnt_a.reshape(-1, 2).astype(np.float64)
                        pts_b = cnt_b.reshape(-1, 2).astype(np.float64)
                        dists    = cdist(pts_a, pts_b)
                        min_dist = float(np.min(dists))
                        idx      = np.unravel_index(np.argmin(dists), dists.shape)
                        dist_m   = {
                            "type":     "distance",
                            "label":    f"{name_a} ↔ {name_b}",
                            "value_px": round(min_dist, 1),
                            "value_mm": round(min_dist * px_to_mm, 2),
                            "pt1":      pts_a[idx[0]].astype(int).tolist(),
                            "pt2":      pts_b[idx[1]].astype(int).tolist(),
                        }
                        all_measurements.append(dist_m)

            if all_measurements and self.enable_postprocessing:
                annotated = draw_measurements_on_image(annotated, all_measurements, px_to_mm)

        # Depth overlay
        if self.camera_settings.show_depth_map:
            try:
                depth_viz, raw_disp = self.predict_depth(frame)
                annotated = cv2.addWeighted(annotated, 1.0, depth_viz, 0.4, 0)
                global_median = float(np.median(raw_disp))
                if global_median > 0:
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det["bbox"])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            obj_disp = float(np.median(raw_disp[y1:y2, x1:x2]))
                            if obj_disp > 0:
                                approx_depth = (global_median / obj_disp) * self.camera_settings.object_distance_mm
                                det["approx_depth_mm"] = round(approx_depth, 1)
                                cv2.putText(annotated, f"D: {det['approx_depth_mm']}mm",
                                            (x1, min(h - 5, y2 + 15)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception as e:
                logger.error("Failed to apply depth metrics: %s", e)

        for det in detections:
            det.pop("idx", None)

        return annotated, detections

    # ------------------------------------------------------------------
    # PyTorch inference
    # ------------------------------------------------------------------

    def _predict_pytorch(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """PyTorch model inference (UNet, DeepLab, SegFormer, etc.)."""
        import torch

        CLASS_NAMES  = ["background", "mechanical_part", "magnet", "circle"]
        CLASS_COLORS = {
            1: (0, 255, 0),    # mechanical_part
            2: (255, 0, 255),  # magnet
            3: (255, 255, 0),  # circle
        }

        model = self.current_model
        if not hasattr(model, "forward") and not hasattr(model, "run"):
            return frame, []

        h, w     = frame.shape[:2]
        img      = cv2.resize(frame, (512, 512))
        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_float = (img_float - mean) / std

        if hasattr(model, "run"):  # ONNX Runtime
            input_name = model.get_inputs()[0].name
            tensor_np  = np.expand_dims(np.transpose(img_float, (2, 0, 1)), axis=0).astype(np.float32)
            output_np  = model.run(None, {input_name: tensor_np})[0]
            output     = torch.from_numpy(output_np)
        else:
            device      = next((p.device for p in model.parameters()), torch.device("cpu"))
            model_dtype = next((p.dtype  for p in model.parameters()), torch.float32)
            tensor = (
                torch.from_numpy(img_float)
                .permute(2, 0, 1).unsqueeze(0)
                .to(device=device, dtype=model_dtype)
            )
            with torch.no_grad():
                output = model(tensor)

        if isinstance(output, dict):
            output = output.get("out", list(output.values())[0])

        num_channels = output.shape[1]

        # ----------------------------------------------------------
        # Multi-class path (≥ 2 output channels)
        # ----------------------------------------------------------
        if num_channels > 1:
            probs        = torch.softmax(output, dim=1).squeeze(0).detach().cpu().to(torch.float32).numpy()
            pred_classes = probs.argmax(axis=0)
            pred_resized = cv2.resize(pred_classes.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

            do_edge_refine = getattr(self, "enable_edge_refinement", False)
            guide_filter   = None
            if do_edge_refine:
                try:
                    guide_img    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    guide_filter = cv2.ximgproc.createGuidedFilter(guide_img, radius=8, eps=50)
                except AttributeError:
                    do_edge_refine = False

            # Pass 1: extraction
            detections: List[Dict] = []
            for cls_id in range(1, num_channels):
                cls_prob = cv2.resize(probs[cls_id], (w, h))
                cls_mask = (cls_prob > self.conf_threshold).astype(np.uint8) * 255

                if do_edge_refine and guide_filter is not None:
                    cls_mask = guide_filter.filter(cls_mask)
                    cls_mask = (cls_mask > 127).astype(np.uint8) * 255

                if cls_mask.sum() == 0:
                    continue

                contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours     = [c for c in contours if cv2.contourArea(c) > 50]
                cls_prob_rs  = cv2.resize(probs[cls_id], (w, h))

                for cnt in contours:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    cnt_mask  = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
                    region_probs = cls_prob_rs[cnt_mask > 0]
                    conf         = float(region_probs.mean()) if len(region_probs) > 0 else 0.5
                    cls_name     = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                    detections.append({
                        "class_id":    cls_id,
                        "class_name":  cls_name.replace("_", " "),
                        "confidence":  round(conf, 4),
                        "bbox":        [float(x), float(y), float(x + cw), float(y + ch)],
                        "has_mask":    True,
                        "measurements": [],
                        "contour":     cnt,
                    })

            # Pass 2: heuristic & filtering
            if self.enable_heuristic:
                detections = apply_spatial_heuristic_correction(detections)
            if self.enable_top_n:
                detections = apply_top_n_filtering(detections)

            # Pass 3: rendering & measurements
            overlay          = frame.copy()
            detection_masks  = []
            merged_class_masks: Dict[str, np.ndarray] = {}

            for det in detections:
                cname = det["class_name"]
                if cname not in merged_class_masks:
                    merged_class_masks[cname] = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(merged_class_masks[cname], [det["contour"]], -1, 255, -1)

            for cname, m in merged_class_masks.items():
                detection_masks.append((cname, m))

            px_to_mm         = self._get_px_to_mm(w, detection_masks, frame)
            all_measurements = []
            edge_center_entries = []
            target_names        = {"mechanical_part", "magnet"}
            cross_only_mode     = bool(self.camera_settings.show_opposite_distances)

            for det_idx, det in enumerate(detections):
                cls_name = det["class_name"]
                cls_id   = det["class_id"]
                cnt      = det["contour"]
                color    = CLASS_COLORS.get(cls_id, (0, 255, 0))

                if self.draw_masks:
                    color_layer = np.zeros_like(frame)
                    cv2.drawContours(color_layer, [cnt], -1, color, -1)
                    overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.35, 0)
                    cv2.drawContours(overlay, [cnt], -1, color, 2, cv2.LINE_AA)

                x1, y1, x2, y2 = det["bbox"]
                if self.draw_boxes:
                    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                if self.draw_labels:
                    cv2.putText(overlay, f"{cls_name} {det['confidence']:.2f}",
                                (int(x1), max(15, int(y1) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if self.camera_settings.enabled:
                    if not cross_only_mode:
                        ms = compute_single_contour_measurements(cnt, px_to_mm, cls_name)
                        det["measurements"] = ms
                        all_measurements.extend(ms)

                    if normalize_measurement_family(cls_name) in target_names:
                        edge_center_entries.append({
                            "name":    cls_name,
                            "points":  compute_edge_center_points(cnt),
                            "det_idx": det_idx,
                        })

            if self.camera_settings.enabled:
                if cross_only_mode:
                    circle_center = None
                    for d in detections:
                        if d["class_name"].lower() == "circle":
                            mo = cv2.moments(d["contour"])
                            if abs(mo.get("m00", 0.0)) > 1e-6:
                                circle_center = (int(mo["m10"] / mo["m00"]), int(mo["m01"] / mo["m00"]))
                            break

                    cross_ms = compute_cross_diametric_opposite_distances(
                        edge_center_entries, px_to_mm, circle_center=circle_center
                    )
                    for m in cross_ms:
                        for d_idx in m.get("det_indices", []):
                            if d_idx is not None and 0 <= d_idx < len(detections):
                                detections[d_idx]["measurements"].append(m)
                    all_measurements.extend(cross_ms)
                else:
                    for i in range(len(detections)):
                        for j in range(i + 1, len(detections)):
                            cnt_a = detections[i]["contour"]
                            cnt_b = detections[j]["contour"]
                            if len(cnt_a) < 2 or len(cnt_b) < 2:
                                continue
                            pts_a    = cnt_a.reshape(-1, 2).astype(np.float64)
                            pts_b    = cnt_b.reshape(-1, 2).astype(np.float64)
                            dists    = cdist(pts_a, pts_b)
                            min_dist = float(np.min(dists))
                            idx      = np.unravel_index(np.argmin(dists), dists.shape)
                            dist_m   = {
                                "type":     "distance",
                                "label":    f"{detections[i]['class_name']} ↔ {detections[j]['class_name']}",
                                "value_px": round(min_dist, 1),
                                "value_mm": round(min_dist * px_to_mm, 2),
                                "pt1":      pts_a[idx[0]].astype(int).tolist(),
                                "pt2":      pts_b[idx[1]].astype(int).tolist(),
                            }
                            detections[i]["measurements"].append(dist_m)
                            detections[j]["measurements"].append(dist_m)
                            all_measurements.append(dist_m)

                    if edge_center_entries:
                        extra_ms = compute_edge_center_distances(
                            edge_center_entries, px_to_mm,
                            show_edges=self.camera_settings.show_edge_distances,
                            show_centers=self.camera_settings.show_center_distances,
                        )
                        all_measurements.extend(extra_ms)
                        if self.camera_settings.show_aligned_pair_distances:
                            all_measurements.extend(
                                compute_aligned_same_class_distances(edge_center_entries, px_to_mm)
                            )

                if all_measurements:
                    overlay = draw_measurements_on_image(overlay, all_measurements, px_to_mm)

            for det in detections:
                det.pop("contour", None)

            return overlay, detections

        # ----------------------------------------------------------
        # Binary path (1 output channel)
        # ----------------------------------------------------------
        mask        = torch.sigmoid(output).squeeze().cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]

        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (w, h))

        do_edge_refine = getattr(self, "enable_edge_refinement", False)
        if do_edge_refine:
            try:
                guide_img    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                guide_filter = cv2.ximgproc.createGuidedFilter(guide_img, radius=8, eps=50)
                mask_resized = guide_filter.filter(mask_resized)
                mask_resized = (mask_resized > 127).astype(np.uint8) * 255
            except AttributeError:
                pass

        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours     = [c for c in contours if cv2.contourArea(c) > 50]

        overlay    = frame.copy()
        green_mask = np.zeros_like(frame)
        green_mask[mask_resized > 127] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.6, green_mask, 0.4, 0)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)

        detections: List[Dict] = []
        for i, cnt in enumerate(contours):
            x, y, cw, ch = cv2.boundingRect(cnt)
            cnt_mask  = np.zeros(mask.shape, dtype=np.uint8)
            cnt_scaled = cnt.copy()
            cnt_scaled[:, :, 0] = (cnt[:, :, 0] * mask.shape[1] / w).astype(int)
            cnt_scaled[:, :, 1] = (cnt[:, :, 1] * mask.shape[0] / h).astype(int)
            cv2.drawContours(cnt_mask, [cnt_scaled], -1, 1, -1)
            region_probs = mask[cnt_mask > 0]
            conf         = float(region_probs.mean()) if len(region_probs) > 0 else 0.5
            detections.append({
                "class_id":    i,
                "class_name":  f"region_{i + 1}",
                "confidence":  round(conf, 4),
                "bbox":        [float(x), float(y), float(x + cw), float(y + ch)],
                "has_mask":    True,
                "measurements": [],
            })

        detection_masks = [("foreground", mask_resized)]
        px_to_mm        = self._get_px_to_mm(w, detection_masks, frame)

        if self.camera_settings.enabled and contours:
            for i, cnt in enumerate(contours):
                ms = compute_single_contour_measurements(cnt, px_to_mm, f"region_{i + 1}")
                if i < len(detections):
                    detections[i]["measurements"] = ms

            for i in range(len(contours)):
                for j in range(i + 1, len(contours)):
                    if len(contours[i]) < 2 or len(contours[j]) < 2:
                        continue
                    pts_a    = contours[i].reshape(-1, 2).astype(np.float64)
                    pts_b    = contours[j].reshape(-1, 2).astype(np.float64)
                    dists    = cdist(pts_a, pts_b)
                    min_dist = float(np.min(dists))
                    idx      = np.unravel_index(np.argmin(dists), dists.shape)
                    dist_m   = {
                        "type":     "distance",
                        "label":    f"region_{i + 1} ↔ region_{j + 1}",
                        "value_px": round(min_dist, 1),
                        "value_mm": round(min_dist * px_to_mm, 2),
                        "pt1":      pts_a[idx[0]].astype(int).tolist(),
                        "pt2":      pts_b[idx[1]].astype(int).tolist(),
                    }
                    if i < len(detections):
                        detections[i]["measurements"].append(dist_m)
                    if j < len(detections):
                        detections[j]["measurements"].append(dist_m)

            for det in detections:
                overlay = draw_measurements_on_image(overlay, det.get("measurements", []), px_to_mm)

        coverage = float(np.count_nonzero(mask_resized) / mask_resized.size)
        if not detections:
            detections = [{
                "class_id":    0,
                "class_name":  "foreground",
                "confidence":  round(float(mask[mask > 0.5].mean()) if np.any(mask > 0.5) else 0, 4),
                "has_mask":    True,
                "mask_coverage": round(coverage, 4),
                "measurements": [],
            }]

        return overlay, detections

    # ------------------------------------------------------------------
    # PyTorch architecture builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pytorch_model(arch: str):
        """Instantiate an empty PyTorch model by architecture name."""
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

        def _build_deeplabv3():
            model = tv_models.segmentation.deeplabv3_mobilenet_v3_large(weights=None, progress=False)
            model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
            model.aux_classifier = None
            return model

        builders = {
            "segformer_b0":        lambda: SegFormerB0Simple(n_classes=4),
            "unet_lightweight":    lambda: UNetLightweight(n_channels=3, n_classes=4, base_filters=32),
            "unet_resnet18":       lambda: UNetResNet18(n_classes=4, pretrained=False),
            "hed":                 lambda: HEDNet(),
            "rcf":                 lambda: RCFNet(),
            "pidinet":             lambda: PiDiNet(),
            "teed":                lambda: TEEDNet(),
            "deeplabv3_mobilenet": _build_deeplabv3,
        }

        builder = builders.get(arch)
        return builder() if builder else None

    # ------------------------------------------------------------------
    # Public info helpers
    # ------------------------------------------------------------------

    def get_models_info(self) -> List[Dict]:
        """Return list of available models with metadata."""
        return [
            {
                "name":         name,
                "display_name": info.get("display_name", name),
                "type":         info["type"],
                "source":       info["source"],
                "active":       name == self.current_model_name,
            }
            for name, info in self.available_models.items()
        ]

    def get_class_labels(self) -> List[str]:
        """Return class label names for the currently loaded model."""
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

        import torch
        if hasattr(self.current_model, "forward"):
            try:
                device = next((p.device for p in self.current_model.parameters()), torch.device("cpu"))
                dummy  = torch.zeros(1, 3, 64, 64, device=device)
                with torch.no_grad():
                    out = self.current_model(dummy)
                if isinstance(out, dict):
                    out = out.get("out", list(out.values())[0])
                if out.shape[1] >= 4:
                    return ["mechanical_part", "magnet", "circle"]
            except Exception:
                pass

        return ["foreground (all classes)"]

    @property
    def fps(self) -> float:
        return self._frame_fps
