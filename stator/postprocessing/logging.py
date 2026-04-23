"""
Inference session logger — writes per-frame metrics to a rotating JSONL file
and optionally summarises the session to MLflow on rotation.
"""

import json
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))


class InferenceLogger:
    """Writes per-frame inference metrics to a JSONL session file.

    Each line is a JSON object with:
        ts, model, fps, latency_ms, num_detections, avg_confidence

    Files are stored under ``outputs/inference_logs/`` and rotate
    when exceeding *max_bytes* (~10 MB).
    """

    MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or (PROJECT_ROOT / "outputs" / "inference_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._session_file = self._new_session_path()
        self._bytes_written = 0
        self._mlflow_stats = self._fresh_mlflow_stats()

    # -- public API -------------------------------------------------------

    def log_frame(
        self,
        model_name: str,
        fps: float,
        latency_ms: float,
        num_detections: int,
        avg_confidence: float,
    ) -> None:
        entry = {
            "ts": round(time.time(), 3),
            "model": model_name,
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms, 2),
            "detections": num_detections,
            "avg_conf": round(avg_confidence, 3),
        }
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        try:
            with open(self._session_file, "a") as f:
                f.write(line)
            self._bytes_written += len(line)
            if self._bytes_written >= self.MAX_BYTES:
                self._rotate()
        except OSError:
            pass  # non-critical; silently skip

        s = self._mlflow_stats
        s["frames"] += 1
        s["total_latency_ms"] += latency_ms
        s["total_fps"] += fps
        s["total_detections"] += num_detections
        s["total_confidence"] += avg_confidence
        if s["model"] is None:
            s["model"] = model_name

    def get_current_session_file(self) -> Path:
        return self._session_file

    def read_last_n(self, n: int = 100) -> list:
        """Read the last *n* entries from the current session file."""
        try:
            with open(self._session_file, "r") as f:
                lines = f.readlines()
            tail = lines[-n:] if len(lines) > n else lines
            return [json.loads(l) for l in tail if l.strip()]
        except (OSError, json.JSONDecodeError):
            return []

    # -- internals --------------------------------------------------------

    def _new_session_path(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"session_{ts}.jsonl"

    def _rotate(self) -> None:
        self._flush_mlflow_session()
        self._session_file = self._new_session_path()
        self._bytes_written = 0
        self._mlflow_stats = self._fresh_mlflow_stats()

    # -- MLflow helpers ---------------------------------------------------

    @staticmethod
    def _fresh_mlflow_stats() -> dict:
        return {
            "frames": 0,
            "total_latency_ms": 0.0,
            "total_fps": 0.0,
            "total_detections": 0,
            "total_confidence": 0.0,
            "model": None,
            "start_ts": time.time(),
        }

    def _flush_mlflow_session(self) -> None:
        """Log an inference session summary run to MLflow (non-blocking, best-effort)."""
        if not (_MLFLOW_AVAILABLE and _MLFLOW_URI):
            return
        s = self._mlflow_stats
        n = s["frames"]
        if n == 0:
            return
        try:
            mlflow.set_tracking_uri(_MLFLOW_URI)
            mlflow.set_experiment("stator-vision/inference")
            run_name = f"inference-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({"model": s["model"] or "unknown"})
                mlflow.log_metrics({
                    "avg_latency_ms":     s["total_latency_ms"]  / n,
                    "avg_fps":            s["total_fps"]          / n,
                    "avg_detections":     s["total_detections"]   / n,
                    "avg_confidence":     s["total_confidence"]   / n,
                    "total_frames":       float(n),
                    "session_duration_s": time.time() - s["start_ts"],
                })
                if self._session_file.exists():
                    mlflow.log_artifact(str(self._session_file), artifact_path="inference_logs")
        except Exception:
            pass  # never crash inference over MLflow
