"""
Flask + WebRTC server for real-time chignon detection.

Architecture
------------
- Flask serves the client UI and REST API endpoints.
- aiortc handles WebRTC peer connections in a background asyncio loop.
- Video frames arrive via WebRTC, are processed by the model, and the
  annotated frames are sent back.  Detection results are pushed over
  a WebRTC DataChannel as JSON.

Endpoints
---------
GET  /                  → Client UI
GET  /api/models        → List available models
POST /api/select-model  → Switch the active model
POST /offer             → WebRTC SDP signaling
GET  /api/mlflow-url    → MLflow tracking UI URL & health
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import base64
import io

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

from inference import ModelManager

from chignon.inference import ChignonModelManager
from files.inference import FileModelManager

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("webrtc-server")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
CLIENT_DIR = os.environ.get("CLIENT_DIR", str(PROJECT_ROOT / "client"))
app = Flask(__name__, static_folder=CLIENT_DIR, static_url_path="/static")
CORS(app)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
model_manager = ModelManager()
chignon_manager = ChignonModelManager()
file_manager = FileModelManager()

def _get_manager(domain: str = "stator"):
    """Return the correct model manager for the given domain."""
    if domain == "chignon":
        return chignon_manager
    if domain == "file":
        return file_manager
    # stator all goes to the main manager
    if domain and domain != "stator":
        model_manager.set_domain(domain)
    return model_manager
peer_connections: dict[str, RTCPeerConnection] = {}
relay = MediaRelay()

class LiveMetricsTracker:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.lock = threading.Lock()
        self.data = []
        self.total_requests = 0
        self.total_errors = 0
        
        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as f:
                    content = f.read()
                    if content.strip():
                        self.data = json.loads(content)
                        self.total_requests = len(self.data)
                        self.total_errors = sum(1 for d in self.data if d.get("is_error"))
            except Exception as e:
                logger.error(f"Failed to load metrics: {e}")

        self.dirty = False
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self):
        while True:
            time.sleep(2.0)
            if self.dirty:
                with self.lock:
                    data_to_write = list(self.data)
                    self.dirty = False
                try:
                    with open(self.filepath, "w") as f:
                        json.dump(data_to_write, f)
                except Exception:
                    pass

    def add_metric(self, latency_ms, is_error=False, model_name=None):
        with self.lock:
            self.total_requests += 1
            if is_error:
                self.total_errors += 1
                
            entry = {
                "timestamp": time.time(),
                "latency_ms": latency_ms,
                "is_error": is_error,
                "model": model_name
            }
            self.data.append(entry)
            
            if len(self.data) > 5000:
                self.data = self.data[-5000:]
                
            self.dirty = True

    def get_stats(self):
        with self.lock:
            recent_latencies = [d["latency_ms"] for d in self.data[-100:] if d["latency_ms"] is not None and not d["is_error"]]
            avg_lat = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0
            
            if len(self.data) >= 2:
                recent = self.data[-100:]
                dt = recent[-1]["timestamp"] - recent[0]["timestamp"]
                throughput = (len(recent) - 1) / dt if dt > 0 else 0
            else:
                throughput = 0
                
            err_rate = (self.total_errors / self.total_requests) * 100 if self.total_requests > 0 else 0
            
            return {
                "avg_latency_ms": round(avg_lat, 2),
                "throughput_fps": round(throughput, 2),
                "error_rate_percent": round(err_rate, 2),
                "total_requests": self.total_requests,
                "total_errors": self.total_errors
            }

os.makedirs(PROJECT_ROOT / "server" / "public", exist_ok=True)
metrics_tracker = LiveMetricsTracker(PROJECT_ROOT / "server" / "public" / "detections.json")

# ---------------------------------------------------------------------------
# Async event loop running in a daemon thread
# ---------------------------------------------------------------------------
_loop = asyncio.new_event_loop()


def _start_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


_thread = threading.Thread(target=_start_loop, args=(_loop,), daemon=True)
_thread.start()


def run_async(coro):
    """Run an asyncio coroutine from synchronous Flask handlers."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=30)


# ═══════════════════════════════════════════════════════════════════════════
# WebRTC Video Transform Track
# ═══════════════════════════════════════════════════════════════════════════

class VideoTransformTrack(MediaStreamTrack):
    """
    Receives a video track from the client, runs model inference on each
    frame, and outputs the annotated frame.  Detection results are pushed
    over a DataChannel attached to the same peer connection.
    """

    kind = "video"

    def __init__(self, track: MediaStreamTrack, pc: RTCPeerConnection):
        super().__init__()
        self.track = track
        self.pc = pc
        self.domain = getattr(pc, "_domain", "stator")
        self.mgr = _get_manager(self.domain)

        # FPS tracking
        self._frame_count = 0
        self._fps_window_start = time.monotonic()
        self._fps = 0.0

        # Skip frames to keep up with real-time
        self._process_every_n = 1  # Process every frame by default

    async def recv(self) -> VideoFrame:
        frame = await self.track.recv()
        self._frame_count += 1

        # FPS calculation (1-second window)
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_window_start = now

        # Convert to numpy BGR
        img = frame.to_ndarray(format="bgr24")

        # Run inference
        t0 = time.monotonic()
        try:
            annotated, detections = self.mgr.predict(img)
            inference_ms = (time.monotonic() - t0) * 1000
            metrics_tracker.add_metric(inference_ms, is_error=False, model_name=self.mgr.current_model_name)
        except Exception as e:
            metrics_tracker.add_metric(None, is_error=True, model_name=self.mgr.current_model_name)
            raise e

        # Push results over DataChannel
        dc = getattr(self.pc, "_results_channel", None)
        if dc is not None and dc.readyState == "open":
            payload = {
                "detections": detections,
                "inference_ms": round(inference_ms, 1),
                "server_fps": round(self._fps, 1),
                "model": self.mgr.available_models.get(
                    self.mgr.current_model_name, {}
                ).get("display_name", self.mgr.current_model_name) or "none",
                "timestamp": round(time.time(), 3),
                "alerts": getattr(self.mgr, "get_latest_alerts", lambda: [])(),
            }
            try:
                dc.send(json.dumps(payload))
            except Exception:
                pass

        # Build output video frame
        new_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


# ═══════════════════════════════════════════════════════════════════════════
# WebRTC Signaling
# ═══════════════════════════════════════════════════════════════════════════

async def _handle_offer(offer: RTCSessionDescription, domain: str = "stator") -> RTCSessionDescription:
    """Create a peer connection, attach video transform, return SDP answer."""
    pc_id = str(uuid.uuid4())[:8]
    pc = RTCPeerConnection()
    pc._domain = domain
    peer_connections[pc_id] = pc

    logger.info("[%s] New peer connection", pc_id)

    # ----- DataChannel ----- #
    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info("[%s] DataChannel '%s' received", pc_id, channel.label)
        pc._results_channel = channel

        @channel.on("message")
        def on_message(message):
            # Client can send commands via DataChannel if needed
            logger.debug("[%s] DataChannel message: %s", pc_id, message)

    # ----- Video Track ----- #
    @pc.on("track")
    def on_track(track: MediaStreamTrack):
        logger.info("[%s] Track received: %s", pc_id, track.kind)
        if track.kind == "video":
            transform = VideoTransformTrack(relay.subscribe(track), pc)
            pc.addTrack(transform)

        @track.on("ended")
        async def on_ended():
            logger.info("[%s] Track %s ended", pc_id, track.kind)

    # ----- Connection lifecycle ----- #
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        state = pc.connectionState
        logger.info("[%s] Connection state: %s", pc_id, state)
        if state in ("failed", "closed"):
            await pc.close()
            peer_connections.pop(pc_id, None)

    # ----- Negotiate ----- #
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return pc.localDescription


# ═══════════════════════════════════════════════════════════════════════════
# Flask Routes
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the client application."""
    return send_from_directory(CLIENT_DIR, "index.html")


@app.route("/client/<path:filename>")
def serve_client_files(filename):
    """Serve static client assets."""
    return send_from_directory(CLIENT_DIR, filename)


# ═══════════════════════════════════════════════════════════════════════════
# API Documentation (Swagger UI + OpenAPI spec)
# ═══════════════════════════════════════════════════════════════════════════

_SWAGGER_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <title>Stator Vision — API Docs</title>
  <link rel=\"stylesheet\" href=\"https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui.css\">
  <style>
    html, body { margin: 0; padding: 0; background: #0e1117; }
    #swagger-ui { max-width: 1200px; margin: 0 auto; }
    .topbar { display: none; }
  </style>
</head>
<body>
  <div id=\"swagger-ui\"></div>
  <script src=\"https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui-bundle.js\"></script>
  <script>
    window.onload = () => {
      window.ui = SwaggerUIBundle({
        url: '/api/openapi.json',
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [SwaggerUIBundle.presets.apis],
        layout: 'BaseLayout',
      });
    };
  </script>
</body>
</html>
"""


def _build_openapi_spec() -> dict:
    """OpenAPI 3.0 spec describing all public REST endpoints."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Stator Vision Inspection API",
            "version": "1.0.0",
            "description": (
                "REST API for the Stator / Chignon / File vision inspection server. "
                "Covers model management, live detection, MindVision camera streaming, "
                "automated inspection sessions, MLflow integration and metrics."
            ),
        },
        "servers": [{"url": "/", "description": "Current host"}],
        "tags": [
            {"name": "Models", "description": "Discover & switch detection models"},
            {"name": "Detection", "description": "Run detection on images & WebRTC streams"},
            {"name": "MindVision", "description": "Industrial camera streaming & capture control"},
            {"name": "Inspection", "description": "Automated stator→chignon→file inspection sessions"},
            {"name": "Metrics", "description": "Live metrics, inference logs and alerts"},
            {"name": "Settings", "description": "Camera & measurement configuration"},
            {"name": "MLflow", "description": "MLflow tracking UI availability"},
        ],
        "components": {
            "schemas": {
                "Error": {
                    "type": "object",
                    "properties": {"error": {"type": "string"}},
                },
                "Ok": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                },
                "ModelInfo": {
                    "type": "object",
                    "properties": {
                        "models": {
                            "type": "array",
                            "items": {"type": "object", "additionalProperties": True},
                        },
                        "current": {"type": "string", "nullable": True},
                    },
                },
                "Measurement": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "label": {"type": "string"},
                        "value_px": {"type": "number"},
                        "value_mm": {"type": "number"},
                    },
                },
                "Detection": {
                    "type": "object",
                    "properties": {
                        "class_id": {"type": "integer"},
                        "class_name": {"type": "string"},
                        "confidence": {"type": "number"},
                        "bbox": {"type": "array", "items": {"type": "number"}},
                        "has_mask": {"type": "boolean"},
                        "measurements": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/Measurement"},
                        },
                    },
                },
                "DetectResponse": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Annotated JPEG (data URL)"},
                        "depth_image": {"type": "string", "nullable": True},
                        "detections": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/Detection"},
                        },
                        "alerts": {"type": "array", "items": {"type": "object"}},
                        "position_message": {"type": "string"},
                        "inference_ms": {"type": "number"},
                        "model": {"type": "string"},
                    },
                },
                "InspectionStatus": {
                    "type": "object",
                    "properties": {
                        "active": {"type": "boolean"},
                        "stage": {
                            "type": "string",
                            "nullable": True,
                            "enum": ["stator", "chignon", "file", None],
                        },
                        "stage_elapsed": {"type": "number"},
                        "stage_total": {"type": "number"},
                        "stage_remaining": {"type": "number"},
                        "total_elapsed": {"type": "number"},
                        "frame_count": {"type": "integer"},
                        "result": {"type": "object", "nullable": True},
                    },
                },
                "InspectionResult": {
                    "type": "object",
                    "properties": {
                        "cancelled": {"type": "boolean"},
                        "duration_s": {"type": "number"},
                        "stator": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string"},
                                    "label": {"type": "string"},
                                    "unit": {"type": "string"},
                                    "count": {"type": "integer"},
                                    "mean": {"type": "number", "nullable": True},
                                    "variance": {"type": "number", "nullable": True},
                                    "std": {"type": "number", "nullable": True},
                                    "validation": {"type": "object", "nullable": True},
                                },
                            },
                        },
                        "chignon": {"type": "array", "items": {"type": "object"}},
                        "file": {
                            "type": "object",
                            "properties": {
                                "decision": {"type": "string", "enum": ["OK", "NOT OK", "UNKNOWN"]},
                                "ok_ratio": {"type": "number", "nullable": True},
                                "samples": {"type": "integer"},
                            },
                        },
                    },
                },
                "Thresholds": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number"},
                            "max": {"type": "number"},
                        },
                    },
                    "example": {
                        "chignon.left_area_mm": {"min": 120.0, "max": 260.0},
                        "stator.magnet.diag-asc": {"min": 40.0, "max": 60.0},
                    },
                },
            }
        },
        "paths": {
            "/api/models": {
                "get": {
                    "tags": ["Models"],
                    "summary": "List available models for a domain",
                    "parameters": [{
                        "name": "domain", "in": "query", "required": False,
                        "schema": {"type": "string", "enum": ["stator", "chignon", "file"], "default": "stator"},
                    }],
                    "responses": {"200": {
                        "description": "Model list",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ModelInfo"}}},
                    }},
                }
            },
            "/api/performance/{model_name}": {
                "get": {
                    "tags": ["Models"],
                    "summary": "Return benchmark/training metrics for a specific model",
                    "parameters": [{
                        "name": "model_name", "in": "path", "required": True,
                        "schema": {"type": "string"},
                    }],
                    "responses": {"200": {"description": "Metrics payload"}},
                }
            },
            "/api/select-model": {
                "post": {
                    "tags": ["Models"],
                    "summary": "Switch the active model",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "model_name": {"type": "string"},
                            "domain": {"type": "string", "enum": ["stator", "chignon", "file"]},
                        },
                        "required": ["model_name"],
                    }}}},
                    "responses": {
                        "200": {"description": "Model loaded"},
                        "400": {"description": "Invalid request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                    },
                }
            },
            "/api/inference-enhancements": {
                "get": {
                    "tags": ["Settings"],
                    "summary": "Get current inference enhancement flags (tracking, edge refine, etc.)",
                    "responses": {"200": {"description": "Current flags"}},
                },
                "post": {
                    "tags": ["Settings"],
                    "summary": "Update inference enhancement flags",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object"}}}},
                    "responses": {"200": {"description": "Updated flags"}},
                },
            },
            "/offer": {
                "post": {
                    "tags": ["Detection"],
                    "summary": "WebRTC SDP offer/answer exchange",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "sdp": {"type": "string"},
                            "type": {"type": "string"},
                            "domain": {"type": "string", "enum": ["stator", "chignon", "file"]},
                        },
                    }}}},
                    "responses": {"200": {"description": "SDP answer"}},
                }
            },
            "/api/detect": {
                "post": {
                    "tags": ["Detection"],
                    "summary": "Run detection on an uploaded image",
                    "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
                        "type": "object",
                        "properties": {
                            "image": {"type": "string", "format": "binary"},
                            "domain": {"type": "string", "enum": ["stator", "chignon", "file"]},
                        },
                        "required": ["image"],
                    }}}},
                    "responses": {
                        "200": {"description": "Annotated image + detections", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/DetectResponse"}}}},
                        "400": {"description": "Bad request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                    },
                }
            },
            "/api/status": {
                "get": {"tags": ["Metrics"], "summary": "Server status + health", "responses": {"200": {"description": "Status"}}}
            },
            "/api/live-metrics": {
                "get": {"tags": ["Metrics"], "summary": "Live aggregated detection metrics", "responses": {"200": {"description": "Metrics"}}}
            },
            "/api/inference-logs": {
                "get": {
                    "tags": ["Metrics"],
                    "summary": "Last N entries of the current JSONL inference session log",
                    "parameters": [{
                        "name": "n", "in": "query", "required": False,
                        "schema": {"type": "integer", "default": 100, "maximum": 500},
                    }],
                    "responses": {"200": {"description": "Log entries"}},
                }
            },
            "/api/inference-alerts": {
                "get": {"tags": ["Metrics"], "summary": "Recent inference validation alerts", "responses": {"200": {"description": "Alert list"}}}
            },
            "/api/mlflow-url": {
                "get": {"tags": ["MLflow"], "summary": "MLflow UI URL and reachability", "responses": {"200": {"description": "MLflow status"}}}
            },
            "/api/settings": {
                "get": {"tags": ["Settings"], "summary": "Get camera + measurement settings", "responses": {"200": {"description": "Settings"}}},
                "post": {
                    "tags": ["Settings"],
                    "summary": "Update camera + measurement settings",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object"}}}},
                    "responses": {"200": {"description": "Updated"}},
                },
            },
            "/api/labels": {
                "get": {"tags": ["Settings"], "summary": "List labels usable as measurement reference", "responses": {"200": {"description": "Labels"}}}
            },
            "/api/mindvision/start": {
                "post": {"tags": ["MindVision"], "summary": "Launch mindvision_capture.py subprocess", "responses": {"200": {"description": "Process started or already running"}}}
            },
            "/api/mindvision/stop": {
                "post": {"tags": ["MindVision"], "summary": "Terminate mindvision_capture.py subprocess", "responses": {"200": {"description": "Process stopped"}}}
            },
            "/api/mindvision/proc-status": {
                "get": {"tags": ["MindVision"], "summary": "Return running state of the MindVision subprocess", "responses": {"200": {"description": "Subprocess state"}}}
            },
            "/api/mindvision/frame": {
                "post": {
                    "tags": ["MindVision"],
                    "summary": "Ingest a raw camera frame (called by mindvision_capture.py)",
                    "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
                        "type": "object",
                        "properties": {"frame": {"type": "string", "format": "binary"}},
                        "required": ["frame"],
                    }}}},
                    "responses": {
                        "200": {"description": "Frame accepted"},
                        "400": {"description": "Decode error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                    },
                }
            },
            "/api/mindvision/stream": {
                "get": {
                    "tags": ["MindVision"],
                    "summary": "MJPEG stream of annotated camera frames",
                    "responses": {"200": {"description": "multipart/x-mixed-replace MJPEG stream", "content": {"multipart/x-mixed-replace": {}}}},
                }
            },
            "/api/mindvision/depth-stream": {
                "get": {"tags": ["MindVision"], "summary": "MJPEG stream of MiDaS depth maps", "responses": {"200": {"description": "MJPEG stream"}}}
            },
            "/api/mindvision/status": {
                "get": {"tags": ["MindVision"], "summary": "MindVision camera connection state", "responses": {"200": {"description": "Camera status"}}}
            },
            "/api/inspection/start": {
                "post": {
                    "tags": ["Inspection"],
                    "summary": "Start a three-stage automated inspection (stator 5s → chignon 5s → file 2s)",
                    "responses": {
                        "200": {"description": "Started", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/InspectionStatus"}}}},
                        "400": {"description": "Camera not streaming", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                        "409": {"description": "Inspection already running", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                    },
                }
            },
            "/api/inspection/cancel": {
                "post": {"tags": ["Inspection"], "summary": "Cancel the current inspection run", "responses": {"200": {"description": "Cancelled", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Ok"}}}}}}
            },
            "/api/inspection/status": {
                "get": {"tags": ["Inspection"], "summary": "Current inspection progress", "responses": {"200": {"description": "Status", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/InspectionStatus"}}}}}}
            },
            "/api/inspection/result": {
                "get": {"tags": ["Inspection"], "summary": "Latest inspection result (mean/variance/validation)", "responses": {"200": {"description": "Result", "content": {"application/json": {"schema": {"type": "object", "properties": {"active": {"type": "boolean"}, "result": {"$ref": "#/components/schemas/InspectionResult"}}}}}}}}
            },
            "/api/inspection/thresholds": {
                "get": {"tags": ["Inspection"], "summary": "Get inspection validation thresholds", "responses": {"200": {"description": "Thresholds", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Thresholds"}}}}}},
                "post": {
                    "tags": ["Inspection"],
                    "summary": "Replace inspection validation thresholds",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Thresholds"}}}},
                    "responses": {"200": {"description": "Saved"}},
                },
            },
            "/api/inspection/durations": {
                "get": {"tags": ["Inspection"], "summary": "Get per-stage durations (seconds)", "responses": {"200": {"description": "Durations"}}},
                "post": {
                    "tags": ["Inspection"],
                    "summary": "Update per-stage durations (seconds)",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "properties": {"stator": {"type": "number"}, "chignon": {"type": "number"}, "file": {"type": "number"}}}}}},
                    "responses": {"200": {"description": "Saved"}, "400": {"description": "Invalid value"}, "409": {"description": "Inspection running"}},
                },
            },
        },
    }


@app.route("/api/docs")
def api_docs_ui():
    """Serve the Swagger UI page for the API."""
    return Response(_SWAGGER_HTML, mimetype="text/html")


@app.route("/api/openapi.json")
def api_openapi_spec():
    """Serve the OpenAPI 3.0 JSON spec consumed by Swagger UI."""
    return jsonify(_build_openapi_spec())


# ---------- REST API ---------- #

@app.route("/api/models", methods=["GET"])
def api_list_models():
    """Return available models and which one is active."""
    domain = request.args.get("domain", "stator")
    mgr = _get_manager(domain)
    return jsonify(
        {
            "models": mgr.get_models_info(),
            "current": mgr.current_model_name,
        }
    )


@app.route("/api/performance/<model_name>", methods=["GET"])
def api_model_performance(model_name):
    """Return benchmark/training performance metrics for a specific model."""
    history_path = PROJECT_ROOT / "outputs" / "results" / "training_logs" / model_name / "training_history.json"
    if not history_path.exists():
        return jsonify({"error": "No performance data found"}), 404
        
    try:
        with open(history_path, "r") as f:
            data = json.load(f)
            
        best_dice = None
        if "epochs" in data:
            best_ep = data.get("best_epoch", 1)
            for ep in data["epochs"]:
                if ep.get("epoch") == best_ep:
                    best_dice = ep.get("val", {}).get("dice")
                    break
                    
        return jsonify({
            "model_name": model_name,
            "best_val_iou": data.get("best_val_iou"),
            "best_val_dice": best_dice,
            "total_train_time_sec": data.get("total_train_time_sec"),
            "peak_gpu_memory_mb": data.get("peak_gpu_memory_mb")
        })
    except Exception as e:
        logger.error(f"Failed to read performance data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/select-model", methods=["POST"])
def api_select_model():
    """Switch the active model."""
    data = request.get_json(force=True)
    domain = data.get("domain", "stator")
    mgr = _get_manager(domain)
    model_name = data.get("model")

    if not model_name:
        return jsonify({"error": "Missing 'model' field"}), 400

    if model_name not in mgr.available_models:
        return jsonify({"error": f"Unknown model: {model_name}"}), 404

    success = mgr.load_model(model_name)
    if success:
        return jsonify({"status": "ok", "model": model_name})
    else:
        return jsonify({"error": f"Failed to load model: {model_name}"}), 500


# ---------- Pipeline Enhancements ---------- #

@app.route("/api/inference-enhancements", methods=["GET", "POST"])
def api_inference_enhancements():
    """Get or update state-of-the-art inference enhancements and post-processing flags."""
    domain = request.args.get("domain", "stator")
    mgr = _get_manager(domain)

    if request.method == "GET":
        return jsonify({
            "enable_tracking": getattr(mgr, "enable_tracking", False),
            "enable_edge_refinement": getattr(mgr, "enable_edge_refinement", False),
            "enable_postprocessing": getattr(mgr, "enable_postprocessing", True),
            "enable_file_color_validation": getattr(mgr, "enable_file_color_validation", True),
            "enable_heuristic": getattr(mgr, "enable_heuristic", True),
            "enable_top_n": getattr(mgr, "enable_top_n", True),
            "draw_boxes": getattr(mgr, "draw_boxes", True),
            "draw_masks": getattr(mgr, "draw_masks", True),
            "draw_labels": getattr(mgr, "draw_labels", True),
            "conf_threshold": getattr(mgr, "conf_threshold", 0.05),
            "domain": domain
        })

    data = request.get_json(force=True)
    
    # Apply to specific manager
    if "enable_tracking" in data:
        mgr.enable_tracking = bool(data["enable_tracking"])
    if "enable_edge_refinement" in data:
        mgr.enable_edge_refinement = bool(data["enable_edge_refinement"])
    if "enable_postprocessing" in data:
        mgr.enable_postprocessing = bool(data["enable_postprocessing"])
    if "enable_file_color_validation" in data:
        mgr.enable_file_color_validation = bool(data["enable_file_color_validation"])
    if "enable_heuristic" in data:
        mgr.enable_heuristic = bool(data["enable_heuristic"])
    if "enable_top_n" in data:
        mgr.enable_top_n = bool(data["enable_top_n"])
    if "draw_boxes" in data:
        mgr.draw_boxes = bool(data["draw_boxes"])
    if "draw_masks" in data:
        mgr.draw_masks = bool(data["draw_masks"])
    if "draw_labels" in data:
        mgr.draw_labels = bool(data["draw_labels"])
    if "conf_threshold" in data:
        mgr.conf_threshold = float(data["conf_threshold"])

    return jsonify({"status": "ok", "domain": domain})


# ---------- WebRTC Signaling ---------- #

@app.route("/offer", methods=["POST"])
def webrtc_offer():
    """
    Receive an SDP offer from the client, create a peer connection,
    and return an SDP answer.
    """
    params = request.get_json(force=True)
    domain = params.get("domain", "stator")
    if domain and domain != "file":
        model_manager.set_domain(domain)

    if "sdp" not in params or "type" not in params:
        return jsonify({"error": "Missing sdp or type"}), 400

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    answer = run_async(_handle_offer(offer, domain=domain))

    return jsonify({"sdp": answer.sdp, "type": answer.type})


# ---------- Image Upload Detection ---------- #

@app.route("/api/detect", methods=["POST"])
def api_detect_image():
    """
    Accept an uploaded image, run model inference, and return
    the annotated image (base64 JPEG) plus detection metadata.
    """
    domain = request.form.get("domain", "stator")
    mgr = _get_manager(domain)

    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Read image bytes → numpy
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        metrics_tracker.add_metric(None, is_error=True, model_name=mgr.current_model_name)
        return jsonify({"error": "Could not decode image"}), 400

    # Run inference
    t0 = time.time()
    try:
        annotated, detections = mgr.predict(img)
        inference_ms = (time.time() - t0) * 1000
        metrics_tracker.add_metric(inference_ms, is_error=False, model_name=mgr.current_model_name)
    except Exception as e:
        metrics_tracker.add_metric(None, is_error=True, model_name=mgr.current_model_name)
        return jsonify({"error": str(e)}), 500

    # Encode annotated image as JPEG → base64
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    depth_b64 = None
    if hasattr(mgr, 'camera_settings') and mgr.camera_settings.show_depth_map:
        try:
            depth_viz = mgr.predict_depth(img)
            _, d_buf = cv2.imencode(".jpg", depth_viz, [cv2.IMWRITE_JPEG_QUALITY, 80])
            depth_b64 = base64.b64encode(d_buf.tobytes()).decode("utf-8")
        except Exception as e:
            logger.error("Upload depth viz failed: %s", e)

    alerts = getattr(mgr, "get_latest_alerts", lambda: [])()
    position_message = getattr(mgr, "get_latest_position_message", lambda: "")()

    return jsonify({
        "image": f"data:image/jpeg;base64,{b64}",
        "depth_image": f"data:image/jpeg;base64,{depth_b64}" if depth_b64 else None,
        "detections": detections,
        "alerts": alerts,
        "position_message": position_message,
        "inference_ms": round(inference_ms, 1),
        "model": mgr.current_model_name,
    })


@app.route("/api/status", methods=["GET"])
def api_status():
    """Health check / status endpoint."""
    model_display = "none"
    if model_manager.current_model_name:
        model_display = model_manager.available_models.get(
            model_manager.current_model_name, {}
        ).get("display_name", model_manager.current_model_name)

    return jsonify(
        {
            "status": "running",
            "active_model": model_display,
            "active_connections": len(peer_connections),
            "available_models": len(model_manager.available_models),
            "mindvision_connected": _mv_latest_frame is not None,
        }
    )

@app.route("/api/live-metrics", methods=["GET"])
def api_live_metrics():
    """Return live aggregated metrics from detections.json."""
    return jsonify(metrics_tracker.get_stats())


@app.route("/api/inference-logs", methods=["GET"])
def api_inference_logs():
    """Return the last N entries from the current JSONL inference session log."""
    n = min(int(request.args.get("n", 100)), 500)
    entries = model_manager._inference_logger.read_last_n(n)
    return jsonify({
        "entries": entries,
        "session_file": str(model_manager._inference_logger.get_current_session_file().name),
        "count": len(entries),
    })


@app.route("/api/inference-alerts", methods=["GET"])
def api_inference_alerts():
    """Return the latest inference validation alerts."""
    return jsonify({"alerts": model_manager.get_latest_alerts()})


# ---------- MLflow ---------- #

@app.route("/api/mlflow-url", methods=["GET"])
def api_mlflow_url():
    """Return the MLflow tracking UI URL and whether the server is reachable.

    The internal Docker URI (e.g. http://mlflow:5002) is rewritten so that
    the browser can reach the dashboard (e.g. http://localhost:5002).
    """
    internal_uri = os.environ.get("MLFLOW_TRACKING_URI", "")

    if not internal_uri:
        return jsonify({"configured": False, "url": None, "reachable": False})

    # Build a browser-accessible URL: replace Docker-internal hostnames
    # (like "mlflow") with the host the browser can reach.
    import urllib.parse
    parsed = urllib.parse.urlparse(internal_uri)
    # If the hostname is not already localhost / 127.0.0.1, swap it out
    browser_host = os.environ.get("MLFLOW_EXTERNAL_HOST", "localhost")
    browser_url = f"{parsed.scheme}://{browser_host}:{parsed.port or 5002}"

    # Quick reachability probe against the *internal* URI (server-side)
    reachable = False
    try:
        import urllib.request
        urllib.request.urlopen(f"{internal_uri}/health", timeout=2)
        reachable = True
    except Exception:
        pass

    return jsonify({
        "configured": True,
        "url": browser_url,
        "reachable": reachable,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Camera Settings (for measurement post-processing)
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    """Return current camera / measurement settings."""
    return jsonify(model_manager.camera_settings.to_dict())


@app.route("/api/labels", methods=["GET"])
def api_get_labels():
    """Return class labels of the currently loaded model."""
    return jsonify({"labels": model_manager.get_class_labels()})


@app.route("/api/settings", methods=["POST"])
def api_update_settings():
    """Update camera / measurement settings.

    Accepts JSON with any subset of:
        sensor_width_mm, focal_length_mm, object_distance_mm, enabled
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    model_manager.camera_settings.update(data)
    logger.info("Camera settings updated: %s", model_manager.camera_settings.to_dict())
    return jsonify(model_manager.camera_settings.to_dict())


# ═══════════════════════════════════════════════════════════════════════════
# MindVision Camera Streaming
# ═══════════════════════════════════════════════════════════════════════════
# The mindvision_capture.py script pushes frames here; the web client
# can view them via MJPEG stream or poll for the latest annotated frame.

import subprocess as _mv_subprocess
import threading as _mv_threading

_mv_latest_frame: bytes | None = None          # Latest raw JPEG from camera
_mv_latest_annotated: bytes | None = None      # Latest annotated JPEG after detection
_mv_latest_depth: bytes | None = None          # Latest depth map JPEG
_mv_latest_detections: list = []               # Latest detection metadata
_mv_latest_model: str | None = None
_mv_latest_inference_ms: float = 0
_mv_frame_lock = _mv_threading.Lock()
_mv_active_domain: str = "stator"             # Domain used for live MV camera inference

# Subprocess handle for mindvision_capture.py when launched from the UI
_mv_proc: "_mv_subprocess.Popen | None" = None
_mv_proc_lock = _mv_threading.Lock()


_MV_LOG_PATH = PROJECT_ROOT / "outputs" / "mindvision_capture.log"


@app.route("/api/mindvision/start", methods=["POST"])
def mv_start():
    """Launch mindvision_capture.py as a managed subprocess."""
    global _mv_proc
    with _mv_proc_lock:
        if _mv_proc is not None and _mv_proc.poll() is None:
            return jsonify({"ok": True, "running": True, "msg": "already running"})

        script = os.path.join(os.path.dirname(__file__), "mindvision_capture.py")
        if not os.path.exists(script):
            return jsonify({"ok": False, "msg": "mindvision_capture.py not found"}), 404

        try:
            _MV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(_MV_LOG_PATH, "w")
            # Point the capture subprocess at this server, regardless of port.
            # request.host_url looks like "http://localhost:5001/"
            server_url = request.host_url.rstrip("/")
            _mv_proc = _mv_subprocess.Popen(
                [sys.executable, script, "--server-url", server_url],
                stdout=log_fh,
                stderr=_mv_subprocess.STDOUT,
            )
            # Give the child a moment to crash on obvious failures (SDK missing,
            # camera already open, etc.) so we can report the reason immediately.
            time.sleep(0.6)
            if _mv_proc.poll() is not None:
                log_fh.close()
                tail = ""
                try:
                    with open(_MV_LOG_PATH) as f:
                        tail = f.read().strip().splitlines()[-6:]
                        tail = "\n".join(tail)
                except Exception:
                    pass
                rc = _mv_proc.returncode
                _mv_proc = None
                return jsonify({
                    "ok": False,
                    "running": False,
                    "msg": f"capture script exited immediately (code {rc})",
                    "log_tail": tail,
                    "log_path": str(_MV_LOG_PATH),
                }), 500
            return jsonify({"ok": True, "running": True, "pid": _mv_proc.pid})
        except Exception as exc:
            return jsonify({"ok": False, "msg": str(exc)}), 500


@app.route("/api/mindvision/stop", methods=["POST"])
def mv_stop():
    """Terminate the managed mindvision_capture.py subprocess."""
    global _mv_proc, _mv_latest_frame, _mv_latest_annotated
    with _mv_proc_lock:
        if _mv_proc is None or _mv_proc.poll() is not None:
            _mv_proc = None
            # Clear stale frame so status shows disconnected
            with _mv_frame_lock:
                _mv_latest_frame = None
                _mv_latest_annotated = None
            return jsonify({"ok": True, "running": False, "msg": "not running"})

        try:
            _mv_proc.terminate()
            try:
                _mv_proc.wait(timeout=4)
            except _mv_subprocess.TimeoutExpired:
                _mv_proc.kill()
            _mv_proc = None
            with _mv_frame_lock:
                _mv_latest_frame = None
                _mv_latest_annotated = None
            return jsonify({"ok": True, "running": False})
        except Exception as exc:
            return jsonify({"ok": False, "msg": str(exc)}), 500


@app.route("/api/mindvision/proc-status")
def mv_proc_status():
    """Return whether the managed capture subprocess is currently running."""
    with _mv_proc_lock:
        running = _mv_proc is not None and _mv_proc.poll() is None
    return jsonify({"running": running})


# ═══════════════════════════════════════════════════════════════════════════
# MLflow Integration
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/mlflow-url")
def mlflow_url():
    """Return the MLflow UI URL and reachability status."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        return jsonify({"available": False, "url": None})

    # Probe the MLflow health endpoint
    try:
        import urllib.request
        urllib.request.urlopen(tracking_uri + "/health", timeout=2)
        reachable = True
    except Exception:
        reachable = False

    # Build a browser-accessible URL:
    # Inside Docker the URI is http://mlflow:5002 (container hostname).
    # Replace with the host-facing address for the browser link.
    browser_url = tracking_uri.replace("http://mlflow:", "http://localhost:")

    return jsonify({"available": reachable, "url": browser_url})


@app.route("/api/mindvision/frame", methods=["POST"])
def mv_receive_frame():
    """
    Receive a frame from the MindVision capture script,
    run detection, and store both raw + annotated for streaming.
    """
    global _mv_latest_frame, _mv_latest_annotated
    global _mv_latest_detections, _mv_latest_model, _mv_latest_inference_ms

    if "frame" not in request.files:
        return jsonify({"error": "No 'frame' in request"}), 400

    file_bytes = request.files["frame"].read()

    # Store raw frame
    with _mv_frame_lock:
        _mv_latest_frame = file_bytes

    # Decode and run detection
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        metrics_tracker.add_metric(None, is_error=True, model_name=model_manager.current_model_name)
        return jsonify({"error": "Could not decode frame"}), 400

    # If an inspection is running, route the frame through the stage-appropriate
    # model manager (stator / chignon / file) so measurements match each stage.
    # Otherwise use whichever domain the UI has selected via /api/mindvision/set-domain.
    insp_stage = _inspection_session.stage if _inspection_session.active else None
    active_mgr = _inspection_manager_for(insp_stage) if insp_stage else _get_manager(_mv_active_domain)

    t0 = time.time()
    try:
        annotated, detections = active_mgr.predict(img)
        inference_ms = (time.time() - t0) * 1000
        metrics_tracker.add_metric(inference_ms, is_error=False, model_name=active_mgr.current_model_name)
    except Exception as e:
        metrics_tracker.add_metric(None, is_error=True, model_name=active_mgr.current_model_name)
        return jsonify({"error": str(e)}), 500

    if insp_stage:
        pos_msg = getattr(active_mgr, "get_latest_position_message", lambda: "")()
        _inspection_session.ingest(insp_stage, detections, pos_msg)

    # Encode annotated frame
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    depth_buf = None
    if hasattr(active_mgr, "camera_settings") and active_mgr.camera_settings.show_depth_map:
        try:
            depth_viz = active_mgr.predict_depth(img)
            _, d_buf = cv2.imencode(".jpg", depth_viz, [cv2.IMWRITE_JPEG_QUALITY, 80])
            depth_buf = d_buf.tobytes()
        except Exception as e:
            logger.error("Depth visualization failed: %s", e)

    with _mv_frame_lock:
        _mv_latest_annotated = buf.tobytes()
        _mv_latest_depth = depth_buf
        _mv_latest_detections = detections
        _mv_latest_model = active_mgr.current_model_name
        _mv_latest_inference_ms = round(inference_ms, 1)

    return jsonify({"ok": True, "inference_ms": round(inference_ms, 1)})


@app.route("/api/mindvision/stream")
def mv_stream():
    """MJPEG stream of annotated MindVision camera frames for the web client."""
    def generate():
        while True:
            with _mv_frame_lock:
                frame = _mv_latest_annotated
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.05)  # ~20 FPS max

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/mindvision/depth-stream")
def mv_depth_stream():
    """MJPEG stream of MiDaS depth estimates for the web client."""
    def generate():
        while True:
            # We check the lock briefly to copy the bytes
            with _mv_frame_lock:
                frame = _mv_latest_depth
            
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                # If depth is disabled, send an empty placeholder or wait
                time.sleep(0.2)
                continue
                
            time.sleep(0.06)  # ~15 FPS max for depth to save bandwidth

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/mindvision/latest")
def mv_latest():
    """Get the latest annotated frame + detections as JSON."""
    with _mv_frame_lock:
        if _mv_latest_annotated is None:
            return jsonify({"connected": False}), 404

        b64 = base64.b64encode(_mv_latest_annotated).decode("utf-8")
        return jsonify({
            "connected": True,
            "image": f"data:image/jpeg;base64,{b64}",
            "detections": _mv_latest_detections,
            "inference_ms": _mv_latest_inference_ms,
            "model": _mv_latest_model,
        })


@app.route("/api/mindvision/status")
def mv_status():
    """Check if MindVision camera is streaming frames."""
    return jsonify({
        "connected": _mv_latest_frame is not None,
        "model": _mv_latest_model,
        "inference_ms": _mv_latest_inference_ms,
    })


@app.route("/api/mindvision/set-domain", methods=["POST"])
def mv_set_domain():
    """Set the active detection domain for the live MindVision camera stream."""
    global _mv_active_domain
    data = request.get_json(force=True)
    domain = data.get("domain", "stator")
    if domain not in ("stator", "chignon", "file"):
        return jsonify({"error": f"Unknown domain: {domain}"}), 400
    _mv_active_domain = domain
    return jsonify({"status": "ok", "domain": _mv_active_domain})


# ═══════════════════════════════════════════════════════════════════════════
# Automated Inspection Session
# ═══════════════════════════════════════════════════════════════════════════
# Runs three sequential stages on live MindVision frames:
#   1. stator  (5 s) — collect cross-diameter distances (magnet, mechanical_part)
#   2. chignon (5 s) — collect left/right chignon surface areas
#   3. file    (2 s) — determine OK / NOT OK position (binary)
# Results per measurement group are reported as mean ± variance, optionally
# compared against user-defined min/max thresholds.

import math as _math
from statistics import mean as _stat_mean, pvariance as _stat_pvariance

_STAGE_DURATIONS: dict[str, float] = {"stator": 5.0, "chignon": 5.0, "file": 2.0}
_STAGE_DURATIONS_LOCK = _mv_threading.Lock()
_STAGE_DURATION_LIMITS = (0.5, 120.0)  # seconds
_STAGE_ORDER = ["stator", "chignon", "file"]


def _stage_duration(stage: str) -> float:
    with _STAGE_DURATIONS_LOCK:
        return float(_STAGE_DURATIONS.get(stage, 0.0))

# Optional user-defined validation thresholds (persisted in memory only).
# Keys:
#   chignon.left_area_mm, chignon.right_area_mm     -> {"min": x, "max": y}
#   stator.<family>.<orientation>_mm                -> {"min": x, "max": y}
#     family      : "magnet" | "mechanical_part"
#     orientation : "diag-asc" | "diag-desc"
_inspection_thresholds: dict = {}
_inspection_thresholds_lock = _mv_threading.Lock()


def _xdiam_key(label: str) -> str | None:
    """Map a cross-diameter label like 'Opposite magnet (diag-asc (TR-BL))'
    to a canonical threshold key 'stator.magnet.diag-asc'."""
    if not label:
        return None
    low = label.lower()
    if "magnet" in low:
        family = "magnet"
    elif "mechanical" in low:
        family = "mechanical_part"
    else:
        return None
    if "diag-asc" in low:
        orient = "diag-asc"
    elif "diag-desc" in low:
        orient = "diag-desc"
    else:
        return None
    return f"stator.{family}.{orient}"


class InspectionSession:
    """State machine for an automated three-stage inspection run."""

    def __init__(self) -> None:
        self.lock = _mv_threading.Lock()
        self.active: bool = False
        self.stage: str | None = None          # "stator" | "chignon" | "file" | None
        self.started_at: float = 0.0
        self.stage_started_at: float = 0.0
        self._thread: _mv_threading.Thread | None = None
        self._cancel = _mv_threading.Event()

        # Per-stage raw buffers
        self._stator_samples: dict[str, list[float]] = {}   # key -> list of mm values
        self._chignon_samples: dict[str, list[float]] = {}  # "left"/"right" -> list of mm² values
        self._file_samples: list[bool] = []                 # True = correct position

        self.result: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> tuple[bool, str]:
        with self.lock:
            if self.active:
                return False, "inspection already running"
            self.active = True
            self.stage = _STAGE_ORDER[0]
            self.started_at = time.time()
            self.stage_started_at = self.started_at
            self._stator_samples = {}
            self._chignon_samples = {}
            self._file_samples = []
            self.result = None
            self._cancel.clear()
        self._thread = _mv_threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True, "started"

    def cancel(self) -> None:
        with self.lock:
            if not self.active:
                return
            self._cancel.set()

    def status(self) -> dict:
        with self.lock:
            if not self.active:
                return {
                    "active": False,
                    "stage": None,
                    "result": self.result,
                }
            now = time.time()
            stage_elapsed = now - self.stage_started_at
            stage_total = _stage_duration(self.stage or "")
            stage_remaining = max(0.0, stage_total - stage_elapsed)
            # Frame counts
            if self.stage == "stator":
                frame_count = max((len(v) for v in self._stator_samples.values()), default=0)
            elif self.stage == "chignon":
                frame_count = max((len(v) for v in self._chignon_samples.values()), default=0)
            elif self.stage == "file":
                frame_count = len(self._file_samples)
            else:
                frame_count = 0
            return {
                "active": True,
                "stage": self.stage,
                "stage_elapsed": round(stage_elapsed, 2),
                "stage_total": stage_total,
                "stage_remaining": round(stage_remaining, 2),
                "total_elapsed": round(now - self.started_at, 2),
                "frame_count": frame_count,
                "result": None,
            }

    # ------------------------------------------------------------------
    # Frame ingestion (called from mv_receive_frame)
    # ------------------------------------------------------------------

    def ingest(self, stage: str, detections: list, position_message: str = "") -> None:
        """Append measurements from the latest detection into the current stage buffer."""
        with self.lock:
            if not self.active or self.stage != stage:
                return
            if stage == "stator":
                # Aggregate one sample per cross-diameter key per frame
                frame_vals: dict[str, float] = {}
                for det in detections or []:
                    for m in det.get("measurements", []) or []:
                        if m.get("type") != "cross_diametric_opposite_distance":
                            continue
                        key = _xdiam_key(m.get("label", ""))
                        if key is None:
                            continue
                        val = m.get("value_mm")
                        if val is None:
                            continue
                        # Each cross-diameter appears twice (once per detection);
                        # dedupe per frame by key.
                        frame_vals[key] = float(val)
                for key, val in frame_vals.items():
                    self._stator_samples.setdefault(key, []).append(val)

            elif stage == "chignon":
                # Chignon detections expose area measurements. Sort by x-center
                # to label left vs right. Collect one area per chignon per frame.
                areas: list[tuple[float, float]] = []  # (x_center, area_mm2)
                for det in detections or []:
                    if "chignon" not in str(det.get("class_name", "")).lower():
                        continue
                    area_mm = None
                    for m in det.get("measurements", []) or []:
                        if m.get("type") == "area":
                            area_mm = m.get("value_mm")
                            break
                    if area_mm is None:
                        continue
                    bbox = det.get("bbox") or [0, 0, 0, 0]
                    x_center = (bbox[0] + bbox[2]) / 2.0
                    areas.append((x_center, float(area_mm)))
                areas.sort(key=lambda t: t[0])
                if len(areas) >= 1:
                    self._chignon_samples.setdefault("left", []).append(areas[0][1])
                if len(areas) >= 2:
                    self._chignon_samples.setdefault("right", []).append(areas[-1][1])

            elif stage == "file":
                # Binary decision from position message. Use prefix match because
                # "correct position" is a substring of "incorrect position", and
                # the fallback "Incorrect: need at least 2 ..." also starts with
                # "incorrect" — both must count as a NOT-OK sample.
                msg = (position_message or "").strip().lower()
                if msg.startswith("correct"):
                    self._file_samples.append(True)
                elif msg.startswith("incorrect"):
                    self._file_samples.append(False)
                # No message → skip

    # ------------------------------------------------------------------
    # Stage transitions (background thread)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            for stage in _STAGE_ORDER:
                if self._cancel.is_set():
                    break
                self._enter_stage(stage)
                end = time.time() + _stage_duration(stage)
                while time.time() < end:
                    if self._cancel.is_set():
                        break
                    time.sleep(0.05)
            if not self._cancel.is_set():
                result = self._aggregate()
            else:
                result = {"cancelled": True}
        except Exception as exc:
            logger.exception("Inspection run failed")
            result = {"error": str(exc)}
        finally:
            with self.lock:
                self.result = result
                self.active = False
                self.stage = None

    def _enter_stage(self, stage: str) -> None:
        with self.lock:
            self.stage = stage
            self.stage_started_at = time.time()
        logger.info("[Inspection] Entering stage: %s", stage)

    # ------------------------------------------------------------------
    # Aggregation + validation
    # ------------------------------------------------------------------

    def _aggregate(self) -> dict:
        with _inspection_thresholds_lock:
            thr = dict(_inspection_thresholds)

        def _summary(values: list[float]) -> dict:
            n = len(values)
            if n == 0:
                return {"count": 0, "mean": None, "variance": None, "std": None}
            mu = _stat_mean(values)
            var = _stat_pvariance(values) if n > 1 else 0.0
            return {
                "count": n,
                "mean": round(mu, 3),
                "variance": round(var, 4),
                "std": round(_math.sqrt(var), 3),
            }

        def _validate(key: str, mu: float | None) -> dict | None:
            if mu is None:
                return None
            bounds = thr.get(key)
            if not bounds:
                return None
            lo = bounds.get("min")
            hi = bounds.get("max")
            ok = True
            reason = None
            if lo is not None and mu < float(lo):
                ok = False
                reason = f"below min {lo}"
            elif hi is not None and mu > float(hi):
                ok = False
                reason = f"above max {hi}"
            return {"valid": ok, "reason": reason, "min": lo, "max": hi}

        # --- Stator ---
        stator_results = []
        for key, values in sorted(self._stator_samples.items()):
            summ = _summary(values)
            stator_results.append({
                "key": key,
                "label": key.replace("stator.", "").replace(".", " "),
                "unit": "mm",
                **summ,
                "validation": _validate(key, summ["mean"]),
            })

        # --- Chignon ---
        chignon_results = []
        for side in ("left", "right"):
            values = self._chignon_samples.get(side, [])
            summ = _summary(values)
            key = f"chignon.{side}_area_mm"
            chignon_results.append({
                "key": key,
                "label": f"{side.capitalize()} chignon area",
                "unit": "mm²",
                **summ,
                "validation": _validate(key, summ["mean"]),
            })

        # --- File ---
        total = len(self._file_samples)
        ok_count = sum(1 for v in self._file_samples if v)
        if total == 0:
            file_decision = {"decision": "UNKNOWN", "ok_ratio": None, "samples": 0}
        else:
            ratio = ok_count / total
            file_decision = {
                "decision": "OK" if ratio >= 0.5 else "NOT OK",
                "ok_ratio": round(ratio, 3),
                "samples": total,
            }

        return {
            "cancelled": False,
            "duration_s": round(time.time() - self.started_at, 2),
            "stator": stator_results,
            "chignon": chignon_results,
            "file": file_decision,
        }


_inspection_session = InspectionSession()


def _inspection_manager_for(stage: str | None):
    """Return the model manager used during a given inspection stage."""
    if stage == "chignon":
        return chignon_manager
    if stage == "file":
        return file_manager
    return model_manager


@app.route("/api/inspection/start", methods=["POST"])
def inspection_start():
    if _mv_latest_frame is None:
        return jsonify({"ok": False, "error": "MindVision camera not streaming"}), 400
    ok, msg = _inspection_session.start()
    if not ok:
        return jsonify({"ok": False, "error": msg}), 409
    return jsonify({"ok": True, "status": _inspection_session.status()})


@app.route("/api/inspection/cancel", methods=["POST"])
def inspection_cancel():
    _inspection_session.cancel()
    return jsonify({"ok": True})


@app.route("/api/inspection/status")
def inspection_status():
    return jsonify(_inspection_session.status())


@app.route("/api/inspection/result")
def inspection_result():
    with _inspection_session.lock:
        return jsonify({"result": _inspection_session.result, "active": _inspection_session.active})


@app.route("/api/inspection/thresholds", methods=["GET", "POST"])
def inspection_thresholds():
    global _inspection_thresholds
    if request.method == "GET":
        with _inspection_thresholds_lock:
            return jsonify(_inspection_thresholds)
    # POST: replace with a new mapping of { key: {min, max} }
    payload = request.get_json(silent=True) or {}
    cleaned: dict = {}
    for key, bounds in payload.items():
        if not isinstance(bounds, dict):
            continue
        lo = bounds.get("min")
        hi = bounds.get("max")
        entry: dict = {}
        if lo not in (None, ""):
            try:
                entry["min"] = float(lo)
            except (TypeError, ValueError):
                pass
        if hi not in (None, ""):
            try:
                entry["max"] = float(hi)
            except (TypeError, ValueError):
                pass
        if entry:
            cleaned[str(key)] = entry
    with _inspection_thresholds_lock:
        _inspection_thresholds = cleaned
    return jsonify({"ok": True, "thresholds": cleaned})


@app.route("/api/inspection/durations", methods=["GET", "POST"])
def inspection_durations():
    """Get or update per-stage durations (seconds) for the inspection session."""
    if request.method == "GET":
        with _STAGE_DURATIONS_LOCK:
            return jsonify(dict(_STAGE_DURATIONS))
    if _inspection_session.active:
        return jsonify({"ok": False, "error": "inspection running"}), 409
    payload = request.get_json(silent=True) or {}
    lo, hi = _STAGE_DURATION_LIMITS
    updated: dict[str, float] = {}
    for stage in _STAGE_ORDER:
        if stage not in payload:
            continue
        try:
            val = float(payload[stage])
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": f"invalid value for {stage}"}), 400
        if not (lo <= val <= hi):
            return jsonify({"ok": False, "error": f"{stage} must be between {lo} and {hi}s"}), 400
        updated[stage] = val
    with _STAGE_DURATIONS_LOCK:
        _STAGE_DURATIONS.update(updated)
        current = dict(_STAGE_DURATIONS)
    return jsonify({"ok": True, "durations": current})


# ═══════════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════════

async def _shutdown():
    """Close all active peer connections."""
    tasks = [pc.close() for pc in peer_connections.values()]
    await asyncio.gather(*tasks)
    peer_connections.clear()


import atexit


def _cleanup():
    try:
        asyncio.run_coroutine_threadsafe(_shutdown(), _loop).result(timeout=5)
    except Exception:
        pass


atexit.register(_cleanup)


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chignon Detection WebRTC Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    parser.add_argument("--model", default=None, help="Model to load at startup")
    parser.add_argument("--debug", action="store_true", help="Flask debug mode")
    args = parser.parse_args()

    if args.model:
        model_manager.load_model(args.model)

    logger.info(
        "Starting server on http://%s:%d  |  Model: %s",
        args.host,
        args.port,
        model_manager.current_model_name or "(none)",
    )

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
