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
            annotated, detections = model_manager.predict(img)
            inference_ms = (time.monotonic() - t0) * 1000
            metrics_tracker.add_metric(inference_ms, is_error=False, model_name=model_manager.current_model_name)
        except Exception as e:
            metrics_tracker.add_metric(None, is_error=True, model_name=model_manager.current_model_name)
            raise e

        # Push results over DataChannel
        dc = getattr(self.pc, "_results_channel", None)
        if dc is not None and dc.readyState == "open":
            payload = {
                "detections": detections,
                "inference_ms": round(inference_ms, 1),
                "server_fps": round(self._fps, 1),
                "model": model_manager.available_models.get(
                    model_manager.current_model_name, {}
                ).get("display_name", model_manager.current_model_name) or "none",
                "timestamp": round(time.time(), 3),
                "alerts": model_manager.get_latest_alerts(),
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

async def _handle_offer(offer: RTCSessionDescription) -> RTCSessionDescription:
    """Create a peer connection, attach video transform, return SDP answer."""
    pc_id = str(uuid.uuid4())[:8]
    pc = RTCPeerConnection()
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


# ---------- REST API ---------- #

@app.route("/api/models", methods=["GET"])
def api_list_models():
    """Return available models and which one is active."""
    return jsonify(
        {
            "models": model_manager.get_models_info(),
            "current": model_manager.current_model_name,
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
    model_name = data.get("model")

    if not model_name:
        return jsonify({"error": "Missing 'model' field"}), 400

    if model_name not in model_manager.available_models:
        return jsonify({"error": f"Unknown model: {model_name}"}), 404

    success = model_manager.load_model(model_name)
    if success:
        return jsonify({"status": "ok", "model": model_name})
    else:
        return jsonify({"error": f"Failed to load model: {model_name}"}), 500


# ---------- Pipeline Enhancements ---------- #

@app.route("/api/inference-settings", methods=["GET", "POST"])
def api_inference_settings():
    """Get or update state-of-the-art inference enhancements."""
    if request.method == "GET":
        return jsonify({
            "enable_tracking": getattr(model_manager, "enable_tracking", False),
            "enable_edge_refinement": getattr(model_manager, "enable_edge_refinement", False)
        })

    data = request.get_json(force=True)
    if "enable_tracking" in data:
        model_manager.enable_tracking = bool(data["enable_tracking"])
    if "enable_edge_refinement" in data:
        model_manager.enable_edge_refinement = bool(data["enable_edge_refinement"])

    return jsonify({"status": "ok"})


# ---------- WebRTC Signaling ---------- #

@app.route("/offer", methods=["POST"])
def webrtc_offer():
    """
    Receive an SDP offer from the client, create a peer connection,
    and return an SDP answer.
    """
    params = request.get_json(force=True)

    if "sdp" not in params or "type" not in params:
        return jsonify({"error": "Missing sdp or type"}), 400

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    answer = run_async(_handle_offer(offer))

    return jsonify({"sdp": answer.sdp, "type": answer.type})


# ---------- Image Upload Detection ---------- #

@app.route("/api/detect", methods=["POST"])
def api_detect_image():
    """
    Accept an uploaded image, run model inference, and return
    the annotated image (base64 JPEG) plus detection metadata.
    """
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
        metrics_tracker.add_metric(None, is_error=True, model_name=model_manager.current_model_name)
        return jsonify({"error": "Could not decode image"}), 400

    # Run inference
    t0 = time.time()
    try:
        annotated, detections = model_manager.predict(img)
        inference_ms = (time.time() - t0) * 1000
        metrics_tracker.add_metric(inference_ms, is_error=False, model_name=model_manager.current_model_name)
    except Exception as e:
        metrics_tracker.add_metric(None, is_error=True, model_name=model_manager.current_model_name)
        return jsonify({"error": str(e)}), 500

    # Encode annotated image as JPEG → base64
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return jsonify({
        "image": f"data:image/jpeg;base64,{b64}",
        "detections": detections,
        "inference_ms": round(inference_ms, 1),
        "model": model_manager.current_model_name,
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

import threading as _mv_threading

_mv_latest_frame: bytes | None = None          # Latest raw JPEG from camera
_mv_latest_annotated: bytes | None = None      # Latest annotated JPEG after detection
_mv_latest_detections: list = []               # Latest detection metadata
_mv_latest_model: str | None = None
_mv_latest_inference_ms: float = 0
_mv_frame_lock = _mv_threading.Lock()


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

    t0 = time.time()
    try:
        annotated, detections = model_manager.predict(img)
        inference_ms = (time.time() - t0) * 1000
        metrics_tracker.add_metric(inference_ms, is_error=False, model_name=model_manager.current_model_name)
    except Exception as e:
        metrics_tracker.add_metric(None, is_error=True, model_name=model_manager.current_model_name)
        return jsonify({"error": str(e)}), 500

    # Encode annotated frame
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

    with _mv_frame_lock:
        _mv_latest_annotated = buf.tobytes()
        _mv_latest_detections = detections
        _mv_latest_model = model_manager.current_model_name
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
