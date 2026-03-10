#!/usr/bin/env python3
"""
MindVision USB Camera — Capture & Detect Script
================================================
Connects to a MindVision industrial camera via USB, grabs frames
continuously, and sends them to the Flask detection server for
chignon segmentation inference.

Architecture:
  [MindVision USB Camera] → [This Script] → HTTP POST /api/detect → [Flask Server]
                                           ← Annotated image + detections ←

The script also streams annotated frames to the server via
POST /api/mindvision/frame so the web client can view them.

Requirements:
  - MindVision SDK installed (libMVSDK.so in /usr/lib/)
  - mvsdk.py (copied from sdkmindvision/demo/python_demo/)
  - Server running on --server-url (default: http://localhost:5000)

Usage:
  python mindvision_capture.py                          # Auto-detect camera, send to localhost:5000
  python mindvision_capture.py --server-url http://HOST:5000
  python mindvision_capture.py --exposure 20            # Set exposure to 20ms
  python mindvision_capture.py --show                   # Also show local OpenCV window
  python mindvision_capture.py --save-dir ./captures    # Save annotated frames to disk
"""

import argparse
import base64
import json
import os
import platform
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add the SDK python_demo directory so we can import mvsdk
SDK_DIR = Path(__file__).resolve().parent.parent / "sdkmindvision" / "demo" / "python_demo"
if str(SDK_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_DIR))

try:
    import mvsdk
except ImportError as e:
    print(f"[ERROR] Cannot import mvsdk: {e}")
    print("Make sure libMVSDK.so is installed (check /usr/lib/libMVSDK.so)")
    print(f"And mvsdk.py is at: {SDK_DIR / 'mvsdk.py'}")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' package required. Install with: pip install requests")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# MindVision Camera Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class MindVisionCamera:
    """Wrapper around the MindVision SDK for clean open/grab/close lifecycle."""

    def __init__(self, camera_index: int = 0, exposure_ms: float = 30.0):
        self.camera_index = camera_index
        self.exposure_ms = exposure_ms
        self.hCamera = 0
        self.pFrameBuffer = 0
        self.cap = None
        self.mono = False
        self._is_open = False

    def list_cameras(self) -> list:
        """Enumerate all connected MindVision cameras."""
        dev_list = mvsdk.CameraEnumerateDevice()
        cameras = []
        for i, dev in enumerate(dev_list):
            cameras.append({
                "index": i,
                "name": dev.GetFriendlyName(),
                "port": dev.GetPortType(),
            })
        return cameras

    def open(self) -> bool:
        """Initialize and start the camera."""
        if self._is_open:
            return True

        dev_list = mvsdk.CameraEnumerateDevice()
        if len(dev_list) < 1:
            print("[ERROR] No MindVision camera found! Check USB connection.")
            return False

        if self.camera_index >= len(dev_list):
            print(f"[ERROR] Camera index {self.camera_index} out of range. "
                  f"Found {len(dev_list)} camera(s).")
            return False

        dev_info = dev_list[self.camera_index]
        print(f"[INFO] Opening camera {self.camera_index}: {dev_info.GetFriendlyName()} "
              f"(port: {dev_info.GetPortType()})")

        try:
            self.hCamera = mvsdk.CameraInit(dev_info, -1, -1)
        except mvsdk.CameraException as e:
            print(f"[ERROR] CameraInit failed ({e.error_code}): {e.message}")
            return False

        # Get camera capabilities
        self.cap = mvsdk.CameraGetCapability(self.hCamera)

        # Mono vs color
        self.mono = (self.cap.sIspCapacity.bMonoSensor != 0)
        if self.mono:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # Continuous capture mode
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)

        # Manual exposure
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, self.exposure_ms * 1000)

        # Start capture thread
        mvsdk.CameraPlay(self.hCamera)

        # Allocate frame buffer (max resolution)
        channels = 1 if self.mono else 3
        buf_size = (self.cap.sResolutionRange.iWidthMax *
                    self.cap.sResolutionRange.iHeightMax * channels)
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(buf_size, 16)

        self._is_open = True
        print(f"[INFO] Camera opened — max resolution: "
              f"{self.cap.sResolutionRange.iWidthMax}x{self.cap.sResolutionRange.iHeightMax}, "
              f"{'mono' if self.mono else 'color'}, exposure: {self.exposure_ms}ms")
        return True

    def grab(self, timeout_ms: int = 200) -> np.ndarray | None:
        """Grab a single frame as a numpy BGR/gray array. Returns None on timeout."""
        if not self._is_open:
            return None

        try:
            pRawData, frame_head = mvsdk.CameraGetImageBuffer(self.hCamera, timeout_ms)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, frame_head)
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

            # On Windows the image is flipped; on Linux it's already correct
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, frame_head, 1)

            # Convert raw buffer → numpy array
            frame_data = (mvsdk.c_ubyte * frame_head.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            channels = 1 if frame_head.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
            frame = frame.reshape((frame_head.iHeight, frame_head.iWidth, channels))

            return frame

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"[WARN] CameraGetImageBuffer failed ({e.error_code}): {e.message}")
            return None

    def close(self):
        """Release camera resources."""
        if self._is_open:
            mvsdk.CameraUnInit(self.hCamera)
            self.hCamera = 0
            self._is_open = False

        if self.pFrameBuffer:
            mvsdk.CameraAlignFree(self.pFrameBuffer)
            self.pFrameBuffer = 0

        print("[INFO] Camera closed")

    @property
    def is_open(self) -> bool:
        return self._is_open


# ═══════════════════════════════════════════════════════════════════════════
# Detection Client
# ═══════════════════════════════════════════════════════════════════════════

class DetectionClient:
    """Sends frames to the Flask server's /api/detect endpoint."""

    def __init__(self, server_url: str, timeout: float = 10.0):
        self.server_url = server_url.rstrip("/")
        self.detect_url = f"{self.server_url}/api/detect"
        self.stream_url = f"{self.server_url}/api/mindvision/frame"
        self.status_url = f"{self.server_url}/api/status"
        self.timeout = timeout
        self.session = requests.Session()

    def check_server(self) -> bool:
        """Check if the detection server is reachable."""
        try:
            r = self.session.get(self.status_url, timeout=3)
            if r.status_code == 200:
                data = r.json()
                print(f"[INFO] Server OK — model: {data.get('active_model', 'none')}, "
                      f"models available: {data.get('available_models', 0)}")
                return True
        except requests.ConnectionError:
            pass
        except Exception as e:
            print(f"[WARN] Server check error: {e}")
        return False

    def detect(self, frame: np.ndarray) -> dict | None:
        """
        Send a frame to /api/detect and return the response.
        Returns dict with keys: image (base64), detections, inference_ms, model
        """
        # Encode frame as JPEG
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_bytes = buf.tobytes()

        try:
            r = self.session.post(
                self.detect_url,
                files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return r.json()
            else:
                print(f"[WARN] Server returned {r.status_code}: {r.text[:200]}")
                return None
        except requests.ConnectionError:
            print("[WARN] Lost connection to server")
            return None
        except requests.Timeout:
            print("[WARN] Server request timed out")
            return None

    def push_frame(self, frame: np.ndarray):
        """Push a raw frame to the server for web client streaming (best-effort)."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        try:
            self.session.post(
                self.stream_url,
                files={"frame": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                timeout=2,
            )
        except Exception:
            pass  # Best-effort, don't block capture loop


# ═══════════════════════════════════════════════════════════════════════════
# Main Capture + Detect Loop
# ═══════════════════════════════════════════════════════════════════════════

def decode_b64_image(b64_string: str) -> np.ndarray | None:
    """Decode a base64 data URI image to numpy BGR array."""
    try:
        # Strip data URI prefix if present
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_string)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def run_capture_loop(args):
    """Main capture-detect loop."""
    # ── Setup camera ─────────────────────────────────────────
    camera = MindVisionCamera(
        camera_index=args.camera_index,
        exposure_ms=args.exposure,
    )

    # List cameras
    cameras = camera.list_cameras()
    if not cameras:
        print("[ERROR] No MindVision cameras detected. Check USB connection and udev rules.")
        print("  Hint: Copy sdkmindvision/88-mvusb.rules to /etc/udev/rules.d/ and reload:")
        print("    sudo cp sdkmindvision/88-mvusb.rules /etc/udev/rules.d/")
        print("    sudo udevadm control --reload-rules && sudo udevadm trigger")
        return

    print(f"\n[INFO] Found {len(cameras)} MindVision camera(s):")
    for cam in cameras:
        print(f"  [{cam['index']}] {cam['name']} (port: {cam['port']})")
    print()

    if not camera.open():
        return

    # ── Setup detection client ────────────────────────────────
    client = DetectionClient(args.server_url, timeout=args.timeout)

    print(f"[INFO] Connecting to server: {args.server_url}")
    if not client.check_server():
        print(f"[WARN] Server not reachable at {args.server_url}")
        print("  Detection will retry on each frame. Start the server and continue.")
    print()

    # ── Save directory ────────────────────────────────────────
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving annotated frames to: {save_dir}")

    # ── Graceful shutdown ─────────────────────────────────────
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n[INFO] Stopping capture...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ── Stats ─────────────────────────────────────────────────
    frame_count = 0
    detect_count = 0
    total_inference_ms = 0
    fps_start = time.time()
    fps_frames = 0

    print("[INFO] Starting capture loop — press Ctrl+C to stop" +
          (", 'q' in window to quit" if args.show else ""))
    print("=" * 60)

    # ── Main loop ─────────────────────────────────────────────
    try:
        while running:
            frame = camera.grab(timeout_ms=500)
            if frame is None:
                continue

            frame_count += 1
            fps_frames += 1

            # Calculate FPS every 2 seconds
            elapsed = time.time() - fps_start
            if elapsed >= 2.0:
                fps = fps_frames / elapsed
                avg_inf = (total_inference_ms / detect_count) if detect_count > 0 else 0
                print(f"\r[STATS] FPS: {fps:.1f} | Frames: {frame_count} | "
                      f"Detections: {detect_count} | Avg inference: {avg_inf:.0f}ms", end="", flush=True)
                fps_frames = 0
                fps_start = time.time()

            # Skip frames to maintain throughput (detect every N frames)
            if args.detect_interval > 1 and (frame_count % args.detect_interval != 0):
                if args.show:
                    display = cv2.resize(frame, (args.display_width, args.display_height))
                    cv2.imshow("MindVision Camera (raw)", display)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
                continue

            # Send to server for detection
            result = client.detect(frame)

            if result:
                detect_count += 1
                total_inference_ms += result.get("inference_ms", 0)

                # Decode annotated image from server
                annotated = decode_b64_image(result.get("image", ""))
                detections = result.get("detections", [])
                model_name = result.get("model", "unknown")
                inf_ms = result.get("inference_ms", 0)

                if detections:
                    det_summary = ", ".join(
                        f"{d.get('class', '?')}:{d.get('confidence', 0):.2f}"
                        for d in detections[:5]
                    )
                    print(f"\n[DET] Frame {frame_count} | {model_name} | "
                          f"{inf_ms:.0f}ms | {len(detections)} detection(s): {det_summary}")

                # Show in local window
                if args.show and annotated is not None:
                    display = cv2.resize(annotated, (args.display_width, args.display_height))
                    # Overlay stats
                    cv2.putText(display, f"Model: {model_name}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, f"Inference: {inf_ms:.0f}ms", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, f"Detections: {len(detections)}", (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("MindVision + Detection", display)

                # Save to disk
                if save_dir and annotated is not None:
                    fname = save_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(fname), annotated)

            elif args.show:
                # No detection result — show raw frame
                display = cv2.resize(frame, (args.display_width, args.display_height))
                cv2.putText(display, "No server response", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("MindVision Camera (raw)", display)

            if args.show:
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

    finally:
        print(f"\n\n[INFO] Capture ended — {frame_count} frames, {detect_count} detections")
        camera.close()
        if args.show:
            cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="MindVision USB Camera — Capture frames and detect chignons via the Flask server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Default: auto camera, server on localhost:5000
  %(prog)s --show                             # Open local OpenCV preview window
  %(prog)s --server-url http://192.168.1.10:5000
  %(prog)s --exposure 50 --camera-index 1     # 50ms exposure, second camera
  %(prog)s --detect-interval 3 --show         # Detect every 3rd frame (faster preview)
  %(prog)s --save-dir ./output_frames         # Save every annotated frame to disk

USB Setup (if camera is not detected):
  sudo cp sdkmindvision/88-mvusb.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules && sudo udevadm trigger
        """,
    )
    parser.add_argument("--server-url", default="http://localhost:5000",
                        help="Detection server URL (default: http://localhost:5000)")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="MindVision camera index (default: 0)")
    parser.add_argument("--exposure", type=float, default=30.0,
                        help="Exposure time in milliseconds (default: 30)")
    parser.add_argument("--detect-interval", type=int, default=1,
                        help="Run detection every N frames (default: 1 = every frame)")
    parser.add_argument("--show", action="store_true",
                        help="Show live preview in OpenCV window")
    parser.add_argument("--display-width", type=int, default=960,
                        help="Preview window width (default: 960)")
    parser.add_argument("--display-height", type=int, default=720,
                        help="Preview window height (default: 720)")
    parser.add_argument("--save-dir", default=None,
                        help="Save annotated frames to this directory")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Server request timeout in seconds (default: 10)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_capture_loop(args)
