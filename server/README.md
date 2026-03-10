# WebRTC Chignon Detection Server

Real-time industrial stator chignon detection served over WebRTC.
The client captures camera frames, streams them to the Flask server via WebRTC,
and receives annotated frames (with segmentation masks and bounding boxes)
plus detection metadata in real-time.

## Architecture

```
┌─────────────────────┐          WebRTC           ┌──────────────────────┐
│      Browser         │ ──── Video Track ────▶   │   Flask + aiortc     │
│   (client/)          │                          │   (server.py)        │
│                      │ ◀── Annotated Track ──── │                      │
│  Camera → localVideo │                          │  inference.py        │
│  remoteVideo ← AI   │ ◀── DataChannel JSON ─── │  ├─ ModelManager     │
│  detections list     │                          │  └─ YOLO / PyTorch   │
└─────────────────────┘                           └──────────────────────┘
```

**Signaling** is done over HTTP (`POST /offer`).  No external TURN/STUN
infrastructure is needed for local-network usage.

## Prerequisites

- Python 3.9+
- A trained model (YOLO weights in `outputs/results/yolo_training/`)
  or the pretrained weights in `weights/`
- A webcam or virtual camera

## Setup

```bash
# From the project root
cd server

# Install dependencies (use the project venv or a new one)
pip install -r requirements.txt

# Start the server
python server.py
```

The server starts on **http://localhost:5000** by default.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `5000` | Bind port |
| `--model` | *(auto)* | Model name to load at startup |
| `--debug` | off | Flask debug mode |

Example:

```bash
python server.py --port 8080 --model yolov26n_seg
```

## Usage

1. Open **http://localhost:5000** in a modern browser (Chrome/Edge recommended).
2. Select a model from the dropdown (the best trained model is auto-selected).
3. Click **Start Detection** — allow camera access when prompted.
4. The **Detection Result** panel shows the annotated frames in real-time.
5. Performance stats (inference time, FPS) and detections appear below.
6. Click **Snapshot** to download a frame as PNG.
7. Switch models on-the-fly with the dropdown — no reconnection needed.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Client UI |
| `GET` | `/api/models` | List available models |
| `POST` | `/api/select-model` | Switch active model `{"model": "name"}` |
| `POST` | `/offer` | WebRTC SDP offer → answer |
| `GET` | `/api/status` | Server health check |

## File Structure

```
server/
├── server.py           # Flask + WebRTC server
├── inference.py        # Model discovery, loading, inference
├── requirements.txt    # Python dependencies
└── README.md           # This file

client/                  # At project root
├── index.html          # Client UI
├── style.css           # Dark-themed styles
└── app.js              # WebRTC client logic
```

