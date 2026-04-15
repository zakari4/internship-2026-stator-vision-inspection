#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# Chignon Detection — Quick Run Script
# ══════════════════════════════════════════════════════════════
# Starts the Flask server which serves both the backend API and
# the client UI on a single port.
#
# Usage:
#   ./run.sh                  Start with defaults (port 5001)
#   ./run.sh --port 8080      Custom port
#   ./run.sh --model yolov8n  Load a specific model at startup
#   ./run.sh --debug          Flask debug / hot-reload mode
#   ./run.sh --mindvision     Start server + MindVision capture pipeline
#   ./run.sh --mindvision --show   Enable OpenCV preview in capture script
# ══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/server"
CLIENT_DIR="$SCRIPT_DIR/client"
DEFAULT_PORT=5001
# MLflow UI (from docker-compose) is published on host port 5002
MLFLOW_UI_PORT="${MLFLOW_UI_PORT:-5002}"
MINDVISION_MODE=false
MINDVISION_SHOW=false
CAPTURE_INTERVAL=5

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✔]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✘]${NC} $*" >&2; }
info() { echo -e "${CYAN}[→]${NC} $*"; }

# ── Parse arguments (pass-through to server.py) ─────────────
PORT="$DEFAULT_PORT"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            EXTRA_ARGS+=("--port" "$2")
            shift 2
            ;;
        --mindvision)
            MINDVISION_MODE=true
            shift
            ;;
        --show)
            MINDVISION_SHOW=true
            shift
            ;;
        --detect-interval)
            CAPTURE_INTERVAL="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Add default port if not explicitly given
if ! printf '%s\n' "${EXTRA_ARGS[@]}" 2>/dev/null | grep -q -- '--port'; then
    EXTRA_ARGS+=("--port" "$DEFAULT_PORT")
fi

# ── Pre-flight checks ───────────────────────────────────────
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Chignon Detection — Test Runner${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""

# Check Python
if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
    err "Python not found. Install Python 3.9+ first."
    exit 1
fi
PYTHON=$(command -v python3 2>/dev/null || command -v python)
log "Python: $($PYTHON --version)"

# Check server files
if [[ ! -f "$SERVER_DIR/server.py" ]]; then
    err "server/server.py not found. Run this script from the project root."
    exit 1
fi

if [[ ! -f "$CLIENT_DIR/index.html" ]]; then
    err "client/index.html not found."
    exit 1
fi

# Check dependencies
info "Checking Python dependencies…"
if ! $PYTHON -c "import flask, flask_cors, aiortc, cv2" 2>/dev/null; then
    warn "Some dependencies are missing. Installing from server/requirements.txt…"
    $PYTHON -m pip install -q -r "$SERVER_DIR/requirements.txt"
    log "Dependencies installed."
else
    log "All dependencies available."
fi

# ── Set environment variables ────────────────────────────────
export PROJECT_ROOT="$SCRIPT_DIR"
export CLIENT_DIR="$CLIENT_DIR"

MV_LIB_DIR="$SCRIPT_DIR/sdkmindvision/lib/x64"

# Prefer local SDK library path for MindVision runtime
if [[ -d "$MV_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$MV_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

wait_for_server() {
    local max_tries=30
    local delay=1
    local url="http://127.0.0.1:${PORT}/api/status"
    local i

    for ((i=1; i<=max_tries; i++)); do
        if $PYTHON -c "import requests; requests.get('${url}', timeout=1).raise_for_status()" 2>/dev/null; then
            return 0
        fi
        sleep "$delay"
    done
    return 1
}

# ── Start ────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}───────────────────────────────────────────────────${NC}"
info "App (frontend)   : http://localhost:${PORT}        (port ${PORT})"
info "API (backend)    : http://localhost:${PORT}/api    (port ${PORT})"
info "API docs         : http://localhost:${PORT}/api/docs"
info "API status       : http://localhost:${PORT}/api/status"
info "MLflow UI        : http://localhost:${MLFLOW_UI_PORT}  (port ${MLFLOW_UI_PORT}, via docker-compose)"
echo -e "${CYAN}───────────────────────────────────────────────────${NC}"
info "Press Ctrl+C to stop."
echo ""

if [[ "$MINDVISION_MODE" == "true" ]]; then
    if [[ ! -f "$SERVER_DIR/mindvision_capture.py" ]]; then
        err "server/mindvision_capture.py not found."
        exit 1
    fi

    if [[ ! -d "$MV_LIB_DIR" ]]; then
        warn "MindVision SDK lib dir not found at: $MV_LIB_DIR"
        warn "If you see libMVSDK.so errors, install/copy SDK libraries there."
    else
        log "MindVision SDK path added: $MV_LIB_DIR"
    fi

    info "Starting Flask server in background..."
    cd "$SERVER_DIR"
    $PYTHON server.py "${EXTRA_ARGS[@]}" &
    SERVER_PID=$!

    cleanup() {
        if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
            info "Stopping server (PID $SERVER_PID)..."
            kill "$SERVER_PID" 2>/dev/null || true
            wait "$SERVER_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT INT TERM

    info "Waiting for server health check..."
    if ! wait_for_server; then
        err "Server did not become ready at http://127.0.0.1:${PORT}/api/status"
        exit 1
    fi
    log "Server is ready."

    CAPTURE_ARGS=(
        "--server-url" "http://127.0.0.1:${PORT}"
        "--detect-interval" "$CAPTURE_INTERVAL"
    )

    if [[ "$MINDVISION_SHOW" == "true" ]]; then
        CAPTURE_ARGS+=("--show")
    fi

    echo ""
    info "Starting MindVision capture..."
    info "Capture URL      : http://127.0.0.1:${PORT}"
    info "Detect interval  : ${CAPTURE_INTERVAL}"
    [[ "$MINDVISION_SHOW" == "true" ]] && warn "--show requires GUI-enabled OpenCV."
    echo ""

    exec $PYTHON "$SERVER_DIR/mindvision_capture.py" "${CAPTURE_ARGS[@]}"
else
    cd "$SERVER_DIR"
    exec $PYTHON server.py "${EXTRA_ARGS[@]}"
fi
