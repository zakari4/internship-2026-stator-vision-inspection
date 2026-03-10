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
# ══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/server"
CLIENT_DIR="$SCRIPT_DIR/client"
DEFAULT_PORT=5001

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

# ── Start ────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}───────────────────────────────────────────────────${NC}"
info "Server (backend) : http://localhost:${PORT}"
info "Frontend (client): http://localhost:${PORT}"
info "API status       : http://localhost:${PORT}/api/status"
echo -e "${CYAN}───────────────────────────────────────────────────${NC}"
info "Press Ctrl+C to stop."
echo ""

cd "$SERVER_DIR"
exec $PYTHON server.py "${EXTRA_ARGS[@]}"
