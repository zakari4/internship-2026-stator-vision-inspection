#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# Chignon Detection — Docker Setup & Run
# ══════════════════════════════════════════════════════════════
# Usage:
#   ./setup_docker.sh          Build and start everything
#   ./setup_docker.sh build    Build images only
#   ./setup_docker.sh up       Start containers (assumes built)
#   ./setup_docker.sh down     Stop and remove containers
#   ./setup_docker.sh logs     Tail server logs
#   ./setup_docker.sh clean    Stop, remove containers + images
# ══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"
SERVER_IMAGE="chignon-server:latest"
SERVER_PORT=5000
CLIENT_PORT=8080

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()   { echo -e "${GREEN}[✔]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
err()   { echo -e "${RED}[✘]${NC} $*" >&2; }
info()  { echo -e "${CYAN}[→]${NC} $*"; }

# ── Pre-flight checks ───────────────────────────────────────
preflight() {
    if ! command -v docker &>/dev/null; then
        err "Docker is not installed. Install it from https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! docker info &>/dev/null; then
        err "Docker daemon is not running. Start it first."
        exit 1
    fi

    # Check compose
    if docker compose version &>/dev/null; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &>/dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        err "Docker Compose not found. Install it from https://docs.docker.com/compose/install/"
        exit 1
    fi

    log "Docker $(docker --version | grep -oP 'version \K[^,]+')"
    log "Compose $($COMPOSE_CMD version --short 2>/dev/null || $COMPOSE_CMD version | grep -oP 'v[\d.]+')"
}

# ── Verify model weights exist ───────────────────────────────
check_weights() {
    local missing=0

    info "Checking model weights..."

    for model in yolov8n_seg yolov8s_seg yolov8m_seg yolov11n_seg yolov11s_seg yolov11m_seg yolov26n_seg yolov26s_seg yolov26m_seg; do
        if [[ ! -f "$PROJECT_DIR/outputs/results/yolo_training/$model/weights/best.pt" ]]; then
            warn "Missing: outputs/results/yolo_training/$model/weights/best.pt"
            missing=$((missing + 1))
        fi
    done

    for f in rtdetr-l.pt rtdetr-x.pt; do
        if [[ ! -f "$PROJECT_DIR/$f" ]]; then
            warn "Missing: $f"
            missing=$((missing + 1))
        fi
    done

    if [[ ! -d "$PROJECT_DIR/weights" ]] || [[ -z "$(ls "$PROJECT_DIR/weights/"*.pt 2>/dev/null)" ]]; then
        warn "Missing: weights/*.pt pretrained models"
        missing=$((missing + 1))
    fi

    if [[ $missing -gt 0 ]]; then
        warn "$missing weight file(s) missing — container will run but those models won't be available"
    else
        log "All model weights found (9 trained + 9 pretrained + 2 RT-DETR)"
    fi
}

# ── Build ────────────────────────────────────────────────────
do_build() {
    info "Building server image (multi-stage, optimized)..."
    cd "$PROJECT_DIR"
    $COMPOSE_CMD -f "$COMPOSE_FILE" build --progress=plain
    log "Build complete"

    # Show image size
    local size
    size=$(docker images "$SERVER_IMAGE" --format "{{.Size}}" 2>/dev/null | head -1)
    if [[ -n "$size" ]]; then
        log "Server image size: $size"
    fi
}

# ── Up ───────────────────────────────────────────────────────
do_up() {
    info "Starting containers..."
    cd "$PROJECT_DIR"
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d

    echo ""
    log "Services running:"
    echo -e "   ${CYAN}Server${NC}  → http://localhost:${SERVER_PORT}"
    echo -e "   ${CYAN}Client${NC}  → http://localhost:${CLIENT_PORT}"
    echo ""
    info "Open ${CYAN}http://localhost:${CLIENT_PORT}${NC} in Chrome/Edge for the full app"
    info "Use '$(basename "$0") logs' to see server output"
    info "Use '$(basename "$0") down' to stop"
}

# ── Down ─────────────────────────────────────────────────────
do_down() {
    info "Stopping containers..."
    cd "$PROJECT_DIR"
    $COMPOSE_CMD -f "$COMPOSE_FILE" down
    log "Containers stopped and removed"
}

# ── Logs ─────────────────────────────────────────────────────
do_logs() {
    cd "$PROJECT_DIR"
    $COMPOSE_CMD -f "$COMPOSE_FILE" logs -f server
}

# ── Clean (remove images too) ────────────────────────────────
do_clean() {
    info "Stopping containers and removing images..."
    cd "$PROJECT_DIR"
    $COMPOSE_CMD -f "$COMPOSE_FILE" down --rmi local --volumes
    log "Cleaned up"
}

# ── Main ─────────────────────────────────────────────────────
main() {
    local cmd="${1:-}"

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Chignon Detection — Docker Setup${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
    echo ""

    preflight

    case "$cmd" in
        build)
            check_weights
            do_build
            ;;
        up)
            do_up
            ;;
        down)
            do_down
            ;;
        logs)
            do_logs
            ;;
        clean)
            do_clean
            ;;
        "")
            # Default: full setup — check, build, run
            check_weights
            do_build
            do_up
            ;;
        *)
            err "Unknown command: $cmd"
            echo "Usage: $(basename "$0") [build|up|down|logs|clean]"
            exit 1
            ;;
    esac
}

main "$@"
