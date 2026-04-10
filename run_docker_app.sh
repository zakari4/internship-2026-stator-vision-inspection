#!/usr/bin/env bash
# Chignon Detection — Docker app runner
#
# Runs or stops the combined Docker image, with optional MindVision capture mode.
#
# Usage:
#   ./run_docker_app.sh run
#   ./run_docker_app.sh run --mindvision
#   ./run_docker_app.sh run --mindvision --camera-index 1 --detect-interval 3
#   ./run_docker_app.sh shutdown
#   ./run_docker_app.sh status
#   ./run_docker_app.sh logs

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-chignon-server-app:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-chignon-server-app}"
HOST_PORT="${HOST_PORT:-5000}"

ENABLE_MINDVISION=false
MV_CAMERA_INDEX=0
MV_DETECT_INTERVAL=1
MV_EXPOSURE=30
MV_SHOW=false
BUILD_IF_MISSING=true

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*" >&2; }
info() { echo -e "${CYAN}[INFO]${NC} $*"; }

usage() {
  cat <<EOF
Usage:
  $(basename "$0") run [options]
  $(basename "$0") shutdown
  $(basename "$0") status
  $(basename "$0") logs

Commands:
  run        Build (if needed) and start the app container
  shutdown   Stop and remove the app container
  status     Show container status
  logs       Follow container logs

Run options:
  --mindvision              Enable MindVision USB camera capture inside container
  --camera-index N          MindVision camera index (default: 0)
  --detect-interval N       Detect every Nth frame (default: 1)
  --exposure MS             Exposure in ms (default: 30)
  --show                    Enable OpenCV preview in capture process
  --port N                  Host port mapped to container 5000 (default: 5000)
  --no-build                Do not build image if missing

Env overrides:
  IMAGE_NAME, CONTAINER_NAME, HOST_PORT
EOF
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    err "Docker is not installed."
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    err "Docker daemon is not reachable. Try using sudo or start Docker service."
    exit 1
  fi
}

image_exists() {
  docker image inspect "$IMAGE_NAME" >/dev/null 2>&1
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"
}

container_running() {
  docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"
}

build_image_if_needed() {
  if image_exists; then
    log "Image found: $IMAGE_NAME"
    return 0
  fi

  if [[ "$BUILD_IF_MISSING" != "true" ]]; then
    err "Image $IMAGE_NAME not found and --no-build was set."
    exit 1
  fi

  info "Image not found. Building from Dockerfile.server_app..."
  (cd "$PROJECT_DIR" && docker build -f Dockerfile.server_app -t "$IMAGE_NAME" .)
  log "Build completed: $IMAGE_NAME"
}

start_capture_in_container() {
  local capture_cmd
  capture_cmd="export LD_LIBRARY_PATH=/app/sdkmindvision/lib/x64:\${LD_LIBRARY_PATH:-}; \
python /app/server/mindvision_capture.py \
  --server-url http://127.0.0.1:5000 \
  --camera-index ${MV_CAMERA_INDEX} \
  --detect-interval ${MV_DETECT_INTERVAL} \
  --exposure ${MV_EXPOSURE}"

  if [[ "$MV_SHOW" == "true" ]]; then
    capture_cmd+=" --show"
    warn "--show in container requires GUI/X11 access."
  fi

  info "Starting MindVision capture process in container..."
  docker exec -d "$CONTAINER_NAME" bash -lc "$capture_cmd"
  log "MindVision capture started."
}

run_container() {
  build_image_if_needed

  if container_exists; then
    info "Removing existing container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi

  local -a run_args
  run_args=(
    -d
    --name "$CONTAINER_NAME"
    --restart unless-stopped
    -p "${HOST_PORT}:5000"
    -e PROJECT_ROOT=/app
    -e MODEL_DIR=/app
    -e CLIENT_DIR=/app/client
    -v "$PROJECT_DIR/outputs/results:/app/outputs/results:ro"
    -v "$PROJECT_DIR/chignon/results:/app/chignon/results:ro"
    -v "$PROJECT_DIR/files/results:/app/files/results:ro"
    -v "$PROJECT_DIR/weights:/app/weights:ro"
  )

  if [[ -d "$PROJECT_DIR/sdkmindvision" ]]; then
    run_args+=( -v "$PROJECT_DIR/sdkmindvision:/app/sdkmindvision:ro" )
  else
    warn "sdkmindvision directory not found. MindVision mode may fail."
  fi

  if [[ "$ENABLE_MINDVISION" == "true" ]]; then
    info "MindVision mode enabled: adding USB device access"
    run_args+=( --privileged )
    if [[ -e /dev/bus/usb ]]; then
      run_args+=( --device /dev/bus/usb:/dev/bus/usb )
    else
      warn "/dev/bus/usb not present on host."
    fi
    if [[ -e /dev/video0 ]]; then
      run_args+=( --device /dev/video0:/dev/video0 )
    fi
  fi

  info "Starting container on http://localhost:${HOST_PORT}"
  docker run "${run_args[@]}" "$IMAGE_NAME" >/dev/null
  log "Container started: $CONTAINER_NAME"

  if [[ "$ENABLE_MINDVISION" == "true" ]]; then
    start_capture_in_container
  fi

  echo ""
  log "Server URL: http://localhost:${HOST_PORT}"
  info "Check status: curl http://localhost:${HOST_PORT}/api/status"
}

shutdown_container() {
  if container_exists; then
    docker rm -f "$CONTAINER_NAME" >/dev/null
    log "Container removed: $CONTAINER_NAME"
  else
    warn "Container not found: $CONTAINER_NAME"
  fi
}

show_status() {
  if container_running; then
    log "Container is running: $CONTAINER_NAME"
    docker ps --filter "name=^${CONTAINER_NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  elif container_exists; then
    warn "Container exists but is not running: $CONTAINER_NAME"
    docker ps -a --filter "name=^${CONTAINER_NAME}$" --format 'table {{.Names}}\t{{.Status}}'
  else
    warn "Container does not exist: $CONTAINER_NAME"
  fi
}

show_logs() {
  if container_exists; then
    docker logs -f "$CONTAINER_NAME"
  else
    err "Container not found: $CONTAINER_NAME"
    exit 1
  fi
}

main() {
  local cmd="${1:-}"
  shift || true

  case "$cmd" in
    run)
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --mindvision)
            ENABLE_MINDVISION=true
            shift
            ;;
          --camera-index)
            MV_CAMERA_INDEX="$2"
            shift 2
            ;;
          --detect-interval)
            MV_DETECT_INTERVAL="$2"
            shift 2
            ;;
          --exposure)
            MV_EXPOSURE="$2"
            shift 2
            ;;
          --show)
            MV_SHOW=true
            shift
            ;;
          --port)
            HOST_PORT="$2"
            shift 2
            ;;
          --no-build)
            BUILD_IF_MISSING=false
            shift
            ;;
          -h|--help)
            usage
            exit 0
            ;;
          *)
            err "Unknown option for run: $1"
            usage
            exit 1
            ;;
        esac
      done

      require_docker
      run_container
      ;;

    shutdown)
      require_docker
      shutdown_container
      ;;

    status)
      require_docker
      show_status
      ;;

    logs)
      require_docker
      show_logs
      ;;

    -h|--help|help|"")
      usage
      ;;

    *)
      err "Unknown command: $cmd"
      usage
      exit 1
      ;;
  esac
}

main "$@"
