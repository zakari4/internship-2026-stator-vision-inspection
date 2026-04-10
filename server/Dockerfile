# ══════════════════════════════════════════════════════════════
# Chignon Detection Server — Optimized Multi-Stage Dockerfile
# ══════════════════════════════════════════════════════════════
# Stage 1: Build — install Python deps with compilers
# Stage 2: Runtime — slim image with only runtime libs + wheels
# Only best.pt weights are copied (not training artifacts)
# ══════════════════════════════════════════════════════════════

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ pkg-config \
        libavdevice-dev libavfilter-dev \
        libopus-dev libvpx-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY server/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim

# Runtime-only system libs (no compilers) — Debian 13 Trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
        libsm6 libxext6 libxrender1 \
        libavdevice61 libavfilter10 \
        libopus0 libvpx9 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /root/.cache

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy project subset allowed by .dockerignore (server/client/domain code + results).
COPY . /app

# ── Environment ──────────────────────────────────────────────
ENV MODEL_DIR=/app \
    PROJECT_ROOT=/app \
    CLIENT_DIR=/app/client \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 5000

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/status')" || exit 1

# ── Run as non-root ─────────────────────────────────────────
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "server/server.py", "--host", "0.0.0.0", "--port", "5000"]
