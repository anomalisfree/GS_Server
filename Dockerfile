# =============================================================================
# GS Training Server – Multi-stage Docker build
#
# Stage 1 (brush-builder): compiles Brush from Rust source
# Stage 2 (runtime):       CUDA + Vulkan + COLMAP + Python FastAPI server
#
# Build:
#   docker build -t gs-server .
#
# Run locally (GPU):
#   docker run --gpus all -p 8080:8080 -v gs-data:/data gs-server
# =============================================================================

# ─── Stage 1: build Brush ─────────────────────────────────────────────────────
FROM rust:1.82-bookworm AS brush-builder

WORKDIR /build

# System deps needed to compile Brush (Vulkan headers, pkg-config for wgpu)
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libvulkan-dev \
        cmake \
        clang \
    && rm -rf /var/lib/apt/lists/*

# Copy only brush source so Docker layer cache stays valid when server code changes
COPY brush/ .

# Build the headless CLI binary (brush-app produces the `brush` binary)
RUN cargo build --release -p brush-app 2>&1


# ─── Stage 2: runtime image ────────────────────────────────────────────────────
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Expose GPU to both CUDA (COLMAP) and Vulkan (Brush)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
# Tell Vulkan loader to use the NVIDIA ICD (mounted by nvidia-container-toolkit)
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

RUN apt-get update && apt-get install -y --no-install-recommends \
        # ── Vulkan for Brush ──────────────────────
        libvulkan1 \
        mesa-vulkan-drivers \
        # ── COLMAP (structure-from-motion) ────────
        colmap \
        # ── Python 3.11 ───────────────────────────
        python3.11 \
        python3.11-dev \
        python3-pip \
        # ── Utilities ────────────────────────────
        curl \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m pip install --upgrade pip

# Copy pre-built Brush binary from Stage 1
COPY --from=brush-builder /build/target/release/brush /usr/local/bin/brush
RUN chmod +x /usr/local/bin/brush

# ─── Python server ─────────────────────────────────────────────────────────────
WORKDIR /app

COPY gs_server/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy server source (changes more often than deps → keep after pip install)
COPY gs_server/ gs_server/

# Persistent data directories (override with -v /host/path:/data)
RUN mkdir -p /data/jobs /data/uploads /data/models
VOLUME ["/data"]

# ─── Environment defaults ──────────────────────────────────────────────────────
ENV GS_HOST=0.0.0.0
ENV GS_PORT=8080
ENV GS_JOBS_DIR=/data/jobs
ENV GS_UPLOADS_DIR=/data/uploads
ENV GS_MODELS_DIR=/data/models
ENV GS_COLMAP_EXE=/usr/bin/colmap
ENV GS_BRUSH_EXE=/usr/local/bin/brush
ENV COLMAP_USE_GPU=true
ENV MASKING_ENABLED=true

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

CMD ["python3.11", "-m", "gs_server"]
