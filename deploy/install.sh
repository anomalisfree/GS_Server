#!/usr/bin/env bash
# =============================================================================
# deploy/install.sh
#
# Full server installation script – run this ONCE on the GCE VM after SSH-ing in.
# Safe to re-run: all steps are idempotent.
#
# What it does:
#   1. Installs NVIDIA drivers + CUDA toolkit
#   2. Installs Vulkan support (for Brush headless GPU rendering)
#   3. Installs COLMAP (via apt)
#   4. Installs Rust toolchain and builds Brush
#   5. Installs Python 3.11 + server dependencies
#   6. Mounts the data disk to /data
#   7. Creates a systemd service so the server starts on boot
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/gs_server}"
BRUSH_DIR="${BRUSH_DIR:-$HOME/brush}"
BRUSH_REPO="https://github.com/ArthurBrussee/brush"
BRUSH_TAG="v0.3.0"                        # версия совместимая с gs_server
DATA_DEVICE="${DATA_DEVICE:-/dev/sdb}"   # secondary disk attached by create_vm.sh
DATA_MOUNT="/data"
SERVICE_NAME="gs-server"
SERVER_PORT=8080

log() { echo -e "\n\033[1;34m>>> $*\033[0m"; }
ok()  { echo -e "\033[1;32m[OK] $*\033[0m"; }

# ── 0. Sanity check ───────────────────────────────────────────────────────────
if [[ ! -d "$REPO_DIR/gs_server" ]]; then
  echo "ERROR: gs_server directory not found: $REPO_DIR/gs_server"
  echo "Upload the server code first:"
  echo "  gcloud compute scp --recurse gs_server gs-training-server:~/gs_server/ --zone=ZONE"
  exit 1
fi

# ── 1. System updates ─────────────────────────────────────────────────────────
log "Updating system packages..."
sudo apt-get update -q
sudo apt-get upgrade -y -q
sudo apt-get install -y --no-install-recommends \
    curl wget git ca-certificates \
    build-essential pkg-config cmake clang \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    libvulkan1 mesa-vulkan-drivers \
    colmap
ok "System packages installed."

# ── 2. NVIDIA drivers ─────────────────────────────────────────────────────────
log "Checking NVIDIA drivers..."
if ! command -v nvidia-smi &>/dev/null; then
  echo "  Installing NVIDIA drivers (this may take several minutes)..."
  sudo apt-get install -y linux-headers-"$(uname -r)" software-properties-common
  # Ubuntu 22.04 driver 535 supports L4 and T4 GPUs
  sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit
  echo "  NVIDIA drivers installed. Reboot required."
  echo ""
  echo "  Run: sudo reboot"
  echo "  Then re-run this script after reboot."
  exit 0
else
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  ok "NVIDIA drivers already present."
fi

# ── 3. CUDA (nvidia-cuda-toolkit) ─────────────────────────────────────────────
log "Checking CUDA..."
if ! nvcc --version &>/dev/null; then
  sudo apt-get install -y nvidia-cuda-toolkit
fi
ok "CUDA ready: $(nvcc --version | grep release | awk '{print $5}' | tr -d ,)"

# ── 4. Rust toolchain ─────────────────────────────────────────────────────────
log "Installing/updating Rust toolchain..."
if ! command -v cargo &>/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
fi
# shellcheck source=/dev/null
source "$HOME/.cargo/env"
rustup update stable
ok "Rust: $(rustc --version)"

# ── 5. Clone and build Brush ──────────────────────────────────────────────────
log "Cloning Brush from GitHub ($BRUSH_TAG)..."
if [[ ! -d "$BRUSH_DIR" ]]; then
  git clone --depth=1 --branch "$BRUSH_TAG" "$BRUSH_REPO" "$BRUSH_DIR"
else
  ok "Brush repo already exists — skipping clone."
fi

log "Building Brush (this takes 10–20 minutes on first run)..."
cd "$BRUSH_DIR"
export CARGO_BUILD_JOBS="$(nproc)"
cargo build --release -p brush-app 2>&1 | tail -10

# Binary may be named 'brush' or 'brush_app' depending on the version
for candidate in brush brush_app; do
  if [[ -f "$BRUSH_DIR/target/release/$candidate" ]]; then
    BRUSH_BIN="$BRUSH_DIR/target/release/$candidate"
    break
  fi
done
if [[ -z "${BRUSH_BIN:-}" ]]; then
  echo "ERROR: Brush binary not found after build!"
  exit 1
fi
ok "Brush binary: $BRUSH_BIN"

# ── 6. Python dependencies ────────────────────────────────────────────────────
log "Installing Python server dependencies..."
cd "$REPO_DIR"
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r gs_server/requirements.txt
ok "Python dependencies installed."

# ── 7. Mount data disk ────────────────────────────────────────────────────────
log "Setting up data disk ($DATA_DEVICE → $DATA_MOUNT)..."
if ! mountpoint -q "$DATA_MOUNT"; then
  if [[ -b "$DATA_DEVICE" ]]; then
    # Format only if no filesystem exists yet
    if ! blkid "$DATA_DEVICE" &>/dev/null; then
      sudo mkfs.ext4 -F "$DATA_DEVICE"
    fi
    sudo mkdir -p "$DATA_MOUNT"
    sudo mount "$DATA_DEVICE" "$DATA_MOUNT"
    # Persist across reboots
    DISK_UUID=$(sudo blkid -s UUID -o value "$DATA_DEVICE")
    if ! grep -q "$DISK_UUID" /etc/fstab; then
      echo "UUID=$DISK_UUID  $DATA_MOUNT  ext4  defaults,nofail  0  2" | sudo tee -a /etc/fstab
    fi
    ok "Data disk mounted at $DATA_MOUNT"
  else
    echo "  WARNING: $DATA_DEVICE not found – using local /data directory."
    sudo mkdir -p "$DATA_MOUNT"
  fi
else
  ok "Data disk already mounted."
fi

sudo mkdir -p "$DATA_MOUNT/jobs" "$DATA_MOUNT/uploads" "$DATA_MOUNT/models"
sudo chown -R "$USER:$USER" "$DATA_MOUNT"

# ── 8. Create systemd service ──────────────────────────────────────────────────
log "Creating systemd service: $SERVICE_NAME..."
# shellcheck source=/dev/null
source "$HOME/.cargo/env"

sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" > /dev/null <<EOF
[Unit]
Description=Gaussian Splatting Training Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR
Environment="GS_HOST=0.0.0.0"
Environment="GS_PORT=$SERVER_PORT"
Environment="GS_JOBS_DIR=$DATA_MOUNT/jobs"
Environment="GS_UPLOADS_DIR=$DATA_MOUNT/uploads"
Environment="GS_MODELS_DIR=$DATA_MOUNT/models"
Environment="GS_COLMAP_EXE=/usr/bin/colmap"
Environment="GS_BRUSH_EXE=$BRUSH_BIN"
Environment="COLMAP_USE_GPU=true"
Environment="MASKING_ENABLED=true"
Environment="PATH=$HOME/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=python3.11 -m gs_server
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=gs-server

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

# ── Done ──────────────────────────────────────────────────────────────────────
EXTERNAL_IP=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip" -H "Metadata-Flavor: Google" 2>/dev/null || hostname -I | awk '{print $1}')

echo ""
echo "============================================================"
echo "  GS Training Server is running!"
echo "============================================================"
echo "  API:  http://${EXTERNAL_IP}:${SERVER_PORT}"
echo "  Docs: http://${EXTERNAL_IP}:${SERVER_PORT}/docs"
echo ""
echo "  Useful commands:"
echo "    sudo systemctl status  $SERVICE_NAME   # check status"
echo "    sudo journalctl -fu    $SERVICE_NAME   # live logs"
echo "    sudo systemctl restart $SERVICE_NAME   # restart"
echo "============================================================"
