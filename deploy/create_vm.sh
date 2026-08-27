#!/usr/bin/env bash
# =============================================================================
# deploy/create_vm.sh
#
# Creates a Google Compute Engine VM with NVIDIA GPU for GS training.
# Run this ONCE on your local machine (requires gcloud CLI authenticated).
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   bash deploy/create_vm.sh [PROJECT_ID] [ZONE]
#
# Defaults:
#   PROJECT_ID  – read from gcloud config
#   ZONE        – us-central1-a  (L4 GPU available here)
# =============================================================================
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT="${1:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${2:-us-central1-a}"
INSTANCE_NAME="gs-training-server"
MACHINE_TYPE="g2-standard-8"          # 8 vCPU, 32 GB RAM + L4 GPU (24 GB VRAM)
GPU_TYPE="nvidia-l4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
DATA_DISK_NAME="gs-data-disk"
DATA_DISK_SIZE="500GB"
DISK_TYPE="pd-ssd"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
FIREWALL_TAG="gs-server"
SERVER_PORT="8080"

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ -z "$PROJECT" ]]; then
  echo "ERROR: No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

echo "========================================="
echo "  Creating GS Training Server on GCP"
echo "========================================="
echo "  Project  : $PROJECT"
echo "  Zone     : $ZONE"
echo "  Machine  : $MACHINE_TYPE + ${GPU_COUNT}x $GPU_TYPE"
echo "  Disk     : Boot ${BOOT_DISK_SIZE}  |  Data ${DATA_DISK_SIZE}"
echo "========================================="
echo ""

# ── Enable required APIs ─────────────────────────────────────────────────────
echo "[1/5] Enabling GCP APIs..."
gcloud services enable compute.googleapis.com \
    artifactregistry.googleapis.com \
    --project="$PROJECT" --quiet

# ── Create persistent data disk (separate so it survives VM deletion) ─────────
echo "[2/5] Creating persistent data disk..."
if gcloud compute disks describe "$DATA_DISK_NAME" --zone="$ZONE" --project="$PROJECT" &>/dev/null; then
  echo "  Data disk already exists – skipping."
else
  gcloud compute disks create "$DATA_DISK_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --size="$DATA_DISK_SIZE" \
    --type="$DISK_TYPE"
  echo "  Created: $DATA_DISK_NAME"
fi

# ── Create firewall rule ──────────────────────────────────────────────────────
echo "[3/5] Creating firewall rule (port ${SERVER_PORT})..."
if gcloud compute firewall-rules describe "allow-gs-server" --project="$PROJECT" &>/dev/null; then
  echo "  Firewall rule already exists – skipping."
else
  gcloud compute firewall-rules create "allow-gs-server" \
    --project="$PROJECT" \
    --direction=INGRESS \
    --action=ALLOW \
    --rules="tcp:${SERVER_PORT}" \
    --target-tags="$FIREWALL_TAG" \
    --description="Allow GS training server API"
  echo "  Created firewall rule: allow-gs-server"
fi

# ── Create the VM ─────────────────────────────────────────────────────────────
echo "[4/5] Creating VM instance..."
gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$BOOT_DISK_SIZE" \
  --boot-disk-type="$DISK_TYPE" \
  --disk="name=${DATA_DISK_NAME},mode=rw,auto-delete=no" \
  --tags="$FIREWALL_TAG" \
  --scopes="cloud-platform" \
  --metadata=startup-script-url=""

echo "  VM created: $INSTANCE_NAME"

# ── Print next steps ──────────────────────────────────────────────────────────
echo ""
echo "[5/5] Next steps:"
echo "----------------------------------------------------"
echo "  1) SSH into the VM:"
echo "     gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT"
echo ""
echo "  2) Upload the project (from your local machine):"
echo "     gcloud compute scp --recurse . $INSTANCE_NAME:~/gs_server \\"
echo "       --zone=$ZONE --project=$PROJECT"
echo ""
echo "  3) Run the install script on the VM:"
echo "     bash ~/gs_server/deploy/install.sh"
echo ""
echo "  4) Server will be available at:"
EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
  --zone="$ZONE" --project="$PROJECT" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || echo "<external-ip>")
echo "     http://${EXTERNAL_IP}:${SERVER_PORT}"
echo "     http://${EXTERNAL_IP}:${SERVER_PORT}/docs"
echo "----------------------------------------------------"
