#!/usr/bin/env bash
set -euo pipefail
log() { echo -e "\n>>> $*"; }

log "Шаг 1: CUDA 12.1 Toolkit от NVIDIA"
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -q
sudo apt-get install -y cuda-toolkit-12-1 2>&1 | tail -5
export PATH=/usr/local/cuda-12.1/bin:$PATH
nvcc --version

log "Шаг 2: Зависимости COLMAP"
sudo apt-get install -y cmake libboost-all-dev libeigen3-dev libflann-dev \
  libfreeimage-dev libmetis-dev libgoogle-glog-dev libgflags-dev \
  libsqlite3-dev libglew-dev libcgal-dev libceres-dev 2>&1 | tail -3

log "Шаг 3: Клонируем COLMAP 3.10"
rm -rf ~/colmap_src
git clone --depth=1 --branch 3.10 https://github.com/colmap/colmap.git ~/colmap_src

log "Шаг 4: Конфигурируем с CUDA 12.1"
mkdir -p ~/colmap_src/build
cd ~/colmap_src/build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ENABLED=ON \
  -DGUI_ENABLED=OFF \
  -DTESTS_ENABLED=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc 2>&1 | tail -10

log "Шаг 5: Компилируем (~15-20 мин)"
make -j8 2>&1 | tail -5

log "Шаг 6: Устанавливаем"
sudo make install

log "=== COLMAP с CUDA 12.1 готов ==="
colmap help | head -2
