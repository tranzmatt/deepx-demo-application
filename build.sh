#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✔]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
fail() { echo -e "${RED}[✘]${NC} $1"; exit 1; }

RESOURCES_URL="https://github.com/sixfab/deepx-demo-application/releases/download/resources-v1.0.0/resources.zip"

log "Installing dependencies..."
sudo apt update -qq
sudo apt install -y \
    qt6-base-dev \
    qt6-multimedia-dev \
    libopencv-dev \
    libcamera-dev \
    cmake \
    build-essential \
    wget \
    unzip

if [ -f "resources.zip" ]; then
    warn "resources.zip already exists, skipping download."
else
    log "Downloading resources.zip..."
    wget -q --show-progress "$RESOURCES_URL" -O resources.zip || fail "Failed to download resources.zip"
fi

log "Extracting resources.zip..."
unzip -o resources.zip -d . && rm -f resources.zip
log "Resources extracted to project root."

QT6_CMAKE_DIR=$(find /usr -name "Qt6Config.cmake" 2>/dev/null | head -1 | xargs dirname)

if [ -z "$QT6_CMAKE_DIR" ]; then
    fail "Qt6Config.cmake not found. Please check your Qt6 installation."
fi

log "Qt6 found: $QT6_CMAKE_DIR"

if [ -d "build" ]; then
    warn "Removing old build/ directory..."
    rm -rf build
fi

log "Creating build directory..."
mkdir build && cd build

log "Running CMake..."
cmake .. -DCMAKE_PREFIX_PATH="$QT6_CMAKE_DIR"

log "Building with $(nproc) cores..."
make -j$(nproc)

log "Build complete!"
