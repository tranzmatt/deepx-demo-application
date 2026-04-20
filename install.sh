#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✔]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
fail() { echo -e "${RED}[✘]${NC} $1"; exit 1; }

REPO_URL="https://github.com/sixfab/deepx-demo-application.git"
REPO_DIR="deepx-demo-application"
BRANCH="master"

if ! command -v git >/dev/null 2>&1; then
    warn "git is not installed. Installing..."
    if command -v apt >/dev/null 2>&1; then
        sudo apt update -qq
        sudo apt install -y git || fail "Failed to install git"
    else
        fail "apt not found. Please install git manually and re-run."
    fi
    log "git installed."
else
    log "git found: $(git --version)"
fi

if [ -d "$REPO_DIR/.git" ]; then
    warn "$REPO_DIR already exists, pulling latest changes..."
    cd "$REPO_DIR"
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    log "Cloning $REPO_URL..."
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR" || fail "Failed to clone repository"
    cd "$REPO_DIR"
fi

[ -f "build.sh" ] || fail "build.sh not found in repository."
chmod +x build.sh

log "Running build.sh..."
./build.sh

log "Installation complete! Run the app with: ./$REPO_DIR/build/cpp_qt_app"

