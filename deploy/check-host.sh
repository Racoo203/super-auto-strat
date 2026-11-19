#!/usr/bin/env bash
# Preflight for the Ubuntu training server: checks everything `make up` relies on.
# Prints what's missing and how to fix it; changes nothing.
set -u
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }
FAILED=0
cd "$(dirname "$0")/.."

echo "Host"
. /etc/os-release 2>/dev/null && ok "$PRETTY_NAME" || warn "unknown OS"
ok "$(nproc) CPU threads, $(free -g | awk '/Mem:/{print $2}') GiB RAM, $(df -h . | awk 'NR==2{print $4}') free disk here"

echo "NVIDIA driver"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  ok "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1), driver $drv"
  [ "${drv%%.*}" -ge 570 ] || bad "driver $drv < 570: torch 2.10 ships CUDA 12.8 (sudo ubuntu-drivers install)"
else
  bad "nvidia-smi not working: sudo ubuntu-drivers install && sudo reboot"
fi

echo "Docker"
if command -v docker >/dev/null; then
  ok "$(docker --version)"
  docker compose version >/dev/null 2>&1 && ok "$(docker compose version)" || bad "docker compose plugin missing (install docker-compose-plugin)"
  if docker info >/dev/null 2>&1; then ok "daemon reachable as $(whoami)"
  else bad "can't reach the daemon as $(whoami): sudo usermod -aG docker $USER, then log out and in"; fi
  systemctl is-enabled docker >/dev/null 2>&1 && ok "docker starts on boot" || warn "docker not enabled on boot: sudo systemctl enable docker"
else
  bad "docker not installed: https://docs.docker.com/engine/install/ubuntu/"
fi

echo "GPU inside containers"
if docker info 2>/dev/null | grep -qi nvidia; then
  if docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L >/dev/null 2>&1; then ok "containers see the GPU"
  else bad "nvidia runtime registered but a test container can't see the GPU"; fi
else
  bad "NVIDIA Container Toolkit not configured: see deploy/README.md step 2"
fi

echo "Repo"
[ -f .env ] && ok ".env present (RUN_NAME=$(grep -E '^RUN_NAME=' .env | cut -d= -f2))" || warn "no .env yet: cp .env.example .env"
git diff --quiet HEAD 2>/dev/null && ok "clean checkout at $(git rev-parse --short HEAD)" || warn "uncommitted changes: runs will be tagged -dirty"

echo
[ "$FAILED" = 0 ] && echo "All checks passed. Next: make test && make smoke && make up" || { echo "Fix the FAIL items above."; exit 1; }
