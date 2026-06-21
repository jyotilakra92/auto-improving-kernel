#!/bin/bash
# CUDA 12.8 toolkit (nvcc) for autokernel + PyTorch cu128.
# Requires NVIDIA driver R550+ (CUDA 12.8 runtime). Driver 535 is too old.
set -euo pipefail

CUDA_RUN="cuda_12.8.0_570.86.10_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/${CUDA_RUN}"
INSTALL_DIR="/usr/local/cuda-12.8"
MIN_DRIVER_MAJOR=550

log() { echo "==> $*"; }

if [[ $(id -u) -eq 0 ]]; then
  echo "Run as normal user (script uses sudo where needed)." >&2
  exit 1
fi

log "GPU driver check"
if ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi failed — fix the GPU driver first." >&2
  exit 1
fi
nvidia-smi | head -3

driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
driver_major=${driver_ver%%.*}
log "Driver version: ${driver_ver}"

if (( driver_major < MIN_DRIVER_MAJOR )); then
  echo
  echo "ERROR: Driver ${driver_ver} is too old for CUDA 12.8 toolkit + PyTorch cu128."
  echo "Your nvidia-smi shows max CUDA ~12.2 on driver 535."
  echo
  echo "Upgrade the driver, reboot, then re-run this script:"
  echo "  sudo apt-get update"
  echo "  sudo apt-get install -y cuda-drivers-560"
  echo "  sudo reboot"
  echo
  echo "After reboot, nvidia-smi should show Driver 560+ and CUDA Version 12.6+."
  exit 1
fi

log "Installing build deps"
sudo apt-get update
sudo apt-get install -y build-essential wget

# GCP Debian often mounts /tmp as tmpfs (~8G). The 5G runfile plus extraction
# needs ~12G+; use home dir on the root disk instead.
workdir="${HOME}/cuda-install"
extract_tmp="${workdir}/extract"
mkdir -p "$extract_tmp"
export TMPDIR="$extract_tmp"
trap 'rm -rf "$extract_tmp"' EXIT
cd "$workdir"

avail_kb=$(df -Pk "$workdir" | awk 'NR==2 {print $4}')
need_kb=$((12 * 1024 * 1024)) # ~12 GiB for runfile + extract
if (( avail_kb < need_kb )); then
  echo "Need ~12 GiB free on ${workdir}; only $(( avail_kb / 1024 / 1024 )) GiB available." >&2
  exit 1
fi

if [[ -f "$CUDA_RUN" ]]; then
  log "Reusing existing ${workdir}/${CUDA_RUN}"
else
  log "Downloading ${CUDA_RUN} (~5 GiB)"
  wget -O "$CUDA_RUN" "$CUDA_URL"
fi

log "Installing toolkit only to ${INSTALL_DIR} (no driver; TMPDIR=${TMPDIR})"
sudo TMPDIR="$TMPDIR" sh "$CUDA_RUN" --silent --toolkit --override --installpath="$INSTALL_DIR"

if [[ ! -x "${INSTALL_DIR}/bin/nvcc" ]]; then
  echo "Install failed: ${INSTALL_DIR}/bin/nvcc missing." >&2
  exit 1
fi

if ! grep -q 'cuda-12.8/bin' ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<EOF

# CUDA 12.8 toolkit (autokernel / PyTorch cu128)
export CUDA_HOME=${INSTALL_DIR}
export PATH=\${CUDA_HOME}/bin:\${PATH}
# stubs/ provides libcuda.so for nvcc/Triton link step (driver lib is runtime-only)
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64/stubs:\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}
EOF
fi

export CUDA_HOME="${INSTALL_DIR}"
export PATH="${CUDA_HOME}/bin:${PATH}"

log "nvcc version"
"${CUDA_HOME}/bin/nvcc" --version

echo
echo "Done. Next:"
echo "  export CUDA_HOME=${INSTALL_DIR}"
echo "  export PATH=\${CUDA_HOME}/bin:\$PATH"
echo "  rm -rf ~/.cache/torch_extensions/*/autokernel_matmul"
echo "  cd ~/auto-improving-demo/autokernel && uv run check_cuda.py && uv run bench.py"
