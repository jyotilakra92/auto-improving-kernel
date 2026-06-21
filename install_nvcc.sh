#!/bin/bash
# Wrapper — use CUDA 12.8 toolkit (matches PyTorch cu128).
# apt cuda-nvcc-13-* causes: "CUDA driver version is insufficient for CUDA runtime version"
exec "$(dirname "$0")/install_cuda128.sh" "$@"
