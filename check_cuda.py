"""
Verify GPU + nvcc before running benchmarks.

    uv run check_cuda.py
"""

from __future__ import annotations

import subprocess
import sys

import torch

from kernel import cuda_toolchain_info


def main() -> None:
    print("=== autokernel CUDA check ===\n")

    if not torch.cuda.is_available():
        print("FAIL: PyTorch sees no GPU. Fix nvidia-smi / driver first.")
        sys.exit(1)

    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA: {torch.version.cuda}")

    try:
        info = cuda_toolchain_info()
    except RuntimeError as err:
        print(f"\nFAIL: {err}")
        sys.exit(1)

    print(f"CUDA_HOME:    {info['cuda_home']}")
    print(f"nvcc:         {info['nvcc']}")
    print(f"nvcc version: {info['nvcc_version']}")

    torch_major = int(torch.version.cuda.split(".")[0])
    nvcc_major = int(info["nvcc_version"].split(".")[0])
    if torch_major != nvcc_major:
        print(
            f"\nNote: nvcc {info['nvcc_version']} vs PyTorch CUDA {torch.version.cuda} "
            "(OK on Debian 13 — we relax PyTorch's version check for JIT compile)."
        )

    subprocess.run([info["nvcc"], "--version"], check=False)
    print("\nOK — run: uv run bench.py")


if __name__ == "__main__":
    main()
