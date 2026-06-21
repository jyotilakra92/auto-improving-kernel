"""
Thin Python wrapper — JIT-compiles kernel.cu on first use.

Do not put kernel logic here; edit kernel.cu instead.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import torch

# Debian 13 apt ships nvcc 13.x; PyTorch wheel is cu128. Allow compile anyway.
import torch.utils.cpp_extension as _cpp_ext

if hasattr(_cpp_ext, "_check_cuda_version"):
    _orig_check_cuda = _cpp_ext._check_cuda_version

    def _relaxed_check_cuda_version(name, version):
        try:
            _orig_check_cuda(name, version)
        except RuntimeError as err:
            if "does not match" not in str(err).lower():
                raise

    _cpp_ext._check_cuda_version = _relaxed_check_cuda_version

from torch.utils.cpp_extension import load

from prepare import K, M, N

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
_CUDA_SRC = os.path.join(_KERNEL_DIR, "kernel.cu")

_mod = None


def _nvcc_version(nvcc: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.check_output([str(nvcc), "--version"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", out)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _nvcc_candidates() -> list[Path]:
    paths: list[Path] = []
    if os.environ.get("CUDA_HOME"):
        paths.append(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc")
    for base in ("/usr/local/cuda", "/usr/lib/nvidia-cuda-toolkit"):
        paths.append(Path(base) / "bin" / "nvcc")
    for pattern in ("/usr/local/cuda-*",):
        paths.extend(Path("/usr/local").glob("cuda-*/bin/nvcc"))
    which = shutil.which("nvcc")
    if which:
        paths.append(Path(which))
    # dedupe, keep order
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _cuda_home_from_nvcc(nvcc: Path) -> Path:
    # .../bin/nvcc -> CUDA root
    return nvcc.parent.parent


def _ensure_cuda_home() -> Path:
    """
    PyTorch cu128 needs CUDA 12.x nvcc at compile AND run time.

    Debian 13 apt only has cuda-nvcc-13-* — that builds against CUDA 13 and fails on
    typical L4 drivers with: "CUDA driver version is insufficient for CUDA runtime version".

    Fix: run ./install_cuda128.sh and set CUDA_HOME=/usr/local/cuda-12.8
    """
    torch_major = int((torch.version.cuda or "0").split(".")[0])
    found: list[tuple[tuple[int, int], Path]] = []

    for nvcc in _nvcc_candidates():
        if not nvcc.is_file():
            continue
        ver = _nvcc_version(nvcc)
        if not ver:
            continue
        found.append((ver, _cuda_home_from_nvcc(nvcc)))

    if not found:
        raise RuntimeError(
            "No system nvcc found. Run: ./install_cuda128.sh\n"
            f"(PyTorch CUDA {torch.version.cuda}; do NOT use apt cuda-nvcc-13-*)"
        )

    # Prefer nvcc major version matching PyTorch (12 for cu128)
    found.sort(key=lambda item: (0 if item[0][0] == torch_major else 1, item[0]))
    ver, root = found[0]

    if ver[0] != torch_major:
        raise RuntimeError(
            f"Found nvcc {ver[0]}.{ver[1]} at {root / 'bin' / 'nvcc'}, "
            f"but PyTorch is CUDA {torch.version.cuda}.\n"
            "apt cuda-nvcc-13-* is incompatible with cu128 on this driver.\n"
            "Run: ./install_cuda128.sh\n"
            "Then: export CUDA_HOME=/usr/local/cuda-12.8\n"
            "      rm -rf ~/.cache/torch_extensions/*/autokernel_matmul"
        )

    os.environ["CUDA_HOME"] = str(root)
    os.environ["PATH"] = str(root / "bin") + os.pathsep + os.environ.get("PATH", "")
    return root


def _cuda_include_flags(cuda_home: Path) -> list[str]:
    flags: list[str] = []
    inc = cuda_home / "include"
    if inc.is_dir():
        flags.append(f"-I{inc}")
    targets = cuda_home / "targets" / "x86_64-linux" / "include"
    if targets.is_dir():
        flags.append(f"-I{targets}")
    torch_include = Path(torch.__file__).resolve().parent / "include"
    if torch_include.is_dir():
        flags.append(f"-I{torch_include}")
    return flags


def cuda_toolchain_info() -> dict:
    root = _ensure_cuda_home()
    nvcc = root / "bin" / "nvcc"
    if not nvcc.is_file():
        nvcc = Path(shutil.which("nvcc") or root / "bin" / "nvcc")
    ver = _nvcc_version(nvcc)
    return {
        "torch_cuda": torch.version.cuda,
        "cuda_home": str(root),
        "nvcc": str(nvcc),
        "nvcc_version": f"{ver[0]}.{ver[1]}" if ver else "unknown",
    }


def _get_module():
    global _mod
    if _mod is None:
        root = _ensure_cuda_home()
        # L4 = Ada sm_89; ensure extension is built for the attached GPU.
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
        _mod = load(
            name="autokernel_matmul",
            sources=[_CUDA_SRC],
            extra_cuda_cflags=["-O3", "--use_fast_math", *_cuda_include_flags(root)],
            verbose=False,
        )
    return _mod


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    m, k = a.shape
    k2, n = b.shape
    assert k == k2 == K and m == M and n == N

    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    _get_module().matmul_forward(a, b, c)
    torch.cuda.synchronize()
    return c
