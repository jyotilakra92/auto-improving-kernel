"""
Fixed benchmark harness for autokernel experiments.

Defines the problem, reference implementation, correctness checks, and timing.
The agent must NOT modify this file.

Usage (normally via bench.py):
    python prepare.py   # sanity check reference + baseline kernel import
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

M = 8192          # rows (e.g. batch * seq)
N = 4096          # hidden dimension
EPS = 1e-5
DTYPE = torch.bfloat16

WARMUP_ITERS = 25
BENCH_ITERS = 200
TIME_BUDGET_S = 120  # wall-clock cap for a single bench run (excluding correctness)

# Correctness tolerances for bf16 RMSNorm
ATOL = 5e-2
RTOL = 5e-2

SEED = 42


@dataclass(frozen=True)
class BenchResult:
    median_us: float
    p95_us: float
    gbytes_s: float
    correctness_ok: bool
    max_abs_err: float
    max_rel_err: float
    bench_seconds: float
    total_seconds: float


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required. autokernel runs on a single NVIDIA GPU with Triton."
        )
    return torch.device("cuda")


def make_inputs(device: torch.device | None = None):
    """Return (x, weight, ref_out) tensors on CUDA."""
    device = device or require_cuda()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED)
    x = torch.randn(M, N, generator=gen, dtype=DTYPE)
    weight = torch.randn(N, generator=gen, dtype=DTYPE)
    x = x.to(device)
    weight = weight.to(device)
    ref = reference_rmsnorm(x, weight)
    return x, weight, ref


@torch.no_grad()
def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Ground-truth RMSNorm in PyTorch (not timed)."""
    x_f = x.float()
    w_f = weight.float()
    rms = torch.sqrt((x_f * x_f).mean(dim=-1, keepdim=True) + EPS)
    return ((x_f / rms) * w_f).to(x.dtype)


def check_correctness(out: torch.Tensor, ref: torch.Tensor) -> tuple[bool, float, float]:
    out_f = out.float()
    ref_f = ref.float()
    abs_err = (out_f - ref_f).abs()
    rel_err = abs_err / ref_f.abs().clamp_min(1e-6)
    max_abs = abs_err.max().item()
    max_rel = rel_err.max().item()
    ok = torch.allclose(out_f, ref_f, atol=ATOL, rtol=RTOL)
    return ok, max_abs, max_rel


def _bytes_moved(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> int:
    return x.numel() * x.element_size() + weight.numel() * weight.element_size() + out.numel() * out.element_size()


def benchmark_kernel(fn, x, weight, ref) -> BenchResult:
    """
    Warm up, verify correctness once, then time `fn(x, weight)`.
    Returns median latency in microseconds (lower is better).
    """
    t_total0 = time.perf_counter()

    # Warmup + compile
    for _ in range(WARMUP_ITERS):
        out = fn(x, weight)
    torch.cuda.synchronize()

    out = fn(x, weight)
    torch.cuda.synchronize()
    ok, max_abs, max_rel = check_correctness(out, ref)
    if not ok:
        return BenchResult(
            median_us=float("inf"),
            p95_us=float("inf"),
            gbytes_s=0.0,
            correctness_ok=False,
            max_abs_err=max_abs,
            max_rel_err=max_rel,
            bench_seconds=0.0,
            total_seconds=time.perf_counter() - t_total0,
        )

    bytes_moved = _bytes_moved(x, weight, out)
    samples_us: list[float] = []
    t_bench0 = time.perf_counter()

    for _ in range(BENCH_ITERS):
        if time.perf_counter() - t_bench0 > TIME_BUDGET_S:
            break
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x, weight)
        torch.cuda.synchronize()
        samples_us.append((time.perf_counter() - t0) * 1e6)

    bench_seconds = time.perf_counter() - t_bench0
    median_us = statistics.median(samples_us)
    p95_us = statistics.quantiles(samples_us, n=20)[-1] if len(samples_us) >= 20 else max(samples_us)
    gbytes_s = (bytes_moved / 1e9) / (median_us / 1e6)

    return BenchResult(
        median_us=median_us,
        p95_us=p95_us,
        gbytes_s=gbytes_s,
        correctness_ok=True,
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        bench_seconds=bench_seconds,
        total_seconds=time.perf_counter() - t_total0,
    )


def theoretical_roofline_gbytes_s(device: torch.device | None = None) -> float:
    """Rough HBM roofline using reported device memory bandwidth (GB/s)."""
    device = device or require_cuda()
    props = torch.cuda.get_device_properties(device)
    # PyTorch reports memory bandwidth indirectly; use a conservative estimate from clock * bus.
    # Fallback: 2 TB/s for H100-class, 900 GB/s for A100, 500 GB/s for consumer.
    name = props.name.lower()
    if "h100" in name or "h200" in name:
        return 3350.0
    if "a100" in name:
        return 2039.0
    if "4090" in name or "4080" in name:
        return 1008.0
    return 900.0


def print_summary(result: BenchResult, label: str = "kernel") -> None:
    roof = theoretical_roofline_gbytes_s()
    pct_roof = 100.0 * result.gbytes_s / roof if roof > 0 and math.isfinite(result.gbytes_s) else 0.0
    print("---")
    print(f"label:            {label}")
    print(f"correct:          {result.correctness_ok}")
    print(f"median_us:        {result.median_us:.3f}")
    print(f"p95_us:           {result.p95_us:.3f}")
    print(f"gbytes_s:         {result.gbytes_s:.2f}")
    print(f"pct_roofline:     {pct_roof:.1f}")
    print(f"max_abs_err:      {result.max_abs_err:.6f}")
    print(f"max_rel_err:      {result.max_rel_err:.6f}")
    print(f"bench_seconds:    {result.bench_seconds:.2f}")
    print(f"total_seconds:    {result.total_seconds:.2f}")
    print(f"problem:          RMSNorm [{M}, {N}] {DTYPE}")
    print(f"iters:            {BENCH_ITERS} (cap {TIME_BUDGET_S}s)")


if __name__ == "__main__":
    device = require_cuda()
    x, weight, ref = make_inputs(device)
    out = reference_rmsnorm(x, weight)
    ok, max_abs, max_rel = check_correctness(out, ref)
    print(f"Reference self-check: ok={ok} max_abs={max_abs:.6f} max_rel={max_rel:.6f}")
    print(f"Device: {torch.cuda.get_device_name()}")
