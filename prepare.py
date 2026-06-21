"""
Fixed benchmark for autokernel MatMul experiments. Do not modify.

What this file does:
  1. Defines the problem size and how long to benchmark.
  2. Builds fixed input matrices A, B and a trusted reference output C.
  3. Times your kernel, checks it matches the reference, prints metrics.

Normal entry point:  uv run bench.py
Sanity check only:   uv run prepare.py
"""

from __future__ import annotations

import time

import torch

# ---------------------------------------------------------------------------
# Problem (fixed) — sized for a single L4 / ~10GB GPU with the naive baseline
# ---------------------------------------------------------------------------

M = 1024
K = 1024
N = 1024
DTYPE = torch.bfloat16
SEED = 42

# How long to benchmark
WARMUP = 5          # untimed runs (also triggers JIT compile on first bench.py run)
TIMED_RUNS = 50     # timed runs (stops early if TIME_LIMIT_S is hit)
TIME_LIMIT_S = 60   # max seconds spent in the timed section

# Pass/fail vs reference (float32 matmul, compared in float32)
ATOL = 0.05
RTOL = 0.05


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (e.g. NVIDIA L4 on Linux).")
    return torch.device("cuda")


def make_inputs(device: torch.device | None = None):
    """Return fixed (A, B, reference_C) on GPU."""
    device = device or require_cuda()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED)
    a = torch.randn(M, K, generator=gen, dtype=DTYPE).to(device)
    b = torch.randn(K, N, generator=gen, dtype=DTYPE).to(device)
    ref = reference_matmul(a, b)
    return a, b, ref


@torch.no_grad()
def reference_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Trusted answer: PyTorch matmul in float32, stored as bfloat16."""
    return torch.matmul(a.float(), b.float()).to(a.dtype)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def check_correctness(got: torch.Tensor, expected: torch.Tensor) -> tuple[bool, float, float]:
    diff = (got.float() - expected.float()).abs()
    rel = diff / expected.float().abs().clamp_min(1e-3)
    max_abs = diff.max().item()
    max_rel = rel.max().item()
    ok = torch.allclose(got.float(), expected.float(), atol=ATOL, rtol=RTOL)
    return ok, max_abs, max_rel


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _time_one_run(fn, a, b) -> float:
    """Run fn(a, b) once; return elapsed microseconds."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(a, b)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6


def benchmark_kernel(fn, a, b, ref):
    """
    Warm up, verify correctness, then time repeated runs.

    fn: callable(a, b) -> C
    Returns a plain dict of metrics (lower median_us is better).
    """
    t_start = time.perf_counter()

    # Warmup (not timed)
    for _ in range(WARMUP):
        fn(a, b)
    torch.cuda.synchronize()

    # Correctness (not timed)
    out = fn(a, b)
    torch.cuda.synchronize()
    correct, max_abs_err, max_rel_err = check_correctness(out, ref)

    if not correct:
        return {
            "correct": False,
            "median_us": float("inf"),
            "p95_us": float("inf"),
            "gbytes_s": 0.0,
            "tflops_s": 0.0,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
            "bench_seconds": 0.0,
            "total_seconds": time.perf_counter() - t_start,
        }

    # Timed runs
    samples_us = []
    t_bench = time.perf_counter()
    for _ in range(TIMED_RUNS):
        if time.perf_counter() - t_bench > TIME_LIMIT_S:
            break
        samples_us.append(_time_one_run(fn, a, b))
    bench_seconds = time.perf_counter() - t_bench

    samples_us.sort()
    median_us = samples_us[len(samples_us) // 2]
    p95_us = samples_us[int(len(samples_us) * 0.95)] if samples_us else float("inf")
    seconds = median_us / 1e6

    # Bytes read/written: A + B + C (bf16)
    bytes_moved = (a.numel() + b.numel() + out.numel()) * out.element_size()
    gbytes_s = (bytes_moved / 1e9) / seconds
    tflops_s = (2.0 * M * K * N / 1e12) / seconds  # 2 FLOPs per output element

    return {
        "correct": True,
        "median_us": median_us,
        "p95_us": p95_us,
        "gbytes_s": gbytes_s,
        "tflops_s": tflops_s,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "bench_seconds": bench_seconds,
        "total_seconds": time.perf_counter() - t_start,
    }


# ---------------------------------------------------------------------------
# Output (grep-friendly lines for the agent loop)
# ---------------------------------------------------------------------------

def print_summary(result: dict, label: str = "kernel") -> None:
    print("---")
    print(f"label:            {label}")
    print(f"correct:          {result['correct']}")
    print(f"median_us:        {result['median_us']:.3f}")
    print(f"p95_us:           {result['p95_us']:.3f}")
    print(f"gbytes_s:         {result['gbytes_s']:.2f}")
    print(f"tflops_s:         {result['tflops_s']:.2f}")
    print(f"max_abs_err:      {result['max_abs_err']:.6f}")
    print(f"max_rel_err:      {result['max_rel_err']:.6f}")
    print(f"bench_seconds:    {result['bench_seconds']:.2f}")
    print(f"total_seconds:    {result['total_seconds']:.2f}")
    print(f"problem:          MatMul [{M},{K}] x [{K},{N}] {DTYPE}")
    print(f"iters:            {TIMED_RUNS} timed (cap {TIME_LIMIT_S}s)")


if __name__ == "__main__":
    dev = require_cuda()
    a, b, ref = make_inputs(dev)
    ok, max_abs, max_rel = check_correctness(reference_matmul(a, b), ref)
    print(f"Reference self-check: ok={ok}  max_abs={max_abs:.6f}  max_rel={max_rel:.6f}")
    print(f"Device: {torch.cuda.get_device_name(dev)}")
