"""
RMSNorm GPU kernel — the ONLY file the autoresearch agent should edit.

    output[i, :] = x[i, :] / rms(x[i, :]) * weight

Baseline: naive row-wise Triton kernel (one program per row).
The agent's job is to reduce median_us while keeping correctness.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from prepare import EPS, N


@triton.jit
def _rmsnorm_row_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
    sq = x * x
    mean_sq = tl.sum(sq, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_sq + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * inv_rms * w
    tl.store(y_ptr + row * stride_ym + cols * stride_yn, y.to(tl.bfloat16), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    m, n = x.shape
    assert n == weight.numel() == N

    y = torch.empty_like(x)
    block = triton.next_power_of_2(n)
    grid = (m,)
    _rmsnorm_row_kernel[grid](
        x,
        weight,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        n,
        EPS,
        BLOCK=block,
    )
    return y
