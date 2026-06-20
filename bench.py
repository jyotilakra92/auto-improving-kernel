"""
Run the fixed autokernel benchmark against kernel.py.

    uv run bench.py

Prints grep-friendly summary lines for the agent loop.
"""

from __future__ import annotations

import torch

from kernel import rmsnorm
from prepare import benchmark_kernel, make_inputs, print_summary, require_cuda


def main() -> None:
    device = require_cuda()
    torch.cuda.manual_seed(42)
    x, weight, ref = make_inputs(device)

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Capability: {torch.cuda.get_device_capability()}")

    result = benchmark_kernel(rmsnorm, x, weight, ref)
    print_summary(result, label="kernel")


if __name__ == "__main__":
    main()
