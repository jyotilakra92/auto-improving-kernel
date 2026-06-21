# autokernel

**Autoresearch for GPU kernels** — give an AI agent a real CUDA C++ kernel, a fixed benchmark, and a keep/revert loop.

Directly inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## How it works

```
prepare.py   — fixed problem, reference, correctness, timing (do not modify)
kernel.cu    — CUDA C++ MatMul kernel (agent modifies this)
kernel.py    — JIT compile/load wrapper (do not modify)
bench.py     — runs benchmark, prints grep-friendly metrics (do not modify)
plot.py      — reads results.tsv, writes progress.png
program.md   — agent instructions ("research org code")
```

## Problem: MatMul

Fixed shape, bfloat16 (tuned for **single L4 / ~50GB GPU**):

```
C = A @ B
A: [1024, 1024]   B: [1024, 1024]   C: [1024, 1024]
```

Baseline kernel: one CUDA block per output, one thread, naive loop over K.

## Quick start

**Requirements:** NVIDIA GPU + driver (`nvidia-smi` works), Linux, Python 3.10+, [uv](https://docs.astral.sh/uv/).

**Important (Debian 13 + PyTorch cu128):**

1. **Driver 535 is too old** (max CUDA 12.2). Upgrade first:
   ```bash
   sudo apt-get update
   sudo apt-get install -y cuda-drivers-560
   sudo reboot
   nvidia-smi   # expect Driver 560+ and CUDA Version 12.6+
   ```

2. **Do not use** `apt install cuda-nvcc-13-*` (CUDA 13 runtime mismatch).

3. Install **CUDA 12.8 toolkit** (nvcc only):
   ```bash
   cd autokernel
   chmod +x install_cuda128.sh
   ./install_cuda128.sh

   export CUDA_HOME=/usr/local/cuda-12.8
   export PATH=$CUDA_HOME/bin:$PATH
   rm -rf ~/.cache/torch_extensions/*/autokernel_matmul

   uv run check_cuda.py
   uv run bench.py
   ```

Key output lines:

```
median_us:        ...    # lower is better
tflops_s:         ...    # higher is better
correct:          True
```

## Progress chart

After experiments append rows to `results.tsv`:

```bash
uv run plot.py    # → progress.png
```

Shows `median_us` and `tflops_s` over time, with kept/discarded/crashed runs and a running-best line.

## Run the agent

```
Read program.md and kick off a new autokernel experiment. Do setup first.
```

## License

MIT
