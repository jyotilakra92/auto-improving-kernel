# autokernel

**Autoresearch for GPU kernels** — give an AI agent a real Triton kernel, a fixed benchmark, and a keep/revert loop. Wake up to faster code.

Directly inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) (autonomous LLM training experiments) and the [AutoKernel](https://github.com/RightNow-AI/autokernel) research line (autonomous kernel optimization on full models).

This repo is a **minimal demo**: one kernel, one metric, one file to edit. It is the kernel equivalent of autoresearch's `train.py` loop.

## How it works

```
prepare.py   — fixed problem, reference, correctness, timing (do not modify)
kernel.py    — Triton RMSNorm implementation (agent modifies this)
bench.py     — runs benchmark, prints grep-friendly metrics (do not modify)
program.md   — agent instructions ("research org code")
```

The agent:

1. Edits `kernel.py`
2. Runs `uv run bench.py`
3. Keeps the commit if `median_us` drops and output is correct
4. Reverts otherwise
5. Logs to `results.tsv` and repeats

## Quick start

**Requirements:** Single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
cd autokernel

# Install dependencies
uv sync

# Baseline benchmark (~30s)
uv run bench.py
```

Expected output includes:

```
median_us:        ...
gbytes_s:         ...
correct:          True
```

## Run the agent

Point Cursor (or another agent) at this repo with permissions to run commands and commit:

```
Read program.md and kick off a new autokernel experiment. Do setup first.
```

The default `program.md` is intentionally minimal — iterate on it like Karpathy suggests for `program.md` in autoresearch.

## Problem: RMSNorm

Shape `[8192, 4096]`, bfloat16 — typical LLM hidden-state normalization. The baseline kernel uses one Triton program per row with a naive reduction. There is plenty of headroom for tiling, vectorization, and occupancy tuning.

## Project layout

```
autokernel/
├── prepare.py    # fixed eval harness
├── kernel.py     # agent-editable Triton kernel
├── bench.py      # benchmark runner
├── program.md    # autonomous agent loop
├── pyproject.toml
└── README.md
```

Sibling directory `../autoresearch/` is the original Karpathy LLM training autoresearch clone.

## Design choices

- **Single editable file** — small diffs, easy review, matches autoresearch philosophy.
- **Correctness-gated** — faster wrong answers are rejected outright.
- **Fixed problem size** — experiments are comparable across commits and machines (same as autoresearch's fixed 5-minute training budget).
- **Triton first** — fast compile/edit cycle for agents; CUDA C++ can be a follow-up fork.

## License

MIT
