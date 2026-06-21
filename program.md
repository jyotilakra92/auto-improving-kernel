# autokernel

This is an experiment to have an LLM autonomously optimize a GPU kernel.

Inspired by [@karpathy/autoresearch](https://github.com/karpathy/autoresearch): one editable file, fixed evaluation, keep-or-revert git loop, run overnight.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun20`). The branch `autokernel/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autokernel/<tag>` from current main/master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — fixed problem definition, reference implementation, correctness checks, timing protocol. **Do not modify.**
   - `kernel.cu` — the CUDA C++ MatMul kernel you optimize. **This is the only file you edit.**
   - `kernel.py` — fixed JIT loader (do not modify).
   - `bench.py` — thin runner. **Do not modify.**
4. **Verify GPU + deps**: CUDA must be available. From this directory run `uv sync` then `uv run bench.py`. If it crashes, fix environment issues before starting the loop.
5. **Initialize `results.tsv`**: Create it with just the header row. Baseline is recorded after the first run.
6. **Confirm and go**: Confirm setup looks good, then start experimenting.

## Problem

Optimize **matrix multiply** `C = A @ B` with fixed shapes:

- `A`: `[1024, 1024]` bfloat16
- `B`: `[1024, 1024]` bfloat16
- `C`: `[1024, 1024]` bfloat16

Sized for a single **L4 (~10–24GB)** GPU so the naive baseline finishes in ~1–2 minutes.

The reference is `reference_matmul()` in `prepare.py` (float32 matmul, cast back to bf16). Your kernel must match within fixed tolerances (`ATOL`, `RTOL` in `prepare.py`).

## Experimentation

Each experiment runs on a **single NVIDIA GPU**. Launch with:

```bash
uv run bench.py > run.log 2>&1
```

**What you CAN do:**
- Modify `kernel.cu` only — tile sizes, shared memory, register blocking, vectorized loads, warp-level primitives, etc.

**What you CANNOT do:**
- Modify `prepare.py`, `bench.py`, or `kernel.py`.
- Change problem sizes, dtypes, tolerances, or timing protocol.
- Install new packages or edit `pyproject.toml`.
- Call cuBLAS/cuDNN or PyTorch ops for the timed path (must be your CUDA kernel in `kernel.cu`).

**Goal: minimize `median_us`** (median kernel latency in microseconds). Lower is better. Also watch **`tflops_s`** (higher is better). Correctness is mandatory.

**Simplicity criterion**: All else being equal, simpler is better.

**First run**: Establish baseline with unmodified `kernel.cu`. First launch also JIT-compiles the extension (~10–30s).

## Output format

After a successful run:

```
---
label:            kernel
correct:          True
median_us:        8500.000
p95_us:           8700.000
gbytes_s:         0.45
tflops_s:         0.25
max_abs_err:      0.015625
max_rel_err:      0.012000
bench_seconds:    25.00
total_seconds:    35.00
problem:          MatMul [1024,1024] x [1024,1024] torch.bfloat16
iters:            50 timed (cap 60s)
```

Extract metrics:

```bash
grep "^median_us:\|^correct:\|^tflops_s:" run.log
```

If `correct: False` or grep is empty, the run failed.

## Logging results

Log to `results.tsv` (tab-separated). Do **not** commit this file.

```
commit	median_us	tflops_s	status	description
```

1. git commit hash (short, 7 chars)
2. `median_us` — use `999999.000` for crashes / incorrect
3. `tflops_s` — use `0.00` for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of the experiment

Example:

```
commit	median_us	tflops_s	status	description
a1b2c3d	8500.000	0.25	keep	baseline: 1 block + 1 thread per output
b2c3d4e	8200.000	2.09	keep	shared-memory 32x32 block tile
c3d4e5f	999999.000	0.00	crash	wrong K-loop bounds
```

## Progress chart

After logging results, generate a graph of metrics over time:

```bash
uv run plot.py
```

Writes **`progress.png`** with two panels:
- **median_us** over experiment # (log scale, lower is better) — green = kept, gray = discarded, red = crash
- **tflops_s** over experiment # (higher is better)
- Green step line = running best across kept experiments

Re-run anytime to refresh the chart as experiments accumulate. You can also run `uv run plot.py --show` on a machine with a display.

## The experiment loop

Branch: `autokernel/<tag>` (e.g. `autokernel/jun20`).

LOOP FOREVER:

1. Inspect current git branch/commit.
2. Edit `kernel.cu` with one experimental idea.
3. `git commit`
4. `uv run bench.py > run.log 2>&1`
5. `grep "^median_us:\|^correct:\|^tflops_s:" run.log`
6. On crash or empty grep: `tail -n 50 run.log`, fix trivial bugs or log `crash` and move on.
7. Append row to `results.tsv`.
8. Regenerate the progress chart: `uv run plot.py` (writes `progress.png`).
9. If `correct: True` and `median_us` **improved** (lower): keep commit.
10. Otherwise: `git reset --hard` to pre-experiment commit.

**Timeout**: If a run exceeds 5 minutes wall clock, kill it and treat as failure.

**Crashes**: CUDA compile errors, OOM, or correctness failures → revert unless a trivial fix is obvious.

**NEVER STOP**: Once the loop starts, do not ask the human whether to continue. Run until manually interrupted. If stuck, try: shared-memory tiling, `float4`/`nv_bfloat162` vector loads, register blocking, double-buffering K tiles, warp-level matrix multiply (WMMA), etc.

## Optimization playbook (hints)

- **Baseline**: one block and one thread per output; inner loop over K. Correct but extremely slow.
- **Tiling**: load tiles of A and B into shared memory; reuse data across threads.
- **Coalescing**: consecutive threads should read consecutive addresses.
- **Compute**: MatMul is often compute-bound at sufficient tile size — push `tflops_s` up.
- **Roofline**: compare `tflops_s` against your GPU's peak bf16/tensor TFLOPS.

## Relation to autoresearch

| autoresearch (LLM) | autokernel (this repo) |
|---|---|
| `train.py` editable | `kernel.cu` editable |
| `prepare.py` fixed eval | `prepare.py` fixed eval |
| metric: `val_bpb` ↓ | metric: `median_us` ↓, `tflops_s` ↑ |
| 5 min training budget | ~2 min bench cap per experiment |
| correctness: loss converges | correctness: matches reference tensor |

For full-model kernel autoresearch (profile → extract → optimize many kernels), see [AutoKernel](https://github.com/RightNow-AI/autokernel).
