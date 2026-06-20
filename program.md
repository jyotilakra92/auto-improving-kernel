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
   - `kernel.py` — the Triton RMSNorm kernel you optimize. **This is the only file you edit.**
   - `bench.py` — thin runner. **Do not modify.**
4. **Verify GPU + deps**: CUDA must be available. From this directory run `uv sync` then `uv run bench.py`. If it crashes, fix environment issues before starting the loop.
5. **Initialize `results.tsv`**: Create it with just the header row. Baseline is recorded after the first run.
6. **Confirm and go**: Confirm setup looks good, then start experimenting.

## Problem

Optimize **RMSNorm** on shape `[8192, 4096]` in **bfloat16**:

```
output[i, :] = x[i, :] / sqrt(mean(x[i,:]^2) + eps) * weight[:]
```

The reference is `reference_rmsnorm()` in `prepare.py`. Your kernel must match within fixed tolerances (`ATOL`, `RTOL` in `prepare.py`).

## Experimentation

Each experiment runs on a **single NVIDIA GPU**. Launch with:

```bash
uv run bench.py > run.log 2>&1
```

**What you CAN do:**
- Modify `kernel.py` only — Triton tile sizes, fusion, parallelization, memory coalescing, multi-row programs, etc.
- Add Triton `@triton.jit` helpers in the same file.

**What you CANNOT do:**
- Modify `prepare.py` or `bench.py`.
- Change problem sizes, dtypes, tolerances, or timing protocol.
- Install new packages or edit `pyproject.toml`.
- Call into `torch.compile`, cuBLAS, or PyTorch ops for the timed path (must be your Triton kernel).

**Goal: minimize `median_us`** (median kernel latency in microseconds). Lower is better. Correctness is mandatory — wrong kernels are crashes.

**Simplicity criterion**: All else equal, simpler is better. A 5% speedup that adds 80 lines of fragile code is probably not worth it. Deleting code and matching speed is a win.

**First run**: Establish baseline with unmodified `kernel.py`.

## Output format

After a successful run:

```
---
label:            kernel
correct:          True
median_us:        142.350
p95_us:           148.120
gbytes_s:         890.42
pct_roofline:     26.6
max_abs_err:      0.001953
max_rel_err:      0.004882
bench_seconds:    28.50
total_seconds:    29.10
problem:          RMSNorm [8192, 4096] torch.bfloat16
iters:            200 (cap 120s)
```

Extract metrics:

```bash
grep "^median_us:\|^correct:\|^gbytes_s:" run.log
```

If `correct: False` or grep is empty, the run failed.

## Logging results

Log to `results.tsv` (tab-separated). Do **not** commit this file.

```
commit	median_us	gbytes_s	status	description
```

1. git commit hash (short, 7 chars)
2. `median_us` — use `999999.000` for crashes / incorrect
3. `gbytes_s` — use `0.00` for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of the experiment

Example:

```
commit	median_us	gbytes_s	status	description
a1b2c3d	142.350	890.42	keep	baseline naive row kernel
b2c3d4e	118.200	1071.50	keep	2-row tile, BLOCK=512
c3d4e5f	999999.000	0.00	crash	wrong mask at tail
```

## The experiment loop

Branch: `autokernel/<tag>` (e.g. `autokernel/jun20`).

LOOP FOREVER:

1. Inspect current git branch/commit.
2. Edit `kernel.py` with one experimental idea.
3. `git commit`
4. `uv run bench.py > run.log 2>&1`
5. `grep "^median_us:\|^correct:\|^gbytes_s:" run.log`
6. On crash or empty grep: `tail -n 50 run.log`, fix trivial bugs or log `crash` and move on.
7. Append row to `results.tsv`.
8. If `correct: True` and `median_us` **improved** (lower): keep commit.
9. Otherwise: `git reset --hard` to pre-experiment commit.

**Timeout**: If a run exceeds 5 minutes wall clock, kill it and treat as failure.

**Crashes**: Triton compile errors, OOM, or correctness failures → revert unless a trivial fix is obvious.

**NEVER STOP**: Once the loop starts, do not ask the human whether to continue. Run until manually interrupted. If stuck, try: larger/smaller `BLOCK`, vectorized loads, multiple rows per program, shared memory partial reductions, persistent kernels, reordering rsqrt vs weight multiply, etc.

## Optimization playbook (hints)

- **Memory**: RMSNorm is bandwidth-bound. Maximize coalesced 128-bit loads/stores along the hidden dim.
- **Occupancy**: One program per row launches 8192 blocks — often fine, but row work may be too fat; try multiple rows per CTA.
- **Reduction**: Partial sums via `tl.sum` over tiles; avoid redundant passes over `x`.
- **Constants**: `eps` and `n_cols` are compile-time friendly in Triton.
- **Roofline**: `pct_roofline` in output — aim to push this up without breaking correctness.

## Relation to autoresearch

| autoresearch (LLM) | autokernel (this repo) |
|---|---|
| `train.py` editable | `kernel.py` editable |
| `prepare.py` fixed eval | `prepare.py` fixed eval |
| metric: `val_bpb` ↓ | metric: `median_us` ↓ |
| 5 min training budget | ~2 min bench cap per experiment |
| correctness: loss converges | correctness: matches reference tensor |

For full-model kernel autoresearch (profile → extract → optimize many kernels), see [AutoKernel](https://github.com/RightNow-AI/autokernel).
