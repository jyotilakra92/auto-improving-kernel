"""
Plot autokernel experiment progress from results.tsv.

    uv run plot.py              # writes progress.png
    uv run plot.py --show       # also open interactive window

Expects results.tsv with columns:
    commit  median_us  tflops_s  status  description
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

RESULTS_PATH = Path(__file__).with_name("results.tsv")
OUTPUT_PATH = Path(__file__).with_name("progress.png")

SENTINEL_US = 999_999.0
FAIL_TFLOPS = 0.0

COLORS = {
    "keep": "#22c55e",
    "discard": "#cbd5e1",
    "crash": "#ef4444",
}
BEST_LINE = "#15803d"
MILESTONE = "#0f766e"


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No {path.name} found. Run experiments first — the agent appends rows after each bench."
        )
    df = pd.read_csv(path, sep="\t")
    required = {"commit", "median_us", "tflops_s", "status", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    df["median_us"] = pd.to_numeric(df["median_us"], errors="coerce")
    df["tflops_s"] = pd.to_numeric(df["tflops_s"], errors="coerce")
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df["experiment"] = range(1, len(df) + 1)
    df["valid"] = (df["median_us"] < SENTINEL_US) & (df["tflops_s"] > FAIL_TFLOPS)
    return df


def add_running_best(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative best over kept experiments — continuous step series."""
    best_us = math.inf
    best_tflops = 0.0
    running_us: list[float] = []
    running_tflops: list[float] = []

    for _, row in df.iterrows():
        if row["status"] == "keep" and row["valid"]:
            best_us = min(best_us, row["median_us"])
            best_tflops = max(best_tflops, row["tflops_s"])
        running_us.append(best_us if math.isfinite(best_us) else float("nan"))
        running_tflops.append(best_tflops if best_tflops > 0 else float("nan"))

    out = df.copy()
    out["running_best_us"] = running_us
    out["running_best_tflops"] = running_tflops
    return out


def baseline_from_kept(df: pd.DataFrame) -> tuple[float, float]:
    kept = df[(df["status"] == "keep") & df["valid"]]
    if kept.empty:
        raise ValueError("No valid kept experiments — need at least one baseline row.")
    row = kept.iloc[0]
    return float(row["median_us"]), float(row["tflops_s"])


def milestone_indices(df: pd.DataFrame) -> list[int]:
    """Experiment numbers where running best improved."""
    indices: list[int] = []
    prev_us = math.inf
    for _, row in df.iterrows():
        if row["status"] != "keep" or not row["valid"]:
            continue
        if row["median_us"] < prev_us - 1e-6:
            indices.append(int(row["experiment"]))
            prev_us = row["median_us"]
    return indices


def milestone_curve(
    df: pd.DataFrame,
    value_col: str,
    *,
    samples: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth monotonic curve through running-best change points.

    Uses PCHIP so the line passes exactly through real improvements
    and stays monotone (no overshoot between milestones).
    """
    xs: list[float] = []
    ys: list[float] = []
    prev: float | None = None
    for _, row in df.iterrows():
        val = float(row[value_col])
        if prev is None or abs(val - prev) > 1e-9:
            xs.append(float(row["experiment"]))
            ys.append(val)
            prev = val

    last_x = float(df["experiment"].iloc[-1])
    if xs[-1] < last_x:
        xs.append(last_x)
        ys.append(ys[-1])

    x_pts = np.asarray(xs)
    y_pts = np.asarray(ys)
    if len(x_pts) < 2:
        return x_pts, y_pts

    x_dense = np.linspace(x_pts[0], x_pts[-1], samples)
    if len(x_pts) == 2:
        y_dense = np.interp(x_dense, x_pts, y_pts)
    else:
        y_dense = PchipInterpolator(x_pts, y_pts)(x_dense)
    return x_dense, y_dense


def plot(df: pd.DataFrame, output: Path) -> None:
    df = add_running_best(df)
    baseline_us, baseline_tflops = baseline_from_kept(df)

    df["speedup"] = baseline_us / df["running_best_us"]
    df["tflops_gain"] = df["running_best_tflops"] / baseline_tflops

    kept = df["status"] == "keep"
    milestones = milestone_indices(df)
    x_tflops, y_tflops = milestone_curve(df, "running_best_tflops")
    x_us, y_us = milestone_curve(df, "running_best_us")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#fafafa")

    valid = df[df["valid"]]

    kept_df = df[kept & df["valid"]]
    best_us = float(kept_df["median_us"].min())
    best_tflops = float(kept_df["tflops_s"].max())
    us_gain = baseline_us / best_us
    tflops_gain = best_tflops / baseline_tflops

    # --- Panel 1: smooth TFLOPS curve ---
    ax = axes[0]
    ax.set_facecolor("#ffffff")
    ax.fill_between(
        x_tflops,
        baseline_tflops,
        y_tflops,
        alpha=0.18,
        color=BEST_LINE,
        label="Gain vs baseline",
    )
    ax.plot(
        x_tflops,
        y_tflops,
        color=BEST_LINE,
        linewidth=2.8,
        label="Running best tflops_s",
        zorder=4,
    )
    for idx in milestones:
        row = df[df["experiment"] == idx].iloc[0]
        ax.scatter(
            idx,
            row["running_best_tflops"],
            s=90,
            c=COLORS["keep"],
            edgecolors="#14532d",
            linewidths=1.2,
            zorder=5,
        )
        if idx == milestones[0] or idx == milestones[-1] or len(milestones) <= 4:
            ax.annotate(
                f"{row['running_best_tflops']:.2f}",
                (idx, row["running_best_tflops"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color=MILESTONE,
                fontweight="bold",
            )
        elif idx == milestones[1]:
            ax.annotate(
                f"{row['running_best_tflops']:.2f}\n{row['description'][:28]}…",
                (idx, row["running_best_tflops"]),
                textcoords="offset points",
                xytext=(12, 8),
                ha="left",
                fontsize=8,
                color="#475569",
            )

    for status in ("discard", "crash"):
        part = valid[valid["status"] == status]
        if part.empty:
            continue
        ax.scatter(
            part["experiment"],
            part["tflops_s"],
            c=COLORS[status],
            s=28,
            alpha=0.75,
            zorder=2,
        )

    ax.axhline(baseline_tflops, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.8, label="Baseline")
    ax.set_ylabel("tflops_s (higher is better)")
    ax.set_title(
        f"MatMul kernel autotuning — NVIDIA L4, bf16 [1024³]  "
        f"(median_us {baseline_us:.0f} → {best_us:.1f} µs, {us_gain:.1f}× faster; "
        f"tflops_s {baseline_tflops:.2f} → {best_tflops:.2f}, {tflops_gain:.1f}×)",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    ax.set_ylim(bottom=baseline_tflops * 0.85)
    ax.legend(loc="upper left", framealpha=0.95)

    # --- Panel 2: smooth latency curve ---
    ax = axes[1]
    ax.set_facecolor("#ffffff")
    ax.plot(
        x_us,
        y_us,
        color=BEST_LINE,
        linewidth=2.5,
        label="Running best median_us",
        zorder=4,
    )
    for status in ("keep", "discard", "crash"):
        part = df[(df["status"] == status) & df["valid"]]
        if part.empty:
            continue
        ax.scatter(
            part["experiment"],
            part["median_us"],
            c=COLORS[status],
            s=55 if status == "keep" else 26,
            alpha=0.95 if status == "keep" else 0.65,
            edgecolors="#14532d" if status == "keep" else "none",
            linewidths=0.6,
            label=status.capitalize(),
            zorder=3 if status == "keep" else 2,
        )
    ax.axhline(baseline_us, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.8, label="Baseline")
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("median_us (µs, lower is better)")
    ax.legend(loc="upper right", ncol=2, framealpha=0.95)
    ax.set_ylim(0, max(valid["median_us"].max() * 1.15, baseline_us * 1.05))

    summary = (
        f"{len(df)} experiments  ·  {len(kept_df)} kept  ·  "
        f"{int((df['status'] == 'crash').sum())} crashes  ·  "
        f"best {best_us:.1f} µs ({us_gain:.1f}×)  ·  "
        f"best {best_tflops:.2f} TFLOPS ({tflops_gain:.1f}×)"
    )
    fig.text(0.5, 0.012, summary, ha="center", fontsize=10, color="#334155")

    legend_handles = [
        mpatches.Patch(color=COLORS["keep"], label="Kept (improved)"),
        mpatches.Patch(color=COLORS["discard"], label="Discarded"),
        mpatches.Patch(color=COLORS["crash"], label="Crash / invalid"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot autokernel results.tsv")
    parser.add_argument("--input", type=Path, default=RESULTS_PATH, help="Path to results.tsv")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output PNG path")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    df = load_results(args.input)
    if len(df) == 0:
        raise SystemExit(f"{args.input} is empty (header only?). Run at least one experiment first.")

    plot(df, args.output)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
