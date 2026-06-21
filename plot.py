"""
Plot autokernel experiment progress from results.tsv.

    uv run plot.py              # writes progress.png
    uv run plot.py --show       # also open interactive window

Expects results.tsv with columns:
    commit  median_us  tflops_s  status  description
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = Path(__file__).with_name("results.tsv")
OUTPUT_PATH = Path(__file__).with_name("progress.png")

COLORS = {
    "keep": "#2ecc71",
    "discard": "#cccccc",
    "crash": "#e74c3c",
}


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
    return df


def running_best(series: pd.Series, mask: pd.Series, better: str) -> pd.Series:
    """Cumulative best over kept experiments only."""
    values = series.where(mask)
    if better == "lower":
        return values.cummin()
    return values.cummax()


def plot(df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("autokernel experiment progress", fontsize=14, fontweight="bold")

    x = df["experiment"]
    is_keep = df["status"] == "keep"
    is_crash = df["status"] == "crash"

    # --- median_us (lower is better) ---
    ax = axes[0]
    for status, color in COLORS.items():
        part = df[df["status"] == status]
        if part.empty:
            continue
        ax.scatter(
            part["experiment"],
            part["median_us"],
            c=color,
            s=50 if status == "keep" else 24,
            alpha=0.9 if status == "keep" else 0.6,
            label=status.capitalize(),
            edgecolors="black" if status == "keep" else "none",
            linewidths=0.4,
            zorder=3 if status == "keep" else 2,
        )

    best_us = running_best(df["median_us"], is_keep, "lower")
    ax.step(x, best_us, where="mid", color="#27ae60", linewidth=2, label="Running best", zorder=4)
    ax.set_ylabel("median_us (lower is better)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # --- tflops_s (higher is better) ---
    ax = axes[1]
    valid_tflops = df[~is_crash]
    for status, color in COLORS.items():
        part = valid_tflops[valid_tflops["status"] == status]
        if part.empty:
            continue
        ax.scatter(
            part["experiment"],
            part["tflops_s"],
            c=color,
            s=50 if status == "keep" else 24,
            alpha=0.9 if status == "keep" else 0.6,
            label=status.capitalize(),
            edgecolors="black" if status == "keep" else "none",
            linewidths=0.4,
            zorder=3 if status == "keep" else 2,
        )

    best_tflops = running_best(df["tflops_s"], is_keep, "higher")
    ax.step(x, best_tflops, where="mid", color="#27ae60", linewidth=2, label="Running best", zorder=4)
    ax.set_xlabel("experiment #")
    ax.set_ylabel("tflops_s (higher is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    kept = df[is_keep]
    if not kept.empty:
        summary = (
            f"experiments: {len(df)}  |  kept: {len(kept)}  |  "
            f"best median_us: {kept['median_us'].min():.1f}  |  "
            f"best tflops_s: {kept['tflops_s'].max():.2f}"
        )
        fig.text(0.5, 0.01, summary, ha="center", fontsize=10, color="#444444")

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output, dpi=150)
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
