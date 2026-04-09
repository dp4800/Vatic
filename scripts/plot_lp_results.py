#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
plot_lp_results.py — Visualize price-taking LP DC shifting results.

Usage:
    python scripts/plot_lp_results.py \
        --out-dir outputs/RTS_JOINT_TEST/2020-01-06 \
        --iter 1
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

_BLUE   = "#4e79a7"
_RED    = "#e15759"
_GREEN  = "#59a14f"
_ORANGE = "#f28e2b"
_GRAY   = "#aaaaaa"


def load_data(out_dir: Path, iteration: int):
    cas_lp_dir = out_dir / "cas-lp" / f"iter{iteration}"
    ct = pd.read_csv(cas_lp_dir / "comparison_table.csv", index_col=0, parse_dates=True)
    dc_sched = pd.read_csv(cas_lp_dir / "lp_dc_schedule.csv", index_col=0, parse_dates=True)
    dc_sched["total"] = dc_sched.sum(axis=1)

    cas_lp = pd.read_csv(cas_lp_dir / "cas_results.csv")
    cas_gm_path = out_dir / "cas-gm" / "cas_results.csv"
    cas_gm = pd.read_csv(cas_gm_path) if cas_gm_path.exists() else None

    return ct, dc_sched, cas_lp, cas_gm


def make_figure(out_dir: Path, iteration: int, out_path: Path):
    ct, dc_sched, cas_lp, cas_gm = load_data(out_dir, iteration)

    hours = np.arange(len(ct))
    hour_labels = ct.index.strftime("%m-%d\n%H:%M")

    # Pick every 6th label for x-axis readability
    tick_step = 6
    xticks = hours[::tick_step]
    xlabels = hour_labels[::tick_step]

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle("Price-Taking LP: DC Load Shifting Results", fontsize=13, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.32,
                          left=0.08, right=0.97, top=0.93, bottom=0.06)

    # ── Panel 1: DC load timeseries ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(hours, ct["dc_load_baseline"], color=_GRAY, lw=1.5, ls="--", label="Baseline DC load")
    ax1.plot(hours, dc_sched["total"],      color=_BLUE, lw=2,   label="LP-optimised DC load")
    ax1.fill_between(hours, ct["dc_load_baseline"], dc_sched["total"],
                     where=dc_sched["total"] > ct["dc_load_baseline"],
                     alpha=0.15, color=_GREEN, label="Shifted up (cheap hours)")
    ax1.fill_between(hours, ct["dc_load_baseline"], dc_sched["total"],
                     where=dc_sched["total"] < ct["dc_load_baseline"],
                     alpha=0.15, color=_RED, label="Shifted down (expensive hours)")
    ax1.set_xticks(xticks); ax1.set_xticklabels(xlabels, fontsize=8)
    ax1.set_ylabel("DC Load (MW)")
    ax1.set_title("DC Load: Baseline vs LP Schedule (48 hours)")
    ax1.legend(loc="upper right", ncol=2)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax1.set_xlim(0, len(hours) - 1)

    # ── Panel 2: LMP timeseries ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2b = ax2.twinx()
    ax2.bar(hours, ct["lmp_baseline"], color=_ORANGE, alpha=0.6, width=0.8, label="LMP")
    ax2b.plot(hours, dc_sched["total"], color=_BLUE, lw=1.5, label="LP load")
    ax2b.plot(hours, ct["dc_load_baseline"], color=_GRAY, lw=1, ls="--", label="Baseline load")
    ax2.set_xticks(xticks); ax2.set_xticklabels(xlabels, fontsize=8)
    ax2.set_ylabel("LMP ($/MWh)", color=_ORANGE)
    ax2b.set_ylabel("DC Load (MW)", color=_BLUE)
    ax2.set_title("LMP vs DC Load")
    ax2.set_xlim(-0.5, len(hours) - 0.5)
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=8)

    # ── Panel 3: Carbon intensity timeseries ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3b = ax3.twinx()
    ax3.bar(hours, ct["ci_baseline"], color=_RED, alpha=0.55, width=0.8, label="Carbon intensity")
    ax3b.plot(hours, dc_sched["total"], color=_BLUE, lw=1.5, label="LP load")
    ax3b.plot(hours, ct["dc_load_baseline"], color=_GRAY, lw=1, ls="--", label="Baseline load")
    ax3.set_xticks(xticks); ax3.set_xticklabels(xlabels, fontsize=8)
    ax3.set_ylabel("CI (kg CO₂/MWh)", color=_RED)
    ax3b.set_ylabel("DC Load (MW)", color=_BLUE)
    ax3.set_title("Carbon Intensity vs DC Load")
    ax3.set_xlim(-0.5, len(hours) - 0.5)
    lines1, labs1 = ax3.get_legend_handles_labels()
    lines2, labs2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=8)

    # ── Panel 4: Load shift magnitude ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    shift = dc_sched["total"].values - ct["dc_load_baseline"].values
    colors = [_GREEN if s > 0 else _RED for s in shift]
    ax4.bar(hours, shift, color=colors, width=0.8)
    ax4.axhline(0, color="black", lw=0.8)
    ax4.set_xticks(xticks); ax4.set_xticklabels(xlabels, fontsize=8)
    ax4.set_ylabel("Shift (MW)")
    ax4.set_title("DC Load Shift vs Baseline")
    ax4.set_xlim(-0.5, len(hours) - 0.5)
    # annotate net
    ax4.text(0.02, 0.05, f"Net: {shift.sum():.2f} MWh (≈0)",
             transform=ax4.transAxes, fontsize=8, color="gray")

    # ── Panel 5: CO₂ reduction comparison bar chart ──────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    labels_co2 = ["LP\n(price-taking)"]
    values_co2 = [cas_lp["carbon_reduction_pct"].values[0]]
    bar_colors = [_BLUE]

    if cas_gm is not None and "carbon_reduction_pct" in cas_gm.columns:
        best_gm = cas_gm["carbon_reduction_pct"].max()
        labels_co2.append("Grid-mix\n(best α)")
        values_co2.append(best_gm)
        bar_colors.append(_ORANGE)

    bars = ax5.bar(labels_co2, values_co2, color=bar_colors, width=0.4)
    for bar, val in zip(bars, values_co2):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax5.set_ylabel("CO₂ Reduction (%)")
    ax5.set_title("CO₂ Reduction by CAS Mode")
    ax5.set_ylim(0, max(values_co2) * 1.25)

    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot LP DC shifting results")
    p.add_argument("--out-dir", required=True, help="Base output directory (e.g. outputs/RTS_JOINT_TEST/2020-01-06)")
    p.add_argument("--iter", type=int, default=1, help="LP iteration to plot (default: 1)")
    p.add_argument("--out-file", default=None, help="Output PNG path (default: <out-dir>/lp_results.png)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_file = Path(args.out_file) if args.out_file else out_dir / "lp_results.png"

    make_figure(out_dir, args.iter, out_file)


if __name__ == "__main__":
    main()
