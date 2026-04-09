#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""3-mode CAS load profile comparison: Grid-Mix vs 24/7 vs LP.

Overlays all three CAS modes on the same axes to show how each reshapes the
average 24-hour demand profile relative to baseline.  Three panels: Annual
average, Winter (Dec–Feb), and Summer (Jun–Aug).

Usage:
    module load anaconda3/2024.10
    python scripts/cas_mode_load_comparison.py [--study-dir PATH]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import CAS_MODES as _CAS_MODES

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CAS_MODES = [{"sim_dir": k, **v} for k, v in _CAS_MODES.items()]

BASELINE_DIR = "baseline"

# Seasonal week groupings (ISO dates of Monday starts)
WINTER_WEEKS = [
    "2018-01-07", "2018-01-21", "2018-02-04", "2018-02-18",
    "2018-12-02", "2018-12-16",
]
SUMMER_WEEKS = [
    "2018-06-03", "2018-06-17", "2018-07-01", "2018-07-15",
    "2018-08-05", "2018-08-19",
]

PANELS = [
    {"title": "Annual Average (24 weeks)",   "weeks": None},      # None = all
    {"title": "Winter (Dec–Feb)",            "weeks": WINTER_WEEKS},
    {"title": "Summer (Jun–Aug)",            "weeks": SUMMER_WEEKS},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_hourly_demand(study: Path, sim_subdir: str,
                       weeks: list[str] | None = None) -> pd.DataFrame:
    """Load hourly_summary.csv Demand from all (or selected) weeks."""
    frames = []
    for week_dir in sorted(study.iterdir()):
        if not week_dir.is_dir():
            continue
        if weeks is not None and week_dir.name not in weeks:
            continue
        csv_path = week_dir / sim_subdir / "hourly_summary.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, usecols=["Date", "Hour", "Demand"])
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No hourly_summary.csv found for {sim_subdir}")

    return pd.concat(frames, ignore_index=True)


def hourly_average(df: pd.DataFrame) -> np.ndarray:
    """Mean demand per hour-of-day (0–23), in GW."""
    return df.groupby("Hour")["Demand"].mean().sort_index().values / 1000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(study: Path, out_path: Path):
    hours = np.arange(24)
    hours_fine = np.linspace(0, 23, 240)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, panel in zip(axes, PANELS):
        wks = panel["weeks"]

        # Baseline
        bl_df = load_hourly_demand(study, BASELINE_DIR, wks)
        bl_avg = hourly_average(bl_df)
        bl_fine = np.interp(hours_fine, hours, bl_avg)

        ax.plot(hours_fine, bl_fine, color="#555555", linewidth=2.2,
                linestyle="--", label="No CAS (baseline)", zorder=4)

        # Overlay all 3 CAS modes
        for mode in CAS_MODES:
            cas_df = load_hourly_demand(study, mode["sim_dir"], wks)
            cas_avg = hourly_average(cas_df)
            cas_fine = np.interp(hours_fine, hours, cas_avg)

            ax.plot(hours_fine, cas_fine, color=mode["color"],
                    linewidth=2, label=mode["label"], zorder=3)

            # Light shading between CAS and baseline
            ax.fill_between(hours_fine, bl_fine, cas_fine,
                            where=cas_fine >= bl_fine, interpolate=True,
                            alpha=0.12, color=mode["color"], zorder=1)
            ax.fill_between(hours_fine, bl_fine, cas_fine,
                            where=cas_fine < bl_fine, interpolate=True,
                            alpha=0.12, color=mode["color"], zorder=1)

        ax.set_title(panel["title"], fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Hour of Day", fontsize=11)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
        ax.set_xticklabels(["12am", "4am", "8am", "12pm", "4pm", "8pm", "11pm"])
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Average System Demand (GW)", fontsize=11)

    fig.suptitle("CAS Mode Comparison: Average 24-Hour Demand Profile",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Texas-7k DC Grid, TX_2018_ANNUAL (24 biweekly VATIC runs)",
             ha="center", fontsize=10, color="#666666")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="3-mode CAS load profile comparison")
    parser.add_argument("--study-dir",
                        default="outputs/TX_2018_ANNUAL",
                        help="Path to TX_2018_ANNUAL output directory")
    parser.add_argument("--output",
                        default="outputs/TX_2018_ANNUAL/cas_mode_load_comparison.png",
                        help="Output image path")
    args = parser.parse_args()

    plot_comparison(Path(args.study_dir), Path(args.output))


if __name__ == "__main__":
    main()
