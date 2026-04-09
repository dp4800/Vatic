#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""Deferral window sensitivity: load profiles across all 3 CAS modes.

Shows how varying the deferral window W (4 h, 8 h, 18 h, 24 h) reshapes
the average 24-hour demand profile.  W is a datacenter flexibility parameter
that applies to all CAS modes; sweep data currently exists for LP CAS, while
Grid-Mix and 24/7 are shown at their default W (12 h) as reference.
Three panels: Average (all sweep weeks), Winter (Jan), and Summer (Jul–Aug).

Usage:
    module load anaconda3/2024.10
    python scripts/cas_deferral_load_comparison.py [--study-dir PATH]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import SWEEP_WEEKS

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

WINTER_SWEEP = ["2018-01-07", "2018-01-21"]
SUMMER_SWEEP = ["2018-07-01", "2018-07-15"]

# Grid-Mix and 24/7 at default W=12 h (no sweep data for these modes)
FIXED_MODES = [
    {"sim_dir": "sim-gm",  "label": "Grid-Mix CAS ($W$=12 h)",  "color": "#FF8C00",
     "linewidth": 1.8, "linestyle": "-"},
    {"sim_dir": "sim-247", "label": "24/7 CAS ($W$=12 h)",       "color": "#8B5CF6",
     "linewidth": 1.8, "linestyle": "-"},
]

# LP CAS deferral sweep — shades of blue with distinct styles
LP_DEFERRAL = [
    {"dir": "sim-lp-deferral-deferral_4h",  "label": "LP CAS $W$=4 h",
     "color": "#93c5fd", "linewidth": 1.6, "linestyle": "-"},
    {"dir": "sim-lp-deferral-deferral_8h",  "label": "LP CAS $W$=8 h",
     "color": "#60a5fa", "linewidth": 1.6, "linestyle": "-"},
    {"dir": "sim-lp-deferral-deferral_18h", "label": "LP CAS $W$=18 h",
     "color": "#3b82f6", "linewidth": 1.6, "linestyle": "-"},
    {"dir": "sim-lp-deferral-deferral_24h", "label": "LP CAS $W$=24 h",
     "color": "#1d4ed8", "linewidth": 1.6, "linestyle": "-"},
]

PANELS = [
    {"title": "Average (8 sweep weeks)", "weeks": SWEEP_WEEKS},
    {"title": "Winter (Jan)",            "weeks": WINTER_SWEEP},
    {"title": "Summer (Jul)",            "weeks": SUMMER_SWEEP},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_hourly_demand(study: Path, sim_subdir: str,
                       weeks: list[str]) -> pd.DataFrame:
    """Load hourly_summary.csv Demand from specified weeks."""
    frames = []
    for week in weeks:
        csv_path = study / week / sim_subdir / "hourly_summary.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, usecols=["Date", "Hour", "Demand"])
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No hourly_summary.csv for {sim_subdir}")
    return pd.concat(frames, ignore_index=True)


def hourly_average(df: pd.DataFrame) -> np.ndarray:
    """Mean demand per hour-of-day (0–23), in GW."""
    return df.groupby("Hour")["Demand"].mean().sort_index().values / 1000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_deferral(study: Path, out_path: Path):
    hours = np.arange(24)
    hours_fine = np.linspace(0, 23, 240)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, panel in zip(axes, PANELS):
        wks = panel["weeks"]

        # Baseline
        bl_df = load_hourly_demand(study, "baseline", wks)
        bl_avg = hourly_average(bl_df)
        bl_fine = np.interp(hours_fine, hours, bl_avg)

        ax.plot(hours_fine, bl_fine, color="#555555", linewidth=2.2,
                linestyle="--", label="No CAS (baseline)", zorder=5)

        # Fixed modes: Grid-Mix and 24/7
        for mode in FIXED_MODES:
            try:
                m_df = load_hourly_demand(study, mode["sim_dir"], wks)
            except FileNotFoundError:
                continue
            m_avg = hourly_average(m_df)
            m_fine = np.interp(hours_fine, hours, m_avg)

            ax.plot(hours_fine, m_fine, color=mode["color"],
                    linewidth=mode["linewidth"], linestyle=mode["linestyle"],
                    label=mode["label"], zorder=4)

        # LP deferral sweep
        for lp in LP_DEFERRAL:
            try:
                lp_df = load_hourly_demand(study, lp["dir"], wks)
            except FileNotFoundError:
                continue
            lp_avg = hourly_average(lp_df)
            lp_fine = np.interp(hours_fine, hours, lp_avg)

            ax.plot(hours_fine, lp_fine, color=lp["color"],
                    linewidth=lp["linewidth"], linestyle=lp["linestyle"],
                    label=lp["label"], zorder=3)

            # Light shading between LP curve and baseline
            ax.fill_between(hours_fine, bl_fine, lp_fine,
                            where=lp_fine >= bl_fine, interpolate=True,
                            alpha=0.06, color=lp["color"], zorder=1)
            ax.fill_between(hours_fine, bl_fine, lp_fine,
                            where=lp_fine < bl_fine, interpolate=True,
                            alpha=0.06, color=lp["color"], zorder=1)

        ax.set_title(panel["title"], fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Hour of Day", fontsize=11)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
        ax.set_xticklabels(["12am", "4am", "8am", "12pm",
                             "4pm", "8pm", "11pm"])
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Average System Demand (GW)", fontsize=11)

    fig.suptitle("Deferral Window $W$ Sensitivity: Average 24-Hour Demand Profile\n"
                 "LP CAS sweep ($W$=4–24 h); Grid-Mix and 24/7 at default $W$=12 h",
                 fontsize=14, fontweight="bold", y=1.04)
    fig.text(0.5, -0.02,
             "Texas-7k DC Grid, TX_2018_ANNUAL (8 biweekly sweep weeks, "
             "2 per season)",
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
        description="Deferral window sensitivity: load profiles, 3 CAS modes")
    parser.add_argument("--study-dir",
                        default="outputs/TX_2018_ANNUAL",
                        help="Path to TX_2018_ANNUAL output directory")
    parser.add_argument("--output",
                        default="outputs/TX_2018_ANNUAL/"
                                "cas_deferral_load_comparison.png",
                        help="Output image path")
    args = parser.parse_args()

    plot_deferral(Path(args.study_dir), Path(args.output))


if __name__ == "__main__":
    main()
