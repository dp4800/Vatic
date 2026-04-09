#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""Average 24-hour demand load profile by CAS mode.

Thesis: Figure 8 — "Representative weekly time series for system-wide grid demand."

Reads hourly_summary.csv from baseline and CAS simulation outputs across
all 24 weekly VATIC runs (TX_2018_ANNUAL), computes hour-of-day averages,
and plots 3-panel comparison (Grid-Mix, 24/7, LP CAS vs. baseline).

Usage:
    module load anaconda3/2024.10
    python scripts/cas_load_profiles.py [--study-dir PATH]
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


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_hourly_demand(study: Path, sim_subdir: str) -> pd.DataFrame:
    """Load and concatenate hourly_summary.csv from all weeks for a sim mode.

    Returns DataFrame with columns: Date, Hour, Demand (MW).
    """
    frames = []
    for week_dir in sorted(study.iterdir()):
        csv_path = week_dir / sim_subdir / "hourly_summary.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, usecols=["Date", "Hour", "Demand"])
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No hourly_summary.csv found for {sim_subdir} in {study}")

    return pd.concat(frames, ignore_index=True)


def hourly_average(df: pd.DataFrame) -> np.ndarray:
    """Compute mean demand per hour-of-day (0–23), in GW."""
    return df.groupby("Hour")["Demand"].mean().sort_index().values / 1000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_profiles(study: Path, out_path: Path):
    # Load baseline
    bl_df = load_hourly_demand(study, BASELINE_DIR)
    bl_avg = hourly_average(bl_df)
    hours = np.arange(25)  # 0–24; hour 24 == hour 0 to close the loop

    # Interpolate for smooth fill_between at crossovers
    hours_fine = np.linspace(0, 24, 240)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Wrap baseline: append hour-0 value at position 24 to close the loop
    bl_avg_wrap = np.append(bl_avg, bl_avg[0])

    for ax, mode in zip(axes, CAS_MODES):
        cas_df = load_hourly_demand(study, mode["sim_dir"])
        cas_avg = hourly_average(cas_df)
        cas_avg_wrap = np.append(cas_avg, cas_avg[0])
        color = mode["color"]

        # Interpolate to fine grid for clean crossover shading
        bl_fine = np.interp(hours_fine, hours, bl_avg_wrap)
        cas_fine = np.interp(hours_fine, hours, cas_avg_wrap)

        # Baseline line
        ax.plot(hours_fine, bl_fine, color="#555555", linewidth=2,
                linestyle="--", label="No CAS (baseline)", zorder=3)

        # CAS line
        ax.plot(hours_fine, cas_fine, color=color, linewidth=2,
                label=mode["label"], zorder=3)

        # Shaded area — single fill with interpolate=True
        ax.fill_between(hours_fine, bl_fine, cas_fine,
                         where=cas_fine >= bl_fine, interpolate=True,
                         alpha=0.25, color=color, zorder=2)
        ax.fill_between(hours_fine, bl_fine, cas_fine,
                         where=cas_fine < bl_fine, interpolate=True,
                         alpha=0.25, color=color, zorder=2)

        ax.set_title(mode["label"], fontsize=14, fontweight="bold",
                     color=color, pad=10)
        ax.set_xlabel("Hour of Day", fontsize=11)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
        ax.set_xticklabels(["12am", "4am", "8am", "12pm", "4pm", "8pm", "12am"])
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Average System Demand (GW)", fontsize=11)

    n_weeks = len([d for d in study.iterdir() if d.is_dir()
                   and (d / BASELINE_DIR).exists()])

    fig.suptitle("Average 24-Hour Demand Load Profile by CAS Mode",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             f"Averaged over {n_weeks} weekly VATIC runs (TX_2018_ANNUAL)",
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
        description="Average 24-hour demand load profile by CAS mode")
    parser.add_argument("--study-dir",
                        default="outputs/TX_2018_ANNUAL",
                        help="Path to TX_2018_ANNUAL output directory")
    parser.add_argument("--output",
                        default="outputs/TX_2018_ANNUAL/cas_load_profiles.png",
                        help="Output image path")
    args = parser.parse_args()

    study = Path(args.study_dir)
    out_path = Path(args.output)
    plot_profiles(study, out_path)


if __name__ == "__main__":
    main()
