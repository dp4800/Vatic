#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""CAS outcome sensitivity: CO₂, cost, water across all 3 modes and 24 weeks.

Thesis: Figure 13 — "Performance of the three CAS modes across all months."

Reads weekly_summary CSVs from TX_2018_ANNUAL and plots % change vs baseline
for each CAS mode (Grid-Mix, 24/7, LP) across all 24 weeks, grouped by metric.

Usage:
    module load anaconda3/2024.10
    python scripts/cas_outcome_sensitivity.py [--study-dir PATH]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from constants import CAS_MODES

# Outlier weeks to exclude per (mode, metric).  value → NaN, gap bridged
# with a dashed line.
# Outlier weeks to exclude per mode (applies to all metrics).
# 2018-07-01 Grid-Mix had anomalous dispatch convergence.
OUTLIER_WEEKS: dict[str, list[str]] = {
    "sim-gm":  ["2018-07-01"],
    "sim-247": ["2018-07-01"],
    "sim-lp":  ["2018-07-01"],
}

# Metrics to plot (must match weekly_summary 'metric' column exactly)
METRICS = [
    {"name": "Total CO2",            "unit": "kt",    "ylabel": "CO₂ Change vs. Baseline (%)",
     "title": "Carbon emissions",   "invert": True},
    {"name": "Operational cost",     "unit": "$M",    "ylabel": "Cost Change vs. Baseline (%)",
     "invert": True},
    {"name": "Water consumption",     "unit": "Bgal",  "ylabel": "Water Consumption Change (%)",
     "invert": True},
    {"name": "Renewables curtailed", "unit": "GWh",   "ylabel": "Renewables Curtailed Change (%)",
     "invert": True},
]

SEASON_COLORS = {
    "Winter": "#3B82F6",
    "Spring": "#22C55E",
    "Summer": "#EF4444",
    "Fall":   "#F59E0B",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_weekly_summaries(study: Path) -> pd.DataFrame:
    """Load and concatenate all weekly_summary CSVs."""
    frames = []
    for week_dir in sorted(study.iterdir()):
        if not week_dir.is_dir():
            continue
        for csv in week_dir.glob("weekly_summary_*.csv"):
            df = pd.read_csv(csv)
            frames.append(df)

    if not frames:
        raise FileNotFoundError("No weekly_summary CSVs found")

    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_outcomes(study: Path, out_path: Path):
    ws = load_weekly_summaries(study)
    ws["week_date"] = pd.to_datetime(ws["week_date"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    for ax, metric_cfg in zip(axes, METRICS):
        mname = metric_cfg["name"]
        rows = ws[ws["metric"] == mname].copy()
        rows = rows.sort_values("week_date")

        for mode_key, mode_cfg in CAS_MODES.items():
            if mode_key not in rows.columns:
                continue

            bl = rows["baseline"].astype(float)
            val = rows[mode_key].astype(float)
            pct_change = (val - bl) / bl * 100.0

            # Mask outlier weeks
            outlier_dates = OUTLIER_WEEKS.get(mode_key, [])
            if outlier_dates:
                mask = rows["week_date"].dt.strftime("%Y-%m-%d").isin(outlier_dates)
                pct_change = pct_change.copy()
                pct_change[mask] = np.nan

            dates = rows["week_date"].values
            vals = pct_change.values

            # Solid line + markers for valid points
            ax.plot(dates, vals, color=mode_cfg["color"],
                    marker=mode_cfg["marker"], markersize=5, linewidth=1.5,
                    label=mode_cfg["label"], zorder=3)

            # Dashed bridge across each NaN gap
            valid = ~np.isnan(vals)
            for i in range(len(vals)):
                if np.isnan(vals[i]):
                    # find nearest valid points before and after
                    before = np.where(valid[:i])[0]
                    after = np.where(valid[i+1:])[0]
                    if len(before) and len(after):
                        j, k = before[-1], after[0] + i + 1
                        ax.plot([dates[j], dates[k]], [vals[j], vals[k]],
                                color=mode_cfg["color"], linewidth=1.0,
                                linestyle="--", alpha=0.5, zorder=2)

        # Zero line
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="-", zorder=1)

        # Season background bands
        for _, row in rows.iterrows():
            season = row.get("season", "")
            if season in SEASON_COLORS:
                ax.axvspan(row["week_date"] - pd.Timedelta(days=3),
                           row["week_date"] + pd.Timedelta(days=3),
                           alpha=0.06, color=SEASON_COLORS[season], zorder=0)

        ax.set_ylabel(metric_cfg["ylabel"], fontsize=10)
        ax.set_title(metric_cfg.get("title", mname), fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    fig.suptitle("CAS Mode Sensitivity: Weekly Outcome Metrics vs. Baseline\n"
                 "(Texas-7k DC Grid, 23 of 24 Biweekly Windows, 2018)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CAS outcome sensitivity across 3 modes and 24 weeks")
    parser.add_argument("--study-dir",
                        default="outputs/TX_2018_ANNUAL",
                        help="Path to TX_2018_ANNUAL output directory")
    parser.add_argument("--output",
                        default="outputs/TX_2018_ANNUAL/cas_outcome_sensitivity.png",
                        help="Output image path")
    args = parser.parse_args()

    plot_outcomes(Path(args.study_dir), Path(args.output))


if __name__ == "__main__":
    main()
