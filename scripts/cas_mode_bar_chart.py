#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""Bar chart: relative CAS mode differences vs. baseline.

Thesis: Figure 12 — "Changes, relative to the baseline, of grid performance
    for the three CAS modes."

Aggregates weekly_summary CSVs across all 24 weeks of TX_2018_ANNUAL and
plots % change vs. baseline for key metrics, grouped by CAS mode.

Usage:
    module load anaconda3/2024.10
    python scripts/cas_mode_bar_chart.py [--study-dir PATH]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import CAS_MODES

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Metrics to plot (must match weekly_summary 'metric' column)
# sign: +1 means "lower is better" (negative % = improvement), shown as-is
METRICS = [
    {"name": "Total CO2",            "short": r"CO$_2$",              "unit": "kt"},
    {"name": "Operational cost",     "short": "Op. Cost",             "unit": "$M"},
    {"name": "Renewables used",      "short": "Renew.\nCoverage",     "unit": "GWh"},
    {"name": "Renewables curtailed", "short": "Renew.\nCurtailed",    "unit": "GWh"},
    {"name": "Load shedding",        "short": "Load\nShedding",       "unit": "MWh"},
    {"name": "Water consumption",     "short": "Water\nConsumption",    "unit": "Bgal"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_weekly(study: Path) -> pd.DataFrame:
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


def compute_pct_change(ws: pd.DataFrame) -> pd.DataFrame:
    """Compute % change vs. baseline for each metric × mode.

    For each metric, restricts to weeks where ALL modes have data, then
    sums and computes (mode - bl) / bl * 100.
    """
    mode_cols = [k for k in CAS_MODES if k in ws.columns]
    rows = []
    for metric_cfg in METRICS:
        mname = metric_cfg["name"]
        mrows = ws[ws["metric"] == mname].copy()
        if mrows.empty:
            continue

        # Convert to numeric
        for col in ["baseline"] + mode_cols:
            mrows[col] = pd.to_numeric(mrows[col], errors="coerce")

        # Keep only weeks where baseline AND all modes have data
        mask = mrows["baseline"].notna()
        for col in mode_cols:
            mask &= mrows[col].notna()
        mrows = mrows[mask]

        if mrows.empty:
            continue

        bl_total = mrows["baseline"].sum()
        if bl_total == 0:
            continue

        for mode_key in mode_cols:
            mode_total = mrows[mode_key].sum()
            pct = (mode_total - bl_total) / bl_total * 100.0
            rows.append({
                "metric": mname,
                "short": metric_cfg["short"],
                "mode": mode_key,
                "baseline_total": bl_total,
                "mode_total": mode_total,
                "pct_change": pct,
                "n_weeks": len(mrows),
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bars(study: Path, out_path: Path):
    ws = load_all_weekly(study)
    df = compute_pct_change(ws)

    n_weeks = ws["week_date"].nunique()
    metric_names = [m["short"] for m in METRICS]
    mode_keys = list(CAS_MODES.keys())
    n_metrics = len(metric_names)
    n_modes = len(mode_keys)

    x = np.arange(n_metrics)
    width = 0.22
    offsets = np.array([-1, 0, 1]) * width

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, mode_key in enumerate(mode_keys):
        mode_df = df[df["mode"] == mode_key]
        vals = []
        for metric_cfg in METRICS:
            row = mode_df[mode_df["metric"] == metric_cfg["name"]]
            vals.append(row["pct_change"].values[0] if len(row) else 0.0)

        style = CAS_MODES[mode_key]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=style["label"], color=style["color"],
                      edgecolor="white", linewidth=0.5, zorder=3)

        # Value labels on bars
        for bar, v in zip(bars, vals):
            y_pos = bar.get_height()
            va = "bottom" if y_pos >= 0 else "top"
            offset = 0.3 if y_pos >= 0 else -0.3
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos + offset,
                    f"{v:+.1f}%", ha="center", va=va, fontsize=7.5,
                    fontweight="bold", color=style["color"])

    ax.axhline(0, color="#333333", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel("Change vs. Baseline (%)", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9, loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    n_matched = int(df["n_weeks"].iloc[0]) if len(df) else 0
    ax.set_title(
        "CAS Mode Impact: Relative Change vs. No-CAS Baseline\n"
        f"Texas-7k DC Grid  |  {n_matched} matched weeks where all modes "
        "have data (TX_2018_ANNUAL)",
        fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Bar chart: CAS mode relative differences vs. baseline")
    parser.add_argument("--study-dir",
                        default="outputs/TX_2018_ANNUAL",
                        help="Path to TX_2018_ANNUAL output directory")
    parser.add_argument("--output",
                        default="outputs/TX_2018_ANNUAL/"
                                "cas_mode_bar_chart.png",
                        help="Output image path")
    args = parser.parse_args()

    plot_bars(Path(args.study_dir), Path(args.output))


if __name__ == "__main__":
    main()
