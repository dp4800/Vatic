#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""Water use analysis: seasonal profiles + fuel decomposition.

Figure 1 (3×4): Hourly water withdrawal, consumption, and withdrawal by
                fuel type — by season, all 3 CAS modes vs. baseline.
Figure 2 (1×1): Annual fuel-type bar chart showing which fuels drive water
                changes for each CAS mode.

Usage:
    module load anaconda3/2024.10
    python scripts/water_analysis.py [--study-dir PATH]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import CAS_MODES, BASELINE_STYLE, SEASONS

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODES = {"baseline": BASELINE_STYLE, **CAS_MODES}

# Short fuel labels for plots
FUEL_SHORT = {
    "NUC (Nuclear)":                   "Nuclear",
    "NG (Natural Gas)":                "Nat. Gas",
    "SUB (Subbituminous Coal)":        "Sub. Coal",
    "LIG (Lignite Coal)":             "Lignite",
    "PUR (Purchased Steam)":          "Purch. Steam",
    "WDS (Wood/Wood Waste Solids)":   "Wood",
}

FUEL_COLORS = {
    "Nuclear":      "#7c3aed",
    "Nat. Gas":     "#f97316",
    "Sub. Coal":    "#374151",
    "Lignite":      "#6b7280",
    "Purch. Steam": "#06b6d4",
    "Wood":         "#22c55e",
}

GAL_TO_BGAL = 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_system_water(study: Path, mode: str,
                      weeks: list[str]) -> pd.DataFrame:
    """Load system_water_hourly.csv for specified weeks."""
    frames = []
    for wk in weeks:
        csv = study / wk / "water" / mode / "system_water_hourly.csv"
        if csv.exists():
            frames.append(pd.read_csv(csv))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_fuel_water(study: Path, mode: str,
                    weeks: list[str]) -> pd.DataFrame:
    """Load fuel_water_hourly.csv for specified weeks."""
    frames = []
    for wk in weeks:
        csv = study / wk / "water" / mode / "fuel_water_hourly.csv"
        if csv.exists():
            frames.append(pd.read_csv(csv))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def hourly_avg_water(df: pd.DataFrame, col: str) -> np.ndarray:
    """Mean of `col` per hour-of-day (0–23), in Bgal."""
    if df.empty:
        return np.full(24, np.nan)
    return (df.groupby("Hour")[col].mean().sort_index().values * GAL_TO_BGAL)


def total_water_by_fuel(df: pd.DataFrame, col: str) -> pd.Series:
    """Total `col` (gal) per fuel, returned with short labels."""
    if df.empty:
        return pd.Series(dtype=float)
    totals = df.groupby("fuel")[col].sum()
    totals.index = totals.index.map(lambda f: FUEL_SHORT.get(f, f))
    return totals


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Seasonal hourly profiles (withdrawal + consumption)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_seasonal_water(study: Path, out_path: Path):
    hours = np.arange(24)
    hours_fine = np.linspace(0, 23, 240)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)

    row_labels = [
        ("total_wd_gal", "Water Withdrawal\n(Bgal/hour)"),
        ("total_wc_gal", "Water Consumption\n(Bgal/hour)"),
    ]

    for col_i, (season, weeks) in enumerate(SEASONS.items()):
        axes[0, col_i].set_title(season, fontsize=13, fontweight="bold", pad=8)

        for row_i, (col, ylabel) in enumerate(row_labels):
            ax = axes[row_i, col_i]

            for mode_key, style in MODES.items():
                df = load_system_water(study, mode_key, weeks)
                avg = hourly_avg_water(df, col)
                avg_fine = np.interp(hours_fine, hours, avg)

                ax.plot(hours_fine, avg_fine,
                        color=style["color"], linestyle=style["linestyle"],
                        linewidth=style["linewidth"], label=style["label"],
                        zorder=3)

            ax.grid(True, alpha=0.3)
            if col_i == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row_i == 1:
                ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
                ax.set_xticklabels(["12am", "4am", "8am", "12pm",
                                     "4pm", "8pm", "11pm"], fontsize=8)
                ax.set_xlabel("Hour of Day", fontsize=9)

    handles = [
        plt.Line2D([0], [0], color=s["color"], linestyle=s["linestyle"],
                    linewidth=s["linewidth"], label=s["label"])
        for s in MODES.values()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "Water Use by Season: Hourly Withdrawal and Consumption Profiles\n"
        "Baseline + 3 CAS Modes  |  Texas-7k DC Grid (6 weeks/season)",
        fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Fuel decomposition — what drives water changes
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fuel_decomposition(study: Path, out_path: Path):
    """Bar chart: per-fuel water withdrawal change (mode − baseline), annual."""
    all_weeks = [w for wks in SEASONS.values() for w in wks]

    # Load baseline fuel totals
    bl_fuel = total_water_by_fuel(
        load_fuel_water(study, "baseline", all_weeks), "wd_gal")

    fuels = sorted(bl_fuel.index, key=lambda f: bl_fuel.get(f, 0),
                   reverse=True)
    # Filter to fuels with non-zero baseline
    fuels = [f for f in fuels if bl_fuel.get(f, 0) > 0]

    n_fuels = len(fuels)
    n_modes = len(CAS_MODES)
    x = np.arange(n_fuels)
    width = 0.25
    offsets = np.array([-1, 0, 1]) * width

    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left panel: absolute change (Bgal) ---
    for i, (mode_key, style) in enumerate(CAS_MODES.items()):
        mode_fuel = total_water_by_fuel(
            load_fuel_water(study, mode_key, all_weeks), "wd_gal")

        deltas = [(mode_fuel.get(f, 0) - bl_fuel.get(f, 0)) * GAL_TO_BGAL
                  for f in fuels]

        bars = ax_abs.bar(x + offsets[i], deltas, width,
                          label=style["label"], color=style["color"],
                          edgecolor="white", linewidth=0.5, zorder=3)

        for bar, d in zip(bars, deltas):
            if abs(d) > 0.001:
                va = "bottom" if d >= 0 else "top"
                off = 0.002 if d >= 0 else -0.002
                ax_abs.text(bar.get_x() + bar.get_width() / 2, d + off,
                            f"{d:+.3f}", ha="center", va=va, fontsize=7,
                            color=style["color"])

    ax_abs.axhline(0, color="#333", linewidth=0.8, zorder=2)
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(fuels, fontsize=10)
    ax_abs.set_ylabel("Change in Water Withdrawal (Bgal)", fontsize=11)
    ax_abs.set_title("Absolute Change by Fuel", fontsize=12, fontweight="bold")
    ax_abs.legend(fontsize=9, framealpha=0.9)
    ax_abs.grid(True, axis="y", alpha=0.3)

    # --- Right panel: % change ---
    for i, (mode_key, style) in enumerate(CAS_MODES.items()):
        mode_fuel = total_water_by_fuel(
            load_fuel_water(study, mode_key, all_weeks), "wd_gal")

        pcts = []
        for f in fuels:
            bl_val = bl_fuel.get(f, 0)
            if bl_val > 0:
                pcts.append((mode_fuel.get(f, 0) - bl_val) / bl_val * 100)
            else:
                pcts.append(0.0)

        bars = ax_pct.bar(x + offsets[i], pcts, width,
                          label=style["label"], color=style["color"],
                          edgecolor="white", linewidth=0.5, zorder=3)

        for bar, p in zip(bars, pcts):
            if abs(p) > 0.5:
                va = "bottom" if p >= 0 else "top"
                off = 0.3 if p >= 0 else -0.3
                ax_pct.text(bar.get_x() + bar.get_width() / 2, p + off,
                            f"{p:+.1f}%", ha="center", va=va, fontsize=7,
                            color=style["color"])

    ax_pct.axhline(0, color="#333", linewidth=0.8, zorder=2)
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(fuels, fontsize=10)
    ax_pct.set_ylabel("Change in Water Withdrawal (%)", fontsize=11)
    ax_pct.set_title("Relative Change by Fuel", fontsize=12, fontweight="bold")
    ax_pct.legend(fontsize=9, framealpha=0.9)
    ax_pct.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Water Withdrawal Change vs. Baseline: Fuel-Type Decomposition\n"
        "Texas-7k DC Grid  |  All 24 biweekly weeks (TX_2018_ANNUAL)",
        fontsize=13, fontweight="bold", y=1.03)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Baseline fuel-type water pie (context)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_baseline_breakdown(study: Path, out_path: Path):
    """Stacked bar: baseline water withdrawal by fuel, per season."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

    for ax, (season, weeks) in zip(axes, SEASONS.items()):
        bl_fuel = total_water_by_fuel(
            load_fuel_water(study, "baseline", weeks), "wd_gal")
        bl_fuel = bl_fuel[bl_fuel > 0].sort_values(ascending=False) * GAL_TO_BGAL

        colors = [FUEL_COLORS.get(f, "#999") for f in bl_fuel.index]
        ax.barh(bl_fuel.index, bl_fuel.values, color=colors,
                edgecolor="white", linewidth=0.5)

        for j, (fuel, val) in enumerate(bl_fuel.items()):
            ax.text(val + 0.002, j, f"{val:.3f}", va="center",
                    fontsize=8, color="#333")

        ax.set_title(season, fontsize=12, fontweight="bold")
        ax.set_xlabel("Withdrawal (Bgal)", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        "Baseline Water Withdrawal by Fuel Type and Season\n"
        "Texas-7k DC Grid (6 weeks/season)",
        fontsize=13, fontweight="bold", y=1.03)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Water use analysis: seasonal profiles + fuel decomposition")
    parser.add_argument("--study-dir",
                        default="outputs/TX_2018_ANNUAL",
                        help="Path to TX_2018_ANNUAL output directory")
    parser.add_argument("--output-dir",
                        default="outputs/TX_2018_ANNUAL",
                        help="Directory for output images")
    args = parser.parse_args()

    study = Path(args.study_dir)
    out = Path(args.output_dir)

    plot_seasonal_water(study, out / "water_seasonal_profiles.png")
    plot_fuel_decomposition(study, out / "water_fuel_decomposition.png")
    plot_baseline_breakdown(study, out / "water_baseline_breakdown.png")


if __name__ == "__main__":
    main()
