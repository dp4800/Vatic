#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""Sensitivity analysis: 24-hour profiles varying α, W, and flex ratio.

Thesis: Figure 14 — "Sensitivity of within-day patterns of demand, emissions
    and operational costs to perturbations of flexibility parameters."

Three columns (one per parameter sweep) × three rows (demand, carbon
emissions, operational cost) showing how average daily profiles change
as each CAS parameter is swept.

Usage:
    module load anaconda3/2024.10
    python scripts/cas_sensitivity_profiles.py [--study-dir PATH]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import EMISSION_FACTORS, FUEL_CATEGORY, SWEEP_WEEKS

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

PANELS = [
    {
        "title": r"$\alpha$ (Carbon Weight)",
        "param_label": r"$\alpha$",
        "curves": [
            {"dir": "sim-lp-alpha-alpha_0.00", "label": r"$\alpha$=0.00 (cost only)",
             "color": "#DC2626", "val": 0.00},
            {"dir": "sim-lp-alpha-alpha_0.25", "label": r"$\alpha$=0.25",
             "color": "#D97706", "val": 0.25},
            {"dir": "sim-lp",                  "label": r"$\alpha$=0.50 (default)",
             "color": "#059669", "val": 0.50},
            {"dir": "sim-lp-alpha-alpha_0.75", "label": r"$\alpha$=0.75",
             "color": "#2563EB", "val": 0.75},
            {"dir": "sim-lp-alpha-alpha_1.00", "label": r"$\alpha$=1.00 (carbon only)",
             "color": "#7C3AED", "val": 1.00},
        ],
    },
    {
        "title": "Deferral Window $W$",
        "param_label": "$W$",
        "curves": [
            {"dir": "sim-lp-deferral-deferral_4h",  "label": "$W$=4 h",
             "color": "#D97706", "val": 4},
            {"dir": "sim-lp-deferral-deferral_8h",  "label": "$W$=8 h (default)",
             "color": "#059669", "val": 8},
            {"dir": "sim-lp-deferral-deferral_18h", "label": "$W$=18 h",
             "color": "#2563EB", "val": 18},
            {"dir": "sim-lp-deferral-deferral_24h", "label": "$W$=24 h",
             "color": "#7C3AED", "val": 24},
        ],
    },
    {
        "title": r"Flexible Workload Ratio $f$",
        "param_label": r"$f$",
        "curves": [
            {"dir": "sim-lp-flex-flex_20pct", "label": r"$f$=20%",
             "color": "#D97706", "val": 20},
            {"dir": "sim-lp",                 "label": r"$f$=30% (default)",
             "color": "#059669", "val": 30},
            {"dir": "sim-lp-flex-flex_40pct", "label": r"$f$=40%",
             "color": "#2563EB", "val": 40},
            {"dir": "sim-lp-flex-flex_50pct", "label": r"$f$=50%",
             "color": "#7C3AED", "val": 50},
        ],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_gen_ef_map(grid_dir: Path) -> dict[str, float]:
    """Generator UID → CO₂ emission factor (tonnes/MWh)."""
    gen_path = grid_dir / 'TX_Data' / 'SourceData' / 'gen.csv'
    gen = pd.read_csv(gen_path)
    uid_to_ef = {}
    for _, row in gen.iterrows():
        uid = str(row['GEN UID'])
        fuel = str(row.get('Fuel', ''))
        category = FUEL_CATEGORY.get(fuel, 'Natural Gas')
        uid_to_ef[uid] = EMISSION_FACTORS.get(category, 0.0)
    return uid_to_ef


def load_hourly_summary(study: Path, sim_subdir: str,
                        weeks: list[str]) -> pd.DataFrame | None:
    """Load hourly_summary with Demand, cost columns from specified weeks."""
    frames = []
    for week in weeks:
        csv_path = study / week / sim_subdir / "hourly_summary.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path,
                             usecols=["Date", "Hour", "Demand",
                                      "FixedCosts", "VariableCosts"])
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_hourly_co2(study: Path, sim_subdir: str, weeks: list[str],
                    uid_to_ef: dict[str, float]) -> pd.DataFrame | None:
    """Compute hourly CO₂ (tonnes) from thermal_detail + emission factors."""
    frames = []
    for week in weeks:
        td_path = study / week / sim_subdir / "thermal_detail.csv"
        if td_path.exists():
            td = pd.read_csv(td_path, usecols=["Date", "Hour", "Generator", "Dispatch"])
            td["co2_t"] = td["Generator"].map(uid_to_ef).fillna(0.0) * td["Dispatch"]
            hourly = td.groupby(["Date", "Hour"], as_index=False)["co2_t"].sum()
            frames.append(hourly)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def hourly_avg(df: pd.DataFrame, col: str) -> np.ndarray:
    """Mean of *col* per hour-of-day (0–23)."""
    return df.groupby("Hour")[col].mean().sort_index().values


# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sensitivity(study: Path, grid_dir: Path, out_path: Path):
    uid_to_ef = build_gen_ef_map(grid_dir)

    hours = np.arange(24)
    hours_fine = np.linspace(0, 23, 240)

    # ── Pre-load baseline data ────────────────────────────────────────────────
    bl_hs = load_hourly_summary(study, "baseline", SWEEP_WEEKS)
    bl_co2 = load_hourly_co2(study, "baseline", SWEEP_WEEKS, uid_to_ef)

    bl_demand = np.interp(hours_fine, hours,
                          hourly_avg(bl_hs, "Demand") / 1000.0)
    bl_co2_avg = np.interp(hours_fine, hours,
                           hourly_avg(bl_co2, "co2_t") / 1000.0)
    bl_hs["cost"] = bl_hs["FixedCosts"] + bl_hs["VariableCosts"]
    bl_cost = np.interp(hours_fine, hours,
                        hourly_avg(bl_hs, "cost") / 1e6)

    # Row definitions: (baseline_fine, column_name, unit_label, ylabel)
    ROWS = [
        (bl_demand,  "demand", "GW",   "Avg. System Demand (GW)"),
        (bl_co2_avg, "co2",    "kt/h", "Avg. CO₂ Emissions (kt/h)"),
        (bl_cost,    "cost",   "$M/h", "Avg. Operational Cost ($M/h)"),
    ]

    # ── Figure: 3 rows × 3 columns ───────────────────────────────────────────
    fig, axes = plt.subplots(len(ROWS), len(PANELS),
                             figsize=(18, 5 * len(ROWS)),
                             sharex=True)

    for col_idx, panel in enumerate(PANELS):
        # Pre-load all curve data once per panel
        curve_data = {}
        for curve in panel["curves"]:
            hs = load_hourly_summary(study, curve["dir"], SWEEP_WEEKS)
            co2 = load_hourly_co2(study, curve["dir"], SWEEP_WEEKS, uid_to_ef)
            if hs is not None:
                hs["cost"] = hs["FixedCosts"] + hs["VariableCosts"]
                curve_data[curve["dir"]] = {
                    "demand": np.interp(hours_fine, hours,
                                        hourly_avg(hs, "Demand") / 1000.0),
                    "cost":   np.interp(hours_fine, hours,
                                        hourly_avg(hs, "cost") / 1e6),
                    "co2":    (np.interp(hours_fine, hours,
                                         hourly_avg(co2, "co2_t") / 1000.0)
                               if co2 is not None else None),
                }

        for row_idx, (bl_fine, row_key, _, ylabel) in enumerate(ROWS):
            ax = axes[row_idx, col_idx]

            # Baseline
            ax.plot(hours_fine, bl_fine, color="#555555", linewidth=2.2,
                    linestyle="--", label="No CAS (baseline)", zorder=4)

            for curve in panel["curves"]:
                cd = curve_data.get(curve["dir"])
                if cd is None or cd[row_key] is None:
                    continue
                cas_fine = cd[row_key]

                ax.plot(hours_fine, cas_fine, color=curve["color"],
                        linewidth=1.8, label=curve["label"], zorder=3)

            ax.grid(True, alpha=0.3)

            # Column titles on top row only
            if row_idx == 0:
                ax.set_title(panel["title"], fontsize=12,
                             fontweight="bold", pad=10)

            # Y-axis label on leftmost column only
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=11)

            # X-axis labels on bottom row only
            if row_idx == len(ROWS) - 1:
                ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
                ax.set_xticklabels(
                    ["12am", "4am", "8am", "12pm", "4pm", "8pm", "11pm"])
                ax.set_xlabel("Hour of Day", fontsize=11)

            # Legend in top-left of each panel (first row only to avoid clutter)
            if row_idx == 0:
                ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Share y-axis within each row
    for row_idx in range(len(ROWS)):
        row_axes = [axes[row_idx, c] for c in range(len(PANELS))]
        ymin = min(a.get_ylim()[0] for a in row_axes)
        ymax = max(a.get_ylim()[1] for a in row_axes)
        for a in row_axes:
            a.set_ylim(ymin, ymax)

    fig.suptitle(
        "LP CAS Parameter Sensitivity: Average 24-Hour Profiles",
        fontsize=15, fontweight="bold", y=1.01)
    fig.text(0.5, -0.01,
             f"Averaged over {len(SWEEP_WEEKS)} weekly VATIC runs "
             "(TX_2018_ANNUAL, 2 per season)",
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
        description="CAS sensitivity: 24-hour demand, CO₂, and cost profiles")
    parser.add_argument("--study-dir",
                        default="outputs/TX_2018_ANNUAL",
                        help="Path to TX_2018_ANNUAL output directory")
    parser.add_argument("--grid-dir",
                        default="vatic/data/grids/Texas-7k",
                        help="Path to grid directory with gen.csv")
    parser.add_argument("--output",
                        default="outputs/TX_2018_ANNUAL/cas_sensitivity_profiles.png",
                        help="Output image path")
    args = parser.parse_args()

    plot_sensitivity(Path(args.study_dir), Path(args.grid_dir),
                     Path(args.output))


if __name__ == "__main__":
    main()
