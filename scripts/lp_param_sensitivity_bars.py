#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""LP parameter sensitivity bar chart: 3-row panel showing how α, W, f
affect the same outcome metrics as cas_mode_bar_chart.

Thesis: Figure 15 — "Sensitivity of average demand, emissions and operational
    costs to perturbations of flexibility parameters."

Each row isolates one parameter (others held at baseline values).
Bars show % change vs. no-CAS baseline, averaged across 8 sweep weeks.

Usage:
    module load anaconda3/2024.10
    python scripts/lp_param_sensitivity_bars.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import EMISSION_FACTORS, FUEL_CATEGORY, SWEEP_WEEKS

BASELINE_COLOR = "#059669"          # consistent emerald for all default values

PARAM_ROWS = [
    {
        "title": r"Carbon-Cost Trade-off ($\alpha$)",
        "param_label": r"$\alpha$",
        "variants": [
            {"dir": "sim-lp-alpha-alpha_0.00", "label": "0.0 (cost only)",
             "color": "#DC2626", "is_baseline": False},
            {"dir": "sim-lp-alpha-alpha_0.25", "label": "0.25",
             "color": "#D97706", "is_baseline": False},
            {"dir": "sim-lp",                  "label": "0.5 (default)",
             "color": BASELINE_COLOR, "is_baseline": True},
            {"dir": "sim-lp-alpha-alpha_0.75", "label": "0.75",
             "color": "#2563EB", "is_baseline": False},
            {"dir": "sim-lp-alpha-alpha_1.00", "label": "1.0 (carbon only)",
             "color": "#7C3AED", "is_baseline": False},
        ],
    },
    {
        "title": r"Deferral Window ($W$)",
        "param_label": r"$W$",
        "variants": [
            {"dir": "sim-lp-deferral-deferral_4h",  "label": "4 h",
             "color": "#D97706", "is_baseline": False},
            {"dir": "sim-lp",                        "label": "8 h (default)",
             "color": BASELINE_COLOR, "is_baseline": True},
            {"dir": "sim-lp-deferral-deferral_18h",  "label": "18 h",
             "color": "#2563EB", "is_baseline": False},
            {"dir": "sim-lp-deferral-deferral_24h",  "label": "24 h",
             "color": "#7C3AED", "is_baseline": False},
        ],
    },
    {
        "title": r"Flexible Workload Ratio ($f$)",
        "param_label": r"$f$",
        "variants": [
            {"dir": "sim-lp-flex-flex_20pct", "label": "20%",
             "color": "#D97706", "is_baseline": False},
            {"dir": "sim-lp",                 "label": "30% (default)",
             "color": BASELINE_COLOR, "is_baseline": True},
            {"dir": "sim-lp-flex-flex_40pct",  "label": "40%",
             "color": "#2563EB", "is_baseline": False},
            {"dir": "sim-lp-flex-flex_50pct",  "label": "50%",
             "color": "#7C3AED", "is_baseline": False},
        ],
    },
]

METRICS = [
    {"key": "co2",   "short": r"CO$_2$"},
    {"key": "cost",  "short": "Op. Cost"},
    {"key": "renew", "short": "Renew.\nCoverage"},
    {"key": "curt",  "short": "Renew.\nCurtailed"},
    {"key": "ls",    "short": "Load\nShedding"},
    {"key": "water", "short": "Water\nConsumption"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data computation
# ═══════════════════════════════════════════════════════════════════════════════

def load_gen_ef_map(grid_dir: Path) -> dict[str, float]:
    for subdir in ['TX_Data/SourceData', 'RTS_Data/SourceData']:
        gen_path = grid_dir / subdir / 'gen.csv'
        if gen_path.exists():
            break
    else:
        raise FileNotFoundError(f"No gen.csv found under {grid_dir}")
    gen = pd.read_csv(gen_path)
    return {
        str(row['GEN UID']): EMISSION_FACTORS.get(
            FUEL_CATEGORY.get(str(row.get('Fuel', '')), 'Natural Gas'), 0.0)
        for _, row in gen.iterrows()
    }


def compute_week_metrics(sim_dir: Path, water_dir: Path,
                         uid_to_ef: dict[str, float]) -> dict[str, float]:
    """Compute raw metrics for one scenario in one week."""
    hs_path = sim_dir / 'hourly_summary.csv'
    td_path = sim_dir / 'thermal_detail.csv'
    if not hs_path.exists():
        return {}

    hs = pd.read_csv(hs_path)
    result = {
        'cost':  (hs['FixedCosts'] + hs['VariableCosts']).sum(),
        'renew': hs['RenewablesUsed'].sum(),
        'curt':  hs['RenewablesCurtailment'].sum(),
        'ls':    hs['LoadShedding'].sum(),
    }

    if td_path.exists():
        td = pd.read_csv(td_path)
        td['co2_t'] = td['Generator'].map(uid_to_ef).fillna(0.0) * td['Dispatch']
        result['co2'] = td['co2_t'].sum()
    else:
        result['co2'] = np.nan

    water_path = water_dir / 'system_water_hourly.csv'
    if water_path.exists():
        water = pd.read_csv(water_path)
        result['water'] = water['total_wc_gal'].sum() if 'total_wc_gal' in water.columns else np.nan
    else:
        result['water'] = np.nan

    return result


def compute_pct_changes(study: Path, grid_dir: Path) -> dict:
    """Compute % change vs baseline for each parameter row × variant × metric.

    Returns dict[row_idx][variant_dir] = {metric_key: pct_change}.
    """
    uid_to_ef = load_gen_ef_map(grid_dir)
    results = {}

    for row_idx, row_cfg in enumerate(PARAM_ROWS):
        results[row_idx] = {}

        for var in row_cfg["variants"]:
            bl_totals = {m["key"]: 0.0 for m in METRICS}
            var_totals = {m["key"]: 0.0 for m in METRICS}
            n_valid = 0

            for week in SWEEP_WEEKS:
                week_dir = study / week

                # Baseline (no-CAS)
                bl = compute_week_metrics(
                    week_dir / 'baseline',
                    week_dir / 'water' / 'baseline',
                    uid_to_ef)

                # Variant
                vr = compute_week_metrics(
                    week_dir / var["dir"],
                    week_dir / 'water' / var["dir"],
                    uid_to_ef)

                if not bl or not vr:
                    continue

                for key in bl_totals:
                    bl_totals[key] += bl.get(key, 0.0)
                    var_totals[key] += vr.get(key, 0.0)
                n_valid += 1

            pcts = {}
            for key in bl_totals:
                bl_val = bl_totals[key]
                if bl_val and bl_val != 0 and not np.isnan(bl_val):
                    pcts[key] = (var_totals[key] - bl_val) / bl_val * 100.0
                else:
                    pcts[key] = 0.0

            results[row_idx][var["dir"]] = pcts
            print(f"  {row_cfg['title']}: {var['label']} ({n_valid} weeks)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sensitivity(study: Path, grid_dir: Path, out_path: Path):
    print("Computing metrics...")
    data = compute_pct_changes(study, grid_dir)

    n_metrics = len(METRICS)
    n_rows = len(PARAM_ROWS)

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4.5 * n_rows),
                             sharex=True)

    for row_idx, (ax, row_cfg) in enumerate(zip(axes, PARAM_ROWS)):
        variants = row_cfg["variants"]
        n_vars = len(variants)

        x = np.arange(n_metrics)
        total_w = 0.7
        bar_w = total_w / n_vars
        offsets = np.linspace(-total_w / 2 + bar_w / 2,
                              total_w / 2 - bar_w / 2, n_vars)

        for i, var in enumerate(variants):
            pcts = data[row_idx][var["dir"]]
            vals = [pcts.get(m["key"], 0.0) for m in METRICS]
            color = var["color"]
            edge = "white"
            lw = 0.5

            bars = ax.bar(x + offsets[i], vals, bar_w,
                          label=var["label"], color=color,
                          edgecolor=edge, linewidth=lw, zorder=3)

            for bar, v in zip(bars, vals):
                y_pos = bar.get_height()
                va = "bottom" if y_pos >= 0 else "top"
                offset = 0.15 if y_pos >= 0 else -0.15
                ax.text(bar.get_x() + bar.get_width() / 2,
                        y_pos + offset, f"{v:+.1f}%",
                        ha="center", va=va, fontsize=6,
                        fontweight="bold" if var["is_baseline"] else "normal",
                        color="#333")

        ax.axhline(0, color="#333", linewidth=0.8, zorder=2)
        ax.set_ylabel("Change vs. Baseline (%)", fontsize=10)
        ax.set_title(row_cfg["title"], fontsize=12, fontweight="bold", pad=8)
        ax.legend(fontsize=8.5, framealpha=0.9, loc="best",
                  title=f"{row_cfg['param_label']}  (* = baseline LP)",
                  title_fontsize=8.5)
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xticks(np.arange(n_metrics))
    axes[-1].set_xticklabels([m["short"] for m in METRICS], fontsize=11)

    fig.suptitle(
        "LP CAS Parameter Sensitivity: Isolated Single-Parameter Sweeps\n"
        f"Texas-7k DC Grid  |  {len(SWEEP_WEEKS)} biweekly sweep windows, 2018",
        fontsize=14, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="LP parameter sensitivity bar chart")
    parser.add_argument("--study-dir", default="outputs/TX_2018_ANNUAL")
    parser.add_argument("--grid-dir", default="vatic/data/grids/Texas-7k")
    parser.add_argument("--output",
                        default="plots/TX_2018_ANNUAL/lp_param_sensitivity_bars.png")
    args = parser.parse_args()
    plot_sensitivity(Path(args.study_dir), Path(args.grid_dir), Path(args.output))


if __name__ == "__main__":
    main()
