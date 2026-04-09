#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""4×3 CAS metrics panel: demand, carbon emissions, renewable coverage, cost.

Thesis: Figure 10 — "Average daily profile for demand load, carbon intensity,
    renewable coverage, and operational cost across all CAS modes."

Reads hourly_summary.csv and thermal_detail.csv from baseline and CAS
simulation outputs across all weekly VATIC runs (TX_2018_ANNUAL), computes
hour-of-day averages, and plots a 4-row × 3-column comparison.

Usage:
    module load anaconda3/2024.10
    python scripts/cas_metrics_panels.py [--study-dir PATH]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import CAS_MODES as _CAS_MODES, EMISSION_FACTORS_BY_FUEL

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CAS_MODES = [{"sim_dir": k, **v} for k, v in _CAS_MODES.items()]

BASELINE_DIR = "baseline"

# Weeks to exclude globally (anomalous dispatch convergence on 2018-07-01).
EXCLUDE_WEEKS: set[str] = {"2018-07-01"}

EMISSION_FACTORS = EMISSION_FACTORS_BY_FUEL

# Row definitions: (metric_key, y-label, transform applied to raw hourly value)
METRICS = [
    ("demand_gw",    "System Demand\n(GW)"),
    ("renew_pct",    "Renewable Coverage\n(%)"),
    ("co2_kt_per_h", "CO₂ Emissions\n(kt/h)"),
    ("ci_kg_mwh",    "Carbon Intensity\n(kg CO₂/MWh)"),
    ("cost_m_per_h", "Operational Cost\n($M/h)"),
]

GEN_CSV = Path("vatic/data/grids/Texas-7k/TX_Data/SourceData/gen.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_gen_lookup(gen_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gen_csv, usecols=["GEN UID", "Fuel"])
    return df.rename(columns={"GEN UID": "Generator"})


def _skip_week(sim_subdir: str, week_dir: Path) -> bool:
    return week_dir.name in EXCLUDE_WEEKS


def load_hourly_metrics(study: Path, sim_subdir: str) -> pd.DataFrame:
    """Load hourly_summary.csv from all weeks; return concatenated DataFrame."""
    frames = []
    for week_dir in sorted(study.iterdir()):
        if _skip_week(sim_subdir, week_dir):
            continue
        csv_path = week_dir / sim_subdir / "hourly_summary.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, usecols=[
                "Date", "Hour", "Demand", "RenewablesUsed",
                "FixedCosts", "VariableCosts",
            ])
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No hourly_summary.csv for {sim_subdir} in {study}")
    return pd.concat(frames, ignore_index=True)


def load_hourly_co2(study: Path, sim_subdir: str,
                    gen_lookup: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly total CO₂ from thermal_detail.csv + gen lookup.

    Returns DataFrame with columns: Date, Hour, CO2_kg.
    """
    frames = []
    for week_dir in sorted(study.iterdir()):
        if _skip_week(sim_subdir, week_dir):
            continue
        th_path = week_dir / sim_subdir / "thermal_detail.csv"
        if not th_path.exists():
            continue

        th = pd.read_csv(th_path, usecols=["Date", "Hour", "Generator", "Dispatch"])
        th = th.merge(gen_lookup, on="Generator", how="left")
        th["CO2_kg"] = th["Dispatch"] * th["Fuel"].map(EMISSION_FACTORS).fillna(0)

        co2_hourly = th.groupby(["Date", "Hour"])["CO2_kg"].sum().reset_index()
        frames.append(co2_hourly[["Date", "Hour", "CO2_kg"]])

    if not frames:
        raise FileNotFoundError(
            f"No thermal_detail.csv for {sim_subdir} in {study}")
    return pd.concat(frames, ignore_index=True)


def compute_hourly_averages(study: Path, sim_subdir: str,
                            gen_lookup: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return dict of metric_key → 24-element array of hour-of-day averages."""
    hs = load_hourly_metrics(study, sim_subdir)
    co2_df = load_hourly_co2(study, sim_subdir, gen_lookup)

    demand_avg = hs.groupby("Hour")["Demand"].mean().sort_index().values
    renew_avg = hs.groupby("Hour")["RenewablesUsed"].mean().sort_index().values
    cost_avg = (
        hs.groupby("Hour")[["FixedCosts", "VariableCosts"]]
        .mean().sum(axis=1).sort_index().values
    )
    co2_avg = co2_df.groupby("Hour")["CO2_kg"].mean().sort_index().values

    return {
        "demand_gw":    demand_avg / 1e3,
        "ci_kg_mwh":    co2_avg / demand_avg,  # kg CO₂ / MWh
        "co2_kt_per_h": co2_avg / 1e6,   # kg → kt
        "renew_pct":    renew_avg / demand_avg * 100,
        "cost_m_per_h": cost_avg / 1e6,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_panels(study: Path, gen_lookup: pd.DataFrame, out_path: Path):
    print("Loading baseline metrics...")
    bl = compute_hourly_averages(study, BASELINE_DIR, gen_lookup)

    hours = np.arange(25)
    hours_fine = np.linspace(0, 24, 240)

    n_rows = len(METRICS)
    n_cols = len(CAS_MODES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows),
                             sharey="row")

    for col, mode in enumerate(CAS_MODES):
        print(f"Loading {mode['label']}...")
        cas = compute_hourly_averages(study, mode["sim_dir"], gen_lookup)
        color = mode["color"]

        for row, (key, ylabel) in enumerate(METRICS):
            ax = axes[row, col]

            bl_wrap = np.append(bl[key], bl[key][0])
            cas_wrap = np.append(cas[key], cas[key][0])
            bl_fine = np.interp(hours_fine, hours, bl_wrap)
            cas_fine = np.interp(hours_fine, hours, cas_wrap)

            ax.plot(hours_fine, bl_fine, color="black", linewidth=2,
                    linestyle="--", zorder=3)
            ax.plot(hours_fine, cas_fine, color="black", linewidth=2,
                    zorder=3)

            ax.fill_between(hours_fine, bl_fine, cas_fine,
                            where=cas_fine >= bl_fine, interpolate=True,
                            alpha=0.25, color=color, zorder=2)
            ax.fill_between(hours_fine, bl_fine, cas_fine,
                            where=cas_fine < bl_fine, interpolate=True,
                            alpha=0.25, color=color, zorder=2)

            ax.set_xlim(0, 24)
            ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
            ax.tick_params(labelsize=15)
            ax.grid(True, alpha=0.3)

            # x-axis labels only on bottom row
            if row == n_rows - 1:
                ax.set_xticklabels(
                    ["12am", "4am", "8am", "12pm", "4pm", "8pm", "12am"])
                ax.set_xlabel("Hour of Day", fontsize=17)
            else:
                ax.set_xticklabels([])

            # y-labels on left column only
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=17)

            # Column titles on top row only
            if row == 0:
                ax.set_title(mode["label"], fontsize=20, fontweight="bold",
                             color=color, pad=12)

    # Shared figure legend (black dashed = baseline, black solid = CAS)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="--",
               label="No CAS (baseline)"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="-",
               label="CAS"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=2, fontsize=16, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(
        "Average 24-Hour Metrics by CAS Mode",
        fontsize=22, fontweight="bold", y=1.04,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Bar chart — aggregate % change vs baseline
# ═══════════════════════════════════════════════════════════════════════════════

BAR_METRICS = [
    ("co2_t",       "Total CO\u2082"),
    ("renew_pct",   "Mean Renewable\nCoverage"),
    ("cost_usd",    "Total Operational\nCost"),
]


def compute_aggregates(study: Path, sim_subdir: str,
                       gen_lookup: pd.DataFrame) -> dict[str, float]:
    """Return scalar aggregates for a scenario across all weeks."""
    hs = load_hourly_metrics(study, sim_subdir)

    total_demand = hs["Demand"].sum()
    total_renew = hs["RenewablesUsed"].sum()
    total_cost = hs["FixedCosts"].sum() + hs["VariableCosts"].sum()

    # Total CO₂ from thermal dispatch
    co2_total = 0.0
    for week_dir in sorted(study.iterdir()):
        if _skip_week(sim_subdir, week_dir):
            continue
        th_path = week_dir / sim_subdir / "thermal_detail.csv"
        if not th_path.exists():
            continue
        th = pd.read_csv(th_path, usecols=["Generator", "Dispatch"])
        th = th.merge(gen_lookup, on="Generator", how="left")
        co2_total += (th["Dispatch"] * th["Fuel"].map(EMISSION_FACTORS).fillna(0)).sum()

    return {
        "co2_t":      co2_total / 1e3,       # kt
        "renew_pct":  total_renew / total_demand * 100,
        "cost_usd":   total_cost,
    }


def plot_bar_chart(study: Path, gen_lookup: pd.DataFrame, out_path: Path):
    """Grouped bar chart: % change vs baseline for each CAS mode."""
    from matplotlib.lines import Line2D

    print("Computing baseline aggregates...")
    bl = compute_aggregates(study, BASELINE_DIR, gen_lookup)

    pct_changes: dict[str, list[float]] = {k: [] for k, _ in BAR_METRICS}
    cas_labels = []

    for mode in CAS_MODES:
        print(f"Computing {mode['label']} aggregates...")
        cas = compute_aggregates(study, mode["sim_dir"], gen_lookup)
        cas_labels.append(mode["label"])
        for key, _ in BAR_METRICS:
            pct_changes[key].append((cas[key] - bl[key]) / bl[key] * 100)

    n_metrics = len(BAR_METRICS)
    n_modes = len(CAS_MODES)
    x = np.arange(n_metrics)
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, mode in enumerate(CAS_MODES):
        vals = [pct_changes[k][i] for k, _ in BAR_METRICS]
        offset = (i - (n_modes - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, color=mode["color"],
                      alpha=0.85, edgecolor="white", linewidth=0.5,
                      label=mode["label"])

        for bar, v in zip(bars, vals):
            va = "bottom" if v >= 0 else "top"
            y_off = 0.15 if v >= 0 else -0.15
            ax.text(bar.get_x() + bar.get_width() / 2, v + y_off,
                    f"{v:+.2f}%", ha="center", va=va, fontsize=10)

    ax.axhline(0, color="black", linewidth=0.8)
    # Pad y-axis so labels on lowest bars don't clip
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in BAR_METRICS], fontsize=13)
    ax.set_ylabel("Change vs. Baseline (%)", fontsize=15)
    ax.set_title("CAS Mode Performance vs. Baseline",
                 fontsize=20, fontweight="bold", pad=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(fontsize=12, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Fuel mix stacked area chart  (1×4: baseline + 3 CAS modes)
# ═══════════════════════════════════════════════════════════════════════════════

FUEL_GROUPS = [
    ("Nuclear",  ["NUC (Nuclear)"],                                       "#E74C3C"),
    ("Coal",     ["SUB (Subbituminous Coal)", "LIG (Lignite Coal)"],       "#636363"),
    ("Gas",      ["NG (Natural Gas)", "OG (Other Gas)",
                  "PUR (Purchased Steam)"],                               "#FF8C00"),
    ("Wind",     ["WND (Wind)"],                                          "#2ECC71"),
    ("Solar",    ["SUN (Solar)"],                                         "#F1C40F"),
    ("Hydro",    ["WAT (Water)"],                                         "#19D3F3"),
    ("Other",    ["PC (Petroleum Coke)", "WDS (Wood/Wood Waste Solids)",
                  "AB (Agricultural By-Products)", "WH (Waste Heat)",
                  "OTH (Other)",
                  "MWH (Electricity use for Energy Storage)"],            "#B0B0B0"),
]


def load_gen_fuel_lookup(gen_csv: Path) -> pd.DataFrame:
    """Load GEN UID → Fuel → FuelGroup mapping."""
    df = pd.read_csv(gen_csv, usecols=["GEN UID", "Fuel"])
    df = df.rename(columns={"GEN UID": "Generator"})

    fuel_to_group = {}
    for group_name, fuels, _ in FUEL_GROUPS:
        for f in fuels:
            fuel_to_group[f] = group_name
    df["FuelGroup"] = df["Fuel"].map(fuel_to_group).fillna("Other")
    return df[["Generator", "FuelGroup"]]


def load_hourly_fuel_mix(study: Path, sim_subdir: str,
                         gen_fuel: pd.DataFrame) -> pd.DataFrame:
    """Load thermal_detail + renew_detail for all weeks.

    Returns DataFrame with columns: Hour, FuelGroup, Dispatch_GW
    (hour-of-day averages in GW).
    """
    frames = []
    for week_dir in sorted(study.iterdir()):
        if _skip_week(sim_subdir, week_dir):
            continue
        # Thermal dispatch
        th_path = week_dir / sim_subdir / "thermal_detail.csv"
        if th_path.exists():
            th = pd.read_csv(th_path, usecols=["Date", "Hour", "Generator",
                                                "Dispatch"])
            frames.append(th.rename(columns={"Dispatch": "MW"}))

        # Renewable output
        rn_path = week_dir / sim_subdir / "renew_detail.csv"
        if rn_path.exists() and rn_path.stat().st_size > 50:
            rn = pd.read_csv(rn_path, usecols=["Date", "Hour", "Generator",
                                                "Output"])
            frames.append(rn.rename(columns={"Output": "MW"}))

    if not frames:
        raise FileNotFoundError(
            f"No thermal/renew detail for {sim_subdir} in {study}")

    all_gen = pd.concat(frames, ignore_index=True)
    all_gen = all_gen.merge(gen_fuel, on="Generator", how="left")
    all_gen["FuelGroup"] = all_gen["FuelGroup"].fillna("Other")

    # Hour-of-day average dispatch per fuel group (MW → GW)
    pivot = (all_gen.groupby(["Date", "Hour", "FuelGroup"])["MW"]
             .sum().reset_index()
             .groupby(["Hour", "FuelGroup"])["MW"].mean().reset_index())
    pivot["GW"] = pivot["MW"] / 1e3
    return pivot[["Hour", "FuelGroup", "GW"]]


def plot_fuel_mix(study: Path, gen_csv: Path, out_path: Path):
    """1×4 stacked area: baseline + 3 CAS modes."""
    gen_fuel = load_gen_fuel_lookup(gen_csv)

    group_names = [g for g, _, _ in FUEL_GROUPS]
    group_colors = {g: c for g, _, c in FUEL_GROUPS}

    scenarios = [{"sim_dir": BASELINE_DIR, "label": "No CAS (Baseline)",
                  "title_color": "black"}]
    for mode in CAS_MODES:
        scenarios.append({"sim_dir": mode["sim_dir"],
                          "label": mode["label"],
                          "title_color": mode["color"]})

    n_cols = len(scenarios)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), sharey=True)

    hours = np.arange(25)
    hours_fine = np.linspace(0, 24, 240)

    for col, scen in enumerate(scenarios):
        ax = axes[col]
        print(f"Loading fuel mix: {scen['label']}...")
        pivot = load_hourly_fuel_mix(study, scen["sim_dir"], gen_fuel)

        # Build stacked arrays (ordered bottom → top)
        stack_fine = []
        labels_used = []
        for grp in group_names:
            sub = pivot[pivot["FuelGroup"] == grp].set_index("Hour")["GW"]
            if sub.empty:
                vals = np.zeros(24)
            else:
                vals = sub.reindex(range(24), fill_value=0).values
            # Midnight-wrap + interpolate
            vals_wrap = np.append(vals, vals[0])
            vals_fine = np.interp(hours_fine, hours, vals_wrap)
            stack_fine.append(vals_fine)
            labels_used.append(grp)

        ax.stackplot(hours_fine, *stack_fine,
                     labels=labels_used,
                     colors=[group_colors[g] for g in labels_used],
                     alpha=0.85)

        ax.set_xlim(0, 24)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
        ax.set_xticklabels(
            ["12am", "4am", "8am", "12pm", "4pm", "8pm", "12am"],
            fontsize=13)
        ax.set_xlabel("Hour of Day", fontsize=15)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_title(scen["label"], fontsize=18, fontweight="bold",
                     color=scen["title_color"], pad=10)

        if col == 0:
            ax.set_ylabel("Generation (GW)", fontsize=15)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center",
               ncol=len(group_names), fontsize=12, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("Average 24-Hour Generation Mix by Fuel Type",
                 fontsize=20, fontweight="bold", y=1.09)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="4×3 CAS metrics panel plot")
    parser.add_argument("--study-dir",
                        default="outputs/TX_2018_ANNUAL",
                        help="Path to TX_2018_ANNUAL output directory")
    parser.add_argument("--output",
                        default="plots/TX_2018_ANNUAL/cas_metrics_panels.png",
                        help="Output image path")
    parser.add_argument("--gen-csv", type=Path, default=GEN_CSV,
                        help="Generator CSV for fuel lookup")
    args = parser.parse_args()

    gen_lookup = load_gen_lookup(args.gen_csv)
    study = Path(args.study_dir)
    out = Path(args.output)
    plot_panels(study, gen_lookup, out)
    plot_bar_chart(study, gen_lookup, out.with_name("cas_bar_comparison.png"))
    plot_fuel_mix(study, args.gen_csv, out.with_name("cas_fuel_mix.png"))


if __name__ == "__main__":
    main()
