#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""DC impact analysis plots: no-DC vs DC baseline comparison across 2018.

Thesis: Figure 9 — "Seasonal estimates for carbon emissions, total costs,
    load shedding, and renewable curtailment."

Generates 4 figures:
  dc_demand_decomposition.png  — Seasonal demand profiles (2×2 area plot)
  dc_system_metrics.png        — CO₂, cost, shedding, curtailment by season
  dc_dispatch_shift.png        — Dispatch by fuel type (stacked bar)
  dc_lmp_impact.png            — LMP distribution shift (box plot)

Usage:
    module load anaconda3/2024.10
    python scripts/dc_impact_plots.py [--study-dir outputs/TX_2018_ANNUAL]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# =============================================================================
# Style — consistent with pareto_analysis.py and tx_datacenter_map.py
# =============================================================================

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "figure.dpi":        200,
})

FUEL_COLORS = {
    "Nuclear":     "#7c3aed",
    "Coal":        "#1e293b",
    "Natural Gas": "#ea580c",
    "Wind":        "#16a34a",
    "Solar":       "#eab308",
    "Hydro":       "#0ea5e9",
    "Storage":     "#8b5cf6",
    "Other":       "#94a3b8",
}

DC_COLOR = "#ec4899"   # hot pink — matches grid map convention
NODC_COLOR = "#3b82f6"  # blue
DC_FILL = "#fce7f3"    # light pink

from constants import EMISSION_FACTORS_BY_FUEL as EMISSION_FACTORS

# Map raw fuel strings to display fuel groups
FUEL_GROUP = {
    "NUC (Nuclear)":            "Nuclear",
    "SUB (Subbituminous Coal)": "Coal",
    "LIG (Lignite Coal)":      "Coal",
    "PC (Petroleum Coke)":     "Coal",
    "NG (Natural Gas)":        "Natural Gas",
    "OG (Other Gas)":          "Natural Gas",
    "PUR (Purchased Steam)":   "Other",
    "WH (Waste Heat)":         "Other",
    "WDS (Wood/Wood Waste Solids)": "Other",
    "AB (Agricultural By-Products)": "Other",
    "OTH (Other)":             "Other",
    "SUN (Solar)":             "Solar",
    "WND (Wind)":              "Wind",
    "WAT (Water)":             "Hydro",
    "MWH (Electricity use for Energy Storage)": "Storage",
}

# Season mapping: month -> season name
SEASON_MAP = {
    1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
    5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall", 12: "Winter",
}

# Representative weeks per season for demand profile panels
REP_WEEKS = {
    "Winter": "2018-01-07",
    "Spring": "2018-04-01",
    "Summer": "2018-07-15",
    "Fall":   "2018-10-21",
}

SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]

# =============================================================================
# Data loaders
# =============================================================================

def load_gen_lookup(repo: Path) -> pd.DataFrame:
    """Load generator -> fuel type mapping from gen.csv."""
    gen_path = repo / "vatic/data/grids/Texas-7k/TX_Data/SourceData/gen.csv"
    gen = pd.read_csv(gen_path)
    return gen[["GEN UID", "Fuel"]].rename(columns={"GEN UID": "Generator"})


def load_hourly(week_dir: Path, scenario: str) -> pd.DataFrame | None:
    """Load hourly_summary.csv for a given scenario."""
    path = week_dir / scenario / "hourly_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["Date"]) + pd.to_timedelta(df["Hour"], unit="h")
    return df


def load_thermal(week_dir: Path, scenario: str) -> pd.DataFrame | None:
    """Load thermal_detail.csv."""
    path = week_dir / scenario / "thermal_detail.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_renew(week_dir: Path, scenario: str) -> pd.DataFrame | None:
    """Load renew_detail.csv."""
    path = week_dir / scenario / "renew_detail.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_bus_injections(repo: Path) -> pd.DataFrame:
    """Load DC load injection timeseries (DAY_AHEAD)."""
    path = (repo / "vatic/data/grids/Texas-7k-DC-REAL/TX_Data/"
            "timeseries_data_files/BusInjections/DAY_AHEAD_bus_injections.csv")
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(
        df[["Year", "Month", "Day"]].rename(columns={"Year": "year", "Month": "month", "Day": "day"})
    ) + pd.to_timedelta(df["Period"], unit="h")
    bus_cols = [c for c in df.columns if c not in ("Year", "Month", "Day", "Period", "DateTime")]
    df["DC_Total_MW"] = df[bus_cols].sum(axis=1)
    return df[["DateTime", "DC_Total_MW"]]


def get_week_dirs(study_dir: Path) -> list[Path]:
    """Return sorted list of week directories (2018-MM-DD)."""
    return sorted(d for d in study_dir.iterdir()
                  if d.is_dir() and d.name.startswith("2018-"))


def assign_season(date_str: str) -> str:
    """Map a date string like '2018-07-15' to a season."""
    month = int(date_str.split("-")[1])
    return SEASON_MAP[month]


# =============================================================================
# Aggregation
# =============================================================================

def _extract_week_metrics(week_dir: Path, scenario: str, label: str,
                          gen_lookup: pd.DataFrame):
    """Extract metrics and dispatch for one week+scenario.

    Returns (record_dict, list_of_dispatch_dicts) or (None, None).
    """
    hourly = load_hourly(week_dir, scenario)
    thermal = load_thermal(week_dir, scenario)
    if hourly is None or thermal is None:
        return None, None

    date_str = week_dir.name
    season = assign_season(date_str)

    th = thermal.merge(gen_lookup, on="Generator", how="left")
    th["FuelGroup"] = th["Fuel"].map(FUEL_GROUP).fillna("Other")
    th["EF"] = th["Fuel"].map(EMISSION_FACTORS).fillna(0)
    th["CO2_kg"] = th["Dispatch"] * th["EF"]

    total_cost = hourly["FixedCosts"].sum() + hourly["VariableCosts"].sum()
    total_co2 = th["CO2_kg"].sum()
    total_shedding = hourly["LoadShedding"].sum()
    total_curtailment = hourly["RenewablesCurtailment"].sum()
    total_demand = hourly["Demand"].sum()

    rec = {
        "week": date_str, "season": season, "scenario": label,
        "cost_M": total_cost / 1e6,
        "co2_kt": total_co2 / 1e6,
        "shedding_GWh": total_shedding / 1e3,
        "curtailment_GWh": total_curtailment / 1e3,
        "demand_GWh": total_demand / 1e3,
    }

    disp_recs = []
    for fuel, mwh in th.groupby("FuelGroup")["Dispatch"].sum().items():
        disp_recs.append({
            "week": date_str, "season": season, "scenario": label,
            "fuel": fuel, "dispatch_GWh": mwh / 1e3,
        })

    return rec, disp_recs


def aggregate_all_weeks(study_dir: Path, gen_lookup: pd.DataFrame):
    """Aggregate hourly and thermal data across all weeks for both scenarios.

    DC ("baseline") metrics use ALL available weeks — actual seasonal totals.
    No DC metrics use the single representative week per season, scaled by
    the number of biweekly periods in that season to approximate the seasonal
    total.

    Returns (records, dispatch_records).
    """
    records = []
    dispatch_records = []

    # Count biweekly periods per season (for No DC scaling)
    season_week_counts: dict[str, int] = {}
    for week_dir in get_week_dirs(study_dir):
        season = assign_season(week_dir.name)
        season_week_counts[season] = season_week_counts.get(season, 0) + 1

    # DC: collect from ALL weeks that have baseline data
    for week_dir in get_week_dirs(study_dir):
        rec, drecs = _extract_week_metrics(
            week_dir, "baseline", "DC", gen_lookup)
        if rec is not None:
            records.append(rec)
            dispatch_records.extend(drecs)

    # No DC: collect from representative weeks, scale to seasonal estimate
    metric_cols = ["cost_M", "co2_kt", "shedding_GWh",
                   "curtailment_GWh", "demand_GWh"]
    for week_dir in get_week_dirs(study_dir):
        if not (week_dir / "no-dc" / "hourly_summary.csv").exists():
            continue
        rec, drecs = _extract_week_metrics(
            week_dir, "no-dc", "No DC", gen_lookup)
        if rec is None:
            continue
        scale = season_week_counts.get(rec["season"], 1)
        for col in metric_cols:
            rec[col] *= scale
        for d in drecs:
            d["dispatch_GWh"] *= scale
        records.append(rec)
        dispatch_records.extend(drecs)

    return pd.DataFrame(records), pd.DataFrame(dispatch_records)


# =============================================================================
# Figure 1: Demand decomposition (2×2 seasonal panels)
# =============================================================================

def fig_demand_decomposition(study_dir: Path, dc_inj: pd.DataFrame, out_dir: Path):
    """2×2 area plot showing DC load component of total demand."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=True)

    for ax, season in zip(axes.flat, SEASON_ORDER):
        date_str = REP_WEEKS[season]
        week_dir = study_dir / date_str

        # DC scenario demand
        hourly_dc = load_hourly(week_dir, "baseline")
        # No-DC scenario demand
        hourly_nodc = load_hourly(week_dir, "no-dc")

        if hourly_dc is None:
            ax.set_title(f"{season} — no DC data")
            continue

        hours = np.arange(len(hourly_dc))

        if hourly_nodc is not None:
            nodc_demand = hourly_nodc["Demand"].values / 1e3  # GW
            dc_demand = hourly_dc["Demand"].values / 1e3
            dc_component = dc_demand - nodc_demand

            ax.fill_between(hours, 0, nodc_demand, alpha=0.4,
                            color=NODC_COLOR, label="Native Texas-7k load")
            ax.fill_between(hours, nodc_demand, nodc_demand + np.maximum(dc_component, 0),
                            alpha=0.5, color=DC_COLOR, label="DC load addition")
            ax.plot(hours, dc_demand, color="#1e293b", lw=0.8, alpha=0.7)
        else:
            # Fallback: use BusInjections data for DC component estimate
            dc_demand = hourly_dc["Demand"].values / 1e3
            start = pd.Timestamp(date_str)
            end = start + pd.Timedelta(days=7)
            mask = (dc_inj["DateTime"] >= start) & (dc_inj["DateTime"] < end)
            dc_load = dc_inj.loc[mask, "DC_Total_MW"].values / 1e3  # GW
            min_len = min(len(dc_demand), len(dc_load))
            native = dc_demand[:min_len] - dc_load[:min_len]

            ax.fill_between(hours[:min_len], 0, native, alpha=0.4,
                            color=NODC_COLOR, label="Native Texas-7k load")
            ax.fill_between(hours[:min_len], native, dc_demand[:min_len],
                            alpha=0.5, color=DC_COLOR, label="DC load addition")
            ax.plot(hours[:min_len], dc_demand[:min_len], color="#1e293b", lw=0.8, alpha=0.7)

        ax.set_title(f"{season} ({date_str})", fontweight="bold")
        ax.set_xlabel("Day")
        ax.set_ylabel("Demand (GW)")
        ax.set_xlim(0, 168)
        ax.set_xticks(np.arange(0, 169, 24))
        ax.set_xticklabels([str(d) for d in range(1, 9)][:len(np.arange(0, 169, 24))])

    axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("Demand Decomposition: Native Texas-7k Load + Data Center Injection",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = out_dir / "dc_demand_decomposition.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# Figure 2: System metrics comparison (grouped bar)
# =============================================================================

def fig_system_metrics(df: pd.DataFrame, out_dir: Path):
    """Grouped bar chart: seasonal aggregates for no-DC vs DC."""
    metrics = [
        ("co2_kt", "CO₂ (kt)", "#64748b"),
        ("cost_M", "Total Cost ($M)", "#0ea5e9"),
        ("shedding_GWh", "Load Shedding (GWh)", "#ef4444"),
        ("curtailment_GWh", "Renewable Curtailment (GWh)", "#eab308"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (col, title, color) in zip(axes.flat, metrics):
        seasonal = (df.groupby(["season", "scenario"])[col].sum()
                    .unstack("scenario")
                    .reindex(SEASON_ORDER))

        x = np.arange(len(SEASON_ORDER))
        w = 0.35

        bars_nodc = ax.bar(x - w/2, seasonal.get("No DC", 0), w,
                           label="No DC", color=NODC_COLOR, alpha=0.8)
        bars_dc = ax.bar(x + w/2, seasonal.get("DC", 0), w,
                         label="DC", color=DC_COLOR, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(SEASON_ORDER)
        ax.set_title(title, fontweight="bold")

        # Add value labels on bars
        for bars in [bars_nodc, bars_dc]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:,.0f}",
                            xy=(bar.get_x() + bar.get_width()/2, max(h, 0)),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    axes[0, 0].legend(fontsize=9, framealpha=0.9)
    fig.suptitle("Seasonal System Metrics: No DC vs DC",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = out_dir / "dc_system_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# Figure 3: Dispatch by fuel type (stacked bar, winter vs summer)
# =============================================================================

def fig_dispatch_shift(disp_df: pd.DataFrame, out_dir: Path):
    """Stacked bar chart: dispatch by fuel type for winter vs summer, no-DC vs DC."""
    fuel_order = ["Nuclear", "Coal", "Natural Gas", "Hydro", "Wind", "Solar", "Storage", "Other"]

    seasons_shown = SEASON_ORDER
    scenarios = ["No DC", "DC"]
    bar_labels = [f"{s}\n{sc}" for s in seasons_shown for sc in scenarios]

    # Aggregate dispatch by season, scenario, fuel
    agg = (disp_df[disp_df["season"].isin(seasons_shown)]
           .groupby(["season", "scenario", "fuel"])["dispatch_GWh"]
           .sum().reset_index())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(bar_labels))
    bottoms = np.zeros(len(bar_labels))

    for fuel in fuel_order:
        heights = []
        for season in seasons_shown:
            for scenario in scenarios:
                val = agg.loc[(agg["season"] == season) &
                              (agg["scenario"] == scenario) &
                              (agg["fuel"] == fuel), "dispatch_GWh"]
                heights.append(val.values[0] if len(val) > 0 else 0)
        heights = np.array(heights)
        ax.bar(x, heights, bottom=bottoms, label=fuel,
               color=FUEL_COLORS.get(fuel, "#94a3b8"), width=0.6)
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("Generation (GWh)")
    ax.set_title("Dispatch by Fuel Type: No DC vs DC by Season",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)

    fig.tight_layout()
    path = out_dir / "dc_dispatch_shift.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# Figure 4: LMP distribution shift (box plot)
# =============================================================================

def fig_lmp_impact(study_dir: Path, out_dir: Path):
    """Box plot of LMP distributions: no-DC vs DC by season."""
    data = {season: {"No DC": [], "DC": []} for season in SEASON_ORDER}

    for week_dir in get_week_dirs(study_dir):
        date_str = week_dir.name
        season = assign_season(date_str)

        # Only include weeks where both scenarios exist
        if not (week_dir / "no-dc" / "hourly_summary.csv").exists():
            continue

        for scenario, label in [("no-dc", "No DC"), ("baseline", "DC")]:
            hourly = load_hourly(week_dir, scenario)
            if hourly is not None:
                data[season][label].extend(hourly["Price"].dropna().tolist())

    fig, ax = plt.subplots(figsize=(10, 6))

    positions = []
    labels = []
    box_data = []
    colors = []

    for i, season in enumerate(SEASON_ORDER):
        for j, (scenario, color) in enumerate([("No DC", NODC_COLOR), ("DC", DC_COLOR)]):
            pos = i * 3 + j
            positions.append(pos)
            labels.append(f"{season}\n{scenario}" if j == 0 else f"\n{scenario}")
            vals = data[season][scenario]
            box_data.append(vals if vals else [0])
            colors.append(color)

    bp = ax.boxplot(box_data, positions=positions, widths=0.8, patch_artist=True,
                    showfliers=False, medianprops=dict(color="white", lw=1.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Custom x labels — just season names centered between pairs
    season_centers = [i * 3 + 0.5 for i in range(len(SEASON_ORDER))]
    ax.set_xticks(season_centers)
    ax.set_xticklabels(SEASON_ORDER)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=NODC_COLOR, alpha=0.7, label="No DC"),
                       Patch(facecolor=DC_COLOR, alpha=0.7, label="DC")],
              fontsize=9, framealpha=0.9)

    ax.set_ylabel("LMP ($/MWh)")
    ax.set_title("Locational Marginal Price Distribution by Season",
                 fontweight="bold", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    fig.tight_layout()
    path = out_dir / "dc_lmp_impact.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DC impact analysis plots")
    parser.add_argument("--study-dir", type=Path,
                        default=Path("outputs/TX_2018_ANNUAL"),
                        help="Path to TX_2018_ANNUAL study directory")
    parser.add_argument("--repo", type=Path,
                        default=Path("/home/dp4800/Documents/GitHub/Vatic"),
                        help="Repository root (for gen.csv and BusInjections)")
    args = parser.parse_args()

    study_dir = args.study_dir.resolve()
    repo = args.repo.resolve()
    out_dir = study_dir
    print(f"Study dir: {study_dir}")
    print(f"Output dir: {out_dir}")

    # Load generator fuel lookup
    gen_lookup = load_gen_lookup(repo)
    print(f"Loaded {len(gen_lookup)} generators from gen.csv")

    # Load DC injection timeseries
    dc_inj = load_bus_injections(repo)
    print(f"Loaded DC injection data: {len(dc_inj)} hours")

    # Check data availability
    weeks = get_week_dirs(study_dir)
    n_nodc = sum(1 for w in weeks if (w / "no-dc" / "hourly_summary.csv").exists())
    n_dc = sum(1 for w in weeks if (w / "baseline" / "hourly_summary.csv").exists())
    print(f"Weeks with no-DC data: {n_nodc}/24")
    print(f"Weeks with DC data:    {n_dc}/24")

    if n_nodc == 0:
        print("ERROR: No no-DC simulation results found. Run the SLURM job first.")
        return

    # Aggregate data
    print("\nAggregating metrics across all weeks...")
    df, disp_df = aggregate_all_weeks(study_dir, gen_lookup)
    print(f"  {len(df)} week×scenario records, {len(disp_df)} dispatch records")

    # Generate figures
    print("\nGenerating figures...")
    fig_demand_decomposition(study_dir, dc_inj, out_dir)
    fig_system_metrics(df, out_dir)
    fig_dispatch_shift(disp_df, out_dir)
    fig_lmp_impact(study_dir, out_dir)

    print("\nDone! All figures saved to", out_dir)


if __name__ == "__main__":
    main()
