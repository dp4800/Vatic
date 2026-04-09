# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
compare_sim_outputs.py
Utilities consumed by compare_cas_modes.py (step 11) to build system-level
comparison DataFrames and plots across multiple VATIC simulation runs.

Public API expected by compare_cas_modes.py:
    build_comparison(sim_dirs, sim_labels, gen_csv, water_dirs=None) -> pd.DataFrame
    _delta_cols(df, baseline_label) -> pd.DataFrame
    make_plots(sim_dirs, sim_labels, gen_csv, comparison_df, out_dir, water_dirs=None)
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Emission factors (kg CO₂ / MWh) — thesis Table AI
# ---------------------------------------------------------------------------
_FALLBACK_KG: dict[str, float] = {
    "coal":    1078.5,   # lignite + subbituminous
    "coke":    1021.2,   # petroleum coke
    "oil":      795.8,
    "ng":       496.3,
    "gas":      496.3,   # alias for NG-based fuels
    "steam":    496.3,   # purchased steam (gas-fired)
    "nuclear":    0.0,
    "wind":       0.0,
    "solar":      0.0,
    "hydro":      0.0,
    "water":      0.0,   # WAT fuel code
    "biomass":   54.0,
    "wood":      54.0,   # WDS fuel code
    "agricultural": 54.0, # AB fuel code
    "storage":    0.0,   # battery / MWH fuel code
    "other":    496.3,   # default to NG proxy
}

def _emission_factors(gen_csv: Path) -> pd.Series:
    """Return per-generator emission factor in kg CO₂/MWh.

    Uses HR_avg_0 × Emissions CO2 Lbs/MMBTU when available (RTS-GMLC style),
    otherwise falls back to fuel-keyword lookup.  Mirrors cas._emission_factors().
    """
    try:
        gen = pd.read_csv(gen_csv)
        gen = gen.set_index("GEN UID")
        has_hr  = "HR_avg_0" in gen.columns
        has_co2 = "Emissions CO2 Lbs/MMBTU" in gen.columns
        if has_hr and has_co2:
            hr      = pd.to_numeric(gen["HR_avg_0"], errors="coerce").fillna(0.0) / 1000.0
            co2_lbs = pd.to_numeric(gen["Emissions CO2 Lbs/MMBTU"], errors="coerce").fillna(0.0)
            return hr * co2_lbs * 0.453592
        # Fallback: fuel-keyword lookup
        fuels = gen["Fuel"]
        return fuels.map(
            lambda f: next((v for k, v in _FALLBACK_KG.items() if k in str(f).lower()), 0.0)
        )
    except Exception:
        return pd.Series(dtype=float)


def _co2_kg(thermal_csv: Path, ef: pd.Series) -> float:
    df = pd.read_csv(thermal_csv)
    df["co2"] = df["Dispatch"] * df["Generator"].map(ef).fillna(0.0)
    return float(df["co2"].sum())


def _dispatch_by_fuel(thermal_csv: Path, gen_csv: Path) -> dict[str, float]:
    try:
        gen  = pd.read_csv(gen_csv)[["GEN UID", "Fuel"]].set_index("GEN UID")
        df   = pd.read_csv(thermal_csv)
        df   = df.join(gen, on="Generator")
        return df.groupby("Fuel")["Dispatch"].sum().to_dict()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Core: build_comparison
# ---------------------------------------------------------------------------

def build_comparison(
    sim_dirs:   Sequence[str | Path],
    sim_labels: Sequence[str],
    gen_csv:    str | Path,
    water_dirs: Sequence[str | Path] | None = None,
) -> pd.DataFrame:
    """
    Build a metrics × simulations DataFrame.

    Rows  = metric names
    Cols  = sim_labels

    Parameters
    ----------
    sim_dirs   : directories containing hourly_summary.csv + thermal_detail.csv
    sim_labels : column labels (first entry is treated as baseline for delta_cols)
    gen_csv    : path to RTS-GMLC gen.csv for fuel look-up
    water_dirs : optional per-sim directories containing system_water_hourly.csv
    """
    gen_csv = Path(gen_csv)
    ef      = _emission_factors(gen_csv)

    records: dict[str, list] = {}

    for i, (sdir, lbl) in enumerate(zip(sim_dirs, sim_labels)):
        sdir = Path(sdir)
        row: dict[str, float] = {}

        # hourly_summary metrics
        hs = sdir / "hourly_summary.csv"
        if hs.exists():
            df = pd.read_csv(hs)
            total_demand = df["Demand"].sum()
            row["total_demand_mwh"]          = float(total_demand)
            row["total_cost_usd"]            = float((df["FixedCosts"] + df["VariableCosts"]).sum())
            row["variable_cost_usd"]         = float(df["VariableCosts"].sum())
            row["fixed_cost_usd"]            = float(df["FixedCosts"].sum())
            row["load_shedding_mwh"]         = float(df["LoadShedding"].sum())
            row["renewables_used_mwh"]       = float(df["RenewablesUsed"].sum())
            row["renewables_curtailed_mwh"]  = float(df["RenewablesCurtailment"].sum())
            row["renewables_available_mwh"]  = float(df["RenewablesAvailable"].sum())
            row["mean_lmp_usd_mwh"]          = float(df["Price"].mean())
            row["lmp_std_usd_mwh"]           = float(df["Price"].std())
            row["peak_lmp_usd_mwh"]          = float(df["Price"].max())
            row["renew_fraction_pct"]        = (
                100.0 * row["renewables_used_mwh"] / total_demand if total_demand > 0 else 0.0
            )

        # thermal_detail CO₂
        td = sdir / "thermal_detail.csv"
        if td.exists():
            co2_kg = _co2_kg(td, ef)
            total_mwh = row.get("total_demand_mwh", 1.0)
            row["total_co2_kg"]          = co2_kg
            row["total_co2_tonnes"]      = co2_kg / 1000.0
            row["mean_ci_kgco2_mwh"]     = co2_kg / total_mwh if total_mwh > 0 else 0.0

            # dispatch by fuel
            for fuel, mwh in _dispatch_by_fuel(td, gen_csv).items():
                row[f"dispatch_{fuel.lower().replace(' ','_')}_mwh"] = float(mwh)

        # water
        wdir = None
        if water_dirs and i < len(water_dirs):
            wdir = Path(water_dirs[i])
        # also check standard location: sdir/../water/<sim_name>/
        if wdir is None:
            candidate = sdir.parent / "water" / sdir.name / "system_water_hourly.csv"
            if candidate.exists():
                wdir = candidate.parent
        if wdir is not None:
            wcsv = wdir / "system_water_hourly.csv"
            if wcsv.exists():
                wdf = pd.read_csv(wcsv)
                row["total_wd_gal"]      = float(wdf["total_wd_gal"].sum())
                row["total_wc_gal"]      = float(wdf["total_wc_gal"].sum())
                total_mwh = row.get("total_demand_mwh", 1.0)
                row["wd_intensity_gal_mwh"] = row["total_wd_gal"] / total_mwh if total_mwh > 0 else 0.0

        records[lbl] = row

    df_out = pd.DataFrame(records)
    return df_out


# ---------------------------------------------------------------------------
# _delta_cols: add Δ and Δ% columns relative to baseline
# ---------------------------------------------------------------------------

def _delta_cols(df: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    """
    Append absolute (Δ) and relative (Δ%) delta columns for every non-baseline
    column, measured against baseline_label.

    Returns a new DataFrame with the same rows and additional Δ / Δ% columns.
    """
    if baseline_label not in df.columns:
        return df.copy()

    out = df.copy()
    bl  = df[baseline_label]

    for col in df.columns:
        if col == baseline_label:
            continue
        delta     = df[col] - bl
        delta_pct = delta / bl.replace(0, np.nan) * 100.0
        out[f"Δ {col}"]   = delta.round(4)
        out[f"Δ% {col}"]  = delta_pct.round(3)

    return out


# ---------------------------------------------------------------------------
# make_plots: time-series and bar summary charts
# ---------------------------------------------------------------------------

def make_plots(
    sim_dirs:       Sequence[str | Path],
    sim_labels:     Sequence[str],
    gen_csv:        str | Path,
    comparison_df:  pd.DataFrame,
    out_dir:        str | Path,
    water_dirs:     Sequence[str | Path] | None = None,
) -> list[Path]:
    """
    Generate comparison figures and write them to out_dir.

    Figures produced:
      system_comparison_bars.png   — key metric bar chart
      dispatch_by_fuel.png         — fuel-mix stacked bars
      lmp_timeseries.png           — hourly LMP overlay
      co2_timeseries.png           — cumulative CO₂ overlay
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colors  = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b",
               "#b07aa1", "#76b7b2", "#ff9da7", "#9c755f"]
    labels  = list(sim_labels)
    generated: list[Path] = []

    # ── 1. Key metric bar chart ───────────────────────────────────────────────
    KEY = ["total_co2_tonnes", "total_cost_usd", "load_shedding_mwh",
           "renewables_used_mwh", "mean_lmp_usd_mwh", "lmp_std_usd_mwh"]
    present = [k for k in KEY if k in comparison_df.index]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(3 * len(present), 4))
        if len(present) == 1:
            axes = [axes]
        for ax, metric in zip(axes, present):
            vals = [comparison_df.loc[metric, lbl] if lbl in comparison_df.columns else 0
                    for lbl in labels]
            ax.bar(range(len(labels)), vals,
                   color=colors[:len(labels)], edgecolor="white")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_title(metric.replace("_", "\n"), fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = out_dir / "system_comparison_bars.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(p)
        print(f"    → {p.name}")

    # ── 2. Fuel dispatch stacked bars ─────────────────────────────────────────
    fuel_rows = [r for r in comparison_df.index
                 if r.startswith("dispatch_") and r.endswith("_mwh")]
    if fuel_rows:
        fuels = [r.replace("dispatch_", "").replace("_mwh", "") for r in fuel_rows]
        vals  = comparison_df.loc[fuel_rows].fillna(0.0)
        fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.5), 5))
        bottoms = np.zeros(len(labels))
        for j, (fuel, row) in enumerate(zip(fuels, fuel_rows)):
            fvals = [vals.loc[row, lbl] if lbl in vals.columns else 0.0 for lbl in labels]
            ax.bar(range(len(labels)), fvals, bottom=bottoms,
                   label=fuel, color=colors[j % len(colors)], edgecolor="white", lw=0.4)
            bottoms += np.array(fvals)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Dispatch (MWh)")
        ax.set_title("Dispatch by fuel")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        fig.tight_layout()
        p = out_dir / "dispatch_by_fuel.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(p)
        print(f"    → {p.name}")

    # ── 3. LMP time-series overlay ────────────────────────────────────────────
    lmp_frames = []
    for sdir, lbl in zip(sim_dirs, sim_labels):
        hs = Path(sdir) / "hourly_summary.csv"
        if hs.exists():
            df = pd.read_csv(hs)
            df["label"] = lbl
            lmp_frames.append(df[["Date", "Hour", "Price", "label"]])
    if lmp_frames:
        all_lmp = pd.concat(lmp_frames, ignore_index=True)
        all_lmp["t"] = all_lmp.groupby("label").cumcount()
        fig, ax = plt.subplots(figsize=(12, 3.5))
        for j, lbl in enumerate(labels):
            sub = all_lmp[all_lmp["label"] == lbl]
            ax.plot(sub["t"].values, sub["Price"].values,
                    color=colors[j % len(colors)], alpha=0.8,
                    lw=0.8, label=lbl)
        ax.set_xlabel("Hour"); ax.set_ylabel("LMP ($/MWh)")
        ax.set_title("Hourly LMP — all simulation runs")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / "lmp_timeseries.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(p)
        print(f"    → {p.name}")

    # ── 4. Cumulative CO₂ overlay ─────────────────────────────────────────────
    gen_csv = Path(gen_csv)
    ef = _emission_factors(gen_csv)
    co2_frames = []
    for sdir, lbl in zip(sim_dirs, sim_labels):
        td = Path(sdir) / "thermal_detail.csv"
        if td.exists():
            df = pd.read_csv(td)
            df["co2_kg"] = df["Dispatch"] * df["Generator"].map(ef).fillna(0.0)
            hourly = df.groupby(["Date", "Hour"])["co2_kg"].sum().reset_index()
            hourly["cumco2_t"] = hourly["co2_kg"].cumsum() / 1000.0
            hourly["t"]  = range(len(hourly))
            hourly["label"] = lbl
            co2_frames.append(hourly[["t", "cumco2_t", "label"]])
    if co2_frames:
        all_co2 = pd.concat(co2_frames, ignore_index=True)
        fig, ax  = plt.subplots(figsize=(12, 3.5))
        for j, lbl in enumerate(labels):
            sub = all_co2[all_co2["label"] == lbl]
            ax.plot(sub["t"].values, sub["cumco2_t"].values,
                    color=colors[j % len(colors)], lw=1.4, label=lbl)
        ax.set_xlabel("Hour"); ax.set_ylabel("Cumulative CO₂ (tonnes)")
        ax.set_title("Cumulative CO₂ — all simulation runs")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / "co2_timeseries.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(p)
        print(f"    → {p.name}")

    return generated
