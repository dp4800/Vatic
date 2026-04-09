#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
run_outputs.py — Per-week thesis figure outputs (14-day / 336-hour simulations).

Produces for one simulation week directory (e.g. outputs/2020-05-04/):
  figures/fig1_1_nodc_vs_dc.png
  figures/fig1_3_seasonal_dispatch.png
  figures/fig1_4_seasonal_water.png
  figures/fig2_2_seasonal_dispatch_battery.png
  figures/fig2_3_co2_water_panel.png
  figures/fig2_4_lmp_cost_panel.png
  figures/fig2_5_co2_hod_all_scenarios.png
  figures/fig3_3_fuel_mix_by_scenario.png
  figures/fig3_4_co2_delta.png
  figures/fig3_5_battery_panel.png      (only if Battery dispatch > 0)
  figures/fig3_6_water_by_scenario.png
  weekly_deltas_{date}.csv
  weekly_summary_{date}.csv
  diagnostics_{date}.json

Usage:
    python scripts/run_outputs.py outputs/2020-05-04 [--force]
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Logger ─────────────────────────────────────────────────────────────────────
log = logging.getLogger("run_outputs")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("  [run_outputs] %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

# ── Constants ──────────────────────────────────────────────────────────────────

SCENARIOS = ["baseline", "sim-gm", "sim-247", "sim-lp"]

SCENARIO_COLORS = {
    'baseline': 'steelblue',
    'sim-gm':   'darkorange',
    'sim-247':  'green',
    'sim-lp':   'crimson',
}

SCENARIO_LABELS = {
    'baseline': 'Baseline',
    'sim-gm':   'Grid-mix',
    'sim-247':  '24/7',
    'sim-lp':   'LP',
}

# Fuel display order (bottom to top in stacked areas)
FUEL_ORDER = [
    'Coal', 'Pet Coke', 'Oil',
    'Gas CC', 'Gas CT', 'Gas ST', 'Other Gas',
    'Nuclear', 'Hydro', 'Biomass',
    'Wind', 'Solar', 'Battery',
]

FUEL_COLORS = {
    'Coal':      'dimgray',
    'Pet Coke':  'slategray',
    'Oil':       'saddlebrown',
    'Gas CC':    'peru',
    'Gas CT':    'wheat',
    'Gas ST':    'burlywood',
    'Other Gas': 'tan',
    'Nuclear':   'steelblue',
    'Hydro':     'royalblue',
    'Biomass':   'olivedrab',
    'Wind':      'mediumseagreen',
    'Solar':     'gold',
    'Battery':   'mediumpurple',
}

# CO2 emission factors (tCO2/MWh) — thesis Table AI
CO2_FACTORS = {
    'Coal':      1.0785,
    'Pet Coke':  1.0212,
    'Oil':       0.7958,
    'Gas CC':    0.4963,
    'Gas CT':    0.4963,
    'Gas ST':    0.4963,
    'Other Gas': 0.4963,
    'Nuclear':   0.0,
    'Hydro':     0.0,
    'Biomass':   0.054,
    'Wind':      0.0,
    'Solar':     0.0,
    'Battery':   0.0,
}

# DC bus names (hardcoded fallback)
DC_BUS_NAMES = [
    'Abel', 'Adler', 'Attar', 'Attlee',
    'Bach', 'Balzac', 'Beethoven',
    'Cabell', 'Caesar', 'Clark',
]

# Season definitions
SEASON_MONTHS = {
    'Winter': (12, 1, 2),
    'Spring': (3, 4, 5),
    'Summer': (6, 7, 8),
    'Fall':   (9, 10, 11),
}
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Fall']
SEASON_TITLES = {
    'Winter': 'Winter (Dec–Feb)',
    'Spring': 'Spring (Mar–May)',
    'Summer': 'Summer (Jun–Aug)',
    'Fall':   'Fall (Sep–Nov)',
}


# ── Fuel parser ────────────────────────────────────────────────────────────────

def _gen_fuel(gen_name: str) -> str:
    """Parse generator name to canonical fuel label.

    Handles both RTS-GMLC naming (e.g. '101_STEAM_1', '123_CC_1')
    and Texas-7k naming (e.g. '298_ConventionalSteamCoal_1',
    '10554_NaturalGasFiredCombinedCycle_ST1').
    """
    parts = gen_name.split('_')
    for p in parts:
        u = p.upper()
        # ── RTS-GMLC short tokens (exact match) ──────────────────────────────
        if u == 'STEAM':         return 'Coal'
        if u == 'CC':            return 'Gas CC'
        if u == 'CT':            return 'Gas CT'
        if u == 'NUCLEAR':       return 'Nuclear'
        if u in ('HYDRO', 'HY'): return 'Hydro'
        if u == 'WIND':          return 'Wind'
        if u in ('PV', 'RTPV', 'SOLAR'): return 'Solar'
        if u == 'STORAGE':       return 'Battery'
        # ── Texas-7k long camelCase tokens (substring match) ─────────────────
        if 'STEAMCOAL' in u or 'CONVENTIONALSTEAM' in u: return 'Coal'
        if 'COMBINEDCYCLE' in u:                          return 'Gas CC'
        if 'COMBUSTIONTURBINE' in u or 'INTERNALCOMBUSTION' in u: return 'Gas CT'
        if 'NATURALGAS' in u and 'STEAM' in u:            return 'Gas ST'
        if 'PETROLEUM' in u or 'PETCOKE' in u:            return 'Pet Coke'
        if 'BIOMASS' in u or 'WOODWASTE' in u or 'WOOD' in u: return 'Biomass'
        if 'OTHERGASES' in u or 'OTHERGAS' in u:          return 'Other Gas'
        if 'WIND' in u:          return 'Wind'
        if 'SOLAR' in u or 'PHOTOVOLTAIC' in u: return 'Solar'
        if 'HYDRO' in u:         return 'Hydro'
        if 'NUCLEAR' in u:       return 'Nuclear'
        if 'BATTER' in u:        return 'Battery'
    return 'Other'


# ── Data loaders ───────────────────────────────────────────────────────────────

def _parse_dt(date_col: pd.Series, hour_col: pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(
        date_col.astype(str) + " " + hour_col.astype(int).astype(str) + ":00",
        format="%Y-%m-%d %H:%M",
    )


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log.warning("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def _load_hourly(sim_dir: Path) -> pd.DataFrame:
    df = _load_csv(sim_dir / "hourly_summary.csv")
    if df.empty:
        return df
    df.index = _parse_dt(df["Date"], df["Hour"])
    return df


def _load_thermal(sim_dir: Path) -> pd.DataFrame:
    return _load_csv(sim_dir / "thermal_detail.csv")


def _load_renew(sim_dir: Path) -> pd.DataFrame:
    return _load_csv(sim_dir / "renew_detail.csv")


def _load_bus(sim_dir: Path) -> pd.DataFrame:
    return _load_csv(sim_dir / "bus_detail.csv")


def _load_storage(sim_dir: Path) -> pd.DataFrame:
    return _load_csv(sim_dir / "storage_detail.csv")


def _fuel_dispatch_hourly(
    thermal_df: pd.DataFrame,
    renew_df: pd.DataFrame,
    n_hours: int,
    hours_index: pd.DatetimeIndex,
    storage_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return (n_hours × len(FUEL_ORDER)) MW dispatch DataFrame.

    Battery discharge (from storage_detail.csv) is included in the 'Battery'
    column.  Only discharge is shown as generation; charging is load and is
    reflected in the demand line, not subtracted from the generation stack.
    """
    result = pd.DataFrame(0.0, index=range(n_hours), columns=FUEL_ORDER)
    # Build hour-index mapping
    dt_to_idx = {dt: i for i, dt in enumerate(hours_index)}

    if not thermal_df.empty:
        th = thermal_df.copy()
        th["_dt"] = _parse_dt(th["Date"], th["Hour"])
        th["_fuel"] = th["Generator"].apply(_gen_fuel)
        th = th[th["_fuel"].isin(FUEL_ORDER)]
        for (dt, fuel), grp in th.groupby(["_dt", "_fuel"]):
            idx = dt_to_idx.get(dt)
            if idx is not None:
                result.at[idx, fuel] = result.at[idx, fuel] + grp["Dispatch"].sum()

    if not renew_df.empty:
        rn = renew_df.copy()
        rn["_dt"] = _parse_dt(rn["Date"], rn["Hour"])
        rn["_fuel"] = rn["Generator"].apply(_gen_fuel)
        rn = rn[rn["_fuel"].isin(FUEL_ORDER)]
        for (dt, fuel), grp in rn.groupby(["_dt", "_fuel"]):
            idx = dt_to_idx.get(dt)
            if idx is not None:
                result.at[idx, fuel] = result.at[idx, fuel] + grp["Output"].sum()

    if storage_df is not None and not storage_df.empty and "Discharge" in storage_df.columns:
        st = storage_df.copy()
        st["_dt"] = _parse_dt(st["Date"], st["Hour"])
        for dt, grp in st.groupby("_dt"):
            idx = dt_to_idx.get(dt)
            if idx is not None:
                result.at[idx, "Battery"] = result.at[idx, "Battery"] + grp["Discharge"].sum()

    return result


def _hourly_co2(fuel_mw: pd.DataFrame) -> pd.Series:
    """Compute hourly CO2 (tCO2/h) from fuel_mw dispatch DataFrame."""
    co2 = pd.Series(0.0, index=fuel_mw.index)
    for fuel, factor in CO2_FACTORS.items():
        if fuel in fuel_mw.columns and factor > 0:
            co2 += fuel_mw[fuel] * factor
    return co2


def _load_water_system(water_dir: Path, scenario: str) -> pd.DataFrame:
    """Load system_water_hourly.csv for a scenario; returns empty df on missing."""
    p = water_dir / scenario / "system_water_hourly.csv"
    return _load_csv(p)


def _lmp_hourly(bus_df: pd.DataFrame, hours_index: pd.DatetimeIndex) -> pd.Series:
    """System-average LMP per hour from bus_detail.csv."""
    out = pd.Series(np.nan, index=range(len(hours_index)))
    if bus_df.empty or "LMP" not in bus_df.columns:
        return out
    bd = bus_df.copy()
    bd["_dt"] = _parse_dt(bd["Date"], bd["Hour"])
    dt_to_idx = {dt: i for i, dt in enumerate(hours_index)}
    for dt, grp in bd.groupby("_dt"):
        idx = dt_to_idx.get(dt)
        if idx is not None:
            out.iloc[idx] = grp["LMP"].mean()
    return out


# ── HOD helper ─────────────────────────────────────────────────────────────────

def _hod_mean(series_or_df, n_hours: int = 336):
    """Return hour-of-day (0-23) mean from a Series or DataFrame indexed 0..n_hours-1."""
    if isinstance(series_or_df, pd.Series):
        tmp = series_or_df.copy()
        tmp.index = tmp.index % 24
        return tmp.groupby(level=0).mean()
    else:
        tmp = series_or_df.copy()
        tmp.index = tmp.index % 24
        return tmp.groupby(level=0).mean()


# ── x-axis helpers ─────────────────────────────────────────────────────────────

def _day_xticks(ax: plt.Axes, n_hours: int = 336) -> None:
    """Set x-ticks at every 24 hours with Day N labels."""
    ticks = list(range(0, n_hours, 24))
    labels = [f"Day {i + 1}" for i in range(len(ticks))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Hour")


def _hod_xticks(ax: plt.Axes) -> None:
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of Day")


# ── Figure save helper ─────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{stem}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("→ %s", path)


# ── Master data loader ──────────────────────────────────────────────────────────

def _load_all(week_dir: Path) -> dict:
    """
    Load all per-scenario data needed for all figures.

    Returns dict keyed by scenario name, each value a sub-dict with:
        hourly, thermal, renew, bus, storage, fuel_mw, co2_h, lmp_h, water_df
    Plus top-level:
        hours_index  (DatetimeIndex, length = n_hours)
        n_hours      (int)

    Also loads 'baseline-nodc' if it exists.
    """
    water_dir = week_dir / "water"

    # Determine time axis from baseline hourly_summary
    baseline_hourly = _load_hourly(week_dir / "baseline")
    if baseline_hourly.empty:
        # Try first available scenario
        for scen in SCENARIOS:
            baseline_hourly = _load_hourly(week_dir / scen)
            if not baseline_hourly.empty:
                break

    hours_index = baseline_hourly.index if not baseline_hourly.empty else pd.DatetimeIndex([])
    n_hours = len(hours_index)

    result: dict = {"hours_index": hours_index, "n_hours": n_hours}

    # Load all standard scenarios
    all_keys = list(SCENARIOS) + ["baseline-nodc"]
    for scen in all_keys:
        sim_dir = week_dir / scen
        if not sim_dir.exists():
            continue
        hourly = _load_hourly(sim_dir)
        if hourly.empty:
            continue
        thermal = _load_thermal(sim_dir)
        renew   = _load_renew(sim_dir)
        bus     = _load_bus(sim_dir)
        storage = _load_storage(sim_dir)

        fuel_mw = _fuel_dispatch_hourly(thermal, renew, n_hours, hours_index, storage)
        co2_h   = _hourly_co2(fuel_mw)
        lmp_h   = _lmp_hourly(bus, hours_index)
        # For baseline-nodc, water lives in water/baseline-nodc if present, else skip
        water_df = _load_water_system(water_dir, scen)

        result[scen] = {
            "hourly":   hourly,
            "thermal":  thermal,
            "renew":    renew,
            "bus":      bus,
            "storage":  storage,
            "fuel_mw":  fuel_mw,    # DataFrame, index 0..n_hours-1
            "co2_h":    co2_h,      # Series, index 0..n_hours-1
            "lmp_h":    lmp_h,      # Series, index 0..n_hours-1
            "water_df": water_df,   # raw CSV df (may be empty)
        }

    return result


# ── Water conversion helpers ───────────────────────────────────────────────────

def _water_series_mgal(water_df: pd.DataFrame, col: str, hours_index: pd.DatetimeIndex) -> pd.Series:
    """Return hourly Mgal series (index 0..n-1) from system_water_hourly.csv."""
    n = len(hours_index)
    out = pd.Series(np.nan, index=range(n))
    if water_df.empty or col not in water_df.columns:
        return out
    df = water_df.copy()
    df["_dt"] = _parse_dt(df["Date"], df["Hour"])
    dt_to_idx = {dt: i for i, dt in enumerate(hours_index)}
    for dt, grp in df.groupby("_dt"):
        idx = dt_to_idx.get(dt)
        if idx is not None:
            out.iloc[idx] = grp[col].sum() / 1e6  # gal → Mgal
    return out


# ── Stacked area helper ────────────────────────────────────────────────────────

def _stacked_area_hod(ax: plt.Axes, fuel_mw: pd.DataFrame) -> None:
    """Draw stacked area HOD dispatch on ax. Returns nothing; modifies ax in place."""
    hod_fw = _hod_mean(fuel_mw)
    x = np.arange(24)
    bottom = np.zeros(24)
    for fuel in FUEL_ORDER:
        if fuel not in hod_fw.columns:
            continue
        vals = hod_fw[fuel].fillna(0.0).values
        if vals.sum() == 0:
            continue
        ax.fill_between(x, bottom, bottom + vals,
                        color=FUEL_COLORS[fuel],
                        label=fuel, alpha=0.92)
        bottom += vals
    _hod_xticks(ax)
    ax.set_xlim(0, 23)


def _pending_placeholder(ax: plt.Axes) -> None:
    """Draw a grey 'Data pending' placeholder on ax."""
    ax.set_facecolor("#f0f0f0")
    ax.text(0.5, 0.5, "Data pending — run annual",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="#888888", style="italic")
    ax.set_xticks([])
    ax.set_yticks([])


def _infer_season(date_str: str) -> str:
    month = int(date_str[5:7])
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


# ── Fig 1.1 — No-DC vs DC baseline HOD dispatch ────────────────────────────────

def _fig1_1(data: dict, fig_dir: Path, week_date: str) -> None:
    stem = "fig1_1_nodc_vs_dc"
    if "baseline" not in data:
        log.warning("fig1_1: no baseline data, skipping")
        return
    if "baseline-nodc" not in data:
        log.warning("fig1_1: no baseline-nodc data, skipping")
        return

    hours_index = data["hours_index"]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    # --- Left panel: no-DC HOD dispatch ---
    nodc_fuel = data["baseline-nodc"]["fuel_mw"]
    nodc_hourly = data["baseline-nodc"]["hourly"]
    _stacked_area_hod(ax_left, nodc_fuel)
    # Demand line (HOD average)
    if "Demand" in nodc_hourly.columns:
        hod_demand = _hod_mean(nodc_hourly["Demand"].reset_index(drop=True))
        ax_left.plot(np.arange(24), hod_demand.values, color="black",
                     lw=1.4, ls="--", label="Demand", zorder=5)
    ax_left.set_ylabel("MW (avg)")
    ax_left.set_title("No Data Centers")

    # --- Right panel: DC baseline HOD dispatch ---
    dc_fuel = data["baseline"]["fuel_mw"]
    dc_hourly = data["baseline"]["hourly"]
    _stacked_area_hod(ax_right, dc_fuel)
    if "Demand" in dc_hourly.columns:
        hod_demand_dc = _hod_mean(dc_hourly["Demand"].reset_index(drop=True))
        ax_right.plot(np.arange(24), hod_demand_dc.values, color="black",
                      lw=1.4, ls="--", label="Demand", zorder=5)
    ax_right.set_ylabel("MW (avg)")
    ax_right.set_title("With Data Centers")

    # Share same y-max across both panels
    ymax = max(ax_left.get_ylim()[1], ax_right.get_ylim()[1])
    ax_left.set_ylim(bottom=0, top=ymax)
    ax_right.set_ylim(bottom=0, top=ymax)

    # Legend on right panel only, upper left, 2 cols
    handles, labels = ax_right.get_legend_handles_labels()
    ax_right.legend(handles, labels, loc="upper left", ncol=2, fontsize=8)

    fig.suptitle(f"HOD avg dispatch by fuel — {week_date}", fontsize=10)
    plt.tight_layout()
    _save_fig(fig, fig_dir, stem)


# ── Fig 1.3 — Seasonal dispatch (HOD, baseline) ────────────────────────────────

def _fig1_3(data: dict, fig_dir: Path, week_date: str) -> None:
    stem = "fig1_3_seasonal_dispatch"
    if "baseline" not in data:
        log.warning("fig1_3: no baseline data, skipping")
        return

    current_season = _infer_season(week_date)
    fuel_mw = data["baseline"]["fuel_mw"]
    hod_fw = _hod_mean(fuel_mw)

    # Compute ymax from available data
    ymax = 0.0
    for fuel in FUEL_ORDER:
        if fuel in hod_fw.columns:
            ymax += hod_fw[fuel].fillna(0.0).max()
    ymax = max(ymax * 1.05, 1.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    x = np.arange(24)

    for ax, season in zip(axes, SEASON_ORDER):
        ax.set_title(SEASON_TITLES[season])
        if season == current_season:
            # Draw actual stacked area
            _stacked_area_hod(ax, fuel_mw)
            ax.set_ylim(bottom=0, top=ymax)
            ax.set_ylabel("MW (avg)")
        else:
            _pending_placeholder(ax)
            ax.set_ylim(bottom=0, top=ymax)

    # Collect legend from a plotted axis
    for ax, season in zip(axes, SEASON_ORDER):
        if season == current_season:
            handles, labels = ax.get_legend_handles_labels()
            break
    else:
        handles, labels = [], []

    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(FUEL_ORDER),
                   fontsize=7, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Hour-of-day avg dispatch by fuel — seasonal", fontsize=10)
    plt.tight_layout()
    _save_fig(fig, fig_dir, stem)


# ── Fig 1.4 — Seasonal water (HOD, baseline) ──────────────────────────────────

def _fig1_4(data: dict, fig_dir: Path, week_date: str) -> None:
    stem = "fig1_4_seasonal_water"
    if "baseline" not in data:
        log.warning("fig1_4: no baseline data, skipping")
        return

    current_season = _infer_season(week_date)
    hours_index = data["hours_index"]
    water_df = data["baseline"]["water_df"]
    wd = _water_series_mgal(water_df, "total_wd_gal", hours_index)
    wc = _water_series_mgal(water_df, "total_wc_gal", hours_index)

    if wd.isna().all() and wc.isna().all():
        log.warning("fig1_4: no water data for baseline, skipping")
        return

    hod_wd = _hod_mean(wd.fillna(0.0))
    hod_wc = _hod_mean(wc.fillna(0.0))
    ymax = max(hod_wd.max(), hod_wc.max()) * 1.1
    ymax = max(ymax, 1e-6)

    x = np.arange(24)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    for ax, season in zip(axes, SEASON_ORDER):
        ax.set_title(SEASON_TITLES[season])
        if season == current_season:
            ax.plot(x, hod_wd.values, color=SCENARIO_COLORS["baseline"],
                    ls="-", lw=1.4, label="Withdrawal")
            ax.plot(x, hod_wc.values, color=SCENARIO_COLORS["baseline"],
                    ls="--", lw=1.2, label="Consumption")
            _hod_xticks(ax)
            ax.set_xlim(0, 23)
            ax.set_ylim(bottom=0, top=ymax)
            ax.set_ylabel("Mgal/h")
            ax.legend(fontsize=8)
        else:
            _pending_placeholder(ax)
            ax.set_ylim(bottom=0, top=ymax)

    fig.suptitle(f"HOD avg water use (baseline) — seasonal — {week_date}", fontsize=10)
    fig.text(0.5, -0.04,
             "Withdrawal = total volume pulled from source.  "
             "Consumption = fraction that does not return.",
             ha="center", fontsize=8, color="#555555")
    plt.tight_layout()
    _save_fig(fig, fig_dir, stem)


# ── Fig 2.2 — Seasonal dispatch with battery (sim-lp) ─────────────────────────

def _fig2_2(data: dict, fig_dir: Path, week_date: str) -> None:
    stem = "fig2_2_seasonal_dispatch_battery"
    scen = "sim-lp"
    if scen not in data:
        log.warning("fig2_2: no sim-lp data, skipping")
        return

    current_season = _infer_season(week_date)
    fuel_mw = data[scen]["fuel_mw"]
    hod_fw = _hod_mean(fuel_mw)

    ymax = 0.0
    for fuel in FUEL_ORDER:
        if fuel in hod_fw.columns:
            ymax += hod_fw[fuel].fillna(0.0).max()
    ymax = max(ymax * 1.05, 1.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)

    for ax, season in zip(axes, SEASON_ORDER):
        ax.set_title(SEASON_TITLES[season])
        if season == current_season:
            _stacked_area_hod(ax, fuel_mw)
            ax.set_ylim(bottom=0, top=ymax)
            ax.set_ylabel("MW (avg)")
            ax.annotate(
                "Battery dispatch is VATIC-managed,\nindependent of CAS",
                xy=(0.02, 0.97), xycoords="axes fraction",
                fontsize=6.5, va="top", color="#6a0dad",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#9b5de5", alpha=0.85),
            )
        else:
            _pending_placeholder(ax)
            ax.set_ylim(bottom=0, top=ymax)

    for ax, season in zip(axes, SEASON_ORDER):
        if season == current_season:
            handles, labels = ax.get_legend_handles_labels()
            break
    else:
        handles, labels = [], []

    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(FUEL_ORDER),
                   fontsize=7, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(f"HOD avg dispatch (LP scenario, incl. Battery) — seasonal — {week_date}",
                 fontsize=10)
    plt.tight_layout()
    _save_fig(fig, fig_dir, stem)


# ── Fig 2.3 — CO2 + water panel ────────────────────────────────────────────────

def _fig2_3(data: dict, fig_dir: Path, week_date: str) -> None:
    stem = "fig2_3_co2_water_panel"
    n_hours = data["n_hours"]
    hours_index = data["hours_index"]
    x = np.arange(n_hours)
    hod_x = np.arange(24)
    season = _infer_season(week_date)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax_co2_ts, ax_co2_hod, ax_wd_ts, ax_wd_hod = axes.flatten()

    co2_all: dict[str, pd.Series] = {}
    wd_all: dict[str, pd.Series] = {}

    for scen in SCENARIOS:
        if scen not in data:
            continue
        co2_all[scen] = data[scen]["co2_h"]
        wd_all[scen] = _water_series_mgal(data[scen]["water_df"], "total_wd_gal", hours_index)

    # Shared y-max for CO2 row
    co2_ts_max = max(
        (s.max() for s in co2_all.values() if not s.isna().all()),
        default=1.0,
    ) * 1.1
    co2_hod_max = max(
        (_hod_mean(s.fillna(0.0)).max() for s in co2_all.values() if not s.isna().all()),
        default=1.0,
    ) * 1.1
    co2_ymax = max(co2_ts_max, co2_hod_max)

    # Shared y-max for water row
    wd_ts_max = max(
        (s.max() for s in wd_all.values() if not s.isna().all()),
        default=1.0,
    ) * 1.1
    wd_hod_max = max(
        (_hod_mean(s.fillna(0.0)).max() for s in wd_all.values() if not s.isna().all()),
        default=1.0,
    ) * 1.1
    wd_ymax = max(wd_ts_max, wd_hod_max)

    # Top-left: CO2 timeseries
    for scen in SCENARIOS:
        if scen not in co2_all:
            continue
        ax_co2_ts.plot(x, co2_all[scen].values, color=SCENARIO_COLORS[scen],
                       lw=1.2, label=SCENARIO_LABELS[scen])
    ax_co2_ts.set_ylim(bottom=0, top=co2_ymax)
    ax_co2_ts.set_ylabel("tCO2/h")
    ax_co2_ts.set_title("Hourly CO2 emissions")
    _day_xticks(ax_co2_ts, n_hours)

    # Top-right: CO2 HOD
    for scen in SCENARIOS:
        if scen not in co2_all:
            continue
        hod = _hod_mean(co2_all[scen].fillna(0.0))
        ax_co2_hod.plot(hod_x, hod.values, color=SCENARIO_COLORS[scen],
                        lw=1.2, label=SCENARIO_LABELS[scen])
    ax_co2_hod.set_ylim(bottom=0, top=co2_ymax)
    ax_co2_hod.set_ylabel("tCO2/h")
    ax_co2_hod.set_title("HOD avg CO2 emissions")
    _hod_xticks(ax_co2_hod)
    ax_co2_hod.set_xlim(0, 23)

    # Bottom-left: water withdrawal timeseries
    for scen in SCENARIOS:
        if scen not in wd_all:
            continue
        ax_wd_ts.plot(x, wd_all[scen].values, color=SCENARIO_COLORS[scen],
                      lw=1.2, label=SCENARIO_LABELS[scen])
    ax_wd_ts.set_ylim(bottom=0, top=wd_ymax if wd_ymax > 0 else 1.0)
    ax_wd_ts.set_ylabel("Mgal/h")
    ax_wd_ts.set_title("Hourly water withdrawal")
    _day_xticks(ax_wd_ts, n_hours)

    # Bottom-right: water withdrawal HOD
    for scen in SCENARIOS:
        if scen not in wd_all:
            continue
        hod = _hod_mean(wd_all[scen].fillna(0.0))
        ax_wd_hod.plot(hod_x, hod.values, color=SCENARIO_COLORS[scen],
                       lw=1.2, label=SCENARIO_LABELS[scen])
    ax_wd_hod.set_ylim(bottom=0, top=wd_ymax if wd_ymax > 0 else 1.0)
    ax_wd_hod.set_ylabel("Mgal/h")
    ax_wd_hod.set_title("HOD avg water withdrawal")
    _hod_xticks(ax_wd_hod)
    ax_wd_hod.set_xlim(0, 23)

    # Shared legend above all panels
    handles, labels = ax_co2_ts.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(SCENARIOS),
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"CO2 and water: all scenarios — {week_date} ({season})",
        y=1.06, fontsize=10,
    )
    plt.tight_layout()
    _save_fig(fig, fig_dir, stem)


# ── Fig 2.4 — LMP + cost panel ─────────────────────────────────────────────────

def _fig2_4(data: dict, fig_dir: Path, week_date: str) -> None:
    stem = "fig2_4_lmp_cost_panel"
    n_hours = data["n_hours"]
    x = np.arange(n_hours)

    fig, (ax_lmp, ax_cost) = plt.subplots(1, 2, figsize=(12, 4))

    lmp_ymax = 0.0
    cost_ymax = 0.0

    for scen in SCENARIOS:
        if scen not in data:
            continue
        lmp_h = data[scen]["lmp_h"]
        ax_lmp.plot(x, lmp_h.values, color=SCENARIO_COLORS[scen],
                    lw=1.2, label=SCENARIO_LABELS[scen])
        valid_lmp = lmp_h.dropna()
        if not valid_lmp.empty:
            lmp_ymax = max(lmp_ymax, valid_lmp.max())

        hourly = data[scen]["hourly"]
        if "VariableCosts" in hourly.columns and "FixedCosts" in hourly.columns:
            op_cost = (hourly["VariableCosts"] + hourly["FixedCosts"]).values
        elif "VariableCosts" in hourly.columns:
            op_cost = hourly["VariableCosts"].values
        else:
            op_cost = np.zeros(n_hours)
        ax_cost.plot(x, op_cost, color=SCENARIO_COLORS[scen],
                     lw=1.2, label=SCENARIO_LABELS[scen])
        cost_ymax = max(cost_ymax, op_cost.max())

    # Shared y-max for both panels
    shared_lmp_top = min(lmp_ymax * 1.1, 500) if lmp_ymax > 0 else 500
    shared_cost_top = cost_ymax * 1.1 if cost_ymax > 0 else 1.0

    ax_lmp.set_ylim(bottom=0, top=shared_lmp_top)
    ax_lmp.set_ylabel("$/MWh")
    ax_lmp.set_title("System-avg LMP")
    ax_lmp.legend(loc="upper right", fontsize=8)
    _day_xticks(ax_lmp, n_hours)

    ax_cost.set_ylim(bottom=0, top=shared_cost_top)
    ax_cost.set_ylabel("$000/h")
    ax_cost.set_title("Hourly operational cost (Variable + Fixed)")
    ax_cost.legend(loc="upper right", fontsize=8)
    _day_xticks(ax_cost, n_hours)

    fig.suptitle(f"LMP and operational cost — {week_date}", fontsize=10)
    plt.tight_layout()
    _save_fig(fig, fig_dir, stem)


# ── Fig 2.5 — HOD CO2 all scenarios ───────────────────────────────────────────

def _fig2_5(data: dict, fig_dir: Path, week_date: str) -> None:
    stem = "fig2_5_co2_hod_all_scenarios"
    x = np.arange(24)

    fig, ax = plt.subplots(figsize=(8, 4))
    for scen in SCENARIOS:
        if scen not in data:
            continue
        hod = _hod_mean(data[scen]["co2_h"].fillna(0.0))
        ax.plot(x, hod.values, color=SCENARIO_COLORS[scen],
                lw=1.2, label=SCENARIO_LABELS[scen])

    ax.set_ylabel("tCO2/h")
    ax.set_title(f"HOD avg CO2 by scenario — {week_date}")
    ax.legend(fontsize=8)
    _hod_xticks(ax)
    ax.set_xlim(0, 23)
    ax.text(0.5, -0.18,
            "(Box plot across all weeks shown in annual summary)",
            ha="center", transform=ax.transAxes, fontsize=8, color="#666666")
    plt.tight_layout()
    _save_fig(fig, fig_dir, stem)


# ── Fig 3.3 — Fuel-mix by scenario ────────────────────────────────────────────

def _fig3_3(data: dict, fig_dir: Path, week_date: str) -> None:
    stem = "fig3_3_fuel_mix_by_scenario"
    present_scenarios = [s for s in SCENARIOS if s in data]
    if not present_scenarios:
        log.warning("fig3_3: no scenario data, skipping")
        return

    # Determine active fuels (>0 in any scenario)
    active_fuels = []
    for fuel in FUEL_ORDER:
        total = sum(
            data[s]["fuel_mw"][fuel].sum()
            for s in present_scenarios
            if fuel in data[s]["fuel_mw"].columns
        )
        if total > 0:
            active_fuels.append(fuel)

    if not active_fuels:
        log.warning("fig3_3: no fuel dispatch found, skipping")
        return

    n_fuels = len(active_fuels)
    n_scen  = len(present_scenarios)
    bar_w   = 0.8 / n_scen
    x       = np.arange(n_fuels)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, scen in enumerate(present_scenarios):
        vals = [data[scen]["fuel_mw"][f].sum() if f in data[scen]["fuel_mw"].columns else 0.0
                for f in active_fuels]
        offset = (i - (n_scen - 1) / 2.0) * bar_w
        ax.bar(x + offset, vals, width=bar_w,
               color=SCENARIO_COLORS[scen], label=SCENARIO_LABELS[scen],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(active_fuels, rotation=30, ha="right")
    ax.set_ylabel("MWh (week total)")
    ax.set_title(f"Fuel-mix by scenario — {week_date}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_fig(fig, fig_dir, stem)


# ── Fig 3.4 — CO2 delta bar ────────────────────────────────────────────────────

def _fig3_4(data: dict, fig_dir: Path, week_date: str) -> None:
    stem = "fig3_4_co2_delta"
    if "baseline" not in data:
        log.warning("fig3_4: no baseline data, skipping")
        return

    baseline_co2 = data["baseline"]["co2_h"].sum()
    non_baseline = [s for s in SCENARIOS if s != "baseline" and s in data]
    if not non_baseline:
        log.warning("fig3_4: no non-baseline scenarios, skipping")
        return

    deltas = [data[s]["co2_h"].sum() - baseline_co2 for s in non_baseline]
    x = np.arange(len(non_baseline))

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, deltas, color=[SCENARIO_COLORS[s] for s in non_baseline],
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8)

    # Annotate bars with value
    ylim_range = max(abs(d) for d in deltas) * 1.3 if deltas else 1.0
    ax.set_ylim(-ylim_range * 1.15, ylim_range * 1.15)
    for bar, val in zip(bars, deltas):
        sign = "+" if val >= 0 else "\u2212"
        label = f"{sign}{abs(val):,.0f} t"
        va = "bottom" if val >= 0 else "top"
        yoff = ylim_range * 0.03
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (yoff if val >= 0 else -yoff),
                label, ha="center", va=va, fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in non_baseline])
    ax.set_ylabel("\u0394tCO2 vs baseline")
    ax.set_title(f"CO2 delta vs baseline — {week_date}")
    plt.tight_layout()
    _save_fig(fig, fig_dir, stem)


# ── Fig 3.5 — Battery panel ────────────────────────────────────────────────────

def _fig3_5(data: dict, fig_dir: Path, week_date: str) -> None:
    stem = "fig3_5_battery_panel"
    scen = "sim-lp"
    if scen not in data:
        log.warning("fig3_5: no sim-lp data, skipping")
        return

    fuel_mw = data[scen]["fuel_mw"]
    if "Battery" not in fuel_mw.columns or fuel_mw["Battery"].sum() == 0:
        log.warning("fig3_5: Battery dispatch is all zero — skipping battery panel")
        return

    n_hours = data["n_hours"]
    hours_index = data["hours_index"]
    x = np.arange(n_hours)
    hod_x = np.arange(24)

    battery_ts = fuel_mw["Battery"].fillna(0.0)
    hod_battery = _hod_mean(battery_ts)

    # Curtailment from renew_detail.csv
    renew_df = data[scen]["renew"]
    curt_ts = pd.Series(0.0, index=range(n_hours))
    if not renew_df.empty and "Curtailment" in renew_df.columns:
        dt_to_idx = {dt: i for i, dt in enumerate(hours_index)}
        rn = renew_df.copy()
        rn["_dt"] = _parse_dt(rn["Date"], rn["Hour"])
        for dt, grp in rn.groupby("_dt"):
            idx = dt_to_idx.get(dt)
            if idx is not None:
                curt_ts.iloc[idx] = grp["Curtailment"].sum()

    hod_curt = _hod_mean(curt_ts)

    # DC load + LMP from bus_detail
    bus_df = data[scen]["bus"]
    dc_load_ts = pd.Series(0.0, index=range(n_hours))
    lmp_ts_bus = pd.Series(np.nan, index=range(n_hours))
    if not bus_df.empty and "Bus" in bus_df.columns:
        dt_to_idx = {dt: i for i, dt in enumerate(hours_index)}
        bd = bus_df.copy()
        bd["_dt"] = _parse_dt(bd["Date"], bd["Hour"])
        dc_bd = bd[bd["Bus"].isin(DC_BUS_NAMES)]
        for dt, grp in dc_bd.groupby("_dt"):
            idx = dt_to_idx.get(dt)
            if idx is not None:
                if "Demand" in grp.columns:
                    dc_load_ts.iloc[idx] = grp["Demand"].sum()
        for dt, grp in bd.groupby("_dt"):
            idx = dt_to_idx.get(dt)
            if idx is not None and "LMP" in grp.columns:
                lmp_ts_bus.iloc[idx] = grp["LMP"].mean()

    hod_dc = _hod_mean(dc_load_ts)
    hod_lmp = _hod_mean(lmp_ts_bus.fillna(method="ffill").fillna(0.0))

    # Shared MW y-max across all 4 subplots
    mw_ymax = max(
        curt_ts.max(), battery_ts.max(),
        hod_curt.max(), hod_battery.max(),
        dc_load_ts.max(), hod_dc.max(),
    ) * 1.1
    mw_ymax = max(mw_ymax, 1.0)

    # Stat annotations
    curt_total = curt_ts.sum()
    batt_total = battery_ts.sum()
    absorption_pct = (batt_total / curt_total * 100.0) if curt_total > 0 else 0.0
    stat_str = (f"Curtailed: {curt_total:,.0f} MWh | "
                f"Battery absorbed: {batt_total:,.0f} MWh | "
                f"Absorption: {absorption_pct:.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_tl, ax_tr, ax_bl, ax_br = axes.flatten()

    # Top-left: curtailment vs battery (timeseries)
    ax_tl.plot(x, curt_ts.values, color="#e07b39", lw=1.2, label="Curtailment")
    ax_tl.plot(x, battery_ts.values, color=FUEL_COLORS["Battery"], lw=1.2, label="Battery")
    ax_tl.set_ylim(bottom=0, top=mw_ymax)
    ax_tl.set_ylabel("MW")
    ax_tl.set_title("Curtailment vs Battery dispatch")
    ax_tl.legend(fontsize=8)
    ax_tl.text(0.01, 0.97, stat_str, transform=ax_tl.transAxes,
               fontsize=6.5, va="top", color="#333333",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaaaaa", alpha=0.85))
    _day_xticks(ax_tl, n_hours)

    # Top-right: HOD curtailment vs battery
    ax_tr.plot(hod_x, hod_curt.values, color="#e07b39", lw=1.2, label="Curtailment")
    ax_tr.plot(hod_x, hod_battery.values, color=FUEL_COLORS["Battery"], lw=1.2, label="Battery")
    ax_tr.set_ylim(bottom=0, top=mw_ymax)
    ax_tr.set_ylabel("MW")
    ax_tr.set_title("HOD curtailment vs Battery")
    ax_tr.legend(fontsize=8)
    _hod_xticks(ax_tr)
    ax_tr.set_xlim(0, 23)

    # Bottom-left: DC load + battery + LMP (twin axis)
    ax_bl.plot(x, dc_load_ts.values, color="#444444", lw=1.2, label="DC load")
    ax_bl.plot(x, battery_ts.values, color=FUEL_COLORS["Battery"], lw=1.2, label="Battery")
    ax_bl.set_ylim(bottom=0, top=mw_ymax)
    ax_bl.set_ylabel("MW")
    ax_bl.set_title("DC load & Battery (LMP on right axis)")
    ax_bl.legend(fontsize=8, loc="upper left")
    if not lmp_ts_bus.isna().all():
        ax_bl2 = ax_bl.twinx()
        ax_bl2.plot(x, lmp_ts_bus.values, color="red", lw=0.8, ls="--", alpha=0.6, label="LMP")
        ax_bl2.set_ylabel("LMP ($/MWh)", color="red")
        ax_bl2.tick_params(axis="y", labelcolor="red")
    _day_xticks(ax_bl, n_hours)

    # Bottom-right: HOD DC load + battery
    ax_br.plot(hod_x, hod_dc.values, color="#444444", lw=1.2, label="DC load")
    ax_br.plot(hod_x, hod_battery.values, color=FUEL_COLORS["Battery"], lw=1.2, label="Battery")
    ax_br.set_ylim(bottom=0, top=mw_ymax)
    ax_br.set_ylabel("MW")
    ax_br.set_title("HOD DC load & Battery")
    ax_br.legend(fontsize=8)
    _hod_xticks(ax_br)
    ax_br.set_xlim(0, 23)

    fig.suptitle(f"Battery dispatch analysis (sim-lp) — {week_date}", fontsize=10)
    plt.tight_layout()
    _save_fig(fig, fig_dir, stem)


# ── Fig 3.6 — Water by scenario ────────────────────────────────────────────────

def _fig3_6(data: dict, fig_dir: Path, week_date: str) -> None:
    stem = "fig3_6_water_by_scenario"
    hours_index = data["hours_index"]
    present_scenarios = [s for s in SCENARIOS if s in data]
    if not present_scenarios:
        log.warning("fig3_6: no scenario data, skipping")
        return

    wd_totals: dict[str, float] = {}
    wc_totals: dict[str, float] = {}
    for scen in present_scenarios:
        wd = _water_series_mgal(data[scen]["water_df"], "total_wd_gal", hours_index)
        wc = _water_series_mgal(data[scen]["water_df"], "total_wc_gal", hours_index)
        wd_totals[scen] = wd.fillna(0.0).sum()
        wc_totals[scen] = wc.fillna(0.0).sum()

    baseline_wd = wd_totals.get("baseline", 0.0)
    x = np.arange(len(present_scenarios))

    import matplotlib.colors as mcolors

    def _lighten(color_name: str, factor: float = 0.5) -> tuple:
        """Return a lightened (more transparent) version of a named color."""
        rgb = mcolors.to_rgb(color_name)
        return tuple(1 - factor * (1 - c) for c in rgb)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x, [wd_totals[s] for s in present_scenarios],
                  color=[SCENARIO_COLORS[s] for s in present_scenarios],
                  edgecolor="white", linewidth=0.5, label="Withdrawal")

    # Lighter overlay for consumption
    ax.bar(x, [wc_totals[s] for s in present_scenarios],
           color=[_lighten(SCENARIO_COLORS[s], 0.45) for s in present_scenarios],
           edgecolor="white", linewidth=0.5, label="Consumption (lighter shade)")

    # Annotate % change vs baseline
    for i, scen in enumerate(present_scenarios):
        if scen == "baseline" or baseline_wd == 0:
            pct_str = "baseline"
        else:
            pct = (wd_totals[scen] - baseline_wd) / baseline_wd * 100.0
            sign = "+" if pct >= 0 else ""
            pct_str = f"{sign}{pct:.1f}%"
        bar = bars[i]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01, pct_str,
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in present_scenarios])
    ax.set_ylabel("Total Mgal")
    ax.set_title(f"Water withdrawal & consumption by scenario — {week_date}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_fig(fig, fig_dir, stem)


# ── Weekly tables ──────────────────────────────────────────────────────────────

def _load_sys_comp(week_dir: Path) -> Optional[pd.DataFrame]:
    p = week_dir / "compare" / "system_comparison.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, index_col=0)
    except Exception as exc:
        log.warning("Could not load system_comparison.csv: %s", exc)
        return None


def _make_weekly_deltas(
    week_date: str,
    season: str,
    sys_comp: pd.DataFrame,
    out_path: Path,
) -> None:
    rows = []
    b_co2  = float(sys_comp.at["total_co2_tonnes",          "baseline"]) if "total_co2_tonnes"          in sys_comp.index and "baseline" in sys_comp.columns else np.nan
    b_cost = float(sys_comp.at["total_cost_usd",            "baseline"]) if "total_cost_usd"            in sys_comp.index and "baseline" in sys_comp.columns else np.nan
    b_wd   = float(sys_comp.at["total_wd_gal",              "baseline"]) if "total_wd_gal"              in sys_comp.index and "baseline" in sys_comp.columns else np.nan
    b_shed = float(sys_comp.at["load_shedding_mwh",         "baseline"]) if "load_shedding_mwh"         in sys_comp.index and "baseline" in sys_comp.columns else np.nan
    b_curt = float(sys_comp.at["renewables_curtailed_mwh",  "baseline"]) if "renewables_curtailed_mwh"  in sys_comp.index and "baseline" in sys_comp.columns else np.nan
    b_coal = float(sys_comp.at["dispatch_Coal_mwh",         "baseline"]) if "dispatch_Coal_mwh"         in sys_comp.index and "baseline" in sys_comp.columns else np.nan

    for scen in ["sim-gm", "sim-247", "sim-lp"]:
        if scen not in sys_comp.columns:
            continue
        def _get(metric):
            return float(sys_comp.at[metric, scen]) if metric in sys_comp.index else np.nan
        co2  = _get("total_co2_tonnes")
        cost = _get("total_cost_usd")
        wd   = _get("total_wd_gal")
        shed = _get("load_shedding_mwh")
        curt = _get("renewables_curtailed_mwh")
        coal = _get("dispatch_Coal_mwh")

        rows.append({
            "week_date":                  week_date,
            "season":                     season,
            "scenario":                   scen,
            "baseline_co2_kt":            round(b_co2  / 1e3, 4) if not np.isnan(b_co2)  else np.nan,
            "delta_co2_t":                round(co2 - b_co2,  2) if not np.isnan(co2) and not np.isnan(b_co2) else np.nan,
            "delta_cost_pct":             round((cost - b_cost) / b_cost * 100, 4) if b_cost and not np.isnan(cost) else np.nan,
            "delta_water_withdrawal_pct": round((wd   - b_wd)   / b_wd   * 100, 4) if b_wd   and not np.isnan(wd)   else np.nan,
            "delta_load_shedding_pct":    round((shed - b_shed) / b_shed * 100, 4) if b_shed and not np.isnan(shed) else np.nan,
            "delta_curtailment_pct":      round((curt - b_curt) / b_curt * 100, 4) if b_curt and not np.isnan(curt) else np.nan,
            "delta_coal_dispatch_pct":    round((coal - b_coal) / b_coal * 100, 4) if b_coal and not np.isnan(coal) else np.nan,
        })

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, float_format="%.4f")
    log.info("→ %s", out_path)


def _make_weekly_summary(
    week_date: str,
    season: str,
    sys_comp: pd.DataFrame,
    out_path: Path,
    commit_flips: dict,
) -> None:
    METRIC_UNITS = {
        "total_co2_tonnes":            ("Total CO2",              "kt",      1e-3),
        "total_cost_usd":              ("Operational cost",        "$M",      1e-6),
        "total_wd_gal":                ("Water withdrawal",        "Bgal",    1e-9),
        "load_shedding_mwh":           ("Load shedding",           "MWh",     1.0),
        "renewables_curtailed_mwh":    ("Renewables curtailed",    "GWh",     1e-3),
        "dispatch_Coal_mwh":           ("Coal dispatch",           "GWh",     1e-3),
        "dispatch_NG_mwh":             ("Gas dispatch",            "GWh",     1e-3),
        "renewables_used_mwh":         ("Renewables used",         "GWh",     1e-3),
        "mean_ci_kgco2_mwh":           ("Carbon intensity",        "kg/MWh",  1.0),
        "avg_lmp_usd_mwh":             ("Avg LMP",                 "$/MWh",   1.0),
        "lolp":                        ("LOLP",                    "fraction", 1.0),
    }

    rows = []
    labels = ["baseline"] + [s for s in ["sim-gm", "sim-247", "sim-lp"]
                              if s in sys_comp.columns]

    for metric, (desc, unit, scale) in METRIC_UNITS.items():
        if metric not in sys_comp.index:
            continue
        row = {"week_date": week_date, "season": season, "metric": desc, "unit": unit}
        vals = {}
        for scen in labels:
            if scen in sys_comp.columns:
                try:
                    v = float(sys_comp.at[metric, scen]) * scale
                    row[scen] = round(v, 4)
                    vals[scen] = v
                except (ValueError, TypeError):
                    pass
        higher_is_better = metric in {"renewables_used_mwh"}
        shifted = {k: v for k, v in vals.items() if k != "baseline"}
        if shifted:
            winner = max(shifted, key=shifted.__getitem__) if higher_is_better \
                     else min(shifted, key=shifted.__getitem__)
            row["winner"] = winner
        rows.append(row)

    for scen, n_flips in commit_flips.items():
        rows.append({
            "week_date": week_date, "season": season,
            "metric": "Commitment flips vs baseline",
            "unit": "gen-hours", scen: n_flips, "winner": "",
        })

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, float_format="%.4f")
    log.info("→ %s", out_path)


def _make_diagnostics(
    week_date: str,
    week_dir: Path,
    data: dict,
    out_path: Path,
) -> dict:
    diag: dict = {"week_date": week_date}
    hours_index = data["hours_index"]

    # Curtailment hours per scenario
    curt_hours: dict[str, int] = {}
    for scen in SCENARIOS:
        if scen not in data:
            continue
        hs = data[scen]["hourly"]
        if "RenewablesCurtailment" in hs.columns:
            curt_hours[scen] = int((hs["RenewablesCurtailment"] > 0).sum())
    diag["curtailment_hours"] = curt_hours

    # Peak LMP per bus (baseline)
    peak_lmp: dict[str, float] = {}
    if "baseline" in data:
        bd = data["baseline"]["bus"]
        if not bd.empty and "LMP" in bd.columns and "Bus" in bd.columns:
            peak_lmp = bd.groupby("Bus")["LMP"].max().round(2).to_dict()
    diag["peak_lmp_by_bus"] = peak_lmp

    # LMP L2 distance (endogeneity signal)
    lmp_l2: dict[str, float] = {}
    if "baseline" in data:
        bd_base = data["baseline"]["bus"]
        if not bd_base.empty and "LMP" in bd_base.columns:
            bd_base = bd_base.copy()
            bd_base["_dt"] = _parse_dt(bd_base["Date"], bd_base["Hour"])
            base_vec = bd_base.set_index(["_dt", "Bus"])["LMP"]
            for scen in ["sim-gm", "sim-247", "sim-lp"]:
                if scen not in data:
                    continue
                bd_s = data[scen]["bus"]
                if bd_s.empty or "LMP" not in bd_s.columns:
                    continue
                bd_s = bd_s.copy()
                bd_s["_dt"] = _parse_dt(bd_s["Date"], bd_s["Hour"])
                scen_vec = bd_s.set_index(["_dt", "Bus"])["LMP"]
                common = base_vec.index.intersection(scen_vec.index)
                if len(common) == 0:
                    continue
                diff = (base_vec[common] - scen_vec[common]).values
                lmp_l2[scen] = round(float(np.linalg.norm(diff)), 4)
    diag["lmp_l2_distance"] = lmp_l2

    # Commitment flips
    commit_flips: dict[str, int] = {}
    if "baseline" in data:
        th_base = data["baseline"]["thermal"]
        if not th_base.empty and "Unit State" in th_base.columns:
            base_state = (th_base.set_index(["Date", "Hour", "Generator"])["Unit State"]
                          .astype(bool))
            for scen in ["sim-gm", "sim-247", "sim-lp"]:
                if scen not in data:
                    continue
                th_s = data[scen]["thermal"]
                if th_s.empty or "Unit State" not in th_s.columns:
                    continue
                scen_state = (th_s.set_index(["Date", "Hour", "Generator"])["Unit State"]
                              .astype(bool))
                common = base_state.index.intersection(scen_state.index)
                flips = int((base_state[common] != scen_state[common]).sum())
                commit_flips[scen] = flips
    diag["commitment_flips"] = commit_flips

    # sim-lp convergence
    iter_csv = week_dir / "sim_lp_iterations" / "iter_convergence.csv"
    if iter_csv.exists():
        try:
            iter_df = pd.read_csv(iter_csv)
            diag["sim_lp_iter_rows"] = len(iter_df)
            if "converged" in iter_df.columns:
                conv_rows = iter_df[iter_df["converged"].astype(bool)]
                diag["sim_lp_converged"] = not conv_rows.empty
                diag["sim_lp_converged_at"] = int(conv_rows.index[0]) if not conv_rows.empty else None
            else:
                diag["sim_lp_converged"] = None
        except Exception:
            diag["sim_lp_converged"] = None
    else:
        iter_json = week_dir / "sim-lp" / "cas_iter_info.json"
        if iter_json.exists():
            try:
                info = json.loads(iter_json.read_text())
                diag["sim_lp_converged"] = info.get("converged_at", 999) < info.get("max_iter", 3)
                diag["sim_lp_converged_at"] = info.get("converged_at")
                diag["sim_lp_max_iter"] = info.get("max_iter")
            except Exception:
                diag["sim_lp_converged"] = None
        else:
            diag["sim_lp_converged"] = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diag, indent=2, default=str))
    log.info("→ %s", out_path)
    return commit_flips


# ── Main entry point ────────────────────────────────────────────────────────────

def run(week_dir: str | Path, *, force: bool = False, figures: bool = True) -> None:
    """Generate all per-week outputs for the given simulation directory."""
    week_dir = Path(week_dir)
    if not week_dir.exists():
        log.warning("week_dir not found: %s", week_dir)
        return

    week_date = week_dir.name
    season = _infer_season(week_date)
    fig_dir = week_dir / "figures"

    log.info("Processing %s (%s)  force=%s", week_date, season, force)

    # Load all data
    data = _load_all(week_dir)
    n_hours = data["n_hours"]
    if n_hours == 0:
        log.warning("No hourly data found in %s — aborting", week_dir)
        return

    # ── Figures ───────────────────────────────────────────────────────────────
    figs_written = 0

    if figures:
        def _maybe(stem: str, fn, *args, **kwargs):
            nonlocal figs_written
            out_png = fig_dir / f"{stem}.png"
            if force or not out_png.exists():
                try:
                    fn(*args, **kwargs)
                    figs_written += 1
                except Exception as exc:
                    import traceback
                    log.warning("Error generating %s: %s\n%s", stem, exc,
                                traceback.format_exc())
            else:
                log.info("skip (exists) %s", out_png.name)

        _maybe("fig1_1_nodc_vs_dc",                _fig1_1, data, fig_dir, week_date)
        _maybe("fig1_3_seasonal_dispatch",          _fig1_3, data, fig_dir, week_date)
        _maybe("fig1_4_seasonal_water",             _fig1_4, data, fig_dir, week_date)
        _maybe("fig2_2_seasonal_dispatch_battery",  _fig2_2, data, fig_dir, week_date)
        _maybe("fig2_3_co2_water_panel",            _fig2_3, data, fig_dir, week_date)
        _maybe("fig2_4_lmp_cost_panel",             _fig2_4, data, fig_dir, week_date)
        _maybe("fig2_5_co2_hod_all_scenarios",      _fig2_5, data, fig_dir, week_date)
        _maybe("fig3_3_fuel_mix_by_scenario",       _fig3_3, data, fig_dir, week_date)
        _maybe("fig3_4_co2_delta",                  _fig3_4, data, fig_dir, week_date)
        _maybe("fig3_5_battery_panel",              _fig3_5, data, fig_dir, week_date)
        _maybe("fig3_6_water_by_scenario",          _fig3_6, data, fig_dir, week_date)

    # ── Tables & diagnostics ──────────────────────────────────────────────────
    diag_path    = week_dir / f"diagnostics_{week_date}.json"
    deltas_path  = week_dir / f"weekly_deltas_{week_date}.csv"
    summary_path = week_dir / f"weekly_summary_{week_date}.csv"

    if force or not diag_path.exists():
        commit_flips = _make_diagnostics(week_date, week_dir, data, diag_path)
    else:
        log.info("skip (exists) %s", diag_path.name)
        try:
            commit_flips = json.loads(diag_path.read_text()).get("commitment_flips", {})
        except Exception:
            commit_flips = {}

    sys_comp = _load_sys_comp(week_dir)
    if sys_comp is not None:
        if force or not deltas_path.exists():
            _make_weekly_deltas(week_date, season, sys_comp, deltas_path)
        else:
            log.info("skip (exists) %s", deltas_path.name)

        if force or not summary_path.exists():
            _make_weekly_summary(week_date, season, sys_comp, summary_path, commit_flips)
        else:
            log.info("skip (exists) %s", summary_path.name)
    else:
        log.warning("system_comparison.csv not found — weekly tables skipped")

    print(f"[run_outputs] {week_date} → {figs_written} figures written")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Generate per-week thesis figures for a 14-day VATIC simulation."
    )
    p.add_argument("week_dir", help="Path to week output dir (e.g. outputs/2020-05-04)")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.week_dir, force=args.force)
