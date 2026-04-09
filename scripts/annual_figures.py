#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
annual_figures.py — Generate all thesis figures from 12-month simulation data.

Outputs go to outputs/annual_figures/{fig_id}.png

Usage:
    python scripts/annual_figures.py
    python scripts/annual_figures.py --out-dir outputs/annual_figures
"""

import argparse
import calendar
import json
import logging
import re
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Global plot theme ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.size":            10,
    "axes.titlesize":       11,
    "axes.labelsize":       10,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "figure.titlesize":     12,
    "figure.titleweight":   "bold",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.color":           "#e0e0e0",
    "grid.linewidth":       0.6,
    "axes.axisbelow":       True,
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "legend.framealpha":    0.85,
    "legend.edgecolor":     "#cccccc",
})

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPTS_DIR  = Path(__file__).parent
VATIC_ROOT   = SCRIPTS_DIR.parent
OUTPUTS_ROOT = VATIC_ROOT / "outputs"
GRIDS_DIR    = VATIC_ROOT / "vatic" / "data" / "grids"

_GRID_REGISTRY: dict[str, str] = {
    "Texas-7k_2030": "TX2030_Data",
    "Texas-7k":      "TX_Data",
    "RTS-GMLC":      "RTS_Data",
}


def _data_dir_for(grid_name: str) -> str:
    for prefix, data_dir in sorted(_GRID_REGISTRY.items(), key=lambda x: len(x[0]), reverse=True):
        if grid_name.startswith(prefix):
            return data_dir
    return "RTS_Data"


def _grid_source_csvs(grid_name: str) -> tuple[Path, Path]:
    """Return (gen_csv, bus_csv) paths for a grid."""
    src = GRIDS_DIR / grid_name / _data_dir_for(grid_name) / "SourceData"
    return src / "gen.csv", src / "bus.csv"


# Defaults (overridden in main() from --grid arg)
GEN_CSV, BUS_CSV = _grid_source_csvs("RTS-GMLC")

# ── Month / season constants ───────────────────────────────────────────────────
_MONTHS_FALLBACK = [
    ("Jan", "2020-01-01", "Winter"),
    ("Feb", "2020-02-01", "Winter"),
    ("Mar", "2020-03-01", "Spring"),
    ("Apr", "2020-04-01", "Spring"),
    ("May", "2020-05-01", "Spring"),
    ("Jun", "2020-06-01", "Summer"),
    ("Jul", "2020-07-01", "Summer"),
    ("Aug", "2020-08-01", "Summer"),
    ("Sep", "2020-09-01", "Fall"),
    ("Oct", "2020-10-01", "Fall"),
    ("Nov", "2020-11-01", "Fall"),
    ("Dec", "2020-12-01", "Winter"),
]

_MONTH_SEASON = {
    1: "Winter", 2: "Winter",  3: "Spring",
    4: "Spring", 5: "Spring",  6: "Summer",
    7: "Summer", 8: "Summer",  9: "Fall",
    10: "Fall",  11: "Fall",   12: "Winter",
}
_DATE_RE = re.compile(r"^\d{4}-(\d{2})-\d{2}$")


def _discover_months(outputs_root: Path) -> list[tuple[str, str, str]]:
    """Scan outputs_root for YYYY-MM-DD dirs that have a completed baseline.

    Returns one entry per calendar month (earliest date found for each month),
    sorted chronologically.  Falls back to an empty list if none are found.
    """
    if not outputs_root.exists():
        return []
    found: dict[int, str] = {}  # month_num → earliest date string
    for d in outputs_root.iterdir():
        if not d.is_dir():
            continue
        m = _DATE_RE.match(d.name)
        if not m:
            continue
        if not (d / "baseline" / "thermal_detail.csv").exists():
            continue
        month_num = int(m.group(1))
        if month_num not in found or d.name < found[month_num]:
            found[month_num] = d.name
    return [
        (calendar.month_abbr[mn], found[mn], _MONTH_SEASON[mn])
        for mn in sorted(found)
    ]


def _init_months(outputs_root: Path = OUTPUTS_ROOT) -> list[tuple[str, str, str]]:
    """Return discovered MONTHS list, falling back to hardcoded 2020 dates."""
    discovered = _discover_months(outputs_root)
    if discovered:
        log.info("MONTHS: discovered %d entries from %s", len(discovered), outputs_root)
        return discovered
    log.info("MONTHS: no outputs found — using hardcoded 2020 fallback")
    return _MONTHS_FALLBACK


MONTHS       = _init_months()
SEASONS      = ["Winter", "Spring", "Summer", "Fall"]
SEASON_DATES = {s: [d for _, d, ss in MONTHS if ss == s] for s in SEASONS}
# One representative date per season: pick the first (chronologically earliest)
SEASON_REP   = {
    s: min(SEASON_DATES[s]) if SEASON_DATES.get(s) else ""
    for s in SEASONS
}

SCENARIOS = ["baseline", "sim-gm", "sim-247", "sim-lp"]
SCENARIO_LABELS = {
    "baseline": "Baseline", "sim-gm": "Grid-Mix",
    "sim-247": "24/7", "sim-lp": "LP",
}

# ── Colors ─────────────────────────────────────────────────────────────────────
FUEL_ORDER = ["Nuclear", "Coal", "NG-CC", "NG-CT", "Oil",
              "Hydro", "Storage", "Wind", "Solar"]
FUEL_COLORS = {
    "Nuclear": "steelblue", "Coal": "dimgray",
    "NG-CC": "peru",        "NG-CT": "wheat",
    "Oil": "darkkhaki",     "Hydro": "royalblue",
    "Storage": "mediumpurple", "Wind": "mediumseagreen", "Solar": "gold",
}
SCENARIO_COLORS = {
    "baseline": "steelblue", "sim-gm": "darkorange",
    "sim-247": "forestgreen", "sim-lp": "crimson",
}
SEASON_COLORS = {
    "Winter": "#4575b4", "Spring": "#74c476",
    "Summer": "#d73027", "Fall":   "#f4a582",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _gen_fuel_map() -> dict:
    """Return {GEN_UID: fuel_group} from gen.csv."""
    gen = pd.read_csv(GEN_CSV)
    mapping = {}
    for _, row in gen.iterrows():
        fuel = row["Fuel"]
        utype = row["Unit Type"]
        if fuel == "Coal":             grp = "Coal"
        elif fuel == "Nuclear":        grp = "Nuclear"
        elif fuel in ("NG", "Gas"):
            grp = "NG-CC" if utype == "CC" else "NG-CT"
        elif fuel == "Oil":            grp = "Oil"
        elif fuel == "Hydro":          grp = "Hydro"
        elif fuel == "Solar":          grp = "Solar"
        elif fuel == "Wind":           grp = "Wind"
        elif fuel == "Storage":        grp = "Storage"
        else:                          grp = None
        if grp:
            mapping[row["GEN UID"]] = grp
    return mapping


def _load_dispatch(date: str, scenario: str, fuel_map: dict) -> pd.DataFrame:
    """
    Return hourly dispatch by fuel group (MW).
    Columns: datetime, Nuclear, Coal, NG-CC, NG-CT, Oil, Hydro, Storage, Wind, Solar, Demand
    """
    base = OUTPUTS_ROOT / date / scenario
    th  = pd.read_csv(base / "thermal_detail.csv")
    re  = pd.read_csv(base / "renew_detail.csv")
    hs  = pd.read_csv(base / "hourly_summary.csv")

    th["fuel"] = th["Generator"].map(fuel_map)
    re["fuel"] = re["Generator"].map(fuel_map)

    th_grp = (th.groupby(["Date", "Hour", "fuel"])["Dispatch"]
                .sum().reset_index().rename(columns={"Dispatch": "MW"}))
    re_grp = (re.groupby(["Date", "Hour", "fuel"])["Output"]
                .sum().reset_index().rename(columns={"Output": "MW"}))

    combined = pd.concat([th_grp, re_grp], ignore_index=True)
    combined = combined[combined["fuel"].notna()]
    pivot = combined.pivot_table(index=["Date", "Hour"], columns="fuel",
                                 values="MW", aggfunc="sum").fillna(0).reset_index()
    pivot["datetime"] = pd.to_datetime(
        pivot["Date"].astype(str) + " " + pivot["Hour"].astype(int).astype(str) + ":00")

    # Merge demand
    hs["datetime"] = pd.to_datetime(
        hs["Date"].astype(str) + " " + hs["Hour"].astype(int).astype(str) + ":00")
    pivot = pivot.merge(hs[["datetime", "Demand"]], on="datetime", how="left")

    for col in FUEL_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0.0

    return pivot.sort_values("datetime").reset_index(drop=True)


def _load_water(date: str, scenario: str) -> pd.DataFrame:
    path = OUTPUTS_ROOT / date / "water" / scenario / "system_water_hourly.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Hour"].astype(int).astype(str) + ":00")
    return df


def _load_hourly(date: str, scenario: str) -> pd.DataFrame:
    path = OUTPUTS_ROOT / date / scenario / "hourly_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Hour"].astype(int).astype(str) + ":00")
    return df


def _load_weekly_summaries() -> pd.DataFrame:
    rows = []
    for _, date, season in MONTHS:
        path = OUTPUTS_ROOT / date / f"weekly_summary_{date}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["date"]   = date
            df["season"] = season
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _co2_series(date: str, scenario: str,
                gen_df: pd.DataFrame) -> pd.Series:
    """Hourly CO2 (metric tonnes) for one scenario.

    Works with both RTS-GMLC gen.csv (HR_avg_0 + Emissions CO2 Lbs/MMBTU)
    and Texas-7k gen.csv (uses fuel-keyword fallback factors).
    """
    # Fallback factors (metric tonnes CO2/MWh) — eGRID 2023 TX / EIA sources.
    _FALLBACK_T: dict[str, float] = {
        "coal": 1.0785, "lignite": 1.0785, "subbituminous": 1.0785,
        "petroleum coke": 1.0212, "oil": 0.7958,
        "natural gas": 0.4963, "ng": 0.4963, "gas": 0.4963,
    }
    base = OUTPUTS_ROOT / date / scenario
    th   = pd.read_csv(base / "thermal_detail.csv")
    if "HR_avg_0" in gen_df.columns and "Emissions CO2 Lbs/MMBTU" in gen_df.columns:
        hr     = gen_df["HR_avg_0"].astype(float) / 1000.0
        ef_lbs = pd.to_numeric(gen_df["Emissions CO2 Lbs/MMBTU"], errors="coerce").fillna(0)
        ef     = hr * ef_lbs * 0.000453592   # metric tonnes CO2/MWh
    else:
        fuel_col = next((c for c in gen_df.columns if c.strip().lower() == "fuel"), None)
        fuels    = gen_df[fuel_col].astype(str).str.lower() if fuel_col else pd.Series([""] * len(gen_df))
        ef       = fuels.map(lambda f: next((v for k, v in _FALLBACK_T.items() if k in f), 0.0))
    ef.index = gen_df["GEN UID"]
    th["co2_t"] = th["Dispatch"] * th["Generator"].map(ef).fillna(0)
    th["datetime"] = pd.to_datetime(
        th["Date"].astype(str) + " " + th["Hour"].astype(int).astype(str) + ":00")
    return th.groupby("datetime")["co2_t"].sum()


# ── Save helper ────────────────────────────────────────────────────────────────
def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{stem}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", p)


# ── Stacked area helper ────────────────────────────────────────────────────────
def _stacked_area(ax, df: pd.DataFrame, x_col: str = "datetime") -> None:
    x      = df[x_col].values
    bottom = np.zeros(len(df))
    for fuel in FUEL_ORDER:
        if fuel not in df.columns:
            continue
        vals = df[fuel].fillna(0).values
        if vals.sum() == 0:
            continue
        ax.fill_between(x, bottom, bottom + vals,
                        color=FUEL_COLORS[fuel], label=fuel, alpha=0.92)
        bottom += vals


def _stacked_area_hod(ax, hod: pd.DataFrame) -> None:
    """hod indexed 0–23 with fuel columns."""
    x      = np.arange(24)
    bottom = np.zeros(24)
    for fuel in FUEL_ORDER:
        if fuel not in hod.columns:
            continue
        vals = hod[fuel].fillna(0).values
        if vals.sum() == 0:
            continue
        ax.fill_between(x, bottom, bottom + vals,
                        color=FUEL_COLORS[fuel], label=fuel, alpha=0.92)
        bottom += vals


def _fuel_legend(ax) -> None:
    handles = [mpatches.Patch(color=FUEL_COLORS[f], label=f)
               for f in FUEL_ORDER if f in FUEL_COLORS]
    ax.legend(handles=handles, loc="upper right", ncol=2)


# ══════════════════════════════════════════════════════════════════════════════
# §1  BASELINE GRID CHARACTERIZATION
# ══════════════════════════════════════════════════════════════════════════════

def fig1_1(out_dir: Path, fuel_map: dict) -> None:
    """Fig 1.1 — Side-by-side winter/summer timeseries dispatch."""
    log.info("Fig 1.1 — baseline dispatch timeseries winter/summer")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    ymax = 0.0
    dfs  = {}
    for season, date in [("Winter", "2020-01-06"), ("Summer", "2020-07-06")]:
        df = _load_dispatch(date, "baseline", fuel_map)
        dfs[season] = df
        ymax = max(ymax, df["Demand"].max() * 1.05)

    for ax, (season, date) in zip(axes, [("Winter", "2020-01-06"), ("Summer", "2020-07-06")]):
        df = dfs[season]
        _stacked_area(ax, df)
        ax.plot(df["datetime"], df["Demand"], color="black",
                linewidth=1.2, label="Demand", zorder=5)
        ax.set_ylim(0, ymax)
        ax.set_title(f"{season} ({date})")
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=30)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(7))

    axes[0].set_ylabel("Dispatch (MW)")
    _fuel_legend(axes[1])
    axes[1].get_legend().set_title("Fuel")
    # Add DC load shade
    fig.suptitle("Baseline Grid Dispatch: Winter vs Summer (with DC Load)")
    fig.tight_layout()
    _save(fig, out_dir, "fig1_1_dispatch_timeseries")


def fig1_2_table(out_dir: Path, fuel_map: dict, gen_df: pd.DataFrame) -> None:
    """Fig 1.2 — Table: annual metrics with vs without DC load."""
    log.info("Fig 1.2 — annual grid metrics table (DC vs no-DC)")

    rows = []
    for _, date, season in MONTHS:
        nodc_path = OUTPUTS_ROOT / date / "baseline-nodc" / "thermal_detail.csv"
        dc_path   = OUTPUTS_ROOT / date / "baseline" / "thermal_detail.csv"
        if not nodc_path.exists() or not dc_path.exists():
            continue

        for label, scenario in [("No DC", "baseline-nodc"), ("With DC", "baseline")]:
            hs  = _load_hourly(date, scenario)
            if hs.empty:
                continue
            co2 = _co2_series(date, scenario, gen_df).sum()
            rows.append({
                "date": date, "season": season, "label": label,
                "co2_kt":    round(co2 / 1000, 1),
                "cost_m":    round(hs["VariableCosts"].sum() / 1e6, 2),
                "shed_mwh":  round(hs["LoadShedding"].sum(), 1),
                "demand_mwh": round(hs["Demand"].sum(), 0),
            })

    if not rows:
        log.warning("Fig 1.2 — no-DC baseline data not yet available, skipping")
        return

    df = pd.DataFrame(rows)

    # Aggregate annually
    agg = (df.groupby("label")[["co2_kt", "cost_m", "shed_mwh", "demand_mwh"]]
             .sum().reset_index())
    agg["renew_pct"] = np.nan   # placeholder — needs renew_detail

    nodc = agg[agg["label"] == "No DC"].iloc[0]
    dc   = agg[agg["label"] == "With DC"].iloc[0]

    metrics = [
        ("Annual CO₂ (kt)",       nodc["co2_kt"],  dc["co2_kt"],  "kt",  True),
        ("Operational cost ($M)",  nodc["cost_m"],  dc["cost_m"],  "$M",  True),
        ("Load shedding (MWh)",    nodc["shed_mwh"],dc["shed_mwh"],"MWh", True),
        ("Total demand (MWh)",     nodc["demand_mwh"],dc["demand_mwh"],"MWh",False),
    ]

    tbl_data  = []
    row_labels = []
    for name, v_nodc, v_dc, unit, _ in metrics:
        delta     = v_dc - v_nodc
        delta_pct = 100 * delta / max(abs(v_nodc), 1)
        tbl_data.append([f"{v_nodc:,.1f}", f"{v_dc:,.1f}",
                         f"{delta:+,.1f}", f"{delta_pct:+.1f}%"])
        row_labels.append(f"{name} ({unit})")

    fig, ax = plt.subplots(figsize=(10, len(metrics) * 0.7 + 1.5))
    ax.axis("off")
    col_labels = ["RTS-GMLC\n(No DC)", "RTS-GMLC\n+Data Centers", "Δ (abs)", "Δ (%)"]
    tbl = ax.table(cellText=tbl_data, rowLabels=row_labels, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)

    # Color header row
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    fig.suptitle("Annual Grid Metrics: Original vs +Data Centers", y=0.98)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_2_annual_metrics_table")


def fig1_2b(out_dir: Path, fuel_map: dict) -> None:
    """Fig 1.2b — 2×2 dispatch timeseries: No DC vs +DC, Winter and Summer."""
    log.info("Fig 1.2b — dispatch timeseries no-DC vs DC, winter/summer")

    season_pairs = [("Winter", "2020-01-06"), ("Summer", "2020-07-06")]
    cases = [("No DC", "baseline-nodc"), ("With DC", "baseline")]

    # Pre-load and compute shared y-axis max
    dfs = {}
    ymax = 0.0
    for season, date in season_pairs:
        for label, scenario in cases:
            path = OUTPUTS_ROOT / date / scenario / "thermal_detail.csv"
            if not path.exists():
                log.warning("Fig 1.2b — missing %s/%s, skipping", date, scenario)
                return
            df = _load_dispatch(date, scenario, fuel_map)
            dfs[(season, label)] = df
            ymax = max(ymax, df["Demand"].max() * 1.05)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

    for row, (season, date) in enumerate(season_pairs):
        for col, (label, scenario) in enumerate(cases):
            ax = axes[row][col]
            df = dfs[(season, label)]
            _stacked_area(ax, df)
            ax.plot(df["datetime"], df["Demand"], color="black",
                    linewidth=1.2, label="Demand", zorder=5)
            ax.set_ylim(0, ymax)
            ax.set_title(f"{season} — {label}  ({date})")
            ax.tick_params(axis="x", rotation=30)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(7))
            if col == 0:
                ax.set_ylabel("Dispatch (MW)")
            if row == 1:
                ax.set_xlabel("Date")

    _fuel_legend(axes[0][1])
    fig.suptitle("Dispatch Timeseries: Without vs With Data Center Load")
    fig.tight_layout()
    _save(fig, out_dir, "fig1_2b_dispatch_nodc_vs_dc")


def fig1_3(out_dir: Path, fuel_map: dict) -> None:
    """Fig 1.3 — Seasonal HOD dispatch, 4 panels."""
    log.info("Fig 1.3 — seasonal HOD dispatch")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    ymax = 0.0
    season_hod = {}
    for season in SEASONS:
        dfs = []
        for date in SEASON_DATES[season]:
            try:
                df = _load_dispatch(date, "baseline", fuel_map)
                df["hod"] = df["datetime"].dt.hour
                dfs.append(df)
            except Exception:
                pass
        if not dfs:
            season_hod[season] = pd.DataFrame()
            continue
        combined = pd.concat(dfs, ignore_index=True)
        hod = combined.groupby("hod")[FUEL_ORDER + ["Demand"]].mean()
        season_hod[season] = hod
        ymax = max(ymax, hod["Demand"].max() * 1.05)

    for ax, season in zip(axes, SEASONS):
        hod = season_hod[season]
        if hod.empty:
            ax.text(0.5, 0.5, "Data pending", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=11)
        else:
            _stacked_area_hod(ax, hod)
            ax.plot(np.arange(24), hod["Demand"], color="black",
                    linewidth=1.4, label="Demand", zorder=5)
        ax.set_ylim(0, ymax if ymax > 0 else 4000)
        ax.set_title(season, color=SEASON_COLORS[season], fontweight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_xticks([0, 6, 12, 18, 23])

    axes[0].set_ylabel("Dispatch (MW)")
    _fuel_legend(axes[-1])
    fig.suptitle("Seasonal Dispatch by Fuel Mix (Hour-of-Day Average)")
    fig.tight_layout()
    _save(fig, out_dir, "fig1_3_seasonal_hod_dispatch")


def fig1_4(out_dir: Path) -> None:
    """Fig 1.4 — Seasonal water withdrawal vs consumption HOD."""
    log.info("Fig 1.4 — seasonal water")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    ymax_wd = 0.0
    season_data = {}
    for season in SEASONS:
        wds, wcs = [], []
        for date in SEASON_DATES[season]:
            w = _load_water(date, "baseline")
            if w.empty:
                continue
            w["hod"] = w["datetime"].dt.hour
            wds.append(w.groupby("hod")["total_wd_gal"].mean())
            wcs.append(w.groupby("hod")["total_wc_gal"].mean())
        if not wds:
            season_data[season] = None
            continue
        wd = pd.concat(wds, axis=1).mean(axis=1) / 1e9   # billion gal
        wc = pd.concat(wcs, axis=1).mean(axis=1) / 1e9
        season_data[season] = (wd, wc)
        ymax_wd = max(ymax_wd, wd.max() * 1.1)

    for ax, season in zip(axes, SEASONS):
        data = season_data.get(season)
        if data is None:
            ax.text(0.5, 0.5, "Data pending", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
        else:
            wd, wc = data
            ax.plot(wd.index, wd.values, color=SEASON_COLORS[season],
                    linewidth=2, label="Withdrawal")
            ax.plot(wc.index, wc.values, color=SEASON_COLORS[season],
                    linewidth=2, linestyle="--", label="Consumption")
        ax.set_title(season, color=SEASON_COLORS[season], fontweight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_ylim(0, ymax_wd if ymax_wd > 0 else 0.1)

    axes[0].set_ylabel("Water (billion gal/hr)")
    axes[0].legend()
    fig.text(0.5, -0.02,
             "Solid = withdrawal (water taken from source);  "
             "Dashed = consumption (water not returned, evaporated/lost)",
             ha="center", fontsize=8, style="italic")
    fig.suptitle("Seasonal Water Withdrawal vs Consumption (Hour-of-Day Average)")
    fig.tight_layout()
    _save(fig, out_dir, "fig1_4_seasonal_water")


# ══════════════════════════════════════════════════════════════════════════════
# §2  CAS ALGORITHM PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def table2_1(out_dir: Path) -> None:
    """Table 2.1 — Annual performance summary, all 3 CAS modes vs baseline."""
    log.info("Table 2.1 — annual CAS performance")
    ws = _load_weekly_summaries()
    if ws.empty:
        return

    # Pivot to wide format per metric
    metrics_of_interest = ["Total CO2", "Operational cost", "Water withdrawal",
                            "Load shedding", "Renewables curtailed"]

    results = {}
    for metric in metrics_of_interest:
        sub = ws[ws["metric"] == metric]
        if sub.empty:
            continue
        totals = sub[["baseline", "sim-gm", "sim-247", "sim-lp"]].sum()
        results[metric] = totals

    if not results:
        return

    df_res = pd.DataFrame(results).T
    df_res["unit"] = ws.drop_duplicates("metric").set_index("metric")["unit"]

    # Compute deltas
    for sc in ["sim-gm", "sim-247", "sim-lp"]:
        df_res[f"Δ{sc}_abs"] = df_res[sc] - df_res["baseline"]
        df_res[f"Δ{sc}_pct"] = 100 * df_res[f"Δ{sc}_abs"] / df_res["baseline"].abs()

    # Build table cells
    row_labels = [f"{m} ({df_res.loc[m,'unit']})" for m in df_res.index if m in df_res.index]
    col_labels = ["Baseline", "Grid-Mix\nΔ abs", "Grid-Mix\nΔ%",
                  "24/7\nΔ abs", "24/7\nΔ%", "LP\nΔ abs", "LP\nΔ%"]

    cell_data = []
    for metric in df_res.index:
        row = df_res.loc[metric]
        b = row["baseline"]
        cells = [f"{b:.1f}"]
        for sc in ["sim-gm", "sim-247", "sim-lp"]:
            cells += [f"{row[f'Δ{sc}_abs']:+.1f}", f"{row[f'Δ{sc}_pct']:+.1f}%"]
        cell_data.append(cells)

    fig, ax = plt.subplots(figsize=(14, len(row_labels) * 0.65 + 2))
    ax.axis("off")
    tbl = ax.table(cellText=cell_data, rowLabels=row_labels, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.5)

    # Header color
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Color Δ% cells
    for i, metric in enumerate(df_res.index, start=1):
        row = df_res.loc[metric]
        for j_offset, sc in enumerate(["sim-gm", "sim-247", "sim-lp"]):
            pct = row[f"Δ{sc}_pct"]
            col_idx = 2 + j_offset * 2   # Δ% column
            if pct < -0.5:
                tbl[(i, col_idx)].set_facecolor("#c8e6c9")
            elif pct > 0.5:
                tbl[(i, col_idx)].set_facecolor("#ffcdd2")

    fig.suptitle("Annual CAS Performance Summary (all 12 months aggregated)", y=0.99)
    fig.tight_layout()
    _save(fig, out_dir, "table2_1_annual_performance")


def fig2_3(out_dir: Path, fuel_map: dict, gen_df: pd.DataFrame) -> None:
    """Fig 2.3 — 2×2: CO2 timeseries (top) + water (bottom), winter/summer."""
    log.info("Fig 2.3 — CO2+water panel 2×2")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    season_pairs = [("Winter", "2020-01-06"), ("Summer", "2020-07-06")]

    # Top row: CO2 timeseries
    co2_ymax = 0.0
    co2_data = {}
    for season, date in season_pairs:
        for sc in SCENARIOS:
            s = _co2_series(date, sc, gen_df)
            co2_data[(season, sc)] = s
            co2_ymax = max(co2_ymax, s.max() * 1.1)

    for ax, (season, date) in zip(axes[0], season_pairs):
        for sc in SCENARIOS:
            s = co2_data[(season, sc)]
            ax.plot(s.index, s.values, color=SCENARIO_COLORS[sc],
                    label=SCENARIO_LABELS[sc], linewidth=1.2)
        ax.set_ylim(0, co2_ymax)
        ax.set_title(f"CO₂ Emissions — {season}")
        ax.set_ylabel("CO₂ (t/hr)")
        ax.tick_params(axis="x", rotation=25)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    axes[0][1].legend()

    # Bottom row: water
    wd_ymax = 0.0
    for ax, (season, date) in zip(axes[1], season_pairs):
        for sc in SCENARIOS:
            w = _load_water(date, sc)
            if w.empty:
                continue
            ax.plot(w["datetime"], w["total_wd_gal"] / 1e9, color=SCENARIO_COLORS[sc],
                    label=f"{SCENARIO_LABELS[sc]} WD", linewidth=1.2)
            ax.plot(w["datetime"], w["total_wc_gal"] / 1e9, color=SCENARIO_COLORS[sc],
                    linewidth=1.2, linestyle="--", label=f"{SCENARIO_LABELS[sc]} WC")
            wd_ymax = max(wd_ymax, w["total_wd_gal"].max() / 1e9 * 1.1)
        ax.set_title(f"Water — {season}")
        ax.set_ylabel("Water (billion gal/hr)")
        ax.tick_params(axis="x", rotation=25)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

    for ax in axes[1]:
        ax.set_ylim(bottom=0)
    axes[1][1].legend()

    fig.suptitle("CO₂ Emissions and Water Use: All CAS Modes\n(Solid = withdrawal, Dashed = consumption)")
    fig.tight_layout()
    _save(fig, out_dir, "fig2_3_co2_water_panel")


def fig2_4(out_dir: Path) -> None:
    """Fig 2.4 — 1×2: hourly operational cost, winter (left) / summer (right)."""
    log.info("Fig 2.4 — operational cost panel 1x2")
    season_pairs = [("Winter", "2020-01-06"), ("Summer", "2020-07-06")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    cost_ymax = 0.0
    for _, date in season_pairs:
        for sc in SCENARIOS:
            hs = _load_hourly(date, sc)
            if not hs.empty:
                cost_ymax = max(cost_ymax, hs["VariableCosts"].max() / 1e6 * 1.1)

    for ax, (season, date) in zip(axes, season_pairs):
        for sc in SCENARIOS:
            hs = _load_hourly(date, sc)
            if hs.empty:
                continue
            ax.plot(hs["datetime"], hs["VariableCosts"] / 1e6,
                    color=SCENARIO_COLORS[sc], linewidth=1.2,
                    label=SCENARIO_LABELS[sc], alpha=0.9)
        ax.set_ylim(0, cost_ymax if cost_ymax > 0 else 1)
        ax.set_title(f"{season}  ({date})")
        ax.tick_params(axis="x", rotation=25)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2fM"))

    axes[0].set_ylabel("Operational Cost ($M/hr)")
    axes[1].legend()
    fig.suptitle("Hourly Operational Cost: Winter vs Summer")
    fig.tight_layout()
    _save(fig, out_dir, "fig2_4_lmp_cost_panel")


def fig2_5(out_dir: Path) -> None:
    """Fig 2.5 — Box & whisker: ΔCO2 vs baseline per season × CAS mode."""
    log.info("Fig 2.5 — CO2 delta box plot")
    ws = _load_weekly_summaries()
    if ws.empty:
        return

    co2 = ws[ws["metric"] == "Total CO2"].copy()
    if co2.empty:
        return

    co2["delta_gm"]  = co2["sim-gm"]  - co2["baseline"]
    co2["delta_247"] = co2["sim-247"] - co2["baseline"]
    co2["delta_lp"]  = co2["sim-lp"]  - co2["baseline"]

    fig, ax = plt.subplots(figsize=(12, 6))
    group_w   = 0.6
    box_w     = group_w / 3 * 0.8
    offsets   = [-group_w / 3, 0, group_w / 3]
    modes     = [("sim-gm", "delta_gm"), ("sim-247", "delta_247"), ("sim-lp", "delta_lp")]

    x_ticks = []
    for i, season in enumerate(SEASONS):
        sub = co2[co2["season"] == season]
        for j, (sc, col) in enumerate(modes):
            vals = sub[col].dropna().values * 1000   # kt → t
            if len(vals) == 0:
                continue
            x = i + offsets[j]
            bp = ax.boxplot(vals, positions=[x], widths=box_w,
                            patch_artist=True, notch=False,
                            medianprops=dict(color="black", linewidth=2))
            for patch in bp["boxes"]:
                patch.set_facecolor(SCENARIO_COLORS[sc])
                patch.set_alpha(0.7)
        x_ticks.append(i)

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(SEASONS)
    ax.set_ylabel("ΔCO₂ vs Baseline (metric tonnes/week)")
    ax.set_title("ΔCO₂ vs Baseline by Season and CAS Mode")

    legend_patches = [mpatches.Patch(color=SCENARIO_COLORS[sc],
                                     label=SCENARIO_LABELS[sc], alpha=0.7)
                      for sc, _ in modes]
    ax.legend(handles=legend_patches)
    fig.suptitle("ΔCO₂ vs Baseline: Weekly Distribution by Season and CAS Mode")
    fig.tight_layout()
    _save(fig, out_dir, "fig2_5_co2_delta_boxplot")


# ══════════════════════════════════════════════════════════════════════════════
# §3  SEASONAL FLEXIBILITY PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

def table3_1(out_dir: Path) -> None:
    """Table 3.1 — Monthly CO2 deltas."""
    log.info("Table 3.1 — monthly CO2 deltas")
    ws = _load_weekly_summaries()
    if ws.empty:
        return

    co2 = ws[ws["metric"] == "Total CO2"][
        ["date", "season", "baseline", "sim-gm", "sim-247", "sim-lp"]].copy()
    co2["Δ gm (t)"]  = ((co2["sim-gm"]  - co2["baseline"]) * 1000).round(0)
    co2["Δ 247 (t)"] = ((co2["sim-247"] - co2["baseline"]) * 1000).round(0)
    co2["Δ lp (t)"]  = ((co2["sim-lp"]  - co2["baseline"]) * 1000).round(0)

    display = co2[["date", "season", "baseline", "Δ gm (t)", "Δ 247 (t)", "Δ lp (t)"]].copy()
    display.columns = ["Week", "Season", "Baseline CO₂ (kt)", "Δ Grid-Mix (t)",
                       "Δ 24/7 (t)", "Δ LP (t)"]
    display["Baseline CO₂ (kt)"] = display["Baseline CO₂ (kt)"].round(1)

    fig, ax = plt.subplots(figsize=(13, len(display) * 0.55 + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=display.values, colLabels=display.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.4)

    # Header
    for j in range(len(display.columns)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Color delta cells
    for i, row in enumerate(display.itertuples(), start=1):
        for j, val in enumerate([row[4], row[5], row[6]], start=3):
            if val < 0:
                tbl[(i, j)].set_facecolor("#c8e6c9")
            elif val > 0:
                tbl[(i, j)].set_facecolor("#ffcdd2")

    fig.suptitle("Monthly CO₂ Deltas vs Baseline (12 representative weeks)", y=0.99)
    _save(fig, out_dir, "table3_1_co2_deltas")


def table3_2(out_dir: Path) -> None:
    """Table 3.2 — Monthly operational cost deltas."""
    log.info("Table 3.2 — monthly cost deltas")
    ws = _load_weekly_summaries()
    if ws.empty:
        return

    cost = ws[ws["metric"] == "Operational cost"][
        ["date", "season", "baseline", "sim-gm", "sim-247", "sim-lp"]].copy()
    cost["Δ gm ($k)"]  = ((cost["sim-gm"]  - cost["baseline"]) * 1000).round(0)
    cost["Δ 247 ($k)"] = ((cost["sim-247"] - cost["baseline"]) * 1000).round(0)
    cost["Δ lp ($k)"]  = ((cost["sim-lp"]  - cost["baseline"]) * 1000).round(0)

    display = cost[["date", "season", "baseline",
                    "Δ gm ($k)", "Δ 247 ($k)", "Δ lp ($k)"]].copy()
    display.columns = ["Week", "Season", "Baseline Cost ($M)", "Δ Grid-Mix ($k)",
                       "Δ 24/7 ($k)", "Δ LP ($k)"]
    display["Baseline Cost ($M)"] = display["Baseline Cost ($M)"].round(2)

    fig, ax = plt.subplots(figsize=(13, len(display) * 0.55 + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=display.values, colLabels=display.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.4)
    for j in range(len(display.columns)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(display.itertuples(), start=1):
        for j, val in enumerate([row[4], row[5], row[6]], start=3):
            if val < 0:
                tbl[(i, j)].set_facecolor("#c8e6c9")
            elif val > 0:
                tbl[(i, j)].set_facecolor("#ffcdd2")

    fig.suptitle("Monthly Operational Cost Deltas vs Baseline (12 representative weeks)", y=0.99)
    _save(fig, out_dir, "table3_2_cost_deltas")


def _seasonal_grouped_bar(ax, ws: pd.DataFrame, metric: str,
                          ylabel: str, title: str,
                          scenarios=SCENARIOS) -> None:
    sub = ws[ws["metric"] == metric]
    agg = sub.groupby("season")[scenarios].sum().reindex(SEASONS)
    n = len(scenarios)
    x = np.arange(len(SEASONS))
    w = 0.7 / n
    for j, sc in enumerate(scenarios):
        offset = (j - (n - 1) / 2) * w
        ax.bar(x + offset, agg[sc], width=w * 0.9,
               color=SCENARIO_COLORS[sc], label=SCENARIO_LABELS[sc], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(SEASONS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def fig3_3(out_dir: Path) -> None:
    log.info("Fig 3.3 — seasonal CO2 totals by scenario")
    ws = _load_weekly_summaries()
    if ws.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    _seasonal_grouped_bar(ax, ws, "Total CO2", "CO₂ (kt)",
                          "Seasonal CO₂ Totals by Scenario")
    fig.suptitle("Seasonal CO₂ Totals by CAS Scenario")
    fig.tight_layout()
    _save(fig, out_dir, "fig3_3_seasonal_co2_by_scenario")


def fig3_4(out_dir: Path) -> None:
    log.info("Fig 3.4 — seasonal CO2 reduction %")
    ws = _load_weekly_summaries()
    if ws.empty:
        return

    co2 = ws[ws["metric"] == "Total CO2"]
    agg = co2.groupby("season")[["baseline", "sim-gm", "sim-247", "sim-lp"]].sum().reindex(SEASONS)
    modes = ["sim-gm", "sim-247", "sim-lp"]
    x = np.arange(len(SEASONS))
    w = 0.7 / len(modes)

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, sc in enumerate(modes):
        pct = 100 * (agg[sc] - agg["baseline"]) / agg["baseline"]
        offset = (j - 1) * w
        ax.bar(x + offset, pct, width=w * 0.9,
               color=SCENARIO_COLORS[sc], label=SCENARIO_LABELS[sc], alpha=0.85)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(SEASONS)
    ax.set_ylabel("CO₂ Change vs Baseline (%)")
    ax.set_title("Seasonal CO₂ Reduction vs Baseline (%)")
    ax.legend()
    fig.suptitle("Seasonal CO₂ Change vs Baseline by CAS Mode (%)")
    fig.tight_layout()
    _save(fig, out_dir, "fig3_4_co2_reduction_pct")


def fig3_6(out_dir: Path) -> None:
    log.info("Fig 3.6 — seasonal water by scenario")
    ws = _load_weekly_summaries()
    if ws.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    _seasonal_grouped_bar(ax, ws, "Water withdrawal", "Water Withdrawal (Bgal)",
                          "Seasonal Water Withdrawal by Scenario")
    fig.suptitle("Seasonal Water Withdrawal by CAS Scenario")
    fig.tight_layout()
    _save(fig, out_dir, "fig3_6_seasonal_water_by_scenario")


def fig3_7(out_dir: Path) -> None:
    log.info("Fig 3.7 — seasonal water reduction %")
    ws = _load_weekly_summaries()
    if ws.empty:
        return

    wd = ws[ws["metric"] == "Water withdrawal"]
    agg = wd.groupby("season")[["baseline", "sim-gm", "sim-247", "sim-lp"]].sum().reindex(SEASONS)
    modes = ["sim-gm", "sim-247", "sim-lp"]
    x = np.arange(len(SEASONS))
    w = 0.7 / len(modes)

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, sc in enumerate(modes):
        pct = 100 * (agg[sc] - agg["baseline"]) / agg["baseline"]
        offset = (j - 1) * w
        ax.bar(x + offset, pct, width=w * 0.9,
               color=SCENARIO_COLORS[sc], label=SCENARIO_LABELS[sc], alpha=0.85)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(SEASONS)
    ax.set_ylabel("Water Withdrawal Change vs Baseline (%)")
    ax.set_title("Seasonal Water Withdrawal Reduction vs Baseline (%)")
    ax.legend()
    fig.suptitle("Seasonal Water Withdrawal Change vs Baseline by CAS Mode (%)")
    fig.tight_layout()
    _save(fig, out_dir, "fig3_7_water_reduction_pct")


# ══════════════════════════════════════════════════════════════════════════════
# §5  SPATIAL PRICE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _load_bus_annual_lmps(scenario: str) -> pd.DataFrame:
    """Return mean annual LMP per bus across all 12 months."""
    frames = []
    for _, date, _ in MONTHS:
        path = OUTPUTS_ROOT / date / scenario / "bus_detail.csv"
        if path.exists():
            df = pd.read_csv(path, usecols=["Bus", "LMP"])
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby("Bus")["LMP"].mean().reset_index()


_STATE_BORDERS_CACHE = VATIC_ROOT / "vatic" / "data" / "ne_states_us.geojson"
_STATE_BORDERS_URL   = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_1_states_provinces.geojson"
)


def _draw_state_borders(
    ax: plt.Axes,
    color: str = "#555",
    lw: float = 0.7,
    alpha: float = 0.55,
    zorder: int = 2,
) -> None:
    """Overlay US state borders on a longitude/latitude axes.

    Downloads Natural Earth 110m state polygons on first call and caches them
    at vatic/data/ne_states_us.geojson.  Silently skips if both download and
    cache are unavailable.
    """
    if not _STATE_BORDERS_CACHE.exists():
        try:
            with urllib.request.urlopen(_STATE_BORDERS_URL, timeout=15) as resp:
                raw = json.loads(resp.read().decode())
            us_features = [
                f for f in raw["features"]
                if f["properties"].get("iso_a2") == "US"
            ]
            _STATE_BORDERS_CACHE.write_text(
                json.dumps({"type": "FeatureCollection", "features": us_features})
            )
            log.info("State borders cached → %s", _STATE_BORDERS_CACHE)
        except Exception as exc:
            log.warning("Could not fetch state borders: %s", exc)
            return

    try:
        geojson = json.loads(_STATE_BORDERS_CACHE.read_text())
    except Exception as exc:
        log.warning("Could not read state borders cache: %s", exc)
        return

    for feature in geojson["features"]:
        geom  = feature["geometry"]
        gtype = geom["type"]
        if gtype == "Polygon":
            polys = [geom["coordinates"]]
        elif gtype == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue
        for poly in polys:
            for ring in poly:           # outer ring + holes
                xs = [pt[0] for pt in ring]
                ys = [pt[1] for pt in ring]
                ax.plot(xs, ys, color=color, lw=lw, alpha=alpha,
                        zorder=zorder, solid_capstyle="round")


def _set_map_extent(ax: plt.Axes, geo: pd.DataFrame, pad: float = 0.8) -> None:
    """Lock the viewport to the bus network bounding box + padding degrees."""
    ax.set_xlim(geo["lng"].min() - pad, geo["lng"].max() + pad)
    ax.set_ylim(geo["lat"].min() - pad, geo["lat"].max() + pad)


def _bus_geo() -> pd.DataFrame:
    buses = pd.read_csv(BUS_CSV)
    buses = buses.rename(columns={"Bus Name": "Bus", "lat": "lat", "lng": "lng"})
    return buses[["Bus", "lat", "lng", "Area"]]


def fig5_1(out_dir: Path) -> None:
    """Fig 5.1 — Network map: k-means cluster assignments."""
    log.info("Fig 5.1 — k-means cluster map")
    geo   = _bus_geo()
    lmps  = _load_bus_annual_lmps("baseline")
    if lmps.empty:
        return

    # Build LMP feature matrix per bus per month
    frames = []
    for _, date, _ in MONTHS:
        path = OUTPUTS_ROOT / date / "baseline" / "bus_detail.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["Bus", "Hour", "LMP"])
        hod = df.groupby(["Bus", "Hour"])["LMP"].mean().unstack(fill_value=0)
        hod.columns = [f"h{c}" for c in hod.columns]
        frames.append(hod)

    if not frames:
        return

    # Average HOD LMP profile per bus across all months
    feat = pd.concat(frames).groupby(level=0).mean()
    scaler = StandardScaler()
    X = scaler.fit_transform(feat.values)

    n_clusters = 8
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    cluster_df = pd.DataFrame({"Bus": feat.index, "cluster": labels})

    merged = geo.merge(cluster_df, on="Bus", how="left")
    merged = merged.merge(lmps, on="Bus", how="left")

    cmap = plt.cm.get_cmap("tab10", n_clusters)
    fig, ax = plt.subplots(figsize=(10, 8))
    for c in range(n_clusters):
        sub = merged[merged["cluster"] == c]
        ax.scatter(sub["lng"], sub["lat"], c=[cmap(c)] * len(sub),
                   s=100, zorder=3, label=str(c), edgecolors="white", linewidths=0.4)

    _draw_state_borders(ax, color="black", lw=1.8, alpha=0.9)
    _set_map_extent(ax, geo)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("RTS-GMLC Bus Cluster Assignments (k=8, LMP HOD profiles)")
    ax.legend(title="Cluster", loc="upper right", markerscale=1.0, framealpha=0.9)
    ax.grid(False)
    ax.set_aspect("equal")
    fig.suptitle("Spatial LMP Clusters: Bus Network (k-means, k=8)")
    fig.tight_layout()
    _save(fig, out_dir, "fig5_1_cluster_map")

    # Save cluster assignments for §6
    cluster_df.to_csv(out_dir / "bus_clusters.csv", index=False)
    return cluster_df


def fig5_2(out_dir: Path) -> None:
    """Fig 5.2 — Network map: annual average LMP by bus."""
    log.info("Fig 5.2 — LMP map baseline")
    geo  = _bus_geo()
    lmps = _load_bus_annual_lmps("baseline")
    if lmps.empty:
        return
    merged = geo.merge(lmps, on="Bus", how="left")

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(merged["lng"], merged["lat"], c=merged["LMP"],
                    cmap="RdYlGn_r", s=150, zorder=3,
                    vmin=merged["LMP"].quantile(0.05),
                    vmax=merged["LMP"].quantile(0.95))
    plt.colorbar(sc, ax=ax, label="Annual Avg LMP ($/MWh)")
    for _, row in merged.iterrows():
        ax.annotate(str(row["Bus"]), (row["lng"], row["lat"]),
                    fontsize=5, alpha=0.5, ha="center", va="bottom")
    _draw_state_borders(ax)
    _set_map_extent(ax, merged)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Annual Average LMP by Bus (Baseline)")
    ax.grid(False)
    ax.set_aspect("equal")
    fig.suptitle("Spatial Distribution of Annual Average LMPs (Baseline)")
    fig.tight_layout()
    _save(fig, out_dir, "fig5_2_lmp_map_baseline")


def fig5_3(out_dir: Path) -> None:
    """Fig 5.3 — Network map: annual average LMP delta per CAS mode."""
    log.info("Fig 5.3 — LMP delta maps")
    geo      = _bus_geo()
    lmps_bl  = _load_bus_annual_lmps("baseline")
    if lmps_bl.empty:
        return

    modes = [("sim-gm", "Grid-Mix"), ("sim-247", "24/7"), ("sim-lp", "LP")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, (sc, label) in zip(axes, modes):
        lmps_sc = _load_bus_annual_lmps(sc)
        if lmps_sc.empty:
            ax.text(0.5, 0.5, f"{label}\nno data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        delta = lmps_bl.merge(lmps_sc, on="Bus", suffixes=("_bl", "_sc"))
        delta["ΔLMP"] = delta["LMP_sc"] - delta["LMP_bl"]
        merged = geo.merge(delta[["Bus", "ΔLMP"]], on="Bus", how="left")

        vabs = max(abs(merged["ΔLMP"].min()), abs(merged["ΔLMP"].max()))
        sc_plot = ax.scatter(merged["lng"], merged["lat"], c=merged["ΔLMP"],
                             cmap="RdBu_r", s=150, zorder=3,
                             vmin=-vabs, vmax=vabs)
        plt.colorbar(sc_plot, ax=ax, label="ΔLMP ($/MWh)")
        _draw_state_borders(ax)
        _set_map_extent(ax, geo)
        ax.set_title(f"{label}")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(False)
        ax.set_aspect("equal")

    fig.suptitle("Annual Average LMP Delta vs Baseline by Bus and CAS Mode")
    fig.tight_layout()
    _save(fig, out_dir, "fig5_3_lmp_delta_maps")


def fig5_4(out_dir: Path) -> None:
    """Fig 5.4 — Network map: cluster re-assignment under each CAS mode."""
    log.info("Fig 5.4 — cluster reassignment maps")
    cluster_csv = out_dir / "bus_clusters.csv"
    if not cluster_csv.exists():
        log.warning("Fig 5.4 — bus_clusters.csv not found, run fig5_1 first")
        return

    geo        = _bus_geo()
    base_cl    = pd.read_csv(cluster_csv)
    n_clusters = base_cl["cluster"].nunique()
    cmap       = plt.cm.get_cmap("tab10", n_clusters)

    modes = [("sim-gm", "Grid-Mix"), ("sim-247", "24/7"), ("sim-lp", "LP")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, (sc, label) in zip(axes, modes):
        # Build LMP feature for this scenario
        frames = []
        for _, date, _ in MONTHS:
            path = OUTPUTS_ROOT / date / sc / "bus_detail.csv"
            if not path.exists():
                continue
            df  = pd.read_csv(path, usecols=["Bus", "Hour", "LMP"])
            hod = df.groupby(["Bus", "Hour"])["LMP"].mean().unstack(fill_value=0)
            hod.columns = [f"h{c}" for c in hod.columns]
            frames.append(hod)
        if not frames:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue

        feat    = pd.concat(frames).groupby(level=0).mean()
        scaler  = StandardScaler()
        X       = scaler.fit_transform(feat.values)
        km      = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        new_cl  = pd.DataFrame({"Bus": feat.index, "cluster_new": km.fit_predict(X)})

        merged  = geo.merge(base_cl, on="Bus", how="left")
        merged  = merged.merge(new_cl, on="Bus", how="left")
        merged["changed"] = merged["cluster"] != merged["cluster_new"]

        for c in range(n_clusters):
            sub = merged[merged["cluster_new"] == c]
            ax.scatter(sub["lng"], sub["lat"], c=[cmap(c)] * len(sub),
                       s=100, zorder=3)
        # Ring annotation for changed buses
        changed = merged[merged["changed"] == True]
        ax.scatter(changed["lng"], changed["lat"], s=200, zorder=4,
                   facecolors="none", edgecolors="black", linewidths=1.5)
        _draw_state_borders(ax, color="black", lw=1.8, alpha=0.9)
        _set_map_extent(ax, geo)
        ax.set_title(f"{label}  ({changed.shape[0]} reassigned)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(False)
        ax.set_aspect("equal")

    fig.suptitle("Cluster Re-assignment Under CAS Modes\n(Rings = buses with changed cluster membership)")
    fig.tight_layout()
    _save(fig, out_dir, "fig5_4_cluster_reassignment")


# ══════════════════════════════════════════════════════════════════════════════
# §6  POLICY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

_SUMMER_DATES = {date for _, date, s in MONTHS if s == "Summer"}
_WINTER_DATES = {date for _, date, s in MONTHS if s == "Winter"}

_TIME_PANELS  = [("6am", 6), ("12pm", 12), ("6pm", 18), ("12am", 0)]


def _load_phi_annual(scenario: str) -> pd.DataFrame:
    """Load and concatenate phi_hourly for all 12 months for one scenario."""
    frames = []
    for _, date, _ in MONTHS:
        path = OUTPUTS_ROOT / date / "environmental_score" / f"{scenario}_phi_hourly.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["date_str"] = date
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fig6_season_plot(
    out_dir: Path,
    phi_all: dict[str, pd.DataFrame],
    col: str,
    ylabel: str,
    suptitle_prefix: str,
    fig_id: str,
) -> None:
    """
    Fig 3.5/3.6-style renderer for fig6_1 and fig6_2.

    Layout: rows = scenarios, cols = time snapshots (6am/12pm/6pm/12am).
    Each panel: grey bars = col (left axis), black line+circles = net export
    (right axis, range = 2× left axis range).  Last x-position = System total.
    Produces {fig_id}_summer.png and {fig_id}_winter.png.
    """
    ref_df     = phi_all["baseline"] if "baseline" in phi_all else next(iter(phi_all.values()))
    microgrids = sorted(ref_df["microgrid"].unique())
    x_labels   = [str(m) for m in microgrids] + ["System"]
    x_pos      = np.arange(len(x_labels))

    n_sc  = len(SCENARIOS)
    n_t   = len(_TIME_PANELS)
    # bar colours: grey for microgrids, dark grey for System
    bar_colors = ["#b0b0b0"] * len(microgrids) + ["#444"]

    for season, dates in [("Summer", _SUMMER_DATES), ("Winter", _WINTER_DATES)]:
        # ── Pre-compute per-microgrid hourly means for every scenario ─────────
        # sc_means[sc][h] = {mg: (col_val, export_val)}
        sc_means: dict[str, dict[int, dict]] = {}
        bl_df = phi_all.get("baseline", pd.DataFrame())
        for sc in SCENARIOS:
            df = phi_all.get(sc, pd.DataFrame())
            sc_means[sc] = {}
            for _, h in _TIME_PANELS:
                sub = df[df["date_str"].isin(dates) & (df["hour"] == h)] if not df.empty else pd.DataFrame()
                bl_sub = bl_df[bl_df["date_str"].isin(dates) & (bl_df["hour"] == h)] if not bl_df.empty else pd.DataFrame()
                mg_data = {}
                for mg in microgrids:
                    sc_row = sub[sub["microgrid"] == mg]
                    bl_row = bl_sub[bl_sub["microgrid"] == mg]
                    sc_val  = sc_row[col].mean()           if not sc_row.empty  else 0.0
                    bl_val  = bl_row[col].mean()           if not bl_row.empty  else 0.0
                    sc_exp  = sc_row["net_export_mw"].mean() if not sc_row.empty else 0.0
                    bl_exp  = bl_row["net_export_mw"].mean() if not bl_row.empty else 0.0
                    mg_data[mg] = {
                        "abs":     sc_val,
                        "delta":   sc_val - bl_val,
                        "exp_abs": sc_exp,
                        "exp_delta": sc_exp - bl_exp,
                    }
                sc_means[sc][h] = mg_data

        # ── Determine y-scales: baseline row uses absolute; delta rows use Δ ─
        y_abs_bl   = 0.0   # baseline absolute scale
        y_abs_dl   = 0.0   # delta scale (non-baseline rows)
        y_abs_exp  = 0.0   # export scale (right axis, shared)

        for sc in SCENARIOS:
            is_bl = (sc == "baseline")
            for _, h in _TIME_PANELS:
                for mg, d in sc_means[sc][h].items():
                    if is_bl:
                        y_abs_bl  = max(y_abs_bl,  abs(d["abs"]))
                        y_abs_exp = max(y_abs_exp, abs(d["exp_abs"]))
                    else:
                        y_abs_dl  = max(y_abs_dl,  abs(d["delta"]))
                        y_abs_exp = max(y_abs_exp, abs(d["exp_delta"]))

        y_abs_bl  = (y_abs_bl  or 100.0) * 1.2
        y_abs_dl  = (y_abs_dl  or 50.0)  * 1.2
        y_abs_exp = (y_abs_exp or 100.0) * 1.2

        fig, axes = plt.subplots(
            n_sc, n_t,
            figsize=(4.5 * n_t, 2.8 * n_sc),
            sharex=True,
        )
        if n_sc == 1:
            axes = axes[np.newaxis, :]

        twin_axes = np.empty_like(axes, dtype=object)
        for i in range(n_sc):
            for j in range(n_t):
                twin_axes[i, j] = axes[i, j].twinx()

        for i, sc in enumerate(SCENARIOS):
            is_bl   = (sc == "baseline")
            y_left  = y_abs_bl if is_bl else y_abs_dl
            col_key = "abs"    if is_bl else "delta"
            exp_key = "exp_abs" if is_bl else "exp_delta"

            for j, (time_label, h) in enumerate(_TIME_PANELS):
                ax  = axes[i, j]
                ax2 = twin_axes[i, j]

                mg_data = sc_means[sc][h]
                bar_vals  = [mg_data[mg][col_key]  for mg in microgrids]
                line_vals = [mg_data[mg][exp_key]  for mg in microgrids]
                # System totals
                bar_vals.append(sum(bar_vals))
                line_vals.append(sum(line_vals))

                # ── Bars (left axis) ─────────────────────────────────────────
                ax.bar(x_pos, bar_vals, color=bar_colors, width=0.65,
                       zorder=3, edgecolor="none")
                ax.axhline(0, color="black", lw=0.6, zorder=4)
                ax.set_ylim(-y_left, y_left)
                ax.yaxis.set_major_locator(mticker.MaxNLocator(5, symmetric=True))
                ax.grid(True, axis="y", alpha=0.2, lw=0.5)

                # ── Line (right axis) ─────────────────────────────────────────
                ax2.plot(x_pos, line_vals, "k-o", ms=3.5, lw=1.2, zorder=5)
                ax2.set_ylim(-y_abs_exp, y_abs_exp)
                ax2.yaxis.set_major_locator(mticker.MaxNLocator(5, symmetric=True))

                if j == n_t - 1:
                    ax2.tick_params(axis="y", labelright=True)
                else:
                    ax2.tick_params(axis="y", labelright=False)

                # ── Column header (top row only) ─────────────────────────────
                if i == 0:
                    ax.set_title(time_label, fontweight="bold", pad=4)

                # ── Row label (left column only) ─────────────────────────────
                if j == 0:
                    row_label = (SCENARIO_LABELS[sc] if is_bl
                                 else f"Δ {SCENARIO_LABELS[sc]}\nvs Baseline")
                    ax.set_ylabel(row_label, labelpad=4, rotation=90)

                # ── X-axis tick labels (bottom row only) ─────────────────────
                if i == n_sc - 1:
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(x_labels, rotation=90)

        # ── Right-axis label ──────────────────────────────────────────────────
        mid = n_sc // 2
        twin_axes[mid, -1].set_ylabel("Net Export (MW)", labelpad=6)

        # ── Figure legend ─────────────────────────────────────────────────────
        legend_handles = [
            mpatches.Patch(facecolor="#b0b0b0",
                           label=f"Row 1: {ylabel} (absolute)"),
            mpatches.Patch(facecolor="#b0b0b0", hatch="//",
                           label="Rows 2–4: Δ vs Baseline"),
            Line2D([0], [0], color="black", marker="o", ms=4,
                   label="Net Export MW (right axis)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, 0.0), framealpha=0.95)

        fig.suptitle(f"{suptitle_prefix} — {season}")
        fig.tight_layout(rect=[0.04, 0.05, 1, 1])
        _save(fig, out_dir, f"{fig_id}_{season.lower()}")


def fig6_1(out_dir: Path) -> None:
    """Fig 6.1 — NCS (phi) by cluster × time of day, summer and winter."""
    log.info("Fig 6.1 — NCS by cluster/time-of-day (summer + winter)")

    phi_all = {sc: df for sc in SCENARIOS
               if not (df := _load_phi_annual(sc)).empty}
    if not phi_all:
        return

    _fig6_season_plot(
        out_dir, phi_all,
        col="phi",
        ylabel="Net Carbon Score (φ)",
        suptitle_prefix="Net Carbon Score by Cluster and Time of Day",
        fig_id="fig6_1_ncs_by_cluster_tod",
    )


def fig6_2(out_dir: Path) -> None:
    """Fig 6.2 — Water intensity by cluster × time of day, summer and winter."""
    log.info("Fig 6.2 — water intensity by cluster/time-of-day (summer + winter)")

    phi_all = {sc: df for sc in SCENARIOS
               if not (df := _load_phi_annual(sc)).empty
               and "wd_gal_mwh" in df.columns}
    if not phi_all:
        return

    _fig6_season_plot(
        out_dir, phi_all,
        col="wd_gal_mwh",
        ylabel="Water Withdrawal Intensity (gal/MWh)",
        suptitle_prefix="Water Withdrawal Intensity by Cluster and Time of Day",
        fig_id="fig6_2_water_intensity_by_cluster_tod",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path,
                   default=OUTPUTS_ROOT / "annual_figures")
    p.add_argument("--grid", type=str, default="RTS-GMLC",
                   help="Base grid name (default: RTS-GMLC). Used to resolve gen.csv/bus.csv.")
    return p.parse_args()


def main():
    global GEN_CSV, BUS_CSV
    args    = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    GEN_CSV, BUS_CSV = _grid_source_csvs(args.grid)
    log.info("Loading generator data from %s", GEN_CSV)
    gen_df   = pd.read_csv(GEN_CSV)
    fuel_map = _gen_fuel_map()

    log.info("Output directory: %s", out_dir)

    # §1
    log.info("=== §1 Baseline Grid Characterization ===")
    fig1_1(out_dir, fuel_map)
    fig1_2_table(out_dir, fuel_map, gen_df)
    fig1_2b(out_dir, fuel_map)
    fig1_3(out_dir, fuel_map)
    fig1_4(out_dir)

    # §2
    log.info("=== §2 CAS Algorithm Performance ===")
    table2_1(out_dir)
    fig2_3(out_dir, fuel_map, gen_df)
    fig2_4(out_dir)
    fig2_5(out_dir)

    # §3
    log.info("=== §3 Seasonal Flexibility Patterns ===")
    table3_1(out_dir)
    table3_2(out_dir)
    fig3_3(out_dir)
    fig3_4(out_dir)
    fig3_6(out_dir)
    fig3_7(out_dir)

    # §5
    log.info("=== §5 Spatial Price Analysis ===")
    fig5_1(out_dir)
    fig5_2(out_dir)
    fig5_3(out_dir)
    fig5_4(out_dir)

    # §6
    log.info("=== §6 Policy Analysis ===")
    fig6_1(out_dir)
    fig6_2(out_dir)

    log.info("=== DONE — all figures in %s ===", out_dir)
    # Print manifest
    pngs = sorted(out_dir.glob("*.png"))
    log.info("Generated %d figures:", len(pngs))
    for p in pngs:
        log.info("  %s", p.name)


if __name__ == "__main__":
    main()
