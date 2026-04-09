"""
Average Daily Profile by Season — Baseline + 3 CAS Modes
Texas-7k with Data Center Load | 2 weeks per season, equally weighted

4 rows × 4 columns:
  Rows:    CO2 Emissions | System Average LMP | Renewable Utilisation | Demand
  Columns: Winter | Spring | Summer | Fall

Output: outputs/cas_daily_load_profile_by_season.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO    = Path(__file__).resolve().parents[2]
ANN_DIR = REPO / "outputs" / "TX_2018_ANNUAL"
GEN_CSV = REPO / "vatic/data/grids/Texas-7k-DC-REAL/TX_Data/SourceData/gen.csv"
OUT     = Path("/scratch/gpfs/SIRCAR/dp4800/vatic/outputs/cas_daily_load_profile_by_season.png")

sys.path.insert(0, str(REPO / "scripts"))
import cas as cas_mod  # noqa: E402

# ── Config ─────────────────────────────────────────────────────────────────────
SEASONS = {
    "Winter": ["2018-01-07", "2018-02-18"],
    "Spring": ["2018-04-01", "2018-05-20"],
    "Summer": ["2018-07-01", "2018-08-19"],
    "Fall":   ["2018-10-07", "2018-11-18"],
}

MODES = {
    "baseline": ("Baseline (no CAS)", "#555555", "-",  2.0),
    "sim-gm":   ("Grid-Mix CAS",      "#FF9800", "--", 1.8),
    "sim-247":  ("24/7 CAS",          "#00BCD4", ":",  1.8),
    "sim-lp":   ("LP CAS (α=0.5)",    "#E53935", "-",  2.2),
}

METRICS = [
    ("co2",    r"CO$_2$ Emissions (tonne/hour)"),
    ("lmp",    "System Average LMP ($/MWh)"),
    ("renew",  "Renewable Utilisation\n(% of available)"),
    ("demand", "Demand (GW)"),
]

DAYTIME = (6, 20)   # hours for yellow shading

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "axes.linewidth":    0.8,
})

# ── Emission factors ──────────────────────────────────────────────────────────
_ef_cache: pd.Series | None = None

def _emission_factors() -> pd.Series:
    global _ef_cache
    if _ef_cache is None:
        _ef_cache = cas_mod._emission_factors(GEN_CSV)
    return _ef_cache


# ── Per-week metric loader ────────────────────────────────────────────────────
def load_week_profiles(date: str, mode: str) -> dict[str, pd.Series]:
    """Return hour-of-day series (index 0-23) for all 4 metrics."""
    sim_dir = ANN_DIR / date / mode

    hs = pd.read_csv(sim_dir / "hourly_summary.csv",
                     usecols=["Hour", "Price", "RenewablesUsed",
                               "RenewablesAvailable", "Demand"])

    # LMP
    lmp = hs.groupby("Hour")["Price"].mean()

    # Renewable utilisation
    hu = hs.groupby("Hour")[["RenewablesUsed", "RenewablesAvailable"]].sum()
    renew = (hu["RenewablesUsed"] / hu["RenewablesAvailable"].replace(0, np.nan) * 100
             ).fillna(0.0)

    # Demand (GW)
    demand = hs.groupby("Hour")["Demand"].mean() / 1e3

    # CO2 (tonnes/hour) from thermal_detail — average over days
    td = pd.read_csv(sim_dir / "thermal_detail.csv",
                     usecols=["Date", "Hour", "Generator", "Dispatch"])
    ef = _emission_factors()
    td["co2_t"] = td["Dispatch"] * td["Generator"].map(ef).fillna(0.0) / 1e3
    co2 = td.groupby(["Date", "Hour"])["co2_t"].sum().groupby(level=1).mean()

    return {"co2": co2, "lmp": lmp, "renew": renew, "demand": demand}


# ── Season averages ───────────────────────────────────────────────────────────
def season_mode_profiles(season_dates: list[str], mode: str
                         ) -> dict[str, pd.Series]:
    """Equal-weight average over the season's weeks."""
    weeks = []
    for d in season_dates:
        path = ANN_DIR / d / mode / "hourly_summary.csv"
        if not path.exists():
            print(f"  MISSING: {d}/{mode} — skipping")
            continue
        try:
            weeks.append(load_week_profiles(d, mode))
        except Exception as e:
            print(f"  ERROR {d}/{mode}: {e}")
    if not weeks:
        return {m: pd.Series(np.nan, index=range(24)) for m, _ in METRICS}
    return {
        metric: pd.concat([w[metric] for w in weeks], axis=1).mean(axis=1)
        for metric, _ in METRICS
    }


# ── Build all data ────────────────────────────────────────────────────────────
print("Loading profiles...")
data: dict[str, dict[str, dict[str, pd.Series]]] = {}
for season, dates in SEASONS.items():
    data[season] = {}
    for mode in MODES:
        print(f"  {season} / {mode}")
        data[season][mode] = season_mode_profiles(dates, mode)

# ── Plot ──────────────────────────────────────────────────────────────────────
n_rows = len(METRICS)
n_cols = len(SEASONS)
hours  = np.arange(24)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(16, 13),
    sharex=True,
)

fig.suptitle(
    "Average Daily Profile by Season — Baseline + 3 CAS Modes\n"
    "Texas-7k with Data Center Load  |  2 weeks per season, equally weighted",
    fontsize=11, fontweight="bold", y=1.01,
)

for col_i, season in enumerate(SEASONS):
    axes[0, col_i].set_title(season, fontsize=11, fontweight="bold", pad=6)

    for row_i, (metric_key, metric_label) in enumerate(METRICS):
        ax = axes[row_i, col_i]

        # Daytime shading
        ax.axvspan(DAYTIME[0], DAYTIME[1], color="#FFF9C4", alpha=0.6, zorder=0)

        for mode_key, (label, color, ls, lw) in MODES.items():
            series = data[season][mode_key][metric_key].reindex(hours)
            ax.plot(hours, series.values, color=color, linestyle=ls,
                    linewidth=lw, label=label, zorder=3)

        # Y-axis label only on first column
        if col_i == 0:
            ax.set_ylabel(metric_label, fontsize=8.5)

        ax.set_xlim(0, 23)
        ax.set_xticks([0, 6, 12, 18, 23])

        # X tick labels only on bottom row
        if row_i == n_rows - 1:
            ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "23:00"],
                               fontsize=8, rotation=20, ha="right")
            ax.set_xlabel("Hour of day", fontsize=8.5)
        else:
            ax.set_xticklabels([])

        ax.tick_params(axis="y", labelsize=8)

        # Demand row: GW formatting
        if metric_key == "demand":
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:.0f}")
            )

        # Renew row: percent formatting
        if metric_key == "renew":
            ax.set_ylim(bottom=0)

# ── Legend (below figure) ──────────────────────────────────────────────────────
handles = [
    plt.Line2D([0], [0], color=c, linestyle=ls, linewidth=lw, label=lbl)
    for _, (lbl, c, ls, lw) in MODES.items()
]
fig.legend(handles=handles, loc="lower center", ncol=4,
           fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.03))

plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
