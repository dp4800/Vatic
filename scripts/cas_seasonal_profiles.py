#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""Average daily profile by season — Baseline + 3 CAS Modes.

Thesis: Figure 11 — "CAS Mode by Fuel Mix."

5 rows × 4 columns:
  Rows:    CO₂ Emissions | System LMP | Renewable Utilisation
           | Renewable Curtailment | Load Shedding
  Columns: Winter | Spring | Summer | Fall

Uses all 24 biweekly weeks (6 per season) from TX_2018_ANNUAL.

Usage:
    module load anaconda3/2024.10
    python scripts/cas_seasonal_profiles.py [--study-dir PATH]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import cas as cas_mod  # noqa: E402

from constants import CAS_MODES, BASELINE_STYLE, SEASONS

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

GEN_CSV = REPO / "vatic/data/grids/Texas-7k-DC-REAL/TX_Data/SourceData/gen.csv"

MODES = {"baseline": BASELINE_STYLE, **CAS_MODES}

METRICS = [
    {"key": "co2",        "ylabel": r"CO$_2$ (tonnes/hour)"},
    {"key": "lmp",        "ylabel": "System LMP\n($/MWh)"},
    {"key": "renew_util", "ylabel": "Renewable\nUtilisation (%)"},
    {"key": "curtail",    "ylabel": "Renewable\nCurtailment (MW)"},
    {"key": "loadshed",   "ylabel": "Load Shedding\n(MW)"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Emission factors (cached)
# ═══════════════════════════════════════════════════════════════════════════════

_ef_cache: pd.Series | None = None

def _emission_factors() -> pd.Series:
    global _ef_cache
    if _ef_cache is None:
        _ef_cache = cas_mod._emission_factors(GEN_CSV)
    return _ef_cache


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_week_profiles(study: Path, date: str, mode: str
                       ) -> dict[str, pd.Series]:
    """Return hour-of-day series (index 0–23) for all metrics."""
    sim_dir = study / date / mode

    hs = pd.read_csv(sim_dir / "hourly_summary.csv",
                     usecols=["Hour", "Price", "RenewablesUsed",
                              "RenewablesAvailable", "RenewablesCurtailment",
                              "LoadShedding"])

    # LMP
    lmp = hs.groupby("Hour")["Price"].mean()

    # Renewable utilisation (%)
    hu = hs.groupby("Hour")[["RenewablesUsed", "RenewablesAvailable"]].sum()
    renew_util = (hu["RenewablesUsed"]
                  / hu["RenewablesAvailable"].replace(0, np.nan) * 100
                  ).fillna(0.0)

    # Curtailment (MW)
    curtail = hs.groupby("Hour")["RenewablesCurtailment"].mean()

    # Load shedding (MW)
    loadshed = hs.groupby("Hour")["LoadShedding"].mean()

    # CO₂ (tonnes/hour) from thermal_detail
    td = pd.read_csv(sim_dir / "thermal_detail.csv",
                     usecols=["Date", "Hour", "Generator", "Dispatch"])
    ef = _emission_factors()
    td["co2_t"] = td["Dispatch"] * td["Generator"].map(ef).fillna(0.0) / 1e3
    co2 = td.groupby(["Date", "Hour"])["co2_t"].sum().groupby(level=1).mean()

    return {"co2": co2, "lmp": lmp, "renew_util": renew_util,
            "curtail": curtail, "loadshed": loadshed}


def season_profiles(study: Path, season_dates: list[str], mode: str
                    ) -> dict[str, pd.Series]:
    """Equal-weight average over the season's weeks."""
    weeks = []
    for d in season_dates:
        path = study / d / mode / "hourly_summary.csv"
        if not path.exists():
            print(f"  MISSING: {d}/{mode} — skipping")
            continue
        try:
            weeks.append(load_week_profiles(study, d, mode))
        except Exception as e:
            print(f"  ERROR {d}/{mode}: {e}")
    if not weeks:
        return {m["key"]: pd.Series(np.nan, index=range(24)) for m in METRICS}
    return {
        m["key"]: pd.concat([w[m["key"]] for w in weeks], axis=1).mean(axis=1)
        for m in METRICS
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_profiles(study: Path, out_path: Path):
    print("Loading profiles...")
    data: dict[str, dict[str, dict[str, pd.Series]]] = {}
    for season, dates in SEASONS.items():
        data[season] = {}
        for mode in MODES:
            print(f"  {season} / {mode}")
            data[season][mode] = season_profiles(study, dates, mode)

    hours = np.arange(24)
    hours_fine = np.linspace(0, 23, 240)
    n_rows = len(METRICS)
    n_cols = len(SEASONS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 16), sharex=True)

    for col_i, season in enumerate(SEASONS):
        axes[0, col_i].set_title(season, fontsize=13, fontweight="bold", pad=8)

        for row_i, metric in enumerate(METRICS):
            ax = axes[row_i, col_i]
            key = metric["key"]

            for mode_key, style in MODES.items():
                series = data[season][mode_key][key].reindex(hours)
                vals = series.values.astype(float)
                vals_fine = np.interp(hours_fine, hours, vals)

                ax.plot(hours_fine, vals_fine,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        label=style["label"],
                        zorder=3)

            ax.grid(True, alpha=0.3)

            if col_i == 0:
                ax.set_ylabel(metric["ylabel"], fontsize=10)

            if key == "renew_util":
                ax.set_ylim(bottom=0, top=105)

            if key in ("curtail", "loadshed"):
                ax.set_ylim(bottom=0)

            if row_i == n_rows - 1:
                ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
                ax.set_xticklabels(["12am", "4am", "8am", "12pm",
                                     "4pm", "8pm", "11pm"], fontsize=8)
                ax.set_xlabel("Hour of Day", fontsize=9)
            else:
                ax.set_xticklabels([])

    # Single shared legend at the bottom
    handles = [
        plt.Line2D([0], [0], color=s["color"], linestyle=s["linestyle"],
                    linewidth=s["linewidth"], label=s["label"])
        for s in MODES.values()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    n_weeks = sum(len(v) for v in SEASONS.values())
    fig.suptitle(
        "Average Daily Profile by Season — Baseline + 3 CAS Modes\n"
        f"Texas-7k DC Grid  |  {n_weeks} biweekly weeks "
        f"({n_weeks // 4} per season)",
        fontsize=14, fontweight="bold", y=1.01)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Seasonal daily profiles: Baseline + 3 CAS modes")
    parser.add_argument("--study-dir",
                        default="outputs/TX_2018_ANNUAL",
                        help="Path to TX_2018_ANNUAL output directory")
    parser.add_argument("--output",
                        default="outputs/TX_2018_ANNUAL/"
                                "cas_seasonal_profiles.png",
                        help="Output image path")
    args = parser.parse_args()

    plot_profiles(Path(args.study_dir), Path(args.output))


if __name__ == "__main__":
    main()
