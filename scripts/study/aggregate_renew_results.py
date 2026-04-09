"""
scripts/study/aggregate_renew_results.py
-----------------------------------------
Aggregate RENEW_STUDY_RTS_2020 results into analysis CSVs.

Reads metrics from each completed sim dir and produces:

  outputs/RENEW_STUDY_RTS_2020/analysis/
    renew_results_wide.csv   — one row per run × mode, all metrics
    renew_summary.csv        — CO₂/cost reduction vs baseline, by group/label/date/mode
    pareto_lp.csv            — LP-CAS (cost_red_pct, co2_red_pct) per portfolio/date

Run after all renew jobs complete:
    module load anaconda3/2024.10
    python scripts/study/aggregate_renew_results.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
REPO      = Path(__file__).resolve().parents[2]
OUT_ROOT  = REPO / "outputs" / "RENEW_STUDY_RTS_2020"
MANIFEST  = REPO / "scripts" / "study" / "renew_manifest.csv"
ANALYSIS  = OUT_ROOT / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

# Use base 15PCT grid for emission factors (new wind/solar/storage → 0 CO₂)
GEN_CSV = REPO / "vatic" / "data" / "grids" / "RTS-GMLC-DC-15PCT" \
        / "RTS_Data" / "SourceData" / "gen.csv"

MODES       = ["baseline", "sim-gm", "sim-lp"]
MODE_LABELS = {"baseline": "Baseline", "sim-gm": "GM-CAS", "sim-lp": "LP-CAS"}

_CO2_KG_MWH = {
    "Coal": 1078.5, "NG": 496.3, "Oil": 795.8,
    "Nuclear": 0.0, "Hydro": 0.0, "Wind": 0.0,
    "Solar": 0.0, "Sync_Cond": 0.0, "Storage": 0.0,
}

# ---------------------------------------------------------------------------

def _load_ef() -> dict[str, float]:
    if not GEN_CSV.exists():
        return {}
    gen = pd.read_csv(GEN_CSV)
    uid_col  = "GEN UID" if "GEN UID" in gen.columns else gen.columns[0]
    fuel_col = next((c for c in gen.columns if "fuel" in c.lower()), None)
    hr_col   = "HR_avg_0" if "HR_avg_0" in gen.columns else None
    co2_col  = next((c for c in gen.columns if "CO2" in c and "Lbs" in c), None)
    ef = {}
    for _, row in gen.iterrows():
        uid = row[uid_col]
        if hr_col and co2_col and pd.notna(row.get(hr_col)) and pd.notna(row.get(co2_col)):
            # HR [BTU/kWh] / 1000 → MMBTU/MWh; × CO2 [lbs/MMBTU] × 0.453592 kg/lb → kg CO2/MWh
            ef[uid] = float(row[hr_col]) / 1000.0 * float(row[co2_col]) * 0.453592
        else:
            fuel = str(row.get(fuel_col, "")).strip() if fuel_col else ""
            ef[uid] = next((v for k, v in _CO2_KG_MWH.items() if k.lower() in fuel.lower()), 0.0)
    return ef


EF = _load_ef()


def _best_lp_dir(run_dir: Path) -> Path:
    """Return the best LP sim dir (prefer sim-lp, fall back to last non-empty joint-i*)."""
    if (run_dir / "sim-lp").is_dir():
        return run_dir / "sim-lp"
    iters = sorted(run_dir.glob("sim-lp-joint-i*"), reverse=True)
    for d in iters:
        td = d / "thermal_detail.csv"
        if td.exists() and td.stat().st_size > 0:
            return d
    return run_dir / "sim-lp"


def metrics_from_dir(sim_dir: Path) -> dict | None:
    td_path = sim_dir / "thermal_detail.csv"
    hs_path = sim_dir / "hourly_summary.csv"
    if (not td_path.exists() or td_path.stat().st_size == 0
            or not hs_path.exists() or hs_path.stat().st_size == 0):
        return None
    td = pd.read_csv(td_path)
    hs = pd.read_csv(hs_path)

    co2_kg = sum(EF.get(g, 0.0) * d
                 for g, d in zip(td["Generator"], td["Dispatch"]))
    cost_usd    = float((hs["FixedCosts"] + hs["VariableCosts"]).sum())
    load_shed   = float(hs["LoadShedding"].sum())
    renew_used  = float(hs["RenewablesUsed"].sum())
    curtail     = float(hs["RenewablesCurtailment"].sum()) / 1e3
    demand      = float(hs["Demand"].sum())
    ci          = co2_kg / demand if demand > 0 else 0.0

    return {
        "co2_kt":          round(co2_kg / 1e6, 4),
        "cost_musd":       round(cost_usd / 1e6, 4),
        "load_shed_mwh":   round(load_shed, 2),
        "renew_used_gwh":  round(renew_used / 1e3, 4),
        "curtail_gwh":     round(curtail, 4),
        "ci_kg_mwh":       round(ci, 4),
    }


def pct_change(new, base) -> float | None:
    if new is None or base is None or base == 0:
        return None
    return round(100 * (new - base) / abs(base), 3)


def reduction_pct(val, base) -> float | None:
    """Positive = reduction vs baseline."""
    if val is None or base is None or base == 0:
        return None
    return round(100 * (base - val) / abs(base), 3)


# ---------------------------------------------------------------------------

def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    wide_rows  = []
    short_rows = []

    for _, run in manifest.iterrows():
        group   = run["group"]
        label   = run["label"]
        date    = str(run["date"])
        wind_mw = run["wind_mw"]
        sol_mw  = run["solar_mw"]
        bat_mwh = run["battery_mwh"]
        dc_grid = run["dc_grid"]

        # Sim dir: out_root / date / mode
        # out_root was set as "outputs/RENEW_STUDY_RTS_2020/{group}/{label}/{date}"
        run_dir = OUT_ROOT / group / label / date / date

        raw = {}
        for mode in ["baseline", "sim-gm"]:
            raw[mode] = metrics_from_dir(run_dir / mode)
        raw["sim-lp"] = metrics_from_dir(_best_lp_dir(run_dir))

        missing = [m for m, v in raw.items() if v is None]
        if missing:
            print(f"  [PARTIAL] {group}/{label}/{date} — missing: {missing}",
                  file=sys.stderr)
        if all(v is None for v in raw.values()):
            print(f"  [SKIP]    {group}/{label}/{date}", file=sys.stderr)
            continue

        bl = raw.get("baseline") or {}

        for mode in MODES:
            m = raw.get(mode) or {}
            row = {
                "group":      group,
                "label":      label,
                "date":       date,
                "mode":       MODE_LABELS[mode],
                "wind_mw":    wind_mw,
                "solar_mw":   sol_mw,
                "battery_mwh": bat_mwh,
                "renew_cap_mw": wind_mw + sol_mw,
                "dc_grid":    dc_grid,
            }
            for metric in ["co2_kt", "cost_musd", "load_shed_mwh",
                           "renew_used_gwh", "curtail_gwh", "ci_kg_mwh"]:
                row[metric] = m.get(metric, np.nan)
            wide_rows.append(row)

        # Summary: reduction vs baseline for GM and LP
        for mode in ["sim-gm", "sim-lp"]:
            m = raw.get(mode) or {}
            short_rows.append({
                "group":       group,
                "label":       label,
                "date":        date,
                "mode":        MODE_LABELS[mode],
                "wind_mw":     wind_mw,
                "solar_mw":    sol_mw,
                "battery_mwh": bat_mwh,
                "renew_cap_mw": wind_mw + sol_mw,
                "co2_kt_bl":   bl.get("co2_kt", np.nan),
                "co2_kt":      m.get("co2_kt", np.nan),
                "cost_musd_bl": bl.get("cost_musd", np.nan),
                "cost_musd":   m.get("cost_musd", np.nan),
                "curtail_gwh": m.get("curtail_gwh", np.nan),
                "co2_red_pct": reduction_pct(m.get("co2_kt"), bl.get("co2_kt")),
                "cost_red_pct": reduction_pct(m.get("cost_musd"), bl.get("cost_musd")),
                "ci_red_pct":  reduction_pct(m.get("ci_kg_mwh"), bl.get("ci_kg_mwh")),
            })

    wide = pd.DataFrame(wide_rows)
    wide.to_csv(ANALYSIS / "renew_results_wide.csv", index=False)
    print(f"Wrote {len(wide)} rows → {ANALYSIS / 'renew_results_wide.csv'}")

    summary = pd.DataFrame(short_rows)
    summary.to_csv(ANALYSIS / "renew_summary.csv", index=False)
    print(f"Wrote {len(summary)} rows → {ANALYSIS / 'renew_summary.csv'}")

    # Pareto CSV: LP-CAS (cost_red_pct, co2_red_pct) per portfolio/date
    pareto = summary[summary["mode"] == "LP-CAS"][
        ["group", "label", "date", "wind_mw", "solar_mw", "battery_mwh",
         "renew_cap_mw", "co2_red_pct", "cost_red_pct", "curtail_gwh"]
    ].copy()
    pareto.to_csv(ANALYSIS / "pareto_lp.csv", index=False)
    print(f"Wrote {len(pareto)} rows → {ANALYSIS / 'pareto_lp.csv'}")

    # Console summary
    print("\n=== LP-CAS CO₂ reduction (%) by group and renewable capacity ===")
    lp = summary[summary["mode"] == "LP-CAS"].copy()
    pivot = lp.pivot_table(
        index=["group", "label", "wind_mw", "solar_mw", "battery_mwh"],
        columns="date",
        values="co2_red_pct",
        aggfunc="first"
    )
    print(pivot.round(2).to_string())

    print("\n=== GM-CAS CO₂ reduction (%) by group ===")
    gm = summary[summary["mode"] == "GM-CAS"].copy()
    pivot_gm = gm.pivot_table(
        index=["group", "label"],
        columns="date",
        values="co2_red_pct",
        aggfunc="first"
    )
    print(pivot_gm.round(2).to_string())


if __name__ == "__main__":
    main()
