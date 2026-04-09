"""
scripts/study/aggregate_results.py
-----------------------------------
Aggregate CAS_STUDY_RTS_2020 results into CSVs for report generation.

For each run, metrics are read from weekly_summary_{date}.csv when available,
or computed directly from thermal_detail.csv + hourly_summary.csv as fallback.

Outputs → outputs/CAS_STUDY_RTS_2020/analysis/
    results_wide.csv        — one row per run, all metrics as columns
    baseline_summary.csv    — 4-season baseline table (long format)
    sensitivity_summary.csv — sensitivity group/param/date/mode table
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
REPO     = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "outputs" / "CAS_STUDY_RTS_2020"
MANIFEST = REPO / "scripts" / "study" / "manifest.csv"
ANALYSIS = OUT_ROOT / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

GEN_CSV  = REPO / "vatic" / "data" / "grids" / "RTS-GMLC-DC-15PCT" / "RTS_Data" / "SourceData" / "gen.csv"

MODES = ["baseline", "sim-gm", "sim-247", "sim-lp"]
MODE_LABELS = {"baseline": "Baseline", "sim-gm": "GM-CAS",
               "sim-247": "247-CAS", "sim-lp": "LP-CAS"}

# Fallback CO2 factors (kg/MWh) for RTS-GMLC fuel types
_CO2_KG_MWH = {
    "Coal": 1001.0, "NG": 469.0, "Oil": 840.0,
    "Nuclear": 0.0, "Hydro": 0.0, "Wind": 0.0,
    "Solar": 0.0, "Sync_Cond": 0.0, "Storage": 0.0,
}

# ---------------------------------------------------------------------------
# Emission factors
# ---------------------------------------------------------------------------

def _load_ef() -> dict[str, float]:
    """Return {gen_uid: kg_CO2/MWh}."""
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

# ---------------------------------------------------------------------------
# Metrics from a single sim directory
# ---------------------------------------------------------------------------

def _lp_sim_dir(run_dir: Path) -> Path:
    """Return best LP sim dir: prefer sim-lp, fall back to last non-empty joint-i* dir."""
    if (run_dir / "sim-lp").exists():
        return run_dir / "sim-lp"
    # Try joint iterations from last to first, skip dirs with empty thermal_detail
    iters = sorted(run_dir.glob("sim-lp-joint-i*"), reverse=True)
    for d in iters:
        td = d / "thermal_detail.csv"
        if td.exists() and td.stat().st_size > 0:
            return d
    return run_dir / "sim-lp"  # will report as missing


def metrics_from_dir(sim_dir: Path) -> dict | None:
    td_path = sim_dir / "thermal_detail.csv"
    hs_path = sim_dir / "hourly_summary.csv"
    if (not td_path.exists() or td_path.stat().st_size == 0 or
            not hs_path.exists() or hs_path.stat().st_size == 0):
        return None

    td = pd.read_csv(td_path)
    hs = pd.read_csv(hs_path)

    co2_kg = sum(EF.get(g, 0.0) * d
                 for g, d in zip(td["Generator"], td["Dispatch"]))

    cost_usd = float((hs["FixedCosts"] + hs["VariableCosts"]).sum())
    load_shed = float(hs["LoadShedding"].sum())
    renew_used = float(hs["RenewablesUsed"].sum())
    renew_avail = float(hs["RenewablesAvailable"].sum())
    curtail_gwh = float(hs["RenewablesCurtailment"].sum()) / 1e3
    demand = float(hs["Demand"].sum())
    ci = co2_kg / demand if demand > 0 else 0.0

    return {
        "co2_kt":       round(co2_kg / 1e6, 4),
        "cost_musd":    round(cost_usd / 1e6, 4),
        "load_shed_mwh": round(load_shed, 2),
        "renew_used_gwh": round(renew_used / 1e3, 4),
        "curtail_gwh":  round(curtail_gwh, 4),
        "ci_kg_mwh":    round(ci, 4),
    }


def load_run(run_dir: Path) -> dict[str, dict]:
    """Load metrics for all 4 modes from a run directory."""
    result = {}
    for mode in ["baseline", "sim-gm", "sim-247"]:
        m = metrics_from_dir(run_dir / mode)
        result[mode] = m
    # LP: prefer sim-lp, fall back to last joint iteration
    lp_dir = _lp_sim_dir(run_dir)
    result["sim-lp"] = metrics_from_dir(lp_dir)
    return result


def reduction_pct(val, baseline) -> float | None:
    if val is None or baseline is None or baseline == 0:
        return None
    return round(100 * (baseline - val) / abs(baseline), 3)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def run_dir_for(phase: str, run_label: str, date: str) -> Path:
    if phase == "baseline":
        return OUT_ROOT / "baseline" / date
    # sensitivity run_label: "alpha/alpha_0.00/2020-01"
    # → sensitivity/alpha/alpha_0.00/2020-01-01
    parts = run_label.split("/")  # [group, param, month_label]
    return OUT_ROOT / "sensitivity" / parts[0] / parts[1] / date


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    rows = []

    for _, run in manifest.iterrows():
        phase     = run["phase"]
        group     = run["group"]
        param     = run["param_label"]
        run_label = run["run_label"]
        date      = str(run["date"])

        rdir = run_dir_for(phase, run_label, date)

        # Try weekly_summary first (fast, pre-computed)
        ws_path = rdir / f"weekly_summary_{date}.csv"
        if ws_path.exists():
            ws  = pd.read_csv(ws_path)
            ws  = ws[ws["metric"].isin(["Total CO2", "Operational cost",
                                        "Water withdrawal", "Load shedding",
                                        "Renewables curtailed", "Renewables used",
                                        "Carbon intensity"])]
            met = {}
            for _, r in ws.iterrows():
                for mode in MODES:
                    if mode in ws.columns:
                        met[f"{r['metric']}__{mode}"] = r.get(mode, np.nan)
            source = "weekly_summary"
        else:
            # Fall back to raw sim dirs
            raw = load_run(rdir)
            missing = [m for m, v in raw.items() if v is None]
            if missing:
                print(f"  [PARTIAL] {rdir.relative_to(REPO)} — missing: {missing}",
                      file=sys.stderr)
            if all(v is None for v in raw.values()):
                print(f"  [SKIP]    {rdir.relative_to(REPO)}", file=sys.stderr)
                continue
            met = {}
            key_map = {
                "Total CO2":           "co2_kt",
                "Operational cost":    "cost_musd",
                "Load shedding":       "load_shed_mwh",
                "Renewables used":     "renew_used_gwh",
                "Renewables curtailed": "curtail_gwh",
                "Carbon intensity":    "ci_kg_mwh",
            }
            for mode in MODES:
                m = raw.get(mode)
                for label, col in key_map.items():
                    met[f"{label}__{mode}"] = m[col] if m else np.nan
            source = "computed"

        row = {
            "phase": phase, "group": group, "param": param,
            "run_label": run_label, "date": date,
            "flex_pct": run["flex_pct"], "lp_alpha": run["lp_alpha"],
            "deferral_h": run["deferral_h"], "max_iter": run["max_iter"],
            "amplitude_mw": run["amplitude_mw"], "source": source,
        }
        row.update(met)

        # Reduction % vs baseline mode
        for label in ["Total CO2", "Operational cost", "Load shedding",
                      "Renewables curtailed", "Carbon intensity"]:
            bl = met.get(f"{label}__baseline", np.nan)
            for mode in ["sim-gm", "sim-247", "sim-lp"]:
                val = met.get(f"{label}__{mode}", np.nan)
                row[f"{label}__{mode}__red_pct"] = reduction_pct(
                    val if not np.isnan(val) else None,
                    bl  if not np.isnan(bl)  else None,
                )

        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(ANALYSIS / "results_wide.csv", index=False)
    print(f"\nWrote {len(results)} rows → {ANALYSIS / 'results_wide.csv'}")

    # ------------------------------------------------------------------
    # Baseline summary (long format)
    # ------------------------------------------------------------------
    bl = results[results["phase"] == "baseline"].copy()
    bl["season"] = bl["date"].map({
        "2020-01-01": "Winter (Jan)",
        "2020-04-01": "Spring (Apr)",
        "2020-07-01": "Summer (Jul)",
        "2020-10-01": "Fall  (Oct)",
    })
    bl_rows = []
    for _, r in bl.iterrows():
        for mode in ["sim-gm", "sim-247", "sim-lp"]:
            for label in ["Total CO2", "Operational cost", "Carbon intensity",
                          "Load shedding", "Renewables curtailed"]:
                bl_rows.append({
                    "season":   r["season"],
                    "mode":     MODE_LABELS[mode],
                    "metric":   label,
                    "baseline": r.get(f"{label}__baseline", np.nan),
                    "value":    r.get(f"{label}__{mode}", np.nan),
                    "red_pct":  r.get(f"{label}__{mode}__red_pct", np.nan),
                })
    pd.DataFrame(bl_rows).to_csv(ANALYSIS / "baseline_summary.csv", index=False)
    print(f"Wrote → {ANALYSIS / 'baseline_summary.csv'}")

    # ------------------------------------------------------------------
    # Sensitivity summary
    # ------------------------------------------------------------------
    sens = results[results["phase"] == "sensitivity"].copy()
    sens_rows = []
    for _, r in sens.iterrows():
        for mode in ["sim-gm", "sim-247", "sim-lp"]:
            sens_rows.append({
                "group":        r["group"],
                "param":        r["param"],
                "date":         r["date"],
                "mode":         MODE_LABELS[mode],
                "flex_pct":     r["flex_pct"],
                "lp_alpha":     r["lp_alpha"],
                "deferral_h":   r["deferral_h"],
                "amplitude_mw": r["amplitude_mw"],
                "co2_red_pct":  r.get("Total CO2__sim-lp__red_pct" if mode == "sim-lp"
                                      else f"Total CO2__{mode}__red_pct", np.nan),
                "cost_red_pct": r.get(f"Operational cost__{mode}__red_pct", np.nan),
                "co2_kt":       r.get(f"Total CO2__{mode}", np.nan),
                "bl_co2_kt":    r.get("Total CO2__baseline", np.nan),
            })
    pd.DataFrame(sens_rows).to_csv(ANALYSIS / "sensitivity_summary.csv", index=False)
    print(f"Wrote → {ANALYSIS / 'sensitivity_summary.csv'}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n=== Baseline CO2 reduction (%) by season and mode ===")
    bl_df = pd.DataFrame(bl_rows)
    pivot = bl_df[bl_df["metric"] == "Total CO2"].pivot_table(
        index="season", columns="mode", values="red_pct"
    )
    print(pivot.round(3).to_string())

    print("\n=== Sensitivity LP-CAS CO2 reduction (%) ===")
    ss = pd.DataFrame(sens_rows)
    lp = ss[ss["mode"] == "LP-CAS"]
    print(lp.groupby(["group", "param", "date"])["co2_red_pct"]
          .first().round(3).to_string())


if __name__ == "__main__":
    main()
