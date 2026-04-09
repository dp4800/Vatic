#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
pareto_eval_scale_nocas.py — Evaluate one portfolio row WITHOUT LP CAS,
using the scaling-based investment approach.

Runs VATIC on the raw scaled investment grid (no CAS load shift) for
each representative week, then compares metrics vs sim-lp baseline.

Usage (called by SLURM array task N):
    python scripts/pareto_eval_scale_nocas.py \\
        --row N \\
        --portfolios scripts/study/pareto_portfolios_scale.csv \\
        --config scripts/study/params_pareto_scale_p1.json

Result JSON: {out_root}/results_nocas/pareto_{NNN}_{hash8}.json

Run this alongside pareto_eval_scale.py to decompose:
    baseline → SCALE-only → SCALE+CAS
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import renew_invest as ri              # noqa: E402
import apply_cas_shift as _acs        # noqa: E402
import scale_investment as si          # noqa: E402


# ── Argument parsing ────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--row",        type=int,  required=True)
    p.add_argument("--portfolios", type=Path,
                   default=REPO / "scripts/study/pareto_portfolios_scale.csv")
    p.add_argument("--config",     type=Path,
                   default=REPO / "scripts/study/params_pareto_scale.json")
    return p.parse_args()


# ── Metric aggregation ─────────────────────────────────────────────────────────

def _aggregate_metrics(per_week: list[dict]) -> dict:
    total_gen = sum(m["total_generation_mwh"] for m in per_week)
    total_co2 = sum(m["total_co2_tonnes"]     for m in per_week)
    return {
        "total_co2_tonnes":         round(total_co2, 1),
        "total_cost_usd":           round(sum(m["total_cost_usd"]           for m in per_week), 0),
        "variable_cost_usd":        round(sum(m["variable_cost_usd"]        for m in per_week), 0),
        "load_shedding_mwh":        round(sum(m["load_shedding_mwh"]        for m in per_week), 2),
        "renewables_used_mwh":      round(sum(m["renewables_used_mwh"]      for m in per_week), 1),
        "renewables_curtailed_mwh": round(sum(m["renewables_curtailed_mwh"] for m in per_week), 1),
        "total_generation_mwh":     round(total_gen, 1),
        "mean_ci_kgco2_mwh":        round(
            total_co2 * 1000 / total_gen if total_gen > 0 else 0.0, 3
        ),
        "avg_lmp_usd_mwh":          round(
            sum(m["avg_lmp_usd_mwh"] * m["total_generation_mwh"] for m in per_week)
            / max(total_gen, 1.0), 3
        ),
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse()

    df = pd.read_csv(args.portfolios)
    if args.row >= len(df):
        sys.exit(f"Row {args.row} out of range (0–{len(df)-1})")

    row = df.iloc[args.row]
    sv = si.ScaleVector(
        wind_scale  = float(row["wind_scale"]),
        solar_scale = float(row["solar_scale"]),
        battery_mw  = float(row["battery_mw"]),
    )
    h = sv.vector_hash()
    print(f"Portfolio {args.row}: wind×{sv.wind_scale} solar×{sv.solar_scale} "
          f"battery={sv.battery_mw:.0f} MW  hash={h}")

    with open(args.config) as f:
        cfg = json.load(f)

    sim_cfg   = cfg["simulation"]
    grid_cfg  = cfg["grid"]
    capex_cfg = cfg.get("capex", {})
    out_root  = REPO / cfg["out_root"]
    result_subdir = cfg.get("result_nocas_subdir", "results_nocas")
    out_root.mkdir(parents=True, exist_ok=True)

    result_path = out_root / result_subdir / f"pareto_{args.row:03d}_{h}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        with open(result_path) as f:
            r = json.load(f)
        if r.get("status") == "ok":
            print(f"Already complete — skipping ({result_path})")
            return

    dc_grid  = grid_cfg["dc_grid"]
    sim_days = int(sim_cfg["days"])

    sim_params = {
        "solver":               sim_cfg["solver"],
        "solver_args":          sim_cfg["solver_args"],
        "threads":              sim_cfg["threads"],
        "ruc_horizon":          sim_cfg["ruc_horizon"],
        "sced_horizon":         sim_cfg["sced_horizon"],
        "ruc_mipgap":           sim_cfg["ruc_mipgap"],
        "reserve_factor":       sim_cfg["reserve_factor"],
        "output_detail":        sim_cfg["output_detail"],
        "thermal_rating_scale": sim_cfg.get("thermal_rating_scale", 1.0),
    }

    multi_week_configs_raw = cfg.get("multi_week_dates", [])
    multi_week_configs     = [(d, REPO / b) for d, b in multi_week_configs_raw]

    scaled_grid = f"{dc_grid}-SCALE-{h}"

    _grid_data_dir = _acs._registry_cfg(dc_grid)["data_dir"]
    gen_csv        = ri.GRIDS_DIR / dc_grid     / _grid_data_dir / "SourceData" / "gen.csv"
    scaled_gen_csv = ri.GRIDS_DIR / scaled_grid / _grid_data_dir / "SourceData" / "gen.csv"

    # Baseline wind/solar totals (for CAPEX on added MW)
    src_gen_df        = pd.read_csv(gen_csv)
    baseline_wind_mw  = float(src_gen_df[src_gen_df["Fuel"] == "WND (Wind)"]["PMax MW"].sum())
    baseline_solar_mw = float(src_gen_df[src_gen_df["Fuel"] == "SUN (Solar)"]["PMax MW"].sum())

    per_week_baselines = []
    multi_week_dates   = []
    for wdate, bdir in multi_week_configs:
        per_week_baselines.append(ri._extract_metrics(bdir, gen_csv))
        multi_week_dates.append(wdate)

    total_gen_b = sum(m["total_generation_mwh"] for m in per_week_baselines)
    total_co2_b = sum(m["total_co2_tonnes"]     for m in per_week_baselines)
    baseline_metrics = {
        "total_co2_tonnes":         round(total_co2_b, 1),
        "total_cost_usd":           round(sum(m["total_cost_usd"]    for m in per_week_baselines), 0),
        "variable_cost_usd":        round(sum(m["variable_cost_usd"] for m in per_week_baselines), 0),
        "load_shedding_mwh":        round(sum(m["load_shedding_mwh"] for m in per_week_baselines), 2),
        "renewables_used_mwh":      round(sum(m["renewables_used_mwh"]      for m in per_week_baselines), 1),
        "renewables_curtailed_mwh": round(sum(m["renewables_curtailed_mwh"] for m in per_week_baselines), 1),
        "total_generation_mwh":     round(total_gen_b, 1),
        "mean_ci_kgco2_mwh":        round(
            total_co2_b * 1000 / total_gen_b if total_gen_b > 0 else 0.0, 3
        ),
        "avg_lmp_usd_mwh":          round(
            sum(m["avg_lmp_usd_mwh"] * m["total_generation_mwh"]
                for m in per_week_baselines) / max(total_gen_b, 1.0), 3
        ),
    }

    added = sv.added_mw(baseline_wind_mw, baseline_solar_mw)
    x_capex = ri.InvestmentVector(
        wind_mw    = added["wind_mw"],
        solar_mw   = added["solar_mw"],
        battery_mw = added["battery_mw"],
    )
    capex_usd    = ri._annualized_capex(x_capex, capex_cfg, sim_days)
    embodied_co2 = ri._annualized_embodied_co2(x_capex, capex_cfg, sim_days)

    result: dict = {
        "hash":             h,
        "investment":       sv.to_dict(),
        "scaled_grid":      scaled_grid,
        "status":           "pending",
        "portfolio_id":     int(args.row),
        "portfolio_group":  str(row.get("group", "")),
        "baseline_metrics": baseline_metrics,
        "added_wind_mw":    round(added["wind_mw"],   1),
        "added_solar_mw":   round(added["solar_mw"],  1),
        "added_battery_mw": round(added["battery_mw"], 1),
    }

    def _save(r: dict) -> None:
        with open(result_path, "w") as fp:
            json.dump(r, fp, indent=2,
                      default=lambda o: int(o) if hasattr(o, "item") else str(o))

    # ── Step 1: Build scaled investment grid (shared with CAS run) ─────────────
    try:
        si._apply_investment_scale(dc_grid, sv, scaled_grid)
    except Exception as exc:
        result["status"] = f"grid_error: {exc}"
        _save(result)
        return

    # ── Step 2: Run VATIC on raw scaled grid (no CAS) per week ────────────────
    inv_sim_dirs: list[Path] = []
    for wdate in multi_week_dates:
        inv_dir = out_root / "sims_inv" / f"{sim_days}d" / h / wdate
        inv_sim_dirs.append(inv_dir)
        if not (inv_dir / "thermal_detail.csv").exists():
            ok = ri._run_vatic(
                scaled_grid, inv_dir, wdate, sim_days,
                sim_params["solver"], sim_params["solver_args"],
                sim_params["threads"], sim_params["ruc_horizon"],
                sim_params["sced_horizon"], sim_params["ruc_mipgap"],
                sim_params["reserve_factor"], sim_params["output_detail"],
                sim_params.get("thermal_rating_scale", 1.0),
            )
            if not ok:
                result["status"] = "vatic_inv_failed"
                _save(result)
                return
        else:
            print(f"  INV VATIC cached for {wdate}")

    # ── Step 3: Extract metrics (use scaled grid gen_csv for correct UIDs) ────
    try:
        per_week = [ri._extract_metrics(d, scaled_gen_csv) for d in inv_sim_dirs]
        metrics  = _aggregate_metrics(per_week)
    except Exception as exc:
        result["status"] = f"metric_error: {exc}"
        _save(result)
        return

    # ── Step 4: Compute objective vs baseline ──────────────────────────────────
    w = {
        "lambda_carbon":      0.5,
        "lambda_cost":        0.3,
        "lambda_reliability": 0.2,
        "lambda_water":       0.0,
    }
    obj = ri._compute_objective(
        metrics, baseline_metrics,
        lambda_carbon=w["lambda_carbon"],
        lambda_cost=w["lambda_cost"],
        lambda_reliability=w["lambda_reliability"],
        lambda_water=w["lambda_water"],
        water_wd_gal=metrics.get("total_wd_gal", 0.0),
        base_water_wd=1.0,
        reliability_penalty=1000.0,
        curtailment_penalty=0.0,
        capex_usd=capex_usd,
        embodied_co2_t=embodied_co2,
    )

    result.update({
        "status":                    "ok",
        "metrics":                   metrics,
        "capex_usd":                 round(capex_usd, 0),
        "embodied_co2_t":            round(embodied_co2, 2),
        "total_cost_with_capex_usd": round(metrics["total_cost_usd"] + capex_usd, 0),
        "objective":                 round(obj, 6),
    })
    _save(result)

    status = result.get("status", "?")
    co2    = result.get("metrics", {}).get("total_co2_tonnes", "N/A")
    cost   = result.get("metrics", {}).get("total_cost_usd", "N/A")
    capex  = result.get("capex_usd", "N/A")
    print(f"Done: status={status}  CO2={co2}t  opex=${cost}  capex=${capex}")
    print(f"Result → {result_path}")


if __name__ == "__main__":
    main()
