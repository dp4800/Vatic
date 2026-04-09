#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
pareto_eval_scale.py — Evaluate one portfolio row with LP CAS, using the
scaling-based investment approach (scale existing wind/solar, add battery).

Usage (called by SLURM array task N):
    python scripts/pareto_eval_scale.py \\
        --row N \\
        --portfolios scripts/study/pareto_portfolios_scale.csv \\
        --config scripts/study/params_pareto_scale_p1.json

Pipeline per task:
  1. Build/reuse scaled grid (Texas-7k-DC-REAL-SCALE-{hash8})
  2. Build/reuse LP CAS-shifted grid using CI+LMP signals from annual sim-lp runs
  3. Run VATIC on the CAS-LP grid for each representative week
  4. Extract and aggregate metrics
  5. Write result JSON to: {out_root}/results/pareto_{NNN}_{hash8}.json

Key differences from pareto_eval.py (the invalid "add" approach):
  - ScaleVector(wind_scale, solar_scale, battery_mw) replaces InvestmentVector
  - Existing generators are SCALED in-place (PMax, timeseries, cost curves)
  - Batteries added from existing template, not a single oversized row
  - CAPEX computed on ADDED capacity = (scale-1) × baseline_total
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import cas as cas_mod                  # noqa: E402
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


# ── LP CAS grid builder (identical logic to pareto_eval.py) ────────────────────

def _build_cas_lp_grid(
    inv_grid: str,
    signal_dirs_by_date: dict,
    sim_days: int,
    alpha: float,
    gen_csv: Path,
    buses: list[str],
    cas_suffix: str = "",
    flexible_ratio: float = 0.0,
) -> str:
    cas_grid     = f"{inv_grid}-CAS-LP{cas_suffix}"
    dst_grid_dir = ri.GRIDS_DIR / cas_grid

    if dst_grid_dir.exists():
        print(f"  CAS-LP grid cached: {cas_grid}")
        return cas_grid

    data_dir_name = _acs._registry_cfg(inv_grid)["data_dir"]
    inject_dir    = (ri.GRIDS_DIR / inv_grid / data_dir_name
                     / "timeseries_data_files" / "BusInjections")

    da_csvs = list(inject_dir.glob("DAY_AHEAD_*.csv"))
    if len(da_csvs) != 1:
        raise RuntimeError(
            f"Expected 1 DA CSV in {inject_dir}, found {len(da_csvs)}"
        )
    da_csv  = da_csvs[0]
    rt_csvs = list(inject_dir.glob("REAL_TIME_*.csv"))
    rt_csv  = rt_csvs[0] if rt_csvs else None

    da_df, da_time_cols, bus_cols = _acs._read_injections(da_csv, [])
    actual_buses = buses if buses else bus_cols
    dt_idx       = _acs._da_to_datetime(da_df)

    shifted_arr = {b: da_df[b].astype(float).values.copy() for b in actual_buses}
    scale_arr   = {b: np.ones(len(dt_idx), dtype=float)     for b in actual_buses}
    dt_to_pos   = pd.Series(np.arange(len(dt_idx), dtype=int), index=dt_idx)

    for wdate, signal_dir in signal_dirs_by_date.items():
        end_date = (datetime.strptime(wdate, "%Y-%m-%d")
                    + timedelta(days=sim_days - 1)).strftime("%Y-%m-%d")

        week_mask = (
            (dt_idx >= pd.Timestamp(wdate, tz="utc")) &
            (dt_idx <  pd.Timestamp(end_date, tz="utc") + pd.Timedelta(days=1))
        )
        week_dt = dt_idx[week_mask]
        if len(week_dt) == 0:
            print(f"  WARNING: no injection rows for {wdate} — skipping LP CAS")
            continue

        ci_df  = cas_mod.compute_carbon_intensity(signal_dir, gen_csv)
        lmp_df = cas_mod.load_lmp(signal_dir, actual_buses,
                                  start_date=wdate, end_date=end_date)
        signal_df = ci_df.join(lmp_df, how="inner")

        bus_load = {}
        for bus in actual_buses:
            vals = da_df.loc[np.array(week_mask), bus].astype(float).values
            bus_load[bus] = pd.Series(vals, index=week_dt)

        combined_hourly = sum(s for s in bus_load.values())
        combined_df = (pd.DataFrame({"avg_dc_power_mw": combined_hourly})
                       .join(signal_df, how="inner"))

        if combined_df.empty:
            print(f"  WARNING: no signal overlap for {wdate} — skipping LP CAS")
            continue

        tot_peak = float(combined_df["avg_dc_power_mw"].max())
        if tot_peak <= 0:
            print(f"  WARNING: zero DC load for {wdate} — skipping LP CAS")
            continue

        max_cap          = tot_peak * 1.3
        shifted_combined = cas_mod.cas_lp(combined_df, max_cap, alpha=alpha,
                                          flexible_ratio=flexible_ratio)

        sf = (shifted_combined["avg_dc_power_mw"]
              / combined_df["avg_dc_power_mw"].replace(0, np.nan)).fillna(1.0)

        c_orig = float((combined_df["avg_dc_power_mw"]
                        * combined_df["carbon_intensity"]).sum())
        c_new  = float((shifted_combined["avg_dc_power_mw"]
                        * combined_df["carbon_intensity"]).sum())
        p_orig = float((combined_df["avg_dc_power_mw"] * combined_df["lmp"]).sum())
        p_new  = float((shifted_combined["avg_dc_power_mw"] * combined_df["lmp"]).sum())
        c_red  = (c_orig - c_new) / c_orig * 100 if c_orig > 0 else 0.0
        p_red  = (p_orig - p_new) / p_orig * 100 if p_orig > 0 else 0.0
        print(f"  LP CAS alpha={alpha}  {wdate}: "
              f"peak={tot_peak:.0f} MW  carbon={c_red:+.1f}%  cost={p_red:+.1f}%")

        for bus in actual_buses:
            orig     = bus_load[bus]
            sf_bus   = sf.reindex(orig.index, fill_value=1.0)
            shifted  = (orig * sf_bus).clip(lower=0.0)
            pos_in_da = dt_to_pos.reindex(orig.index).dropna().astype(int)
            shifted_arr[bus][pos_in_da.values] = (
                shifted.reindex(pos_in_da.index).values
            )
            scale_arr[bus][pos_in_da.values] = (
                sf_bus.reindex(pos_in_da.index).values
            )

    shift_series = {b: pd.Series(shifted_arr[b], index=dt_idx) for b in actual_buses}
    scale_series = {b: pd.Series(scale_arr[b],   index=dt_idx) for b in actual_buses}

    shutil.copytree(ri.GRIDS_DIR / inv_grid, dst_grid_dir)
    src_init = ri.INIT_DIR / inv_grid
    dst_init = ri.INIT_DIR / cas_grid
    if src_init.is_dir():
        if dst_init.exists():
            shutil.rmtree(dst_init)
        shutil.copytree(src_init, dst_init)

    dst_inject_dir = (dst_grid_dir / data_dir_name
                      / "timeseries_data_files" / "BusInjections")

    new_da = _acs._apply_da_shift(
        da_df, da_time_cols, bus_cols, actual_buses, shift_series
    )
    new_da.to_csv(dst_inject_dir / da_csv.name, index=False)

    if rt_csv:
        rt_df, rt_time_cols, rt_bus_cols = _acs._read_injections(rt_csv, [])
        rt_pph = max(
            1,
            (int(rt_df["Period"].max()) - int(rt_df["Period"].min()) + 1) // 24,
        )
        new_rt = _acs._apply_rt_shift(
            rt_df, rt_time_cols, rt_bus_cols, actual_buses, scale_series, rt_pph
        )
        new_rt.to_csv(dst_inject_dir / rt_csv.name, index=False)

    print(f"  Created CAS-LP grid: {cas_grid}")
    return cas_grid


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
    lp_alpha    = float(cfg.get("lp_alpha", 0.5))
    flex_ratio  = float(cfg.get("flexible_ratio", 0.0))
    cas_suffix = cfg.get("cas_grid_suffix", "")
    result_subdir = cfg.get("result_subdir", "results")
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
    buses    = grid_cfg.get("dc_buses") or []
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
    gen_csv        = ri.GRIDS_DIR / dc_grid       / _grid_data_dir / "SourceData" / "gen.csv"
    scaled_gen_csv = ri.GRIDS_DIR / scaled_grid   / _grid_data_dir / "SourceData" / "gen.csv"

    # Baseline wind/solar totals from source grid (for CAPEX on added MW)
    src_gen_df        = pd.read_csv(gen_csv)
    baseline_wind_mw  = float(src_gen_df[src_gen_df["Fuel"] == "WND (Wind)"]["PMax MW"].sum())
    baseline_solar_mw = float(src_gen_df[src_gen_df["Fuel"] == "SUN (Solar)"]["PMax MW"].sum())

    # Baseline metrics from annual sim-lp (used only for baseline comparison)
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

    # CAPEX on ADDED capacity only (scaling above baseline)
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
        "lp_alpha":         lp_alpha,
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

    # ── Step 1: Build scaled investment grid ───────────────────────────────────
    try:
        si._apply_investment_scale(dc_grid, sv, scaled_grid)
    except Exception as exc:
        result["status"] = f"grid_error: {exc}"
        _save(result)
        return

    # ── Step 2: Run VATIC on scaled grid (no CAS) to get correct CI signal ────
    # Signal dirs use the scaled grid's own dispatch, not the baseline sim-lp,
    # so that CAS shifts load based on carbon intensity from the actual
    # renewable mix of each portfolio.
    signal_dirs: dict = {}
    for wdate in multi_week_dates:
        inv_dir = out_root / "sims_inv" / f"{sim_days}d" / h / wdate
        signal_dirs[wdate] = inv_dir
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
                result["status"] = "vatic_signal_failed"
                _save(result)
                return
        else:
            print(f"  Signal sim cached for {wdate}")

    # ── Step 3: Build LP CAS-shifted grid ─────────────────────────────────────
    try:
        cas_grid = _build_cas_lp_grid(
            scaled_grid, signal_dirs, sim_days, lp_alpha, gen_csv, buses,
            cas_suffix, flexible_ratio=flex_ratio,
        )
    except Exception as exc:
        result["status"] = f"cas_grid_error: {exc}"
        _save(result)
        return

    # ── Step 4: Run VATIC on CAS-LP grid per week ─────────────────────────────
    cas_sim_dirs: list[Path] = []
    for wdate in multi_week_dates:
        cas_dir = out_root / "sims_cas" / f"{sim_days}d" / h / wdate
        cas_sim_dirs.append(cas_dir)
        if not (cas_dir / "thermal_detail.csv").exists():
            ok = ri._run_vatic(
                cas_grid, cas_dir, wdate, sim_days,
                sim_params["solver"], sim_params["solver_args"],
                sim_params["threads"], sim_params["ruc_horizon"],
                sim_params["sced_horizon"], sim_params["ruc_mipgap"],
                sim_params["reserve_factor"], sim_params["output_detail"],
                sim_params.get("thermal_rating_scale", 1.0),
            )
            if not ok:
                result["status"] = "vatic_cas_failed"
                _save(result)
                return
        else:
            print(f"  CAS VATIC cached for {wdate}")

    # ── Step 5: Extract metrics (use scaled grid gen_csv for correct UIDs) ────
    try:
        per_week = [ri._extract_metrics(d, scaled_gen_csv) for d in cas_sim_dirs]
        metrics  = _aggregate_metrics(per_week)
    except Exception as exc:
        result["status"] = f"metric_error: {exc}"
        _save(result)
        return

    # ── Step 6: Compute weighted objective vs baseline ─────────────────────────
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
