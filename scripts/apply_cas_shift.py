#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
apply_cas_shift.py

Copy a VATIC grid and rewrite its BusInjections timeseries to reflect a
carbon-aware-scheduled (CAS) workload shift.  The resulting grid can be run
directly with vatic-det and compared against the pre-shift baseline.

Two scheduling modes (--mode):

  grid-mix  (default)
      Signal  : hourly grid carbon intensity from the baseline simulation.
      Metric  : carbon cost reduction (%).
      Algorithm: cas_grid_mix — shift flexible load toward low-CI hours.

  24_7
      Signal  : hourly renewable surplus (renewable supply − DC demand).
      Metric  : 24/7 renewable coverage (%).
      Algorithm: cas_24_7 — shift flexible load toward hours where renewable
                 supply exceeds DC demand.
      Extra args: --renew-generators, --renew-fraction.

Workflow
--------
1. A baseline simulation (--sim-dir) supplies the scheduling signal
   (carbon intensity for grid-mix; renewable output for 24/7).
2. The CAS algorithm shifts the flexible fraction of each hour's DC load
   while respecting a server capacity cap and daily energy balance
   (see scripts/cas.py for details).
3. Both the DAY_AHEAD and REAL_TIME BusInjections CSVs in the new grid copy
   are updated to reflect the shifted schedule:
     - DA (hourly): direct output of the CAS scheduler.
     - RT (sub-hourly): each period within a given hour is scaled by the
       same factor as the DA shift for that hour, preserving intra-hour
       variation.

All other grid files (gen.csv, bus.csv, timeseries_pointers.csv, thermal/
renewable timeseries, initial state, …) are copied unchanged.

Usage
-----
    # Grid-mix mode
    python scripts/apply_cas_shift.py \\
        --source-grid    RTS-GMLC-DC          \\
        --output-grid    RTS-GMLC-DC-CAS      \\
        --sim-dir        outputs/dc/2020-05-04 \\
        --buses          Abel Adams            \\
        --extra-capacity     30               \\
        --flexible-work-ratio 30

    # 24/7 mode (contracted 10% of grid-wide renewables)
    python scripts/apply_cas_shift.py \\
        --source-grid    RTS-GMLC-DC-RE       \\
        --output-grid    RTS-GMLC-DC-RE-247   \\
        --sim-dir        outputs/dc-re/2020-05-04 \\
        --buses          Abel Adams            \\
        --mode           24_7                  \\
        --renew-fraction 0.10                  \\
        --extra-capacity     30               \\
        --flexible-work-ratio 30

Then run the new grid normally:
    bash scripts/run_simulation.sh --grid RTS-GMLC-DC-RE-247 \\
        --date 2020-05-04 --days 1 --scenario dc-247

And compare:
    python scripts/compare_sim_outputs.py \\
        --sim-dirs outputs/dc-re/2020-05-04 outputs/dc-247/2020-05-04 \\
        --labels   baseline cas-247
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
import cas  # noqa: E402

# ---------------------------------------------------------------------------
# Grid registry (data_dir name per grid family)
# ---------------------------------------------------------------------------
GRID_REGISTRY: dict[str, dict] = {
    "RTS-GMLC":      {"data_dir": "RTS_Data"},
    "Texas-7k_2030": {"data_dir": "TX2030_Data"},
    "Texas-7k":      {"data_dir": "TX_Data"},
}

_VATIC_ROOT = _SCRIPTS_DIR.parent
_GRIDS_DIR  = _VATIC_ROOT / "vatic" / "data" / "grids"
_INIT_DIR   = _GRIDS_DIR / "initial-state"


def _registry_cfg(grid: str) -> dict:
    for key in sorted(GRID_REGISTRY, key=len, reverse=True):
        if grid.startswith(key):
            return GRID_REGISTRY[key]
    sys.exit(f"Unsupported grid '{grid}'. Known prefixes: {list(GRID_REGISTRY)}")


# ---------------------------------------------------------------------------
# BusInjections parsing & writing (period-based RTS-GMLC format)
# ---------------------------------------------------------------------------

def _read_injections(csv_path: Path, buses: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Read a BusInjections CSV, return (df, time_cols, bus_cols)."""
    df = pd.read_csv(csv_path)
    if "Period" in df.columns:
        time_cols = ["Year", "Month", "Day", "Period"]
    elif "Forecast_time" in df.columns:
        time_cols = ["Issue_time", "Forecast_time"]
    elif "Time" in df.columns:
        time_cols = ["Time"]
    else:
        sys.exit(f"Cannot detect time format in {csv_path}")

    bus_cols = [c for c in df.columns if c not in time_cols]
    missing  = [b for b in buses if b not in bus_cols]
    if missing:
        sys.exit(f"Bus(es) {missing} not found in {csv_path}. Available: {bus_cols}")
    return df, time_cols, bus_cols


def _da_to_datetime(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Convert period-based DA injection rows to UTC DatetimeIndex (hourly)."""
    period_offset = 1 if df["Period"].min() >= 1 else 0
    return pd.to_datetime(
        {"year": df["Year"], "month": df["Month"],
         "day":  df["Day"],  "hour":  df["Period"] - period_offset},
        utc=True,
    )


def _apply_da_shift(
    da_df: pd.DataFrame,
    time_cols: list[str],
    bus_cols: list[str],
    buses: list[str],
    shift_series: dict[str, pd.Series],   # bus → hourly shifted MW series
) -> pd.DataFrame:
    """Overwrite bus columns in a DA injection DataFrame with shifted values."""
    out = da_df.copy()
    dt_idx = _da_to_datetime(da_df)

    for bus in buses:
        shifted = shift_series[bus]
        # Reindex using the DatetimeIndex directly (not .values) to preserve
        # timezone-aware matching
        shifted_aligned = shifted.reindex(dt_idx)
        out[bus] = shifted_aligned.values

    return out


def _apply_rt_shift(
    rt_df: pd.DataFrame,
    time_cols: list[str],
    bus_cols: list[str],
    buses: list[str],
    scale_factors: dict[str, pd.Series],  # bus → hourly scale factor indexed by UTC hour timestamp
    rt_periods_per_hour: int,
) -> pd.DataFrame:
    """Scale RT injection values by the same hourly factor applied to DA."""
    out = rt_df.copy()

    # Build per-row UTC hour timestamp from Period column.
    # Period may be 1-based (RTS-GMLC) or 0-based (Texas-7k) — detect and normalize.
    period = rt_df["Period"].astype(int)
    period_offset = 1 if period.min() >= 1 else 0
    hour   = (period - period_offset) // rt_periods_per_hour

    dt_hour = pd.to_datetime(
        {"year": rt_df["Year"], "month": rt_df["Month"],
         "day":  rt_df["Day"],  "hour":  hour},
        utc=True,
    )

    for bus in buses:
        sf = scale_factors[bus]
        sf_aligned = sf.reindex(dt_hour, fill_value=1.0)
        out[bus] = (rt_df[bus].astype(float) * sf_aligned.values).clip(lower=0.0)

    return out


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source-grid", required=True,
                   help="Source grid name (must have BusInjections, e.g. RTS-GMLC-DC).")
    p.add_argument("--output-grid", required=True,
                   help="Name for the new CAS grid (e.g. RTS-GMLC-DC-CAS).")
    p.add_argument("--sim-dir", required=True, type=Path,
                   help="Vatic output dir from a baseline run on --source-grid "
                        "(used to compute the hourly carbon intensity signal).")
    p.add_argument("--buses", required=True, nargs="+", metavar="BUS",
                   help="DC bus names (must match columns in BusInjections CSVs).")
    p.add_argument("--mode", choices=["grid-mix", "24_7", "lp"], default="grid-mix",
                   help="Scheduling mode: grid-mix (minimize CI), 24_7 "
                        "(maximize renewable coverage), or lp (LP-optimal, "
                        "minimises α·LMP + (1-α)·CI). Default: grid-mix.")
    p.add_argument("--renew-generators", nargs="*", default=None, metavar="GEN_UID",
                   help="[24_7 only] Restrict renewable supply to these generator UIDs. "
                        "Defaults to all Solar+Wind in the simulation.")
    p.add_argument("--renew-fraction", type=float, default=1.0, metavar="FRAC",
                   help="[24_7 only] Scale renewable supply by this factor (0–1). "
                        "Models a contracted share of grid-wide renewable output "
                        "(e.g. 0.10 = 10%% of total generation). Default: 1.0.")
    p.add_argument("--extra-capacity", type=float, default=30.0, metavar="PCT",
                   help="Server headroom above current DC peak in %% (default: 30).")
    p.add_argument("--flexible-work-ratio", type=float, default=30.0, metavar="PCT",
                   help="[grid-mix/24_7] Percentage of each hour's load that may be deferred (default: 30).")
    p.add_argument("--deferral-window", type=int, default=12, metavar="H",
                   help="[grid-mix/24_7] Max hours a flexible job may be deferred (default: 12).")
    p.add_argument("--alpha", type=float, default=0.5, metavar="ALPHA",
                   help="[lp] Trade-off weight: 0=carbon-only, 1=cost-only (default: 0.5).")
    p.add_argument("--ramp-rate", type=float, default=None, metavar="MW/H",
                   help="[lp] Max load change between hours (MW/h). Default: unconstrained.")
    p.add_argument("--perturb-sim-dir", type=Path, default=None, metavar="DIR",
                   help="Vatic simulation directory for a perturbed (higher DC load) run. "
                        "[grid-mix] When supplied, uses marginal CI (Δco₂/Δload) as the "
                        "shifting signal instead of average CI, correcting for committed-"
                        "baseload distortion. "
                        "[lp] Enables price-anticipating QP via β=∂LMP/∂D.")
    p.add_argument("--start-date", default=None,
                   help="Start of the shift window YYYY-MM-DD (defaults to sim-dir name).")
    p.add_argument("--end-date", default=None,
                   help="End of the shift window YYYY-MM-DD (defaults to start).")
    p.add_argument("--vatic-root", type=Path, default=None,
                   help="VATIC repo root (auto-detected by default).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.output_grid == args.source_grid:
        sys.exit("--output-grid must differ from --source-grid.")

    if args.vatic_root:
        root      = args.vatic_root.resolve()
        grids_dir = root / "vatic" / "data" / "grids"
        init_dir  = grids_dir / "initial-state"
    else:
        grids_dir = _GRIDS_DIR
        init_dir  = _INIT_DIR

    cfg          = _registry_cfg(args.source_grid)
    src_grid_dir = grids_dir / args.source_grid
    src_data_dir = src_grid_dir / cfg["data_dir"]
    src_init_dir = init_dir / args.source_grid
    sim_dir      = args.sim_dir.resolve()

    if not src_data_dir.is_dir():
        sys.exit(f"Source grid data directory not found: {src_data_dir}")
    if not sim_dir.is_dir():
        sys.exit(f"sim-dir not found: {sim_dir}")

    inject_dir = src_data_dir / "timeseries_data_files" / "BusInjections"
    if not inject_dir.is_dir():
        sys.exit(
            f"BusInjections directory not found in source grid: {inject_dir}\n"
            "Run add_datacenter_load.py to create a grid with BusInjections."
        )

    gen_csv = src_data_dir / "SourceData" / "gen.csv"

    start_date = args.start_date or sim_dir.name
    end_date   = args.end_date   or start_date

    # -----------------------------------------------------------------------
    # 1. Copy grid + initial-state
    # -----------------------------------------------------------------------
    dst_grid_dir = grids_dir / args.output_grid
    dst_data_dir = dst_grid_dir / cfg["data_dir"]
    dst_init_dir = init_dir / args.output_grid

    print(f"\n[1/5] Copying grid '{args.source_grid}' → '{args.output_grid}'")
    if dst_grid_dir.exists():
        answer = input(f"  '{dst_grid_dir}' already exists. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            sys.exit("Aborted.")
        shutil.rmtree(dst_grid_dir)
    shutil.copytree(src_grid_dir, dst_grid_dir)
    print(f"  Copied → {dst_grid_dir}")

    print(f"\n[2/5] Copying initial-state '{args.source_grid}' → '{args.output_grid}'")
    if src_init_dir.is_dir():
        if dst_init_dir.exists():
            shutil.rmtree(dst_init_dir)
        shutil.copytree(src_init_dir, dst_init_dir)
        print(f"  Copied → {dst_init_dir}")
    else:
        print("  No initial-state dir found — skipping.")

    # -----------------------------------------------------------------------
    # 2. Load scheduling signal from baseline simulation
    # -----------------------------------------------------------------------
    if args.mode == "grid-mix":
        if args.perturb_sim_dir is not None:
            pdir = args.perturb_sim_dir.resolve()
            if not pdir.is_dir():
                sys.exit(f"--perturb-sim-dir not found: {pdir}")
            print(f"\n[3/5] Computing marginal CI from '{sim_dir.name}' vs '{pdir.name}'")
            mci_df    = cas.compute_marginal_ci(sim_dir, pdir, args.buses, gen_csv)
            signal_df = mci_df[["marginal_ci", "avg_ci_base"]].rename(
                columns={"marginal_ci": "carbon_intensity"}
            )
            print(f"  Hours: {len(signal_df)}  "
                  f"Mean marginal CI: {signal_df['carbon_intensity'].mean():.2f}  "
                  f"avg CI: {signal_df['avg_ci_base'].mean():.2f} kg CO₂/MWh")
        else:
            print(f"\n[3/5] Computing avg carbon intensity from '{sim_dir.name}'")
            signal_df = cas.compute_carbon_intensity(sim_dir, gen_csv)
            print(f"  Hours: {len(signal_df)}  "
                  f"Mean CI: {signal_df['carbon_intensity'].mean():.2f} kg CO₂/MWh")
    elif args.mode == "lp":
        print(f"\n[3/5] Computing CI + LMPs from '{sim_dir.name}'")
        ci_df  = cas.compute_carbon_intensity(sim_dir, gen_csv)
        lmp_df = cas.load_lmp(sim_dir, args.buses,
                              start_date=start_date, end_date=end_date)
        signal_df = ci_df.join(lmp_df, how="inner")
        print(f"  Hours: {len(signal_df)}  "
              f"Mean CI: {signal_df['carbon_intensity'].mean():.2f} kg CO2/MWh  "
              f"Mean LMP: {signal_df['lmp'].mean():.2f} $/MWh")

        # Price sensitivity β = ∂LMP/∂D (optional — switches LP → QP)
        lp_beta_series = None
        if args.perturb_sim_dir is not None:
            pdir = args.perturb_sim_dir.resolve()
            if not pdir.is_dir():
                sys.exit(f"--perturb-sim-dir not found: {pdir}")
            print(f"  Computing β from '{pdir.name}' …")
            beta_df = cas.compute_price_sensitivity(sim_dir, pdir, args.buses)
            lp_beta_series = beta_df["beta"]
            print(f"  Mean β={lp_beta_series.mean():.4f}  "
                  f"non-zero={int((lp_beta_series > 0).sum())}/{len(lp_beta_series)} hours  "
                  f"→ price-anticipating QP")
    else:
        gens = args.renew_generators or None
        frac_str = f"  ×{args.renew_fraction:.2f}" if args.renew_fraction != 1.0 else ""
        label = f"[{', '.join(gens)}]" if gens else "[all Solar+Wind]"
        print(f"\n[3/5] Loading renewable supply from '{sim_dir.name}' {label}{frac_str}")
        signal_df = cas.load_renewable_supply(sim_dir, gen_csv, generators=gens)
        if args.renew_fraction != 1.0:
            signal_df = signal_df * args.renew_fraction
        print(f"  Hours: {len(signal_df)}  "
              f"Mean supply: {signal_df['tot_renewable'].mean():.1f} MW")

    # -----------------------------------------------------------------------
    # 3. Load DA injection and build CAS-shifted profile per bus
    # -----------------------------------------------------------------------
    lp_mode_tag = ""
    if args.mode == "lp":
        lp_mode_tag = (" [QP]" if getattr(args, "perturb_sim_dir", None) else " [LP]")
    print(f"\n[4/5] Applying CAS {args.mode}{lp_mode_tag} shift  "
          f"(extra_cap={args.extra_capacity}%, flex={args.flexible_work_ratio}%)")

    da_csvs = list(inject_dir.glob("DAY_AHEAD_*.csv"))
    rt_csvs = list(inject_dir.glob("REAL_TIME_*.csv"))
    if len(da_csvs) != 1:
        sys.exit(f"Expected exactly one DA CSV in {inject_dir}, found {len(da_csvs)}.")
    da_csv = da_csvs[0]
    if len(rt_csvs) == 0:
        # RT file missing — use DA as RT (T7k uses hourly resolution for both)
        rt_csv = inject_dir / da_csv.name.replace("DAY_AHEAD_", "REAL_TIME_")
        shutil.copy(da_csv, rt_csv)
        print(f"  Note: RT injection file missing; copied DA → {rt_csv.name}")
    elif len(rt_csvs) != 1:
        sys.exit(f"Expected exactly one RT CSV in {inject_dir}, found {len(rt_csvs)}.")
    else:
        rt_csv = rt_csvs[0]

    da_df, da_time_cols, da_bus_cols = _read_injections(da_csv, args.buses)
    rt_df, rt_time_cols, rt_bus_cols = _read_injections(rt_csv, args.buses)

    # Detect RT periods per hour.
    # RTS-GMLC: 1-based periods (1–288 for 5-min, 1–24 for hourly).
    # Texas-7k:  0-based periods (0–23 for hourly).
    rt_min_period = int(rt_df["Period"].min())
    rt_max_period = int(rt_df["Period"].max())
    n_periods     = rt_max_period - rt_min_period + 1   # 24 (hourly) or 288 (5-min)
    rt_periods_per_hour = max(1, n_periods // 24)       # e.g. 288//24=12, 24//24=1

    shifted_series: dict[str, pd.Series]   = {}
    scale_factors:  dict[str, pd.Series]   = {}

    def _bus_window(bus: str) -> pd.DataFrame:
        """Return hourly DC power for one bus filtered to the shift window."""
        bus_da = da_df[da_time_cols + [bus]].copy()
        dt_idx = _da_to_datetime(bus_da)
        hourly = pd.DataFrame(
            {"avg_dc_power_mw": bus_da[bus].astype(float).values},
            index=dt_idx,
        ).sort_index()
        return hourly[
            (hourly.index >= pd.Timestamp(start_date, tz="utc")) &
            (hourly.index <  pd.Timestamp(end_date,   tz="utc") + pd.Timedelta(days=1))
        ]

    if args.mode == "lp":
        # ---- LP/QP: schedule on combined load, distribute via scale factors ----
        combined_hourly = sum(_bus_window(b)["avg_dc_power_mw"] for b in args.buses)
        combined_df = pd.DataFrame({"avg_dc_power_mw": combined_hourly}).join(
            signal_df, how="inner"
        )
        if combined_df.empty:
            sys.exit(
                f"No overlapping hours between BusInjections and LP signals.\n"
                f"  Injection range: {combined_hourly.index[0]} – {combined_hourly.index[-1]}\n"
                f"  Signal range   : {signal_df.index[0]} – {signal_df.index[-1]}"
            )

        tot_peak = float(combined_df["avg_dc_power_mw"].max())
        max_cap  = tot_peak * (1.0 + args.extra_capacity / 100.0)

        shifted_combined = cas.cas_lp(combined_df, max_cap,
                                      alpha=args.alpha, ramp_rate=args.ramp_rate,
                                      beta=lp_beta_series)

        orig_carbon   = cas.carbon_cost(combined_df)
        new_carbon    = cas.carbon_cost(shifted_combined)
        orig_lmp_cost = float((combined_df["avg_dc_power_mw"] * combined_df["lmp"]).sum())
        new_lmp_cost  = float((shifted_combined["avg_dc_power_mw"] * combined_df["lmp"]).sum())
        carbon_red = (orig_carbon - new_carbon)    / orig_carbon   * 100 if orig_carbon   > 0 else 0.0
        cost_red   = (orig_lmp_cost - new_lmp_cost) / orig_lmp_cost * 100 if orig_lmp_cost > 0 else 0.0
        print(f"  Combined ({len(args.buses)} buses)  "
              f"peak={tot_peak:.1f} MW  max_cap={max_cap:.1f} MW  "
              f"carbon={carbon_red:.2f}%  cost={cost_red:.2f}%  (α={args.alpha})")

        combined_orig = combined_df["avg_dc_power_mw"]
        sf_combined   = (shifted_combined["avg_dc_power_mw"]
                         / combined_orig.replace(0, np.nan)).fillna(1.0)

        for bus in args.buses:
            bw = _bus_window(bus)
            orig_bus   = bw["avg_dc_power_mw"]
            sf_aligned = sf_combined.reindex(orig_bus.index, fill_value=1.0)
            shifted_series[bus] = (orig_bus * sf_aligned).clip(lower=0.0)
            scale_factors[bus]  = sf_aligned

    elif args.mode == "24_7":
        # ---- 24/7: schedule on combined load, distribute via scale factors ----
        #
        # Renewable supply is a shared resource across all DC buses; scheduling
        # must be evaluated against the total load to avoid trivial 100% coverage
        # when individual bus load < total renewable supply.
        combined_hourly = sum(_bus_window(b)["avg_dc_power_mw"] for b in args.buses)
        combined_df = pd.DataFrame({"avg_dc_power_mw": combined_hourly}).join(
            signal_df, how="inner"
        )
        if combined_df.empty:
            sys.exit(
                f"No overlapping hours between BusInjections and renewable supply signal.\n"
                f"  Injection range: {combined_hourly.index[0]} – {combined_hourly.index[-1]}\n"
                f"  Signal range   : {signal_df.index[0]} – {signal_df.index[-1]}"
            )

        tot_peak = float(combined_df["avg_dc_power_mw"].max())
        max_cap  = tot_peak * (1.0 + args.extra_capacity / 100.0)

        shifted_combined = cas.cas_24_7(combined_df, args.flexible_work_ratio, max_cap, window=args.deferral_window)
        orig_cov = cas.calculate_coverage(combined_df)
        new_cov  = cas.calculate_coverage(shifted_combined)
        print(f"  Combined ({len(args.buses)} buses)  "
              f"peak={tot_peak:.1f} MW  max_cap={max_cap:.1f} MW  "
              f"coverage {orig_cov:.1f}% → {new_cov:.1f}%  (+{new_cov - orig_cov:.1f} pp)")

        # Per-bus scale factor = same as combined shift ratio
        combined_orig = combined_df["avg_dc_power_mw"]
        sf_combined   = (shifted_combined["avg_dc_power_mw"]
                         / combined_orig.replace(0, np.nan)).fillna(1.0)

        for bus in args.buses:
            bw = _bus_window(bus)
            orig_bus = bw["avg_dc_power_mw"]
            # Apply combined scale factor to each bus
            sf_aligned = sf_combined.reindex(orig_bus.index, fill_value=1.0)
            shifted_bus = (orig_bus * sf_aligned).clip(lower=0.0)
            shifted_series[bus] = shifted_bus
            scale_factors[bus]  = sf_aligned

    else:
        # ---- grid-mix: run per bus independently (CI is a price signal) ----
        for bus in args.buses:
            bus_window = _bus_window(bus)
            df_all = bus_window.join(signal_df, how="inner")
            if df_all.empty:
                sys.exit(
                    f"No overlapping hours between BusInjections and CI signal "
                    f"for bus '{bus}'.\n"
                    f"  Injection range: {bus_window.index[0]} – {bus_window.index[-1]}\n"
                    f"  Signal range   : {signal_df.index[0]} – {signal_df.index[-1]}"
                )

            cur_peak = float(df_all["avg_dc_power_mw"].max())
            max_cap  = cur_peak * (1.0 + args.extra_capacity / 100.0)

            shifted    = cas.cas_grid_mix(df_all, args.flexible_work_ratio, max_cap, window=args.deferral_window)
            orig_carbon = cas.carbon_cost(df_all)
            new_carbon  = cas.carbon_cost(shifted)
            metric      = (orig_carbon - new_carbon) / orig_carbon * 100 if orig_carbon > 0 else 0.0

            print(f"  {bus:20s}  peak={cur_peak:.1f} MW  max_cap={max_cap:.1f} MW  "
                  f"carbon reduction={metric:.2f}%")

            shifted_series[bus] = shifted["avg_dc_power_mw"]
            orig  = df_all["avg_dc_power_mw"]
            sf    = shifted["avg_dc_power_mw"] / orig.replace(0, np.nan)
            scale_factors[bus] = sf.fillna(1.0)

    # -----------------------------------------------------------------------
    # 4. Write shifted injections into the new grid copy
    # -----------------------------------------------------------------------
    print(f"\n[5/5] Writing shifted BusInjections to '{args.output_grid}'")

    dst_inject_dir = dst_data_dir / "timeseries_data_files" / "BusInjections"
    dst_da_csv = dst_inject_dir / da_csv.name
    dst_rt_csv = dst_inject_dir / rt_csv.name

    # DA — overwrite the simulated date range; leave other dates unchanged
    new_da = _apply_da_shift(da_df, da_time_cols, da_bus_cols,
                             args.buses, shifted_series)
    new_da.to_csv(dst_da_csv, index=False)
    print(f"  DA written → {dst_da_csv.name}")

    # RT — scale by the same hourly factor
    new_rt = _apply_rt_shift(rt_df, rt_time_cols, rt_bus_cols,
                             args.buses, scale_factors, rt_periods_per_hour)
    new_rt.to_csv(dst_rt_csv, index=False)
    print(f"  RT written → {dst_rt_csv.name}  ({rt_periods_per_hour} periods/hour)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    mode_detail = ""
    if args.mode == "24_7":
        gens_str = ", ".join(args.renew_generators) if args.renew_generators else "all Solar+Wind"
        mode_detail = (
            f"  Renewable generators : {gens_str}\n"
            f"  Renewable fraction   : {args.renew_fraction:.2f}\n"
        )
    elif args.mode == "lp":
        qp_tag = f"yes ({args.perturb_sim_dir.name})" if args.perturb_sim_dir else "no (LP)"
        mode_detail = (
            f"  Alpha (cost/carbon)  : {args.alpha}\n"
            f"  Ramp rate            : {args.ramp_rate} MW/h\n"
            f"  Price-anticipating QP: {qp_tag}\n"
        )

    print(f"""
Done!  New grid: '{args.output_grid}'

CAS parameters:
  Mode                 : {args.mode}
{mode_detail}  Extra capacity       : {args.extra_capacity}%
  Flexible work ratio  : {args.flexible_work_ratio}%
  Shift window         : {start_date} – {end_date}

Only BusInjections were modified.  All other grid files are unchanged.

Next steps:
  bash scripts/run_simulation.sh --grid {args.output_grid} \\
      --date {start_date} --days 1 --scenario dc-cas

  python scripts/compare_sim_outputs.py \\
      --sim-dirs outputs/<baseline>/{start_date} outputs/dc-cas/{start_date} \\
      --labels   baseline cas \\
      --gen-csv  vatic/data/grids/{args.source_grid}/{cfg['data_dir']}/SourceData/gen.csv
""")


if __name__ == "__main__":
    main()
