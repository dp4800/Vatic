# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
cas.py — Carbon-Aware Scheduling utilities for VATIC simulations.

Core functions
--------------
compute_carbon_intensity
    Build an hourly grid *average* carbon-intensity series (kg CO2 / MWh_delivered)
    from vatic simulation outputs (thermal_detail.csv, renew_detail.csv)
    joined with per-generator emission factors from gen.csv.

compute_marginal_ci
    Build an hourly *marginal* carbon-intensity series (kg CO2 / MWh of
    additional DC load) from a paired baseline + perturbed simulation.
    More accurate than average CI for quantifying the true emission impact
    of shifting flexible load.

load_renewable_supply
    Build an hourly renewable generation series (MW) from renew_detail.csv,
    optionally filtered to specific generator names (e.g. co-located units).

load_lmp
    Extract per-bus hourly LMPs from VATIC's bus_detail.csv output.

cas_grid_mix  [mode: grid-mix]
    Causal rolling-window shift toward low-CI hours.  Flexible work arriving
    at hour t may be deferred up to ``window`` hours (default 12).
    Signal: grid carbon intensity.  Metric: carbon reduction %.

cas_24_7  [mode: 24/7]
    Causal rolling-window shift toward high-renewable hours.  Same deferral
    window as cas_grid_mix.  Signal: renewable surplus.
    Metric: 24/7 renewable coverage %.

cas_lp  [mode: lp]
    LP-optimal scheduler.  Minimises α·LMP + (1-α)·CI subject to:
      - Daily energy balance (all arrived work is processed)
      - Causal backlog constraint (work can only be deferred, not advanced)
      - Server capacity ceiling
      - Optional ramp-rate limit on inter-hour load changes
    Signal: both LMP and CI.  Metric: carbon reduction % and cost reduction %.

load_dc_power
    Parse a VATIC BusInjections CSV into an hourly DC power series.

carbon_cost / calculate_coverage
    Scalar summary helpers used by the analysis scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Carbon intensity from VATIC simulation outputs
# ---------------------------------------------------------------------------

# Fallback emission factors (kg CO2/MWh) by fuel keyword, for grids that
# lack HR_avg_0 / Emissions CO2 Lbs/MMBTU columns (e.g. Texas-7k).
# Sources: EPA eGRID 2023 Texas median (inputs/egrid2023_data_rev2.xlsx,
# PLNT23 sheet) and EIA CO2 coefficients (inputs/co2_vol_mass.xlsx).
# Values in kg CO2/MWh = tCO2/MWh × 1000.
_FUEL_FALLBACK_KG = {
    "coal":           1078.5,   # thesis Table AI: 1.0785 tCO₂/MWh
    "lignite":        1078.5,
    "subbituminous":  1078.5,
    "petroleum coke": 1021.2,   # thesis Table AI: 1.0212 tCO₂/MWh
    "pet coke":       1021.2,
    "oil":             795.8,   # thesis Table AI: 0.7958 tCO₂/MWh
    "natural gas":     496.3,   # thesis Table AI: 0.4963 tCO₂/MWh
    "ng":              496.3,
    "gas":             496.3,
    "biomass":          54.0,   # thesis Table AI: 0.054 tCO₂/MWh
    "wood":             54.0,
}


def _emission_factors(gen_csv: str | Path) -> pd.Series:
    """Return per-generator emission factor in kg CO2 / MWh.

    For grids with RTS-GMLC-style columns (HR_avg_0, Emissions CO2 Lbs/MMBTU),
    factors are computed per-generator from source data.  For grids that lack
    these columns (e.g. Texas-7k), a fuel-keyword fallback is used instead.

    Formula (when columns are present):
        EF [kg CO2/MWh] = HR_avg_0 [BTU/kWh] / 1000          # → MMBTU/MWh
                          * CO2 [lbs/MMBTU]
                          * 0.453592                           # lbs → kg
    """
    gen_df = pd.read_csv(gen_csv)

    has_hr  = "HR_avg_0" in gen_df.columns
    has_co2 = "Emissions CO2 Lbs/MMBTU" in gen_df.columns

    if has_hr and has_co2:
        hr      = pd.to_numeric(gen_df["HR_avg_0"], errors="coerce").fillna(0.0) / 1000.0
        co2_lbs = pd.to_numeric(gen_df["Emissions CO2 Lbs/MMBTU"], errors="coerce").fillna(0.0)
        ef      = hr * co2_lbs * 0.453592
    else:
        # Fallback: match fuel keyword → kg CO2/MWh
        uid_col  = "GEN UID" if "GEN UID" in gen_df.columns else gen_df.columns[0]
        # Prefer exact "Fuel" or "Unit Type" columns; avoid price/numeric columns.
        _fuel_candidates = ["Fuel", "Unit Type", "fuel_type", "Fuel Type"]
        fuel_col = next((c for c in _fuel_candidates if c in gen_df.columns), None)
        if fuel_col is None:
            fuel_col = next(
                (c for c in gen_df.columns
                 if "fuel" in c.lower() and "price" not in c.lower() and "cost" not in c.lower()),
                None,
            )
        fuels    = gen_df[fuel_col].astype(str).str.lower() if fuel_col else pd.Series([""] * len(gen_df))
        ef = fuels.map(
            lambda f: next((v for k, v in _FUEL_FALLBACK_KG.items() if k in f), 0.0)
        )

    ef.index = gen_df["GEN UID"]
    return ef


def compute_carbon_intensity(
    sim_dir: str | Path,
    gen_csv: str | Path,
) -> pd.DataFrame:
    """Compute hourly grid average carbon intensity from a vatic sim run.

    Carbon intensity is defined as:
        CI(t) = sum_i[Dispatch_i(t) * EF_i]
                / (sum_i[Dispatch_i(t)] + sum_j[Output_j(t)])

    where i indexes thermal generators and j indexes renewables, so that
    renewable generation dilutes the carbon content of delivered electricity.

    Parameters
    ----------
    sim_dir : path
        Directory containing thermal_detail.csv and renew_detail.csv produced
        by a vatic simulation run.
    gen_csv : path
        gen.csv for the grid (used to look up heat rates and emission factors).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC, hourly), single column ``carbon_intensity``
        in kg CO2 / MWh.  Hours with zero total generation return 0.
    """
    sim_dir = Path(sim_dir)
    ef = _emission_factors(gen_csv)

    # --- Thermal dispatch --------------------------------------------------
    thermal = pd.read_csv(sim_dir / "thermal_detail.csv")
    thermal["datetime"] = pd.to_datetime(
        thermal["Date"] + " " + thermal["Hour"].astype(str) + ":00",
        format="%Y-%m-%d %H:%M",
        utc=True,
    )
    # kg CO2 emitted per generator per hour = Dispatch [MWh in 1 hour] * EF
    thermal["co2_kg"] = thermal["Dispatch"] * thermal["Generator"].map(ef).fillna(0.0)

    hourly_co2  = thermal.groupby("datetime")["co2_kg"].sum()
    hourly_th   = thermal.groupby("datetime")["Dispatch"].sum()

    # --- Renewable output --------------------------------------------------
    renew = pd.read_csv(sim_dir / "renew_detail.csv")
    renew["datetime"] = pd.to_datetime(
        renew["Date"] + " " + renew["Hour"].astype(str) + ":00",
        format="%Y-%m-%d %H:%M",
        utc=True,
    )
    hourly_re = renew.groupby("datetime")["Output"].sum()

    # --- Carbon intensity --------------------------------------------------
    idx = hourly_co2.index.union(hourly_th.index).union(hourly_re.index)
    co2   = hourly_co2.reindex(idx, fill_value=0.0)
    th    = hourly_th.reindex(idx, fill_value=0.0)
    re    = hourly_re.reindex(idx, fill_value=0.0)
    total = th + re

    ci = pd.Series(
        np.where(total > 0, co2 / total, 0.0),
        index=idx,
        name="carbon_intensity",
    )
    return ci.sort_index().to_frame()


# ---------------------------------------------------------------------------
# DC power from BusInjections
# ---------------------------------------------------------------------------

def load_dc_power(
    inject_csv: str | Path,
    buses: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Parse a VATIC BusInjections CSV into an hourly DC power series.

    Supports the RTS-GMLC Period-based format (Year/Month/Day/Period 1-24)
    as well as an ISO-timestamp format (column ``Time`` or ``Forecast_time``).

    Parameters
    ----------
    inject_csv : path
        Path to a DAY_AHEAD_bus_injections.csv.
    buses : list of str, optional
        Bus name columns to sum.  Defaults to all non-time columns.
    start_date, end_date : str, optional
        Inclusive date range filter in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC, hourly), single column ``avg_dc_power_mw``
        containing the sum of the selected bus injections per hour.
    """
    df = pd.read_csv(inject_csv)

    # Detect time format
    if "Period" in df.columns:
        # RTS-GMLC uses 1-based periods (1-24); Texas-7k uses 0-based (0-23).
        period_offset = 1 if df["Period"].min() >= 1 else 0
        times = pd.to_datetime(
            {
                "year":   df["Year"],
                "month":  df["Month"],
                "day":    df["Day"],
                "hour":   df["Period"] - period_offset,
            },
            utc=True,
        )
        time_cols = ["Year", "Month", "Day", "Period"]
    elif "Forecast_time" in df.columns:
        times = pd.to_datetime(df["Forecast_time"], utc=True)
        time_cols = ["Issue_time", "Forecast_time"]
    elif "Time" in df.columns:
        times = pd.to_datetime(df["Time"], utc=True)
        time_cols = ["Time"]
    else:
        raise ValueError(
            f"Cannot detect time format in {inject_csv}. "
            "Expected 'Period', 'Forecast_time', or 'Time' column."
        )

    value_cols = [c for c in df.columns if c not in time_cols]
    if buses:
        missing = [b for b in buses if b not in value_cols]
        if missing:
            raise ValueError(f"Bus(es) {missing} not found in {inject_csv}. "
                             f"Available: {value_cols}")
        value_cols = list(buses)

    data = df[value_cols].copy()
    data.index = times
    data.index.name = "datetime"

    # Resample to hourly (DA injections are already hourly for RTS-GMLC)
    hourly = data.resample("h").mean()
    hourly["avg_dc_power_mw"] = hourly[value_cols].sum(axis=1)
    result = hourly[["avg_dc_power_mw"]].sort_index()

    if start_date:
        result = result[result.index >= pd.Timestamp(start_date, tz="utc")]
    if end_date:
        result = result[result.index < pd.Timestamp(end_date, tz="utc")
                                      + pd.Timedelta(days=1)]
    return result


# ---------------------------------------------------------------------------
# LMP from VATIC simulation output
# ---------------------------------------------------------------------------

def compute_price_sensitivity(
    sim_base_dir: str | Path,
    sim_perturb_dir: str | Path,
    buses: Sequence[str],
) -> pd.DataFrame:
    """Estimate hourly price sensitivity β = ∂LMP/∂D from two VATIC runs.

    Run a baseline simulation and a perturbed simulation (DC load scaled by
    +ε, e.g. 10%) on the same grid, then compute:

        β(t) = (LMP_perturb(t) − LMP_base(t)) / (D_perturb(t) − D_base(t))

    β captures the local slope of the supply curve at each hour.  When
    positive, it penalises demand concentration in the price-anticipating QP
    objective, naturally spreading shifted load across hours.

    Parameters
    ----------
    sim_base_dir : path
        VATIC output directory for the baseline run.
    sim_perturb_dir : path
        VATIC output directory for the perturbed run (higher DC load).
    buses : list of str
        DC bus names (must exist in both bus_detail.csv files).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC, hourly), columns:
            beta        ∂LMP/∂D  ($/MWh per MW)
            lmp_base    baseline LMP  ($/MWh)
            delta_lmp   LMP change    ($/MWh)
            delta_d     demand change (MW)
    """
    def _load(sim_dir: Path) -> pd.DataFrame:
        df = pd.read_csv(Path(sim_dir) / "bus_detail.csv")
        df = df[df["Bus"].isin(buses)].copy()
        df["datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Hour"].astype(str) + ":00",
            format="%Y-%m-%d %H:%M", utc=True,
        )
        df["lmp_w"] = df["LMP"] * df["Demand"]
        by_hour = df.groupby("datetime")[["lmp_w", "Demand"]].sum()
        lmp_simple = df.groupby("datetime")["LMP"].mean()
        lmp_vals = np.where(
            by_hour["Demand"] > 0,
            by_hour["lmp_w"] / by_hour["Demand"],
            lmp_simple,
        )
        return pd.DataFrame(
            {"lmp": lmp_vals, "demand": by_hour["Demand"].values},
            index=by_hour.index,
        )

    base    = _load(sim_base_dir)
    perturb = _load(sim_perturb_dir)

    aligned = base.join(perturb, lsuffix="_base", rsuffix="_perturb")
    delta_lmp = aligned["lmp_perturb"]    - aligned["lmp_base"]
    delta_d   = aligned["demand_perturb"] - aligned["demand_base"]

    # β = ΔP/ΔD; guard against zero demand change (same load hours)
    beta = delta_lmp / delta_d.replace(0, np.nan)
    beta = beta.fillna(0.0).clip(lower=0.0)   # enforce non-negativity (convexity)

    return pd.DataFrame({
        "beta":      beta,
        "lmp_base":  aligned["lmp_base"],
        "delta_lmp": delta_lmp,
        "delta_d":   delta_d,
    }).sort_index()


def compute_marginal_ci(
    sim_base_dir:    str | Path,
    sim_perturb_dir: str | Path,
    buses:           Sequence[str],
    gen_csv:         str | Path,
) -> pd.DataFrame:
    """Compute hourly marginal carbon intensity (kg CO₂ / MWh) from paired sims.

    Average carbon intensity (total CO₂ / total MWh) is diluted by zero-emission
    nuclear and hydro and understates the true emission impact of an extra MW of
    load.  The marginal rate γ = ΔCO₂/ΔD captures which generator actually turns
    on (or up) at each hour in response to the DC load perturbation.

    Parameters
    ----------
    sim_base_dir : path
        VATIC output directory for the baseline run.
    sim_perturb_dir : path
        VATIC output directory for the perturbed run (higher DC load).
    buses : list of str
        DC bus names (used to compute ΔD — the load delta at the DC buses).
    gen_csv : path
        gen.csv for the grid (heat rates and emission factors).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC, hourly), columns:
            marginal_ci   γ = ΔCO₂/ΔD  (kg CO₂/MWh of additional load)
            avg_ci_base   average CI of baseline run  (kg CO₂/MWh)
            delta_co2_kg  CO₂ difference between runs (kg/h)
            delta_d       demand difference at DC buses (MW)
    """
    ef = _emission_factors(gen_csv)

    def _hourly_co2(sim_dir: Path) -> pd.Series:
        thermal = pd.read_csv(Path(sim_dir) / "thermal_detail.csv")
        thermal["datetime"] = pd.to_datetime(
            thermal["Date"] + " " + thermal["Hour"].astype(str) + ":00",
            format="%Y-%m-%d %H:%M", utc=True,
        )
        thermal["co2_kg"] = thermal["Dispatch"] * thermal["Generator"].map(ef).fillna(0.0)
        return thermal.groupby("datetime")["co2_kg"].sum()

    def _hourly_demand(sim_dir: Path) -> pd.Series:
        df = pd.read_csv(Path(sim_dir) / "bus_detail.csv")
        df = df[df["Bus"].isin(buses)].copy()
        df["datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Hour"].astype(str) + ":00",
            format="%Y-%m-%d %H:%M", utc=True,
        )
        return df.groupby("datetime")["Demand"].sum()

    co2_base    = _hourly_co2(sim_base_dir)
    co2_perturb = _hourly_co2(sim_perturb_dir)
    d_base      = _hourly_demand(sim_base_dir)
    d_perturb   = _hourly_demand(sim_perturb_dir)

    idx         = co2_base.index.union(co2_perturb.index)
    delta_co2   = co2_perturb.reindex(idx, fill_value=0.0) - co2_base.reindex(idx, fill_value=0.0)
    delta_d     = d_perturb.reindex(idx, fill_value=0.0)   - d_base.reindex(idx, fill_value=0.0)

    # γ = ΔCO₂ [kg/h] / ΔD [MWh/h] = kg CO₂ / MWh marginal load
    gamma = delta_co2 / delta_d.replace(0, np.nan)
    gamma = gamma.fillna(0.0).clip(lower=0.0)   # non-negative: more load → more CO₂

    # Also return average CI of baseline for comparison
    avg_ci_base = compute_carbon_intensity(sim_base_dir, gen_csv)["carbon_intensity"]
    avg_ci_base = avg_ci_base.reindex(idx, fill_value=0.0)

    return pd.DataFrame({
        "marginal_ci":  gamma,
        "avg_ci_base":  avg_ci_base,
        "delta_co2_kg": delta_co2,
        "delta_d":      delta_d,
    }).sort_index()


def load_lmp(
    sim_dir: str | Path,
    buses: Sequence[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Extract hourly average LMP across DC buses from a VATIC simulation.

    VATIC writes bus_detail.csv with columns Date, Hour, Bus, Demand,
    Mismatch, LMP.  This function averages the LMP across the requested
    buses (load-weighted if demand is non-zero, otherwise simple average).

    Parameters
    ----------
    sim_dir : path
        VATIC simulation output directory (contains bus_detail.csv).
    buses : list of str
        Bus names to average LMP over (should match DC load buses).
    start_date, end_date : str, optional
        Inclusive date range filter in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC, hourly), single column ``lmp`` ($/MWh).
    """
    sim_dir = Path(sim_dir)
    df = pd.read_csv(sim_dir / "bus_detail.csv")
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Hour"].astype(str) + ":00",
        format="%Y-%m-%d %H:%M",
        utc=True,
    )
    df = df[df["Bus"].isin(buses)]

    # Load-weighted average LMP across buses (vectorised, no groupby.apply)
    df["lmp_w"] = df["LMP"] * df["Demand"]
    by_hour    = df.groupby("datetime")[["lmp_w", "Demand"]].sum()
    lmp_simple = df.groupby("datetime")["LMP"].mean()
    lmp_vals   = np.where(
        by_hour["Demand"] > 0,
        by_hour["lmp_w"] / by_hour["Demand"],
        lmp_simple,
    )
    lmp = pd.DataFrame({"lmp": lmp_vals}, index=by_hour.index).sort_index()

    if start_date:
        lmp = lmp[lmp.index >= pd.Timestamp(start_date, tz="utc")]
    if end_date:
        lmp = lmp[lmp.index < pd.Timestamp(end_date, tz="utc") + pd.Timedelta(days=1)]
    return lmp


# ---------------------------------------------------------------------------
# LP-optimal Carbon-Aware Scheduler
# ---------------------------------------------------------------------------

def cas_lp(
    df_all: pd.DataFrame,
    max_cap: float,
    alpha: float = 0.5,
    ramp_rate: float | None = None,
    horizon: int = 24,
    deferral_window: int = 0,
    beta: pd.Series | None = None,
    flexible_ratio: float = 0.0,
) -> pd.DataFrame:
    """LP/QP-optimal DC load scheduler minimising cost and carbon jointly.

    **Price-taking LP** (default, ``beta=None``):

        min   Σ_τ ( α·LMP̂_τ + (1-α)·ĈI_τ ) · x_τ

    **Price-anticipating QP** (when ``beta`` is supplied):

        min   Σ_τ [ α·(LMP_base_τ + β_τ·(x_τ−A_τ)) + (1-α)·ĈI_τ ] · x_τ
            = Σ_τ  α·β_τ·x_τ²  +  [α·(LMP_base_τ−β_τ·A_τ) + (1-α)·ĈI_τ]·x_τ

        The quadratic β·x² term penalises demand concentration: the more
        load shifted into an hour, the more expensive that hour becomes in
        the model, automatically spreading the shift and avoiding the
        "2am laundry" over-concentration problem.

        β(τ) = ∂LMP/∂D estimated from a VATIC perturbation run via
        ``compute_price_sensitivity()``.

    Both modes share the same constraints:
        Σ_τ x_τ  = Σ_τ A_τ                          (energy balance)
        Σ_{i≤τ} x_i ≤ Σ_{i≤τ} A_i  ∀τ             (causal backlog)
        Σ_{i≤τ} x_i ≥ Σ_{i≤τ-W} A_i  ∀τ≥W          (deferral window, if W>0)
        0 ≤ x_τ ≤ max_cap                           (server capacity)
        |x_τ − x_{τ−1}| ≤ ramp_rate ∀τ>0           (ramp, optional)

    Parameters
    ----------
    df_all : pd.DataFrame
        DatetimeIndex (UTC, hourly).  Must contain ``avg_dc_power_mw``
        plus at least one of ``lmp`` or ``carbon_intensity``.
    max_cap : float
        Server capacity ceiling (MW).
    alpha : float
        Trade-off weight: 1.0 = cost-only, 0.0 = carbon-only.
    ramp_rate : float or None
        Max load change between hours (MW/h).  None = unconstrained.
    horizon : int
        Planning window in hours (default: 24).
    deferral_window : int
        Maximum hours a unit of work may be deferred (0 = unconstrained).
        Enforces: Σ_{i≤τ} x_i ≥ Σ_{i≤τ-W} A_i for τ ≥ W.
    beta : pd.Series or None
        Price sensitivity ∂LMP/∂D ($/MWh per MW) indexed by UTC datetime.
        When provided, switches from LP to price-anticipating QP.
        Obtain via ``compute_price_sensitivity()``.

    Returns
    -------
    pd.DataFrame
        Copy of df_all with ``avg_dc_power_mw`` replaced by the optimal
        scheduled profile.
    """
    from scipy.optimize import minimize as _minimize

    if "lmp" not in df_all.columns and "carbon_intensity" not in df_all.columns:
        raise ValueError("df_all must contain at least one of 'lmp' or 'carbon_intensity'.")

    use_qp = beta is not None
    result  = df_all.copy()

    for _, grp in df_all.groupby(df_all.index.normalize()):
        day_idx = grp.index
        for chunk_start in range(0, len(day_idx), horizon):
            chunk_idx = day_idx[chunk_start: chunk_start + horizon]
            H     = len(chunk_idx)
            chunk = grp.loc[chunk_idx]

            A      = chunk["avg_dc_power_mw"].values.astype(float)
            has_lmp = "lmp" in chunk.columns
            has_ci  = "carbon_intensity" in chunk.columns

            lmp_raw = chunk["lmp"].values.astype(float) if has_lmp else np.zeros(H)
            ci_raw  = chunk["carbon_intensity"].values.astype(float) if has_ci else np.zeros(H)

            lmp_mean = lmp_raw.mean() if lmp_raw.mean() > 0 else 1.0
            ci_mean  = ci_raw.mean()  if ci_raw.mean()  > 0 else 1.0

            cum_A  = np.cumsum(A)
            bounds = [(A[t] * (1.0 - flexible_ratio), max_cap) for t in range(H)]

            # ---- Build shared linear constraint functions ----
            def _constraints(A_arr, cum_A_arr, rr):
                cons = []
                # Energy equality: Σ x = Σ A
                cons.append({
                    "type": "eq",
                    "fun":  lambda x, s=A_arr.sum(): x.sum() - s,
                    "jac":  lambda x: np.ones(len(x)),
                })
                # Causal backlog: Σ_{i≤τ} x_i ≤ Σ_{i≤τ} A_i
                for tau in range(H - 1):
                    ca = float(cum_A_arr[tau])
                    cons.append({
                        "type": "ineq",
                        "fun":  lambda x, t=tau, b=ca: b - x[:t+1].sum(),
                        "jac":  lambda x, t=tau: np.array(
                            [-1.0 if i <= t else 0.0 for i in range(len(x))]
                        ),
                    })
                # Deferral window: Σ_{i≤τ} x_i ≥ Σ_{i≤τ-W} A_i  for τ ≥ W
                if deferral_window > 0:
                    for tau in range(deferral_window, H):
                        ca_dw = float(cum_A_arr[tau - deferral_window])
                        cons.append({
                            "type": "ineq",
                            "fun":  lambda x, t=tau, b=ca_dw: x[:t+1].sum() - b,
                            "jac":  lambda x, t=tau: np.array(
                                [1.0 if i <= t else 0.0 for i in range(len(x))]
                            ),
                        })
                # Ramp rate
                if rr is not None:
                    for tau in range(1, H):
                        cons.append({
                            "type": "ineq",
                            "fun":  lambda x, t=tau, r=rr: r - (x[t] - x[t-1]),
                            "jac":  lambda x, t=tau: np.array(
                                [0.0 if i not in (t-1, t) else
                                 (1.0 if i == t-1 else -1.0)
                                 for i in range(len(x))]
                            ),
                        })
                        cons.append({
                            "type": "ineq",
                            "fun":  lambda x, t=tau, r=rr: r - (x[t-1] - x[t]),
                            "jac":  lambda x, t=tau: np.array(
                                [0.0 if i not in (t-1, t) else
                                 (-1.0 if i == t-1 else 1.0)
                                 for i in range(len(x))]
                            ),
                        })
                return cons

            if not use_qp:
                # ---- Price-taking LP via linprog (faster) ----
                lmp_norm = lmp_raw / lmp_mean
                ci_norm  = ci_raw  / ci_mean
                c_lp     = alpha * lmp_norm + (1.0 - alpha) * ci_norm

                A_ub_rows, b_ub_rows = [], []
                for tau in range(H - 1):
                    row = np.zeros(H); row[:tau+1] = 1.0
                    A_ub_rows.append(row); b_ub_rows.append(cum_A[tau])
                if ramp_rate is not None:
                    for tau in range(1, H):
                        up = np.zeros(H); up[tau] =  1.0; up[tau-1] = -1.0
                        dn = np.zeros(H); dn[tau] = -1.0; dn[tau-1] =  1.0
                        A_ub_rows += [up, dn]
                        b_ub_rows += [ramp_rate, ramp_rate]
                if deferral_window > 0:
                    for tau in range(deferral_window, H):
                        row = np.zeros(H); row[:tau+1] = -1.0
                        A_ub_rows.append(row)
                        b_ub_rows.append(-cum_A[tau - deferral_window])

                A_ub_m = np.array(A_ub_rows) if A_ub_rows else None
                b_ub_v = np.array(b_ub_rows) if b_ub_rows else None

                res = linprog(c_lp,
                              A_ub=A_ub_m, b_ub=b_ub_v,
                              A_eq=np.ones((1, H)), b_eq=np.array([A.sum()]),
                              bounds=bounds, method="highs")
                if res.success:
                    result.loc[chunk_idx, "avg_dc_power_mw"] = res.x

            else:
                # ---- Price-anticipating QP via scipy minimize (SLSQP) ----
                beta_raw = beta.reindex(chunk_idx).fillna(0.0).values.astype(float)

                # Quadratic coefficients: q_τ = α·β_τ
                q = alpha * beta_raw

                # Linear coefficients: c_τ = α·(LMP_base - β·A) + (1-α)·CI_norm
                ci_norm = ci_raw / ci_mean
                c_qp    = alpha * (lmp_raw - beta_raw * A) + (1.0 - alpha) * ci_norm * lmp_mean

                def objective(x):
                    return float(np.dot(q, x**2) + np.dot(c_qp, x))

                def jac(x):
                    return 2.0 * q * x + c_qp

                x0 = A.copy()  # warm-start from original load
                res = _minimize(
                    objective, x0, jac=jac, method="SLSQP",
                    bounds=bounds,
                    constraints=_constraints(A, cum_A, ramp_rate),
                    options={"ftol": 1e-9, "maxiter": 1000},
                )
                if res.success:
                    result.loc[chunk_idx, "avg_dc_power_mw"] = res.x

    return result


# ---------------------------------------------------------------------------
# Carbon-Aware Scheduler
# ---------------------------------------------------------------------------

def cas_grid_mix(
    df_all: pd.DataFrame,
    flexible_work_ratio: float,
    max_cap: float,
    window: int = 12,
) -> pd.DataFrame:
    """Causal rolling-window shift of flexible DC load toward low-CI hours.

    Processes hours sequentially. Flexible work arriving at hour t may be
    deferred up to ``window`` hours (deadline = t + window). At each step:

      1. New flexible energy (load(t) * fr) is added to the pool with its
         deadline.
      2. Pool items whose deadline has arrived are force-consumed at t; any
         overflow beyond max_cap spills to t+1.
      3. Remaining items are committed greedily in earliest-deadline order
         using a look-ahead rule: commit at t if CI(t) ≤ min(CI(t+1 … deadline)).

    End-of-simulation drain: items from the last ``window`` hours whose
    deadline exceeds the simulation length are distributed into the lowest-CI
    available slots in the tail, then any residual is absorbed at T-1
    regardless of max_cap to preserve energy balance.

    Parameters
    ----------
    df_all : pd.DataFrame
        DatetimeIndex (UTC, hourly).  Must contain:
            avg_dc_power_mw   DC load in MW
            carbon_intensity  Grid CI in kg CO₂/MWh
    flexible_work_ratio : float
        Percentage (0–100) of each hour's load that may be deferred.
    max_cap : float
        Maximum DC load in MW (server capacity ceiling).
    window : int
        Maximum deferral horizon in hours (default 12).

    Returns
    -------
    pd.DataFrame
        Copy of df_all with ``avg_dc_power_mw`` replaced by the scheduled
        profile.  All other columns are preserved unchanged.
    """
    fr       = flexible_work_ratio / 100.0
    T        = len(df_all)
    load     = df_all["avg_dc_power_mw"].values.astype(float)
    ci       = df_all["carbon_intensity"].values.astype(float)
    schedule = load * (1.0 - fr)   # non-deferrable baseline
    pool     = []                   # list of [mwh_remaining, deadline_index]

    for t in range(T):
        # --- 1. New flexible work arrives at t ---
        pool.append([load[t] * fr, t + window])

        # --- 2. Force-expire: deadline <= t must be consumed now ---
        headroom  = max_cap - schedule[t]
        surviving = []
        for item in pool:
            amount, deadline = item
            if deadline <= t:
                absorb       = min(amount, max(0.0, headroom))
                schedule[t] += absorb
                headroom    -= absorb
                spill        = amount - absorb
                if spill > 1e-6:
                    if t + 1 < T:
                        surviving.append([spill, t + 1])   # spill to next hour
                    else:
                        schedule[t] += spill               # absorb at last hour
            else:
                surviving.append(item)
        pool = surviving

        # --- 3. Look-ahead commit (earliest deadline first) ---
        pool.sort(key=lambda x: x[1])
        headroom = max_cap - schedule[t]
        for item in pool:
            if headroom < 1e-6:
                break
            amount, deadline = item
            w_end       = min(deadline, T)
            best_future = float(np.min(ci[t + 1 : w_end])) if t + 1 < w_end else np.inf
            if ci[t] <= best_future:
                absorb       = min(amount, headroom)
                schedule[t] += absorb
                headroom    -= absorb
                item[0]     -= absorb

        pool = [item for item in pool if item[0] > 1e-6]

    # --- End-of-simulation drain ---
    # Items from the last `window` hours whose deadline exceeds T couldn't be
    # consumed by the look-ahead (max_cap was the binding constraint). Place
    # them into the lowest-CI available slots in the tail; absorb any residual
    # at T-1 to guarantee energy balance.
    if pool:
        remaining  = sum(item[0] for item in pool)
        tail_start = max(0, T - window)
        order      = np.argsort(ci[tail_start:]) + tail_start   # lowest CI first
        for h in order:
            if remaining < 1e-6:
                break
            headroom = max_cap - schedule[h]
            if headroom > 0:
                absorb       = min(remaining, headroom)
                schedule[h] += absorb
                remaining   -= absorb
        if remaining > 1e-6:
            schedule[T - 1] += remaining   # last resort: absorb past max_cap

    result = df_all.copy()
    result["avg_dc_power_mw"] = schedule
    return result


# ---------------------------------------------------------------------------
# Renewable supply from VATIC simulation outputs
# ---------------------------------------------------------------------------

def load_renewable_supply(
    sim_dir: str | Path,
    gen_csv: str | Path,
    generators: Sequence[str] | None = None,
    fuels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build an hourly renewable generation series from a vatic sim run.

    Parameters
    ----------
    sim_dir : path
        Vatic simulation output directory (contains renew_detail.csv).
    gen_csv : path
        gen.csv for the grid (used to look up Fuel labels).
    generators : list of str, optional
        Restrict to specific GEN UIDs (e.g. the co-located units added by
        add_renewables.py).  Defaults to all non-dispatchable generators.
    fuels : list of str, optional
        Restrict by fuel type (e.g. ['Solar', 'Wind']).  Defaults to both.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC, hourly), columns:
            tot_renewable   MW — sum of selected generators' output
            solar_mw        MW — solar component
            wind_mw         MW — wind component
    """
    sim_dir = Path(sim_dir)
    gen_df  = pd.read_csv(gen_csv)
    fuel_map = dict(zip(gen_df["GEN UID"], gen_df["Fuel"]))

    renew = pd.read_csv(sim_dir / "renew_detail.csv")
    renew["datetime"] = pd.to_datetime(
        renew["Date"] + " " + renew["Hour"].astype(str) + ":00",
        format="%Y-%m-%d %H:%M",
        utc=True,
    )
    renew["Fuel"] = renew["Generator"].map(fuel_map).fillna("Unknown")

    # Normalise fuel labels: TX uses "WND (Wind)"/"SUN (Solar)", RTS uses "Wind"/"Solar".
    renew["_fuel_norm"] = (renew["Fuel"]
                           .str.replace(r".*\bWind\b.*",  "Wind",  regex=True, case=False)
                           .str.replace(r".*\bSolar\b.*", "Solar", regex=True, case=False))

    if generators is not None:
        renew = renew[renew["Generator"].isin(generators)]
    if fuels is not None:
        renew = renew[renew["_fuel_norm"].isin(fuels)]
    else:
        renew = renew[renew["_fuel_norm"].isin(["Solar", "Wind"])]

    solar = (renew[renew["_fuel_norm"] == "Solar"]
             .groupby("datetime")["Output"].sum()
             .rename("solar_mw"))
    wind  = (renew[renew["_fuel_norm"] == "Wind"]
             .groupby("datetime")["Output"].sum()
             .rename("wind_mw"))

    result = pd.concat([solar, wind], axis=1).fillna(0.0)
    result["tot_renewable"] = result["solar_mw"] + result["wind_mw"]
    return result.sort_index()


# ---------------------------------------------------------------------------
# 24/7 Carbon-Aware Scheduler
# ---------------------------------------------------------------------------

def cas_24_7(
    df_all: pd.DataFrame,
    flexible_work_ratio: float,
    max_cap: float,
    window: int = 12,
) -> pd.DataFrame:
    """Causal rolling-window shift of flexible DC load toward high-renewable hours.

    Processes hours sequentially. Flexible work arriving at hour t may be
    deferred up to ``window`` hours. Force-expiry and overflow spill follow
    the same rules as cas_grid_mix.

    Look-ahead commit rule: commit pool energy at t if the current renewable
    surplus estimate (R(t) − min_load(t)) is at least as large as the maximum
    surplus estimate anywhere in the remaining deferral window.  The surplus
    estimate uses the fixed min_load baseline for all hours so the signal is
    stable across the scheduling loop.

    Committed energy fills the coverage band (up to R(t)) before the capacity
    ceiling (up to max_cap), preserving the 24/7 coverage priority.

    Parameters
    ----------
    df_all : pd.DataFrame
        DatetimeIndex (UTC, hourly).  Must contain:
            avg_dc_power_mw   DC load in MW
            tot_renewable     Available renewable supply in MW
    flexible_work_ratio : float
        Percentage (0–100) of each hour's load that may be deferred.
    max_cap : float
        Maximum DC load in MW (server capacity ceiling).
    window : int
        Maximum deferral horizon in hours (default 12).

    Returns
    -------
    pd.DataFrame
        Copy of df_all with ``avg_dc_power_mw`` replaced by the scheduled
        profile.  All other columns (including tot_renewable) are preserved.
    """
    fr               = flexible_work_ratio / 100.0
    T                = len(df_all)
    load             = df_all["avg_dc_power_mw"].values.astype(float)
    renew            = df_all["tot_renewable"].values.astype(float)
    min_load         = load * (1.0 - fr)
    schedule         = min_load.copy()
    surplus_estimate = renew - min_load     # fixed look-ahead signal
    pool             = []                   # [mwh_remaining, deadline_index]

    for t in range(T):
        # --- 1. New flexible work arrives ---
        pool.append([load[t] * fr, t + window])

        # --- 2. Force-expire ---
        headroom  = max_cap - schedule[t]
        surviving = []
        for item in pool:
            amount, deadline = item
            if deadline <= t:
                absorb       = min(amount, max(0.0, headroom))
                schedule[t] += absorb
                headroom    -= absorb
                spill        = amount - absorb
                if spill > 1e-6:
                    if t + 1 < T:
                        surviving.append([spill, t + 1])
                    else:
                        schedule[t] += spill
            else:
                surviving.append(item)
        pool = surviving

        # --- 3. Look-ahead commit (earliest deadline first) ---
        pool.sort(key=lambda x: x[1])
        for item in pool:
            if max_cap - schedule[t] < 1e-6:
                break
            amount, deadline = item
            w_end       = min(deadline, T)
            best_future = float(np.max(surplus_estimate[t + 1 : w_end])) if t + 1 < w_end else -np.inf
            if surplus_estimate[t] >= best_future:
                # Coverage pass: fill up to R(t) first
                absorb       = min(amount, max(0.0, renew[t] - schedule[t]), max_cap - schedule[t])
                schedule[t] += absorb
                item[0]     -= absorb
                # Energy-balance pass: fill remaining up to max_cap
                if item[0] > 1e-6:
                    absorb       = min(item[0], max_cap - schedule[t])
                    schedule[t] += absorb
                    item[0]     -= absorb

        pool = [item for item in pool if item[0] > 1e-6]

    # --- End-of-simulation drain ---
    # Items from the last `window` hours whose deadline exceeds T couldn't be
    # consumed by the look-ahead. Place them into the highest-surplus available
    # slots in the tail (coverage first, then capacity); absorb any residual at
    # T-1 to guarantee energy balance.
    if pool:
        remaining  = sum(item[0] for item in pool)
        tail_start = max(0, T - window)
        order      = np.argsort(-surplus_estimate[tail_start:]) + tail_start  # highest surplus first
        for h in order:
            if remaining < 1e-6:
                break
            # Coverage pass
            cov_head = max(0.0, renew[h] - schedule[h])
            if cov_head > 0:
                absorb       = min(remaining, cov_head, max_cap - schedule[h])
                schedule[h] += absorb
                remaining   -= absorb
            # Capacity pass
            if remaining > 1e-6:
                cap_head = max_cap - schedule[h]
                if cap_head > 0:
                    absorb       = min(remaining, cap_head)
                    schedule[h] += absorb
                    remaining   -= absorb
        if remaining > 1e-6:
            schedule[T - 1] += remaining   # last resort: absorb past max_cap

    result = df_all.copy()
    result["avg_dc_power_mw"] = schedule
    return result


# ---------------------------------------------------------------------------
# Battery storage and scalar summary helpers
# ---------------------------------------------------------------------------

def calculate_battery_capacity(
    tot_renewable: pd.Series,
    dc_load: pd.Series,
) -> float:
    """Minimum battery capacity (MWh) needed for 100% 24/7 renewable coverage.

    The battery charges greedily whenever renewable supply exceeds DC load and
    discharges whenever DC load exceeds supply.  Charging speed is unconstrained
    (can absorb the full hourly surplus in one hour).

    The minimum capacity is derived analytically from the cumulative net-energy
    series, allowing for an optimal initial state-of-charge:

        net(t)   = ren(t) − load(t)
        cum(t)   = Σ net(0..t)   (prefix sum, with cum[-1] = 0)
        S0       = max(0, −min(cum))   # initial charge to avoid underflow
        C        = S0 + max(cum)        # capacity to avoid overflow

    Parameters
    ----------
    tot_renewable : pd.Series
        Hourly renewable supply aligned to dc_load's index (MW).
    dc_load : pd.Series
        Hourly DC load (MW).

    Returns
    -------
    float
        Minimum battery capacity in MWh.
        Returns 0.0 if renewable supply ≥ load at every hour (no battery needed).
        Returns NaN if total renewable energy < total load energy (infeasible).
    """
    ren  = tot_renewable.reindex(dc_load.index).fillna(0.0).values
    load = dc_load.values
    net  = ren - load

    # Already fully covered: no battery needed
    if (net >= -1e-9).all():
        return 0.0

    # Infeasible: total energy deficit cannot be bridged
    if net.sum() < -1e-9:
        return float("nan")

    # Minimum buffer with optimal initial SoC
    cum = np.concatenate([[0.0], np.cumsum(net)])
    S0  = max(0.0, -float(cum.min()))   # initial charge to prevent underflow
    C   = S0 + float(cum.max())          # total capacity to prevent overflow
    return float(max(0.0, C))


def carbon_cost(df: pd.DataFrame) -> float:
    """Compute total carbon cost: sum(load * CI) in kg CO2."""
    return float((df["avg_dc_power_mw"] * df["carbon_intensity"]).sum())


def calculate_coverage(df: pd.DataFrame) -> float:
    """Compute 24/7 renewable coverage as a percentage.

    Coverage(%) = sum_t min(R(t), D(t)) / sum_t D(t) * 100

    where R(t) = tot_renewable and D(t) = avg_dc_power_mw.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``tot_renewable`` and ``avg_dc_power_mw``.

    Returns
    -------
    float
        Coverage percentage in [0, 100].
    """
    covered = np.minimum(
        df["tot_renewable"].values,
        df["avg_dc_power_mw"].values,
    ).sum()
    total = df["avg_dc_power_mw"].sum()
    return float(covered / total * 100) if total > 0 else 0.0
