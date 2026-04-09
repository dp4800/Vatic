#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
sim_lp_joint.py — Joint LP for carbon-aware data center scheduling.

Problem
-------
The iterative sim-lp approach (cas_lp) computes LMPs from a baseline
simulation, optimises the DC shift against those prices, then re-runs VATIC
with the shifted load.  The LMPs in the re-run differ from the ones used to
compute the shift — the signal is endogenous.

The joint LP resolves this by making DC scheduling variables *e[b,t]* and SCED
dispatch variables *p[g,t]* part of the same optimisation.  The prices the
scheduler sees are exactly consistent with the dispatch they induce: they
emerge as the dual variables of the power-balance equality constraint.

Formulation
-----------
Given a fixed unit-commitment schedule u[g,t] from a baseline VATIC run,
solve:

    min   α  · Σ_{g,t} mc[g]  · p[g,t]           (operational cost)
        + (1-α) · Σ_{b,t} CI[t] · e[b,t] / CI_mean  (DC carbon exposure)

subject to
    Power balance       Σ_g p[g,t] + renew[t] = L_nodc[t] + Σ_b e[b,t]   ∀t
    DC energy balance   Σ_{t∈day} e[b,t]  = Σ_{t∈day} d0[b,t]            ∀b,day
    DC causal           Σ_{τ≤t} e[b,τ]   ≤ Σ_{τ≤t} d0[b,τ]              ∀b,t
    DC deferral window  Σ_{τ≤t} e[b,τ]   ≥ Σ_{τ≤t-W} d0[b,τ]            ∀b,t
    Gen ramp-up         p[g,t] - p[g,t-1] ≤ RampUp[g]                    ∀g,t>0
    Gen ramp-down       p[g,t-1] - p[g,t] ≤ RampDown[g]                  ∀g,t>0
    Line flows          |PTDF · Pnet[t]|   ≤ ThermalLimit                 ∀lines,t
    Bounds              PMin·u ≤ p ≤ PMax·u;   lb_e ≤ e ≤ ub_e

The PTDF matrix is derived from first principles using the network susceptance
structure in branch.csv (DC power-flow approximation, reference bus = Arne/113).

Outputs
-------
- New DAY_AHEAD_bus_injections.csv in the output grid's BusInjections dir
- comparison_table.csv: joint-LP predicted LMPs vs baseline LMPs, predicted
  vs simulated CO₂ and cost (post-shift VATIC dir optional)

Integration with main.py
------------------------
Controlled by params.json  "cas": {"lp_mode": "joint" | "iterative"}.
When lp_mode == "joint", main.py calls run() instead of the
analyze_cas.py + apply_cas_shift.py pipeline for the LP step.

CLI usage
---------
    python scripts/sim_lp_joint.py \\
        --baseline-dir   outputs/2020-02-17/baseline \\
        --grid           RTS-GMLC-DC \\
        --output-grid    RTS-GMLC-DC-CAS-LP \\
        --buses          Abel Adler Attar Attlee Bach Balzac Beethoven Cabell Caesar Clark \\
        --start-date     2020-02-17 \\
        --days           7 \\
        --alpha          0.5 \\
        --deferral-window 12 \\
        --flexible-ratio  0.30 \\
        --headroom        0.30 \\
        --post-shift-dir  outputs/2020-02-17/sim-lp     # optional, for residuals
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

_SCRIPTS_DIR = Path(__file__).resolve().parent
_VATIC_ROOT  = _SCRIPTS_DIR.parent
_GRIDS_DIR   = _VATIC_ROOT / "vatic" / "data" / "grids"

GRID_REGISTRY = {
    "RTS-GMLC":      "RTS_Data",
    "Texas-7k_2030": "TX2030_Data",
    "Texas-7k":      "TX_Data",
}

RENEWABLE_FUELS = {"Wind", "Solar", "Hydro", "CSP", "RTPV"}

# Fallback emission factors (kg CO2/MWh) for grids without HR/emission columns.
# Source: EPA eGRID 2023 TX median + EIA CO2 coefficients.
_FALLBACK_EF_KG: dict[str, float] = {
    "coal": 1078.5, "lignite": 1078.5, "subbituminous": 1078.5,
    "petroleum coke": 1021.2, "oil": 795.8,
    "natural gas": 496.3, "ng": 496.3, "gas": 496.3,
    "biomass": 54.0, "wood": 54.0,
}


def _marginal_cost(gen_df: pd.DataFrame) -> pd.Series:
    """Return marginal cost ($/MWh) indexed by GEN UID.

    RTS-GMLC: HR_avg_0 [BTU/kWh] * FuelPrice [$/MMBTU] + VOM [$/MWh]
    Texas-7k: MWh Price 1 [$/MWh] + Variable O&M [$/MWh]  (piecewise cost curve)
    """
    if "HR_avg_0" in gen_df.columns and "Fuel Price $/MMBTU" in gen_df.columns:
        hr  = pd.to_numeric(gen_df["HR_avg_0"], errors="coerce").fillna(0.0) / 1000.0
        fp  = pd.to_numeric(gen_df["Fuel Price $/MMBTU"], errors="coerce").fillna(0.0)
        vom_col = "VOM" if "VOM" in gen_df.columns else None
        vom = pd.to_numeric(gen_df[vom_col], errors="coerce").fillna(0.0) if vom_col else 0.0
        return hr * fp + vom
    # Texas-7k piecewise cost curve: first break-point price + variable O&M
    mc = pd.Series(0.0, index=gen_df.index)
    if "MWh Price 1" in gen_df.columns:
        mc = mc + pd.to_numeric(gen_df["MWh Price 1"], errors="coerce").fillna(0.0)
    vom_col = "Variable O&M" if "Variable O&M" in gen_df.columns else None
    if vom_col:
        mc = mc + pd.to_numeric(gen_df[vom_col], errors="coerce").fillna(0.0)
    return mc


def _ef_co2(gen_df: pd.DataFrame) -> pd.Series:
    """Return CO2 emission factor (kg CO2/MWh) indexed by GEN UID."""
    if "HR_avg_0" in gen_df.columns and "Emissions CO2 Lbs/MMBTU" in gen_df.columns:
        hr      = pd.to_numeric(gen_df["HR_avg_0"], errors="coerce").fillna(0.0) / 1000.0
        co2_lbs = pd.to_numeric(gen_df["Emissions CO2 Lbs/MMBTU"], errors="coerce").fillna(0.0)
        return hr * co2_lbs * 0.453592   # kg CO2/MWh
    fuel_col = next((c for c in gen_df.columns if c.strip().lower() == "fuel"), None)
    fuels    = gen_df[fuel_col].astype(str).str.lower() if fuel_col else pd.Series([""] * len(gen_df), index=gen_df.index)
    return fuels.map(lambda f: next((v for k, v in _FALLBACK_EF_KG.items() if k in f), 0.0))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PTDF construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_ptdf(branch_csv: Path, bus_csv: Path) -> dict:
    """Build PTDF matrix from branch and bus data (DC power-flow approximation).

    Uses the B-matrix method:
        B_bus · θ = P_net   →   θ = B_red⁻¹ · P_net_red
        f_l  = (θ_from − θ_to) / X_l  =  B_f · θ
        PTDF = B_f_red · B_red⁻¹   (padded with 0 column at reference bus)

    Transformer branches (Tr Ratio ≠ 0) are treated as plain reactances in
    the DC approximation (tap ratio ignored).

    Returns dict with keys:
        ptdf             np.ndarray (n_lines, n_buses)
        bus_ids          sorted list[int]
        bus_name_by_id   dict int→str
        bus_idx          dict int→int  (Bus ID → column index)
        line_uids        list[str]
        thermal_limits   np.ndarray (n_lines,)  MW continuous ratings
        ref_bus_id       int
    """
    branch_df = pd.read_csv(branch_csv)
    bus_df    = pd.read_csv(bus_csv)

    bus_ids      = sorted(bus_df["Bus ID"].astype(int).tolist())
    bus_idx      = {bid: i for i, bid in enumerate(bus_ids)}
    bus_name_by_id = dict(zip(bus_df["Bus ID"].astype(int), bus_df["Bus Name"]))
    n = len(bus_ids)

    # Reference bus (slack)
    ref_mask = bus_df["Bus Type"].str.strip() == "Ref"
    if not ref_mask.any():
        warnings.warn("No 'Ref' bus found; using first bus as reference.")
        ref_bus_id = bus_ids[0]
    else:
        ref_bus_id = int(bus_df.loc[ref_mask, "Bus ID"].iloc[0])
    ref_idx = bus_idx[ref_bus_id]

    # Build B_bus and B_f
    B_bus = np.zeros((n, n))
    n_lines = len(branch_df)
    B_f    = np.zeros((n_lines, n))
    thermal_limits = np.zeros(n_lines)
    line_uids      = []

    for li, (_, row) in enumerate(branch_df.iterrows()):
        fb  = int(row["From Bus"])
        tb  = int(row["To Bus"])
        X   = float(row["X"])
        fi  = bus_idx[fb]
        ti  = bus_idx[tb]
        lim = float(row["Cont Rating"])
        if np.isnan(lim) or lim <= 0:
            lim = 9999.0
        line_uids.append(str(row["UID"]))
        thermal_limits[li] = lim

        if X == 0.0 or np.isnan(X):
            continue
        b_l = 1.0 / X
        B_bus[fi, fi] += b_l
        B_bus[ti, ti] += b_l
        B_bus[fi, ti] -= b_l
        B_bus[ti, fi] -= b_l
        B_f[li, fi]    =  b_l
        B_f[li, ti]    = -b_l

    # Remove reference bus row/col
    non_ref  = [i for i in range(n) if i != ref_idx]
    B_red    = B_bus[np.ix_(non_ref, non_ref)]   # (n-1) × (n-1)
    B_f_red  = B_f[:, non_ref]                    # n_lines × (n-1)

    # PTDF_red = B_f_red @ inv(B_red)
    # Solve B_red.T @ X = B_f_red.T  →  X = PTDF_red.T
    try:
        PTDF_red = np.linalg.solve(B_red.T, B_f_red.T).T
    except np.linalg.LinAlgError:
        # Fallback: pseudoinverse if singular
        warnings.warn("B_red singular; using pseudoinverse for PTDF.")
        PTDF_red = (B_f_red @ np.linalg.pinv(B_red))

    PTDF = np.zeros((n_lines, n))
    PTDF[:, non_ref] = PTDF_red

    return {
        "ptdf":           PTDF,
        "bus_ids":        bus_ids,
        "bus_name_by_id": bus_name_by_id,
        "bus_idx":        bus_idx,
        "line_uids":      line_uids,
        "thermal_limits": thermal_limits,
        "ref_bus_id":     ref_bus_id,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Baseline data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_datetime_col(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        df["Date"] + " " + df["Hour"].astype(str) + ":00",
        format="%Y-%m-%d %H:%M", utc=True,
    )


def load_baseline(
    baseline_dir: Path,
    gen_csv: Path,
    bus_csv: Path,
    inject_csv: Path,
    dc_buses: Sequence[str],
    start_date: str,
    n_days: int,
    commit_dir: Path | None = None,
) -> dict:
    """Load all data needed to build the joint LP.

    Parameters
    ----------
    baseline_dir  : path to a VATIC simulation output directory
    gen_csv       : path to SourceData/gen.csv
    bus_csv       : path to SourceData/bus.csv
    inject_csv    : path to DAY_AHEAD_bus_injections.csv
    dc_buses      : list of DC bus names
    start_date    : 'YYYY-MM-DD'
    n_days        : simulation length

    Returns dict with keys:
        hours          DatetimeIndex, length T
        thermal_gens   list of gen UIDs (thermal, ordered)
        gen_info       DataFrame  index=GEN UID, cols: bus_id, PMin, PMax, ramp_mwh, mc
        commit         DataFrame  (T, n_thermal)  unit state 0/1
        p_init         Series  GEN UID → initial dispatch (MW) before hour 0
        renew_by_bus   DataFrame  (T, n_buses)  MW, indexed by DatetimeIndex
        L_nodc         DataFrame  (T, n_buses)  MW non-DC load per bus
        d0             DataFrame  (T, n_dc_buses)  MW baseline DC injection per bus
        lmp_baseline   DataFrame  (T, n_buses)  $/MWh LMPs from baseline
        ci_baseline    Series     (T,)  kg CO₂/MWh
        bus_name_to_id dict  bus_name → bus_id (int)
        dc_buses       list of DC bus names (validated)
    """
    baseline_dir = Path(baseline_dir)
    t_start = pd.Timestamp(start_date, tz="utc")
    t_end   = t_start + pd.Timedelta(hours=n_days * 24 - 1)
    hours   = pd.date_range(t_start, t_end, freq="h", tz="utc")
    T       = len(hours)

    # ── Gen metadata ──────────────────────────────────────────────────────────
    gen_df = pd.read_csv(gen_csv)
    gen_df["GEN UID"] = gen_df["GEN UID"].astype(str)
    gen_df = gen_df.set_index("GEN UID")

    bus_id_map = gen_df["Bus ID"].astype(int).to_dict()

    # Marginal cost ($/MWh) — handles RTS-GMLC and Texas-7k cost curve formats
    mc = _marginal_cost(gen_df)

    ramp_mwmin = pd.to_numeric(gen_df["Ramp Rate MW/Min"], errors="coerce").fillna(0.0)
    ramp_mwh   = ramp_mwmin * 60.0   # MW/h

    gen_info = pd.DataFrame({
        "bus_id": gen_df["Bus ID"].astype(int),
        "PMin":   pd.to_numeric(gen_df["PMin MW"], errors="coerce").fillna(0.0),
        "PMax":   pd.to_numeric(gen_df["PMax MW"], errors="coerce").fillna(0.0),
        "ramp_mwh": ramp_mwh,
        "mc":     mc,
        "fuel":   gen_df["Fuel"],
    })

    # ── Thermal commitment from thermal_detail.csv ────────────────────────────
    # commit_dir overrides the source of the unit-commitment schedule so that
    # an iterative caller can feed in a post-shift VATIC run's UC without
    # changing the CI/LMP signal (which always comes from baseline_dir).
    _commit_src = Path(commit_dir) if commit_dir is not None else baseline_dir
    th = pd.read_csv(_commit_src / "thermal_detail.csv")
    th["datetime"] = _parse_datetime_col(th)
    th = th[(th["datetime"] >= hours[0]) & (th["datetime"] <= hours[-1])]

    # Pivot: rows=datetime, cols=Generator, values=Unit State
    commit = (
        th.pivot_table(index="datetime", columns="Generator",
                       values="Unit State", aggfunc="first")
        .reindex(hours)
        .fillna(0)
        .astype(float)   # bools or ints → float first
        .astype(int)
    )
    dispatch_pivot = (
        th.pivot_table(index="datetime", columns="Generator",
                       values="Dispatch", aggfunc="first")
        .reindex(hours)
        .fillna(0.0)
    )

    thermal_gens = list(commit.columns)

    # Initial dispatch for ramp constraint at t=0
    # Use the very first hour's dispatch as the "previous hour" warm-start
    p_init = dispatch_pivot.iloc[0]  # Series: GEN UID → MW

    # Per-generator per-hour max observed baseline dispatch — used in
    # build_and_solve() to relax PMax bounds for storage / emergency dispatch.
    p_baseline_max = dispatch_pivot.max(axis=0)   # Series: GEN UID → max MW

    # ── Renewable output by bus ───────────────────────────────────────────────
    re = pd.read_csv(baseline_dir / "renew_detail.csv")
    re["datetime"] = _parse_datetime_col(re)
    re = re[(re["datetime"] >= hours[0]) & (re["datetime"] <= hours[-1])]
    re["bus_id"] = re["Generator"].map(bus_id_map)
    re = re.dropna(subset=["bus_id"])
    re["bus_id"] = re["bus_id"].astype(int)

    bus_df_   = pd.read_csv(bus_csv)
    all_bus_ids   = sorted(bus_df_["Bus ID"].astype(int).tolist())
    bus_name_to_id = dict(zip(bus_df_["Bus Name"].str.strip(), bus_df_["Bus ID"].astype(int)))

    renew_by_bus = (
        re.groupby(["datetime", "bus_id"])["Output"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=hours, columns=all_bus_ids, fill_value=0.0)
    )

    # ── Bus demand from bus_detail.csv ────────────────────────────────────────
    bd = pd.read_csv(baseline_dir / "bus_detail.csv")
    bd["datetime"] = _parse_datetime_col(bd)
    bd["bus_id"]   = bd["Bus"].map(bus_name_to_id)
    bd = bd[(bd["datetime"] >= hours[0]) & (bd["datetime"] <= hours[-1])]
    bd = bd.dropna(subset=["bus_id"])
    bd["bus_id"] = bd["bus_id"].astype(int)

    demand_by_bus = (
        bd.pivot_table(index="datetime", columns="bus_id", values="Demand", aggfunc="first")
        .reindex(index=hours, columns=all_bus_ids, fill_value=0.0)
    )
    lmp_by_bus = (
        bd.pivot_table(index="datetime", columns="bus_id", values="LMP", aggfunc="first")
        .reindex(index=hours, columns=all_bus_ids, fill_value=np.nan)
    )
    # Rename columns to bus_name for lmp output
    id_to_name = {v: k for k, v in bus_name_to_id.items()}
    lmp_baseline = lmp_by_bus.rename(columns=id_to_name)

    # ── DC injection baseline d0 ──────────────────────────────────────────────
    inj = pd.read_csv(inject_csv)
    if "Period" in inj.columns:
        _period_offset = 1 if inj["Period"].min() >= 1 else 0
        inj["datetime"] = pd.to_datetime(
            {"year": inj["Year"], "month": inj["Month"],
             "day":  inj["Day"],  "hour":  inj["Period"] - _period_offset},
            utc=True,
        )
    elif "Forecast_time" in inj.columns:
        inj["datetime"] = pd.to_datetime(inj["Forecast_time"], utc=True)
    else:
        inj["datetime"] = pd.to_datetime(inj["Time"], utc=True)

    inj = inj.set_index("datetime")
    dc_buses_valid = [b for b in dc_buses if b in inj.columns]
    missing = [b for b in dc_buses if b not in inj.columns]
    if missing:
        warnings.warn(f"DC buses not found in BusInjections CSV: {missing}")

    d0 = (
        inj[dc_buses_valid]
        .resample("h").mean()
        .reindex(hours, fill_value=0.0)
    )

    # ── Non-DC load per bus: L_nodc[b,t] = demand[b,t] − d0_bus[b,t] ─────────
    L_nodc = demand_by_bus.copy()
    for bname in dc_buses_valid:
        bid = bus_name_to_id[bname]
        if bid in L_nodc.columns:
            L_nodc[bid] = (L_nodc[bid] - d0[bname]).clip(lower=0.0)

    # ── Carbon intensity from baseline (used as CI signal in objective) ────────
    ef_co2 = _ef_co2(gen_df)   # kg CO₂/MWh

    # CI signal always comes from baseline_dir so the carbon objective stays
    # stable across iterations even as the UC is refreshed from commit_dir.
    if commit_dir is not None:
        _th_ci = pd.read_csv(baseline_dir / "thermal_detail.csv")
        _th_ci["datetime"] = _parse_datetime_col(_th_ci)
        _th_ci = _th_ci[(_th_ci["datetime"] >= hours[0]) & (_th_ci["datetime"] <= hours[-1])]
        th2 = _th_ci.copy()
    else:
        th2 = th.copy()
    th2["co2_kg"] = th2["Dispatch"] * th2["Generator"].map(ef_co2).fillna(0.0)
    hourly_co2 = th2.groupby("datetime")["co2_kg"].sum().reindex(hours, fill_value=0.0)
    hourly_th  = th2.groupby("datetime")["Dispatch"].sum().reindex(hours, fill_value=0.0)
    hourly_re  = re.groupby("datetime")["Output"].sum().reindex(hours, fill_value=0.0)
    total_gen  = hourly_th + hourly_re
    ci_baseline = pd.Series(
        np.where(total_gen > 0, hourly_co2 / total_gen, 0.0),
        index=hours, name="carbon_intensity",
    )

    # ── Thermal dispatch aggregated by bus (for price-taking LP PTDF) ────────────
    _valid_gens = [g for g in dispatch_pivot.columns if g in gen_info.index]
    if _valid_gens:
        _dc = dispatch_pivot[_valid_gens].copy()
        _dc.columns = [int(gen_info.loc[g, "bus_id"]) for g in _valid_gens]
        p_baseline_by_bus = _dc.T.groupby(level=0).sum().T
        p_baseline_by_bus = p_baseline_by_bus.reindex(columns=all_bus_ids, fill_value=0.0)
    else:
        p_baseline_by_bus = pd.DataFrame(0.0, index=hours, columns=all_bus_ids)

    return {
        "hours":              hours,
        "T":                  T,
        "thermal_gens":       thermal_gens,
        "gen_info":           gen_info,
        "commit":             commit,
        "p_init":             p_init,
        "p_baseline_max":     p_baseline_max,
        "p_baseline_by_bus":  p_baseline_by_bus,
        "renew_by_bus":       renew_by_bus,
        "L_nodc":             L_nodc,
        "d0":                 d0,
        "lmp_baseline":       lmp_baseline,
        "ci_baseline":        ci_baseline,
        "all_bus_ids":        all_bus_ids,
        "bus_name_to_id":     bus_name_to_id,
        "dc_buses":           dc_buses_valid,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Joint LP construction and solution
# ═══════════════════════════════════════════════════════════════════════════════

def build_and_solve(
    data: dict,
    ptdf_data: dict,
    alpha: float = 0.5,
    deferral_window: int = 12,
    flexible_ratio: float = 0.30,
    headroom: float = 0.30,
    ptdf_threshold: float = 1e-4,
    lf_margin: float = 0.10,
) -> dict:
    """Build and solve the joint LP.

    Parameters
    ----------
    data            : output of load_baseline()
    ptdf_data       : output of build_ptdf()
    alpha           : weight on operational cost (0=carbon-only, 1=cost-only)
    deferral_window : max deferral in hours
    flexible_ratio  : fraction of DC load that is flexible
    headroom        : max capacity increase above baseline peak (fraction)
    ptdf_threshold  : PTDF entries below this are treated as zero (line screening)

    Returns dict with keys:
        status           str  ('optimal' | 'infeasible' | ...)
        p_opt            DataFrame  (T, n_thermal)  optimal dispatch MW
        e_opt            DataFrame  (T, n_dc)       optimal DC injection MW
        lmp_predicted    DataFrame  (T, n_buses)    predicted LMPs $/MWh
        obj_value        float
        total_cost_usd   float
        total_co2_kg     float
    """
    hours        = data["hours"]
    T            = data["T"]
    gens         = data["thermal_gens"]
    gen_info     = data["gen_info"]
    commit       = data["commit"]
    p_init_s     = data["p_init"]
    renew_by_bus = data["renew_by_bus"]
    L_nodc       = data["L_nodc"]
    d0           = data["d0"]
    ci           = data["ci_baseline"].values
    dc_buses     = data["dc_buses"]
    bus_name_to_id = data["bus_name_to_id"]
    all_bus_ids  = data["all_bus_ids"]

    PTDF          = ptdf_data["ptdf"]              # (n_lines, n_buses)
    bus_idx       = ptdf_data["bus_idx"]            # bus_id → col index in PTDF
    thermal_lims  = ptdf_data["thermal_limits"]    # (n_lines,)
    n_lines       = PTDF.shape[0]

    G    = len(gens)
    B_dc = len(dc_buses)

    # Variable layout:
    #   p[g, t]  → g*T + t                     g=0..G-1
    #   e[b, t]  → G*T + b*T + t               b=0..B_dc-1
    #   ls[t]    → (G+B_dc)*T + t              load-shedding slack (penalised)
    #   og[t]    → (G+B_dc)*T + T + t          over-generation slack (free)
    n_var = (G + B_dc + 2) * T

    def p_idx(g, t):  return g * T + t
    def e_idx(b, t):  return G * T + b * T + t
    def ls_idx(t):    return (G + B_dc) * T + t
    def og_idx(t):    return (G + B_dc) * T + T + t

    # Load-shedding penalty: much larger than any marginal cost so the LP only
    # sheds load when there's genuinely not enough committed capacity.
    LS_PENALTY = 10_000.0   # $/MWh

    # ── Objective ─────────────────────────────────────────────────────────────
    mc_vals  = np.array([gen_info.loc[g, "mc"] if g in gen_info.index else 0.0
                          for g in gens])
    mc_mean  = float(mc_vals[mc_vals > 0].mean()) if (mc_vals > 0).any() else 1.0
    ci_mean  = float(ci[ci > 0].mean()) if (ci > 0).any() else 1.0

    c = np.zeros(n_var)
    for g_i, g in enumerate(gens):
        for t in range(T):
            c[p_idx(g_i, t)] = alpha * mc_vals[g_i] / mc_mean

    for b_i, bname in enumerate(dc_buses):
        for t in range(T):
            c[e_idx(b_i, t)] = (1.0 - alpha) * ci[t] / ci_mean

    for t in range(T):
        c[ls_idx(t)] = LS_PENALTY   # heavily penalise load shedding
        c[og_idx(t)] = 0.0          # over-generation is free (curtailment)

    # ── Variable bounds ───────────────────────────────────────────────────────
    lb = np.zeros(n_var)
    ub = np.full(n_var, np.inf)   # slacks are unbounded above

    p_baseline_max = data.get("p_baseline_max", pd.Series(dtype=float))

    for g_i, g in enumerate(gens):
        pmax = gen_info.loc[g, "PMax"] if g in gen_info.index else 0.0
        # Relax upper bound to cover any observed baseline emergency dispatch
        # (e.g. storage discharge, reserve sharing beyond nominal PMax).
        pmax_eff = max(pmax, float(p_baseline_max.get(g, pmax)))
        u_arr = commit[g].values if g in commit.columns else np.zeros(T, dtype=int)
        for t in range(T):
            # lb = 0 (not PMin*u): relaxing must-run avoids over-generation surplus
            # that drives the power-balance dual (LMP) to zero.  The commitment
            # status still gates the upper bound, so uncommitted generators are fixed
            # at 0 and committed generators can dispatch anywhere in [0, PMax_eff].
            lb[p_idx(g_i, t)] = 0.0
            ub[p_idx(g_i, t)] = pmax_eff * u_arr[t]

    dc_max_cap = {}
    for b_i, bname in enumerate(dc_buses):
        d0_b   = d0[bname].values
        peak_b = float(d0_b.max())
        cap_b  = peak_b * (1.0 + headroom)
        dc_max_cap[bname] = cap_b
        for t in range(T):
            lb[e_idx(b_i, t)] = max(0.0, d0_b[t] * (1.0 - flexible_ratio))
            ub[e_idx(b_i, t)] = cap_b

    # Slack variables: ls ≥ 0 (unbounded above), og ≥ 0 (unbounded above)
    for t in range(T):
        lb[ls_idx(t)] = 0.0;  ub[ls_idx(t)] = np.inf
        lb[og_idx(t)] = 0.0;  ub[og_idx(t)] = np.inf

    bounds = [(lb[i], ub[i] if np.isfinite(ub[i]) else None) for i in range(n_var)]

    # ── Equality constraints ──────────────────────────────────────────────────
    # Block 1: Power balance  (T rows)
    # Block 2: DC daily energy balance  (B_dc * n_days rows)
    n_days      = T // 24
    n_eq        = T + B_dc * n_days
    eq_rows, eq_cols, eq_vals = [], [], []
    b_eq = np.zeros(n_eq)

    # Precompute per-bus renewable and load totals (vectorised)
    # renew_total[t] = sum over all buses of renew_by_bus[bus_id][t]
    renew_total = renew_by_bus.values.sum(axis=1)  # (T,)
    L_nodc_total = L_nodc.values.sum(axis=1)       # (T,)

    # Power balance rows
    # sum_g p[g,t] + ls[t] − og[t] − sum_b e[b,t] = L_nodc_total[t] − renew_total[t]
    # ls[t] ≥ 0: emergency supply (load shedding proxy) — penalised at LS_PENALTY
    # og[t] ≥ 0: curtailment / over-generation sink (absorbs must-run surplus) — free
    # This sign convention ensures that when generation is insufficient, ls must
    # increase (expensive), so the LP dispatches thermal generators.  When there
    # is a surplus (e.g. must-run units), og absorbs it at zero cost.
    for t in range(T):
        for g_i in range(G):
            eq_rows.append(t); eq_cols.append(p_idx(g_i, t)); eq_vals.append(1.0)
        for b_i in range(B_dc):
            eq_rows.append(t); eq_cols.append(e_idx(b_i, t)); eq_vals.append(-1.0)
        eq_rows.append(t); eq_cols.append(og_idx(t)); eq_vals.append(-1.0)   # sink
        eq_rows.append(t); eq_cols.append(ls_idx(t)); eq_vals.append(1.0)    # source
        b_eq[t] = L_nodc_total[t] - renew_total[t]

    # DC daily energy balance rows
    for b_i, bname in enumerate(dc_buses):
        d0_b = d0[bname].values
        for day in range(n_days):
            row_eq = T + b_i * n_days + day
            t_start_d = day * 24
            t_end_d   = t_start_d + 24
            for t in range(t_start_d, t_end_d):
                eq_rows.append(row_eq)
                eq_cols.append(e_idx(b_i, t))
                eq_vals.append(1.0)
            b_eq[row_eq] = d0_b[t_start_d:t_end_d].sum()

    A_eq = sparse.csr_matrix(
        (eq_vals, (eq_rows, eq_cols)), shape=(n_eq, n_var)
    )

    # ── Inequality constraints ─────────────────────────────────────────────────
    # We build in COO format and convert at the end.
    ub_rows, ub_cols, ub_vals = [], [], []
    b_ub_list = []
    row_ctr = 0

    # DC causal backlog: cumsum(e[b,0..t]) ≤ cumsum(d0[b,0..t])  per day
    for b_i, bname in enumerate(dc_buses):
        d0_b = d0[bname].values
        for day in range(n_days):
            cum_d0 = 0.0
            for h in range(24 - 1):   # skip last hour (enforced by energy balance)
                t = day * 24 + h
                cum_d0 += d0_b[t]
                for hh in range(h + 1):
                    tt = day * 24 + hh
                    ub_rows.append(row_ctr); ub_cols.append(e_idx(b_i, tt)); ub_vals.append(1.0)
                b_ub_list.append(cum_d0)
                row_ctr += 1

    # DC deferral window (within-day): cumsum_within_day(e[b,0..h]) ≥ cumsum_within_day(d0[b,0..h-W])
    # Ensures work arriving at within-day hour τ is done by τ+W.
    # Reset per day so cross-day cumulation doesn't inflate the RHS.
    # → -sum_{τ=0}^{h} e[b, day*24+τ] ≤ -sum_{τ=0}^{h-W} d0[b, day*24+τ]
    for b_i, bname in enumerate(dc_buses):
        d0_b = d0[bname].values
        for day in range(n_days):
            d0_day = d0_b[day * 24: (day + 1) * 24]          # 24 values for this day
            cum_d0_day = np.concatenate([[0.0], np.cumsum(d0_day)])   # len 25
            for h in range(deferral_window, 24):
                # RHS = cumsum of d0 within this day up to h-W (0-indexed)
                rhs_neg = -float(cum_d0_day[h - deferral_window + 1])
                for hh in range(h + 1):
                    tt = day * 24 + hh
                    ub_rows.append(row_ctr); ub_cols.append(e_idx(b_i, tt)); ub_vals.append(-1.0)
                b_ub_list.append(rhs_neg)
                row_ctr += 1

    # Generator ramp up/down
    for g_i, g in enumerate(gens):
        ramp = gen_info.loc[g, "ramp_mwh"] if g in gen_info.index else 9999.0
        if ramp <= 0:
            ramp = 9999.0
        p_prev = float(p_init_s.get(g, 0.0))

        for t in range(T):
            p_cur_col  = p_idx(g_i, t)
            if t == 0:
                # ramp up vs initial: p[g,0] ≤ p_init + ramp
                ub_rows.append(row_ctr); ub_cols.append(p_cur_col); ub_vals.append(1.0)
                b_ub_list.append(p_prev + ramp)
                row_ctr += 1
                # ramp down vs initial: p_init - p[g,0] ≤ ramp
                ub_rows.append(row_ctr); ub_cols.append(p_cur_col); ub_vals.append(-1.0)
                b_ub_list.append(ramp - p_prev)
                row_ctr += 1
            else:
                p_prev_col = p_idx(g_i, t - 1)
                # ramp up
                ub_rows.append(row_ctr); ub_cols.append(p_cur_col);  ub_vals.append(1.0)
                ub_rows.append(row_ctr); ub_cols.append(p_prev_col); ub_vals.append(-1.0)
                b_ub_list.append(ramp)
                row_ctr += 1
                # ramp down
                ub_rows.append(row_ctr); ub_cols.append(p_cur_col);  ub_vals.append(-1.0)
                ub_rows.append(row_ctr); ub_cols.append(p_prev_col); ub_vals.append(1.0)
                b_ub_list.append(ramp)
                row_ctr += 1

    # Line flow constraints (PTDF-based)
    # fixed_flow[l,t] = Σ_b PTDF[l,b] * (renew[b,t] - L_nodc[b,t])
    # Each bus b in PTDF corresponds to column bus_idx[bus_id]
    ptdf_col_for_bus = {}   # bus_id → PTDF column index (=bus_idx[bus_id])
    for bid, bi in bus_idx.items():
        ptdf_col_for_bus[bid] = bi

    # Build PTDF columns for generators and DC buses
    gen_ptdf_cols = np.array([
        ptdf_col_for_bus.get(
            gen_info.loc[g, "bus_id"] if g in gen_info.index else -1, 0
        )
        for g in gens
    ])  # (G,)

    dc_ptdf_cols = np.array([
        ptdf_col_for_bus.get(bus_name_to_id.get(bname, -1), 0)
        for bname in dc_buses
    ])  # (B_dc,)

    # Fixed flow contributions: renewable minus non-DC load
    # renew_by_bus cols are all_bus_ids; L_nodc cols are all_bus_ids
    net_fixed = renew_by_bus.values - L_nodc.values  # (T, n_all_buses)
    # Reindex to match PTDF columns
    # PTDF columns = bus_idx order (bus_ids sorted)
    ptdf_bus_order = ptdf_data["bus_ids"]
    # Map all_bus_ids → PTDF column positions
    # Build a (T, n_ptdf_buses) fixed_net matrix aligned to PTDF columns
    n_ptdf = PTDF.shape[1]
    fixed_net_aligned = np.zeros((T, n_ptdf))
    for col_i, bid in enumerate(all_bus_ids):
        if bid in bus_idx:
            fixed_net_aligned[:, bus_idx[bid]] += net_fixed[:, col_i]

    fixed_flow = (PTDF @ fixed_net_aligned.T).T   # (T, n_lines)

    # Screen lines: only include lines where max(|PTDF entry|) > threshold
    # for generators or DC buses (to avoid adding rows for irrelevant lines)
    ptdf_gen   = PTDF[:, gen_ptdf_cols]    # (n_lines, G)
    ptdf_dc    = PTDF[:, dc_ptdf_cols]     # (n_lines, B_dc)
    active_lines = np.where(
        (np.abs(ptdf_gen).max(axis=1) > ptdf_threshold) |
        (np.abs(ptdf_dc).max(axis=1)  > ptdf_threshold)
    )[0]

    # Effective thermal limits: raise any line's limit to cover the PTDF-computed
    # fixed flow plus a margin.  This prevents infeasibility caused by approximation
    # errors between our DC-PTDF model and the actual VATIC power-flow model.
    # The baseline VATIC dispatch is always feasible by construction; the LP should
    # respect at least that same feasible region.
    thermal_lims_eff = thermal_lims.copy()
    max_fixed_abs = np.abs(fixed_flow).max(axis=0)  # (n_lines,) max |fixed_flow| over time
    for li in active_lines:
        thermal_lims_eff[li] = max(thermal_lims[li], max_fixed_abs[li] * (1.0 + lf_margin))

    print(f"  Active lines (|PTDF| > {ptdf_threshold}): {len(active_lines)} / {n_lines}")

    for li in active_lines:
        limit = thermal_lims_eff[li]
        if limit <= 0:
            continue
        for t in range(T):
            ff = float(fixed_flow[t, li])

            # Upper: Σ_g PTDF[l,bus(g)] p[g,t] - Σ_b PTDF[l,bus(b)] e[b,t] ≤ limit - ff
            for g_i in range(G):
                v = float(ptdf_gen[li, g_i])
                if abs(v) > ptdf_threshold:
                    ub_rows.append(row_ctr); ub_cols.append(p_idx(g_i, t)); ub_vals.append(v)
            for b_i in range(B_dc):
                v = float(ptdf_dc[li, b_i])
                if abs(v) > ptdf_threshold:
                    ub_rows.append(row_ctr); ub_cols.append(e_idx(b_i, t)); ub_vals.append(-v)
            b_ub_list.append(limit - ff)
            row_ctr += 1

            # Lower: -Σ_g p[g,t]·PTDF + Σ_b e[b,t]·PTDF ≤ limit + ff
            for g_i in range(G):
                v = float(ptdf_gen[li, g_i])
                if abs(v) > ptdf_threshold:
                    ub_rows.append(row_ctr); ub_cols.append(p_idx(g_i, t)); ub_vals.append(-v)
            for b_i in range(B_dc):
                v = float(ptdf_dc[li, b_i])
                if abs(v) > ptdf_threshold:
                    ub_rows.append(row_ctr); ub_cols.append(e_idx(b_i, t)); ub_vals.append(v)
            b_ub_list.append(limit + ff)
            row_ctr += 1

    b_ub = np.array(b_ub_list)
    if not np.all(np.isfinite(b_ub)):
        n_bad = np.sum(~np.isfinite(b_ub))
        import warnings
        warnings.warn(f"b_ub has {n_bad} non-finite values; clamping to ±1e9")
        b_ub = np.where(np.isfinite(b_ub), b_ub, np.sign(b_ub + 1e-300) * 1e9)
    A_ub = sparse.csr_matrix(
        (ub_vals, (ub_rows, ub_cols)), shape=(row_ctr, n_var)
    )

    print(f"  LP size: {n_var} vars, {n_eq} eq, {row_ctr} ineq")

    # ── Solve ─────────────────────────────────────────────────────────────────
    result = linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"disp": False, "presolve": True, "time_limit": 600.0},
    )

    status = {0: "optimal", 1: "iteration_limit", 2: "infeasible",
              3: "unbounded", 4: "other"}.get(result.status, "unknown")
    obj_str = f"{result.fun:.4f}" if result.fun is not None else "None"
    print(f"  HiGHS status: {status} (code {result.status})  obj={obj_str}")
    if hasattr(result, "message"):
        print(f"  Message: {result.message}")

    if result.status != 0:
        return {"status": status, "result": result}

    x = result.x

    # ── Unpack solution ────────────────────────────────────────────────────────
    p_arr = np.array([[x[p_idx(g_i, t)] for t in range(T)] for g_i in range(G)])
    e_arr = np.array([[x[e_idx(b_i, t)] for t in range(T)] for b_i in range(B_dc)])
    ls_arr = np.array([x[ls_idx(t)] for t in range(T)])
    og_arr = np.array([x[og_idx(t)] for t in range(T)])

    if ls_arr.sum() > 1.0:
        print(f"  [warn] Load shedding in LP: {ls_arr.sum():.1f} MWh total, "
              f"max hour={ls_arr.max():.1f} MW at t={ls_arr.argmax()}")
    if og_arr.sum() > 1.0:
        print(f"  [info] Over-generation in LP: {og_arr.sum():.1f} MWh total")

    p_opt = pd.DataFrame(p_arr.T, index=hours, columns=gens)
    e_opt = pd.DataFrame(e_arr.T, index=hours, columns=dc_buses)

    # ── Predicted LMPs from dual variables ────────────────────────────────────
    # Dual of power balance equality (rows 0..T-1 in A_eq)
    lambda_t = result.eqlin.marginals[:T]   # $/MWh  (shadow price per hour)

    # Dual of line flow constraints (upper/lower pairs for active lines)
    # LMP[b,t] = λ[t] + Σ_l PTDF[l,b] · (μ_lower[l,t] − μ_upper[l,t])
    # marginals for ineq constraints are non-negative (dual of Ax≤b has μ≥0)
    mu_ub = result.ineqlin.marginals   # shape (row_ctr,)

    # Identify which ineq rows correspond to line flows
    n_dc_causal   = B_dc * n_days * (24 - 1)
    n_dc_deferral = B_dc * n_days * (24 - deferral_window)
    n_ramp        = G * T * 2
    line_flow_start = n_dc_causal + n_dc_deferral + n_ramp
    n_line_flow_rows = 2 * len(active_lines) * T

    mu_line = mu_ub[line_flow_start: line_flow_start + n_line_flow_rows]

    # Build per-bus predicted LMP
    id_to_name = {v: k for k, v in bus_name_to_id.items()}
    lmp_pred = np.outer(lambda_t, np.ones(n_ptdf))   # (T, n_buses)

    pair = 0
    for li in active_lines:
        limit = thermal_lims_eff[li]
        if limit <= 0:
            continue
        for t in range(T):
            mu_upper = mu_line[pair * 2]       # upper flow constraint dual
            mu_lower = mu_line[pair * 2 + 1]   # lower flow constraint dual
            # Contribution to bus LMPs: PTDF[l,b] * (mu_lower - mu_upper)
            lmp_pred[t] += PTDF[li] * (mu_lower - mu_upper)
            pair += 1

    lmp_predicted = pd.DataFrame(
        lmp_pred,
        index=hours,
        columns=[id_to_name.get(bid, str(bid)) for bid in ptdf_bus_order],
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    ef_co2 = {}
    for g in gens:
        ef_co2[g] = 0.0   # simplified: use CI signal for DC carbon only

    # Operational cost = Σ mc[g] * p[g,t]
    total_cost = float(sum(
        mc_vals[g_i] * p_arr[g_i].sum() for g_i in range(G)
    ))
    # DC carbon = Σ CI[t] * e[b,t]  (approx: system CI as proxy)
    total_dc_co2 = float(sum(
        ci[t] * e_arr[:, t].sum() for t in range(T)
    ))

    return {
        "status":        status,
        "p_opt":         p_opt,
        "e_opt":         e_opt,
        "lmp_predicted": lmp_predicted,
        "lambda_t":      pd.Series(lambda_t, index=hours, name="energy_price"),
        "obj_value":     float(result.fun),
        "total_cost_usd": total_cost,
        "total_dc_co2_kg": total_dc_co2,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. Price-taking LP with PTDF network constraints  (default formulation)
# ═══════════════════════════════════════════════════════════════════════════════

def build_and_solve_price_taking(
    data: dict,
    ptdf_data: dict,
    alpha: float = 0.5,
    deferral_window: int = 12,
    flexible_ratio: float = 0.30,
    headroom: float = 0.30,
    ptdf_threshold: float = 1e-4,
    lf_margin: float = 0.10,
) -> dict:
    """Price-taking LP with PTDF network constraints.

    Key differences from build_and_solve():
    - No thermal dispatch variables: baseline VATIC dispatch is fixed.
    - Per-bus VATIC LMPs used directly as the exogenous cost signal.
    - Decision variables are only e[b,t]  (B_dc × T).
    - PTDF constraints enforce line flow feasibility relative to the
      baseline VATIC flows.

    This resolves the dual-collapse problem in build_and_solve(): with fixed
    thermal dispatch, the interior-point degeneracy that drove predicted LMPs
    to near-zero is eliminated.  The LMP signal comes from an actual market-
    clearing simulation rather than from LP dual variables.

    Returns the same dict structure as build_and_solve(); lmp_predicted is None
    (the exogenous price signal is lmp_baseline, already in data).
    """
    hours          = data["hours"]
    T              = data["T"]
    dc_buses       = data["dc_buses"]
    d0             = data["d0"]
    ci             = data["ci_baseline"].values
    lmp_base       = data["lmp_baseline"]       # DataFrame (T, bus_name)
    bus_name_to_id = data["bus_name_to_id"]
    all_bus_ids    = data["all_bus_ids"]
    renew_by_bus   = data["renew_by_bus"]       # DataFrame (T, bus_id)
    L_nodc         = data["L_nodc"]             # DataFrame (T, bus_id)
    p_by_bus       = data["p_baseline_by_bus"]  # DataFrame (T, bus_id)

    PTDF         = ptdf_data["ptdf"]
    bus_idx      = ptdf_data["bus_idx"]
    thermal_lims = ptdf_data["thermal_limits"]
    n_lines      = PTDF.shape[0]
    n_ptdf       = PTDF.shape[1]

    B_dc  = len(dc_buses)
    n_var = B_dc * T

    def e_idx(b, t): return b * T + t

    # ── Objective ─────────────────────────────────────────────────────────────
    # c[b,t] = α * lmp[b,t] / lmp_mean + (1-α) * CI[t] / CI_mean
    lmp_dc = {}
    for bname in dc_buses:
        if bname in lmp_base.columns:
            lmp_dc[bname] = lmp_base[bname].reindex(hours).fillna(0.0).values
        else:
            lmp_dc[bname] = lmp_base.mean(axis=1).reindex(hours).fillna(0.0).values

    all_lmps = np.concatenate(list(lmp_dc.values()))
    lmp_mean = float(all_lmps[all_lmps > 0].mean()) if (all_lmps > 0).any() else 1.0
    ci_mean  = float(ci[ci > 0].mean()) if (ci > 0).any() else 1.0

    c = np.zeros(n_var)
    for b_i, bname in enumerate(dc_buses):
        lmp_b = lmp_dc[bname]
        for t in range(T):
            c[e_idx(b_i, t)] = (
                alpha       * lmp_b[t] / lmp_mean
                + (1.0 - alpha) * ci[t]   / ci_mean
            )

    # ── Variable bounds ───────────────────────────────────────────────────────
    bounds = []
    for b_i, bname in enumerate(dc_buses):
        d0_b   = d0[bname].values
        peak_b = float(d0_b.max())
        cap_b  = peak_b * (1.0 + headroom)
        for t in range(T):
            lb_t = max(0.0, d0_b[t] * (1.0 - flexible_ratio))
            bounds.append((lb_t, cap_b))

    # ── Equality: DC daily energy balance ─────────────────────────────────────
    n_days = T // 24
    n_eq   = B_dc * n_days
    eq_rows, eq_cols, eq_vals = [], [], []
    b_eq = np.zeros(n_eq)
    for b_i, bname in enumerate(dc_buses):
        d0_b = d0[bname].values
        for day in range(n_days):
            row_eq    = b_i * n_days + day
            t_start_d = day * 24
            t_end_d   = t_start_d + 24
            for t in range(t_start_d, t_end_d):
                eq_rows.append(row_eq)
                eq_cols.append(e_idx(b_i, t))
                eq_vals.append(1.0)
            b_eq[row_eq] = d0_b[t_start_d:t_end_d].sum()
    A_eq = sparse.csr_matrix((eq_vals, (eq_rows, eq_cols)), shape=(n_eq, n_var))

    # ── Inequality constraints ─────────────────────────────────────────────────
    ub_rows, ub_cols, ub_vals = [], [], []
    b_ub_list = []
    row_ctr   = 0

    # DC causal: ∑_{τ≤t} e[b,τ] ≤ ∑_{τ≤t} d0[b,τ]  (within each day)
    for b_i, bname in enumerate(dc_buses):
        d0_b = d0[bname].values
        for day in range(n_days):
            cum_d0 = 0.0
            for h in range(24 - 1):
                t = day * 24 + h
                cum_d0 += d0_b[t]
                for hh in range(h + 1):
                    tt = day * 24 + hh
                    ub_rows.append(row_ctr)
                    ub_cols.append(e_idx(b_i, tt))
                    ub_vals.append(1.0)
                b_ub_list.append(cum_d0)
                row_ctr += 1

    # DC deferral: ∑_{τ≤t} e[b,τ] ≥ ∑_{τ≤t-W} d0[b,τ]
    for b_i, bname in enumerate(dc_buses):
        d0_b = d0[bname].values
        for day in range(n_days):
            d0_day     = d0_b[day * 24: (day + 1) * 24]
            cum_d0_day = np.concatenate([[0.0], np.cumsum(d0_day)])
            for h in range(deferral_window, 24):
                rhs_neg = -float(cum_d0_day[h - deferral_window + 1])
                for hh in range(h + 1):
                    tt = day * 24 + hh
                    ub_rows.append(row_ctr)
                    ub_cols.append(e_idx(b_i, tt))
                    ub_vals.append(-1.0)
                b_ub_list.append(rhs_neg)
                row_ctr += 1

    # PTDF line flow constraints
    # Baseline net injection per bus:
    #   p_thermal[bus,t] + renew[bus,t] - L_nodc[bus,t] - d0_dc[bus,t]
    # This gives the same flows VATIC produced (up to the DC approximation).
    # After the LP, DC load changes from d0 to e, perturbing each flow by
    #   Δflow[l,t] = -PTDF[l, dc_bus] * (e[b,t] - d0[b,t])
    # (DC load increase → net injection decrease → opposing sign to PTDF entry)
    full_net = np.zeros((T, n_ptdf))
    for bid in all_bus_ids:
        if bid not in bus_idx:
            continue
        bi = bus_idx[bid]
        if bid in renew_by_bus.columns:
            full_net[:, bi] += renew_by_bus[bid].values
        if bid in L_nodc.columns:
            full_net[:, bi] -= L_nodc[bid].values
        if bid in p_by_bus.columns:
            full_net[:, bi] += p_by_bus[bid].values
    for bname in dc_buses:
        bid = bus_name_to_id.get(bname)
        if bid is not None and bid in bus_idx:
            full_net[:, bus_idx[bid]] -= d0[bname].values

    np.nan_to_num(full_net, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    baseline_flow = (PTDF @ full_net.T).T          # (T, n_lines)

    dc_ptdf_cols = np.array([
        bus_idx.get(bus_name_to_id.get(bname, -1), 0)
        for bname in dc_buses
    ])
    ptdf_dc = PTDF[:, dc_ptdf_cols]                 # (n_lines, B_dc)

    # Precompute baseline DC-load contribution to each line flow:
    #   d0_c[t, l] = Σ_b PTDF[l, ptdf_col(b)] * d0[b,t]
    d0_dc_mat = np.zeros((T, n_ptdf))
    for b_i, bname in enumerate(dc_buses):
        bid = bus_name_to_id.get(bname)
        if bid is not None and bid in bus_idx:
            d0_dc_mat[:, bus_idx[bid]] += d0[bname].values
    d0_c = (PTDF @ d0_dc_mat.T).T                  # (T, n_lines)

    active_lines = np.where(np.abs(ptdf_dc).max(axis=1) > ptdf_threshold)[0]

    thermal_lims_eff = thermal_lims.copy()
    max_baseline_abs = np.abs(baseline_flow).max(axis=0)
    for li in active_lines:
        thermal_lims_eff[li] = max(
            thermal_lims[li],
            max_baseline_abs[li] * (1.0 + lf_margin),
        )

    print(f"  Active lines (|PTDF| > {ptdf_threshold}): {len(active_lines)} / {n_lines}")

    for li in active_lines:
        limit = thermal_lims_eff[li]
        if limit <= 0:
            continue
        for t in range(T):
            f0   = float(baseline_flow[t, li])
            dc_c = float(d0_c[t, li])

            # Upper bound: flow_lp[l,t] ≤ +limit
            #   f0 - Σ_b ptdf_dc[l,b]*(e[b,t]-d0[b,t]) ≤ limit
            #   -Σ_b ptdf_dc[l,b]*e[b,t] ≤ limit - f0 - dc_c
            for b_i in range(B_dc):
                v = float(ptdf_dc[li, b_i])
                if abs(v) > ptdf_threshold:
                    ub_rows.append(row_ctr)
                    ub_cols.append(e_idx(b_i, t))
                    ub_vals.append(-v)
            b_ub_list.append(limit - f0 - dc_c)
            row_ctr += 1

            # Lower bound: flow_lp[l,t] ≥ -limit
            #   Σ_b ptdf_dc[l,b]*e[b,t] ≤ limit + f0 + dc_c
            for b_i in range(B_dc):
                v = float(ptdf_dc[li, b_i])
                if abs(v) > ptdf_threshold:
                    ub_rows.append(row_ctr)
                    ub_cols.append(e_idx(b_i, t))
                    ub_vals.append(v)
            b_ub_list.append(limit + f0 + dc_c)
            row_ctr += 1

    b_ub = np.array(b_ub_list) if b_ub_list else np.zeros(0)
    np.nan_to_num(b_ub, copy=False, nan=0.0, posinf=1e9, neginf=-1e9)
    A_ub = sparse.csr_matrix(
        (ub_vals, (ub_rows, ub_cols)), shape=(row_ctr, n_var)
    ) if row_ctr > 0 else sparse.csr_matrix((0, n_var))

    print(f"  LP size: {n_var} vars, {n_eq} eq, {row_ctr} ineq")

    # ── Solve ─────────────────────────────────────────────────────────────────
    result = linprog(
        c,
        A_ub=A_ub if row_ctr > 0 else None, b_ub=b_ub if row_ctr > 0 else None,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"disp": False, "presolve": True, "time_limit": 600.0},
    )

    status = {0: "optimal", 1: "iteration_limit", 2: "infeasible",
              3: "unbounded", 4: "other"}.get(result.status, "unknown")
    obj_str = f"{result.fun:.4f}" if result.fun is not None else "None"
    print(f"  HiGHS status: {status} (code {result.status})  obj={obj_str}")
    if hasattr(result, "message"):
        print(f"  Message: {result.message}")

    if result.status != 0:
        return {"status": status, "result": result}

    x    = result.x
    e_arr = np.array([[x[e_idx(b_i, t)] for t in range(T)] for b_i in range(B_dc)])
    e_opt = pd.DataFrame(e_arr.T, index=hours, columns=dc_buses)

    # Sanity check: net energy shift should be ≈ 0 (energy balance enforced)
    d0_arr = np.array([d0[b].values for b in dc_buses])
    e_delta_total = (e_arr - d0_arr).sum()
    print(f"  Net DC energy shift: {e_delta_total:+.2f} MWh (≈0 from energy balance)")

    # Peak-load shift diagnostic
    d0_total  = d0_arr.sum(axis=0)   # (T,) total baseline DC load per hour
    e_total   = e_arr.sum(axis=0)    # (T,) total LP DC load per hour
    shift_max = float(np.abs(e_total - d0_total).max())
    print(f"  Max hourly DC shift: {shift_max:.2f} MW")

    total_cost = float(sum(
        lmp_dc[bname][t] * e_arr[b_i, t]
        for b_i, bname in enumerate(dc_buses)
        for t in range(T)
    ))
    total_dc_co2 = float(sum(
        ci[t] * e_arr[:, t].sum() for t in range(T)
    ))

    return {
        "status":           status,
        "p_opt":            None,     # no thermal dispatch variables
        "e_opt":            e_opt,
        "lmp_predicted":    None,     # price signal was exogenous lmp_baseline
        "lambda_t":         pd.Series(np.zeros(T), index=hours, name="energy_price"),
        "obj_value":        float(result.fun),
        "total_cost_usd":   total_cost,
        "total_dc_co2_kg":  total_dc_co2,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Write output BusInjections
# ═══════════════════════════════════════════════════════════════════════════════

def write_bus_injections(
    e_opt: pd.DataFrame,
    d0: pd.DataFrame,
    source_inject_dir: Path,
    output_inject_dir: Path,
    dc_buses: Sequence[str],
) -> None:
    """Write new DAY_AHEAD and REAL_TIME BusInjections CSVs.

    DA CSV: direct output of e_opt (one row per hour, RTS-GMLC Period format).
    RT CSV: each sub-hourly period within a given hour is scaled by the ratio
            e_opt[b,t] / d0[b,t], preserving intra-hour shape.
    """
    output_inject_dir.mkdir(parents=True, exist_ok=True)

    # ── DAY_AHEAD ─────────────────────────────────────────────────────────────
    da_src_files = list(source_inject_dir.glob("DAY_AHEAD_*.csv"))
    if not da_src_files:
        raise FileNotFoundError(f"No DAY_AHEAD_*.csv in {source_inject_dir}")
    da_src = pd.read_csv(da_src_files[0])

    time_cols = ["Year", "Month", "Day", "Period"]
    non_bus_cols = [c for c in da_src.columns if c in time_cols]

    # Build output DA dataframe
    out_da = da_src[non_bus_cols].copy()
    for bname in dc_buses:
        if bname in e_opt.columns:
            # Map DatetimeIndex → RTS-GMLC Period rows
            vals = e_opt[bname].values
            if len(vals) == len(out_da):
                out_da[bname] = vals
            else:
                # Subset match by date/period
                out_da[bname] = da_src[bname].values if bname in da_src.columns else 0.0

    da_out_path = output_inject_dir / da_src_files[0].name
    out_da.to_csv(da_out_path, index=False)
    print(f"  Wrote DA injections → {da_out_path}")

    # ── REAL_TIME ─────────────────────────────────────────────────────────────
    rt_src_files = list(source_inject_dir.glob("REAL_TIME_*.csv"))
    if rt_src_files:
        rt_src = pd.read_csv(rt_src_files[0])
        out_rt = rt_src.copy()

        # Compute hourly ratio e_opt / d0 per bus
        ratio = pd.DataFrame(index=e_opt.index)
        for bname in dc_buses:
            if bname in e_opt.columns and bname in d0.columns:
                d0_b = d0[bname].values
                e_b  = e_opt[bname].values
                r    = np.where(d0_b > 1e-6, e_b / d0_b, 1.0)
                ratio[bname] = r

        # Determine time column
        if "Period" in rt_src.columns:
            _rt_offset = 1 if rt_src["Period"].min() >= 1 else 0
            _rt_pph    = max(1, rt_src["Period"].nunique() // 24)  # periods per hour
            rt_times = pd.to_datetime(
                {"year":  rt_src["Year"],  "month": rt_src["Month"],
                 "day":   rt_src["Day"],   "hour":  (rt_src["Period"] - _rt_offset) // _rt_pph},
                utc=True,
            )
        else:
            rt_times = pd.to_datetime(
                rt_src.get("Forecast_time", rt_src.get("Time")), utc=True
            )
            rt_times = rt_times.dt.floor("h")

        for bname in dc_buses:
            if bname in rt_src.columns and bname in ratio.columns:
                hourly_ratio = ratio[bname]
                # Map each RT row to its hourly ratio
                rt_idx = pd.DatetimeIndex(rt_times).tz_convert("UTC") if rt_times.dt.tz else pd.DatetimeIndex(rt_times).tz_localize("UTC")
                rt_ratio = hourly_ratio.reindex(rt_idx).ffill().fillna(1.0)
                out_rt[bname] = rt_src[bname].values * rt_ratio.values

        rt_out_path = output_inject_dir / rt_src_files[0].name
        out_rt.to_csv(rt_out_path, index=False)
        print(f"  Wrote RT injections → {rt_out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Comparison table
# ═══════════════════════════════════════════════════════════════════════════════

def build_comparison_table(
    lp_result: dict,
    data: dict,
    post_shift_dir: Path | None = None,
    gen_csv: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Compare joint-LP predictions vs baseline and post-shift VATIC.

    Columns
    -------
    hour            UTC timestamp
    lmp_baseline    avg DC-bus LMP from baseline VATIC ($/MWh)
    lmp_predicted   avg DC-bus LMP predicted by joint LP ($/MWh)
    lmp_shifted     avg DC-bus LMP from post-shift VATIC ($/MWh, if available)
    dc_load_baseline  total DC load MW (baseline)
    dc_load_lp        total DC load MW (joint LP output)
    ci_baseline     carbon intensity from baseline (kg CO₂/MWh)
    """
    dc_buses   = data["dc_buses"]
    hours      = data["hours"]
    bus_name_to_id = data["bus_name_to_id"]

    lmp_base = data["lmp_baseline"]
    lmp_pred = lp_result.get("lmp_predicted")
    e_opt    = lp_result.get("e_opt")
    d0       = data["d0"]
    ci       = data["ci_baseline"]

    def _avg_dc_lmp(lmp_df: pd.DataFrame) -> pd.Series:
        cols = [c for c in lmp_df.columns if c in dc_buses]
        return lmp_df[cols].mean(axis=1) if cols else pd.Series(np.nan, index=hours)

    rows = {
        "lmp_baseline": _avg_dc_lmp(lmp_base).reindex(hours),
        "dc_load_baseline": d0.sum(axis=1).reindex(hours),
        "ci_baseline": ci.reindex(hours),
    }

    if lmp_pred is not None:
        rows["lmp_predicted"] = _avg_dc_lmp(lmp_pred).reindex(hours)

    if e_opt is not None:
        rows["dc_load_lp"] = e_opt.sum(axis=1).reindex(hours)

    # Post-shift VATIC results
    if post_shift_dir is not None and Path(post_shift_dir).is_dir():
        post_bd = pd.read_csv(Path(post_shift_dir) / "bus_detail.csv")
        post_bd["datetime"] = _parse_datetime_col(post_bd)
        id_map = bus_name_to_id
        post_bd["bus_id"] = post_bd["Bus"].map(lambda n: id_map.get(n))
        post_bd = post_bd.dropna(subset=["bus_id"])
        dc_ids  = [bus_name_to_id[b] for b in dc_buses if b in bus_name_to_id]
        dc_lmp_post = (
            post_bd[post_bd["bus_id"].isin(dc_ids)]
            .groupby("datetime")["LMP"].mean()
            .reindex(hours)
        )
        rows["lmp_shifted"] = dc_lmp_post

        # Residual: predicted − simulated
        if "lmp_predicted" in rows:
            rows["lmp_residual"] = rows["lmp_predicted"] - rows["lmp_shifted"]

        # Post-shift CO₂ and cost
        if gen_csv is not None:
            gdf = pd.read_csv(gen_csv)
            gdf = gdf.set_index("GEN UID")
            ef  = _ef_co2(gdf)   # kg CO2/MWh

            th_post = pd.read_csv(Path(post_shift_dir) / "thermal_detail.csv")
            th_post["datetime"] = _parse_datetime_col(th_post)
            th_post["co2"] = th_post["Dispatch"] * th_post["Generator"].map(ef).fillna(0.0)
            hourly_co2_post = th_post.groupby("datetime")["co2"].sum().reindex(hours, fill_value=0.0)
            rows["co2_shifted_kg"] = hourly_co2_post

    table = pd.DataFrame(rows, index=hours)
    table.index.name = "hour"

    if out_path is not None:
        table.to_csv(out_path)
        print(f"  Comparison table → {out_path}")

    # Print summary
    print("\n  ── Comparison summary ────────────────────────────────────────")
    for col in ["lmp_baseline", "lmp_predicted", "lmp_shifted", "lmp_residual"]:
        if col in table.columns:
            s = table[col].dropna()
            print(f"  {col:25s}: mean={s.mean():.2f}  std={s.std():.2f}  "
                  f"min={s.min():.2f}  max={s.max():.2f}  $/MWh")
    for col in ["dc_load_baseline", "dc_load_lp"]:
        if col in table.columns:
            s = table[col].dropna()
            print(f"  {col:25s}: mean={s.mean():.1f} MW  total={s.sum():.0f} MWh")
    if "lmp_residual" in table.columns:
        res = table["lmp_residual"].dropna()
        print(f"\n  Residual RMSE (predicted − simulated LMP): "
              f"{float(np.sqrt((res**2).mean())):.3f} $/MWh")
        print(f"  Residual MAE : {float(res.abs().mean()):.3f} $/MWh")
        pct = float(res.abs().mean() / table["lmp_baseline"].mean() * 100)
        print(f"  MAE as % of baseline mean LMP: {pct:.1f}%")
    print("  ──────────────────────────────────────────────────────────────\n")

    return table


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Grid helper
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_grid(grid_name: str):
    """Return (data_dir_name, grid_path) for a named grid."""
    prefix = next(
        (k for k in sorted(GRID_REGISTRY, key=len, reverse=True)
         if grid_name.startswith(k)),
        None,
    )
    if prefix is None:
        raise ValueError(f"Unknown grid prefix for '{grid_name}'. "
                         f"Known: {list(GRID_REGISTRY)}")
    data_dir_name = GRID_REGISTRY[prefix]
    grid_path     = _GRIDS_DIR / grid_name
    return data_dir_name, grid_path


def _copy_grid(source_grid: str, output_grid: str) -> Path:
    """Copy source grid directory and initial-state to output_grid (if not already present)."""
    import shutil
    _, src_path = _resolve_grid(source_grid)
    out_path    = _GRIDS_DIR / output_grid
    if not out_path.exists():
        shutil.copytree(src_path, out_path)
        print(f"  Copied grid {source_grid} → {output_grid}")
    else:
        print(f"  Output grid already exists: {output_grid}")

    # Also copy initial-state directory (required by VATIC loaders)
    init_src = _GRIDS_DIR / "initial-state" / source_grid
    init_dst = _GRIDS_DIR / "initial-state" / output_grid
    if init_src.exists() and not init_dst.exists():
        shutil.copytree(init_src, init_dst)
        print(f"  Copied initial-state {source_grid} → {output_grid}")

    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Main entry point (callable from main.py and as CLI)
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    baseline_dir: str | Path,
    grid: str,
    output_grid: str,
    dc_buses: Sequence[str],
    start_date: str,
    n_days: int,
    alpha: float = 0.5,
    deferral_window: int = 12,
    flexible_ratio: float = 0.30,
    headroom: float = 0.30,
    post_shift_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    ptdf_threshold: float = 1e-4,
    commit_dir: str | Path | None = None,
) -> dict:
    """End-to-end joint LP run.  Returns the lp_result dict from build_and_solve().

    Parameters mirror the CLI flags.  Callable from main.py as a drop-in
    for the analyze_cas.py + apply_cas_shift.py LP step when
    params.json["cas"]["lp_mode"] == "joint".
    """
    baseline_dir = Path(baseline_dir)
    out_dir      = Path(out_dir) if out_dir else baseline_dir.parent / "cas-lp-joint"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir_name, grid_path = _resolve_grid(grid)
    src_data = grid_path / data_dir_name
    gen_csv    = src_data / "SourceData" / "gen.csv"
    bus_csv    = src_data / "SourceData" / "bus.csv"
    branch_csv = src_data / "SourceData" / "branch.csv"
    inject_dir = src_data / "timeseries_data_files" / "BusInjections"

    da_files = list(inject_dir.glob("DAY_AHEAD_*.csv"))
    if not da_files:
        raise FileNotFoundError(f"No DAY_AHEAD_*.csv in {inject_dir}")
    inject_csv = da_files[0]

    print("═" * 60)
    print("  sim_lp_joint: Building PTDF matrix …")
    ptdf_data = build_ptdf(branch_csv, bus_csv)
    print(f"  Reference bus: {ptdf_data['ref_bus_id']} "
          f"({ptdf_data['bus_name_by_id'].get(ptdf_data['ref_bus_id'], '?')})")
    print(f"  Buses: {ptdf_data['ptdf'].shape[1]}  Lines: {ptdf_data['ptdf'].shape[0]}")

    print("\n  Loading baseline data …")
    data = load_baseline(
        baseline_dir, gen_csv, bus_csv, inject_csv,
        dc_buses, start_date, n_days,
        commit_dir=Path(commit_dir) if commit_dir is not None else None,
    )
    print(f"  Thermal generators: {len(data['thermal_gens'])}")
    print(f"  DC buses: {data['dc_buses']}")
    print(f"  Hours: {data['T']}  ({start_date}, {n_days} days)")

    print("\n  Building and solving price-taking LP (network-aware) …")
    lp_result = build_and_solve_price_taking(
        data, ptdf_data,
        alpha=alpha,
        deferral_window=deferral_window,
        flexible_ratio=flexible_ratio,
        headroom=headroom,
        ptdf_threshold=ptdf_threshold,
    )

    if lp_result["status"] != "optimal":
        print(f"  [WARNING] LP did not solve to optimality: {lp_result['status']}")
        return lp_result

    # ── Write output grid ────────────────────────────────────────────────────
    print(f"\n  Writing output BusInjections for grid '{output_grid}' …")
    out_grid_path  = _copy_grid(grid, output_grid)
    out_data_dir   = out_grid_path / data_dir_name
    out_inject_dir = out_data_dir / "timeseries_data_files" / "BusInjections"

    write_bus_injections(
        lp_result["e_opt"], data["d0"],
        inject_dir, out_inject_dir, data["dc_buses"],
    )

    # ── Comparison table ─────────────────────────────────────────────────────
    print("\n  Building comparison table …")
    build_comparison_table(
        lp_result, data,
        post_shift_dir=post_shift_dir,
        gen_csv=gen_csv,
        out_path=out_dir / "comparison_table.csv",
    )

    # Save LP predicted schedule
    e_path = out_dir / "lp_dc_schedule.csv"
    lp_result["e_opt"].to_csv(e_path)
    print(f"  DC schedule → {e_path}")

    if lp_result.get("lmp_predicted") is not None:
        lmp_path = out_dir / "lp_predicted_lmps.csv"
        lp_result["lmp_predicted"].to_csv(lmp_path)
        print(f"  Predicted LMPs → {lmp_path}")

    # ── cas_results.csv — compatible with compare_cas_modes.py ───────────────
    # Single row representing the joint LP at alpha=lp_alpha (compare script
    # detects LP mode by presence of "alpha" column).
    ct = pd.read_csv(out_dir / "comparison_table.csv", index_col=0, parse_dates=True)
    if "ci_baseline" in ct.columns and "dc_load_lp" in ct.columns:
        co2_base     = (ct["ci_baseline"] * ct["dc_load_baseline"]).sum()
        co2_lp       = (ct["ci_baseline"] * ct["dc_load_lp"]).sum()
        carbon_red   = float((co2_base - co2_lp) / max(co2_base, 1e-9) * 100)
        cost_base    = (ct["lmp_baseline"] * ct["dc_load_baseline"]).sum()
        cost_lp      = (ct["lmp_baseline"] * ct["dc_load_lp"]).sum()  # baseline prices: apples-to-apples
        cost_red     = float((cost_base - cost_lp) / max(cost_base, 1e-9) * 100)
        cas_row = pd.DataFrame([{
            "alpha":                alpha,
            "carbon_reduction_pct": round(carbon_red, 4),
            "cost_reduction_pct":   round(cost_red, 4),
            "total_co2_kg":         round(co2_lp, 1),
            "total_cost_usd":       round(cost_lp, 2),
        }])
        cas_path = out_dir / "cas_results.csv"
        cas_row.to_csv(cas_path, index=False)
        print(f"  CAS results → {cas_path}")

    print("\n  Done.")
    return lp_result


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline-dir",  required=True, type=Path)
    p.add_argument("--grid",          required=True,
                   help="Source grid name with BusInjections (e.g. RTS-GMLC-DC)")
    p.add_argument("--output-grid",   required=True,
                   help="Output grid name for the shifted schedule")
    p.add_argument("--buses",         required=True, nargs="+", metavar="BUS")
    p.add_argument("--start-date",    required=True, metavar="YYYY-MM-DD")
    p.add_argument("--days",          type=int, default=7)
    p.add_argument("--alpha",         type=float, default=0.5,
                   help="Weight on operational cost (0=carbon-only, 1=cost-only)")
    p.add_argument("--deferral-window", type=int, default=12, metavar="H")
    p.add_argument("--flexible-ratio",  type=float, default=0.30,
                   help="Fraction of DC load that is flexible (0–1)")
    p.add_argument("--headroom",        type=float, default=0.30,
                   help="Max capacity above baseline peak (fraction)")
    p.add_argument("--post-shift-dir",  type=Path, default=None,
                   help="Post-shift VATIC output dir for residual validation")
    p.add_argument("--out-dir",         type=Path, default=None,
                   help="Directory for comparison outputs (default: baseline-dir/../cas-lp-joint)")
    p.add_argument("--ptdf-threshold",  type=float, default=1e-4,
                   help="PTDF magnitude below which line constraints are skipped")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run(
        baseline_dir    = args.baseline_dir,
        grid            = args.grid,
        output_grid     = args.output_grid,
        dc_buses        = args.buses,
        start_date      = args.start_date,
        n_days          = args.days,
        alpha           = args.alpha,
        deferral_window = args.deferral_window,
        flexible_ratio  = args.flexible_ratio,
        headroom        = args.headroom,
        post_shift_dir  = args.post_shift_dir,
        out_dir         = args.out_dir,
        ptdf_threshold  = args.ptdf_threshold,
    )


if __name__ == "__main__":
    main()
