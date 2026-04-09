#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
renew_invest.py — Simulation-optimisation for renewable + storage investment.

Solves:
    x* = argmin_x  f(x, VATIC(x))

where x = (wind_mw, solar_mw, battery_mw) and f is a weighted combination of
carbon emissions, production cost, water use, and reliability penalties.

Three-stage search strategy (each stage runs VATIC for every candidate):
  Stage 1 — coarse screening   (default 30 evaluations, Latin Hypercube)
  Stage 2 — local refinement   (default 20 evaluations, neighbourhood of top-3)
  Stage 3 — fine tuning        (default 10 evaluations, tight neighbourhood of best)

Features
--------
  Caching     : identical investment vectors are never re-run (JSON cache +
                presence of thermal_detail.csv in the output dir).
  Parallelism : up to --workers VATIC jobs run concurrently (distinct grid names
                ensure no file contention between workers).
  Fault tolerance : failed runs are logged; the search continues.
  Incremental : re-running the script resumes from cache; only new candidates
                are evaluated.

Usage
-----
    # Balanced (carbon + cost + reliability)
    python scripts/renew_invest.py --config scripts/params_winter.json

    # Carbon-focused sweep, 4 parallel workers
    python scripts/renew_invest.py \\
        --config scripts/params_winter.json \\
        --lambda-carbon 1.0 --lambda-cost 0.0 \\
        --workers 4

    # Custom search bounds
    python scripts/renew_invest.py \\
        --config scripts/params_winter.json \\
        --wind-max 3000 --solar-max 3000 --battery-max 1000

Battery note
------------
Battery storage is added by cloning the existing STORAGE unit at bus 313.
PMax (discharge power) is set to battery_mw / n_buses per bus.  VATIC uses
this power capacity together with the roundtrip efficiency from the reference
unit.  For zero battery_mw the investment still runs; storage capacity is
unchanged from the base grid.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent
VATIC_ROOT  = SCRIPTS_DIR.parent
GRIDS_DIR   = VATIC_ROOT / "vatic" / "data" / "grids"
INIT_DIR    = GRIDS_DIR / "initial-state"

sys.path.insert(0, str(SCRIPTS_DIR))
import cas                    # noqa: E402  emission factor lookup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Investment vector
# ---------------------------------------------------------------------------

@dataclass
class InvestmentVector:
    wind_mw:    float = 0.0   # total new wind capacity (MW), split across buses
    solar_mw:   float = 0.0   # total new solar capacity (MW), split across buses
    battery_mw: float = 0.0   # total new battery power (MW), split across buses

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InvestmentVector":
        return cls(**{k: float(v) for k, v in d.items()
                      if k in ("wind_mw", "solar_mw", "battery_mw")})

    def rounded(self, step: float = 10.0) -> "InvestmentVector":
        """Round each component to nearest step (for cache key stability)."""
        return InvestmentVector(
            wind_mw=round(self.wind_mw / step) * step,
            solar_mw=round(self.solar_mw / step) * step,
            battery_mw=round(self.battery_mw / step) * step,
        )

    def vector_hash(self, step: float = 10.0) -> str:
        """8-character SHA-256 hex of rounded vector."""
        r = self.rounded(step)
        key = f"{r.wind_mw:.0f}_{r.solar_mw:.0f}_{r.battery_mw:.0f}"
        return hashlib.sha256(key.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Transmission-constrained siting
# ---------------------------------------------------------------------------

def find_candidate_buses(
    grid_dir:  Path,
    dc_buses:  list[str],
    max_hops:  int = 2,
) -> list[str]:
    """Return all bus names reachable from any DC bus within *max_hops* branches.

    Uses the grid's branch.csv to build an undirected adjacency graph, then
    performs BFS from each DC bus.  Restricting new renewable/storage capacity
    to these buses provides a simple approximation of transmission headroom:
    capacity sited close to the data-centre load avoids long-distance wheeling
    that would stress lines between regions.

    Parameters
    ----------
    grid_dir : Path
        Root directory of the grid (e.g. ``vatic/data/grids/RTS-GMLC``).
    dc_buses : list[str]
        Names of the buses hosting the data-centre load injection.
    max_hops : int
        Maximum number of branch hops from any DC bus (default 2).

    Returns
    -------
    list[str]
        Sorted list of bus names (includes the DC buses themselves).
        Falls back to *dc_buses* on any I/O or parsing error.
    """
    try:
        bus_csv    = grid_dir / "RTS_Data" / "SourceData" / "bus.csv"
        branch_csv = grid_dir / "RTS_Data" / "SourceData" / "branch.csv"
        bus_df     = pd.read_csv(bus_csv)
        branch_df  = pd.read_csv(branch_csv)

        id_to_name = dict(zip(bus_df["Bus ID"].astype(int),
                              bus_df["Bus Name"].astype(str)))
        name_to_id = {v: k for k, v in id_to_name.items()}

        # Build undirected adjacency: bus_id → set of adjacent bus_ids
        adj: dict[int, set] = {bid: set() for bid in id_to_name}
        for _, row in branch_df.iterrows():
            a, b = int(row["From Bus"]), int(row["To Bus"])
            if a in adj:
                adj[a].add(b)
            if b in adj:
                adj[b].add(a)

        # BFS from all DC buses simultaneously
        seeds = {name_to_id[n] for n in dc_buses if n in name_to_id}
        visited: set[int] = set(seeds)
        frontier = set(seeds)
        for _ in range(max_hops):
            next_frontier: set[int] = set()
            for node in frontier:
                for nb in adj.get(node, set()):
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier

        candidates = sorted(id_to_name[bid] for bid in visited if bid in id_to_name)
        log.info(
            "Transmission siting: %d candidate buses within %d hops of DC buses",
            len(candidates), max_hops,
        )
        return candidates

    except Exception as exc:
        log.warning("find_candidate_buses failed (%s) — falling back to dc_buses", exc)
        return list(dc_buses)


# ---------------------------------------------------------------------------
# Capital cost accounting
# ---------------------------------------------------------------------------

def _crf(r: float, n: int) -> float:
    """Capital Recovery Factor: annual payment per $1 of overnight capital.

    CRF(r, n) = r(1+r)^n / ((1+r)^n − 1)

    Parameters
    ----------
    r : float
        Real discount rate (e.g. 0.05 for 5%).
    n : int
        Asset lifetime in years.
    """
    if r == 0.0:
        return 1.0 / n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def _annualized_capex(x: InvestmentVector, capex: dict, sim_days: int) -> float:
    """Annualised CAPEX + fixed O&M prorated to the simulation window (USD).

    Each technology's overnight capital cost is converted to an equal annual
    payment using its own CRF (discount rate, lifetime), fixed O&M is added,
    and the total is scaled from one year to the simulation period.

    Default cost assumptions (NREL ATB 2024 Moderate, inflated to 2025 USD):
        Source: ATB 2024 Moderate scenario (2022$), inflated via CPI-U (BLS/FRED)
          2022 avg 291.80 → 2025 avg 322.15 → multiplier ×1.104
        Wind    : $1,635/kW overnight ($1,481 × 1.104), $35/kW/yr O&M, 25-yr life
        Solar PV: $1,522/kW overnight ($1,379 × 1.104), $24/kW/yr O&M, 25-yr life
        Battery : $1,326/kW overnight ($1,201 × 1.104), $29/kW/yr O&M, 15-yr life
                  4-hour Li-ion (NMC/LFP); RTE = 85% per Cole & Karmakar (2023),
                  NREL/TP-6A20-85332 — consistent with NREL ATB 2024 utility-scale
                  battery storage assumption across all durations.

    Discount rate (5% real):
        ATB 2024b technology-specific nominal after-tax WACCs (R&D Only case) imply
        real rates of ~3.3–3.7% for wind/solar/battery (nominal WACC 5.9–6.3%,
        deflated by ATB inflation assumption of 2.5%).  A 5% real rate is used as a
        conservative midpoint between this ATB-implied range and the 7–8% real rates
        common in grid-planning and LCOE sensitivity studies (e.g. IEA WEO, NREL
        grid-integration work), providing a modest upward adjustment that reflects
        merchant/policy uncertainty without overstating financing costs.

    Parameters
    ----------
    x : InvestmentVector
        Installed capacity to price.
    capex : dict
        Overrides for any cost assumption.  Keys (all optional):
            wind_per_mw, solar_per_mw, battery_per_mw    ($/MW overnight)
            wind_om_per_mw_yr, solar_om_per_mw_yr,
            battery_om_per_mw_yr                          ($/MW/yr fixed O&M)
            discount_rate                                 (real, fraction)
            wind_life_yr, solar_life_yr, battery_life_yr  (integer years)
            battery_capacity_value_usd_per_mw_yr          ($/MW/yr capacity
                value credit subtracted from battery annualised cost;
                defaults to 0; set to ~50000 for NREL 2024 estimate)
    sim_days : int
        Simulation window length used to prorate annual costs.

    Returns
    -------
    float
        USD cost attributable to the simulation window.
    """
    r    = float(capex.get("discount_rate",    0.05))
    frac = sim_days / 365.0

    wind_ann = x.wind_mw * (
        float(capex.get("wind_per_mw",      1_635_000.0)) * _crf(r, int(capex.get("wind_life_yr",    25)))
        + float(capex.get("wind_om_per_mw_yr",    35_000.0))
    )
    solar_ann = x.solar_mw * (
        float(capex.get("solar_per_mw",     1_522_000.0)) * _crf(r, int(capex.get("solar_life_yr",   25)))
        + float(capex.get("solar_om_per_mw_yr",   24_000.0))
    )
    battery_ann = x.battery_mw * (
        float(capex.get("battery_per_mw",   1_326_000.0)) * _crf(r, int(capex.get("battery_life_yr", 15)))
        + float(capex.get("battery_om_per_mw_yr", 29_000.0))
    )
    battery_capacity_value_usd_per_mw_yr = float(
        capex.get("battery_capacity_value_usd_per_mw_yr", 0.0)
    )
    battery_ann = battery_ann - x.battery_mw * battery_capacity_value_usd_per_mw_yr
    battery_ann = max(battery_ann, 0.0)
    return (wind_ann + solar_ann + battery_ann) * frac


def _annualized_embodied_co2(x: InvestmentVector, capex: dict, sim_days: int) -> float:
    """Annualised embodied (lifecycle construction) CO₂ prorated to sim window (tonnes).

    Embodied carbon covers manufacturing, transport, and installation of the
    asset.  It is spread uniformly over the asset lifetime so that it can be
    compared directly to operational CO₂ on the same time basis.

    Default values (IPCC AR6 median lifecycle, construction phase only):
        Wind    : 700 t CO₂/MW installed  (25-yr life → 28 t/MW/yr)
        Solar PV: 1500 t CO₂/MW installed (25-yr life → 60 t/MW/yr)
        Battery : 600 t CO₂/MW installed  (15-yr life → 40 t/MW/yr)

    Configurable via capex dict keys:
        wind_embodied_tco2_per_mw, solar_embodied_tco2_per_mw,
        battery_embodied_tco2_per_mw
    """
    frac = sim_days / 365.0

    wind_e    = x.wind_mw    * float(capex.get("wind_embodied_tco2_per_mw",    700.0)) \
                / int(capex.get("wind_life_yr",    25))
    solar_e   = x.solar_mw   * float(capex.get("solar_embodied_tco2_per_mw",  1500.0)) \
                / int(capex.get("solar_life_yr",   25))
    battery_e = x.battery_mw * float(capex.get("battery_embodied_tco2_per_mw", 600.0)) \
                / int(capex.get("battery_life_yr", 15))

    return (wind_e + solar_e + battery_e) * frac


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class ResultCache:
    """Persistent JSON cache keyed by vector hash."""

    def __init__(self, cache_file: Path) -> None:
        self._path = cache_file
        self._data: dict[str, dict] = {}
        if cache_file.exists():
            with open(cache_file) as f:
                self._data = json.load(f)
            log.info("Cache loaded: %d entries from %s", len(self._data), cache_file)

    def contains(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def put(self, key: str, result: dict) -> None:
        self._data[key] = result
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2,
                      default=lambda o: int(o) if hasattr(o, "item") else str(o))

    def all_results(self) -> list[dict]:
        return list(self._data.values())


# ---------------------------------------------------------------------------
# Grid modification
# ---------------------------------------------------------------------------

def _apply_investment_t7k(
    x:        InvestmentVector,
    data_dir: Path,
    gen_csv:  Path,
    gen_df:   pd.DataFrame,
) -> None:
    """Apply wind/solar/battery investment to a Texas-7k-family grid.

    Timeseries CSVs use GEN UID as column names (no pointers file needed).
    New generators are added proportional to existing capacity at each site.
    """
    # ── Wind ─────────────────────────────────────────────────────────────────
    if x.wind_mw > 0:
        wind_ts_dir = data_dir / "timeseries_data_files" / "WIND"
        da_files = list(wind_ts_dir.glob("DAY_AHEAD_*.csv"))
        rt_files = list(wind_ts_dir.glob("REAL_TIME_*.csv"))
        if da_files and rt_files:
            da_wind = pd.read_csv(da_files[0])
            rt_wind = pd.read_csv(rt_files[0])
            wind_gens = gen_df[
                gen_df["Fuel"].str.contains("WND|Wind", case=False, na=False)
            ].copy()
            total_pmax = float(wind_gens["PMax MW"].sum())
            if total_pmax > 0:
                new_rows: list[dict] = []
                da_new: dict[str, pd.Series] = {}
                rt_new: dict[str, pd.Series] = {}
                for _, wgen in wind_gens.iterrows():
                    ref_uid = str(wgen["GEN UID"])
                    if ref_uid not in da_wind.columns:
                        continue
                    new_pmax = round(x.wind_mw * float(wgen["PMax MW"]) / total_pmax, 1)
                    if new_pmax < 0.1:
                        continue
                    new_uid = f"{ref_uid}_INV"
                    scale = new_pmax / max(float(wgen["PMax MW"]), 1.0)
                    da_new[new_uid] = (da_wind[ref_uid] * scale).clip(lower=0).round(3)
                    rt_new[new_uid] = (
                        (rt_wind[ref_uid] * scale).clip(lower=0).round(3)
                        if ref_uid in rt_wind.columns
                        else da_new[new_uid]
                    )
                    new_row = wgen.to_dict()
                    new_row["GEN UID"] = new_uid
                    new_row["PMax MW"] = new_pmax
                    new_row["PMin MW"] = 0.0
                    new_rows.append(new_row)
                if new_rows:
                    da_wind = pd.concat(
                        [da_wind, pd.DataFrame(da_new)], axis=1
                    )
                    rt_wind = pd.concat(
                        [rt_wind, pd.DataFrame(rt_new)], axis=1
                    )
                    gen_df = pd.concat(
                        [gen_df, pd.DataFrame(new_rows)], ignore_index=True
                    )
                    da_wind.to_csv(da_files[0], index=False)
                    rt_wind.to_csv(rt_files[0], index=False)
                    log.info("[T7k] Wind: +%.0f MW across %d sites", x.wind_mw, len(new_rows))

    # ── Solar ─────────────────────────────────────────────────────────────────
    if x.solar_mw > 0:
        pv_ts_dir = data_dir / "timeseries_data_files" / "PV"
        da_files  = list(pv_ts_dir.glob("DAY_AHEAD_*.csv"))
        rt_files  = list(pv_ts_dir.glob("REAL_TIME_*.csv"))
        if da_files and rt_files:
            da_pv = pd.read_csv(da_files[0])
            rt_pv = pd.read_csv(rt_files[0])
            solar_gens = gen_df[
                gen_df["Fuel"].str.contains("SUN|Solar", case=False, na=False)
            ].copy()
            total_pmax = float(solar_gens["PMax MW"].sum())
            if total_pmax > 0:
                new_rows = []
                da_new = {}
                rt_new = {}
                for _, sgen in solar_gens.iterrows():
                    ref_uid = str(sgen["GEN UID"])
                    if ref_uid not in da_pv.columns:
                        continue
                    new_pmax = round(x.solar_mw * float(sgen["PMax MW"]) / total_pmax, 1)
                    if new_pmax < 0.1:
                        continue
                    new_uid = f"{ref_uid}_INV"
                    scale = new_pmax / max(float(sgen["PMax MW"]), 1.0)
                    da_new[new_uid] = (da_pv[ref_uid] * scale).clip(lower=0).round(3)
                    rt_new[new_uid] = (
                        (rt_pv[ref_uid] * scale).clip(lower=0).round(3)
                        if ref_uid in rt_pv.columns
                        else da_new[new_uid]
                    )
                    new_row = sgen.to_dict()
                    new_row["GEN UID"] = new_uid
                    new_row["PMax MW"] = new_pmax
                    new_row["PMin MW"] = 0.0
                    new_rows.append(new_row)
                if new_rows:
                    da_pv = pd.concat(
                        [da_pv, pd.DataFrame(da_new)], axis=1
                    )
                    rt_pv = pd.concat(
                        [rt_pv, pd.DataFrame(rt_new)], axis=1
                    )
                    gen_df = pd.concat(
                        [gen_df, pd.DataFrame(new_rows)], ignore_index=True
                    )
                    da_pv.to_csv(da_files[0], index=False)
                    rt_pv.to_csv(rt_files[0], index=False)
                    log.info("[T7k] Solar: +%.0f MW across %d sites", x.solar_mw, len(new_rows))

    # ── Battery ───────────────────────────────────────────────────────────────
    if x.battery_mw > 0:
        batt_mask = gen_df["Unit Type"] == "Batteries"
        if batt_mask.any():
            template = gen_df[batt_mask].iloc[0].to_dict()
            ref_pmax  = max(float(template.get("PMax MW", 1.0)), 1.0)
            scale     = x.battery_mw / ref_pmax
            template["GEN UID"]          = f"{int(template['Bus ID'])}_Battery_INV"
            template["PMax MW"]          = x.battery_mw
            template["PMin MW"]          = 0.0
            template["Ramp Rate MW/Min"] = x.battery_mw
            gen_df = pd.concat(
                [gen_df, pd.DataFrame([template])], ignore_index=True
            )
            log.info("[T7k] Battery: +%.0f MW", x.battery_mw)
        else:
            log.warning("[T7k] No Batteries template found — skipping battery addition")

    gen_df.to_csv(gen_csv, index=False)


def _apply_investment(
    source_grid: str,
    x:           InvestmentVector,
    output_grid: str,
    buses:       list[str],
) -> None:
    """
    Build a modified copy of source_grid with the investment x applied.
    Supports Texas-7k (TX_Data / TX2030_Data) and RTS-GMLC (RTS_Data) grids.
    Does nothing if output_grid already exists (cache hit).
    """
    dst_dir = GRIDS_DIR / output_grid
    if dst_dir.exists():
        return  # already built — cache hit

    import apply_cas_shift as _acs  # lazy import
    data_dir_name = _acs._registry_cfg(source_grid)["data_dir"]

    src_grid_dir = GRIDS_DIR / source_grid
    dst_grid_dir = dst_dir
    dst_data_dir = dst_grid_dir / data_dir_name
    src_init_dir = INIT_DIR / source_grid
    dst_init_dir = INIT_DIR / output_grid

    # Copy grid (and initial-state if present)
    shutil.copytree(src_grid_dir, dst_grid_dir)
    if src_init_dir.is_dir():
        if dst_init_dir.exists():
            shutil.rmtree(dst_init_dir)
        shutil.copytree(src_init_dir, dst_init_dir)

    gen_csv = dst_data_dir / "SourceData" / "gen.csv"
    gen_df  = pd.read_csv(gen_csv)
    gen_df["Bus ID"] = gen_df["Bus ID"].astype(int)

    # ── Texas-7k family ───────────────────────────────────────────────────────
    if data_dir_name in ("TX_Data", "TX2030_Data"):
        _apply_investment_t7k(x, dst_data_dir, gen_csv, gen_df)
        return

    # ── RTS-GMLC family ───────────────────────────────────────────────────────
    import add_renewables as ar  # lazy import
    if x.wind_mw > 0:
        gen_df = ar.add_wind(gen_df, dst_grid_dir, x.wind_mw)
    if x.solar_mw > 0:
        gen_df = ar.add_solar(gen_df, dst_grid_dir, x.solar_mw)
    if x.battery_mw > 0:
        gen_df = ar.add_battery(gen_df, dst_grid_dir, x.battery_mw)
    ar._write_gen(gen_df, dst_grid_dir)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _extract_metrics(sim_dir: Path, gen_csv: Path) -> dict:
    """Return a flat metrics dict from a completed VATIC output directory."""
    hourly  = pd.read_csv(sim_dir / "hourly_summary.csv")
    thermal = pd.read_csv(sim_dir / "thermal_detail.csv")
    renew   = pd.read_csv(sim_dir / "renew_detail.csv")

    ef = cas._emission_factors(gen_csv)
    thermal["co2_t"] = thermal["Dispatch"] * thermal["Generator"].map(ef).fillna(0.0)

    total_mwh  = thermal["Dispatch"].sum() + renew["Output"].sum()
    total_co2  = thermal["co2_t"].sum() / 1000.0      # metric tonnes
    mean_ci    = (total_co2 * 1000.0 / total_mwh) if total_mwh > 0 else 0.0  # kg/MWh

    return {
        "total_co2_tonnes":         round(total_co2, 1),
        "mean_ci_kgco2_mwh":        round(mean_ci, 3),
        "total_cost_usd":           round(hourly["VariableCosts"].sum() + hourly["FixedCosts"].sum(), 0),
        "variable_cost_usd":        round(hourly["VariableCosts"].sum(), 0),
        "avg_lmp_usd_mwh":          round(hourly["Price"].mean(), 3),
        "load_shedding_mwh":        round(hourly["LoadShedding"].sum(), 2),
        "renewables_used_mwh":      round(hourly["RenewablesUsed"].sum(), 1),
        "renewables_curtailed_mwh": round(hourly["RenewablesCurtailment"].sum(), 1),
        "total_generation_mwh":     round(total_mwh, 1),
    }


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def _compute_objective(
    metrics:          dict,
    baseline:         dict,
    lambda_carbon:    float,
    lambda_cost:      float,
    lambda_reliability: float,
    lambda_water:     float = 0.0,
    water_wd_gal:     float = 0.0,
    base_water_wd:    float = 1.0,
    reliability_penalty:  float = 1000.0,
    curtailment_penalty:  float = 0.0,
    capex_usd:        float = 0.0,
    embodied_co2_t:   float = 0.0,
) -> float:
    """Weighted objective (lower = better).

    Each term is normalised by the baseline value so weights are dimensionless.
    Capital costs (``capex_usd``) are added to the candidate's operational cost
    before normalisation, so expensive investments are penalised relative to
    the baseline operational cost.

    ``curtailment_penalty`` ($/MWh) directly penalises curtailed renewable
    generation, discouraging the optimiser from over-building beyond what the
    grid can absorb.

    ``embodied_co2_t`` (tonnes) is the annualised lifecycle construction carbon
    for the investment, added to operational CO₂ before normalisation so the
    carbon term reflects full lifecycle emissions.
    """
    b_co2  = baseline["total_co2_tonnes"]   or 1.0
    b_cost = baseline["total_cost_usd"]     or 1.0
    b_wd   = base_water_wd                  or 1.0

    total_co2        = metrics["total_co2_tonnes"] + embodied_co2_t
    carbon_term      = (total_co2 / b_co2)                                            if b_co2 > 0 else 0.0
    cost_term        = ((metrics["total_cost_usd"] + capex_usd) / b_cost)             if b_cost > 0 else 0.0
    shed             = metrics.get("load_shedding_mwh",      0.0)
    curtailed        = metrics.get("renewables_curtailed_mwh", 0.0)
    reliability_term = (reliability_penalty  * shed      / b_cost)                    if b_cost > 0 else 0.0
    curtailment_term = (curtailment_penalty  * curtailed / b_cost)                    if b_cost > 0 else 0.0
    water_term       = (water_wd_gal / b_wd)                                          if b_wd > 0 else 0.0

    return (
        lambda_carbon      * carbon_term
      + lambda_cost        * cost_term
      + lambda_reliability * reliability_term
      + lambda_cost        * curtailment_term
      + lambda_water       * water_term
    )


# ---------------------------------------------------------------------------
# VATIC runner (subprocess)
# ---------------------------------------------------------------------------

def _run_vatic(
    grid:          str,
    out_dir:       Path,
    sim_date:      str,
    sim_days:      int,
    solver:        str,
    solver_args:   str,
    threads:       int,
    ruc_horizon:   int,
    sced_horizon:  int,
    ruc_mipgap:    float,
    reserve_factor: float,
    output_detail: int,
    thermal_rating_scale: float = 1.0,
) -> bool:
    """
    Run vatic-det for one grid/date combination.
    Returns True on success, False on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bash_cmd = (
        f"module load gurobi/10.0.1 && "
        f"vatic-det {grid} {sim_date} {sim_days}"
        f" --solver {solver}"
        f" --solver-args {solver_args}"
        f" --threads {threads}"
        f" --output-detail {output_detail}"
        f" --ruc-horizon {ruc_horizon}"
        f" --sced-horizon {sced_horizon}"
        f" --ruc-mipgap {ruc_mipgap}"
        f" --reserve-factor {reserve_factor}"
        f" --thermal-rating-scale {thermal_rating_scale}"
        f" --csv --lmps"
        f" --out-dir {out_dir}"
    )
    try:
        result = subprocess.run(bash_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("VATIC failed for grid=%s:\n%s", grid, result.stderr[-2000:])
            return False
        return True
    except Exception as exc:
        log.error("VATIC subprocess error for grid=%s: %s", grid, exc)
        return False


# ---------------------------------------------------------------------------
# Single evaluation (picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _evaluate(args: dict) -> dict:
    """
    Full pipeline for one investment vector:
      1. Build modified grid
      2. Run VATIC (once per week when multi_week_dates is set)
      3. Extract metrics (summed across weeks when multi-week)
      4. Compute objective

    args dict keys:
        x_dict, source_grid, buses, gen_csv_str,
        sim_date, sim_days, sim_params (solver etc.),
        out_root_str, baseline_metrics, weights, water_base_wd,
        reliability_penalty,
        multi_week_dates     (optional list of ISO date strings),
        multi_week_baselines (optional list of per-week baseline metric dicts)
    """
    x       = InvestmentVector.from_dict(args["x_dict"])
    h       = x.vector_hash()
    source  = args["source_grid"]
    buses   = args["buses"]
    gen_csv = Path(args["gen_csv_str"])
    out_root = Path(args["out_root_str"])
    sim_days = args["sim_days"]

    output_grid = f"{source}-INV-{h}"

    # ── Determine simulation directories (Change 3: sim_days prefix) ─────────
    multi_week_dates = args.get("multi_week_dates") or []

    if multi_week_dates:
        # Multi-week: one sub-dir per date, under sims/{sim_days}d/{hash}/{date}
        sim_dirs = [
            out_root / "sims" / f"{sim_days}d" / h / d
            for d in multi_week_dates
        ]
        sim_dir = sim_dirs[0]   # canonical sim_dir for cache result field
    else:
        # Single-week: sims/{sim_days}d/{hash}
        sim_dir  = out_root / "sims" / f"{sim_days}d" / h
        sim_dirs = [sim_dir]

    capex_cfg    = args.get("capex_cfg", {})
    capex_usd    = _annualized_capex(x, capex_cfg, sim_days)
    embodied_co2 = _annualized_embodied_co2(x, capex_cfg, sim_days)

    result: dict = {
        "hash":             h,
        "investment":       x.to_dict(),
        "output_grid":      output_grid,
        "sim_dir":          str(sim_dir),
        "status":           "pending",
        "metrics":          {},
        "capex_usd":        round(capex_usd, 0),
        "objective":        float("inf"),
    }

    t0 = time.time()

    # ── Build grid (once, shared across all weeks) ────────────────────────────
    try:
        _apply_investment(source, x, output_grid, buses)
    except Exception as exc:
        result["status"] = f"grid_error: {exc}"
        log.error("[%s] Grid build failed: %s", h, exc)
        return result

    # ── Run VATIC (one call per week) ─────────────────────────────────────────
    sp = args["sim_params"]
    dates_to_run = multi_week_dates if multi_week_dates else [args["sim_date"]]

    for wdate, wdir in zip(dates_to_run, sim_dirs):
        if not (wdir / "thermal_detail.csv").exists():
            ok = _run_vatic(
                output_grid, wdir,
                wdate, sim_days,
                sp["solver"], sp["solver_args"], sp["threads"],
                sp["ruc_horizon"], sp["sced_horizon"],
                sp["ruc_mipgap"], sp["reserve_factor"],
                sp["output_detail"],
                sp.get("thermal_rating_scale", 1.0),
            )
            if not ok:
                result["status"] = "vatic_failed"
                return result
        else:
            log.info("[%s] VATIC output cached for date=%s — skipping", h, wdate)

    # ── Extract metrics (sum additive fields across weeks) ────────────────────
    try:
        if len(sim_dirs) == 1:
            metrics = _extract_metrics(sim_dirs[0], gen_csv)
        else:
            # Sum additive metrics across all weeks; recompute derived ones.
            per_week = [_extract_metrics(d, gen_csv) for d in sim_dirs]
            metrics = {
                "total_co2_tonnes":         round(sum(m["total_co2_tonnes"]         for m in per_week), 1),
                "total_cost_usd":           round(sum(m["total_cost_usd"]           for m in per_week), 0),
                "variable_cost_usd":        round(sum(m["variable_cost_usd"]        for m in per_week), 0),
                "load_shedding_mwh":        round(sum(m["load_shedding_mwh"]        for m in per_week), 2),
                "renewables_used_mwh":      round(sum(m["renewables_used_mwh"]      for m in per_week), 1),
                "renewables_curtailed_mwh": round(sum(m["renewables_curtailed_mwh"] for m in per_week), 1),
                "total_generation_mwh":     round(sum(m["total_generation_mwh"]     for m in per_week), 1),
            }
            total_gen = metrics["total_generation_mwh"]
            total_co2 = metrics["total_co2_tonnes"]
            metrics["mean_ci_kgco2_mwh"] = round(
                (total_co2 * 1000.0 / total_gen) if total_gen > 0 else 0.0, 3
            )
            # avg_lmp: weighted by generation across weeks
            avg_lmp = sum(
                m["avg_lmp_usd_mwh"] * m["total_generation_mwh"]
                for m in per_week
            ) / max(total_gen, 1.0)
            metrics["avg_lmp_usd_mwh"] = round(avg_lmp, 3)
    except Exception as exc:
        result["status"] = f"metric_error: {exc}"
        log.error("[%s] Metric extraction failed: %s", h, exc)
        return result

    # ── Compute objective ─────────────────────────────────────────────────────
    w = args["weights"]
    obj = _compute_objective(
        metrics,
        args["baseline_metrics"],
        lambda_carbon=w["lambda_carbon"],
        lambda_cost=w["lambda_cost"],
        lambda_reliability=w["lambda_reliability"],
        lambda_water=w["lambda_water"],
        water_wd_gal=metrics.get("total_wd_gal", 0.0),
        base_water_wd=args["water_base_wd"],
        reliability_penalty=args["reliability_penalty"],
        curtailment_penalty=args.get("curtailment_penalty", 0.0),
        capex_usd=capex_usd,
        embodied_co2_t=embodied_co2,
    )

    elapsed = time.time() - t0
    result.update({
        "status":                    "ok",
        "metrics":                   metrics,
        "capex_usd":                 round(capex_usd, 0),
        "embodied_co2_t":            round(embodied_co2, 2),
        "total_cost_with_capex_usd": round(metrics["total_cost_usd"] + capex_usd, 0),
        "objective":                 round(obj, 6),
        "elapsed_s":                 round(elapsed, 1),
    })
    log.info(
        "[%s] wind=%.0f solar=%.0f bat=%.0f | co2=%.1ft cost=%.0f$ shed=%.1f curt=%.1f | obj=%.4f  (%.0fs)",
        h, x.wind_mw, x.solar_mw, x.battery_mw,
        metrics["total_co2_tonnes"], metrics["total_cost_usd"],
        metrics["load_shedding_mwh"], metrics.get("renewables_curtailed_mwh", 0.0),
        obj, elapsed,
    )
    return result


# ---------------------------------------------------------------------------
# Search: Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def _lhs_samples(
    n:     int,
    bounds: list[tuple[float, float]],
    seed:  int = 42,
) -> np.ndarray:
    """
    Return (n, d) array of Latin Hypercube samples in [lo, hi] per dimension.
    Uses scipy.stats.qmc if available, falls back to stratified random.
    """
    d = len(bounds)
    try:
        from scipy.stats.qmc import LatinHypercube, scale as qmc_scale
        sampler = LatinHypercube(d=d, seed=seed)
        unit    = sampler.random(n=n)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        return qmc_scale(unit, lo, hi)
    except ImportError:
        rng = np.random.default_rng(seed)
        lo  = np.array([b[0] for b in bounds], dtype=float)
        hi  = np.array([b[1] for b in bounds], dtype=float)
        cuts = np.linspace(0, 1, n + 1)
        pts  = np.empty((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            u    = rng.uniform(cuts[:-1], cuts[1:])[perm]
            pts[:, j] = lo[j] + u * (hi[j] - lo[j])
        return pts


def _neighborhood_samples(
    center:     InvestmentVector,
    n:          int,
    bounds:     list[tuple[float, float]],
    sigma_frac: float = 0.15,
    seed:       int = 0,
) -> list[InvestmentVector]:
    """
    Generate n perturbations around center using Gaussian noise clipped to bounds.
    sigma = sigma_frac * (hi - lo) for each dimension.
    """
    rng    = np.random.default_rng(seed)
    lo     = np.array([b[0] for b in bounds])
    hi     = np.array([b[1] for b in bounds])
    sigma  = sigma_frac * (hi - lo)
    base   = np.array([center.wind_mw, center.solar_mw, center.battery_mw])
    samples = []
    for _ in range(n):
        perturbed = np.clip(base + rng.normal(0, sigma), lo, hi)
        samples.append(InvestmentVector(
            wind_mw=float(perturbed[0]),
            solar_mw=float(perturbed[1]),
            battery_mw=float(perturbed[2]),
        ))
    return samples


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def _run_stage(
    candidates: list[InvestmentVector],
    cache:      ResultCache,
    base_args:  dict,
    n_workers:  int,
    stage_name: str,
) -> list[dict]:
    """
    Evaluate all candidates (skipping cached ones) using up to n_workers
    parallel VATIC runs.  Returns list of result dicts (cached + new).
    """
    to_run:    list[tuple[InvestmentVector, str]] = []
    cached_res: list[dict] = []

    for x in candidates:
        h = x.vector_hash()
        if cache.contains(h):
            r = cache.get(h)
            if r and r.get("status") == "ok":
                cached_res.append(r)
                log.info("[%s] %s: cache hit", h, stage_name)
                continue
        to_run.append((x, h))

    log.info("%s: %d candidates, %d cached, %d to run",
             stage_name, len(candidates), len(cached_res), len(to_run))

    new_results: list[dict] = []
    if to_run:
        eval_args = [
            {**base_args, "x_dict": x.to_dict()} for x, _ in to_run
        ]
        if n_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
                for result in pool.map(_evaluate, eval_args):
                    cache.put(result["hash"], result)
                    if result["status"] == "ok":
                        new_results.append(result)
                    else:
                        log.warning("Run %s failed: %s", result["hash"], result["status"])
        else:
            for ea in eval_args:
                result = _evaluate(ea)
                cache.put(result["hash"], result)
                if result["status"] == "ok":
                    new_results.append(result)
                else:
                    log.warning("Run %s failed: %s", result["hash"], result["status"])

    return cached_res + new_results


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _build_results_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r.get("metrics", {})
        capex_usd = r.get("capex_usd", 0.0)
        row = {
            "hash":                      r["hash"],
            "output_grid":               r.get("output_grid", ""),
            "wind_mw":                   r["investment"]["wind_mw"],
            "solar_mw":                  r["investment"]["solar_mw"],
            "battery_mw":                r["investment"]["battery_mw"],
            "objective":                 r.get("objective", float("inf")),
            "total_co2_tonnes":          m.get("total_co2_tonnes"),
            "total_cost_usd":            m.get("total_cost_usd"),
            "capex_usd":                 capex_usd,
            "total_cost_with_capex_usd": r.get("total_cost_with_capex_usd",
                                               (m.get("total_cost_usd") or 0) + capex_usd),
            "variable_cost_usd":         m.get("variable_cost_usd"),
            "avg_lmp_usd_mwh":           m.get("avg_lmp_usd_mwh"),
            "load_shedding_mwh":         m.get("load_shedding_mwh"),
            "renewables_used_mwh":       m.get("renewables_used_mwh"),
            "renewables_curtailed_mwh":  m.get("renewables_curtailed_mwh"),
            "mean_ci_kgco2_mwh":         m.get("mean_ci_kgco2_mwh"),
            "status":                    r.get("status"),
        }
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
    return df


def _plot_results(df: pd.DataFrame, baseline: dict, out_dir: Path) -> None:
    """Scatter plots: wind vs solar coloured by objective, and Pareto CO₂ vs cost."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtk
    except ImportError:
        return

    import matplotlib.ticker as mtk

    def _afmt(ax, axis="y"):
        """Consistent formatter — same decimal places for every tick inc. 0."""
        lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
        span = max(abs(hi), abs(lo), 1e-12)
        if span >= 1e6:
            return mtk.FuncFormatter(lambda v, _: f"{v/1e6:,.1f}M")
        if span >= 1e3:
            return mtk.FuncFormatter(lambda v, _: f"{v:,.0f}")
        if span >= 10:
            return mtk.FuncFormatter(lambda v, _: f"{v:.1f}")
        return mtk.FuncFormatter(lambda v, _: f"{v:.3f}")

    df_ok = df[df["status"] == "ok"].copy()
    if df_ok.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── 1. Wind vs Solar coloured by objective ──────────────────────────────
    sc = axes[0].scatter(
        df_ok["wind_mw"], df_ok["solar_mw"],
        c=df_ok["objective"], cmap="RdYlGn_r", s=60, edgecolors="gray", lw=0.5
    )
    axes[0].set_xlabel("Wind added (MW)")
    axes[0].set_ylabel("Solar added (MW)")
    axes[0].set_title("Search landscape: Wind vs Solar")
    plt.colorbar(sc, ax=axes[0], label="Objective")
    # Mark best
    best = df_ok.iloc[0]
    axes[0].scatter(best["wind_mw"], best["solar_mw"],
                    marker="*", s=250, c="gold", edgecolors="black", zorder=5)
    axes[0].xaxis.set_major_locator(mtk.MaxNLocator(6))
    axes[0].yaxis.set_major_locator(mtk.MaxNLocator(6))
    axes[0].xaxis.set_major_formatter(_afmt(axes[0], axis="x"))
    axes[0].yaxis.set_major_formatter(_afmt(axes[0]))

    # ── 2. CO₂ vs Cost Pareto frontier ──────────────────────────────────────
    axes[1].scatter(df_ok["total_cost_usd"] / 1e6, df_ok["total_co2_tonnes"] / 1e3,
                    c=df_ok["objective"], cmap="RdYlGn_r", s=60, edgecolors="gray", lw=0.5)
    # Baseline marker
    axes[1].scatter(
        baseline["total_cost_usd"] / 1e6, baseline["total_co2_tonnes"] / 1e3,
        marker="D", s=100, c="red", label="Baseline", zorder=5
    )
    axes[1].scatter(best["total_cost_usd"] / 1e6, best["total_co2_tonnes"] / 1e3,
                    marker="*", s=250, c="gold", edgecolors="black",
                    label="Best", zorder=6)
    axes[1].set_xlabel("Total Cost ($M)")
    axes[1].set_ylabel("CO₂ (kt)")
    axes[1].set_title("Cost vs Carbon tradeoff")
    axes[1].legend(fontsize=8)
    axes[1].xaxis.set_major_locator(mtk.MaxNLocator(5))
    axes[1].yaxis.set_major_locator(mtk.MaxNLocator(5))
    axes[1].xaxis.set_major_formatter(_afmt(axes[1], axis="x"))
    axes[1].yaxis.set_major_formatter(_afmt(axes[1]))

    # ── 3. Objective distribution ────────────────────────────────────────────
    axes[2].hist(df_ok["objective"], bins=20, color="#4e79a7", edgecolor="white")
    axes[2].axvline(best["objective"], color="gold", lw=2, label=f"Best: {best['objective']:.4f}")
    axes[2].set_xlabel("Objective value")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Objective distribution across candidates")
    axes[2].legend(fontsize=8)
    axes[2].xaxis.set_major_locator(mtk.MaxNLocator(5))
    axes[2].xaxis.set_major_formatter(_afmt(axes[2], axis="x"))
    axes[2].yaxis.set_major_formatter(_afmt(axes[2]))

    fig.suptitle("Renewable Investment Optimisation Results", fontsize=12)
    fig.tight_layout()
    p = out_dir / "renew_opt_results.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    log.info("Plot → %s", p)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_optimisation(
    source_grid:   str,
    dc_grid:       str,
    buses:         list[str],
    gen_csv:       Path,
    sim_date:      str,
    sim_days:      int,
    sim_params:    dict,
    out_dir:       Path,
    baseline_dir:  Path,
    bounds:        dict,            # {wind_max, solar_max, battery_max}
    weights:       dict,            # {lambda_carbon, lambda_cost, lambda_reliability, lambda_water}
    stage_n:       tuple[int, int, int],
    reliability_penalty:  float = 1000.0,
    curtailment_penalty:  float = 0.0,
    n_workers:     int = 1,
    seed:          int = 42,
    capex_cfg:     dict | None = None,  # capital cost assumptions (see _annualized_capex)
    tx_max_hops:   int = 0,             # 0 = no topology constraint; N>0 = BFS hops from dc buses
    multi_week_configs: "list[tuple[str, Path]] | None" = None,
        # list of (sim_date, baseline_dir) pairs for multi-week evaluation.
        # When None, falls back to single-week using sim_date + baseline_dir.
) -> pd.DataFrame:
    """
    Run the three-stage simulation-optimisation loop.
    Returns a DataFrame of all evaluated candidates sorted by objective.

    When *tx_max_hops* > 0, new wind/solar/battery capacity is sited only on
    buses reachable from the DC buses within that many branch hops (transmission-
    constrained siting).  Set to 0 (default) to use *buses* directly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ResultCache(out_dir / "cache.json")

    # ── Resolve siting buses (may be topology-expanded) ───────────────────────
    if tx_max_hops > 0:
        base_grid_dir = GRIDS_DIR / source_grid
        siting_buses = find_candidate_buses(base_grid_dir, buses, max_hops=tx_max_hops)
        log.info("Transmission-constrained siting: %d buses (hops=%d)", len(siting_buses), tx_max_hops)
    else:
        siting_buses = list(buses)

    # ── Compute baseline metrics ──────────────────────────────────────────────
    if multi_week_configs:
        # Multi-week: sum baseline metrics across all weeks.
        log.info(
            "Loading baseline metrics for %d weeks: %s",
            len(multi_week_configs),
            [d for d, _ in multi_week_configs],
        )
        per_week_baselines: list[dict] = []
        for _wdate, _bdir in multi_week_configs:
            log.info("  Loading baseline for %s from %s", _wdate, _bdir)
            per_week_baselines.append(_extract_metrics(_bdir, gen_csv))
        # Sum additive fields; recompute derived ones
        baseline_metrics = {
            "total_co2_tonnes":         round(sum(m["total_co2_tonnes"]         for m in per_week_baselines), 1),
            "total_cost_usd":           round(sum(m["total_cost_usd"]           for m in per_week_baselines), 0),
            "variable_cost_usd":        round(sum(m["variable_cost_usd"]        for m in per_week_baselines), 0),
            "load_shedding_mwh":        round(sum(m["load_shedding_mwh"]        for m in per_week_baselines), 2),
            "renewables_used_mwh":      round(sum(m["renewables_used_mwh"]      for m in per_week_baselines), 1),
            "renewables_curtailed_mwh": round(sum(m["renewables_curtailed_mwh"] for m in per_week_baselines), 1),
            "total_generation_mwh":     round(sum(m["total_generation_mwh"]     for m in per_week_baselines), 1),
        }
        _btotal_gen = baseline_metrics["total_generation_mwh"]
        _btotal_co2 = baseline_metrics["total_co2_tonnes"]
        baseline_metrics["mean_ci_kgco2_mwh"] = round(
            (_btotal_co2 * 1000.0 / _btotal_gen) if _btotal_gen > 0 else 0.0, 3
        )
        _bavg_lmp = sum(
            m["avg_lmp_usd_mwh"] * m["total_generation_mwh"]
            for m in per_week_baselines
        ) / max(_btotal_gen, 1.0)
        baseline_metrics["avg_lmp_usd_mwh"] = round(_bavg_lmp, 3)
        water_base_wd = baseline_metrics.get("total_wd_gal", 1.0)
        multi_week_dates    = [d for d, _ in multi_week_configs]
        multi_week_baselines = per_week_baselines
    else:
        log.info("Loading baseline metrics from %s", baseline_dir)
        baseline_metrics = _extract_metrics(baseline_dir, gen_csv)
        water_base_wd    = baseline_metrics.get("total_wd_gal", 1.0)
        multi_week_dates    = []
        multi_week_baselines = []

    # Base args shared by all evaluate calls
    base_args = {
        "source_grid":          dc_grid,
        "buses":                siting_buses,
        "gen_csv_str":          str(gen_csv),
        "sim_date":             sim_date,
        "sim_days":             sim_days,
        "sim_params":           sim_params,
        "out_root_str":         str(out_dir),
        "baseline_metrics":     baseline_metrics,
        "weights":              weights,
        "water_base_wd":        water_base_wd,
        "reliability_penalty":  reliability_penalty,
        "curtailment_penalty":  curtailment_penalty,
        "capex_cfg":            capex_cfg or {},
        "multi_week_dates":     multi_week_dates,
        "multi_week_baselines": multi_week_baselines,
    }

    dim_bounds = [
        (0.0, bounds["wind_max"]),
        (0.0, bounds["solar_max"]),
        (0.0, bounds["battery_max"]),
    ]

    n1, n2, n3 = stage_n
    all_results: list[dict] = []

    # ══ Stage 1: coarse LHS screen ═══════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 1: Coarse screening (%d LHS candidates)", n1)
    log.info("=" * 60)
    def _colocate(candidates: list[InvestmentVector]) -> list[InvestmentVector]:
        """Clip battery_mw to wind_mw + solar_mw (co-location constraint).

        Prevents the optimizer from building a battery larger than the paired
        renewable capacity, which would result in the battery primarily charging
        from fossil sources.  Battery energy storage = power * duration_h, and
        a battery can only sustainably charge from what the co-located renewables
        produce.  This is an approximation: on an AC bus the electrons are
        indistinguishable by source, but sizing battery <= wind+solar ensures
        the optimizer pairs storage with a plausible renewable charging fleet.
        """
        out = []
        for x in candidates:
            max_bat = x.wind_mw + x.solar_mw
            out.append(InvestmentVector(
                wind_mw    = x.wind_mw,
                solar_mw   = x.solar_mw,
                battery_mw = min(x.battery_mw, max_bat),
            ))
        return out

    pts1  = _lhs_samples(n1, dim_bounds, seed=seed)
    cands1 = _colocate([InvestmentVector(w, s, b) for w, s, b in pts1])
    # Always include zero-investment baseline
    cands1.append(InvestmentVector(0, 0, 0))

    res1 = _run_stage(cands1, cache, base_args, n_workers, "Stage1")
    all_results.extend(res1)

    if not res1:
        log.error("Stage 1 produced no successful results — aborting")
        return _build_results_table(all_results)

    res1_sorted = sorted(res1, key=lambda r: r["objective"])
    top_k       = res1_sorted[:3]
    log.info("Stage 1 best: obj=%.4f  wind=%.0f  solar=%.0f  bat=%.0f",
             top_k[0]["objective"],
             top_k[0]["investment"]["wind_mw"],
             top_k[0]["investment"]["solar_mw"],
             top_k[0]["investment"]["battery_mw"])

    # ══ Stage 2: local refinement around top-3 ═══════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 2: Local refinement (%d candidates around top-3)", n2)
    log.info("=" * 60)
    cands2: list[InvestmentVector] = []
    per_center = max(1, n2 // len(top_k))
    for i, tr in enumerate(top_k):
        center = InvestmentVector.from_dict(tr["investment"])
        cands2.extend(_neighborhood_samples(
            center, per_center, dim_bounds, sigma_frac=0.15, seed=seed + i + 1
        ))
    cands2 = _colocate(cands2)

    res2 = _run_stage(cands2, cache, base_args, n_workers, "Stage2")
    all_results.extend(res2)

    combined = sorted(res1 + res2, key=lambda r: r["objective"])
    best_so_far = combined[0]
    log.info("After Stage 2 best: obj=%.4f  wind=%.0f  solar=%.0f  bat=%.0f",
             best_so_far["objective"],
             best_so_far["investment"]["wind_mw"],
             best_so_far["investment"]["solar_mw"],
             best_so_far["investment"]["battery_mw"])

    # ══ Stage 3: fine tuning around best ═════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 3: Fine tuning (%d candidates around best)", n3)
    log.info("=" * 60)
    best_vec = InvestmentVector.from_dict(best_so_far["investment"])
    cands3   = _colocate(_neighborhood_samples(
        best_vec, n3, dim_bounds, sigma_frac=0.05, seed=seed + 99
    ))
    res3 = _run_stage(cands3, cache, base_args, n_workers, "Stage3")
    all_results.extend(res3)

    # ── Final table ───────────────────────────────────────────────────────────
    df = _build_results_table(all_results)

    csv_path = out_dir / "renew_opt_results.csv"
    df.to_csv(csv_path, index=False)
    log.info("Full results → %s", csv_path)

    best = df.iloc[0]
    log.info("=" * 60)
    log.info("OPTIMAL PORTFOLIO")
    log.info("  Wind added  : %.0f MW", best["wind_mw"])
    log.info("  Solar added : %.0f MW", best["solar_mw"])
    log.info("  Battery     : %.0f MW", best["battery_mw"])
    log.info("  Objective   : %.4f", best["objective"])
    log.info("  CO₂         : %.0f tonnes  (Δ vs baseline: %+.1f%%)",
             best["total_co2_tonnes"],
             (best["total_co2_tonnes"] / baseline_metrics["total_co2_tonnes"] - 1) * 100
             if baseline_metrics["total_co2_tonnes"] > 0 else 0)
    log.info("  Opex        : $%.0f  (Δ vs baseline: %+.1f%%)",
             best["total_cost_usd"],
             (best["total_cost_usd"] / baseline_metrics["total_cost_usd"] - 1) * 100
             if baseline_metrics["total_cost_usd"] > 0 else 0)
    log.info("  CAPEX (sim) : $%.0f  (prorated to %d-day window)",
             best.get("capex_usd", 0), sim_days)
    log.info("  Total cost  : $%.0f  (opex + capex)",
             best.get("total_cost_with_capex_usd", best["total_cost_usd"]))
    log.info("=" * 60)

    _plot_results(df[df["status"] == "ok"], baseline_metrics, out_dir)

    # ── Clean up non-winning candidate grids ──────────────────────────────────
    # Each candidate grid is ~230 MB; only the winner is needed downstream.
    winning_grid = best["output_grid"]
    import shutil
    removed, kept = 0, 0
    for row in df.itertuples():
        grid = row.output_grid
        if not grid or grid == winning_grid:
            kept += 1
            continue
        for d in [GRIDS_DIR / grid, INIT_DIR / grid]:
            if d.exists():
                shutil.rmtree(d)
        removed += 1
    if removed:
        log.info("Cleaned up %d non-winning candidate grids (kept: %s)", removed, winning_grid)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path,
                   default=Path("scripts/params.json"),
                   help="VATIC params JSON (default: scripts/params.json)")
    p.add_argument("--baseline-dir", type=Path, default=None,
                   help="Pre-existing baseline sim dir. Defaults to <out-root>/baseline")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: outputs/<date>/renew_opt)")

    # Search bounds
    p.add_argument("--wind-max",    type=float, default=5000.0)
    p.add_argument("--solar-max",   type=float, default=5000.0)
    p.add_argument("--battery-max", type=float, default=2000.0)

    # Objective weights
    p.add_argument("--lambda-carbon",      type=float, default=0.5)
    p.add_argument("--lambda-cost",        type=float, default=0.3)
    p.add_argument("--lambda-reliability", type=float, default=0.2)
    p.add_argument("--lambda-water",       type=float, default=0.0)
    p.add_argument("--reliability-penalty", type=float, default=1000.0,
                   help="Multiplier on load-shedding in objective (default: 1000)")
    p.add_argument("--curtailment-penalty", type=float, default=0.0,
                   help="Penalty per MWh of curtailed renewable generation (default: 0)")

    # Search size
    p.add_argument("--stage1-n", type=int, default=30)
    p.add_argument("--stage2-n", type=int, default=20)
    p.add_argument("--stage3-n", type=int, default=10)
    p.add_argument("--seed",     type=int, default=42)

    # Execution
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel VATIC jobs (default: 1 = serial)")

    # Capital costs — NREL ATB 2024 Moderate (2022$) inflated to 2025 USD via CPI-U (BLS/FRED)
    p.add_argument("--wind-per-mw",       type=float, default=1_635_000.0,
                   help="Overnight CAPEX for wind ($/MW 2025$, default: 1635000)")
    p.add_argument("--solar-per-mw",      type=float, default=1_522_000.0,
                   help="Overnight CAPEX for solar PV ($/MW 2025$, default: 1522000)")
    p.add_argument("--battery-per-mw",    type=float, default=1_326_000.0,
                   help="Overnight CAPEX for 2-hr Li-ion battery ($/MW 2025$, default: 1326000)")
    p.add_argument("--wind-om-per-mw-yr", type=float, default=35_000.0,
                   help="Fixed O&M for wind ($/MW/yr 2025$, default: 35000)")
    p.add_argument("--solar-om-per-mw-yr",type=float, default=24_000.0,
                   help="Fixed O&M for solar PV ($/MW/yr 2025$, default: 24000)")
    p.add_argument("--battery-om-per-mw-yr", type=float, default=29_000.0,
                   help="Fixed O&M for battery ($/MW/yr 2025$, default: 29000)")
    p.add_argument("--discount-rate",     type=float, default=0.05,
                   help="Real discount rate for CRF (default: 0.05)")
    p.add_argument("--wind-life-yr",      type=int,   default=25)
    p.add_argument("--solar-life-yr",     type=int,   default=25)
    p.add_argument("--battery-life-yr",   type=int,   default=15)
    p.add_argument("--battery-capacity-value", type=float, default=0.0,
                   help="Capacity value credit for battery storage ($/MW/yr, "
                        "default: 0 — set to ~50000 for NREL 2024 estimate)")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.config.exists():
        sys.exit(f"Config not found: {args.config}")

    with open(args.config) as f:
        cfg = json.load(f)
    cfg.pop("_comment", None)

    g   = cfg["grid"]
    sim = cfg["simulation"]

    BASE_GRID  = g["base_grid"]
    DC_GRID    = g["dc_grid"]
    DC_BUSES   = g["dc_buses"]
    SIM_DATE   = sim["date"]
    SIM_DAYS   = sim["days"]

    from datetime import date, timedelta
    sim_end = (date.fromisoformat(SIM_DATE) + timedelta(days=SIM_DAYS - 1)).isoformat()
    _ = sim_end  # used by CAS analysis if needed

    out_root    = Path("outputs") / SIM_DATE
    out_dir     = args.out_dir or (out_root / "renew_opt")
    baseline_dir = args.baseline_dir or (out_root / "baseline")

    if not baseline_dir.exists():
        sys.exit(
            f"Baseline simulation directory not found: {baseline_dir}\n"
            "Run main.py first (steps 1-2) to generate the baseline VATIC outputs."
        )

    gen_csv = (GRIDS_DIR / BASE_GRID / "RTS_Data" / "SourceData" / "gen.csv")
    if not gen_csv.exists():
        sys.exit(f"gen.csv not found: {gen_csv}")

    sim_params = {
        "solver":               sim.get("solver",               "gurobi"),
        "solver_args":          sim.get("solver_args",          "Cuts=1 Presolve=1"),
        "threads":              sim.get("threads",               8),
        "ruc_horizon":          sim.get("ruc_horizon",          48),
        "sced_horizon":         sim.get("sced_horizon",          4),
        "ruc_mipgap":           sim.get("ruc_mipgap",           0.01),
        "reserve_factor":       sim.get("reserve_factor",       0.05),
        "output_detail":        sim.get("output_detail",         2),
        "thermal_rating_scale": sim.get("thermal_rating_scale", 1.0),
    }

    # Build capex config from CLI args (or fall back to params.json if present)
    capex_from_cfg = cfg.get("capex", {})
    capex_cfg = {
        "wind_per_mw":          capex_from_cfg.get("wind_per_mw",          args.wind_per_mw),
        "solar_per_mw":         capex_from_cfg.get("solar_per_mw",         args.solar_per_mw),
        "battery_per_mw":       capex_from_cfg.get("battery_per_mw",       args.battery_per_mw),
        "wind_om_per_mw_yr":    capex_from_cfg.get("wind_om_per_mw_yr",    args.wind_om_per_mw_yr),
        "solar_om_per_mw_yr":   capex_from_cfg.get("solar_om_per_mw_yr",   args.solar_om_per_mw_yr),
        "battery_om_per_mw_yr": capex_from_cfg.get("battery_om_per_mw_yr", args.battery_om_per_mw_yr),
        "discount_rate":        capex_from_cfg.get("discount_rate",        args.discount_rate),
        "wind_life_yr":         capex_from_cfg.get("wind_life_yr",         args.wind_life_yr),
        "solar_life_yr":        capex_from_cfg.get("solar_life_yr",        args.solar_life_yr),
        "battery_life_yr":      capex_from_cfg.get("battery_life_yr",      args.battery_life_yr),
        "battery_capacity_value_usd_per_mw_yr": capex_from_cfg.get(
            "battery_capacity_value_usd_per_mw_yr", args.battery_capacity_value
        ),
    }

    run_optimisation(
        source_grid   = BASE_GRID,
        dc_grid       = DC_GRID,
        buses         = DC_BUSES,
        gen_csv       = gen_csv,
        sim_date      = SIM_DATE,
        sim_days      = SIM_DAYS,
        sim_params    = sim_params,
        out_dir       = out_dir,
        baseline_dir  = baseline_dir,
        bounds        = {
            "wind_max":    args.wind_max,
            "solar_max":   args.solar_max,
            "battery_max": args.battery_max,
        },
        weights       = {
            "lambda_carbon":      args.lambda_carbon,
            "lambda_cost":        args.lambda_cost,
            "lambda_reliability": args.lambda_reliability,
            "lambda_water":       args.lambda_water,
        },
        reliability_penalty  = cfg.get("renew_opt", {}).get("reliability_penalty",  args.reliability_penalty),
        curtailment_penalty  = cfg.get("renew_opt", {}).get("curtailment_penalty",  args.curtailment_penalty),
        n_workers     = args.workers,
        stage_n       = (args.stage1_n, args.stage2_n, args.stage3_n),
        seed          = args.seed,
        capex_cfg     = capex_cfg,
    )


if __name__ == "__main__":
    main()
