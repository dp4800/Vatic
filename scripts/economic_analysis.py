#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""Economic and Capital Cost Analysis for TX_PARETO_SCALE portfolios.

Implements life-cycle investment analysis per the methodology:
  C_lifetime = CAPEX + (C_opex_annual_2030 + C_fom_annual) × annuity_factor(r, H)

Reads Vatic hourly FixedCosts + VariableCosts from both sims_inv (baseline
operation) and sims_cas (CAS-optimised operation) for each portfolio, inflates
from 2018$ to 2030$ via CPI, adds overnight capital costs and fixed O&M for
new renewables, and discounts over a 30-year planning horizon.

Usage:
    module load anaconda3/2024.10
    python scripts/economic_analysis.py [--study-dir PATH]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# Parameters — Tables AV, AIII, AVI
# ═══════════════════════════════════════════════════════════════════════════════

# Table AV: CPI inflation adjustment (2018 → 2030)
CPI_FACTOR_2018_TO_2030 = 1.440  # CPI-U 251.01 → 361.45

# Table AIII: Overnight Capital Cost (2030$/kW, ATB 2030 Conservative, R&D Only)
OCC_WIND_PER_KW = 1388.0       # Land-based wind
OCC_SOLAR_PER_KW = 1180.0      # Utility-scale PV
OCC_BATTERY_PER_KW = 1612.0    # 4-hour Li-ion battery storage

# Table AIII: Fixed O&M (2030$/kW-yr, ATB 2030 Conservative)
FOM_WIND_PER_KW_YR = 30.91     # Land-based wind
FOM_SOLAR_PER_KW_YR = 19.73    # Utility-scale PV
FOM_BATTERY_PER_KW_YR = 40.31  # 4-hour Li-ion battery storage

# Table AVI: Capacity-weighted average real after-tax WACC (ATB 2030 Conservative)
# Weighted by installed capacity in the Texas-7k baseline grid (104,914 MW):
#   Coal 5.4%, NG 5.4%, Nuclear 4.7%, Biopower 4.7%, Wind 3.7%,
#   Solar 3.4%, Hydro 3.8%, Battery 3.4%
# → capacity-weighted average = 4.9%
DISCOUNT_RATE = 0.049

# Planning horizon (ATB cost recovery period)
HORIZON_YEARS = 30

# Simulation structure
SIM_WEEKS = ["2018-01-07", "2018-07-15"]
SIM_DAYS_PER_WEEK = 7
TOTAL_SIM_DAYS = len(SIM_WEEKS) * SIM_DAYS_PER_WEEK  # 14


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def annuity_factor(r: float, h: int) -> float:
    """Present value annuity factor: (1 - (1+r)^{-h}) / r."""
    if r == 0:
        return float(h)
    return (1.0 - (1.0 + r) ** (-h)) / r


def load_portfolios(results_dir: Path) -> list[dict]:
    """Load all pareto JSON result files, sorted by portfolio_id."""
    portfolios = []
    for fp in sorted(results_dir.glob("pareto_*.json")):
        with open(fp) as f:
            portfolios.append(json.load(f))
    portfolios.sort(key=lambda d: d["portfolio_id"])
    return portfolios


def compute_opex_from_hourly(sim_base: Path, portfolio_hash: str) -> dict:
    """Read hourly_summary.csv for both weeks and compute OPEX metrics.

    Returns dict with:
        opex_annual_2018  – annual OPEX extrapolated from sim days (2018$)
        opex_annual_2030  – inflated to 2030$
        daily_opex        – list of daily totals (2018$)
        avg_lmp           – demand-weighted average LMP across all hours
    """
    daily_opex_list = []
    total_price_x_demand = 0.0
    total_demand = 0.0

    for week in SIM_WEEKS:
        csv_path = sim_base / portfolio_hash / week / "hourly_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing: {csv_path}")

        df = pd.read_csv(csv_path)
        df["OPEX"] = df["FixedCosts"] + df["VariableCosts"]

        # Daily aggregation
        daily = df.groupby("Date")["OPEX"].sum()
        daily_opex_list.extend(daily.values.tolist())

        # Demand-weighted LMP
        total_price_x_demand += (df["Price"] * df["Demand"]).sum()
        total_demand += df["Demand"].sum()

    avg_daily_opex = np.mean(daily_opex_list)
    opex_annual_2018 = avg_daily_opex * 365.0
    opex_annual_2030 = opex_annual_2018 * CPI_FACTOR_2018_TO_2030

    avg_lmp = total_price_x_demand / total_demand if total_demand > 0 else 0.0

    return {
        "opex_annual_2018": opex_annual_2018,
        "opex_annual_2030": opex_annual_2030,
        "daily_opex": daily_opex_list,
        "avg_lmp": avg_lmp,
    }


def compute_capex(wind_mw: float, solar_mw: float,
                  battery_mw: float) -> dict:
    """Compute overnight capital cost (one-time, 2030$) per technology."""
    capex_wind = OCC_WIND_PER_KW * wind_mw * 1000.0
    capex_solar = OCC_SOLAR_PER_KW * solar_mw * 1000.0
    capex_battery = OCC_BATTERY_PER_KW * battery_mw * 1000.0
    return {
        "capex_wind": capex_wind,
        "capex_solar": capex_solar,
        "capex_battery": capex_battery,
        "capex_total": capex_wind + capex_solar + capex_battery,
    }


def compute_fom_annual(wind_mw: float, solar_mw: float,
                       battery_mw: float) -> float:
    """Annual fixed O&M for new renewables (2030$/yr)."""
    return (FOM_WIND_PER_KW_YR * wind_mw * 1000.0
            + FOM_SOLAR_PER_KW_YR * solar_mw * 1000.0
            + FOM_BATTERY_PER_KW_YR * battery_mw * 1000.0)


def compute_lifetime_cost(capex_total: float, opex_annual_2030: float,
                          fom_annual: float, r: float, h: int) -> float:
    """C_lifetime = CAPEX + (OPEX_annual + FOM_annual) × annuity_factor."""
    af = annuity_factor(r, h)
    return capex_total + (opex_annual_2030 + fom_annual) * af


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Life-cycle economic analysis for TX_PARETO_SCALE")
    parser.add_argument(
        "--study-dir",
        default="outputs/TX_PARETO_SCALE",
        help="Path to TX_PARETO_SCALE output directory")
    args = parser.parse_args()

    study = Path(args.study_dir)
    results_dir = study / "results"
    sims_inv_base = study / "sims_inv" / "7d"
    sims_cas_base = study / "sims_cas" / "7d"

    # Load portfolio definitions
    portfolios = load_portfolios(results_dir)
    print(f"Loaded {len(portfolios)} portfolios from {results_dir}\n")

    rows = []
    for p in portfolios:
        pid = p["portfolio_id"]
        h = p["hash"]
        group = p.get("portfolio_group", "")
        wind_mw = p.get("added_wind_mw", 0.0)
        solar_mw = p.get("added_solar_mw", 0.0)
        battery_mw = p.get("added_battery_mw", 0.0)

        # CAPEX (one-time, 2030$)
        cx = compute_capex(wind_mw, solar_mw, battery_mw)

        # Fixed O&M (annual, 2030$)
        fom = compute_fom_annual(wind_mw, solar_mw, battery_mw)

        # OPEX from sims_inv (baseline operation)
        inv = compute_opex_from_hourly(sims_inv_base, h)

        # OPEX from sims_cas (CAS-optimised operation)
        cas = compute_opex_from_hourly(sims_cas_base, h)

        # Lifetime costs
        lt_inv = compute_lifetime_cost(
            cx["capex_total"], inv["opex_annual_2030"], fom,
            DISCOUNT_RATE, HORIZON_YEARS)
        lt_cas = compute_lifetime_cost(
            cx["capex_total"], cas["opex_annual_2030"], fom,
            DISCOUNT_RATE, HORIZON_YEARS)

        rows.append({
            "portfolio_id": pid,
            "hash": h,
            "group": group,
            "added_wind_mw": wind_mw,
            "added_solar_mw": solar_mw,
            "added_battery_mw": battery_mw,
            "opex_annual_2018_inv": inv["opex_annual_2018"],
            "opex_annual_2030_inv": inv["opex_annual_2030"],
            "opex_annual_2018_cas": cas["opex_annual_2018"],
            "opex_annual_2030_cas": cas["opex_annual_2030"],
            "fom_annual_2030": fom,
            "capex_2030": cx["capex_total"],
            "discount_rate": DISCOUNT_RATE,
            "annuity_factor": annuity_factor(DISCOUNT_RATE, HORIZON_YEARS),
            "lifetime_cost_inv": lt_inv,
            "lifetime_cost_cas": lt_cas,
            "avg_lmp_inv": inv["avg_lmp"],
            "avg_lmp_cas": cas["avg_lmp"],
        })

    df = pd.DataFrame(rows)

    # Compute deltas vs baseline (portfolio_id == 0)
    bl = df.loc[df["portfolio_id"] == 0].iloc[0]
    for mode in ("inv", "cas"):
        bl_lt = bl[f"lifetime_cost_{mode}"]
        df[f"delta_lifetime_{mode}_usd"] = df[f"lifetime_cost_{mode}"] - bl_lt
        df[f"delta_lifetime_{mode}_pct"] = (
            (df[f"lifetime_cost_{mode}"] - bl_lt) / bl_lt * 100.0
        )

    # Save CSV
    out_csv = study / "economic_analysis.csv"
    df.to_csv(out_csv, index=False, float_format="%.2f")
    print(f"Saved: {out_csv}\n")

    # Print summary
    print("=" * 120)
    print(f"{'ID':>3}  {'Group':<18}  {'Wind':>8}  {'Solar':>8}  {'Batt':>8}"
          f"  {'CAPEX($B)':>10}  {'OPEX_inv($B/yr)':>16}  {'LT_inv($B)':>11}"
          f"  {'LT_cas($B)':>11}  {'Δ_inv%':>7}  {'Δ_cas%':>7}"
          f"  {'LMP_inv':>8}  {'LMP_cas':>8}")
    print("-" * 120)

    for _, r in df.iterrows():
        print(f"{r['portfolio_id']:3.0f}  {r['group']:<18}"
              f"  {r['added_wind_mw']:8.0f}  {r['added_solar_mw']:8.0f}"
              f"  {r['added_battery_mw']:8.0f}"
              f"  {r['capex_2030']/1e9:10.3f}"
              f"  {r['opex_annual_2030_inv']/1e9:16.3f}"
              f"  {r['lifetime_cost_inv']/1e9:11.3f}"
              f"  {r['lifetime_cost_cas']/1e9:11.3f}"
              f"  {r['delta_lifetime_inv_pct']:7.1f}"
              f"  {r['delta_lifetime_cas_pct']:7.1f}"
              f"  {r['avg_lmp_inv']:8.2f}"
              f"  {r['avg_lmp_cas']:8.2f}")

    print("=" * 120)
    print(f"\nParameters: CPI ×{CPI_FACTOR_2018_TO_2030}, H={HORIZON_YEARS}yr,"
          f" OCC($/kW): wind={OCC_WIND_PER_KW}, solar={OCC_SOLAR_PER_KW},"
          f" battery={OCC_BATTERY_PER_KW}")
    print(f"FOM($/kW-yr): wind={FOM_WIND_PER_KW_YR},"
          f" solar={FOM_SOLAR_PER_KW_YR}, battery={FOM_BATTERY_PER_KW_YR}")
    print(f"Discount rate: {DISCOUNT_RATE:.1%} (capacity-weighted WACC)")


if __name__ == "__main__":
    main()
