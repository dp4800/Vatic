#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
reliability.py — LOLP, LOLE, and EUE reliability metrics.

Definitions
-----------
LOLP  Loss of Load Probability  : fraction of hours with any load shedding
LOLE  Loss of Load Expectation  : expected hours of shedding per simulation period
EUE   Expected Unserved Energy  : total MWh of unmet demand (= load_shedding_mwh)

These are the industry-standard IEEE/NERC reliability indices.  LOLP and LOLE
are closely related (LOLE = LOLP * total_hours) but both are reported because
LOLE in hours/year is the most common planning standard (often 0.1 h/yr = 1 day
in 10 years).

Usage
-----
    import reliability
    r = reliability.compute_reliability(sim_dir)
    # r = {'lolp': 0.0012, 'lole_h': 2, 'eue_mwh': 45.3, 'total_h': 168}
"""

from pathlib import Path

import pandas as pd


def compute_reliability(sim_dir: Path) -> dict:
    """Return LOLP, LOLE, and EUE metrics from a completed VATIC simulation.

    Parameters
    ----------
    sim_dir : Path
        Directory containing hourly_summary.csv from a vatic-det run.

    Returns
    -------
    dict with keys:
        lolp      float  Loss of Load Probability (fraction, 0–1)
        lole_h    int    Loss of Load Expectation (hours in this period)
        eue_mwh   float  Expected Unserved Energy (MWh)
        total_h   int    Total simulated hours
    """
    csv = sim_dir / "hourly_summary.csv"
    if not csv.exists():
        return {"lolp": float("nan"), "lole_h": 0, "eue_mwh": 0.0, "total_h": 0}

    hourly = pd.read_csv(csv)
    shed = hourly["LoadShedding"].clip(lower=0)
    n_hours = len(hourly)
    shed_hours = int((shed > 0).sum())

    return {
        "lolp":    round(shed_hours / n_hours, 6) if n_hours > 0 else 0.0,
        "lole_h":  shed_hours,
        "eue_mwh": round(float(shed.sum()), 2),
        "total_h": n_hours,
    }


def annualize_reliability(r: dict, days_in_month: int, sim_days: int) -> dict:
    """Scale per-simulation reliability metrics to a full calendar month.

    LOLP is a probability — it does not scale with time, so it is returned
    as-is (it already represents the hourly average for the simulated period).
    LOLE and EUE are scaled by (days_in_month / sim_days).

    Parameters
    ----------
    r : dict
        Output of compute_reliability().
    days_in_month : int
        Calendar days in the month being scaled to.
    sim_days : int
        Length of the simulation window.

    Returns
    -------
    dict with the same keys as compute_reliability().
    """
    scale = days_in_month / sim_days if sim_days > 0 else 1.0
    return {
        "lolp":    r["lolp"],
        "lole_h":  round(r["lole_h"] * scale, 2),
        "eue_mwh": round(r["eue_mwh"] * scale, 2),
        "total_h": round(r["total_h"] * scale),
    }


if __name__ == "__main__":
    import argparse, json, sys

    p = argparse.ArgumentParser(description="Compute reliability metrics for a sim dir.")
    p.add_argument("sim_dir", type=Path, help="Path to a vatic-det output directory")
    args = p.parse_args()

    r = compute_reliability(args.sim_dir)
    print(json.dumps(r, indent=2))
    print(f"\nLOLP  = {r['lolp']:.4%}  ({r['lole_h']} / {r['total_h']} hours)")
    print(f"LOLE  = {r['lole_h']} h/period")
    print(f"EUE   = {r['eue_mwh']:.1f} MWh")
