#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
plot_cost_carbon.py — Lifetime cost vs. total carbon emissions (no-CAS).

Loads TX_PARETO/results_nocas JSON files and plots:
  X: Lifetime CO₂ emissions (Mt CO₂, 25-year annualised from 8 representative weeks)
  Y: Lifetime total cost ($B, opex + annualised CAPEX × 25 years)

Usage:
    python scripts/plot_cost_carbon.py [--out <path>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUTS    = Path("/scratch/gpfs/SIRCAR/dp4800/vatic/outputs")
RESULTS_DIR = OUTPUTS / "TX_PARETO" / "results_nocas"

SIM_DAYS    = 8 * 7    # 8 representative weeks × 7 days
NUM_WEEKS   = 8        # weeks in multi_week_dates
LIFETIME_YR = 25       # wind / solar plant lifetime

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

GROUP_CFG = {
    "baseline":  dict(color="#555555", marker="*",  label="Baseline (no new capacity)", s=260, zorder=6),
    "wind_only": dict(color="#1E88E5", marker="^",  label="Wind only",  s=90,  zorder=4),
    "solar_only":dict(color="#FB8C00", marker="s",  label="Solar only", s=90,  zorder=4),
}


def load_results(d: Path) -> list[dict]:
    rows = []
    for f in sorted(d.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if r.get("status") == "ok":
            rows.append(r)
    return rows


def pareto_front(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Boolean mask — non-dominated under minimisation of both axes."""
    n = len(xs)
    dom = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and xs[j] <= xs[i] and ys[j] <= ys[i] and (xs[j] < xs[i] or ys[j] < ys[i]):
                dom[i] = True
                break
    return ~dom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out", type=Path, default=OUTPUTS / "cost_vs_carbon_nocas.png")
    args = ap.parse_args()

    results = load_results(args.results_dir)
    if not results:
        sys.exit(f"No valid results in {args.results_dir}")
    print(f"Loaded {len(results)} portfolios")

    ann  = 365 / SIM_DAYS          # sim-period → annual scale
    life = ann * LIFETIME_YR       # sim-period → 25-year lifetime scale

    # Existing results have a CAPEX bug: _annualized_capex was called with
    # sim_days=7 (one week) but opex metrics cover 8 weeks.  Correct here by
    # adding the 7 missing weeks of CAPEX back before scaling to lifetime.
    opex_arr  = np.array([r["metrics"]["total_cost_usd"] for r in results])
    capex_1wk = np.array([r["capex_usd"]                for r in results])
    capex_arr = capex_1wk * NUM_WEEKS   # corrected to full 8-week window

    co2_mt  = np.array([r["metrics"]["total_co2_tonnes"] for r in results]) * life / 1e6
    cost_b  = (opex_arr + capex_arr) * life / 1e9   # corrected total
    capex_b = capex_arr * life / 1e9
    wind_gw = np.array([r["investment"].get("wind_mw",  0) / 1e3 for r in results])
    sol_gw  = np.array([r["investment"].get("solar_mw", 0) / 1e3 for r in results])
    groups  = [r.get("portfolio_group", "other") for r in results]

    # Pareto front
    pm   = pareto_front(co2_mt, cost_b)
    fidx = np.where(pm)[0][np.argsort(co2_mt[pm])]   # left → right

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(right=0.82)

    for grp, cfg in GROUP_CFG.items():
        idx = [i for i, g in enumerate(groups) if g == grp]
        if not idx:
            continue
        xg = co2_mt[idx]; yg = cost_b[idx]; pg = pm[idx]
        ax.scatter(xg[~pg], yg[~pg],
                   color=cfg["color"], marker=cfg["marker"],
                   s=cfg["s"], alpha=0.70, linewidths=0.5,
                   edgecolors="white", zorder=cfg["zorder"],
                   label=cfg["label"])
        if pg.any():
            ax.scatter(xg[pg], yg[pg],
                       color=cfg["color"], marker=cfg["marker"],
                       s=cfg["s"] * 1.8, alpha=1.0,
                       linewidths=1.5, edgecolors="black",
                       zorder=cfg["zorder"] + 2)

    # Pareto frontier line
    ax.plot(co2_mt[fidx], cost_b[fidx],
            color="#B71C1C", lw=2.0, ls="--",
            zorder=5, label="Pareto frontier")

    # ── Annotations ───────────────────────────────────────────────────────────
    xspan = co2_mt.max() - co2_mt.min()
    yspan = cost_b.max() - cost_b.min()

    for pi in fidx:
        r   = results[pi]
        grp = r.get("portfolio_group", "")
        inv = r["investment"]
        w, s = inv.get("wind_mw", 0), inv.get("solar_mw", 0)

        if grp == "baseline":
            lbl = "Baseline"
            off = (xspan * 0.015, yspan * 0.025)
        elif grp == "wind_only":
            lbl = f"W {w/1e3:.1f} GW"
            off = (xspan * 0.01, -yspan * 0.04)
        else:
            lbl = f"S {s/1e3:.1f} GW"
            off = (-xspan * 0.01, yspan * 0.03)

        ax.annotate(
            lbl,
            xy=(co2_mt[pi], cost_b[pi]),
            xytext=(co2_mt[pi] + off[0], cost_b[pi] + off[1]),
            fontsize=8, color="#222222",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#555555"),
        )

    # ── Installed-capacity colorbar (wind+solar GW) ────────────────────────────
    total_gw = wind_gw + sol_gw
    sc = ax.scatter(co2_mt, cost_b,
                    c=total_gw, cmap="YlGn",
                    s=0,   # invisible markers; only drives colorbar
                    vmin=0, vmax=total_gw.max())
    cax = fig.add_axes([0.84, 0.15, 0.025, 0.65])
    cb  = fig.colorbar(sc, cax=cax)
    cb.set_label("New capacity (GW)", fontsize=9)

    # ── Axes & labels ─────────────────────────────────────────────────────────
    ax.set_xlabel("Lifetime CO₂ emissions  (Mt CO₂,  25-yr horizon)", fontsize=12)
    ax.set_ylabel("Lifetime total cost  ($B,  25-yr horizon)\n"
                  "grid opex + annualised CAPEX", fontsize=11)
    ax.set_title(
        "Texas 7k  —  No-CAS: Lifetime Cost vs. Carbon Emissions\n"
        f"({len(results)} portfolios, Pareto-optimal highlighted)",
        fontsize=11, fontweight="bold",
    )

    handles = [
        mpatches.Patch(color=cfg["color"], label=cfg["label"])
        for grp, cfg in GROUP_CFG.items() if any(g == grp for g in groups)
    ]
    handles.append(plt.Line2D([0], [0], color="#B71C1C", lw=2, ls="--",
                               label="Pareto frontier"))
    ax.legend(handles=handles, fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, lw=0.4, alpha=0.35)

    # ── Console table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"{'Group':<12} {'Portfolio':<14} "
          f"{'CO₂ Mt (25yr)':>14} {'Cost $B (25yr)':>14} {'CAPEX $B':>9}")
    print(f"{'─'*68}")
    for i in np.argsort(co2_mt):
        r   = results[i]
        inv = r["investment"]
        w   = inv.get("wind_mw",  0)
        s   = inv.get("solar_mw", 0)
        tag = "★" if pm[i] else " "
        lbl = f"W{w/1e3:.1f}+S{s/1e3:.1f} GW" if (w or s) else "Baseline"
        print(f"{tag}{r.get('portfolio_group','?'):<11} {lbl:<14} "
              f"{co2_mt[i]:>14.1f} {cost_b[i]:>14.1f} {capex_b[i]:>9.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {args.out}")
    print(f"Saved → {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
