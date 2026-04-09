#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""Unified Pareto analysis for TX renewable investment studies.

Thesis: Figure 16 — "Pareto frontier with CAS and without LP CAS, from
    various configurations of wind, solar, and battery storage."

Supports both TX_PARETO (8-week sweep) and TX_PARETO_SCALE (2-week scaling)
studies.  Generates one image file per figure.

Figures:
  pareto_cost_carbon.png      Annual CO2 vs annual total cost
  pareto_winwin.png           Win-win landscape (% change from baseline)
  pareto_efficiency.png       Investment efficiency frontier
  pareto_breakdown.png        Efficiency-front portfolio breakdown (stacked bar)
  pareto_lifetime_{mode}.png  Lifetime cost vs CO2  (needs economic_analysis.csv)
  pareto_combined.png         CAS vs no-CAS overlay (needs dual result sets)

Usage:
    module load anaconda3/2024.10
    python scripts/pareto_analysis.py --study-dir outputs/TX_PARETO_SCALE --sim-days 14
    python scripts/pareto_analysis.py --study-dir outputs/TX_PARETO --sim-days 56
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# =============================================================================
# Style
# =============================================================================

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
})

# Base technology group -> visual style.
# Colors aligned with tx_datacenter_map.py generator palette.
GROUP_PALETTE: dict[str, dict] = {
    "baseline":      dict(color="#333333", marker="*",  label="Baseline",         s=180),
    "wind_only":     dict(color="#16a34a", marker="^",  label="Wind only",        s=80),
    "solar_only":    dict(color="#eab308", marker="v",  label="Solar only",       s=80),
    "battery_only":  dict(color="#7c3aed", marker="D",  label="Battery only",     s=80),
    "wind_solar":    dict(color="#0d9488", marker="o",  label="Wind + Solar",     s=90),
    "equal_ws":      dict(color="#0d9488", marker="o",  label="Equal W+S",        s=90),
    "wind_battery":  dict(color="#2563eb", marker=">",  label="Wind + Battery",   s=80),
    "solar_battery": dict(color="#ea580c", marker="<",  label="Solar + Battery",  s=80),
    "joint":         dict(color="#e11d48", marker="p",  label="W+S+Battery",      s=90),
    "balanced":      dict(color="#e11d48", marker="p",  label="Balanced W+S+B",   s=90),
    "wind_heavy":    dict(color="#059669", marker="H",  label="Wind-heavy W+S",   s=80),
    "solar_heavy":   dict(color="#d97706", marker="H",  label="Solar-heavy W+S",  s=80),
}

# L/M/H scale tiers: (size_multiplier, alpha)
SCALE_TIER = {"L": (0.7, 0.55), "M": (1.0, 0.75), "H": (1.4, 1.0)}

# Technology colors for stacked bars
TECH_COLORS = {"wind": "#16a34a", "solar": "#eab308", "battery": "#7c3aed"}


# =============================================================================
# Group helpers
# =============================================================================

def _group_base(group: str) -> str:
    """Strip L/M/H suffix to get the base technology group."""
    if group == "baseline":
        return "baseline"
    for suffix in ("_L", "_M", "_H"):
        if group.endswith(suffix):
            return group[: -len(suffix)]
    return group


def _group_tier(group: str) -> str:
    """Get L/M/H tier; unsuffixed groups default to M."""
    for suffix in ("_L", "_M", "_H"):
        if group.endswith(suffix):
            return suffix[-1]
    return "M"


def _get_style(group: str) -> dict:
    base = _group_base(group)
    tier = _group_tier(group)
    pal = GROUP_PALETTE.get(
        base, dict(color="#888888", marker="o", label=base, s=70)
    )
    s_mult, alpha = SCALE_TIER.get(tier, (1.0, 0.75))
    return {**pal, "s": pal["s"] * s_mult, "alpha": alpha}


# =============================================================================
# Data loading
# =============================================================================

def load_json_results(results_dir: Path) -> pd.DataFrame:
    """Load pareto_*.json results into a DataFrame."""
    rows: list[dict] = []
    for fp in sorted(results_dir.glob("pareto_*.json")):
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        if d.get("status") != "ok":
            continue
        inv = d.get("investment", {})
        rows.append(dict(
            portfolio_id=d.get("portfolio_id", 0),
            hash=d["hash"],
            group=d.get("portfolio_group", "other"),
            wind_mw=d.get("added_wind_mw", inv.get("wind_mw", 0)),
            solar_mw=d.get("added_solar_mw", inv.get("solar_mw", 0)),
            battery_mw=d.get("added_battery_mw", inv.get("battery_mw", 0)),
            co2_tonnes=d["metrics"]["total_co2_tonnes"],
            opex_usd=d["metrics"]["total_cost_usd"],
            capex_usd=d.get("capex_usd", 0),
            total_cost_usd=d.get("total_cost_with_capex_usd", 0),
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["total_mw"] = df["wind_mw"] + df["solar_mw"] + df["battery_mw"]
        df["group_base"] = df["group"].apply(_group_base)
    return df


def merge_phase_results(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Merge P1 (winter+summer) and P2 (spring+fall) results by hash.

    Sums operational metrics (opex, CO2) across phases;
    keeps capex once (same investment, different sim weeks).
    Only includes portfolios present in both phases (inner join).
    """
    SUM_COLS = ["co2_tonnes", "opex_usd"]
    KEEP_COLS = ["portfolio_id", "hash", "group", "wind_mw", "solar_mw",
                 "battery_mw", "capex_usd"]

    common = set(df1["hash"]) & set(df2["hash"])
    d1 = df1[df1["hash"].isin(common)].copy()
    d2 = df2[df2["hash"].isin(common)].set_index("hash")

    merged = d1[KEEP_COLS].copy()
    for col in SUM_COLS:
        merged[col] = d1[col].values + d2.loc[d1["hash"].values, col].values

    merged["total_cost_usd"] = merged["opex_usd"] + merged["capex_usd"]
    merged["total_mw"] = merged["wind_mw"] + merged["solar_mw"] + merged["battery_mw"]
    merged["group_base"] = merged["group"].apply(_group_base)
    return merged


# =============================================================================
# Pareto utilities
# =============================================================================

def find_pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of non-dominated points (minimise both), sorted by ascending x."""
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if (x[j] <= x[i] and y[j] <= y[i]
                    and (x[j] < x[i] or y[j] < y[i])):
                is_pareto[i] = False
                break
    idx = np.where(is_pareto)[0]
    return idx[np.argsort(x[idx])]


def _make_label(row) -> str:
    parts = []
    w = row.get("wind_mw") or row.get("added_wind_mw") or 0
    s = row.get("solar_mw") or row.get("added_solar_mw") or 0
    b = row.get("battery_mw") or row.get("added_battery_mw") or 0
    if w > 0:
        parts.append(f"W{w / 1e3:.1f}k")
    if s > 0:
        parts.append(f"S{s / 1e3:.1f}k")
    if b > 0:
        parts.append(f"B{b:.0f}")
    return "+".join(parts) if parts else "Baseline"


# =============================================================================
# Shared plot helpers
# =============================================================================

def _scatter_by_group(ax, df, x_col, y_col, *,
                      pareto_idx=None, size_by_mw=False):
    """Scatter points coloured by group; Pareto points get black edge."""
    pareto_set = set(pareto_idx) if pareto_idx is not None else set()
    plotted: set[str] = set()

    for i, (_, row) in enumerate(df.iterrows()):
        sty = _get_style(row["group"])
        base = row.get("group_base", _group_base(row["group"]))

        label = None
        if base not in plotted:
            label = sty["label"]
            plotted.add(base)

        sz = sty["s"]
        if size_by_mw and row.get("total_mw", 0) > 0:
            sz = np.clip(row["total_mw"] / 150, 30, 280)

        on_front = i in pareto_set
        ax.scatter(
            row[x_col], row[y_col],
            marker=sty["marker"], c=sty["color"],
            s=sz * (1.6 if on_front else 1.0),
            alpha=1.0 if on_front else sty["alpha"],
            edgecolors="black" if on_front else "white",
            linewidths=1.2 if on_front else 0.4,
            label=label, zorder=6 if on_front else 3,
        )


def _draw_front(ax, x, y, idx, **kw):
    defaults = dict(color="#9f1239", linewidth=1.8, linestyle="--",
                    alpha=0.7, zorder=5, label="Pareto front")
    defaults.update(kw)
    ax.plot(x[idx], y[idx], **defaults)


def _shade_dominated(ax, x, y, idx, color="#9f1239"):
    px, py = x[idx], y[idx]
    ymax = ax.get_ylim()[1]
    x_ext = np.concatenate([[px[0]], px, [px[-1]]])
    y_ext = np.concatenate([[ymax], py, [ymax]])
    ax.fill(x_ext, y_ext, alpha=0.04, color=color, zorder=0)


def _baseline_crosshairs(ax, bx, by):
    ax.axhline(by, color="#888888", lw=0.7, ls=":", alpha=0.5, zorder=1)
    ax.axvline(bx, color="#888888", lw=0.7, ls=":", alpha=0.5, zorder=1)


def _annotate_front(ax, df, xcol, ycol, idx, *, use_pid=False, max_n=None):
    sel = idx[:max_n] if max_n else idx
    for i in sel:
        row = df.iloc[i]
        txt = f"P{int(row['portfolio_id'])}" if use_pid else _make_label(row)
        ax.annotate(txt, (row[xcol], row[ycol]),
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=7, alpha=0.85, color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))


def _quadrant_labels(ax, x_vals=None, y_vals=None):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    kw = dict(fontsize=8, alpha=0.4, ha="center", va="center",
              fontweight="bold")
    mx, my = (xlim[0]) / 2, (ylim[0]) / 2
    px, py = xlim[1] / 2, ylim[1] / 2

    defs = [
        (mx, my, "Win-Win\nCO\u2082 + Cost", "#16a34a",
         lambda x, y: (x < 0) & (y < 0)),
        (mx, py, "CO\u2082 Win\nCost Trade-off", "#9f1239",
         lambda x, y: (x < 0) & (y >= 0)),
        (px, my, "Cost Win\nCO\u2082 Trade-off", "#888888",
         lambda x, y: (x >= 0) & (y < 0)),
        (px, py, "Lose-Lose", "#dc2626",
         lambda x, y: (x >= 0) & (y >= 0)),
    ]
    for xp, yp, text, color, mask_fn in defs:
        extra = ""
        if x_vals is not None and y_vals is not None:
            n = int(np.sum(mask_fn(x_vals, y_vals)))
            extra = f"\n({n})"
        ax.text(xp, yp, text + extra, color=color, **kw)


def _group_legend(df, extra_handles=None):
    seen: set[str] = set()
    handles = []
    for base in df["group_base"].unique():
        if base not in GROUP_PALETTE or base in seen:
            continue
        seen.add(base)
        sty = GROUP_PALETTE[base]
        handles.append(mpatches.Patch(color=sty["color"], label=sty["label"]))
    if extra_handles:
        handles.extend(extra_handles)
    return handles


def _save(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Figure 1: Annual Cost vs. Carbon
# =============================================================================

def fig_cost_carbon(df: pd.DataFrame, sim_days: int, out_dir: Path):
    scale = 365 / sim_days
    df = df.copy()
    df["co2_mt"] = df["co2_tonnes"] * scale / 1e6
    df["cost_b"] = df["total_cost_usd"] * scale / 1e9

    x, y = df["co2_mt"].values, df["cost_b"].values
    pidx = find_pareto_front(x, y)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    bl = df[df["group_base"] == "baseline"]
    if not bl.empty:
        _baseline_crosshairs(ax, bl["co2_mt"].iloc[0], bl["cost_b"].iloc[0])

    _scatter_by_group(ax, df, "co2_mt", "cost_b",
                      pareto_idx=pidx, size_by_mw=True)
    _draw_front(ax, x, y, pidx)
    _shade_dominated(ax, x, y, pidx)
    _annotate_front(ax, df, "co2_mt", "cost_b", pidx)

    ax.set_xlabel("Annual CO\u2082 Emissions (Mt/yr)", fontsize=12)
    ax.set_ylabel("Annual Total Cost ($B/yr)\n"
                   "(grid opex + annualised CAPEX)", fontsize=11)
    ax.set_title("Annual Cost vs. Carbon Emissions\n"
                 f"({len(df)} portfolios, annualised from "
                 f"{sim_days}-day simulation)",
                 fontsize=12, fontweight="bold")

    extra = [Line2D([0], [0], color="#9f1239", lw=1.8, ls="--",
                    label="Pareto front")]
    ax.legend(handles=_group_legend(df, extra),
              fontsize=8.5, ncol=2, loc="upper right", framealpha=0.9)

    _save(fig, out_dir / "pareto_cost_carbon.png")


# =============================================================================
# Figure 2: Win-Win Landscape
# =============================================================================

def fig_winwin(df: pd.DataFrame, out_dir: Path):
    bl = df[df["group_base"] == "baseline"]
    if bl.empty:
        print("  Skipping win-win: no baseline found")
        return

    bl_co2 = bl["co2_tonnes"].iloc[0]
    bl_cost = bl["total_cost_usd"].iloc[0]

    df = df.copy()
    df["co2_pct"] = (df["co2_tonnes"] - bl_co2) / bl_co2 * 100
    df["cost_pct"] = (df["total_cost_usd"] - bl_cost) / bl_cost * 100

    x, y = df["co2_pct"].values, df["cost_pct"].values
    pidx = find_pareto_front(x, y)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    _scatter_by_group(ax, df, "co2_pct", "cost_pct",
                      pareto_idx=pidx, size_by_mw=True)
    _draw_front(ax, x, y, pidx)
    _annotate_front(ax, df, "co2_pct", "cost_pct", pidx, max_n=5)

    ax.axhline(0, color="gray", lw=0.7, ls=":")
    ax.axvline(0, color="gray", lw=0.7, ls=":")
    _quadrant_labels(ax, x_vals=x, y_vals=y)

    ax.set_xlabel("CO\u2082 Change vs. Baseline (%)", fontsize=12)
    ax.set_ylabel("Total Cost Change vs. Baseline (%)", fontsize=11)
    ax.set_title("Win-Win Landscape\n"
                 f"({len(df)} portfolios)", fontsize=12, fontweight="bold")

    extra = [Line2D([0], [0], color="#9f1239", lw=1.8, ls="--",
                    label="Pareto front")]
    ax.legend(handles=_group_legend(df, extra),
              fontsize=8.5, loc="upper left", framealpha=0.9)

    _save(fig, out_dir / "pareto_winwin.png")


# =============================================================================
# Figure 3: Investment Efficiency Frontier
# =============================================================================

def fig_efficiency(df: pd.DataFrame, sim_days: int, out_dir: Path):
    bl = df[df["group_base"] == "baseline"]
    if bl.empty:
        print("  Skipping efficiency: no baseline found")
        return None

    bl_co2 = bl["co2_tonnes"].iloc[0]
    scale = 365 / sim_days

    df = df.copy()
    df["co2_red_pct"] = -(df["co2_tonnes"] - bl_co2) / bl_co2 * 100
    df["capex_b"] = df["capex_usd"] * scale / 1e9

    # Pareto: maximise reduction, minimise CAPEX
    neg_red = -df["co2_red_pct"].values
    capex = df["capex_b"].values
    pidx = find_pareto_front(neg_red, capex)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    _scatter_by_group(ax, df, "co2_red_pct", "capex_b",
                      pareto_idx=pidx, size_by_mw=True)

    # Frontier line sorted left-to-right on positive reduction axis
    fx = df["co2_red_pct"].values[pidx]
    fy = capex[pidx]
    order = np.argsort(fx)
    ax.plot(fx[order], fy[order], color="#2563eb", lw=2.0, ls="--",
            zorder=5, alpha=0.8, marker="o", markersize=4,
            label="Efficiency frontier")

    _annotate_front(ax, df, "co2_red_pct", "capex_b", pidx, max_n=5)

    ax.axhline(0, color="gray", lw=0.7, ls=":")
    ax.axvline(0, color="gray", lw=0.7, ls=":")

    ax.set_xlabel("CO\u2082 Reduction vs. Baseline (%)", fontsize=12)
    ax.set_ylabel("Annualised CAPEX ($B/yr)", fontsize=11)
    ax.set_title("Investment Efficiency Frontier\n"
                 f"({len(df)} portfolios, {len(pidx)} on frontier)",
                 fontsize=12, fontweight="bold")

    extra = [Line2D([0], [0], color="#2563eb", lw=2.0, ls="--",
                    label="Efficiency frontier")]
    ax.legend(handles=_group_legend(df, extra),
              fontsize=8.5, loc="upper left", framealpha=0.9)

    _save(fig, out_dir / "pareto_efficiency.png")
    return pidx


# =============================================================================
# Figure 4: Efficiency-Front Portfolio Breakdown (stacked bar)
# =============================================================================

def fig_breakdown(df: pd.DataFrame, eff_idx: np.ndarray | None,
                  sim_days: int, out_dir: Path):
    bl = df[df["group_base"] == "baseline"]
    if bl.empty or eff_idx is None or len(eff_idx) == 0:
        print("  Skipping breakdown: no efficiency frontier computed")
        return

    bl_co2 = bl["co2_tonnes"].iloc[0]
    bl_cost = bl["total_cost_usd"].iloc[0]
    scale = 365 / sim_days

    df = df.copy()
    df["co2_red_pct"] = -(df["co2_tonnes"] - bl_co2) / bl_co2 * 100
    df["cost_pct"] = (df["total_cost_usd"] - bl_cost) / bl_cost * 100
    df["capex_b"] = df["capex_usd"] * scale / 1e9

    # Sort efficiency-front by CO2 reduction descending (best at top)
    eidx = eff_idx[np.argsort(-df["co2_red_pct"].values[eff_idx])]
    rows = df.iloc[eidx]

    wind_gw = rows["wind_mw"].values / 1e3
    solar_gw = rows["solar_mw"].values / 1e3
    bat_gw = rows["battery_mw"].values / 1e3
    co2_red = rows["co2_red_pct"].values
    cost_ch = rows["cost_pct"].values
    capex = rows["capex_b"].values

    ylabels = [f"{c:.1f}%\n{_make_label(r)}"
               for c, (_, r) in zip(co2_red, rows.iterrows())]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.7 * len(eidx) + 1.5)))

    ypos = np.arange(len(eidx))
    ax.barh(ypos, wind_gw, color=TECH_COLORS["wind"],
            height=0.6, label="Wind")
    ax.barh(ypos, solar_gw, color=TECH_COLORS["solar"],
            height=0.6, left=wind_gw, label="Solar")
    ax.barh(ypos, bat_gw, color=TECH_COLORS["battery"],
            height=0.6, left=wind_gw + solar_gw, label="Battery")

    for i in range(len(eidx)):
        total = wind_gw[i] + solar_gw[i] + bat_gw[i]
        ax.text(max(total, 0.1) + 0.15, ypos[i],
                f"cost {cost_ch[i]:+.1f}%  CAPEX ${capex[i]:.2f}B",
                va="center", fontsize=7.5, color="#333333")

    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_xlabel("Installed Capacity (GW)", fontsize=11)
    ax.set_title("Efficiency-Front Portfolios\n"
                 "(sorted by CO\u2082 reduction)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    _save(fig, out_dir / "pareto_breakdown.png")


# =============================================================================
# Figure 5: Lifetime Cost vs. CO2 (single mode)
# =============================================================================

def fig_lifetime(df_json: pd.DataFrame, econ: pd.DataFrame,
                 mode: str, sim_days: int, out_dir: Path):
    """mode = 'cas' or 'inv'."""
    lt_col = f"lifetime_cost_{mode}"
    if lt_col not in econ.columns:
        print(f"  Skipping lifetime ({mode}): column {lt_col} not found")
        return

    # Match JSON CO2 to econ lifetime cost by hash
    co2_map = dict(zip(df_json["hash"], df_json["co2_tonnes"]))
    df = econ.copy()
    df["co2_annual_mt"] = df["hash"].map(co2_map) / sim_days * 365 / 1e6
    df["lt_cost_b"] = df[lt_col] / 1e9
    df.dropna(subset=["co2_annual_mt", "lt_cost_b"], inplace=True)
    df["group_base"] = df["group"].apply(_group_base)
    df["total_mw"] = (df.get("added_wind_mw", df.get("wind_mw", 0))
                      + df.get("added_solar_mw", df.get("solar_mw", 0))
                      + df.get("added_battery_mw", df.get("battery_mw", 0)))

    x, y = df["co2_annual_mt"].values, df["lt_cost_b"].values
    pidx = find_pareto_front(x, y)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    bl = df[df["group_base"] == "baseline"]
    if not bl.empty:
        _baseline_crosshairs(ax, bl["co2_annual_mt"].iloc[0],
                             bl["lt_cost_b"].iloc[0])

    _scatter_by_group(ax, df, "co2_annual_mt", "lt_cost_b",
                      pareto_idx=pidx, size_by_mw=True)
    _draw_front(ax, x, y, pidx)
    _shade_dominated(ax, x, y, pidx)
    _annotate_front(ax, df, "co2_annual_mt", "lt_cost_b", pidx, use_pid=True)

    mode_lbl = "CAS-optimised" if mode == "cas" else "Baseline dispatch"
    ax.set_xlabel("Annual CO\u2082 Emissions (Mt/yr)", fontsize=12)
    ax.set_ylabel("30-Year Lifetime Cost (billion 2030$)", fontsize=12)
    ax.set_title(f"Lifetime Cost vs. Annual Emissions\n"
                 f"({mode_lbl}, {len(df)} portfolios)",
                 fontsize=12, fontweight="bold")

    extra = [Line2D([0], [0], color="#9f1239", lw=1.8, ls="--",
                    label="Pareto front")]
    ax.legend(handles=_group_legend(df, extra),
              fontsize=8.5, loc="upper right", framealpha=0.9)

    _save(fig, out_dir / f"pareto_lifetime_{mode}.png")


# =============================================================================
# Figure 6: Combined CAS vs. no-CAS
# =============================================================================

def fig_combined(df_cas: pd.DataFrame, df_nocas: pd.DataFrame,
                 econ: pd.DataFrame | None, sim_days: int, out_dir: Path):
    """Overlay CAS (filled) and no-CAS (open) on one plot."""
    # Determine x/y columns based on whether we have lifetime data
    use_lifetime = (econ is not None
                    and "lifetime_cost_cas" in econ.columns
                    and "lifetime_cost_inv" in econ.columns)

    if use_lifetime:
        co2_cas = dict(zip(df_cas["hash"], df_cas["co2_tonnes"]))
        co2_inv = dict(zip(df_nocas["hash"], df_nocas["co2_tonnes"]))
        df = econ.copy()
        df["x_cas"] = df["hash"].map(co2_cas) / sim_days * 365 / 1e6
        df["y_cas"] = df["lifetime_cost_cas"] / 1e9
        df["x_inv"] = df["hash"].map(co2_inv) / sim_days * 365 / 1e6
        df["y_inv"] = df["lifetime_cost_inv"] / 1e9
        y_label = "30-Year Lifetime Cost (billion 2030$)"
    else:
        # Merge on hash; use annualised total cost
        scale = 365 / sim_days
        merged = df_cas.merge(df_nocas, on="hash", suffixes=("_cas", "_nocas"))
        df = merged.copy()
        df["group"] = df["group_cas"]
        df["x_cas"] = df["co2_tonnes_cas"] * scale / 1e6
        df["y_cas"] = df["total_cost_usd_cas"] * scale / 1e9
        df["x_inv"] = df["co2_tonnes_nocas"] * scale / 1e6
        df["y_inv"] = df["total_cost_usd_nocas"] * scale / 1e9
        y_label = "Annual Total Cost ($B/yr)"

    df.dropna(subset=["x_cas", "y_cas", "x_inv", "y_inv"], inplace=True)
    df["group_base"] = df["group"].apply(_group_base)

    x_cas, y_cas = df["x_cas"].values, df["y_cas"].values
    x_inv, y_inv = df["x_inv"].values, df["y_inv"].values

    pidx_cas = find_pareto_front(x_cas, y_cas)
    pidx_inv = find_pareto_front(x_inv, y_inv)

    fig, ax = plt.subplots(figsize=(10, 7))

    plotted: set[str] = set()

    for i, (_, row) in enumerate(df.iterrows()):
        sty = _get_style(row["group"])
        base = row["group_base"]

        label = None
        if base not in plotted:
            label = sty["label"]
            plotted.add(base)

        on_cas = i in set(pidx_cas)
        on_inv = i in set(pidx_inv)

        # CAS: filled
        ax.scatter(row["x_cas"], row["y_cas"],
                   marker=sty["marker"], c=sty["color"],
                   s=sty["s"] * (1.5 if on_cas else 1.0),
                   alpha=1.0 if on_cas else sty["alpha"],
                   edgecolors="black" if on_cas else "white",
                   linewidths=1.0 if on_cas else 0.4,
                   label=label, zorder=5 if on_cas else 3)

        # No-CAS: open
        ax.scatter(row["x_inv"], row["y_inv"],
                   marker=sty["marker"], facecolors="none",
                   edgecolors=sty["color"],
                   s=sty["s"] * (1.5 if on_inv else 1.0),
                   linewidths=1.5 if on_inv else 1.0, zorder=4)

        # Arrow from no-CAS to CAS (only for Pareto-front portfolios)
        if on_cas or on_inv:
            ax.annotate(
                "", xy=(row["x_cas"], row["y_cas"]),
                xytext=(row["x_inv"], row["y_inv"]),
                arrowprops=dict(arrowstyle="->", color=sty["color"],
                                lw=1.0, alpha=0.5),
                zorder=2)
        else:
            ax.plot([row["x_inv"], row["x_cas"]],
                    [row["y_inv"], row["y_cas"]],
                    color=sty["color"], lw=0.4, alpha=0.25, zorder=1)

    # Pareto front lines
    ax.plot(x_cas[pidx_cas], y_cas[pidx_cas],
            "k-", lw=1.5, alpha=0.7, zorder=5, label="Pareto front (CAS)")
    ax.plot(x_inv[pidx_inv], y_inv[pidx_inv],
            "k--", lw=1.5, alpha=0.5, zorder=5, label="Pareto front (no-CAS)")

    # Annotate CAS Pareto points
    for i in pidx_cas:
        row = df.iloc[i]
        pid = row.get("portfolio_id", i)
        ax.annotate(f"P{int(pid)}", (row["x_cas"], row["y_cas"]),
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=7, alpha=0.85, color="#333333")

    # Mode legend entries
    ax.scatter([], [], marker="s", c="gray", s=60, edgecolors="black",
               linewidths=0.5, label="CAS (filled)")
    ax.scatter([], [], marker="s", facecolors="none", edgecolors="gray",
               s=60, linewidths=1.5, label="No-CAS (open)")

    ax.set_xlabel("Annual CO\u2082 Emissions (Mt/yr)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title("CAS vs. No-CAS Pareto Comparison\n"
                 f"({len(df)} portfolios)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    _save(fig, out_dir / "pareto_combined.png")


# =============================================================================
# Summary table
# =============================================================================

def _print_summary(df: pd.DataFrame, sim_days: int):
    scale = 365 / sim_days
    print(f"\n{'=' * 80}")
    print(f"PORTFOLIO SUMMARY ({len(df)} portfolios)")
    print(f"{'=' * 80}")
    print(f"{'ID':>3}  {'Group':<18}  {'Wind GW':>8}  {'Solar GW':>9}  "
          f"{'Bat GW':>7}  {'CO2 Mt/yr':>10}  {'Cost $B/yr':>11}")
    print("-" * 80)

    co2_mt = df["co2_tonnes"] * scale / 1e6
    cost_b = df["total_cost_usd"] * scale / 1e9

    for i, (_, r) in enumerate(df.iterrows()):
        print(f"{r['portfolio_id']:3.0f}  {r['group']:<18}"
              f"  {r['wind_mw'] / 1e3:>8.1f}  {r['solar_mw'] / 1e3:>9.1f}"
              f"  {r['battery_mw'] / 1e3:>7.2f}"
              f"  {co2_mt.iloc[i]:>10.2f}  {cost_b.iloc[i]:>11.2f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Pareto analysis for TX renewable investment studies")
    parser.add_argument("--study-dir",
                        default="/scratch/gpfs/SIRCAR/dp4800/vatic/outputs/"
                                "TX_PARETO_SCALE",
                        help="Path to study output directory")
    parser.add_argument("--sim-days", type=int, default=None,
                        help="Simulation days (auto-detected if not given)")
    parser.add_argument("--plots", nargs="*",
                        default=["all"],
                        choices=["all", "cost-carbon", "winwin", "efficiency",
                                 "breakdown", "lifetime", "combined"],
                        help="Which figures to generate")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (defaults to study-dir)")
    args = parser.parse_args()

    study = Path(args.study_dir)
    out_dir = Path(args.out_dir) if args.out_dir else study
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect sim_days from study name if not specified
    if args.sim_days is None:
        if "PARETO_SCALE" in study.name:
            sim_days = 14
        elif "PARETO" in study.name:
            sim_days = 56
        else:
            sim_days = 14
        print(f"Auto-detected sim_days={sim_days} for {study.name}")
    else:
        sim_days = args.sim_days

    plots = set(args.plots)
    do_all = "all" in plots

    # Load data — auto-merge P1 + P2 if both exist
    results_dir = study / "results"
    nocas_dir = study / "results_nocas"
    results_p2_dir = study / "results_p2"
    nocas_p2_dir = study / "results_nocas_p2"
    econ_csv = study / "economic_analysis.csv"

    df = load_json_results(results_dir)
    if df.empty:
        print(f"No valid results in {results_dir}")
        return

    # Merge P2 if available
    if results_p2_dir.exists():
        df_p2 = load_json_results(results_p2_dir)
        if not df_p2.empty:
            print(f"Merging P1 ({len(df)}) + P2 ({len(df_p2)}) CAS results")
            df = merge_phase_results(df, df_p2)
            sim_days = sim_days * 2  # 14 → 28
            print(f"  Updated sim_days to {sim_days}")

    df_nocas = None
    if nocas_dir.exists():
        df_nocas = load_json_results(nocas_dir)
        if df_nocas.empty:
            df_nocas = None

    if df_nocas is not None and nocas_p2_dir.exists():
        df_nocas_p2 = load_json_results(nocas_p2_dir)
        if not df_nocas_p2.empty:
            print(f"Merging P1 ({len(df_nocas)}) + P2 ({len(df_nocas_p2)}) no-CAS results")
            df_nocas = merge_phase_results(df_nocas, df_nocas_p2)

    econ = None
    if econ_csv.exists():
        econ = pd.read_csv(econ_csv)

    print(f"Loaded {len(df)} portfolios (CAS)")
    if df_nocas is not None:
        print(f"Loaded {len(df_nocas)} portfolios (no-CAS)")
    if econ is not None:
        print(f"Loaded economic analysis ({len(econ)} rows)")

    _print_summary(df, sim_days)
    print()

    # Generate figures
    eff_idx = None

    if do_all or "cost-carbon" in plots:
        print("Figure: cost-carbon")
        fig_cost_carbon(df, sim_days, out_dir)

    if do_all or "winwin" in plots:
        print("Figure: win-win landscape")
        fig_winwin(df, out_dir)

    if do_all or "efficiency" in plots:
        print("Figure: efficiency frontier")
        eff_idx = fig_efficiency(df, sim_days, out_dir)

    if do_all or "breakdown" in plots:
        print("Figure: efficiency-front breakdown")
        if eff_idx is None:
            # Compute efficiency frontier if not already done
            bl = df[df["group_base"] == "baseline"]
            if not bl.empty:
                bl_co2 = bl["co2_tonnes"].iloc[0]
                co2_red = -(df["co2_tonnes"].values - bl_co2) / bl_co2 * 100
                capex = df["capex_usd"].values * 365 / sim_days / 1e9
                eff_idx = find_pareto_front(-co2_red, capex)
        fig_breakdown(df, eff_idx, sim_days, out_dir)

    if do_all or "lifetime" in plots:
        if econ is not None:
            for mode in ("cas", "inv"):
                print(f"Figure: lifetime ({mode})")
                # Use appropriate JSON source for CO2
                src = df if mode == "cas" else (df_nocas if df_nocas is not None
                                                else df)
                fig_lifetime(src, econ, mode, sim_days, out_dir)
        else:
            print("  Skipping lifetime plots: no economic_analysis.csv")

    if do_all or "combined" in plots:
        if df_nocas is not None:
            print("Figure: combined CAS vs no-CAS")
            fig_combined(df, df_nocas, econ, sim_days, out_dir)
        else:
            print("  Skipping combined: no results_nocas/ directory")

    print("\nDone.")


if __name__ == "__main__":
    main()
