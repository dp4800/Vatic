#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
cost_analysis.py — Cost & Pareto analysis helpers for the HTML report.

Generates:
  • Grouped % change bar chart (CO₂ + cost components vs baseline)
  • LP alpha trade-off: per-month dual-axis lines + annual average
  • Monthly chronological cost–carbon path (Jan → Dec)
  • HTML tables for CAPEX, portfolio, and projected performance
"""
import base64
import csv
import io
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

SC_LABELS = {
    "baseline": "Baseline (DC, no shift)",
    "sim-gm":   "Grid-Mix (CAS-GM)",
    "sim-247":  "24/7 Renewable (CAS-247)",
    "sim-lp":   "LP Price-Taking (CAS-LP, α=0)",
}
SC_COLORS = {
    "baseline": "#4682b4",
    "sim-gm":   "#e07b00",
    "sim-247":  "#228b22",
    "sim-lp":   "#c0152e",
}
SC_MARKERS = {
    "baseline": "o",
    "sim-gm":   "s",
    "sim-247":  "^",
    "sim-lp":   "D",
}

MONTHS = [
    "2020-01-06","2020-02-03","2020-03-02","2020-04-06",
    "2020-05-04","2020-06-01","2020-07-06","2020-08-03",
    "2020-09-07","2020-10-05","2020-11-02","2020-12-07",
]

MONTH_LABELS = {
    "2020-01-06": "Jan", "2020-02-03": "Feb", "2020-03-02": "Mar",
    "2020-04-06": "Apr", "2020-05-04": "May", "2020-06-01": "Jun",
    "2020-07-06": "Jul", "2020-08-03": "Aug", "2020-09-07": "Sep",
    "2020-10-05": "Oct", "2020-11-02": "Nov", "2020-12-07": "Dec",
}

SEASON_MAP = {
    "2020-01-06": ("Jan", "Winter", "#4a90d9"),
    "2020-02-03": ("Feb", "Winter", "#4a90d9"),
    "2020-03-02": ("Mar", "Spring", "#5ba85a"),
    "2020-04-06": ("Apr", "Spring", "#5ba85a"),
    "2020-05-04": ("May", "Spring", "#5ba85a"),
    "2020-06-01": ("Jun", "Summer", "#d9823a"),
    "2020-07-06": ("Jul", "Summer", "#d9823a"),
    "2020-08-03": ("Aug", "Summer", "#d9823a"),
    "2020-09-07": ("Sep", "Fall",   "#9b6bbf"),
    "2020-10-05": ("Oct", "Fall",   "#9b6bbf"),
    "2020-11-02": ("Nov", "Fall",   "#9b6bbf"),
    "2020-12-07": ("Dec", "Winter", "#4a90d9"),
}

SEASON_PATCHES = [
    mpatches.Patch(color="#4a90d9", label="Winter"),
    mpatches.Patch(color="#5ba85a", label="Spring"),
    mpatches.Patch(color="#d9823a", label="Summer"),
    mpatches.Patch(color="#9b6bbf", label="Fall"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crf(r: float, n: int) -> float:
    """Capital Recovery Factor."""
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def load_system_totals() -> dict[str, dict[str, float]]:
    """Annual totals from system_comparison.csv: metric → scenario → value."""
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for dt in MONTHS:
        p = OUTPUTS_DIR / dt / "compare" / "system_comparison.csv"
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                metric = row[""]
                for sc in ["baseline", "sim-gm", "sim-247", "sim-lp"]:
                    try:
                        totals[metric][sc] += float(row.get(sc, 0) or 0)
                    except ValueError:
                        pass
    return {k: dict(v) for k, v in totals.items()}


def load_lp_pareto() -> dict[float, dict[str, float]]:
    """Annual-average LP alpha sweep data from cas-lp/cas_results.csv."""
    by_alpha: dict[float, dict] = defaultdict(lambda: {"carbon": 0.0, "dc_cost": 0.0, "n": 0})
    for dt in MONTHS:
        p = OUTPUTS_DIR / dt / "cas-lp" / "cas_results.csv"
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                a = float(row["alpha"])
                by_alpha[a]["carbon"]  += float(row["carbon_reduction_pct"])
                by_alpha[a]["dc_cost"] += float(row["cost_reduction_pct"])
                by_alpha[a]["n"]       += 1
    return {a: {"carbon_pct": d["carbon"] / d["n"],
                "dc_cost_pct": d["dc_cost"] / d["n"]}
            for a, d in by_alpha.items()}


def load_lp_pareto_by_month() -> dict[str, dict[float, dict[str, float]]]:
    """Per-month LP alpha sweep: {date: {alpha: {carbon_pct, dc_cost_pct}}}"""
    result: dict[str, dict[float, dict]] = {}
    for dt in MONTHS:
        p = OUTPUTS_DIR / dt / "cas-lp" / "cas_results.csv"
        if not p.exists():
            continue
        result[dt] = {}
        with open(p) as f:
            for row in csv.DictReader(f):
                a = float(row["alpha"])
                result[dt][a] = {
                    "carbon_pct": float(row["carbon_reduction_pct"]),
                    "dc_cost_pct": float(row["cost_reduction_pct"]),
                }
    return result


def load_dc_flex_capex() -> float:
    """Annual DC flexibility CAPEX (USD) — read from any month's dc_flex_capex.json."""
    for dt in MONTHS:
        p = OUTPUTS_DIR / dt / "dc_flex_capex.json"
        if p.exists():
            import json
            with open(p) as f:
                d = json.load(f)
            return float(d.get("dc_flex_capex_annual_usd", 0.0))
    return 0.0


def _load_by_month_sys() -> dict[str, dict[str, dict]]:
    """Per-month system_comparison.csv rows: {date: {metric: row_dict}}"""
    by_month: dict[str, dict[str, dict]] = {}
    for dt in MONTHS:
        p = OUTPUTS_DIR / dt / "compare" / "system_comparison.csv"
        if not p.exists():
            continue
        by_month[dt] = {}
        with open(p) as f:
            for row in csv.DictReader(f):
                by_month[dt][row[""]] = row
    return by_month


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Grouped % change bar chart vs baseline
# ─────────────────────────────────────────────────────────────────────────────

def fig_pareto_system(sys: dict, dc_flex_capex_usd: float) -> str:
    """
    Grouped % change bar chart: CO₂ emissions, variable cost, fixed cost,
    and total op cost vs baseline — one group per CAS scenario.
    Returns base64-encoded PNG.
    """
    co2   = sys.get("total_co2_tonnes",  {})
    var   = sys.get("variable_cost_usd", {})
    fixed = sys.get("fixed_cost_usd",    {})

    base_co2   = co2.get("baseline",   1.0)
    base_var   = var.get("baseline",   1.0)
    base_fixed = fixed.get("baseline", 1.0)
    base_total = base_var + base_fixed

    def pct(val, base):
        return (val - base) / abs(base) * 100 if base else 0.0

    scenarios  = ["sim-gm", "sim-247", "sim-lp"]
    sc_names   = ["Grid-Mix\n(CAS-GM)", "24/7 Renewable\n(CAS-247)", "LP Price-Taking\n(CAS-LP)"]
    sc_colors  = [SC_COLORS["sim-gm"], SC_COLORS["sim-247"], SC_COLORS["sim-lp"]]

    metrics       = ["CO₂ Emissions", "Variable Cost", "Fixed Cost", "Total Op. Cost"]
    metric_colors = ["#5a9fd4", "#e07b00", "#6aab6a", "#555"]
    metric_hatches = ["", "//", "xx", ""]

    x      = np.arange(len(scenarios))
    n_met  = len(metrics)
    width  = 0.18
    offsets = np.linspace(-(n_met - 1) / 2 * width, (n_met - 1) / 2 * width, n_met)

    fig, ax = plt.subplots(figsize=(11, 6))

    for j, (metric, mc, hatch, offset) in enumerate(
        zip(metrics, metric_colors, metric_hatches, offsets)
    ):
        vals = []
        for sc in scenarios:
            if j == 0:
                v = pct(co2.get(sc, 0), base_co2)
            elif j == 1:
                v = pct(var.get(sc, 0), base_var)
            elif j == 2:
                v = pct(fixed.get(sc, 0), base_fixed)
            else:
                v = pct(var.get(sc, 0) + fixed.get(sc, 0), base_total)
            vals.append(v)

        rects = ax.bar(x + offset, vals, width, label=metric, color=mc,
                       hatch=hatch, alpha=0.82, edgecolor="white", linewidth=0.6)

        for rect, val in zip(rects, vals):
            va    = "bottom" if val >= 0 else "top"
            ypos  = val + (0.08 if val >= 0 else -0.08)
            ax.text(rect.get_x() + rect.get_width() / 2, ypos,
                    f"{val:+.1f}%", ha="center", va=va,
                    fontsize=7.2, color=mc, fontweight="bold")

    ax.axhline(0, color="#444", lw=0.9, linestyle="--", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(sc_names, fontsize=11)
    ax.set_ylabel("Annual change vs baseline (%)", fontsize=11)
    ax.set_title("CAS Scenarios: Annual Change vs Baseline\n"
                 "(Operational cost components + CO₂ emissions)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.30)
    ax.tick_params(axis="x", which="both", length=0)

    flex_M = dc_flex_capex_usd / 1e6
    ax.text(0.98, 0.02,
            f"DC flexibility CAPEX (${flex_M:.0f}M/yr)\nnot included in cost bars",
            transform=ax.transAxes, fontsize=8.5, color="#888", style="italic",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.85))

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — LP Alpha Trade-off (per-month lines + annual average)
# ─────────────────────────────────────────────────────────────────────────────

def fig_lp_alpha_tradeoff(pareto: dict[float, dict]) -> str:
    """
    Two-panel figure:
    (a) Dual-axis: carbon reduction % and DC cost reduction % vs alpha.
        Light per-month lines + bold annual-average.
    (b) Trade-off frontier in (carbon%, cost%) space — annual average, α=0.1–1.
    """
    by_month = load_lp_pareto_by_month()

    # Annual-average lines, excluding α=0 from trend (mark it as outlier)
    alphas_all  = sorted(pareto.keys())
    alphas_trend = [a for a in alphas_all if a > 0]
    avg_carbon   = [pareto[a]["carbon_pct"]  for a in alphas_trend]
    avg_cost     = [pareto[a]["dc_cost_pct"] for a in alphas_trend]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("LP Price-Taking: Effect of Carbon Weight α on DC Load Shifting",
                 fontsize=13, fontweight="bold")

    # ── Panel A: dual-axis alpha sweep ───────────────────────────────────────
    ax1b = ax1.twinx()

    # Per-month lines (faint, season-coloured)
    for dt in MONTHS:
        if dt not in by_month:
            continue
        _, _, color = SEASON_MAP.get(dt, ("?", "?", "#888"))
        month_data  = by_month[dt]
        alphas_m    = sorted(a for a in month_data if a > 0)
        if not alphas_m:
            continue
        c_vals = [month_data[a]["carbon_pct"]  for a in alphas_m]
        k_vals = [month_data[a]["dc_cost_pct"] for a in alphas_m]
        ax1.plot(alphas_m, c_vals, color=color, alpha=0.30, lw=1.1, zorder=2)
        ax1b.plot(alphas_m, k_vals, color=color, alpha=0.18, lw=1.0,
                  linestyle="--", zorder=1)

    # Annual average — bold
    ax1.plot(alphas_trend, avg_carbon, color="#1a5276", lw=2.8, zorder=5,
             label="Annual avg — CO₂ red. %")
    ax1b.plot(alphas_trend, avg_cost, color="#b03a2e", lw=2.8, linestyle="--",
              zorder=5, label="Annual avg — cost red. %")

    # Mark α=0 as outlier (x both axes)
    if 0.0 in pareto:
        ax1.scatter([0.0], [pareto[0.0]["carbon_pct"]], s=110, color="#1a5276",
                    marker="x", zorder=7, linewidths=2.5)
        ax1b.scatter([0.0], [pareto[0.0]["dc_cost_pct"]], s=110, color="#b03a2e",
                     marker="x", zorder=7, linewidths=2.5)
        ax1.annotate("α=0 excluded\n(see note)",
                     xy=(0.0, pareto[0.0]["carbon_pct"]),
                     xytext=(0.15, pareto[0.0]["carbon_pct"] + 1.8),
                     fontsize=8.5, color="#666",
                     arrowprops=dict(arrowstyle="-|>", color="#aaa", lw=1.0))

    ax1.set_xlabel("Carbon weight α  (0 = pure cost-min, 1 = pure CO₂-min)", fontsize=10)
    ax1.set_ylabel("DC Carbon Reduction (%)", fontsize=11, color="#1a5276")
    ax1b.set_ylabel("DC Cost Reduction (%)", fontsize=11, color="#b03a2e")
    ax1.tick_params(axis="y", labelcolor="#1a5276")
    ax1b.tick_params(axis="y", labelcolor="#b03a2e")
    ax1.set_title("(a) α sweep: carbon & cost reduction per month\n"
                  "(α=0 excluded from lines — see note below)", fontsize=10.5)
    ax1.grid(True, alpha=0.28)

    # Combined legend: avg lines + season colours
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + SEASON_PATCHES,
               labs1  + labs2  + [p.get_label() for p in SEASON_PATCHES],
               fontsize=8.0, loc="lower left", ncol=2, framealpha=0.9)

    # α=0 anomaly note
    ax1.text(0.02, 0.02,
             "α=0 anomaly: pure cost-min optimizer finds a\n"
             "more expensive solution than α=0.1 in most months\n"
             "(price signals already proxy for carbon signals)",
             transform=ax1.transAxes, fontsize=7.8, color="#555",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff8e1",
                       edgecolor="#ccc", alpha=0.92))

    # ── Panel B: trade-off frontier (annual avg, α=0.1–1) ───────────────────
    ax2.plot(avg_carbon, avg_cost, "-o", color="#2e6da4", lw=2.2, ms=7, zorder=4)
    ax2.fill_between(avg_carbon, avg_cost,
                     min(avg_cost) * 0.97, alpha=0.08, color="#2e6da4")

    for a, c, k in zip(alphas_trend, avg_carbon, avg_cost):
        if a in (0.1, 0.5, 1.0):
            ax2.annotate(f"α={a:.1f}", (c, k),
                         textcoords="offset points", xytext=(5, 5),
                         fontsize=9, color="#333")

    ax2.set_xlabel("DC Carbon Reduction (% of DC carbon footprint)", fontsize=11)
    ax2.set_ylabel("DC Cost Reduction (%)", fontsize=11)
    ax2.set_title("(b) Annual-average trade-off frontier\n"
                  "(α=0.1 → 1.0 only; each point = annual average across 12 months)",
                  fontsize=10.5)
    ax2.grid(True, alpha=0.32)
    ax2.text(0.05, 0.06,
             "Key insight: increasing α (toward pure CO₂-min)\nreduces cost savings more than carbon savings\n"
             "→ price signals already approximate carbon signals",
             transform=ax2.transAxes, fontsize=8.5, color="#555",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0f8ff",
                       edgecolor="#bcd", alpha=0.92))

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Monthly chronological cost–carbon path
# ─────────────────────────────────────────────────────────────────────────────

def fig_monthly_pareto(sys: dict) -> str:
    """
    Per-month cost vs CO₂ connected Jan → Dec with directional arrows.
    Two panels: baseline and LP scenario. Labels in white-background boxes,
    offset perpendicular to path direction to minimise overlap.
    """
    by_month = _load_by_month_sys()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.suptitle("Monthly Operational Cost vs CO₂ — Chronological Path (Jan → Dec)",
                 fontsize=13, fontweight="bold")

    for ax, sc, title in zip(
        axes,
        ["baseline", "sim-lp"],
        ["(a) Baseline (DC, no shift)", "(b) LP Price-Taking (CAS-LP, α=0)"]
    ):
        xs, ys, labels, colors = [], [], [], []
        for dt in MONTHS:
            if dt not in by_month:
                continue
            d  = by_month[dt]
            x  = float(d.get("total_co2_tonnes", {}).get(sc, 0) or 0) / 1000  # kt
            y  = (float(d.get("variable_cost_usd", {}).get(sc, 0) or 0)
                + float(d.get("fixed_cost_usd",    {}).get(sc, 0) or 0)) / 1e6  # $M
            lbl, _, color = SEASON_MAP.get(dt, ("?", "?", "#888"))
            xs.append(x); ys.append(y)
            labels.append(lbl); colors.append(color)

        if len(xs) < 2:
            continue

        xr = max(xs) - min(xs) or 1.0
        yr = max(ys) - min(ys) or 1.0

        # Grey connecting line
        ax.plot(xs, ys, "-", color="#ccc", lw=1.6, zorder=1)

        # Directional arrows (mid-segment)
        for i in range(len(xs) - 1):
            mx = (xs[i] + xs[i + 1]) / 2
            my = (ys[i] + ys[i + 1]) / 2
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            ax.annotate("",
                        xy=(mx + dx * 0.01, my + dy * 0.01),
                        xytext=(mx - dx * 0.01, my - dy * 0.01),
                        arrowprops=dict(arrowstyle="-|>", color="#aaa",
                                        lw=0.8, mutation_scale=11))

        # Scatter dots (season colour)
        for i in range(len(xs)):
            ax.scatter(xs[i], ys[i], s=75, color=colors[i], zorder=5,
                       edgecolors="white", linewidths=0.9)

        # Labels offset perpendicular to local path direction
        n = len(labels)
        PERP_SCALE = 0.13   # fraction of axis range
        for i in range(n):
            if i < n - 1:
                ddx = xs[i + 1] - xs[i]
                ddy = ys[i + 1] - ys[i]
            else:
                ddx = xs[i] - xs[i - 1]
                ddy = ys[i] - ys[i - 1]
            norm = math.sqrt((ddx / xr) ** 2 + (ddy / yr) ** 2) or 1.0
            # Perpendicular direction (normalised to data units)
            px = (-ddy / yr / norm) * xr * PERP_SCALE
            py = ( ddx / xr / norm) * yr * PERP_SCALE
            ax.annotate(labels[i],
                        xy=(xs[i], ys[i]),
                        xytext=(xs[i] + px, ys[i] + py),
                        fontsize=8.2, color=colors[i], fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor=colors[i], alpha=0.80, linewidth=0.8),
                        arrowprops=dict(arrowstyle="-", color=colors[i],
                                        lw=0.5, alpha=0.5))

        # Mark Jan (start) and Dec (end)
        ax.scatter([xs[0]], [ys[0]], s=130, color=colors[0], zorder=6,
                   marker="o", edgecolors="#333", linewidths=1.2)
        ax.scatter([xs[-1]], [ys[-1]], s=130, color=colors[-1], zorder=6,
                   marker="s", edgecolors="#333", linewidths=1.2)

        ax.set_xlabel("CO₂ (kt, 14-day window)", fontsize=10)
        ax.set_ylabel("Operational Cost ($M, 14-day window)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.28)

    # Legend: seasons + start/end markers
    start_patch = mpatches.Patch(color="#999", label="● Jan (start)")
    end_patch   = mpatches.Patch(color="#999", label="■ Dec (end)")
    fig.legend(handles=SEASON_PATCHES + [start_patch, end_patch],
               loc="lower center", ncol=6,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.06), framealpha=0.9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# HTML table builders
# ─────────────────────────────────────────────────────────────────────────────

def _crf_val(r: float, n: int) -> float:
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def html_portfolio_table() -> str:
    """
    Table: optimal portfolio financial analysis (wind, solar, battery).
    Uses CAPEX assumptions from params_scaled.json.
    Reference portfolio based on representative investment optimization outputs.
    """
    r = 0.07
    assets = [
        # name, mw, capex_per_kw, life_yr, om_per_mw_yr
        ("Onshore Wind",   2968, 1400, 25, 29_000),
        ("Solar PV",       5000, 1100, 25, 17_000),
        ("Battery (1-hr)", 112,  1200, 15, 10_000),
    ]
    hdrs = ["Asset", "Optimal MW", "CAPEX ($/kW)", "Life (yr)", "CRF",
            "Ann. CAPEX ($M)", "Ann. O&M ($M)", "Total Ann. Cost ($M)"]
    hdr_html = "".join(f'<th data-col="{i}">{h}</th>' for i, h in enumerate(hdrs))
    rows_html = ""
    totals = [0.0, 0.0, 0.0]
    for name, mw, capex_kw, life, om_mw in assets:
        crf = _crf_val(r, life)
        ann_capex = mw * capex_kw * 1000 * crf / 1e6   # $M
        ann_om    = mw * om_mw / 1e6                     # $M
        total_ann = ann_capex + ann_om
        totals[0] += ann_capex; totals[1] += ann_om; totals[2] += total_ann
        rows_html += f"""<tr>
  <td><strong>{name}</strong></td>
  <td class="num">{mw:,}</td>
  <td class="num">{capex_kw:,}</td>
  <td class="num">{life}</td>
  <td class="num">{crf:.4f}</td>
  <td class="num">{ann_capex:.1f}</td>
  <td class="num">{ann_om:.1f}</td>
  <td class="num"><strong>{total_ann:.1f}</strong></td>
</tr>"""
    rows_html += f"""<tr style="background:#f4f7fb;font-weight:700">
  <td colspan="5">Total</td>
  <td class="num">{totals[0]:.1f}</td>
  <td class="num">{totals[1]:.1f}</td>
  <td class="num">{totals[2]:.1f}</td>
</tr>"""
    return f"""<div class="tbl-wrap">
<table class="sortable">
  <thead><tr>{hdr_html}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>"""


def html_capex_comparison_table() -> str:
    """
    Table: CAPEX comparison across renewable, storage, and fossil sources.
    Reference values from NREL ATB 2024 and EIA.
    """
    sources = [
        # name, capex_kw_low, capex_kw_high, life, om_fixed, source
        ("Onshore Wind",        1200, 1600, 25, "29,000",  "NREL ATB 2024"),
        ("Offshore Wind",       3000, 4500, 25, "87,000",  "NREL ATB 2024"),
        ("Utility Solar PV",     900, 1300, 25, "17,000",  "NREL ATB 2024"),
        ("Battery (4-hr BESS)", 1100, 1400, 15, "10,000",  "NREL ATB 2024"),
        ("Nuclear (AP1000)",    6000, 9000, 60, "120,000", "EIA 2023"),
        ("Coal (new build)",    3600, 4500, 40, "40,000",  "EIA 2023"),
        ("CCGT",                 800, 1100, 30, "12,000",  "EIA 2023"),
        ("Combustion Turbine",   600,  800, 25, "7,000",   "EIA 2023"),
        ("DC Flex. (30% HDR)",  2000, 2000,  5, "N/A",     "Assumed (server CAPEX)"),
    ]
    hdrs = ["Source", "CAPEX Range ($/kW)", "Economic Life (yr)",
            "Fixed O&M ($/MW-yr)", "Reference"]
    hdr_html = "".join(f'<th data-col="{i}">{h}</th>' for i, h in enumerate(hdrs))
    rows_html = ""
    for name, lo, hi, life, om, ref in sources:
        is_renew = name in ("Onshore Wind","Offshore Wind","Utility Solar PV","Battery (4-hr BESS)")
        is_flex  = "DC Flex" in name
        style = ""
        if is_renew:
            style = ' style="background:#f5fbf5"'
        elif is_flex:
            style = ' style="background:#fff8f0"'
        rng = f"{lo:,}–{hi:,}" if lo != hi else f"{lo:,}"
        rows_html += f"""<tr{style}>
  <td><strong>{name}</strong></td>
  <td class="num">{rng}</td>
  <td class="num">{life}</td>
  <td class="num">{om}</td>
  <td style="font-size:12px;color:#666">{ref}</td>
</tr>"""
    return f"""<div class="tbl-wrap">
<table class="sortable">
  <thead><tr>{hdr_html}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>"""


def html_projected_performance(sys: dict, dc_flex_capex_usd: float) -> str:
    """
    Table: Projected Annual Performance — Baseline vs each CAS scenario.
    """
    co2_t = sys.get("total_co2_tonnes",  {})
    var_v = sys.get("variable_cost_usd", {})
    fix_v = sys.get("fixed_cost_usd",    {})

    flex_M = dc_flex_capex_usd / 1e6

    scenarios = ["baseline", "sim-gm", "sim-247", "sim-lp"]
    sc_names  = ["Baseline (no shift)", "Grid-Mix", "24/7 Renewable", "LP Price-Taking"]

    hdrs = ["Metric", "Unit"] + sc_names
    hdr_html = "".join(f'<th data-col="{i}">{h}</th>' for i, h in enumerate(hdrs))

    def row(label, unit, values, fmt=".1f", best_low=True):
        cells = f"<td><strong>{label}</strong></td><td>{unit}</td>"
        best_idx = values.index(min(values) if best_low else max(values))
        for i, v in enumerate(values):
            style = ' style="background:#e8f5e9;font-weight:700"' if i == best_idx else ""
            cells += f'<td class="num"{style}>{v:{fmt}}</td>'
        return f"<tr>{cells}</tr>"

    op_costs   = [(var_v.get(sc, 0) + fix_v.get(sc, 0)) / 1e6 for sc in scenarios]
    flex_costs = [0.0 if sc == "baseline" else flex_M for sc in scenarios]
    total_sys  = [op + fl for op, fl in zip(op_costs, flex_costs)]
    co2_mt     = [co2_t.get(sc, 0) / 1e6 for sc in scenarios]

    base_co2   = co2_mt[0]
    base_total = total_sys[0]

    rows_html = ""
    rows_html += row("Operational cost (var + fixed)", "$M/yr", op_costs)
    rows_html += row("DC flexibility CAPEX", "$M/yr", flex_costs)
    rows_html += row("Total system cost", "$M/yr", total_sys)
    rows_html += ("<tr style='background:#f9f9f9'><td colspan='6' "
                  "style='padding:.3rem .75rem;font-size:12px;color:#888'>CO₂ metrics</td></tr>")
    rows_html += row("Annual CO₂", "Mt/yr", co2_mt, fmt=".3f")

    def delta_row(label, unit, base_val, vals):
        cells = f"<td><em>{label}</em></td><td>{unit}</td>"
        for i, v in enumerate(vals):
            d   = v - base_val
            pct = (d / abs(base_val) * 100) if base_val else 0
            sign  = "+" if d > 0 else ""
            color = "#228b22" if d < 0 else ("#c0152e" if d > 0 else "#555")
            cells += (f'<td class="num" style="color:{color};font-size:12.5px">'
                      f'{sign}{d:.2f} ({sign}{pct:.1f}%)</td>')
        return f"<tr>{cells}</tr>"

    rows_html += delta_row("  Δ vs Baseline — cost", "$M/yr", base_total, total_sys)
    rows_html += delta_row("  Δ vs Baseline — CO₂",  "Mt/yr", base_co2,   co2_mt)

    return f"""<div class="tbl-wrap">
<table class="sortable">
  <thead><tr>{hdr_html}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>"""


def html_cas_flex_cost() -> str:
    """
    Table: CAS flexibility infrastructure cost breakdown.
    """
    import json
    capex = {}
    for dt in MONTHS:
        p = OUTPUTS_DIR / dt / "dc_flex_capex.json"
        if p.exists():
            with open(p) as f:
                capex = json.load(f)
            break
    if not capex:
        return "<p>dc_flex_capex.json not found.</p>"

    peak    = capex["dc_peak_mw"]
    extra   = capex["extra_capacity_mw"]
    cpkw    = capex["server_capex_per_kw_usd"]
    life    = capex["server_life_yr"]
    total_c = extra * 1000 * cpkw / 1e6   # $M total capital
    r       = 0.07
    crf_val = _crf_val(r, life)
    ann     = capex["dc_flex_capex_annual_usd"] / 1e6

    rows = [
        ("Total DC peak load",               f"{peak:.1f} MW",       "843 MW across 51 buses"),
        ("Flexibility headroom",             "30% of peak",          "CAS parameter: extra_capacity_pct"),
        ("Extra server capacity required",   f"{extra:.1f} MW",      f"{peak:.1f} × 30%"),
        ("Server CAPEX assumption",          f"${cpkw:,.0f}/kW",     "Data-center server infrastructure"),
        ("Equipment economic life",          f"{life} years",        "Assumed server refresh cycle"),
        ("Discount rate",                    "7%",                   "Real discount rate"),
        ("Capital Recovery Factor",          f"{crf_val:.4f}",       f"CRF(7%, {life}yr)"),
        ("Total capital outlay",             f"${total_c:,.1f}M",    "252.4 MW × $2,000/kW"),
        ("Annualized CAPEX",                 f"${ann:.1f}M/yr",      "Applies to all CAS scenarios"),
    ]
    rows_html = "".join(
        f'<div class="param-k">{k}</div>'
        f'<div class="param-v">{v} '
        f'<span style="color:#888;font-size:12px">({note})</span></div>'
        for k, v, note in rows
    )
    return f'<div class="param-grid">{rows_html}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def build_cost_section() -> dict[str, str]:
    """
    Return dict of named HTML/base64 strings for the cost analysis section.
    Keys: pareto_fig, alpha_fig, monthly_pareto_fig,
          portfolio_table, capex_table, performance_table, flex_cost_table
    """
    sys_totals    = load_system_totals()
    lp_pareto     = load_lp_pareto()
    dc_flex_capex = load_dc_flex_capex()

    return {
        "pareto_fig":         fig_pareto_system(sys_totals, dc_flex_capex),
        "alpha_fig":          fig_lp_alpha_tradeoff(lp_pareto),
        "monthly_pareto_fig": fig_monthly_pareto(sys_totals),
        "portfolio_table":    html_portfolio_table(),
        "capex_table":        html_capex_comparison_table(),
        "performance_table":  html_projected_performance(sys_totals, dc_flex_capex),
        "flex_cost_table":    html_cas_flex_cost(),
    }


if __name__ == "__main__":
    print("Generating cost analysis components …")
    parts = build_cost_section()
    for k, v in parts.items():
        tag  = "figure" if "fig" in k else "table"
        size = len(v) // 1024
        print(f"  {k}: {tag} ({size} KB)")
    print("Done.")
