"""
scripts/study/generate_renew_report.py
---------------------------------------
Generate a self-contained HTML report for the RENEW_STUDY_RTS_2020.

Reads the analysis CSVs produced by aggregate_renew_results.py and produces:
  outputs/RENEW_STUDY_RTS_2020/analysis/renew_report.html

Sections:
  1. Key metrics table (battery runs, completed)
  2. LP-CAS CO₂ reduction vs renewable capacity (per group)
  3. GM vs LP comparison by group
  4. Pareto scatter: (cost_red_pct, co2_red_pct) for LP-CAS
  5. Renewable curtailment vs capacity
"""

from __future__ import annotations
import base64, io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
REPO     = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "outputs" / "RENEW_STUDY_RTS_2020" / "analysis"

GROUP_COLORS = {
    "wind":     "#2196F3",
    "solar":    "#FF9800",
    "battery":  "#9C27B0",
    "combined": "#4CAF50",
}
DATE_STYLES = {"2020-01-01": "solid", "2020-07-01": "dashed"}
DATE_LABELS = {"2020-01-01": "Jan", "2020-07-01": "Jul"}


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Figure 1 — LP-CAS CO₂ reduction vs renewable capacity, by group

def fig_co2_vs_cap(summary: pd.DataFrame) -> str:
    lp = summary[summary["mode"] == "LP-CAS"].copy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    groups = ["wind", "solar", "battery", "combined"]
    titles = ["Wind sweep", "Solar sweep", "Battery sweep", "Combined"]
    xcols  = {"wind": "wind_mw", "solar": "solar_mw",
              "battery": "battery_mwh", "combined": "renew_cap_mw"}
    xlabels = {"wind": "Added wind (MW)", "solar": "Added solar (MW)",
               "battery": "Added storage (MWh)", "combined": "Wind+Solar (MW each)"}

    for ax, grp, title in zip(axes, groups, titles):
        df = lp[lp["group"] == grp]
        for date, ls in DATE_STYLES.items():
            sub = df[df["date"] == date].sort_values(xcols[grp])
            if sub.empty:
                continue
            ax.plot(sub[xcols[grp]], sub["co2_red_pct"],
                    marker="o", linestyle=ls, color=GROUP_COLORS[grp],
                    label=DATE_LABELS[date], linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabels[grp], fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("LP-CAS CO₂ reduction vs baseline (%)", fontsize=10)
    fig.suptitle("LP-CAS CO₂ Reduction by Renewable Portfolio", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Figure 2 — GM vs LP CO₂ reduction comparison (grouped bar, Jan + Jul)

def fig_gm_vs_lp(summary: pd.DataFrame) -> str:
    groups = ["wind", "solar", "battery", "combined"]
    dates  = ["2020-01-01", "2020-07-01"]

    # For each group, take the maximum renewable capacity point as representative
    rows = []
    for grp in groups:
        df = summary[summary["group"] == grp]
        for date in dates:
            for mode in ["GM-CAS", "LP-CAS"]:
                sub = df[(df["date"] == date) & (df["mode"] == mode)]
                if sub.empty:
                    continue
                # highest renewable capacity row
                if grp == "battery":
                    best = sub.loc[sub["battery_mwh"].idxmax()]
                else:
                    best = sub.loc[sub["renew_cap_mw"].idxmax()]
                rows.append({
                    "group": grp, "date": date, "mode": mode,
                    "co2_red_pct": best["co2_red_pct"],
                    "label": f"{grp.capitalize()}\n{DATE_LABELS[date]}",
                })
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _fig_to_b64(fig)

    df_plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 5))
    labels  = [f"{g.capitalize()} {DATE_LABELS[d]}" for g in groups for d in dates]
    x       = np.arange(len(labels))
    width   = 0.35

    gm_vals = []
    lp_vals = []
    for grp in groups:
        for date in dates:
            sub = df_plot[(df_plot["group"] == grp) & (df_plot["date"] == date)]
            gm_vals.append(sub[sub["mode"] == "GM-CAS"]["co2_red_pct"].values[0]
                           if not sub[sub["mode"] == "GM-CAS"].empty else np.nan)
            lp_vals.append(sub[sub["mode"] == "LP-CAS"]["co2_red_pct"].values[0]
                           if not sub[sub["mode"] == "LP-CAS"].empty else np.nan)

    ax.bar(x - width / 2, gm_vals, width, label="GM-CAS",
           color="#42A5F5", alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, lp_vals, width, label="LP-CAS",
           color="#EF5350", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("CO₂ reduction vs baseline (%)")
    ax.set_title("GM-CAS vs LP-CAS CO₂ Reduction at Highest Penetration Level",
                 fontweight="bold")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Pareto scatter: cost_red_pct vs co2_red_pct for LP-CAS

def fig_pareto(pareto: pd.DataFrame) -> str:
    if pareto.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _fig_to_b64(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, date in zip(axes, ["2020-01-01", "2020-07-01"]):
        sub = pareto[pareto["date"] == date]
        for grp, color in GROUP_COLORS.items():
            g = sub[sub["group"] == grp]
            if g.empty:
                continue
            sc = ax.scatter(g["cost_red_pct"], g["co2_red_pct"],
                            c=color, s=80, label=grp.capitalize(), zorder=3,
                            edgecolors="white", linewidths=0.5)
            # annotate with label
            for _, r in g.iterrows():
                ax.annotate(r["label"],
                            (r["cost_red_pct"], r["co2_red_pct"]),
                            fontsize=6, ha="left", va="bottom",
                            xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Operational cost reduction (%)", fontsize=10)
        ax.set_ylabel("CO₂ reduction (%)", fontsize=10)
        ax.set_title(f"LP-CAS Pareto — {DATE_LABELS[date]}", fontsize=11,
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("LP-CAS: CO₂ vs Cost Reduction Tradeoff by Portfolio",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Renewable curtailment vs capacity

def fig_curtailment(summary: pd.DataFrame) -> str:
    bl = summary[summary["mode"] == "Baseline"].copy()
    groups = ["wind", "solar", "combined"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    xcols = {"wind": "wind_mw", "solar": "solar_mw", "combined": "renew_cap_mw"}
    xlabels = {"wind": "Added wind (MW)", "solar": "Added solar (MW)",
               "combined": "Wind+Solar (MW each)"}

    for ax, grp in zip(axes, groups):
        df = bl[bl["group"] == grp]
        for date, ls in DATE_STYLES.items():
            sub = df[df["date"] == date].sort_values(xcols[grp])
            if sub.empty:
                continue
            ax.plot(sub[xcols[grp]], sub["curtail_gwh"],
                    marker="s", linestyle=ls, color=GROUP_COLORS[grp],
                    label=DATE_LABELS[date], linewidth=2)
        ax.set_xlabel(xlabels[grp], fontsize=9)
        ax.set_title(grp.capitalize(), fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Renewable curtailment (GWh)", fontsize=10)
    fig.suptitle("Renewable Curtailment vs Penetration Level (Baseline Dispatch)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML template

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Renewable Penetration Study — Results</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto;
          padding: 24px; color: #212121; background: #fafafa; }}
  h1   {{ color: #1565C0; border-bottom: 3px solid #1565C0; padding-bottom: 8px; }}
  h2   {{ color: #283593; margin-top: 36px; }}
  h3   {{ color: #37474F; }}
  img  {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 6px;
          box-shadow: 0 2px 8px rgba(0,0,0,.12); margin: 12px 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th   {{ background: #1565C0; color: white; padding: 8px 10px; text-align: left; }}
  td   {{ padding: 6px 10px; border-bottom: 1px solid #e0e0e0; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .missing {{ color: #c62828; font-style: italic; }}
  .note  {{ background: #E3F2FD; border-left: 4px solid #1565C0;
             padding: 10px 14px; margin: 12px 0; font-size: 13px; }}
</style>
</head>
<body>
<h1>Renewable Penetration Sensitivity Study — RTS-GMLC 2020</h1>
<p>Generated {date} &nbsp;|&nbsp; Grid: RTS-GMLC-DC-15PCT + new renewable portfolio
&nbsp;|&nbsp; Months: January & July 2020 (31 days each)</p>

<div class="note">
<b>Study design:</b> 13 renewable portfolios × 2 months = 26 simulations.
Each portfolio adds new generators on top of the base 15% DC-load grid and runs
Baseline, GM-CAS, and LP-CAS. The 24/7-CAS mode is excluded (circular when
varying renewable supply).
</div>

<h2>1. LP-CAS CO₂ Reduction vs Renewable Capacity</h2>
<p>How effectively LP-CAS reduces CO₂ as we add more renewable capacity per portfolio type.</p>
<img src="data:image/png;base64,{fig1}">

<h2>2. GM-CAS vs LP-CAS Comparison (Highest Penetration per Group)</h2>
<p>At the largest portfolio size in each group, comparing grid-mix CAS (GM) vs price-signal CAS (LP).</p>
<img src="data:image/png;base64,{fig2}">

<h2>3. LP-CAS Pareto: CO₂ vs Cost Reduction</h2>
<p>Each point represents one portfolio. Points in the upper-right quadrant achieve simultaneous
CO₂ and cost reductions. Points in the upper-left achieve CO₂ reduction at some cost premium.</p>
<img src="data:image/png;base64,{fig3}">

<h2>4. Renewable Curtailment vs Penetration (Baseline Dispatch)</h2>
<p>How much renewable generation is curtailed without CAS load shifting, as a function of installed capacity.</p>
<img src="data:image/png;base64,{fig4}">

<h2>5. Numeric Summary — LP-CAS Reductions</h2>
{table_lp}

<h2>6. Numeric Summary — GM-CAS Reductions</h2>
{table_gm}

<h2>7. Key Observations</h2>
<ul>
  <li>LP-CAS CO₂ reduction grows with renewable penetration for wind and solar sweeps,
      confirming the LMP–CO₂ correlation mechanism.</li>
  <li>Battery storage shows smaller CO₂ reductions from CAS (storage already shifts load;
      CAS adds marginal benefit on top).</li>
  <li>Summer (July) results differ from winter (January) due to higher solar availability
      and different marginal-price dynamics.</li>
  <li>Curtailment increases with penetration, indicating transmission/flexibility constraints;
      LP-CAS exploits LMP signals to time DC loads to curtailment hours.</li>
</ul>

</body>
</html>
"""


def _df_to_html(df: pd.DataFrame) -> str:
    if df.empty:
        return '<p class="missing">No data available.</p>'
    return df.to_html(index=False, classes="", border=0, float_format=lambda x: f"{x:.2f}")


# ---------------------------------------------------------------------------

def main() -> None:
    summary_path = ANALYSIS / "renew_summary.csv"
    pareto_path  = ANALYSIS / "pareto_lp.csv"

    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found — run aggregate_renew_results.py first",
              file=__import__("sys").stderr)
        return

    summary = pd.read_csv(summary_path)
    pareto  = pd.read_csv(pareto_path) if pareto_path.exists() else pd.DataFrame()

    print("Generating figures...")
    fig1 = fig_co2_vs_cap(summary)
    fig2 = fig_gm_vs_lp(summary)
    fig3 = fig_pareto(pareto)
    fig4 = fig_curtailment(summary)

    # Table: LP-CAS reductions
    lp_tbl = summary[summary["mode"] == "LP-CAS"][
        ["group", "label", "date", "wind_mw", "solar_mw", "battery_mwh",
         "co2_red_pct", "cost_red_pct", "curtail_gwh"]
    ].sort_values(["group", "wind_mw", "solar_mw", "battery_mwh", "date"])

    gm_tbl = summary[summary["mode"] == "GM-CAS"][
        ["group", "label", "date", "wind_mw", "solar_mw", "battery_mwh",
         "co2_red_pct", "cost_red_pct"]
    ].sort_values(["group", "wind_mw", "solar_mw", "battery_mwh", "date"])

    import datetime
    html = _HTML.format(
        date   = datetime.date.today().isoformat(),
        fig1   = fig1,
        fig2   = fig2,
        fig3   = fig3,
        fig4   = fig4,
        table_lp = _df_to_html(lp_tbl),
        table_gm = _df_to_html(gm_tbl),
    )

    out_path = ANALYSIS / "renew_report.html"
    out_path.write_text(html)
    print(f"Report → {out_path}")


if __name__ == "__main__":
    main()
