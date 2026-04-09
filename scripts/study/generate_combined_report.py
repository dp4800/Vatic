"""
scripts/study/generate_combined_report.py
------------------------------------------
Generate a single combined HTML report covering:
  Part 1 — CAS Study    (CAS_STUDY_RTS_2020)
  Part 2 — Renew Study  (RENEW_STUDY_RTS_2020)

Prerequisites:
    python scripts/study/aggregate_results.py
    python scripts/study/aggregate_renew_results.py

Output:
    outputs/combined_report_RTS_2020.html
"""
from __future__ import annotations
import base64, io, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
REPO        = Path(__file__).resolve().parents[2]
CAS_ANAL    = REPO / "outputs" / "CAS_STUDY_RTS_2020"  / "analysis"
RENEW_ANAL  = REPO / "outputs" / "RENEW_STUDY_RTS_2020" / "analysis"
OUT_HTML    = REPO / "outputs" / "combined_report_RTS_2020.html"

SEASON_ORDER  = ["Winter (Jan)", "Spring (Apr)", "Summer (Jul)", "Fall  (Oct)"]
MODE_COLORS   = {"LP-CAS": "#1f77b4", "GM-CAS": "#ff7f0e", "247-CAS": "#2ca02c"}
GROUP_COLORS  = {"wind": "#1f77b4", "solar": "#ff7f0e",
                 "battery": "#9467bd", "combined": "#2ca02c"}
SEASON_COLORS = {"Winter (Jan)": "#4472C4", "Spring (Apr)": "#70AD47",
                 "Summer (Jul)": "#FF0000", "Fall  (Oct)": "#FFC000"}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def img_tag(b64: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:900px;">'


def color_cell(val, good_positive=True) -> str:
    if pd.isna(val):
        return ""
    v = float(val)
    if abs(v) < 0.05:
        return ""
    color = ("#d4edda" if v > 0 else "#f8d7da") if good_positive else \
            ("#f8d7da" if v > 0 else "#d4edda")
    return f'style="background:{color}"'


def df_to_html(df: pd.DataFrame, pct_cols=None, good_positive=True) -> str:
    pct_cols = pct_cols or []
    rows = ["<table class='data-table'><thead><tr>"]
    rows += [f"<th>{c}</th>" for c in df.columns]
    rows.append("</tr></thead><tbody>")
    for _, r in df.iterrows():
        rows.append("<tr>")
        for c in df.columns:
            v = r[c]
            style, cell = "", ("" if pd.isna(v) else str(v))
            if c in pct_cols and not pd.isna(v):
                cell  = f"{float(v):+.3f}%"
                style = color_cell(v, good_positive)
            rows.append(f"<td {style}>{cell}</td>")
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def section(title: str, content: str, id_: str = "") -> str:
    id_attr = f' id="{id_}"' if id_ else ""
    return f"<h2{id_attr}>{title}</h2>\n{content}\n"


def callout(text: str, kind: str = "") -> str:
    cls = f"callout {kind}".strip()
    return f'<div class="{cls}">{text}</div>'


def metric_box(label: str, value: str) -> str:
    return (f'<div class="metric-box">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>')


# ---------------------------------------------------------------------------
# CAS figures
# ---------------------------------------------------------------------------

def fig_baseline_co2(bl: pd.DataFrame) -> str:
    co2 = bl[bl["metric"] == "Total CO2"].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    x, w = np.arange(len(SEASON_ORDER)), 0.25
    for i, mode in enumerate(["GM-CAS", "247-CAS", "LP-CAS"]):
        vals = [co2[(co2["season"] == s) & (co2["mode"] == mode)]["red_pct"].values
                for s in SEASON_ORDER]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + (i-1)*w, vals, w, label=mode, color=MODE_COLORS[mode])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(SEASON_ORDER)
    ax.set_ylabel("CO₂ Reduction (%)"); ax.legend()
    ax.set_title("Baseline: CO₂ Reduction by Season and CAS Mode")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    fig.tight_layout(); return fig_to_b64(fig)


def fig_baseline_cost(bl: pd.DataFrame) -> str:
    cost = bl[bl["metric"] == "Operational cost"].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    x, w = np.arange(len(SEASON_ORDER)), 0.25
    for i, mode in enumerate(["GM-CAS", "247-CAS", "LP-CAS"]):
        vals = [cost[(cost["season"] == s) & (cost["mode"] == mode)]["red_pct"].values
                for s in SEASON_ORDER]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + (i-1)*w, vals, w, label=mode, color=MODE_COLORS[mode])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(SEASON_ORDER)
    ax.set_ylabel("Cost Reduction (%)"); ax.legend()
    ax.set_title("Baseline: Operational Cost Reduction by Season and CAS Mode")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    fig.tight_layout(); return fig_to_b64(fig)


def fig_sensitivity(ss: pd.DataFrame, group: str, x_col: str,
                    x_label: str, title: str) -> str:
    sub = ss[ss["group"] == group].copy()
    if sub.empty:
        return ""
    dates = sorted(sub["date"].unique())
    fig, axes = plt.subplots(1, len(dates), figsize=(5*len(dates), 4),
                             sharey=True, squeeze=False)
    for j, date in enumerate(dates):
        ax = axes[0][j]
        season = "Winter" if "01-01" in date else "Summer"
        d = sub[sub["date"] == date]
        for mode in ["GM-CAS", "247-CAS", "LP-CAS"]:
            dm = d[d["mode"] == mode].sort_values(x_col)
            if dm.empty: continue
            ax.plot(dm[x_col], dm["co2_red_pct"], "o-",
                    color=MODE_COLORS[mode], label=mode, linewidth=1.8)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_title(f"{season} ({date})"); ax.set_xlabel(x_label)
        if j == 0: ax.set_ylabel("CO₂ Reduction (%)")
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(); return fig_to_b64(fig)


def fig_heatmap(ss: pd.DataFrame) -> str:
    lp = ss[ss["mode"] == "LP-CAS"].copy()
    lp["label_str"] = lp["group"] + "/" + lp["param"] + "/" + lp["date"].str[5:7]
    fig, ax = plt.subplots(figsize=(5, max(4, len(lp)*0.35 + 1)))
    colors = [("#2ca02c" if v >= 1 else "#1f77b4" if v >= 0 else "#d62728")
              for v in lp["co2_red_pct"].fillna(0)]
    bars = ax.barh(lp["label_str"], lp["co2_red_pct"].fillna(0), color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LP-CAS CO₂ Reduction (%)")
    ax.set_title("Sensitivity Overview — LP-CAS CO₂ Reduction")
    for bar, val in zip(bars, lp["co2_red_pct"].fillna(0)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:+.2f}%", va="center", fontsize=7)
    fig.tight_layout(); return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Renew figures
# ---------------------------------------------------------------------------

def fig_renew_co2_by_group(rs: pd.DataFrame) -> str:
    lp = rs[rs["mode"] == "LP-CAS"].copy()
    groups = ["wind", "solar", "battery", "combined"]
    dates  = sorted(lp["date"].unique())
    fig, axes = plt.subplots(1, len(dates), figsize=(6*len(dates), 5),
                             sharey=True, squeeze=False)
    for j, date in enumerate(dates):
        ax = axes[0][j]
        season = "Winter" if "01-01" in date else "Summer"
        d = lp[lp["date"] == date]
        for grp in groups:
            g = d[d["group"] == grp].copy()
            if g.empty: continue
            cap = g["renew_cap_mw"].where(g["renew_cap_mw"] > 0, g["battery_mwh"])
            g = g.copy(); g["cap"] = cap
            g = g.sort_values("cap")
            ax.plot(g["cap"], g["co2_red_pct"], "o-",
                    color=GROUP_COLORS.get(grp, "gray"), label=grp.capitalize(),
                    linewidth=1.8)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_title(f"{season} ({date})")
        ax.set_xlabel("Renewable/Battery Capacity (MW or MWh)")
        if j == 0: ax.set_ylabel("LP-CAS CO₂ Reduction (%)")
        ax.legend(fontsize=8)
    fig.suptitle("Renewable Infrastructure: LP-CAS CO₂ Reduction vs Capacity",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); return fig_to_b64(fig)


def fig_renew_pareto(pareto: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    dates = sorted(pareto["date"].unique())
    for j, (ax, date) in enumerate(zip(axes, dates)):
        season = "Winter" if "01-01" in date else "Summer"
        d = pareto[pareto["date"] == date]
        for grp in ["wind", "solar", "battery", "combined"]:
            g = d[d["group"] == grp]
            if g.empty: continue
            ax.scatter(g["cost_red_pct"], g["co2_red_pct"],
                       color=GROUP_COLORS.get(grp, "gray"), s=80,
                       label=grp.capitalize(), zorder=3)
            for _, row in g.iterrows():
                ax.annotate(row["label"],
                            (row["cost_red_pct"], row["co2_red_pct"]),
                            fontsize=6, textcoords="offset points", xytext=(4, 2))
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Cost Reduction (%)"); ax.set_title(f"{season} ({date})")
        if j == 0: ax.set_ylabel("CO₂ Reduction (%)")
        ax.legend(fontsize=8)
    fig.suptitle("Pareto Frontier: LP-CAS CO₂ vs Cost Reduction by Portfolio",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); return fig_to_b64(fig)


def fig_renew_vs_cas_baseline(rs: pd.DataFrame, cas_bl: pd.DataFrame) -> str:
    """Bar chart: baseline CAS (15PCT) vs best renew portfolio per season."""
    lp_renew = rs[rs["mode"] == "LP-CAS"].copy()
    dates = sorted(lp_renew["date"].unique())
    season_map = {"2020-01-01": "Winter (Jan)", "2020-07-01": "Summer (Jul)"}

    cas_lp = cas_bl[(cas_bl["metric"] == "Total CO2") & (cas_bl["mode"] == "LP-CAS")]

    groups = ["Baseline CAS\n(15PCT DC)", "Best Wind", "Best Solar",
              "Best Battery", "Best Combined"]
    colors = ["#7f7f7f", GROUP_COLORS["wind"], GROUP_COLORS["solar"],
              GROUP_COLORS["battery"], GROUP_COLORS["combined"]]

    fig, axes = plt.subplots(1, len(dates), figsize=(7*len(dates), 5),
                             sharey=True, squeeze=False)
    for j, (ax, date) in enumerate(zip(axes[0], dates)):
        season = season_map.get(date, date)
        d_renew = lp_renew[lp_renew["date"] == date]

        # Baseline CAS value from cas_baseline_summary
        cas_row = cas_lp[cas_lp["season"] == season]
        cas_val = float(cas_row["red_pct"].values[0]) if not cas_row.empty else 0.0

        vals = [cas_val]
        for grp in ["wind", "solar", "battery", "combined"]:
            g = d_renew[d_renew["group"] == grp]
            vals.append(float(g["co2_red_pct"].max()) if not g.empty else np.nan)

        bars = ax.bar(groups, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(cas_val, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel("LP-CAS CO₂ Reduction (%)") if j == 0 else None
        ax.set_title(f"{season} ({date})")
        ax.tick_params(axis="x", labelsize=8)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                        f"{val:.2f}%", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Best Portfolio per Group vs Baseline CAS (LP-CAS CO₂ Reduction)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
body { font-family: Arial, sans-serif; max-width: 1150px; margin: 40px auto;
       padding: 0 20px; color: #333; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 6px;
     margin-top: 40px; }
h3 { color: #555; }
.part-header { background: #2c3e50; color: white; padding: 10px 20px;
               border-radius: 6px; margin: 40px 0 20px 0; font-size: 1.3em;
               font-weight: bold; }
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;
               margin: 20px 0; }
.metric-box { background: #f0f4f8; border-left: 4px solid #3498db;
              border-radius: 4px; padding: 12px 16px; }
.metric-box .label { font-size: 0.8em; color: #888; text-transform: uppercase; }
.metric-box .value { font-size: 1.8em; font-weight: bold; color: #2c3e50; }
.callout { background: #eaf4fb; border-left: 4px solid #3498db;
           padding: 12px 16px; margin: 16px 0; border-radius: 4px; }
.warning { background: #fef9e7; border-left: 4px solid #f39c12; }
.finding { background: #eafaf1; border-left: 4px solid #27ae60; }
table.data-table { border-collapse: collapse; width: 100%; font-size: 0.85em;
                   margin: 12px 0; }
table.data-table th { background: #2c3e50; color: white; padding: 6px 10px;
                      text-align: left; }
table.data-table td { padding: 5px 10px; border-bottom: 1px solid #eee; }
table.data-table tr:hover { background: #f5f5f5; }
img { border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
nav { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;
      padding: 16px 24px; margin-bottom: 30px; }
nav ul { margin: 0; padding-left: 20px; }
nav li { margin: 4px 0; }
nav a { color: #3498db; text-decoration: none; }
nav a:hover { text-decoration: underline; }
"""


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_html(cas_bl, cas_ss, cas_wide, renew_rs, renew_pareto) -> str:
    parts = [f"<html><head><meta charset='utf-8'>"
             f"<title>RTS-GMLC 2020 Combined Study Report</title>"
             f"<style>{CSS}</style></head><body>"]

    parts.append("<h1>RTS-GMLC 2020 Combined Study Report</h1>")
    parts.append("<p>Carbon-Aware Scheduling (CAS) + Renewable Infrastructure sensitivity "
                 "on the RTS-GMLC grid, 2020 data. 15% baseline DC load (571 MW). "
                 "28 CAS runs + 26 renewable infrastructure runs.</p>")

    # TOC
    parts.append("""<nav><strong>Contents</strong><ul>
      <li><a href="#part1">Part 1 — CAS Study</a><ul>
        <li><a href="#baseline">1.1 Seasonal Baseline</a></li>
        <li><a href="#flex">1.2 Sensitivity: Flexibility</a></li>
        <li><a href="#alpha">1.3 Sensitivity: LP Alpha</a></li>
        <li><a href="#deferral">1.4 Sensitivity: Deferral Window</a></li>
        <li><a href="#load_growth">1.5 Sensitivity: Load Growth</a></li>
        <li><a href="#heatmap">1.6 Sensitivity Overview</a></li>
      </ul></li>
      <li><a href="#part2">Part 2 — Renewable Infrastructure Study</a><ul>
        <li><a href="#renew_co2">2.1 CO₂ Reduction by Portfolio</a></li>
        <li><a href="#renew_pareto">2.2 Cost–CO₂ Pareto Frontier</a></li>
        <li><a href="#renew_vs_cas">2.3 Best Portfolios vs Baseline CAS</a></li>
        <li><a href="#renew_table">2.4 Full Results Table</a></li>
      </ul></li>
      <li><a href="#limitations">Limitations</a></li>
    </ul></nav>""")

    # ── PART 1: CAS ──────────────────────────────────────────────────────────
    parts.append('<div class="part-header" id="part1">Part 1 — Carbon-Aware Scheduling Study</div>')

    # Key metrics
    bl_lp    = cas_bl[(cas_bl["metric"] == "Total CO2") & (cas_bl["mode"] == "LP-CAS")]
    best_co2 = bl_lp["red_pct"].max()
    best_ssn = bl_lp.loc[bl_lp["red_pct"].idxmax(), "season"] if not bl_lp.empty else "—"
    avg_co2  = bl_lp["red_pct"].mean()
    sens_lp  = cas_ss[cas_ss["mode"] == "LP-CAS"]
    best_sens = sens_lp["co2_red_pct"].max()

    parts.append('<div class="metric-grid">')
    parts.append(metric_box("Best Baseline LP-CAS CO₂ Red.", f"{best_co2:.2f}%"))
    parts.append(metric_box("Best Season", best_ssn))
    parts.append(metric_box("Avg Baseline LP-CAS CO₂ Red.", f"{avg_co2:.2f}%"))
    parts.append(metric_box("Best Sensitivity LP-CAS CO₂ Red.", f"{best_sens:.2f}%"))
    parts.append(metric_box("Total CAS Runs", str(len(cas_wide))))
    parts.append(metric_box("DC Load Added", "571 MW (15% avg demand)"))
    parts.append("</div>")

    # 1.1 Baseline
    bl_table = cas_bl[cas_bl["metric"] == "Total CO2"].pivot_table(
        index="season", columns="mode", values="red_pct"
    ).reindex(SEASON_ORDER).round(3)
    baseline_html = (
        '<div class="two-col">'
        + img_tag(fig_baseline_co2(cas_bl)) + img_tag(fig_baseline_cost(cas_bl))
        + "</div>"
        + callout("<strong>Key finding:</strong> Winter (Jan) shows the largest CO₂ "
                  "reductions across all CAS modes (~1.7–2.2% LP-CAS). Summer (Jul) "
                  "reductions are near zero due to low LMP variance (σ≈3.7 $/MWh).", "finding")
        + callout("<strong>Spring anomaly:</strong> April 247-CAS shows −0.64%. "
                  "High load-shedding in April (MILP stochasticity from 571 MW DC injection) "
                  "contaminates the comparison.", "warning")
        + "<h3>CO₂ Reduction (%) by Season and Mode</h3>"
        + df_to_html(bl_table.reset_index(), pct_cols=["GM-CAS","247-CAS","LP-CAS"])
    )
    parts.append(section("1.1 Seasonal Baseline Results", baseline_html, "baseline"))

    # 1.2–1.5 Sensitivity
    for id_, title, group, x_col, x_label, finding in [
        ("flex",   "1.2 Sensitivity: Flexibility Budget",      "flex",     "flex_pct",
         "Flexibility (%)",
         "CO₂ reduction increases monotonically with flexibility in January "
         "(20%→50%: 1.2→2.8%). Summer is near-zero at all levels."),
        ("alpha",  "1.3 Sensitivity: LP Alpha (Cost/CO₂ Weight)", "alpha", "lp_alpha",
         "LP Alpha (CO₂ weight)",
         "LP-CAS CO₂ reduction is flat across α=0–1 in January (~1.8–2.1%), "
         "confirming LMP is a reasonable CO₂ proxy. In July, α=1.0 gives slightly "
         "negative reductions when renewables dominate pricing."),
        ("deferral","1.4 Sensitivity: Deferral Window",          "deferral","deferral_h",
         "Deferral Window (hours)",
         "Completely flat across 4–24h in January (~2.1%). The LP saturates at "
         "~8h shifts naturally; wider windows grant flexibility the LP does not exploit."),
    ]:
        fig_b64 = fig_sensitivity(cas_ss, group, x_col, x_label,
                                  f"Sensitivity: {x_label} → CO₂ Reduction")
        parts.append(section(title,
            img_tag(fig_b64) + callout(f"<strong>Finding:</strong> {finding}", "finding"),
            id_))

    # 1.5 Load growth
    lg = cas_ss[cas_ss["group"] == "load_growth"]
    lg_val = (f"{lg[lg['mode']=='LP-CAS']['co2_red_pct'].values[0]:.2f}%"
              if not lg.empty else "N/A")
    parts.append(section("1.5 Sensitivity: Load Growth",
        callout(f"<strong>25% DC load growth (amp_25pct):</strong> LP-CAS CO₂ reduction "
                f"Jan = {lg_val} vs 2.21% at 15% DC load. Larger DC shifts more load but "
                "tightens reserve margins. July run completed (0.11% reduction).", "finding"),
        "load_growth"))

    # 1.6 Heatmap
    parts.append(section("1.6 Sensitivity Overview", img_tag(fig_heatmap(cas_ss), "65%"), "heatmap"))

    # ── PART 2: RENEW ─────────────────────────────────────────────────────────
    parts.append('<div class="part-header" id="part2">Part 2 — Renewable Infrastructure Study</div>')

    lp_renew = renew_rs[renew_rs["mode"] == "LP-CAS"]
    best_wind   = lp_renew[lp_renew["group"]=="wind"]["co2_red_pct"].max()
    best_solar  = lp_renew[lp_renew["group"]=="solar"]["co2_red_pct"].max()
    best_batt   = lp_renew[lp_renew["group"]=="battery"]["co2_red_pct"].max()
    best_comb   = lp_renew[lp_renew["group"]=="combined"]["co2_red_pct"].max()

    parts.append('<div class="metric-grid">')
    parts.append(metric_box("Best Wind LP-CAS CO₂ Red.", f"{best_wind:.2f}%"))
    parts.append(metric_box("Best Solar LP-CAS CO₂ Red.", f"{best_solar:.2f}%"))
    parts.append(metric_box("Best Battery LP-CAS CO₂ Red.", f"{best_batt:.2f}%"))
    parts.append(metric_box("Best Combined LP-CAS CO₂ Red.", f"{best_comb:.2f}%"))
    parts.append(metric_box("Portfolios Tested", "17 (6 wind, 6 solar, 3 battery, 2 combined)"))
    parts.append(metric_box("Seasons", "Winter (Jan) + Summer (Jul)"))
    parts.append("</div>")

    # 2.1 CO₂ by portfolio
    co2_fig = fig_renew_co2_by_group(renew_rs)
    parts.append(section("2.1 CO₂ Reduction by Portfolio Type",
        img_tag(co2_fig)
        + callout("<strong>Finding:</strong> Wind portfolios show the strongest LP-CAS "
                  "CO₂ reduction in January (W3000: 3.53%). Solar is effective at moderate "
                  "capacity (S1000: 2.89%) but plateaus at higher levels. Battery "
                  "performance mirrors the baseline CAS (B500: 2.35%), confirming storage "
                  "primarily enables temporal shifting already captured by LP-CAS. "
                  "Summer reductions are near zero across all types, consistent with "
                  "the baseline CAS seasonality.", "finding"),
        "renew_co2"))

    # 2.2 Pareto
    pareto_fig = fig_renew_pareto(renew_pareto)
    parts.append(section("2.2 Cost–CO₂ Pareto Frontier",
        img_tag(pareto_fig)
        + callout("<strong>Finding:</strong> In January most portfolios improve both "
                  "cost and CO₂ (upper-right quadrant). W3000 dominates on CO₂; "
                  "combined W3000S3000 achieves the best cost reduction alongside "
                  "strong CO₂ reductions. In July nearly all portfolios cluster near "
                  "zero, with some combined portfolios slightly worsening CO₂ "
                  "(renewable curtailment effect).", "finding"),
        "renew_pareto"))

    # 2.3 vs CAS baseline
    vs_fig = fig_renew_vs_cas_baseline(renew_rs, cas_bl)
    parts.append(section("2.3 Best Portfolios vs Baseline CAS",
        img_tag(vs_fig)
        + callout("<strong>Key finding:</strong> Adding renewable infrastructure "
                  "significantly amplifies LP-CAS CO₂ reduction beyond the baseline "
                  "15% DC scenario. W3000 (3.53%) is 60% better than baseline CAS "
                  "(2.21%) in January. This confirms that CAS and renewable build-out "
                  "are complementary strategies.", "finding"),
        "renew_vs_cas"))

    # 2.4 Table
    lp_table = lp_renew[["group","label","date","wind_mw","solar_mw",
                          "battery_mwh","co2_red_pct","cost_red_pct","curtail_gwh"]].copy()
    lp_table = lp_table.sort_values(["date","group","label"]).round(3)
    lp_table.columns = ["Group","Portfolio","Date","Wind MW","Solar MW","Battery MWh",
                        "CO₂ Red %","Cost Red %","Curtail GWh"]
    parts.append(section("2.4 Full LP-CAS Results Table",
        df_to_html(lp_table, pct_cols=["CO₂ Red %","Cost Red %"]),
        "renew_table"))

    # ── Limitations ──────────────────────────────────────────────────────────
    parts.append(section("Limitations & Caveats", """
    <ul>
      <li><strong>MILP stochasticity (~0.14% CO₂ variance):</strong> Gurobi branching
      introduces run-to-run variance comparable to effect sizes. Patterns across
      seasons and groups are defensible; individual-run magnitudes are not.</li>
      <li><strong>Load shedding artefact:</strong> The 571 MW DC injection triggers
      load shedding in some hours (esp. April), slightly inflating CO₂ reductions
      independent of CAS.</li>
      <li><strong>LMP ≠ CO₂ proxy in high-renewable hours:</strong> When renewables
      curtail (low/negative LMP), LP-CAS may shift load into curtailment hours,
      increasing CO₂. Summer α=1.0 and some combined portfolios show this effect.</li>
      <li><strong>Small DC relative to grid:</strong> 571 MW ≈ 15% of ~3,809 MW
      average demand. DC load is large enough to affect results but small relative
      to thermal commitment decisions.</li>
      <li><strong>Renewable portfolios are additive overlays:</strong> New wind/solar
      is added on top of existing RTS-GMLC capacity; no retirement or grid upgrade
      modelled.</li>
    </ul>
    """, "limitations"))

    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------

def main() -> None:
    cas_bl   = pd.read_csv(CAS_ANAL   / "baseline_summary.csv")
    cas_ss   = pd.read_csv(CAS_ANAL   / "sensitivity_summary.csv")
    cas_wide = pd.read_csv(CAS_ANAL   / "results_wide.csv")
    renew_rs = pd.read_csv(RENEW_ANAL / "renew_summary.csv")
    pareto   = pd.read_csv(RENEW_ANAL / "pareto_lp.csv")

    html = build_html(cas_bl, cas_ss, cas_wide, renew_rs, pareto)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Report written → {OUT_HTML}")


if __name__ == "__main__":
    main()
