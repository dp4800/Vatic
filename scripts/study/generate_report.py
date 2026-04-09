"""
scripts/study/generate_report.py
----------------------------------
Generate a self-contained HTML report for CAS_STUDY_RTS_2020.

Run aggregate_results.py first, then:
    module load anaconda3/2024.10
    python3 scripts/study/generate_report.py

Output: outputs/CAS_STUDY_RTS_2020/analysis/report.html
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
REPO     = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "outputs" / "CAS_STUDY_RTS_2020" / "analysis"
OUT_HTML = ANALYSIS / "report.html"

SEASON_ORDER = ["Winter (Jan)", "Spring (Apr)", "Summer (Jul)", "Fall  (Oct)"]
MODE_COLORS  = {"LP-CAS": "#1f77b4", "GM-CAS": "#ff7f0e", "247-CAS": "#2ca02c"}
GROUP_ORDER  = ["load_growth", "flex", "alpha", "deferral"]
GROUP_LABELS = {"load_growth": "Load Growth", "flex": "Flexibility",
                "alpha": "LP α (cost/CO₂ weight)", "deferral": "Deferral Window"}
SEASON_COLORS = {"Winter (Jan)": "#4472C4", "Spring (Apr)": "#70AD47",
                 "Summer (Jul)": "#FF0000", "Fall  (Oct)": "#FFC000"}

# ---------------------------------------------------------------------------
# Helpers
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
    """Return inline style for red/green cell colouring."""
    if pd.isna(val):
        return ""
    v = float(val)
    if abs(v) < 0.05:
        return ""
    color = ("#d4edda" if v > 0 else "#f8d7da") if good_positive else \
            ("#f8d7da" if v > 0 else "#d4edda")
    return f'style="background:{color}"'


def df_to_html(df: pd.DataFrame, pct_cols=None, highlight_col=None,
               good_positive=True) -> str:
    pct_cols = pct_cols or []
    rows = ["<table class='data-table'><thead><tr>"]
    rows += [f"<th>{c}</th>" for c in df.columns]
    rows.append("</tr></thead><tbody>")
    for _, r in df.iterrows():
        rows.append("<tr>")
        for c in df.columns:
            v = r[c]
            style = ""
            cell = "" if pd.isna(v) else str(v)
            if c in pct_cols and not pd.isna(v):
                cell = f"{float(v):+.3f}%"
                style = color_cell(v, good_positive)
            rows.append(f"<td {style}>{cell}</td>")
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_baseline_co2(bl: pd.DataFrame) -> str:
    co2 = bl[bl["metric"] == "Total CO2"].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(SEASON_ORDER))
    w = 0.25
    for i, mode in enumerate(["GM-CAS", "247-CAS", "LP-CAS"]):
        vals = [co2[(co2["season"] == s) & (co2["mode"] == mode)]["red_pct"].values
                for s in SEASON_ORDER]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + (i-1)*w, vals, w, label=mode, color=MODE_COLORS[mode])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(SEASON_ORDER)
    ax.set_ylabel("CO₂ Reduction (%)")
    ax.set_title("Baseline: CO₂ Reduction by Season and CAS Mode")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_baseline_cost(bl: pd.DataFrame) -> str:
    cost = bl[bl["metric"] == "Operational cost"].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(SEASON_ORDER))
    w = 0.25
    for i, mode in enumerate(["GM-CAS", "247-CAS", "LP-CAS"]):
        vals = [cost[(cost["season"] == s) & (cost["mode"] == mode)]["red_pct"].values
                for s in SEASON_ORDER]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + (i-1)*w, vals, w, label=mode, color=MODE_COLORS[mode])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(SEASON_ORDER)
    ax.set_ylabel("Cost Reduction (%)")
    ax.set_title("Baseline: Operational Cost Reduction by Season and CAS Mode")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    fig.tight_layout()
    return fig_to_b64(fig)


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
            if dm.empty:
                continue
            ax.plot(dm[x_col], dm["co2_red_pct"], "o-",
                    color=MODE_COLORS[mode], label=mode, linewidth=1.8)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_title(f"{season} ({date})")
        ax.set_xlabel(x_label)
        if j == 0:
            ax.set_ylabel("CO₂ Reduction (%)")
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_heatmap(ss: pd.DataFrame) -> str:
    lp = ss[ss["mode"] == "LP-CAS"].copy()
    lp["label"] = lp["group"] + "/" + lp["param"] + "/" + lp["date"].str[5:7]

    fig, ax = plt.subplots(figsize=(5, max(4, len(lp)*0.35 + 1)))
    colors = [("#2ca02c" if v >= 1 else "#1f77b4" if v >= 0 else "#d62728")
              for v in lp["co2_red_pct"].fillna(0)]
    bars = ax.barh(lp["label"], lp["co2_red_pct"].fillna(0), color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LP-CAS CO₂ Reduction (%)")
    ax.set_title("Sensitivity Overview — LP-CAS CO₂ Reduction")
    for bar, val in zip(bars, lp["co2_red_pct"].fillna(0)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:+.2f}%", va="center", fontsize=7)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CSS = """
body { font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto;
       padding: 0 20px; color: #333; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 6px;
     margin-top: 40px; }
h3 { color: #555; }
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
"""

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


def build_html(bl: pd.DataFrame, ss: pd.DataFrame, wide: pd.DataFrame) -> str:
    parts = [f"<html><head><meta charset='utf-8'>"
             f"<title>CAS Study — RTS-GMLC 2020</title>"
             f"<style>{CSS}</style></head><body>"]
    parts.append("<h1>Carbon-Aware Scheduling Study — RTS-GMLC 2020</h1>")
    parts.append("<p>15% DC load growth baseline (571 MW across 10 buses). "
                 "4 baseline months (Jan/Apr/Jul/Oct). "
                 "Sensitivity: Jan + Jul for flex, alpha, deferral, load growth groups.</p>")

    # ── Key metrics ──────────────────────────────────────────────────────────
    bl_lp = bl[(bl["metric"] == "Total CO2") & (bl["mode"] == "LP-CAS")]
    best_co2 = bl_lp["red_pct"].max()
    best_season = bl_lp.loc[bl_lp["red_pct"].idxmax(), "season"] if not bl_lp.empty else "—"
    avg_co2 = bl_lp["red_pct"].mean()

    sens_lp = ss[ss["mode"] == "LP-CAS"]
    best_sens = sens_lp["co2_red_pct"].max()

    parts.append('<div class="metric-grid">')
    parts.append(metric_box("Best Baseline LP-CAS CO₂ Reduction", f"{best_co2:.2f}%"))
    parts.append(metric_box("Best Season", best_season))
    parts.append(metric_box("Avg Baseline LP-CAS CO₂ Red.", f"{avg_co2:.2f}%"))
    parts.append(metric_box("Best Sensitivity LP-CAS CO₂ Red.", f"{best_sens:.2f}%"))
    n_runs = len(wide)
    parts.append(metric_box("Total Runs", str(n_runs)))
    parts.append(metric_box("DC Load Added", "571 MW (15% of avg demand)"))
    parts.append("</div>")

    # ── Baseline ─────────────────────────────────────────────────────────────
    bl_co2_fig  = fig_baseline_co2(bl)
    bl_cost_fig = fig_baseline_cost(bl)
    bl_table = bl[(bl["metric"] == "Total CO2")].pivot_table(
        index="season", columns="mode", values="red_pct"
    ).reindex(SEASON_ORDER).round(3)

    baseline_html = (
        '<div class="two-col">'
        + img_tag(bl_co2_fig) + img_tag(bl_cost_fig)
        + "</div>"
        + callout("<strong>Key finding:</strong> Winter (Jan) consistently shows the "
                  "largest CO₂ reductions across all CAS modes (~1.7–2.2% for LP-CAS). "
                  "Summer (Jul) reductions are near zero, consistent with lower LMP "
                  "variance (σ≈3.7 $/MWh) reducing the LP's ability to exploit temporal "
                  "price signals.", "finding")
        + callout("<strong>Spring anomaly:</strong> April 247-CAS shows −0.64% "
                  "(CO₂ increases). High load-shedding in April (MILP stochasticity "
                  "artefact from the 571 MW DC injection) contaminates the comparison. "
                  "See limitations section.", "warning")
        + "<h3>CO₂ Reduction (%) by Season and Mode</h3>"
        + df_to_html(bl_table.reset_index(),
                     pct_cols=["GM-CAS", "247-CAS", "LP-CAS"])
    )
    parts.append(section("1. Seasonal Baseline Results", baseline_html, "baseline"))

    # ── Sensitivity ───────────────────────────────────────────────────────────

    # Flex
    flex_fig = fig_sensitivity(
        ss, "flex", "flex_pct", "Flexibility (%)",
        "Sensitivity: Flexibility Budget → LP-CAS CO₂ Reduction"
    )
    flex_html = (
        img_tag(flex_fig)
        + callout("<strong>Finding:</strong> CO₂ reduction increases monotonically with "
                  "flexibility in January (20%→50% flex: 1.2→2.8%). In July, all flex "
                  "values produce near-zero reductions, confirming the seasonal LMP "
                  "variance dependency.", "finding")
    )
    parts.append(section("2. Sensitivity: Flexibility Budget", flex_html, "flex"))

    # Alpha
    alpha_fig = fig_sensitivity(
        ss, "alpha", "lp_alpha", "LP Alpha (CO₂ weight)",
        "Sensitivity: LP α → CO₂ Reduction"
    )
    alpha_html = (
        img_tag(alpha_fig)
        + callout("<strong>Finding:</strong> LP-CAS CO₂ reduction is relatively flat "
                  "across α (0–1) in January (~1.8–2.1%), confirming that LMP already "
                  "serves as a reasonable CO₂ proxy when fossil generators set prices. "
                  "In July, α=1.0 (pure CO₂ minimization) gives slightly negative "
                  "reductions, suggesting price and CI decorrelate when renewables "
                  "dominate.", "finding")
        + callout("<strong>Note:</strong> GM-CAS and 247-CAS lines are constant across "
                  "α — these modes don't use the alpha parameter. They serve as "
                  "reference baselines.", "callout")
    )
    parts.append(section("3. Sensitivity: LP Alpha (Cost/CO₂ Weighting)", alpha_html, "alpha"))

    # Deferral
    def_fig = fig_sensitivity(
        ss, "deferral", "deferral_h", "Deferral Window (hours)",
        "Sensitivity: Deferral Window → CO₂ Reduction"
    )
    def_html = (
        img_tag(def_fig)
        + callout("<strong>Finding:</strong> LP-CAS CO₂ reduction is flat across "
                  "deferral windows of 4–24h in January (~2.1%). The LP naturally "
                  "saturates at ~8h shifts based on the LMP signal; wider windows "
                  "grant more flexibility but the LP does not exploit it. This mirrors "
                  "the finding from the old sensitivity study.", "finding")
    )
    parts.append(section("4. Sensitivity: Deferral Window", def_html, "deferral"))

    # Load growth
    lg_data = ss[ss["group"] == "load_growth"]
    lg_html = callout(
        "<strong>Note:</strong> Only January load_growth/amp_25pct completed "
        "(July run lost to disk quota). LP-CAS CO₂ reduction at 25% DC load "
        f"(95.2 MW/bus): "
        + (f"{lg_data[lg_data['mode']=='LP-CAS']['co2_red_pct'].values[0]:.2f}%" if not lg_data.empty else "N/A")
        + " vs 2.21% at 15%. Larger DC load gives the LP more load to shift but "
        "also tightens reserve margins.", "warning"
    )
    parts.append(section("5. Sensitivity: Load Growth", lg_html, "load_growth"))

    # Overview heatmap
    hm_fig = fig_heatmap(ss)
    hm_html = img_tag(hm_fig, "70%")
    parts.append(section("6. Sensitivity Overview Heatmap", hm_html, "heatmap"))

    # ── Limitations ────────────────────────────────────────────────────────
    lim_html = """
    <ul>
      <li><strong>MILP stochasticity (~0.14% CO₂ variance):</strong> RUC/SCED are
      MIP problems; Gurobi branching introduces run-to-run variance comparable to
      the effect sizes being measured. Patterns across seasons are defensible;
      individual-run magnitudes are not.</li>
      <li><strong>Load shedding artefact:</strong> The 571 MW DC injection (15% of
      avg demand) is large enough to trigger load shedding in some hours, particularly
      April. Shed load reduces demand and hence CO₂ independently of CAS, slightly
      inflating measured reductions.</li>
      <li><strong>LMP ≠ CO₂ proxy in high-renewable hours:</strong> LP-CAS optimises
      on price. When renewables curtail (low or negative LMP), LP may shift load into
      curtailment hours, increasing CO₂. July α=1.0 shows this effect.</li>
      <li><strong>Missing runs:</strong> load_growth/amp_25pct/2020-07 lost to disk
      quota; 5 July sensitivity LP-i1 simulations used i0 fallback (single LP
      iteration instead of two).</li>
      <li><strong>Small DC relative to grid:</strong> 571 MW ≈ 15% of ~3,809 MW
      average demand. DC load is large enough to affect results but still small
      relative to thermal commitment decisions.</li>
    </ul>
    """
    parts.append(section("7. Limitations & Caveats", lim_html, "limitations"))

    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------

def main() -> None:
    bl = pd.read_csv(ANALYSIS / "baseline_summary.csv")
    ss = pd.read_csv(ANALYSIS / "sensitivity_summary.csv")
    wide = pd.read_csv(ANALYSIS / "results_wide.csv")

    html = build_html(bl, ss, wide)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Report written → {OUT_HTML}")


if __name__ == "__main__":
    main()
