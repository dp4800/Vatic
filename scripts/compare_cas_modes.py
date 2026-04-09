#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
compare_cas_modes.py — Compare CAS analysis results across modes and configurations.

Two input groups can be combined freely:

  Group A — CAS analysis directories (from analyze_cas.py):
    --cas-dirs  DIR [DIR ...]   paired with
    --cas-labels LABEL [LABEL ...]
    Each directory must contain a cas_results.csv.
    Mode (grid-mix / 24_7 / lp) is auto-detected from the CSV columns.

  Group B — Full VATIC simulation directories (optional):
    --sim-dirs  DIR [DIR ...]   paired with
    --sim-labels LABEL [LABEL ...]
    --gen-csv   PATH
    Produces system-level comparison plots (LMP, dispatch, CO₂, water).

Outputs (all written to --out-dir):
  cas_mode_summary.csv          — best metrics per CAS configuration
  compare_best_metrics.png      — bar chart of best carbon/coverage per config
  system_comparison.csv         — full Δ table (Group B runs vs baseline)
  compare_timeseries.png        — 7-panel time series (all scenarios)
  compare_dispatch_by_fuel.png  — grouped bar: total dispatch per fuel × scenario
  compare_renew_coverage.png    — RE availability & coverage (time series, diurnal, summary)
  dispatch_<label>.png          — hourly stacked dispatch per scenario
  compare_key_metrics.png       — key metrics comparison (horizontal bars)
  policy_figure1_<label>.png    — Ackon (2025) Fig 1: Shapley allocations + net exports

Usage
-----
  # Compare CAS analytical results from three different modes
  python scripts/compare_cas_modes.py \\
      --cas-dirs   outputs/cas_2020-05-04 \\
                   outputs/cas_247_frac10_2020-05-04 \\
                   outputs/cas_lp_2020-05-04 \\
      --cas-labels grid-mix 24_7 lp \\
      --out-dir    outputs/cas_comparison/2020-05-04

  # Full pipeline: CAS analysis + VATIC re-run comparison + Policy Figure 1
  python scripts/compare_cas_modes.py \\
      --cas-dirs   outputs/cas_2020-05-04 outputs/cas_lp_2020-05-04 \\
      --cas-labels grid-mix lp \\
      --sim-dirs   outputs/baseline outputs/sim-gm outputs/sim-247 outputs/sim-lp \\
      --sim-labels baseline sim-gm sim-247 sim-lp \\
      --gen-csv    vatic/data/grids/RTS-GMLC-DC/RTS_Data/SourceData/gen.csv \\
      --water-dir  outputs/water \\
      --env-score-dir outputs/environmental_score \\
      --out-dir    outputs/cas_comparison/2020-05-04
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
import compare_sim_outputs as cso      # noqa: E402
import reliability                     # noqa: E402

_VATIC_ROOT = _SCRIPTS_DIR.parent
_COLORS = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#b07aa1",
           "#76b7b2", "#ff9da7", "#9c755f"]


# ---------------------------------------------------------------------------
# Mode detection and summary extraction
# ---------------------------------------------------------------------------

def detect_mode(csv_path: Path) -> str:
    """Detect CAS mode from cas_results.csv column schema."""
    cols = set(pd.read_csv(csv_path, nrows=0).columns)
    if "alpha" in cols:
        return "lp"
    if "balanced_coverage" in cols:
        return "24_7"
    return "grid-mix"


def extract_summary(df: pd.DataFrame, mode: str, label: str) -> dict:
    """Return a flat dict of the most informative scalar summaries."""
    s: dict = {"label": label, "mode": mode}

    if mode == "lp":
        best_c = df.loc[df["carbon_reduction_pct"].idxmax()]
        best_s = df.loc[df["cost_reduction_pct"].idxmax()]
        mid    = df.iloc[(df["alpha"] - 0.5).abs().argsort().iloc[0]]
        s.update({
            "best_carbon_reduction_pct":    round(float(best_c["carbon_reduction_pct"]), 2),
            "best_carbon_alpha":            round(float(best_c["alpha"]), 2),
            # NOTE: dc_sched_cost_reduction_pct measures the *datacenter's*
            # internal LP scheduling cost (DC load × LMP), not the grid's total
            # operating cost.  See grid_total_cost_change_pct for the latter.
            "dc_sched_cost_reduction_pct":  round(float(best_s["cost_reduction_pct"]), 2),
            "best_cost_alpha":              round(float(best_s["alpha"]), 2),
            "mid_carbon_reduction_pct":     round(float(mid["carbon_reduction_pct"]), 2),
            "mid_dc_sched_cost_red_pct":    round(float(mid["cost_reduction_pct"]), 2),
            "mid_alpha":                    round(float(mid["alpha"]), 2),
            "grid_total_cost_change_pct":   None,  # filled after sim comparison
            "best_coverage_pct":            None,
            "best_coverage_gain_pct":       None,
            "unshifted_coverage_pct":       None,
            "best_extra_capacity":          None,
            "best_flex_ratio":              None,
        })

    elif mode == "24_7":
        best = df.loc[df["balanced_coverage"].idxmax()]
        s.update({
            "best_carbon_reduction_pct": None,
            "best_carbon_alpha":         None,
            "best_cost_reduction_pct":   None,
            "best_cost_alpha":           None,
            "mid_carbon_reduction_pct":  None,
            "mid_cost_reduction_pct":    None,
            "mid_alpha":                 None,
            "best_coverage_pct":         round(float(best["balanced_coverage"]), 2),
            "best_coverage_gain_pct":    round(float(best["coverage_gain_pct"]), 1),
            "unshifted_coverage_pct":    round(float(best["imbalanced_coverage"]), 2),
            "best_extra_capacity":       int(best["extra_capacity"]),
            "best_flex_ratio":           int(best["flexible_work_ratio"]),
        })

    else:  # grid-mix
        best = df.loc[df["carbon_reduction_pct"].idxmax()]
        s.update({
            "best_carbon_reduction_pct": round(float(best["carbon_reduction_pct"]), 2),
            "best_carbon_alpha":         None,
            "best_cost_reduction_pct":   None,
            "best_cost_alpha":           None,
            "mid_carbon_reduction_pct":  None,
            "mid_cost_reduction_pct":    None,
            "mid_alpha":                 None,
            "best_coverage_pct":         None,
            "best_coverage_gain_pct":    None,
            "unshifted_coverage_pct":    None,
            "best_extra_capacity":       int(best["extra_capacity"]),
            "best_flex_ratio":           int(best["flexible_work_ratio"]),
        })

    return s


# ---------------------------------------------------------------------------
# Numerical table helpers
# ---------------------------------------------------------------------------

def print_summary_table(summaries: list[dict]) -> None:
    df = pd.DataFrame(summaries).set_index("label")
    print(f"\n{'='*72}")
    print("  CAS MODE SUMMARY")
    print(f"{'='*72}")

    carbon_cols = ["mode", "best_carbon_reduction_pct", "best_extra_capacity",
                   "best_flex_ratio", "best_carbon_alpha"]
    sub = df[[c for c in carbon_cols if c in df.columns]].dropna(
        subset=["best_carbon_reduction_pct"], how="all"
    )
    if not sub.empty:
        print("\nCarbon reduction (best achievable):")
        print(sub.to_string())

    cov_cols = ["mode", "unshifted_coverage_pct", "best_coverage_pct",
                "best_coverage_gain_pct", "best_extra_capacity", "best_flex_ratio"]
    sub = df[[c for c in cov_cols if c in df.columns]].dropna(
        subset=["best_coverage_pct"], how="all"
    )
    if not sub.empty:
        print("\n24/7 renewable coverage (best achievable):")
        print(sub.to_string())

    lp_cols = ["mode", "best_carbon_reduction_pct", "dc_sched_cost_reduction_pct",
               "grid_total_cost_change_pct",
               "mid_carbon_reduction_pct", "mid_dc_sched_cost_red_pct"]
    sub = df[df["mode"] == "lp"][[c for c in lp_cols if c in df.columns]]
    if not sub.empty:
        print("\nLP Pareto summary (α=0 carbon-only / α=0.5 balanced / α=1 cost-only):")
        print("  (dc_sched_cost: DC load×LMP objective; grid_total_cost: full grid Δ%)")
        print(sub.to_string())


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _axis_fmt(ax, axis: str = "y"):
    """Consistent tick formatter — same decimal places for all ticks including 0."""
    import matplotlib.ticker as _mt
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    span = max(abs(hi), abs(lo), 1e-12)
    if span >= 1e6:
        return _mt.FuncFormatter(lambda v, _: f"{v/1e6:,.1f}M")
    if span >= 1e3:
        return _mt.FuncFormatter(lambda v, _: f"{v:,.0f}")
    if span >= 10:
        return _mt.FuncFormatter(lambda v, _: f"{v:.1f}")
    return _mt.FuncFormatter(lambda v, _: f"{v:.2f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_best_metrics(
    summaries: list[dict],
    out_dir: Path,
    dfs: dict[str, tuple[str, pd.DataFrame]] | None = None,
) -> Path | None:
    """
    Two-panel summary of best achievable CAS metrics.

    Left panel  — LP Pareto scatter (carbon vs cost trade-off) when LP data available.
    Right panel — Grouped horizontal bars: carbon reduction & cost reduction per config
                  (stacked alongside 24/7 coverage metrics on a secondary chart if needed).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.ticker as mticker
    except ImportError:
        return None

    lp_summaries   = [s for s in summaries if s["mode"] == "lp"]
    gm_summaries   = [s for s in summaries if s["mode"] == "grid-mix"]
    s247_summaries = [s for s in summaries if s["mode"] == "24_7"]

    has_lp        = bool(lp_summaries)
    has_carbon    = bool(lp_summaries or gm_summaries)
    has_coverage  = bool(s247_summaries)

    if not has_lp and not has_carbon and not has_coverage:
        return None

    # ── Figure layout ──────────────────────────────────────────────────────
    n_panels = int(has_lp) + int(has_carbon) + int(has_coverage)
    fig_w    = max(5, 4.5 * n_panels)
    fig_h    = max(4.0, 0.55 * len(summaries) + 2.0)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), squeeze=False)
    panel_iter = iter(axes[0])

    COLORS = {
        "carbon": "#59a14f",
        "cost":   "#4e79a7",
        "cover":  "#f28e2b",
        "cover_base": "#d6c4a0",
    }

    # ── Panel 1: LP Pareto scatter ─────────────────────────────────────────
    if has_lp:
        ax = next(panel_iter)
        palette = _COLORS  # module-level colour list

        # Draw one Pareto curve per LP config
        for i, s in enumerate(lp_summaries):
            label = s["label"]
            color = palette[i % len(palette)]

            if dfs and label in dfs:
                _, df = dfs[label]
                df_s  = df.sort_values("alpha")
                ax.plot(
                    df_s["cost_reduction_pct"],
                    df_s["carbon_reduction_pct"],
                    "-o", color=color, markersize=4, linewidth=1.5,
                    label=label, zorder=3,
                )
                # Annotate the endpoints with α values
                for row in [df_s.iloc[0], df_s.iloc[-1]]:
                    ax.annotate(
                        f"α={row['alpha']:.1f}",
                        (row["cost_reduction_pct"], row["carbon_reduction_pct"]),
                        fontsize=6.5, color=color,
                        xytext=(3, 3), textcoords="offset points",
                    )
            else:
                # Fall back to just the two extreme points from summary
                ax.scatter(
                    [s["best_cost_reduction_pct"]],
                    [s["best_carbon_reduction_pct"]],
                    color=color, s=60, zorder=3, label=label,
                )

        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.axvline(0, color="gray", lw=0.7, ls="--")
        ax.set_xlabel("Cost Reduction (%)", fontsize=9)
        ax.set_ylabel("Carbon Reduction (%)", fontsize=9)
        ax.set_title("LP Carbon–Cost Trade-off\n(Pareto front, α from 0→1)", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_formatter(_axis_fmt(ax, axis="x"))
        ax.yaxis.set_major_formatter(_axis_fmt(ax))

        # Add grid-mix points on the same axes as reference dots
        for s in gm_summaries:
            ax.scatter(
                [0], [s["best_carbon_reduction_pct"]],
                marker="D", color="gray", s=50, zorder=4,
                label=f"{s['label']} (gm)",
            )
            ax.annotate(
                s["label"], (0, s["best_carbon_reduction_pct"]),
                fontsize=6.5, color="gray",
                xytext=(4, -8), textcoords="offset points",
            )

    # ── Panel 2: Grouped bars — carbon & cost reduction ────────────────────
    if has_carbon:
        ax = next(panel_iter)
        # All configs that have carbon or cost data
        bar_configs = [
            s for s in summaries
            if s.get("best_carbon_reduction_pct") is not None
            or s.get("best_cost_reduction_pct") is not None
        ]
        labels  = [s["label"] for s in bar_configs]
        carbons = [s.get("best_carbon_reduction_pct") or 0.0 for s in bar_configs]
        costs   = [s.get("best_cost_reduction_pct")   or 0.0 for s in bar_configs]

        y = np.arange(len(labels))
        h = 0.35
        ax.barh(y - h/2, carbons, height=h, color=COLORS["carbon"],
                edgecolor="white", label="Carbon reduction")
        ax.barh(y + h/2, costs,   height=h, color=COLORS["cost"],
                edgecolor="white", label="Cost reduction")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Reduction vs. baseline (%)", fontsize=9)
        ax.set_title("Best Achievable Reductions\n(each at optimal α)", fontsize=9)
        ax.axvline(0, color="gray", lw=0.8)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(_axis_fmt(ax, axis="x"))

        # Value annotations
        for i, (c, s_) in enumerate(zip(carbons, bar_configs)):
            suffix = f"  [α={s_['best_carbon_alpha']:.1f}]" if s_.get("best_carbon_alpha") is not None else ""
            ax.text(max(c, 0) + 0.1, i - h/2, f"{c:.1f}%{suffix}",
                    ha="left", va="center", fontsize=7, color=COLORS["carbon"])
        for i, (c, s_) in enumerate(zip(costs, bar_configs)):
            suffix = f"  [α={s_['best_cost_alpha']:.1f}]" if s_.get("best_cost_alpha") is not None else ""
            ax.text(max(c, 0) + 0.1, i + h/2, f"{c:.1f}%{suffix}",
                    ha="left", va="center", fontsize=7, color=COLORS["cost"])

        legend_patches = [
            mpatches.Patch(color=COLORS["carbon"], label="Carbon reduction (best α)"),
            mpatches.Patch(color=COLORS["cost"],   label="Cost reduction (best α)"),
        ]
        ax.legend(handles=legend_patches, fontsize=7.5, framealpha=0.7)

    # ── Panel 3: 24/7 coverage ─────────────────────────────────────────────
    if has_coverage:
        ax = next(panel_iter)
        labels    = [s["label"] for s in s247_summaries]
        unshifted = [s.get("unshifted_coverage_pct") or 0.0 for s in s247_summaries]
        balanced  = [s.get("best_coverage_pct")      or 0.0 for s in s247_summaries]
        gains     = [s.get("best_coverage_gain_pct")  or 0.0 for s in s247_summaries]

        y = np.arange(len(labels))
        ax.barh(y, balanced,  color=COLORS["cover"],      edgecolor="white", height=0.55,
                label="Balanced coverage")
        ax.barh(y, unshifted, color=COLORS["cover_base"], edgecolor="white", height=0.55,
                label="Unshifted coverage")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("RE Coverage (%)", fontsize=9)
        ax.set_title("24/7 RE Coverage\n(unshifted vs. balanced)", fontsize=9)
        ax.axvline(0, color="gray", lw=0.8)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(_axis_fmt(ax, axis="x"))

        for i, (bal, gain) in enumerate(zip(balanced, gains)):
            sign = "+" if gain >= 0 else ""
            ax.text(bal + 0.3, i, f"{bal:.1f}%  ({sign}{gain:.1f} pp)",
                    ha="left", va="center", fontsize=7.5)

        legend_patches = [
            mpatches.Patch(color=COLORS["cover_base"], label="Unshifted (no CAS)"),
            mpatches.Patch(color=COLORS["cover"],      label="Balanced (with CAS)"),
        ]
        ax.legend(handles=legend_patches, fontsize=7.5, framealpha=0.7)

    fig.suptitle("CAS Mode Comparison — Best Achievable Metrics", fontsize=11, y=1.01)
    fig.tight_layout()
    p = out_dir / "compare_best_metrics.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _plot_policy_figure1(
    label: str,
    phi_hourly_csv: Path,
    out_dir: Path,
) -> Path | None:
    """
    Ackon (2025) Figure 1 (extended): Shapley allocations, net exports, and
    water withdrawal per microgrid at four representative hours
    (06:00, 12:00, 18:00, 00:00).

    Layout — 4-row × 2-col grid (rows 1-2: φ panels; rows 3-4: W panels):
      Top half  — NCS Shapley allocation (bars, left y) + net export (scatter,
                  right y).  Last bar = grand-coalition total.
      Bottom half — Water withdrawal intensity (gal/MWh, blue bars) per
                   microgrid at the same representative hours.  When water
                   data is absent from the CSV the bottom rows are hidden.

    Right axis is set to 2× the scale of the left axis so that net-export
    magnitudes are visually comparable to allocation magnitudes without
    dominating the chart.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as _mt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        return None

    if not phi_hourly_csv.exists():
        print(f"  [skip] policy figure — {phi_hourly_csv} not found")
        return None

    df = pd.read_csv(phi_hourly_csv)
    target_hours = [6, 12, 18, 0]
    has_water = "wd_gal_mwh" in df.columns

    # Pick first available date for each target hour
    snapshots: list[tuple[int, str, pd.DataFrame]] = []
    for h in target_hours:
        sub = df[df["hour"] == h]
        if sub.empty:
            continue
        date_val  = sub["date"].iloc[0]
        hour_data = (
            sub[sub["date"] == date_val]
            .sort_values("microgrid")
            .reset_index(drop=True)
        )
        snapshots.append((h, date_val, hour_data))

    if not snapshots:
        print(f"  [skip] policy figure — no matching hours in {phi_hourly_csv.name}")
        return None

    ncols     = 2
    phi_nrows = (len(snapshots) + 1) // 2   # rows for φ panels
    w_nrows   = phi_nrows if has_water else 0
    total_rows = phi_nrows + w_nrows

    row_heights = [4.5] * phi_nrows + [3.0] * w_nrows
    fig, axes = plt.subplots(
        total_rows, ncols,
        figsize=(12, sum(row_heights)),
        gridspec_kw={"height_ratios": row_heights},
        squeeze=False,
    )

    # ── Top half: NCS Shapley φ + net export ──────────────────────────────────
    phi_axes_flat = axes[:phi_nrows].flatten()

    for i, (hour_val, date_val, hour_data) in enumerate(snapshots):
        ax  = phi_axes_flat[i]
        ax2 = ax.twinx()

        microgrids  = hour_data["microgrid"].tolist()
        phi_vals    = hour_data["phi"].tolist()
        net_exp     = hour_data["net_export_mw"].tolist()
        grand_total = sum(phi_vals)

        x_labels = [f"MG{k}" for k in microgrids] + ["Grid"]
        phi_all  = phi_vals + [grand_total]
        x        = np.arange(len(x_labels))

        bar_colors = []
        for j, v in enumerate(phi_all):
            if j == len(phi_all) - 1:
                bar_colors.append("#4e79a7")
            elif v >= 0:
                bar_colors.append("#59a14f")
            else:
                bar_colors.append("#e15759")

        ax.bar(x, phi_all, color=bar_colors, edgecolor="white", width=0.6, zorder=3)
        ax2.scatter(x[:-1], net_exp, color="black", marker="o", s=35, zorder=5)

        lim = max((abs(v) for v in phi_all), default=1.0) * 1.2
        lim = lim or 1.0
        ax.set_ylim(-lim, lim)
        ax2.set_ylim(-2.0 * lim, 2.0 * lim)

        hour_str = f"{hour_val:02d}:00"
        ax.set_title(f"φ — {hour_str}  ({date_val})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("NCS Allocation (φ)", fontsize=9)
        ax2.set_ylabel("Net Export (MW)", fontsize=9)
        ax.axhline(0, color="gray", lw=0.8, zorder=1)
        ax.yaxis.set_major_formatter(_axis_fmt(ax))
        ax2.yaxis.set_major_formatter(_axis_fmt(ax2))

    for j in range(len(snapshots), len(phi_axes_flat)):
        phi_axes_flat[j].set_visible(False)

    # ── Bottom half: Water withdrawal intensity ────────────────────────────────
    if has_water:
        w_axes_flat = axes[phi_nrows:].flatten()

        # Global colour scale: lightest at 0, darkest at max across all snapshots
        all_wd = [v for _, _, hd in snapshots for v in hd["wd_gal_mwh"].tolist()]
        wd_max = max(all_wd) if all_wd else 1.0

        for i, (hour_val, date_val, hour_data) in enumerate(snapshots):
            ax_w = w_axes_flat[i]

            microgrids = hour_data["microgrid"].tolist()
            wd_vals    = hour_data["wd_gal_mwh"].tolist()
            wc_vals    = hour_data["wc_gal_mwh"].tolist() if "wc_gal_mwh" in hour_data.columns else [0.0] * len(wd_vals)
            x_mg       = np.arange(len(microgrids))
            x_labels_mg = [f"MG{k}" for k in microgrids]

            # Withdrawal as solid bars; consumption as hatched overlay
            bar_w = 0.4
            bars_wd = ax_w.bar(
                x_mg - bar_w / 2, wd_vals, width=bar_w,
                color="#1f77b4", alpha=0.80, edgecolor="white",
                label="Withdrawal (gal/MWh)",
            )
            bars_wc = ax_w.bar(
                x_mg + bar_w / 2, wc_vals, width=bar_w,
                color="#aec7e8", alpha=0.85, edgecolor="white",
                hatch="//",
                label="Consumption (gal/MWh)",
            )

            # Annotate withdrawal values above bars
            for bar, v in zip(bars_wd, wd_vals):
                if v > 0:
                    ax_w.text(
                        bar.get_x() + bar.get_width() / 2,
                        v * 1.02,
                        f"{v:.0f}",
                        ha="center", va="bottom", fontsize=6, color="#1f77b4",
                    )

            hour_str = f"{hour_val:02d}:00"
            ax_w.set_title(f"Water — {hour_str}  ({date_val})", fontsize=9)
            ax_w.set_xticks(x_mg)
            ax_w.set_xticklabels(x_labels_mg, rotation=45, ha="right", fontsize=7)
            ax_w.set_ylabel("Water Intensity\n(gal/MWh)", fontsize=8)
            ax_w.set_ylim(0, wd_max * 1.25 or 1.0)
            ax_w.yaxis.set_major_formatter(_axis_fmt(ax_w))
            ax_w.grid(axis="y", alpha=0.25)
            if i == 0:
                ax_w.legend(fontsize=7, loc="upper right")

        for j in range(len(snapshots), len(w_axes_flat)):
            w_axes_flat[j].set_visible(False)

    # ── Figure-level legend & titles ──────────────────────────────────────────
    legend_elements = [
        Patch(facecolor="#59a14f", label="Positive φ allocation"),
        Patch(facecolor="#e15759", label="Negative φ allocation"),
        Patch(facecolor="#4e79a7", label="Grid total (φ)"),
        Line2D([0], [0], marker="o", color="black", lw=0, markersize=6,
               label="Net export (MW)"),
    ]
    if has_water:
        legend_elements += [
            Patch(facecolor="#1f77b4", alpha=0.80, label="Water withdrawal (gal/MWh)"),
            Patch(facecolor="#aec7e8", alpha=0.85, hatch="//", label="Water consumption (gal/MWh)"),
        ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=min(len(legend_elements), 3),
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Policy Analysis: NCS Allocations, Net Exports & Water Withdrawal — {label}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    path = out_dir / f"policy_figure1_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Group A: CAS analysis dirs
    p.add_argument("--cas-dirs", nargs="+", type=Path, default=[], metavar="DIR",
                   help="CAS analysis output dirs (each must contain cas_results.csv).")
    p.add_argument("--cas-labels", nargs="+", default=[], metavar="LABEL",
                   help="Labels for each --cas-dir (same count).")
    # Group B: VATIC re-run sim dirs
    p.add_argument("--sim-dirs", nargs="*", type=Path, default=[], metavar="DIR",
                   help="VATIC simulation output dirs for system-level comparison.")
    p.add_argument("--sim-labels", nargs="*", default=[], metavar="LABEL",
                   help="Labels for each --sim-dir (same count).")
    p.add_argument("--gen-csv", type=Path, default=None,
                   help="gen.csv for emission factor lookup. Defaults to RTS-GMLC.")
    # Optional enrichment dirs
    p.add_argument("--water-dir", type=Path, default=None, metavar="DIR",
                   help="Base directory for water analysis outputs. "
                        "Expected subdirs: <water-dir>/<sim-label>/system_water_hourly.csv")
    p.add_argument("--env-score-dir", type=Path, default=None, metavar="DIR",
                   help="Directory with phi_hourly CSVs. "
                        "Expected files: <env-score-dir>/<sim-label>_phi_hourly.csv")
    p.add_argument("--baseline-label", default=None, metavar="LABEL",
                   help="Which sim-label is the baseline for delta comparison "
                        "(default: first --sim-label).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: ./cas_comparison).")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip all plot generation.")
    p.add_argument("--renew-capex-json", type=Path, default=None, metavar="FILE",
                   help="Path to renew_capex.json written by main.py when renew_opt "
                        "apply_best=true. When supplied, adds annualised CAPEX rows "
                        "(based on the actual portfolio built) to system_comparison.csv "
                        "and computes cost-per-tCO2-avoided.")
    p.add_argument("--dc-flex-capex-json", type=Path, default=None, metavar="FILE",
                   help="Path to dc_flex_capex.json written by main.py. When supplied, "
                        "adds dc_flex_capex_annual_usd row (cost of DC over-provisioning "
                        "buffer) to system_comparison.csv for all non-baseline scenarios.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.cas_dirs and not args.sim_dirs:
        sys.exit("Provide at least --cas-dirs or --sim-dirs.")
    if args.cas_dirs and len(args.cas_dirs) != len(args.cas_labels):
        sys.exit("--cas-dirs and --cas-labels must have the same count.")
    if args.sim_dirs and len(args.sim_dirs) != len(args.sim_labels):
        sys.exit("--sim-dirs and --sim-labels must have the same count.")
    if args.sim_dirs and len(args.sim_dirs) < 2:
        sys.exit("Provide at least two --sim-dirs to compare.")

    for d in args.cas_dirs:
        if not (d / "cas_results.csv").exists():
            sys.exit(f"cas_results.csv not found in {d}")
    for d in args.sim_dirs:
        if not d.is_dir():
            sys.exit(f"sim-dir not found: {d}")

    gen_csv = args.gen_csv
    if gen_csv is None:
        gen_csv = (_VATIC_ROOT / "vatic" / "data" / "grids" /
                   "RTS-GMLC" / "RTS_Data" / "SourceData" / "gen.csv")
        if args.sim_dirs:
            print(f"--gen-csv not specified, using default: {gen_csv}")
    if args.sim_dirs and not gen_csv.exists():
        sys.exit(f"gen.csv not found: {gen_csv}")

    out_dir = args.out_dir or Path("outputs") / "cas_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_label = args.baseline_label or (args.sim_labels[0] if args.sim_labels else None)

    # Build water_dirs list aligned with sim_labels
    water_dirs: list[Path | None] | None = None
    if args.sim_dirs and args.water_dir:
        water_dirs = [
            args.water_dir / lbl
            for lbl in args.sim_labels
        ]

    # ── Group A: load CAS analysis results ───────────────────────────────────
    summaries: list[dict] = []
    dfs: dict[str, tuple[str, pd.DataFrame]] = {}

    if args.cas_dirs:
        print(f"\nLoading {len(args.cas_dirs)} CAS analysis result(s)…")
        for d, label in zip(args.cas_dirs, args.cas_labels):
            csv_path = d / "cas_results.csv"
            mode     = detect_mode(csv_path)
            df       = pd.read_csv(csv_path)
            s        = extract_summary(df, mode, label)
            summaries.append(s)
            dfs[label] = (mode, df)
            print(f"  {label:20s}  mode={mode:8s}  rows={len(df)}")

        print_summary_table(summaries)

    # ── Group B: system-level comparison ─────────────────────────────────────
    system_comparison = None
    if args.sim_dirs:
        print(f"\nComputing system-level stats for {len(args.sim_dirs)} sim run(s)…")
        system_comparison = cso.build_comparison(
            list(args.sim_dirs), list(args.sim_labels), gen_csv,
            water_dirs=water_dirs,
        )

        # ── Append LOLP / LOLE reliability rows ──────────────────────────────
        rel_rows: dict[str, list] = {"lolp": [], "lole_h": []}
        for lbl, sdir in zip(args.sim_labels, args.sim_dirs):
            r = reliability.compute_reliability(Path(sdir))
            rel_rows["lolp"].append(r["lolp"])
            rel_rows["lole_h"].append(r["lole_h"])
        for metric, vals in rel_rows.items():
            system_comparison.loc[metric] = vals
        delta_df = cso._delta_cols(system_comparison, args.sim_labels[0])

        KEY_METRICS = [
            "mean_ci_kgco2_mwh", "total_co2_tonnes",
            "variable_cost_usd", "total_cost_usd",
            "peak_lmp_usd_mwh", "lmp_std_usd_mwh",
            "total_demand_mwh", "load_shedding_mwh",
            "lolp", "lole_h",
            "renewables_used_mwh", "renewables_curtailed_mwh",
            "total_wd_gal", "wd_intensity_gal_mwh",
        ]
        print(f"\n{'='*72}")
        print("  SYSTEM-LEVEL KEY METRICS")
        print(f"{'='*72}")
        rows = [m for m in KEY_METRICS if m in delta_df.index]
        print(delta_df.loc[rows].to_string(float_format="{:,.2f}".format))

        fuel_rows = [r for r in delta_df.index if r.startswith("dispatch_")]
        if fuel_rows:
            print(f"\n{'='*72}")
            print("  DISPATCH BY FUEL (MWh)")
            print(f"{'='*72}")
            print(delta_df.loc[fuel_rows].to_string(float_format="{:,.1f}".format))

        # ── CAPEX rows (only when an investment grid was applied) ─────────────
        # capex_annual_usd is 0 for the baseline (no new infrastructure) and
        # equals the full annualised cost of the portfolio for every shifted
        # mode — they all run on the same investment-enhanced grid.
        if args.renew_capex_json and args.renew_capex_json.exists():
            import json as _json
            _capex = _json.load(open(args.renew_capex_json))
            _capex_ann  = float(_capex.get("capex_annual_usd", 0.0))
            _embod_ann  = float(_capex.get("embodied_co2_annual_t", 0.0))

            # capex_annual_usd row
            _capex_row = {lbl: (0.0 if lbl == baseline_label else _capex_ann)
                          for lbl in args.sim_labels}
            system_comparison.loc["capex_annual_usd"] = _capex_row

            # embodied_co2_annual_t row
            _embod_row = {lbl: (0.0 if lbl == baseline_label else _embod_ann)
                          for lbl in args.sim_labels}
            system_comparison.loc["embodied_co2_annual_t"] = _embod_row

            # total_cost_incl_capex_usd = operational cost + CAPEX
            if "total_cost_usd" in system_comparison.index:
                system_comparison.loc["total_cost_incl_capex_usd"] = (
                    system_comparison.loc["total_cost_usd"]
                    + system_comparison.loc["capex_annual_usd"]
                )

            # cost_per_tco2_avoided_usd = extra total cost / CO₂ avoided
            # Positive = paying to reduce CO₂; negative = saving money AND reducing CO₂
            if ("total_cost_incl_capex_usd" in system_comparison.index
                    and "total_co2_tonnes" in system_comparison.index):
                _base_cost = float(system_comparison.at["total_cost_incl_capex_usd", baseline_label])
                _base_co2  = float(system_comparison.at["total_co2_tonnes",          baseline_label])
                _cpp_row   = {}
                for lbl in args.sim_labels:
                    if lbl == baseline_label:
                        _cpp_row[lbl] = 0.0
                        continue
                    _sim_cost = float(system_comparison.at["total_cost_incl_capex_usd", lbl])
                    _sim_co2  = float(system_comparison.at["total_co2_tonnes",          lbl])
                    _avoided  = _base_co2 - _sim_co2
                    _extra    = _sim_cost - _base_cost
                    _cpp_row[lbl] = round(_extra / _avoided, 2) if abs(_avoided) > 0.01 else float("nan")
                system_comparison.loc["cost_per_tco2_avoided_usd"] = _cpp_row

            # Rebuild delta_df to include the new rows
            delta_df = cso._delta_cols(system_comparison, args.sim_labels[0])

            print(f"\n  [renew_opt] CAPEX applied: wind={_capex.get('wind_mw',0):.0f} MW  "
                  f"solar={_capex.get('solar_mw',0):.0f} MW  "
                  f"battery={_capex.get('battery_mw',0):.0f} MW  "
                  f"→ ${_capex_ann/1e6:.1f}M/yr annualised")

        # ── DC flexibility CAPEX (cost of server over-provisioning buffer) ────
        # Applied to every non-baseline scenario — all CAS modes require the
        # same headroom, regardless of whether renewables were invested in.
        if args.dc_flex_capex_json and args.dc_flex_capex_json.exists():
            import json as _json
            _dc = _json.load(open(args.dc_flex_capex_json))
            _dc_flex_ann = float(_dc.get("dc_flex_capex_sim_usd", 0.0))

            _dc_flex_row = {lbl: (0.0 if lbl == baseline_label else _dc_flex_ann)
                            for lbl in args.sim_labels}
            system_comparison.loc["dc_flex_capex_annual_usd"] = _dc_flex_row

            # Re-derive total_cost_incl_capex_usd including DC flex CAPEX
            _renew_row = system_comparison.loc["capex_annual_usd"] \
                if "capex_annual_usd" in system_comparison.index \
                else pd.Series(0.0, index=args.sim_labels)
            if "total_cost_usd" in system_comparison.index:
                system_comparison.loc["total_cost_incl_capex_usd"] = (
                    system_comparison.loc["total_cost_usd"]
                    + _renew_row
                    + system_comparison.loc["dc_flex_capex_annual_usd"]
                )

            # Recompute cost_per_tco2_avoided with updated total cost
            if ("total_cost_incl_capex_usd" in system_comparison.index
                    and "total_co2_tonnes" in system_comparison.index):
                _base_cost = float(system_comparison.at["total_cost_incl_capex_usd", baseline_label])
                _base_co2  = float(system_comparison.at["total_co2_tonnes",          baseline_label])
                _cpp_row   = {}
                for lbl in args.sim_labels:
                    if lbl == baseline_label:
                        _cpp_row[lbl] = 0.0
                        continue
                    _sim_cost = float(system_comparison.at["total_cost_incl_capex_usd", lbl])
                    _sim_co2  = float(system_comparison.at["total_co2_tonnes",          lbl])
                    _avoided  = _base_co2 - _sim_co2
                    _extra    = _sim_cost - _base_cost
                    _cpp_row[lbl] = round(_extra / _avoided, 2) if abs(_avoided) > 0.01 else float("nan")
                system_comparison.loc["cost_per_tco2_avoided_usd"] = _cpp_row

            # Rebuild delta_df to include all new rows
            delta_df = cso._delta_cols(system_comparison, args.sim_labels[0])

            print(f"\n  [dc_flex]   DC over-provisioning: {_dc.get('extra_capacity_mw',0):.1f} MW "
                  f"extra → ${_dc.get('dc_flex_capex_annual_usd',0)/1e6:.2f}M/yr "
                  f"(${_dc_flex_ann/1e3:.0f}K/sim-period)")

        if "cost_per_tco2_avoided_usd" in system_comparison.index:
            for lbl in [l for l in args.sim_labels if l != baseline_label]:
                cpp = system_comparison.at["cost_per_tco2_avoided_usd", lbl]
                if cpp == cpp:  # skip nan
                    print(f"     {lbl}: ${cpp:,.0f}/tCO₂ avoided (incl. all CAPEX)")

        sys_csv = out_dir / "system_comparison.csv"
        delta_df.to_csv(sys_csv)
        print(f"\nSystem comparison → {sys_csv}")

        # Back-fill grid_total_cost_change_pct into LP summaries.
        # Match each CAS label to the best-matching sim label by substring:
        #   "lp" → sim-lp,  "grid-mix"/"gm" → sim-gm,  "24_7"/"247" → sim-247
        _slug = {"lp": "lp", "grid-mix": "gm", "gm": "gm",
                 "24_7": "247", "247": "247"}
        non_base = [l for l in args.sim_labels if l != baseline_label]
        for s in summaries:
            if s["mode"] != "lp":
                continue
            suffix = _slug.get(s["label"].lower())
            if suffix is None:
                # try substring match on the label itself
                suffix = s["label"].replace("-", "").replace("_", "").lower()
            matched = next((l for l in non_base
                            if suffix in l.replace("-", "").replace("_", "").lower()), None)
            if matched:
                col = f"Δ% ({matched} vs {baseline_label})"
                if "total_cost_usd" in delta_df.index and col in delta_df.columns:
                    s["grid_total_cost_change_pct"] = round(
                        float(delta_df.at["total_cost_usd", col]), 2
                    )

    # Write summary CSV after back-filling grid_total_cost_change_pct
    if summaries:
        summary_df = pd.DataFrame(summaries).set_index("label")
        csv_path   = out_dir / "cas_mode_summary.csv"
        summary_df.to_csv(csv_path)
        print(f"\nSummary table → {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.no_plots:
        print(f"\nAll outputs in: {out_dir}")
        return

    print("\nGenerating plots…")
    generated: list[Path] = []

    # 1. Best metrics bar chart (CAS Group A)
    if summaries:
        p = _plot_best_metrics(summaries, out_dir, dfs if args.cas_dirs else None)
        if p:
            print(f"  → {p.name}")
            generated.append(p)

    # 2. System comparison plots (time series, stacked dispatch, key metrics)
    if args.sim_dirs and system_comparison is not None:
        print("  System-level plots:")
        cso.make_plots(
            list(args.sim_dirs), list(args.sim_labels),
            gen_csv, system_comparison, out_dir,
            water_dirs=water_dirs,
        )

    # 3. Policy Analysis Figure 1 — one per sim label that has phi_hourly data
    if args.env_score_dir and args.sim_dirs:
        for label in args.sim_labels:
            phi_csv = args.env_score_dir / f"{label}_phi_hourly.csv"
            p = _plot_policy_figure1(label, phi_csv, out_dir)
            if p:
                print(f"  → {p.name}")
                generated.append(p)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
