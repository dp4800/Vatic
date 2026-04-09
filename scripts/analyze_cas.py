#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
analyze_cas.py — Carbon-Aware Scheduling analysis for VATIC simulations.

Supports two scheduling modes (--mode):

  grid-mix  (default)
      Signal  : hourly grid carbon intensity from the simulation.
      Metric  : carbon reduction (%).
      Algorithm: cas_grid_mix — shift flexible load toward low-CI hours.
      Outputs : cas_results.csv, cas_timeseries.png, cas_surface.png (3D).

  24_7
      Signal  : hourly renewable surplus (renewable supply − DC demand).
      Metric  : 24/7 renewable coverage (%).
      Algorithm: cas_24_7 — shift flexible load toward hours where renewable
                 supply exceeds DC demand.
      Outputs : cas_results.csv, cas_timeseries.png, cas_heatmap.png.
      Extra arg: --renew-generators (optional, to restrict to co-located units).

Usage
-----
    # Grid-mix mode
    python scripts/analyze_cas.py \\
        --sim-dir  outputs/dc/2020-05-04 \\
        --grid     RTS-GMLC-DC \\
        --buses    Abel Adams \\
        --mode     grid-mix

    # 24/7 mode (using all grid renewables as supply)
    python scripts/analyze_cas.py \\
        --sim-dir  outputs/dc-re/2020-05-04 \\
        --grid     RTS-GMLC-DC-RE \\
        --buses    Abel Adams \\
        --mode     24_7

    # 24/7 mode restricted to co-located generators only
    python scripts/analyze_cas.py \\
        --sim-dir  outputs/dc-re/2020-05-04 \\
        --grid     RTS-GMLC-DC-RE \\
        --buses    Abel Adams \\
        --mode     24_7 \\
        --renew-generators 101_PV_5 102_PV_3 101_WIND_1 102_WIND_1 \\
        --max-extra-capacity      100 \\
        --max-flexible-work-ratio 100

    # LP mode — price-taking (no perturbation run needed)
    python scripts/analyze_cas.py \\
        --sim-dir  outputs/dc/2020-05-04 \\
        --grid     RTS-GMLC-DC \\
        --buses    Abel Adams \\
        --mode     lp \\
        --extra-capacity 30 \\
        --alpha-steps 11

    # LP mode — price-anticipating QP (supply a perturbed-DC-load simulation)
    python scripts/analyze_cas.py \\
        --sim-dir        outputs/dc/2020-05-04 \\
        --perturb-sim-dir outputs/dc-perturb/2020-05-04 \\
        --grid           RTS-GMLC-DC \\
        --buses          Abel Adams \\
        --mode           lp \\
        --extra-capacity 30 \\
        --alpha-steps    11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
import cas  # noqa: E402

GRID_REGISTRY: dict[str, dict] = {
    "RTS-GMLC":      {"data_dir": "RTS_Data"},
    "Texas-7k_2030": {"data_dir": "TX2030_Data"},
    "Texas-7k":      {"data_dir": "TX_Data"},
}
_VATIC_ROOT = _SCRIPTS_DIR.parent
_GRIDS_DIR  = _VATIC_ROOT / "vatic" / "data" / "grids"


def _resolve_paths(grid: str) -> tuple[Path, Path]:
    prefix = next(
        (k for k in sorted(GRID_REGISTRY, key=len, reverse=True) if grid.startswith(k)),
        None,
    )
    if prefix is None:
        sys.exit(f"Unsupported grid '{grid}'. Known prefixes: {list(GRID_REGISTRY)}")
    cfg    = GRID_REGISTRY[prefix]
    data   = _GRIDS_DIR / grid / cfg["data_dir"]
    inject = data / "timeseries_data_files" / "BusInjections"
    gen_csv = data / "SourceData" / "gen.csv"
    return inject, gen_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sim-dir", required=True, type=Path,
                   help="Vatic simulation output directory.")
    p.add_argument("--grid", required=True,
                   help="Grid name with BusInjections (e.g. RTS-GMLC-DC).")
    p.add_argument("--buses", required=True, nargs="+", metavar="BUS",
                   help="DC bus names to sum for the DC power profile.")
    p.add_argument("--mode", choices=["grid-mix", "24_7", "lp"], default="grid-mix",
                   help="Scheduling mode: grid-mix (minimize CI), 24_7 "
                        "(maximize renewable coverage), or lp (LP-optimal, "
                        "minimises α·LMP + (1-α)·CI). Default: grid-mix.")
    p.add_argument("--renew-generators", nargs="*", default=None, metavar="GEN_UID",
                   help="[24_7 only] Restrict renewable supply to these generator UIDs. "
                        "Defaults to all Solar+Wind in the simulation.")
    p.add_argument("--renew-fraction", type=float, default=1.0, metavar="FRAC",
                   help="[24_7 only] Scale renewable supply by this factor (0–1). "
                        "Models a contracted share of grid-wide renewable output "
                        "(e.g. 0.10 = 10%% of total renewable generation). Default: 1.0.")
    p.add_argument("--start-date", default=None,
                   help="Start of analysis window YYYY-MM-DD.")
    p.add_argument("--end-date", default=None,
                   help="End of analysis window YYYY-MM-DD (inclusive).")
    p.add_argument("--max-extra-capacity", type=int, default=50, metavar="PCT",
                   help="Max server headroom above current peak (%%, default: 50).")
    p.add_argument("--max-flexible-work-ratio", type=int, default=50, metavar="PCT",
                   help="Max flexible workload ratio (%%, default: 50).")
    p.add_argument("--step", type=int, default=10,
                   help="Step size for parameter sweep (default: 10).")
    p.add_argument("--deferral-window", type=int, default=12, metavar="H",
                   help="[grid-mix/24_7] Max hours a flexible job may be deferred (default: 12).")
    p.add_argument("--extra-capacity", type=float, default=30.0, metavar="PCT",
                   help="[lp] Server headroom above current DC peak (%%, default: 30).")
    p.add_argument("--alpha-steps", type=int, default=11, metavar="N",
                   help="[lp] Number of alpha values to sweep from 0 to 1 (default: 11).")
    p.add_argument("--ramp-rate", type=float, default=None, metavar="MW/H",
                   help="[lp] Max load change between hours (MW/h). Default: unconstrained.")
    p.add_argument("--perturb-sim-dir", type=Path, default=None, metavar="DIR",
                   help="[lp] Vatic simulation directory for a perturbed (higher DC load) run. "
                        "When supplied, enables the price-anticipating QP via β=∂LMP/∂D "
                        "estimated from the baseline vs perturbed runs.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: sim-dir/cas_analysis).")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib output.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    sim_dir = args.sim_dir.resolve()
    if not sim_dir.is_dir():
        sys.exit(f"sim-dir not found: {sim_dir}")

    out_dir = args.out_dir or sim_dir / "cas_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = args.start_date or sim_dir.name
    end_date   = args.end_date   or start_date

    # -----------------------------------------------------------------------
    # Load DC power
    # -----------------------------------------------------------------------
    inject_dir, gen_csv = _resolve_paths(args.grid)
    if not inject_dir.is_dir():
        sys.exit(f"BusInjections directory not found: {inject_dir}")

    inject_csvs = list(inject_dir.glob("DAY_AHEAD_*.csv"))
    if len(inject_csvs) != 1:
        sys.exit(f"Expected one DAY_AHEAD_*.csv in {inject_dir}, found {len(inject_csvs)}")

    print(f"Mode              : {args.mode}")
    print(f"Loading DC power  : {inject_csvs[0].name}")
    dc_power = cas.load_dc_power(
        inject_csvs[0], buses=args.buses,
        start_date=start_date, end_date=end_date,
    )

    # -----------------------------------------------------------------------
    # Load signal (CI or renewable supply) and build df_all
    # -----------------------------------------------------------------------
    if args.mode == "grid-mix":
        if args.perturb_sim_dir is not None:
            pdir = args.perturb_sim_dir.resolve()
            if not pdir.is_dir():
                sys.exit(f"--perturb-sim-dir not found: {pdir}")
            print(f"Computing marginal CI : {sim_dir.name} vs {pdir.name}")
            mci_df    = cas.compute_marginal_ci(sim_dir, pdir, args.buses, gen_csv)
            signal_df = mci_df[["marginal_ci", "avg_ci_base"]].rename(
                columns={"marginal_ci": "carbon_intensity"}
            )
            print(f"  Mean marginal CI={mci_df['marginal_ci'].mean():.2f}  "
                  f"avg CI={mci_df['avg_ci_base'].mean():.2f} kg CO₂/MWh")
        else:
            print(f"Computing avg CI  : {sim_dir.name}")
            signal_df = cas.compute_carbon_intensity(sim_dir, gen_csv)
        df_all = dc_power.join(signal_df, how="inner")
        signal_col  = "carbon_intensity"
        signal_label = "Carbon Intensity (kg CO₂/MWh)"

    elif args.mode == "lp":
        beta_series = None
        if args.perturb_sim_dir is not None:
            pdir = args.perturb_sim_dir.resolve()
            if not pdir.is_dir():
                sys.exit(f"--perturb-sim-dir not found: {pdir}")
            print(f"Computing marginal CI : {sim_dir.name} vs {pdir.name}")
            mci_df = cas.compute_marginal_ci(sim_dir, pdir, args.buses, gen_csv)
            ci_df  = mci_df[["marginal_ci", "avg_ci_base"]].rename(
                columns={"marginal_ci": "carbon_intensity"}
            )
            print(f"  Mean marginal CI={mci_df['marginal_ci'].mean():.2f}  "
                  f"avg CI={mci_df['avg_ci_base'].mean():.2f} kg CO₂/MWh")
            print(f"Computing β           : {sim_dir.name} vs {pdir.name}")
            beta_df     = cas.compute_price_sensitivity(sim_dir, pdir, args.buses)
            beta_series = beta_df["beta"]
            print(f"  Mean β={beta_series.mean():.4f}  max β={beta_series.max():.4f}  "
                  f"non-zero={int((beta_series > 0).sum())}/{len(beta_series)} hours")
        else:
            print(f"Computing avg CI  : {sim_dir.name}")
            ci_df = cas.compute_carbon_intensity(sim_dir, gen_csv)
        print(f"Loading LMPs      : {sim_dir.name}  buses={args.buses}")
        lmp_df = cas.load_lmp(sim_dir, args.buses,
                              start_date=start_date, end_date=end_date)
        df_all = dc_power.join(ci_df, how="inner").join(lmp_df, how="inner")
        signal_col  = "carbon_intensity"   # used for timeseries overlay
        signal_label = "Carbon Intensity (kg CO₂/MWh)"

    else:  # 24_7
        gens = args.renew_generators or None
        label_suffix = f" [{', '.join(gens)}]" if gens else " [all Solar+Wind]"
        frac_str = f"  ×{args.renew_fraction:.2f}" if args.renew_fraction != 1.0 else ""
        print(f"Loading renewables: {sim_dir.name}{label_suffix}{frac_str}")
        renew_df = cas.load_renewable_supply(sim_dir, gen_csv, generators=gens)
        if args.renew_fraction != 1.0:
            renew_df = renew_df * args.renew_fraction
        df_all = dc_power.join(renew_df, how="inner")
        signal_col  = "tot_renewable"
        signal_label = "Renewable Supply (MW)"

    if df_all.empty:
        sys.exit("No overlapping hours between DC power and simulation signal.")

    n_hours  = len(df_all)
    n_days   = n_hours / 24
    cur_peak = float(df_all["avg_dc_power_mw"].max())

    print(f"Analysis window   : {df_all.index[0].date()} – {df_all.index[-1].date()} "
          f"({n_hours} hours, {n_days:.1f} days)")
    print(f"DC peak load      : {cur_peak:.1f} MW")
    _ci_type = "marginal" if args.perturb_sim_dir else "avg"
    if args.mode == "grid-mix":
        print(f"Mean grid CI ({_ci_type:8s}): {df_all['carbon_intensity'].mean():.2f} kg CO₂/MWh")
    elif args.mode == "lp":
        print(f"Mean grid CI ({_ci_type:8s}): {df_all['carbon_intensity'].mean():.2f} kg CO₂/MWh")
        print(f"Mean LMP          : {df_all['lmp'].mean():.2f} $/MWh")
    else:
        base_cov = cas.calculate_coverage(df_all)
        print(f"Unshifted coverage: {base_cov:.1f}%")
        print(f"Mean renew supply : {df_all['tot_renewable'].mean():.1f} MW")

    # -----------------------------------------------------------------------
    # Parametric sweep
    # -----------------------------------------------------------------------
    results_rows = []
    first_shifted = None

    if args.mode == "lp":
        # LP mode: sweep alpha (cost/carbon trade-off) at fixed server capacity
        max_cap = cur_peak * (1.0 + args.extra_capacity / 100.0)
        alphas  = np.linspace(0.0, 1.0, args.alpha_steps)

        orig_carbon = cas.carbon_cost(df_all)
        orig_cost   = float((df_all["avg_dc_power_mw"] * df_all["lmp"]).sum())

        mode_tag = "QP (price-anticipating)" if beta_series is not None else "LP (price-taking)"
        print(f"\nSweeping alpha (0=carbon-only … 1=cost-only)  [{mode_tag}]  "
              f"max_cap={max_cap:.1f} MW  ramp={args.ramp_rate} MW/h …")

        for alpha in alphas:
            shifted = cas.cas_lp(df_all, max_cap,
                                 alpha=alpha, ramp_rate=args.ramp_rate,
                                 deferral_window=args.deferral_window,
                                 beta=beta_series)
            new_carbon = cas.carbon_cost(shifted)
            new_cost   = float((shifted["avg_dc_power_mw"] * df_all["lmp"]).sum())
            carbon_red = (orig_carbon - new_carbon) / orig_carbon * 100 if orig_carbon > 0 else 0.0
            cost_red   = (orig_cost   - new_cost)   / orig_cost   * 100 if orig_cost   > 0 else 0.0
            row = {
                "alpha":              round(float(alpha), 3),
                "carbon_reduction_pct": round(carbon_red, 3),
                "cost_reduction_pct":   round(cost_red,   3),
                "total_co2_kg":         round(new_carbon, 1),
                "total_cost_usd":       round(new_cost,   2),
            }
            results_rows.append(row)
            if first_shifted is None:
                first_shifted = shifted

    else:
        step        = args.step
        extra_caps  = range(step, args.max_extra_capacity + 1, step)
        flex_ratios = range(step, args.max_flexible_work_ratio + 1, step)

        if args.mode == "grid-mix":
            baseline_metric = cas.carbon_cost(df_all)
        else:
            baseline_metric = cas.calculate_coverage(df_all)

        print(f"\nSweeping extra_capacity × flexible_work_ratio …")

        for extra_cap in extra_caps:
            max_cap = cur_peak * (1.0 + extra_cap / 100.0)
            for flex_ratio in flex_ratios:
                if args.mode == "grid-mix":
                    shifted        = cas.cas_grid_mix(df_all, flex_ratio, max_cap, window=args.deferral_window)
                    shifted_metric = cas.carbon_cost(shifted)
                    delta_pct = (
                        (baseline_metric - shifted_metric) / baseline_metric * 100
                        if baseline_metric > 0 else 0.0
                    )
                    row = {
                        "extra_capacity":       extra_cap,
                        "flexible_work_ratio":  flex_ratio,
                        "imbalanced_cost":      round(baseline_metric, 1),
                        "balanced_cost":        round(shifted_metric, 1),
                        "carbon_reduction_pct": round(delta_pct, 3),
                    }
                else:
                    shifted      = cas.cas_24_7(df_all, flex_ratio, max_cap, window=args.deferral_window)
                    coverage_pct = cas.calculate_coverage(shifted)
                    row = {
                        "extra_capacity":      extra_cap,
                        "flexible_work_ratio": flex_ratio,
                        "imbalanced_coverage": round(baseline_metric, 1),
                        "balanced_coverage":   round(coverage_pct, 1),
                        "coverage_gain_pct":   round(coverage_pct - baseline_metric, 1),
                    }
                results_rows.append(row)
                if first_shifted is None:
                    first_shifted = shifted

    results = pd.DataFrame(results_rows)
    csv_path = out_dir / "cas_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"Results saved → {csv_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    if args.mode == "lp":
        print(f"\nalpha → carbon_reduction%  /  cost_reduction%:")
        for _, row in results.iterrows():
            print(f"  α={row['alpha']:.2f}  carbon={row['carbon_reduction_pct']:+.2f}%"
                  f"  cost={row['cost_reduction_pct']:+.2f}%")
    else:
        value_col = "carbon_reduction_pct" if args.mode == "grid-mix" else "balanced_coverage"
        pivot = results.pivot(
            index="flexible_work_ratio",
            columns="extra_capacity",
            values=value_col,
        )
        label = "Carbon Reduction (%)" if args.mode == "grid-mix" else "24/7 Coverage (%)"
        print(f"\n{label} — rows=flexible_work_ratio, cols=extra_capacity:")
        print(pivot.to_string())

        best_idx = results[value_col].idxmax()
        best     = results.loc[best_idx]
        if args.mode == "grid-mix":
            print(f"\nBest: {best[value_col]:.2f}% reduction  "
                  f"(extra_cap={int(best['extra_capacity'])}%, "
                  f"flex={int(best['flexible_work_ratio'])}%)")
        else:
            print(f"\nBest: {best[value_col]:.1f}% coverage  "
                  f"(+{best['coverage_gain_pct']:.1f} pp vs unshifted {baseline_metric:.1f}%)  "
                  f"extra_cap={int(best['extra_capacity'])}%, "
                  f"flex={int(best['flexible_work_ratio'])}%")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    if args.no_plots:
        return

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    # ---- Time-series plot (first / α=0.5 combo) ---------------------------
    if args.mode == "lp":
        max_cap0 = cur_peak * (1.0 + args.extra_capacity / 100.0)
        combo_label = f"α={results['alpha'].iloc[len(results)//2]:.2f}"
    else:
        ec0 = int(results["extra_capacity"].iloc[0])
        fr0 = int(results["flexible_work_ratio"].iloc[0])
        max_cap0 = cur_peak * (1.0 + ec0 / 100.0)
        combo_label = f"extra_cap={ec0}%  flex={fr0}%"
    hours = np.arange(n_hours)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(hours, df_all["avg_dc_power_mw"].values,
             color="red", lw=2.5, label="Original DC load")
    ax1.plot(hours, first_shifted["avg_dc_power_mw"].values,
             color="steelblue", lw=2.5, label="Shifted DC load")
    ax1.axhline(max_cap0, color="black", linestyle="--", lw=1.5,
                label=f"Max cap ({max_cap0:.0f} MW)")
    ax1.set_ylabel("DC Load (MW)")
    ax1.set_xlabel("Hour")
    ax1.set_ylim(0, max_cap0 * 1.15)

    ax2 = ax1.twinx()
    ax2.plot(hours, df_all[signal_col].values,
             color="gray", lw=1.5, alpha=0.7, linestyle="--",
             label=signal_label)
    if args.mode == "24_7":
        # Shade hours where renewable supply > original DC demand (green surplus)
        surplus = df_all["tot_renewable"].values - df_all["avg_dc_power_mw"].values
        ax1.fill_between(hours, 0, df_all["avg_dc_power_mw"].values,
                         where=(surplus >= 0), alpha=0.10, color="green",
                         label="Renewable surplus hours")
    ax2.set_ylabel(signal_label)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    mode_str = {"grid-mix": "Grid-Mix", "24_7": "24/7", "lp": "LP"}[args.mode]
    ax1.set_title(
        f"CAS {mode_str} — {args.grid}  |  {combo_label}  "
        f"({df_all.index[0].date()} – {df_all.index[-1].date()})"
    )
    day_ticks  = [i * 24 for i in range(int(n_days) + 1) if i * 24 <= n_hours]
    day_labels = [
        (df_all.index[0] + pd.Timedelta(hours=i * 24)).strftime("%m/%d")
        for i in range(len(day_ticks))
    ]
    ax1.set_xticks(day_ticks)
    ax1.set_xticklabels(day_labels)

    fig.tight_layout()
    ts_path = out_dir / "cas_timeseries.png"
    fig.savefig(ts_path, dpi=150)
    print(f"Time-series plot  → {ts_path}")
    plt.close(fig)

    # ---- Sweep summary plot -----------------------------------------------
    if args.mode == "lp":
        # Pareto curve: carbon reduction vs cost reduction as alpha varies
        fig_p, ax_p = plt.subplots(figsize=(8, 6))
        sc = ax_p.scatter(
            results["cost_reduction_pct"],
            results["carbon_reduction_pct"],
            c=results["alpha"], cmap=plt.cm.RdYlGn_r,
            s=80, zorder=3,
        )
        # Annotate alpha values
        for _, row in results.iterrows():
            ax_p.annotate(
                f"α={row['alpha']:.1f}",
                (row["cost_reduction_pct"], row["carbon_reduction_pct"]),
                textcoords="offset points", xytext=(5, 3), fontsize=7,
            )
        fig_p.colorbar(sc, ax=ax_p, label="α (0=carbon-only, 1=cost-only)")
        ax_p.set_xlabel("Cost Reduction [%]", labelpad=12)
        ax_p.set_ylabel("Carbon Reduction [%]", labelpad=12)
        ax_p.set_title(
            f"CAS LP Pareto Front — {args.grid}\n"
            f"max_cap={max_cap0:.0f} MW  ramp={args.ramp_rate} MW/h  "
            f"({df_all.index[0].date()} – {df_all.index[-1].date()})"
        )
        ax_p.axhline(0, color="gray", lw=0.8, ls="--")
        ax_p.axvline(0, color="gray", lw=0.8, ls="--")
        fig_p.tight_layout()
        sweep_path = out_dir / "cas_pareto.png"
        fig_p.savefig(sweep_path, dpi=150)
        plt.close(fig_p)

    elif args.mode == "grid-mix":
        # 3-D surface (same as before)
        fig3d = plt.figure(figsize=(9, 7))
        ax3d = fig3d.add_subplot(111, projection="3d")
        surf = ax3d.plot_trisurf(
            results["extra_capacity"].values,
            results["flexible_work_ratio"].values,
            results[value_col].values,
            cmap=plt.cm.Greens, linewidth=0.2, alpha=0.9,
        )
        fig3d.colorbar(surf, ax=ax3d, shrink=0.5, label="Carbon Reduction (%)")
        ax3d.set_xlabel("Extra Capacity [%]", labelpad=18)
        ax3d.set_ylabel("Flexible Workload [%]", labelpad=18)
        ax3d.set_zlabel("Carbon\nReduction [%]", labelpad=18)
        ax3d.set_title(f"CAS Grid-Mix Carbon Reduction — {args.grid}")
        ax3d.dist = 13
        sweep_path = out_dir / "cas_surface.png"
        fig3d.savefig(sweep_path, dpi=150, bbox_inches="tight")
        plt.close(fig3d)

    else:  # 24_7
        # Heatmap (matching CarbonExplorer 24/7 output)
        try:
            import seaborn as sns
            use_sns = True
        except ImportError:
            use_sns = False

        fig_hm, ax_hm = plt.subplots(figsize=(9, 6))
        pivot_int = pivot.round(1)

        if use_sns:
            sns.heatmap(
                pivot_int,
                cmap=plt.cm.Greens,
                ax=ax_hm,
                annot=True,
                fmt=".1f",
                cbar_kws={"label": "24/7 Coverage [%]"},
            )
        else:
            im = ax_hm.imshow(
                pivot_int.values, cmap=plt.cm.Greens,
                aspect="auto", origin="lower",
            )
            ax_hm.set_xticks(range(len(pivot_int.columns)))
            ax_hm.set_xticklabels(pivot_int.columns)
            ax_hm.set_yticks(range(len(pivot_int.index)))
            ax_hm.set_yticklabels(pivot_int.index)
            for i in range(len(pivot_int.index)):
                for j in range(len(pivot_int.columns)):
                    ax_hm.text(j, i, f"{pivot_int.values[i, j]:.1f}",
                               ha="center", va="center", fontsize=8)
            fig_hm.colorbar(im, ax=ax_hm, label="24/7 Coverage [%]")

        ax_hm.set_xlabel("Flexible Workload [%]", labelpad=12)
        ax_hm.set_ylabel("Extra Capacity [%]", labelpad=12)
        ax_hm.set_title(f"CAS 24/7 Renewable Coverage — {args.grid}")
        fig_hm.tight_layout()
        sweep_path = out_dir / "cas_heatmap.png"
        fig_hm.savefig(sweep_path, dpi=150)
        plt.close(fig_hm)

    print(f"Sweep plot        → {sweep_path}")


if __name__ == "__main__":
    main()
