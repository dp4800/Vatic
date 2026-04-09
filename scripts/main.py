#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
main.py — End-to-end CAS workflow orchestration.

Runs the full Carbon-Aware Scheduling pipeline:
  1.  Create DC grid                  (add_datacenter_load.py)
  2.  Run baseline DC simulation      (vatic-det)
  3.  (Optional) Renew investment opt (renew_invest.py)
  4.  (Optional) Perturb simulation   (vatic-det, for QP beta estimation)
  5.  Analyze CAS — grid-mix          (analyze_cas.py)
  6.  Analyze CAS — 24/7              (analyze_cas.py)
  7.  Analyze CAS — LP / QP           (analyze_cas.py)
  8.  Apply grid-mix shift + run      (apply_cas_shift.py + vatic-det)
  9.  Apply 24/7 shift + run          (apply_cas_shift.py + vatic-det)
  10. Apply LP/QP shift + run         (apply_cas_shift.py + vatic-det)
  11. Water use analysis              (water_use.py — all sim dirs)
  12. Net Carbon Score                (net_carbon_score.py — all sim dirs)
  13. Compare all outputs             (compare_cas_modes.py)

All parameters are read from a JSON config file (default: params.json next
to this script).  Pass --config <path> to use a different file.

Re-running is safe: each step is skipped if its primary output already exists.

Usage
-----
    python scripts/main.py
    python scripts/main.py --config scripts/params.json
    python scripts/main.py --config experiments/winter.json
"""

import argparse
import json
import hashlib
import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

# Allow sibling scripts to be imported directly
sys.path.insert(0, str(Path(__file__).parent))
import water_use        # noqa: E402  (local scripts/ module)
import net_carbon_score # noqa: E402  (local scripts/ module)
import renew_invest     # noqa: E402  (local scripts/ module)
import sim_lp_joint     # noqa: E402  (local scripts/ module)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG LOADING
# ──────────────────────────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).parent
DEFAULT_CONFIG = SCRIPTS_DIR / "params.json"


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    # strip top-level comment key if present
    cfg.pop("_comment", None)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full CAS workflow from a JSON config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, metavar="FILE",
        help=f"Path to JSON params file (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--date", type=str, default=None, metavar="YYYY-MM-DD",
        help="Override simulation start date from the config file.",
    )
    p.add_argument(
        "--days", type=int, default=None, metavar="N",
        help="Override simulation duration (days) from the config file.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

_step_counter = 0


def step(name: str) -> None:
    global _step_counter
    _step_counter += 1
    print(f"\n{'='*72}")
    print(f"  STEP {_step_counter}: {name}")
    print(f"{'='*72}")


def skip_if(path: Path, label: str) -> bool:
    """Return True (and print) when *path* already exists — caller should skip."""
    if path.exists():
        print(f"  [skip] {label} — {path} already exists")
        return True
    return False


def run(cmd, *, shell: bool = False, check: bool = True) -> None:
    """Run *cmd*, streaming output to stdout/stderr.  Exit on failure."""
    display = cmd if shell else " ".join(str(c) for c in cmd)
    print(f"  $ {display}\n")
    result = subprocess.run(cmd, shell=shell)
    if check and result.returncode != 0:
        sys.exit(f"\n[ERROR] Command failed (exit {result.returncode}).")


def py(*args) -> None:
    """Run a Python script using the same interpreter that launched main.py."""
    run([sys.executable] + [str(a) for a in args])


# ──────────────────────────────────────────────────────────────────────────────
# GRID REGISTRY
# ──────────────────────────────────────────────────────────────────────────────

_GRID_REGISTRY: dict[str, str] = {
    "Texas-7k_2030": "TX2030_Data",
    "Texas-7k":      "TX_Data",
    "RTS-GMLC":      "RTS_Data",   # also matches all RTS-GMLC-DC-* derived grids
}


def _data_dir_for(grid_name: str) -> str:
    """Return the SourceData parent directory name for a grid.

    Uses prefix matching so derived grids (e.g. 'RTS-GMLC-DC-SCALED-CAS-LP')
    resolve correctly.  Falls back to 'RTS_Data' for unknown grids.
    """
    for prefix, data_dir in sorted(_GRID_REGISTRY.items(), key=lambda x: len(x[0]), reverse=True):
        if grid_name.startswith(prefix):
            return data_dir
    return "RTS_Data"


# ──────────────────────────────────────────────────────────────────────────────
# WORKFLOW STEPS
# ──────────────────────────────────────────────────────────────────────────────

def create_dc_grid(grids_dir: Path, grid_name: str, amplitude: float,
                   base_grid: str, dc_buses: list[str],
                   variation: float, period_hours: float, phase_hours: float) -> None:
    # BusInjections dir is created by add_datacenter_load.py — reliable marker
    marker = grids_dir / grid_name / _data_dir_for(grid_name) / "timeseries_data_files" / "BusInjections"
    if skip_if(marker, f"grid {grid_name}"):
        return
    py(
        SCRIPTS_DIR / "add_datacenter_load.py",
        "--grid",         base_grid,
        "--output-grid",  grid_name,
        "--buses",        *dc_buses,
        "--amplitude-mw", str(amplitude),
        "--variation",    str(variation),
        "--period-hours", str(period_hours),
        "--phase-hours",  str(phase_hours),
        "--force",
    )


def run_vatic(grid: str, out_dir: Path, sim_date: str, sim_days: int,
              solver: str, solver_args: str, threads: int, output_detail: int,
              ruc_horizon: int, sced_horizon: int, ruc_mipgap: float,
              reserve_factor: float,
              gurobi_module: str = "gurobi/10.0.1",
              thermal_rating_scale: float = 1.0) -> None:
    """Launch vatic-det inside bash after loading the Gurobi module."""
    bash_cmd = (
        f"module load {gurobi_module} && "
        f"vatic-det {grid} {sim_date} {sim_days}"
        f" --solver {solver}"
        f" --solver-args {solver_args}"
        f" --threads {threads}"
        f" --output-detail {output_detail}"
        f" --ruc-horizon {ruc_horizon}"
        f" --sced-horizon {sced_horizon}"
        f" --ruc-mipgap {ruc_mipgap}"
        f" --reserve-factor {reserve_factor}"
        f" --thermal-rating-scale {thermal_rating_scale}"
        f" --csv --lmps"
        f" --out-dir {out_dir}"
    )
    run(bash_cmd, shell=True)


def _check_vatic_output(out_dir: Path) -> None:
    """Raise RuntimeError if vatic-det did not produce expected output files."""
    required = ["thermal_detail.csv", "renew_detail.csv", "bus_detail.csv"]
    missing  = [f for f in required if not (out_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"vatic-det output incomplete in {out_dir}: missing {missing}"
        )
    empty = [f for f in required if (out_dir / f).stat().st_size == 0]
    if empty:
        raise RuntimeError(
            f"vatic-det output files are empty in {out_dir}: {empty}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# CAS ITERATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _total_co2(sim_dir: Path, gen_csv: Path) -> float:
    """Return total CO₂ emitted (metric tonnes) from a completed vatic sim.

    Used as the convergence signal for the iterated CAS loop.
    Handles both RTS-GMLC gen.csv (has HR_avg_0 + Emissions CO2 Lbs/MMBTU)
    and Texas-7k gen.csv (lacks those columns; falls back to fuel-keyword factors).
    """
    import pandas as pd
    # Fallback factors (kg CO2/MWh) for grids without HR/emission columns.
    # Source: EPA eGRID 2023 TX median + EIA CO2 coefficients.
    _FALLBACK_KG: dict[str, float] = {
        "coal": 1078.5, "lignite": 1078.5, "subbituminous": 1078.5,
        "petroleum coke": 1021.2, "oil": 795.8,
        "natural gas": 496.3, "ng": 496.3, "gas": 496.3,
    }
    thermal = pd.read_csv(sim_dir / "thermal_detail.csv")
    gen_df  = pd.read_csv(gen_csv)
    if "HR_avg_0" in gen_df.columns and "Emissions CO2 Lbs/MMBTU" in gen_df.columns:
        hr      = pd.to_numeric(gen_df["HR_avg_0"], errors="coerce").fillna(0.0) / 1000.0
        co2_lbs = pd.to_numeric(gen_df["Emissions CO2 Lbs/MMBTU"], errors="coerce").fillna(0.0)
        ef      = hr * co2_lbs * 0.453592      # kg CO2/MWh
    else:
        fuel_col = next((c for c in gen_df.columns if c.strip().lower() == "fuel"), None)
        fuels    = gen_df[fuel_col].astype(str).str.lower() if fuel_col else pd.Series([""] * len(gen_df))
        ef       = fuels.map(lambda f: next((v for k, v in _FALLBACK_KG.items() if k in f), 0.0))
    ef.index = gen_df["GEN UID"]
    thermal["co2_kg"] = thermal["Dispatch"] * thermal["Generator"].map(ef).fillna(0.0)
    return float(thermal["co2_kg"].sum() / 1000.0)  # metric tonnes


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not args.config.exists():
        sys.exit(f"[ERROR] Config file not found: {args.config}")

    cfg = load_config(args.config)
    print(f"Config: {args.config.resolve()}")

    # ── Unpack config sections ─────────────────────────────────────────────────
    g   = cfg["grid"]
    sim = cfg["simulation"]
    cas = cfg["cas"]
    qp  = cfg["qp"]

    BASE_GRID             = g["base_grid"]
    DC_GRID               = g["dc_grid"]
    DC_BUSES              = g.get("dc_buses", [])
    AMPLITUDE_MW          = g.get("amplitude_mw", 0.0)
    VARIATION             = g.get("variation", 0.05)
    PERIOD_HOURS          = g.get("period_hours", 24.0)
    PHASE_HOURS           = g.get("phase_hours", 0.0)
    THERMAL_RATING_SCALE  = g.get("thermal_rating_scale", 1.0)

    # TX real-datacenter mode: use add_tx_datacenters.py instead of add_datacenter_load.py
    DC_CREATION_MODE = g.get("dc_creation_mode", "sinusoidal")   # "sinusoidal" | "tx_datacenters"
    DC_TOTAL_MW      = g.get("total_mw", None)
    DC_CSV           = g.get("dc_csv", None)
    # "explicit" (dc_buses list from config) | "from_grid" (read from BusInjections CSV)
    DC_BUSES_MODE    = g.get("dc_buses_mode", "explicit")

    SIM_DATE        = args.date if args.date else sim["date"]
    SIM_DAYS        = args.days if args.days else sim["days"]
    SOLVER          = sim["solver"]
    SOLVER_ARGS     = sim["solver_args"]
    THREADS         = sim["threads"]
    RUC_HORIZON     = sim["ruc_horizon"]
    SCED_HORIZON    = sim["sced_horizon"]
    RUC_MIPGAP      = sim["ruc_mipgap"]
    RESERVE_FACTOR  = sim["reserve_factor"]
    OUTPUT_DETAIL   = sim["output_detail"]
    GUROBI_MODULE   = sim.get("gurobi_module", "gurobi/10.0.1")

    CAS_EXTRA_CAP   = cas["extra_capacity_pct"]
    CAS_FLEX_RATIO  = cas["flexible_ratio_pct"]
    CAS_RENEW_FRAC  = cas["renew_fraction"]
    CAS_LP_ALPHA    = cas["lp_alpha"]
    CAS_LP_ALPHA_N  = cas["lp_alpha_steps"]
    CAS_DEFERRAL_W  = cas.get("deferral_window_hours", 12)
    CAS_LP_MODE     = cas.get("lp_mode", "iterative")   # "joint" | "iterative"
    CAS_LP_FLEX_R   = cas.get("flexible_ratio_pct", CAS_FLEX_RATIO) / 100.0
    CAS_LP_HEADROOM = cas.get("headroom_pct", CAS_EXTRA_CAP) / 100.0

    QP_ENABLED       = qp["enabled"]
    QP_PERTURB_SCALE = qp["perturb_scale"]

    # Optional sections — fall back to sensible defaults if absent
    water_cfg    = cfg.get("water", {})
    EIA_FILE     = Path(water_cfg.get("eia_file", "inputs/Cooling_Boiler_Generator_Data_Texas_2024.csv"))
    WATER_ENABLED = water_cfg.get("enabled", EIA_FILE.exists())

    analytics_cfg  = cfg.get("analytics", {})
    NCS_K          = analytics_cfg.get("ncs_K", 16)
    NCS_SCENARIOS  = analytics_cfg.get("ncs_scenarios", 500)
    NCS_ENABLED    = analytics_cfg.get("ncs_enabled", True)
    CARBON_WEIGHT  = analytics_cfg.get("carbon_weight", 0.5)
    WATER_WEIGHT   = analytics_cfg.get("water_weight",  0.5)

    renew_opt_cfg      = cfg.get("renew_opt", {})
    RENEW_OPT_ENABLED  = renew_opt_cfg.get("enabled",      False)
    RENEW_OPT_APPLY    = renew_opt_cfg.get("apply_best",   False)

    cas_iter_cfg = cfg.get("cas_iter", {})
    CAS_MAX_ITER = int(cas_iter_cfg.get("max_iter",     1))
    CAS_ITER_TOL = float(cas_iter_cfg.get("tol_co2_pct", 0.5))

    lp_iter_cfg          = cfg.get("sim_lp_iter", {})
    LP_ITER_MODE         = lp_iter_cfg.get("mode",                 "single")    # "single" | "iterative"
    LP_ITER_MAX          = int(lp_iter_cfg.get("max_iterations",    5))
    LP_ITER_THRESH       = float(lp_iter_cfg.get("convergence_threshold", 0.01))
    LP_ITER_METRIC       = lp_iter_cfg.get("convergence_metric",   "co2_delta") # "co2_delta" | "lmp_l2"
    LP_ITER_ON_OSC       = lp_iter_cfg.get("on_oscillation",       "best")      # "best" | "average"

    joint_lp_iter_cfg = cfg.get("joint_lp_iter", {})
    JOINT_LP_MAX_ITER = int(joint_lp_iter_cfg.get("max_iter", 1))
    JOINT_LP_TOL      = float(joint_lp_iter_cfg.get("tol",    0.5))  # CO₂ convergence %

    # Optional tag to namespace LP output dirs, allowing multiple sensitivity
    # variants to share the same out_root (and thus reuse a common baseline).
    CAS_LP_TAG = cfg.get("cas_lp_tag", "")
    _lp_sfx    = f"-{CAS_LP_TAG}" if CAS_LP_TAG else ""

    dc_flex_cfg  = cfg.get("dc_flexibility", {})

    # ── Derived values ─────────────────────────────────────────────────────────
    SIM_END_DATE  = (date.fromisoformat(SIM_DATE) + timedelta(days=SIM_DAYS - 1)).isoformat()

    VATIC_ROOT    = SCRIPTS_DIR.parent
    GRIDS_DIR     = VATIC_ROOT / "vatic" / "data" / "grids"
    BASE_GRID_DIR = GRIDS_DIR / BASE_GRID   # canonical gen.csv / bus.csv source
    OUT_ROOT      = Path(cfg.get("out_root", "outputs")) / SIM_DATE

    # PERTURB_GRID and CAS_GRID_* are derived after the optional renew_opt step
    # (which may override DC_GRID with an investment-enhanced grid).

    DIR_BASELINE  = OUT_ROOT / "baseline"
    DIR_PERTURB   = OUT_ROOT / "perturb"
    DIR_CAS_GM    = OUT_ROOT / "cas-gm"
    DIR_CAS_247   = OUT_ROOT / "cas-247"
    DIR_CAS_LP    = OUT_ROOT / f"cas-lp{_lp_sfx}"
    DIR_SIM_GM    = OUT_ROOT / "sim-gm"
    DIR_SIM_247   = OUT_ROOT / "sim-247"
    DIR_SIM_LP    = OUT_ROOT / f"sim-lp{_lp_sfx}"
    DIR_COMPARE   = OUT_ROOT / "compare"

    # Sim dirs that receive post-processing (baseline + all three shifted runs)
    SIM_RUNS = [
        ("baseline", DIR_BASELINE),
        ("sim-gm",   DIR_SIM_GM),
        ("sim-247",  DIR_SIM_247),
        (f"sim-lp{_lp_sfx}", DIR_SIM_LP),
    ]

    # Convenience wrapper that closes over sim params
    def vatic(grid: str, out_dir: Path) -> None:
        run_vatic(grid, out_dir, SIM_DATE, SIM_DAYS, SOLVER, SOLVER_ARGS,
                  THREADS, OUTPUT_DETAIL, RUC_HORIZON, SCED_HORIZON,
                  RUC_MIPGAP, RESERVE_FACTOR, GUROBI_MODULE,
                  thermal_rating_scale=THERMAL_RATING_SCALE)
        _check_vatic_output(out_dir)

    # gen.csv used for CO₂ convergence checks during CAS iteration
    GEN_CSV_PATH = BASE_GRID_DIR / _data_dir_for(BASE_GRID) / "SourceData" / "gen.csv"

    def _iter_cas(
        label:            str,
        cas_grid_base:    str,
        dir_sim_final:    Path,
        make_apply_args,  # callable(output_grid: str, signal_dir: Path) -> list
        make_analyze_args=None,  # callable(signal_dir: Path) -> list | None
        cas_base_dir: Path | None = None,  # output dir for updated analyze_cas results
    ) -> None:
        """Run one CAS mode with optional fixed-point iteration.

        Iterates: [analyze_cas] → apply shift → vatic → check CO₂ convergence
        → repeat.  Always writes the final result to dir_sim_final.

        When CAS_MAX_ITER == 1 the behaviour is identical to the original
        single-pass approach (no intermediate directories are created).

        If make_analyze_args is supplied and CAS_MAX_ITER > 1, re-runs
        analyze_cas.py between iterations using the previous iteration's
        simulation as the signal source. This corrects for the mismatch
        between baseline CI and post-shift actual CI (the grid-mix CO₂ sign
        issue: marginal generator after shifting may differ from the CI signal
        computed against the unshifted baseline).
        """
        signal_dir = DIR_BASELINE
        prev_co2:  float | None = None
        final_i    = 0

        for i in range(CAS_MAX_ITER):
            sfx       = f"-I{i}" if CAS_MAX_ITER > 1 else ""
            iter_grid = f"{cas_grid_base}{sfx}"
            iter_sim  = OUT_ROOT / f"sim-{label}-i{i}" if CAS_MAX_ITER > 1 else dir_sim_final

            # Re-run analyze_cas with updated signal (iterations > 0 only).
            # This corrects the CI signal: the initial analyze_cas used the
            # baseline simulation, but after the first shift the marginal
            # generator may differ, causing the predicted CO₂ savings to be
            # wrong. Re-running against the previous iteration's simulation
            # builds a more accurate signal for the next shift.
            if i > 0 and make_analyze_args is not None:
                _iter_cas_dir = (cas_base_dir or OUT_ROOT / f"cas-{label}") / f"iter{i}"
                _iter_cas_dir.mkdir(parents=True, exist_ok=True)
                if not (_iter_cas_dir / "cas_results.csv").exists():
                    _ana_extra = make_analyze_args(signal_dir)
                    if _ana_extra is not None:
                        py(
                            SCRIPTS_DIR / "analyze_cas.py",
                            "--sim-dir", signal_dir,
                            "--out-dir", _iter_cas_dir,
                            *_ana_extra,
                        )
                        print(f"  [CAS iter {i}] analyze_cas updated signal → {_iter_cas_dir}")

            # Apply CAS shift using signal from previous simulation
            marker = (GRIDS_DIR / iter_grid / _data_dir_for(iter_grid)
                      / "timeseries_data_files" / "BusInjections")
            if not marker.exists():
                py(SCRIPTS_DIR / "apply_cas_shift.py",
                   *make_apply_args(iter_grid, signal_dir))

            # Run vatic on shifted grid
            if not (iter_sim / "thermal_detail.csv").exists():
                iter_sim.mkdir(parents=True, exist_ok=True)
                vatic(iter_grid, iter_sim)

            # Convergence check (needs at least one previous result)
            curr_co2 = _total_co2(iter_sim, GEN_CSV_PATH)
            if prev_co2 is not None:
                delta_pct = abs(curr_co2 - prev_co2) / max(abs(prev_co2), 1.0) * 100
                print(f"  [CAS iter {i}/{CAS_MAX_ITER}]  Δ CO₂ = {delta_pct:.3f}%"
                      f"  (tol = {CAS_ITER_TOL}%)")
                if delta_pct < CAS_ITER_TOL:
                    print(f"  [converged at iteration {i}]")
                    final_i    = i
                    signal_dir = iter_sim
                    break

            prev_co2   = curr_co2
            signal_dir = iter_sim
            final_i    = i

        # When using intermediate dirs, move the converged result to the
        # canonical output path and clean up temporary dirs / grids.
        if CAS_MAX_ITER > 1:
            final_iter_sim = OUT_ROOT / f"sim-{label}-i{final_i}"
            shutil.move(str(final_iter_sim), str(dir_sim_final))

            # Record convergence metadata alongside the simulation output
            with open(dir_sim_final / "cas_iter_info.json", "w") as _f:
                json.dump({"converged_at": final_i,
                           "max_iter":     CAS_MAX_ITER,
                           "tol_co2_pct":  CAS_ITER_TOL}, _f, indent=2)

            # Remove intermediate simulation directories
            for j in range(final_i + 1):
                d = OUT_ROOT / f"sim-{label}-i{j}"
                if d.exists():
                    shutil.rmtree(d)

            # Promote the winning iteration grid to the canonical name;
            # delete the rest to reclaim disk space (~230 MB each).
            for j in range(CAS_MAX_ITER):
                g = GRIDS_DIR / f"{cas_grid_base}-I{j}"
                if not g.exists():
                    continue
                if j == final_i:
                    target = GRIDS_DIR / cas_grid_base
                    if not target.exists():
                        g.rename(target)
                    else:
                        shutil.rmtree(g)
                else:
                    shutil.rmtree(g)

    # ── LP-specific iterative refinement ──────────────────────────────────────

    def _lmp_l2(sim_a: Path, sim_b: Path) -> float:
        """L2 distance between LMP vectors from two VATIC bus_detail.csv files."""
        import pandas as _pd, numpy as _np
        def _load_lmps(p: Path) -> _pd.Series:
            bd = _pd.read_csv(p / "bus_detail.csv")
            bd["_dt"] = _pd.to_datetime(
                bd["Date"].astype(str) + " " + bd["Hour"].astype(int).astype(str) + ":00",
                format="%Y-%m-%d %H:%M",
            )
            return bd.set_index(["_dt", "Bus"])["LMP"]
        try:
            va = _load_lmps(sim_a)
            vb = _load_lmps(sim_b)
            common = va.index.intersection(vb.index)
            return float(_np.linalg.norm((va[common] - vb[common]).values))
        except Exception:
            return float("nan")

    def _avg_injections(grid_a: str, grid_b: str, out_grid: str) -> None:
        """Create out_grid by averaging BusInjections of grid_a and grid_b."""
        import pandas as _pd
        for ts in ("DAY_AHEAD", "REAL_TIME"):
            for suffix in ("bus_injections",):
                pa = (GRIDS_DIR / grid_a / _data_dir_for(grid_a) / "timeseries_data_files"
                      / "BusInjections" / f"{ts}_{suffix}.csv")
                pb = (GRIDS_DIR / grid_b / _data_dir_for(grid_b) / "timeseries_data_files"
                      / "BusInjections" / f"{ts}_{suffix}.csv")
                if not pa.exists() or not pb.exists():
                    continue
                da = _pd.read_csv(pa)
                db = _pd.read_csv(pb)
                id_cols = [c for c in da.columns if c in {"Year","Month","Day","Period"}]
                num_cols = [c for c in da.columns if c not in id_cols]
                avg = da.copy()
                avg[num_cols] = (da[num_cols].values + db[num_cols].values) / 2.0
                out_dir = (GRIDS_DIR / out_grid / _data_dir_for(out_grid) / "timeseries_data_files"
                           / "BusInjections")
                out_dir.mkdir(parents=True, exist_ok=True)
                avg.to_csv(out_dir / f"{ts}_{suffix}.csv", index=False)

    def _iter_lp(
        cas_grid_base:  str,
        dir_sim_final:  Path,
        make_apply_args,  # callable(output_grid: str, signal_dir: Path) -> list
    ) -> None:
        """LP-specific iterative refinement with oscillation detection and
        per-iteration diagnostics.

        Iteration 0 uses baseline LMPs (identical to single-pass).
        Iteration k > 0 uses LMPs from iteration k−1's VATIC output.
        Convergence is checked via CO₂ delta % change or LMP L2 distance.
        Oscillation is detected when the CO₂ delta worsens vs k−2.
        On oscillation (or hard stop): apply on_oscillation strategy (best/average).
        """
        import csv as _csv, numpy as _np

        diag_dir = dir_sim_final.parent / "sim_lp_iterations"
        diag_dir.mkdir(parents=True, exist_ok=True)

        signal_dir   = DIR_BASELINE
        prev_co2:    float | None  = None
        prev2_co2:   float | None  = None          # k−2 for oscillation detection
        prev_sim:    Path  | None  = None           # sim dir from previous iteration
        all_co2:     list[tuple[int, float]] = []   # (iter, co2_t)
        all_l2:      list[tuple[int, float]] = []   # (iter, lmp_l2)
        conv_rows:   list[dict]  = []
        final_i      = 0
        termination  = "hard_stop"
        n_iter       = LP_ITER_MAX if LP_ITER_MODE == "iterative" else 1

        for i in range(n_iter):
            sfx       = f"-I{i}" if n_iter > 1 else ""
            iter_grid = f"{cas_grid_base}{sfx}"
            iter_sim  = OUT_ROOT / f"sim-lp{_lp_sfx}-i{i}" if n_iter > 1 else dir_sim_final

            # Apply shift using signal from previous simulation
            marker = (GRIDS_DIR / iter_grid / _data_dir_for(iter_grid)
                      / "timeseries_data_files" / "BusInjections")
            if not marker.exists():
                py(SCRIPTS_DIR / "apply_cas_shift.py",
                   *make_apply_args(iter_grid, signal_dir))

            # Save this iteration's BusInjections schedule as diagnostic
            sched_src = GRIDS_DIR / iter_grid / _data_dir_for(iter_grid) / "timeseries_data_files" / "BusInjections"
            sched_dst = diag_dir / f"iter_{i}_schedule"
            if sched_src.exists() and not sched_dst.exists():
                shutil.copytree(sched_src, sched_dst)

            # Run VATIC
            if not (iter_sim / "thermal_detail.csv").exists():
                iter_sim.mkdir(parents=True, exist_ok=True)
                vatic(iter_grid, iter_sim)

            # CO₂ delta vs baseline
            curr_co2   = _total_co2(iter_sim, GEN_CSV_PATH)
            base_co2   = _total_co2(DIR_BASELINE, GEN_CSV_PATH)
            co2_delta  = curr_co2 - base_co2

            # LMP L2 distance (vs previous iteration, or vs baseline for i=0)
            ref_sim   = prev_sim if prev_sim is not None else DIR_BASELINE
            l2        = _lmp_l2(ref_sim, iter_sim)

            all_co2.append((i, curr_co2))
            all_l2.append((i, l2))

            # Save per-iteration scalar files
            (diag_dir / f"iter_{i}_co2_delta.txt").write_text(
                f"iter={i}  co2_delta_t={co2_delta:.2f}  baseline_co2_t={base_co2:.2f}\n"
            )
            (diag_dir / f"iter_{i}_lmp_l2.txt").write_text(
                f"iter={i}  lmp_l2_distance={l2:.4f}\n"
            )

            # Oscillation: CO₂ delta worsened vs two iterations ago
            oscillating = False
            if prev2_co2 is not None:
                # "worsened" means further from baseline (larger abs delta) and in wrong direction
                if abs(co2_delta) > abs(prev2_co2 - base_co2):
                    oscillating = True
                    print(f"  [LP iter {i}] oscillation detected — CO₂Δ={co2_delta:.1f}t"
                          f" worse than iter {i-2} ({prev2_co2-base_co2:.1f}t)")

            # Convergence check
            converged = False
            if prev_co2 is not None:
                if LP_ITER_METRIC == "lmp_l2":
                    conv_val = l2
                    tol_str  = f"L2={l2:.2f} (tol={LP_ITER_THRESH})"
                    converged = (l2 < LP_ITER_THRESH)
                else:  # co2_delta
                    delta_chg_pct = (abs(co2_delta - (prev_co2 - base_co2))
                                     / max(abs(prev_co2 - base_co2), 1.0)) * 100.0
                    tol_str  = f"ΔCO₂chg={delta_chg_pct:.3f}% (tol={LP_ITER_THRESH*100:.1f}%)"
                    converged = (delta_chg_pct / 100.0 < LP_ITER_THRESH)

                print(f"  [LP iter {i}/{n_iter}]  CO₂Δ={co2_delta:.0f}t  {tol_str}")

            conv_rows.append({
                "iter":            i,
                "co2_delta_t":     round(co2_delta, 2),
                "lmp_l2_distance": round(l2, 4) if l2 == l2 else "",
                "converged":       converged,
                "oscillating":     oscillating,
                "termination":     "",   # filled in at end
            })

            final_i    = i
            prev2_co2  = prev_co2
            prev_co2   = curr_co2
            prev_sim   = iter_sim

            if converged:
                termination = "converged"
                print(f"  [LP iter] converged at iteration {i}")
                break

            if oscillating and LP_ITER_ON_OSC == "best":
                termination = "oscillation_best"
                # Select the iterate with the best (most negative) CO₂ delta
                best_i = min(all_co2, key=lambda x: x[1])[0]
                if best_i != i:
                    print(f"  [LP iter] oscillation → selecting best iterate: {best_i}"
                          f" (CO₂Δ={all_co2[best_i][1]-base_co2:.0f}t)")
                    final_i = best_i
                break

            if oscillating and LP_ITER_ON_OSC == "average":
                termination = "oscillation_average"
                # Average the last two iteration grids
                prev_grid = f"{cas_grid_base}-I{i-1}" if n_iter > 1 else cas_grid_base
                avg_grid  = f"{cas_grid_base}-AVG"
                print(f"  [LP iter] oscillation → averaging iter {i-1} and {i}")
                if not (GRIDS_DIR / avg_grid).exists():
                    shutil.copytree(GRIDS_DIR / iter_grid, GRIDS_DIR / avg_grid)
                    _avg_injections(prev_grid, iter_grid, avg_grid)
                # Run VATIC on averaged grid
                avg_sim = OUT_ROOT / f"sim-lp{_lp_sfx}-avg"
                if not (avg_sim / "thermal_detail.csv").exists():
                    avg_sim.mkdir(parents=True, exist_ok=True)
                    vatic(avg_grid, avg_sim)
                final_i  = -1   # sentinel: use avg_sim
                prev_sim = avg_sim
                break

        # Annotate termination reason in last row
        if conv_rows:
            conv_rows[-1]["termination"] = termination

        # Write iter_convergence.csv
        conv_path = diag_dir / "iter_convergence.csv"
        with open(conv_path, "w", newline="") as _f:
            w = _csv.DictWriter(_f, fieldnames=[
                "iter","co2_delta_t","lmp_l2_distance","converged","oscillating","termination"])
            w.writeheader(); w.writerows(conv_rows)
        print(f"  [LP iter] convergence log → {conv_path}")

        # Promote winning simulation to canonical output path
        if n_iter > 1:
            if final_i == -1:
                winning_sim = OUT_ROOT / f"sim-lp{_lp_sfx}-avg"
            else:
                winning_sim = OUT_ROOT / f"sim-lp{_lp_sfx}-i{final_i}"

            if winning_sim != dir_sim_final:
                shutil.move(str(winning_sim), str(dir_sim_final))

            # Record which iter was selected
            with open(dir_sim_final / "cas_iter_info.json", "w") as _f:
                json.dump({
                    "converged_at":  final_i,
                    "max_iter":      n_iter,
                    "tol_co2_pct":   LP_ITER_THRESH * 100,
                    "termination":   termination,
                    "convergence_metric": LP_ITER_METRIC,
                    "on_oscillation": LP_ITER_ON_OSC,
                }, _f, indent=2)

            # Clean up intermediate sim dirs
            for j in range(LP_ITER_MAX):
                d = OUT_ROOT / f"sim-lp{_lp_sfx}-i{j}"
                if d.exists():
                    shutil.rmtree(d)
            avg_sim_dir = OUT_ROOT / f"sim-lp{_lp_sfx}-avg"
            if avg_sim_dir.exists() and avg_sim_dir != dir_sim_final:
                shutil.rmtree(avg_sim_dir)

            # Promote winning grid and clean intermediates
            for j in range(LP_ITER_MAX):
                g = GRIDS_DIR / f"{cas_grid_base}-I{j}"
                if not g.exists():
                    continue
                if j == (final_i if final_i >= 0 else LP_ITER_MAX - 1):
                    target = GRIDS_DIR / cas_grid_base
                    if not target.exists():
                        g.rename(target)
                    else:
                        shutil.rmtree(g)
                else:
                    shutil.rmtree(g)
            avg_grid_dir = GRIDS_DIR / f"{cas_grid_base}-AVG"
            if avg_grid_dir.exists():
                shutil.rmtree(avg_grid_dir)
        else:
            # Single-pass: still write a one-row iter_convergence.csv
            pass   # already written above

    def _iter_joint_lp(cas_grid_base: str, dir_sim_final: Path) -> None:
        """Joint LP with UC refresh between iterations.

        Iteration 0 uses the baseline unit commitment.
        Each subsequent iteration feeds the previous VATIC run's thermal
        commitment back into the joint LP so the dispatch variables stay
        consistent with the load profile they produce.

        The CI/LMP signal always comes from DIR_BASELINE (unchanged across
        iterations) so the carbon objective is stable.

        Convergence: CO₂ change between consecutive VATIC runs < JOINT_LP_TOL %.
        Cost metrics in cas_results.csv use baseline LMPs (not predicted duals).
        """
        diag_dir = dir_sim_final.parent / "sim_lp_joint_iterations"
        diag_dir.mkdir(parents=True, exist_ok=True)

        commit_dir: Path | None = None   # iteration 0 → baseline UC
        prev_co2:   float | None = None
        final_i     = 0

        for i in range(JOINT_LP_MAX_ITER):
            multi     = JOINT_LP_MAX_ITER > 1
            sfx       = f"-I{i}" if multi else ""
            iter_grid = f"{cas_grid_base}{sfx}"
            iter_sim  = OUT_ROOT / f"sim-lp{_lp_sfx}-joint-i{i}" if multi else dir_sim_final
            iter_cas  = DIR_CAS_LP / f"iter{i}"          if multi else DIR_CAS_LP
            iter_cas.mkdir(parents=True, exist_ok=True)

            # Solve joint LP (skip if grid already built from a prior run)
            _marker = (GRIDS_DIR / iter_grid / _data_dir_for(iter_grid)
                       / "timeseries_data_files" / "BusInjections")
            if not _marker.exists():
                print(f"  [joint LP iter {i}] solving LP "
                      f"(UC from {'baseline' if commit_dir is None else f'iter {i-1} VATIC'}) …")
                sim_lp_joint.run(
                    baseline_dir    = DIR_BASELINE,
                    grid            = DC_GRID,
                    output_grid     = iter_grid,
                    dc_buses        = DC_BUSES,
                    start_date      = SIM_DATE,
                    n_days          = SIM_DAYS,
                    alpha           = CAS_LP_ALPHA,
                    deferral_window = CAS_DEFERRAL_W,
                    flexible_ratio  = CAS_LP_FLEX_R,
                    headroom        = CAS_LP_HEADROOM,
                    out_dir         = iter_cas,
                    commit_dir      = commit_dir,
                )

            # Run VATIC on the shifted grid
            _td = iter_sim / "thermal_detail.csv"
            if _td.exists() and _td.stat().st_size == 0:
                _td.unlink()   # remove empty file left by a killed previous run
            if not _td.exists():
                iter_sim.mkdir(parents=True, exist_ok=True)
                vatic(iter_grid, iter_sim)

            # Convergence check via CO₂ delta (same metric as _iter_cas)
            curr_co2 = _total_co2(iter_sim, GEN_CSV_PATH)
            if prev_co2 is not None:
                delta_pct = abs(curr_co2 - prev_co2) / max(abs(prev_co2), 1.0) * 100
                print(f"  [joint LP iter {i}/{JOINT_LP_MAX_ITER}]  "
                      f"CO₂={curr_co2:.0f}t  Δ={delta_pct:.3f}%  (tol={JOINT_LP_TOL}%)")
                if delta_pct < JOINT_LP_TOL:
                    print(f"  [joint LP iter] converged at iteration {i}")
                    final_i = i
                    break

            prev_co2   = curr_co2
            commit_dir = iter_sim   # next iteration reads UC from this VATIC output
            final_i    = i

        # Promote winning iteration to canonical paths
        if JOINT_LP_MAX_ITER > 1:
            winning_sim = OUT_ROOT / f"sim-lp{_lp_sfx}-joint-i{final_i}"
            winning_cas = DIR_CAS_LP / f"iter{final_i}"
            if winning_sim.exists() and winning_sim != dir_sim_final:
                shutil.move(str(winning_sim), str(dir_sim_final))
            if winning_cas.exists():
                for fp in winning_cas.iterdir():
                    dst = DIR_CAS_LP / fp.name
                    if not dst.exists():
                        shutil.copy2(fp, dst)
            with open(dir_sim_final / "cas_iter_info.json", "w") as _f:
                json.dump({"converged_at": final_i,
                           "max_iter":     JOINT_LP_MAX_ITER,
                           "tol_co2_pct":  JOINT_LP_TOL}, _f, indent=2)
            for j in range(JOINT_LP_MAX_ITER):
                d = OUT_ROOT / f"sim-lp{_lp_sfx}-joint-i{j}"
                if d.exists() and d != dir_sim_final:
                    shutil.rmtree(d)
                g = GRIDS_DIR / f"{cas_grid_base}-I{j}"
                if not g.exists():
                    continue
                target = GRIDS_DIR / cas_grid_base
                if j == final_i and not target.exists():
                    g.rename(target)
                else:
                    shutil.rmtree(g)

        # Build comparison table against post-shift VATIC actuals
        _cmp_path = DIR_CAS_LP / "comparison_table.csv"
        if not _cmp_path.exists() and dir_sim_final.exists():
            _ddn, _gp   = sim_lp_joint._resolve_grid(DC_GRID)
            _gen_csv    = _gp / _ddn / "SourceData" / "gen.csv"
            _bus_csv    = _gp / _ddn / "SourceData" / "bus.csv"
            _branch_csv = _gp / _ddn / "SourceData" / "branch.csv"
            _inj_dir    = _gp / _ddn / "timeseries_data_files" / "BusInjections"
            _inj_csv    = list(_inj_dir.glob("DAY_AHEAD_*.csv"))[0]
            _d = sim_lp_joint.load_baseline(
                DIR_BASELINE, _gen_csv, _bus_csv, _inj_csv,
                DC_BUSES, SIM_DATE, SIM_DAYS,
            )
            _e_csv   = DIR_CAS_LP / "lp_dc_schedule.csv"
            _lmp_csv = DIR_CAS_LP / "lp_predicted_lmps.csv"
            if _e_csv.exists() and _lmp_csv.exists():
                import pandas as _pd
                _lp_res = {
                    "status":        "optimal",
                    "e_opt":         _pd.read_csv(_e_csv,   index_col=0, parse_dates=True),
                    "lmp_predicted": _pd.read_csv(_lmp_csv, index_col=0, parse_dates=True),
                }
                sim_lp_joint.build_comparison_table(
                    _lp_res, _d,
                    post_shift_dir = dir_sim_final,
                    gen_csv        = _gen_csv,
                    out_path       = _cmp_path,
                )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── 1. Create DC grid ──────────────────────────────────────────────────────
    step("Create datacenter grid")
    if DC_CREATION_MODE == "tx_datacenters":
        _tx_marker = (GRIDS_DIR / DC_GRID / _data_dir_for(DC_GRID)
                      / "timeseries_data_files" / "BusInjections")
        if skip_if(_tx_marker, f"grid {DC_GRID}"):
            pass
        else:
            _tx_args = [
                SCRIPTS_DIR / "add_tx_datacenters.py",
                "--grid",         BASE_GRID,
                "--output-grid",  DC_GRID,
                "--total-mw",     str(DC_TOTAL_MW),
                "--variation",    str(VARIATION),
                "--period-hours", str(PERIOD_HOURS),
                "--phase-hours",  str(PHASE_HOURS),
            ]
            if DC_CSV:
                _tx_args += ["--dc-csv", DC_CSV]
            py(*_tx_args)
    else:
        create_dc_grid(GRIDS_DIR, DC_GRID, AMPLITUDE_MW, BASE_GRID, DC_BUSES,
                       VARIATION, PERIOD_HOURS, PHASE_HOURS)

    # Resolve DC_BUSES from the grid's BusInjections CSV when dc_buses_mode=from_grid
    if DC_BUSES_MODE == "from_grid":
        import pandas as _pd_buses
        _inj_dir = (GRIDS_DIR / DC_GRID / _data_dir_for(DC_GRID)
                    / "timeseries_data_files" / "BusInjections")
        _inj_csvs = list(_inj_dir.glob("DAY_AHEAD_*.csv"))
        if _inj_csvs:
            _inj_df   = _pd_buses.read_csv(_inj_csvs[0], nrows=1)
            _time_cols = {"Year", "Month", "Day", "Period",
                          "Issue_time", "Forecast_time", "Time"}
            DC_BUSES = [c for c in _inj_df.columns if c not in _time_cols]
            print(f"  [dc_buses] resolved {len(DC_BUSES)} buses from BusInjections CSV")
        else:
            sys.exit(f"[ERROR] dc_buses_mode=from_grid but no BusInjections CSV in {_inj_dir}")

    # ── 2. Run baseline DC simulation ──────────────────────────────────────────
    step("Run baseline DC simulation")
    if not skip_if(DIR_BASELINE / "thermal_detail.csv", "baseline simulation"):
        DIR_BASELINE.mkdir(parents=True, exist_ok=True)
        vatic(DC_GRID, DIR_BASELINE)

    # ── 3. (Optional) Renewable investment optimisation ────────────────────────
    # Finds the optimal (wind_mw, solar_mw, battery_mw) via a 3-stage simulation-
    # optimisation loop.  If apply_best=true the best investment grid replaces
    # DC_GRID for all downstream CAS analysis steps.
    if RENEW_OPT_ENABLED:
        step("Renewable investment optimisation")
        renew_out     = OUT_ROOT / "renew_opt"
        results_csv   = renew_out / "renew_opt_results.csv"
        _gen_csv      = BASE_GRID_DIR / _data_dir_for(BASE_GRID) / "SourceData" / "gen.csv"
        _sim_params   = {
            "solver":        SOLVER,
            "solver_args":   SOLVER_ARGS,
            "threads":       THREADS,
            "ruc_horizon":   RUC_HORIZON,
            "sced_horizon":  SCED_HORIZON,
            "ruc_mipgap":    RUC_MIPGAP,
            "reserve_factor": RESERVE_FACTOR,
            "output_detail": OUTPUT_DETAIL,
        }
        _bounds  = {
            "wind_max":    renew_opt_cfg.get("wind_max_mw",    5000.0),
            "solar_max":   renew_opt_cfg.get("solar_max_mw",   5000.0),
            "battery_max": renew_opt_cfg.get("battery_max_mw", 2000.0),
        }
        _weights = {
            "lambda_carbon":      renew_opt_cfg.get("lambda_carbon",      0.5),
            "lambda_cost":        renew_opt_cfg.get("lambda_cost",        0.3),
            "lambda_reliability": renew_opt_cfg.get("lambda_reliability", 0.2),
            "lambda_water":       renew_opt_cfg.get("lambda_water",       0.0),
        }

        if results_csv.exists():
            import pandas as _pd
            _df_opt = _pd.read_csv(results_csv)
            if RENEW_OPT_APPLY and not _df_opt.empty:
                DC_GRID = _df_opt.iloc[0]["output_grid"]
                print(f"  [skip] renew_opt done — best grid applied: {DC_GRID}")
            else:
                print(f"  [skip] renew_opt already done — {results_csv}")
        else:
            _mw_date_strs = renew_opt_cfg.get("multi_week_dates", [])
            _multi_week_configs = (
                [
                    (d, Path(cfg.get("out_root", "outputs")) / d / "baseline")
                    for d in _mw_date_strs
                ]
                if _mw_date_strs else None
            )
            _df_opt = renew_invest.run_optimisation(
                source_grid        = BASE_GRID,
                dc_grid            = DC_GRID,
                buses              = DC_BUSES,
                gen_csv            = _gen_csv,
                sim_date           = SIM_DATE,
                sim_days           = SIM_DAYS,
                sim_params         = _sim_params,
                out_dir            = renew_out,
                baseline_dir       = DIR_BASELINE,
                bounds             = _bounds,
                weights            = _weights,
                reliability_penalty  = renew_opt_cfg.get("reliability_penalty",  1000.0),
                curtailment_penalty  = renew_opt_cfg.get("curtailment_penalty",     0.0),
                n_workers          = renew_opt_cfg.get("workers", 1),
                stage_n            = (
                    renew_opt_cfg.get("stage1_n", 30),
                    renew_opt_cfg.get("stage2_n", 20),
                    renew_opt_cfg.get("stage3_n", 10),
                ),
                seed               = renew_opt_cfg.get("seed", 42),
                capex_cfg          = cfg.get("capex", {}),
                tx_max_hops        = renew_opt_cfg.get("tx_max_hops", 0),
                multi_week_configs = _multi_week_configs,
            )
            if RENEW_OPT_APPLY and not _df_opt.empty and _df_opt.iloc[0]["status"] == "ok":
                DC_GRID = _df_opt.iloc[0]["output_grid"]
                print(f"  [renew_opt] Best investment grid applied: {DC_GRID}")

        # Write renew_capex.json when an investment grid was applied.
        # This records what was actually built so downstream comparisons can
        # include the annualised CAPEX in their cost totals.
        RENEW_CAPEX_JSON = OUT_ROOT / "renew_capex.json"
        if RENEW_OPT_APPLY and not RENEW_CAPEX_JSON.exists():
            try:
                import pandas as _pd
                _df_opt = _pd.read_csv(results_csv)
                if not _df_opt.empty and _df_opt.iloc[0].get("status") == "ok":
                    _best   = _df_opt.iloc[0]
                    _x      = renew_invest.InvestmentVector(
                        wind_mw    = float(_best.get("wind_mw",    0)),
                        solar_mw   = float(_best.get("solar_mw",   0)),
                        battery_mw = float(_best.get("battery_mw", 0)),
                    )
                    _capex_cfg   = cfg.get("capex", {})
                    _capex_ann   = renew_invest._annualized_capex(_x, _capex_cfg, SIM_DAYS)
                    _embodied_ann = renew_invest._annualized_embodied_co2(_x, _capex_cfg, SIM_DAYS)
                    with open(RENEW_CAPEX_JSON, "w") as _f:
                        json.dump({
                            "wind_mw":             _x.wind_mw,
                            "solar_mw":            _x.solar_mw,
                            "battery_mw":          _x.battery_mw,
                            "capex_annual_usd":    round(_capex_ann, 0),
                            "embodied_co2_annual_t": round(_embodied_ann, 2),
                            "sim_days":            SIM_DAYS,
                        }, _f, indent=2)
                    print(f"  [renew_opt] CAPEX summary → {RENEW_CAPEX_JSON}")
            except Exception as _e:
                print(f"  [warn] Could not write renew_capex.json: {_e}")

    # ── Write DC flexibility CAPEX JSON ────────────────────────────────────────
    # Over-provisioning cost: DC needs extra_capacity_pct% headroom above nominal
    # load to support flexible workload shifting. This infrastructure cost is
    # borne by any scenario that uses CAS (all non-baseline modes).
    DC_FLEX_CAPEX_JSON = OUT_ROOT / "dc_flex_capex.json"
    if not DC_FLEX_CAPEX_JSON.exists():
        _dc_peak_mw  = AMPLITUDE_MW * len(DC_BUSES)
        _extra_mw    = _dc_peak_mw * CAS_EXTRA_CAP / 100.0
        _s_capex_kw  = float(dc_flex_cfg.get("server_capex_per_kw",  2000.0))
        _s_life_yr   = int(dc_flex_cfg.get("server_life_yr",         5))
        _s_rate      = float(dc_flex_cfg.get("discount_rate",        0.07))
        _total_capex = _extra_mw * 1000.0 * _s_capex_kw
        _crf = _s_rate * (1 + _s_rate) ** _s_life_yr / ((1 + _s_rate) ** _s_life_yr - 1)
        _dc_capex_ann = _total_capex * _crf
        _dc_capex_sim = _dc_capex_ann * SIM_DAYS / 365.0
        with open(DC_FLEX_CAPEX_JSON, "w") as _f:
            json.dump({
                "dc_peak_mw":              round(_dc_peak_mw, 1),
                "extra_capacity_mw":       round(_extra_mw, 1),
                "server_capex_per_kw_usd": _s_capex_kw,
                "server_life_yr":          _s_life_yr,
                "dc_flex_capex_annual_usd": round(_dc_capex_ann, 0),
                "dc_flex_capex_sim_usd":    round(_dc_capex_sim, 2),
                "sim_days":                 SIM_DAYS,
            }, _f, indent=2)
        print(f"  DC flexibility CAPEX: {_extra_mw:.1f} MW extra → "
              f"${_dc_capex_ann/1e6:.2f}M/yr  ({DC_FLEX_CAPEX_JSON})")

    # Derived grid names — include date tag so parallel/sequential multi-month
    # runs don't share CAS grids whose BusInjections are date-specific.
    _DATE_TAG     = SIM_DATE.replace("-", "")   # e.g. "20200106"
    PERTURB_GRID  = f"{DC_GRID}-PERTURB"
    CAS_GRID_GM   = f"{DC_GRID}-CAS-GM-{_DATE_TAG}"
    CAS_GRID_247  = f"{DC_GRID}-CAS-247-{_DATE_TAG}"
    _lp_key   = f"{CAS_LP_ALPHA:.4f}_{CAS_FLEX_RATIO}_{CAS_DEFERRAL_W}_{JOINT_LP_MAX_ITER}"
    _lp_hash  = hashlib.sha256(_lp_key.encode()).hexdigest()[:8]
    CAS_GRID_LP   = f"{DC_GRID}-CAS-LP-{_DATE_TAG}-{_lp_hash}"

    # ── 4. (Optional) Perturb simulation for QP β estimation ──────────────────
    if QP_ENABLED:
        step("Create perturb grid and run perturb simulation")
        create_dc_grid(GRIDS_DIR, PERTURB_GRID, AMPLITUDE_MW * QP_PERTURB_SCALE,
                       BASE_GRID, DC_BUSES, VARIATION, PERIOD_HOURS, PHASE_HOURS)
        if not skip_if(DIR_PERTURB / "thermal_detail.csv", "perturb simulation"):
            DIR_PERTURB.mkdir(parents=True, exist_ok=True)
            vatic(PERTURB_GRID, DIR_PERTURB)

    # ── 5. Analyze CAS — grid-mix ─────────────────────────────────────────────
    step("Analyze CAS — grid-mix")
    if not skip_if(DIR_CAS_GM / "cas_results.csv", "CAS grid-mix analysis"):
        gm_args = [
            SCRIPTS_DIR / "analyze_cas.py",
            "--sim-dir",                  DIR_BASELINE,
            "--grid",                     DC_GRID,
            "--buses",                    *DC_BUSES,
            "--mode",                     "grid-mix",
            "--start-date",               SIM_DATE,
            "--end-date",                 SIM_END_DATE,
            "--max-extra-capacity",       str(CAS_EXTRA_CAP),
            "--max-flexible-work-ratio",  str(CAS_FLEX_RATIO),
            "--deferral-window",          str(CAS_DEFERRAL_W),
            "--out-dir",                  DIR_CAS_GM,
        ]
        if QP_ENABLED:
            gm_args += ["--perturb-sim-dir", DIR_PERTURB]
        py(*gm_args)

    # ── 6. Analyze CAS — 24/7 ─────────────────────────────────────────────────
    step("Analyze CAS — 24/7")
    if not skip_if(DIR_CAS_247 / "cas_results.csv", "CAS 24/7 analysis"):
        s247_args = [
            SCRIPTS_DIR / "analyze_cas.py",
            "--sim-dir",                  DIR_BASELINE,
            "--grid",                     DC_GRID,
            "--buses",                    *DC_BUSES,
            "--mode",                     "24_7",
            "--start-date",               SIM_DATE,
            "--end-date",                 SIM_END_DATE,
            "--renew-fraction",           str(CAS_RENEW_FRAC),
            "--max-extra-capacity",       str(CAS_EXTRA_CAP),
            "--max-flexible-work-ratio",  str(CAS_FLEX_RATIO),
            "--deferral-window",          str(CAS_DEFERRAL_W),
            "--out-dir",                  DIR_CAS_247,
        ]
        if QP_ENABLED:
            s247_args += ["--perturb-sim-dir", DIR_PERTURB]
        py(*s247_args)

    # ── 7. Analyze CAS — LP (price-taking / joint) or QP (price-anticipating) ──
    step("Analyze CAS — LP / QP")
    if CAS_LP_MODE == "joint":
        # LP solve and VATIC are interleaved in step 10 via _iter_joint_lp().
        # For JOINT_LP_MAX_ITER == 1 the behaviour is identical to the original
        # single-pass; for > 1 the UC is refreshed between iterations.
        DIR_CAS_LP.mkdir(parents=True, exist_ok=True)
        print(f"  [joint LP] LP solve and simulation handled in step 10 "
              f"(max_iter={JOINT_LP_MAX_ITER}, tol={JOINT_LP_TOL}%)")
    else:
        if not skip_if(DIR_CAS_LP / "cas_results.csv", "CAS LP/QP analysis"):
            lp_args = [
                SCRIPTS_DIR / "analyze_cas.py",
                "--sim-dir",        DIR_BASELINE,
                "--grid",           DC_GRID,
                "--buses",          *DC_BUSES,
                "--mode",           "lp",
                "--start-date",     SIM_DATE,
                "--end-date",       SIM_END_DATE,
                "--extra-capacity", str(CAS_EXTRA_CAP),
                "--alpha-steps",    str(CAS_LP_ALPHA_N),
                "--out-dir",        DIR_CAS_LP,
            ]
            if QP_ENABLED:
                lp_args += ["--perturb-sim-dir", DIR_PERTURB]
            py(*lp_args)

    # ── 8. Apply grid-mix shift → new grid → shifted simulation ───────────────
    step("Apply CAS grid-mix shift and run shifted simulation"
         + (f" (up to {CAS_MAX_ITER} iterations)" if CAS_MAX_ITER > 1 else ""))
    if not skip_if(DIR_SIM_GM / "thermal_detail.csv", "shifted grid-mix simulation"):
        if CAS_MAX_ITER <= 1:
            gm_marker = GRIDS_DIR / CAS_GRID_GM / _data_dir_for(CAS_GRID_GM) / "timeseries_data_files" / "BusInjections"
            if not gm_marker.exists():
                _gm_apply = [
                    SCRIPTS_DIR / "apply_cas_shift.py",
                    "--source-grid",         DC_GRID,
                    "--output-grid",         CAS_GRID_GM,
                    "--sim-dir",             DIR_BASELINE,
                    "--buses",               *DC_BUSES,
                    "--mode",                "grid-mix",
                    "--start-date",          SIM_DATE,
                    "--end-date",            SIM_END_DATE,
                    "--extra-capacity",      str(CAS_EXTRA_CAP),
                    "--flexible-work-ratio", str(CAS_FLEX_RATIO),
                    "--deferral-window",     str(CAS_DEFERRAL_W),
                ]
                if QP_ENABLED:
                    _gm_apply += ["--perturb-sim-dir", DIR_PERTURB]
                py(*_gm_apply)
            DIR_SIM_GM.mkdir(parents=True, exist_ok=True)
            vatic(CAS_GRID_GM, DIR_SIM_GM)
            shutil.rmtree(GRIDS_DIR / CAS_GRID_GM, ignore_errors=True)
            shutil.rmtree(GRIDS_DIR / "initial-state" / CAS_GRID_GM, ignore_errors=True)
        else:
            _iter_cas(
                "gm", CAS_GRID_GM, DIR_SIM_GM,
                make_apply_args=lambda grid, sig: [
                    "--source-grid",         DC_GRID,
                    "--output-grid",         grid,
                    "--sim-dir",             str(sig),
                    "--buses",               *DC_BUSES,
                    "--mode",                "grid-mix",
                    "--start-date",          SIM_DATE,
                    "--end-date",            SIM_END_DATE,
                    "--extra-capacity",      str(CAS_EXTRA_CAP),
                    "--flexible-work-ratio", str(CAS_FLEX_RATIO),
                    "--deferral-window",     str(CAS_DEFERRAL_W),
                    # Marginal CI for i=0 (signal=baseline); for i>0 the
                    # shifted sim's average CI is the appropriate signal.
                    *(["--perturb-sim-dir", str(DIR_PERTURB)]
                      if QP_ENABLED and sig == DIR_BASELINE else []),
                ],
                make_analyze_args=lambda prev_sig_dir: [
                    # Re-analyze with updated CI signal between iterations.
                    # Only mode/grid/bus args; --sim-dir and --out-dir
                    # are supplied by _iter_cas itself.
                    # Note: do NOT pass --perturb-sim-dir here — we have no
                    # perturbed version of the iterated simulation dirs.
                    # Average CI of the shifted sim converges toward the true
                    # marginal signal over iterations.
                    "--grid",                    DC_GRID,
                    "--buses",                   *DC_BUSES,
                    "--mode",                    "grid-mix",
                    "--start-date",              SIM_DATE,
                    "--end-date",                SIM_END_DATE,
                    "--max-extra-capacity",      str(CAS_EXTRA_CAP),
                    "--max-flexible-work-ratio", str(CAS_FLEX_RATIO),
                    "--deferral-window",         str(CAS_DEFERRAL_W),
                ],
                cas_base_dir=DIR_CAS_GM,
            )

    # ── 9. Apply 24/7 shift → new grid → shifted simulation ───────────────────
    step("Apply CAS 24/7 shift and run shifted simulation"
         + (f" (up to {CAS_MAX_ITER} iterations)" if CAS_MAX_ITER > 1 else ""))
    if not skip_if(DIR_SIM_247 / "thermal_detail.csv", "shifted 24/7 simulation"):
        if CAS_MAX_ITER <= 1:
            g247_marker = GRIDS_DIR / CAS_GRID_247 / _data_dir_for(CAS_GRID_247) / "timeseries_data_files" / "BusInjections"
            if not g247_marker.exists():
                py(
                    SCRIPTS_DIR / "apply_cas_shift.py",
                    "--source-grid",         DC_GRID,
                    "--output-grid",         CAS_GRID_247,
                    "--sim-dir",             DIR_BASELINE,
                    "--buses",               *DC_BUSES,
                    "--mode",                "24_7",
                    "--start-date",          SIM_DATE,
                    "--end-date",            SIM_END_DATE,
                    "--renew-fraction",      str(CAS_RENEW_FRAC),
                    "--extra-capacity",      str(CAS_EXTRA_CAP),
                    "--flexible-work-ratio", str(CAS_FLEX_RATIO),
                    "--deferral-window",     str(CAS_DEFERRAL_W),
                )
            DIR_SIM_247.mkdir(parents=True, exist_ok=True)
            vatic(CAS_GRID_247, DIR_SIM_247)
            shutil.rmtree(GRIDS_DIR / CAS_GRID_247, ignore_errors=True)
            shutil.rmtree(GRIDS_DIR / "initial-state" / CAS_GRID_247, ignore_errors=True)
        else:
            _iter_cas(
                "247", CAS_GRID_247, DIR_SIM_247,
                lambda grid, sig: [
                    "--source-grid",         DC_GRID,
                    "--output-grid",         grid,
                    "--sim-dir",             str(sig),
                    "--buses",               *DC_BUSES,
                    "--mode",                "24_7",
                    "--start-date",          SIM_DATE,
                    "--end-date",            SIM_END_DATE,
                    "--renew-fraction",      str(CAS_RENEW_FRAC),
                    "--extra-capacity",      str(CAS_EXTRA_CAP),
                    "--flexible-work-ratio", str(CAS_FLEX_RATIO),
                    "--deferral-window",     str(CAS_DEFERRAL_W),
                ],
            )

    # ── 10. Apply LP/QP shift → new grid → shifted simulation ─────────────────
    step("Apply CAS LP/QP shift and run shifted simulation"
         + (f" (up to {CAS_MAX_ITER} iterations)" if CAS_MAX_ITER > 1 else ""))
    if not skip_if(DIR_SIM_LP / "thermal_detail.csv", "shifted LP/QP simulation"):
        if CAS_LP_MODE == "joint":
            _iter_joint_lp(CAS_GRID_LP, DIR_SIM_LP)
            shutil.rmtree(GRIDS_DIR / CAS_GRID_LP, ignore_errors=True)
            shutil.rmtree(GRIDS_DIR / "initial-state" / CAS_GRID_LP, ignore_errors=True)
        else:
            # Shared apply-args factory for both single and iterative LP modes
            def _lp_apply_args(grid: str, sig: Path) -> list:
                args = [
                    "--source-grid",         DC_GRID,
                    "--output-grid",         grid,
                    "--sim-dir",             str(sig),
                    "--buses",               *DC_BUSES,
                    "--mode",                "lp",
                    "--start-date",          SIM_DATE,
                    "--end-date",            SIM_END_DATE,
                    "--alpha",               str(CAS_LP_ALPHA),
                    "--extra-capacity",      str(CAS_EXTRA_CAP),
                    "--flexible-work-ratio", str(CAS_FLEX_RATIO),
                ]
                if QP_ENABLED:
                    args += ["--perturb-sim-dir", str(DIR_PERTURB)]
                return args

            if LP_ITER_MODE == "iterative":
                _iter_lp(CAS_GRID_LP, DIR_SIM_LP, _lp_apply_args)
            else:
                # Single-pass (default): apply once from baseline, run VATIC
                lp_marker = GRIDS_DIR / CAS_GRID_LP / _data_dir_for(CAS_GRID_LP) / "timeseries_data_files" / "BusInjections"
                if not lp_marker.exists():
                    py(SCRIPTS_DIR / "apply_cas_shift.py",
                       *_lp_apply_args(CAS_GRID_LP, DIR_BASELINE))
                DIR_SIM_LP.mkdir(parents=True, exist_ok=True)
                _td_lp = DIR_SIM_LP / "thermal_detail.csv"
                if _td_lp.exists() and _td_lp.stat().st_size == 0:
                    _td_lp.unlink()
                if not _td_lp.exists():
                    vatic(CAS_GRID_LP, DIR_SIM_LP)
                shutil.rmtree(GRIDS_DIR / CAS_GRID_LP, ignore_errors=True)
                shutil.rmtree(GRIDS_DIR / "initial-state" / CAS_GRID_LP, ignore_errors=True)
                # Single-pass still writes a one-row iter_convergence.csv
                import csv as _csv
                diag_dir = DIR_SIM_LP.parent / "sim_lp_iterations"
                diag_dir.mkdir(parents=True, exist_ok=True)
                conv_path = diag_dir / "iter_convergence.csv"
                if not conv_path.exists() and (DIR_SIM_LP / "thermal_detail.csv").exists():
                    base_co2 = _total_co2(DIR_BASELINE, GEN_CSV_PATH)
                    curr_co2 = _total_co2(DIR_SIM_LP,   GEN_CSV_PATH)
                    with open(conv_path, "w", newline="") as _f:
                        w = _csv.DictWriter(_f, fieldnames=[
                            "iter","co2_delta_t","lmp_l2_distance",
                            "converged","oscillating","termination"])
                        w.writeheader()
                        w.writerow({
                            "iter": 0,
                            "co2_delta_t": round(curr_co2 - base_co2, 2),
                            "lmp_l2_distance": "",
                            "converged": True,
                            "oscillating": False,
                            "termination": "single_pass",
                        })

    # ── 11. Water use — all simulation runs ───────────────────────────────────
    # Runs before compare so that water metrics are available in the comparison.
    step("Water use analysis (all simulation runs)")
    RATE_CSV = OUT_ROOT / "gen_water_rates.csv"
    if WATER_ENABLED:
        if not EIA_FILE.exists():
            print(f"  [warn] EIA file not found ({EIA_FILE}) — skipping water analysis")
        else:
            if not RATE_CSV.exists():
                import pandas as pd
                gen_df = pd.read_csv(BASE_GRID_DIR / _data_dir_for(BASE_GRID) / "SourceData" / "gen.csv")
                water_use.build_water_rates(EIA_FILE, gen_df, out_csv=RATE_CSV)
            else:
                print(f"  [skip] gen_water_rates.csv already exists — {RATE_CSV}")

            # Load rates once; reuse the DataFrame for all sim runs to avoid
            # re-reading the CSV on every call.
            import pandas as _pd_water
            _water_rates_df = _pd_water.read_csv(RATE_CSV) if RATE_CSV.exists() else None

            for label, sim_dir in SIM_RUNS:
                if not sim_dir.exists():
                    print(f"  [skip] {label} — simulation directory absent")
                    continue
                water_dir = OUT_ROOT / "water" / label
                marker    = water_dir / "system_water_hourly.csv"
                if skip_if(marker, f"water/{label}"):
                    continue
                water_use.run(
                    sim_dir        = sim_dir,
                    grid_dir       = BASE_GRID_DIR,
                    eia_path       = EIA_FILE,
                    out_dir        = water_dir,
                    rate_csv       = RATE_CSV,
                    water_rates_df = _water_rates_df,
                )
    else:
        print("  [skip] water analysis disabled (no eia_file configured)")

    # ── 12. Environmental score (NCS + water) — all simulation runs ───────────
    # Runs before compare so that phi_hourly CSVs are available for Policy Fig 1.
    step("Environmental score — Net Carbon Score + water intensity (all runs)")
    if NCS_ENABLED:
        water_rates_df = None
        if RATE_CSV.exists():
            import pandas as pd
            water_rates_df = pd.read_csv(RATE_CSV)

        # Compute the K-means partition once from the baseline grid so that
        # every scenario uses identical cluster boundaries.
        _ncs_bus2mg = None
        _bus_df, _ = net_carbon_score._load_grid_data(BASE_GRID_DIR)
        _ncs_bus2mg = net_carbon_score._kmeans_partition(_bus_df, NCS_K)
        print(f"  [NCS] K-means partition fixed on baseline grid (K={NCS_K})")

        for label, sim_dir in SIM_RUNS:
            if not sim_dir.exists():
                print(f"  [skip] {label} — simulation directory absent")
                continue
            out_csv        = OUT_ROOT / "environmental_score" / f"{label}.csv"
            out_hourly_csv = OUT_ROOT / "environmental_score" / f"{label}_phi_hourly.csv"
            if skip_if(out_csv, f"environmental_score/{label}"):
                continue
            net_carbon_score.run(
                sim_dir        = sim_dir,
                grid_dir       = BASE_GRID_DIR,
                K              = NCS_K,
                S              = NCS_SCENARIOS,
                out_csv        = out_csv,
                out_hourly_csv = out_hourly_csv,
                water_rates    = water_rates_df,
                carbon_weight  = CARBON_WEIGHT,
                water_weight   = WATER_WEIGHT,
                bus2mg         = _ncs_bus2mg,
            )
    else:
        print("  [skip] environmental score disabled (analytics.ncs_enabled = false)")

    # ── 13. Compare all outputs ────────────────────────────────────────────────
    # Runs last so water and env-score data are available for comparison plots
    # and Policy Analysis Figure 1 (requires phi_hourly CSVs from step 12).
    step("Compare all CAS outputs")
    DIR_COMPARE.mkdir(parents=True, exist_ok=True)
    compare_args = [
        SCRIPTS_DIR / "compare_cas_modes.py",
        "--cas-dirs",
            DIR_CAS_GM, DIR_CAS_247, DIR_CAS_LP,
        "--cas-labels",
            "grid-mix", "24_7", "lp",
        "--sim-dirs",
            DIR_BASELINE, DIR_SIM_GM, DIR_SIM_247, DIR_SIM_LP,
        "--sim-labels",
            "baseline", "sim-gm", "sim-247", f"sim-lp{_lp_sfx}",
        "--baseline-label", "baseline",
        "--out-dir",    DIR_COMPARE,
        "--gen-csv",    GEN_CSV_PATH,
    ]
    water_base     = OUT_ROOT / "water"
    env_score_base = OUT_ROOT / "environmental_score"
    if WATER_ENABLED and water_base.exists():
        compare_args += ["--water-dir", water_base]
    if NCS_ENABLED and env_score_base.exists():
        compare_args += ["--env-score-dir", env_score_base]
    if RENEW_OPT_ENABLED and RENEW_OPT_APPLY and (OUT_ROOT / "renew_capex.json").exists():
        compare_args += ["--renew-capex-json", OUT_ROOT / "renew_capex.json"]
    if DC_FLEX_CAPEX_JSON.exists():
        compare_args += ["--dc-flex-capex-json", DC_FLEX_CAPEX_JSON]
    compare_args += ["--no-plots"]
    run([sys.executable] + [str(a) for a in compare_args], check=False)

    # ── STEP 13: Standardised per-week outputs (panels, deltas, diagnostics) ──
    step("Per-week standardised outputs (panels, weekly_deltas, diagnostics)")
    try:
        import sys as _sys
        _sys.path.insert(0, str(SCRIPTS_DIR))
        import run_outputs as _ro
        _ro.run(OUT_ROOT, figures=False)
    except Exception as _e:
        print(f"  [warn] run_outputs failed: {_e}")

    print(f"\n{'='*72}")
    print(f"  DONE — results in {OUT_ROOT}/")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
