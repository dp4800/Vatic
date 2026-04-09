#!/usr/bin/env python3
"""
generate_tx_configs.py — Generate all config JSONs and tx_manifest.csv for the TX CAS study.

Study design (mirrors CAS_STUDY_RTS_2020 but for Texas-7k, 2018 data):
------------------------------------------------------------------------
Baseline (4 runs):
  Jan, Apr, Jul, Oct  ×  default params (flex=30%, α=0.5, defer=12h)

Sensitivity (24 runs):
  Jan + Jul  ×  one parameter varied at a time, all others at default:
    Load growth : +25% DC load  (Texas-7k-DC-SCALED, total_mw=25375)
    Flex        : 20%, 40%, 50%
    Alpha       : 0.00, 0.25, 0.75, 1.00
    Deferral    : 4h, 8h, 18h, 24h

Usage:
    module load anaconda3/2024.10
    python scripts/study/generate_tx_configs.py

NOTE: Before submitting sensitivity jobs, pre-build the SCALED DC grid:
    python scripts/main.py --config <any load_growth config>
  or run add_tx_datacenters.py directly with --total-mw 25375.
"""

import csv, json
from pathlib import Path

REPO    = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "scripts" / "study" / "tx_configs"
CFG_DIR.mkdir(parents=True, exist_ok=True)

# ── TX-specific constants ──────────────────────────────────────────────────
BASE_GRID      = "Texas-7k"
DC_GRID_BASE   = "Texas-7k-DC"           # 20 300 MW baseline DC load
DC_GRID_SCALED = "Texas-7k-DC-SCALED"    # 25 375 MW (+25%) for load_growth
DC_TOTAL_MW    = 20300
DC_TOTAL_SCALED = 25375                  # 20300 × 1.25

DC_CSV = "inputs/TX_Data_Center_Info.csv"

THERMAL_RATING_SCALE = 1.15   # relax N-1 contingency limits → normal operating limits

# ── Defaults ───────────────────────────────────────────────────────────────
DEF = dict(
    dc_grid   = DC_GRID_BASE,
    total_mw  = DC_TOTAL_MW,
    flex      = 30,
    alpha     = 0.5,
    deferral  = 12,
)

# ── Month tables ───────────────────────────────────────────────────────────
# TX data: 2018-01-02 through 2018-12-30
BASELINE_MONTHS = [
    ("2018-01-02", 30, "Jan"),
    ("2018-04-01", 30, "Apr"),
    ("2018-07-01", 31, "Jul"),
    ("2018-10-01", 31, "Oct"),
]
SENS_MONTHS = [
    ("2018-01-02", 30, "2018-01"),
    ("2018-07-01", 31, "2018-07"),
]

# ── Sensitivity groups ─────────────────────────────────────────────────────
SENS_GROUPS = [
    ("load_growth", "amp_25pct",    dict(dc_grid=DC_GRID_SCALED, total_mw=DC_TOTAL_SCALED)),
    ("flex",        "flex_20pct",   dict(flex=20)),
    ("flex",        "flex_40pct",   dict(flex=40)),
    ("flex",        "flex_50pct",   dict(flex=50)),
    ("alpha",       "alpha_0.00",   dict(alpha=0.00)),
    ("alpha",       "alpha_0.25",   dict(alpha=0.25)),
    ("alpha",       "alpha_0.75",   dict(alpha=0.75)),
    ("alpha",       "alpha_1.00",   dict(alpha=1.00)),
    ("deferral",    "deferral_4h",  dict(deferral=4)),
    ("deferral",    "deferral_8h",  dict(deferral=8)),
    ("deferral",    "deferral_18h", dict(deferral=18)),
    ("deferral",    "deferral_24h", dict(deferral=24)),
]


def make_config(date, days, dc_grid, total_mw, flex, alpha, deferral, out_root) -> dict:
    return {
        "grid": {
            "base_grid":            BASE_GRID,
            "dc_grid":              dc_grid,
            "dc_creation_mode":     "tx_datacenters",
            "dc_buses_mode":        "from_grid",
            "dc_csv":               DC_CSV,
            "total_mw":             total_mw,
            "variation":            0.05,
            "period_hours":         24.0,
            "phase_hours":          0.0,
            "thermal_rating_scale": THERMAL_RATING_SCALE,
        },
        "simulation": {
            "solver":         "gurobi",
            "gurobi_module":  "gurobi/10.0.1",
            "solver_args":    "Cuts=1 Presolve=1 Heuristics=0.03 TimeLimit=300",
            "threads":        8,
            "ruc_horizon":    24,
            "sced_horizon":   1,
            "ruc_mipgap":     0.05,
            "reserve_factor": 0.05,
            "output_detail":  1,
            "date":           date,
            "days":           days,
        },
        "cas": {
            "extra_capacity_pct":    flex,
            "flexible_ratio_pct":    flex,
            "renew_fraction":        0.1,
            "lp_alpha":              alpha,
            "lp_alpha_steps":        3,
            "deferral_window_hours": deferral,
            "lp_mode":               "iterative",
            "headroom_pct":          flex,
        },
        "qp":        {"enabled": False, "perturb_scale": 1.05},
        "renew_opt": {"enabled": False, "apply_best":    False},
        "cas_iter":  {"max_iter": 1,    "tol_co2_pct":   0.5},
        "sim_lp_iter":   {"mode": "single", "max_iterations": 1},
        "joint_lp_iter": {"max_iter": 1,    "tol": 0.5},
        "analytics": {
            "ncs_enabled": False, "ncs_K": 16, "ncs_scenarios": 100,
            "carbon_weight": 0.5, "water_weight": 0.5,
        },
        "out_root": out_root,
    }


# ── Generate configs and manifest ──────────────────────────────────────────
manifest = []
run_id   = 0

# ── Baseline ──────────────────────────────────────────────────────────────
for date, days, mo_abbrev in BASELINE_MONTHS:
    out_root = "outputs/CAS_STUDY_TX_2018/baseline"
    cfg   = make_config(date, days, DEF["dc_grid"], DEF["total_mw"],
                        DEF["flex"], DEF["alpha"], DEF["deferral"], out_root)
    fname = CFG_DIR / f"tx_baseline__{date}.json"
    fname.write_text(json.dumps(cfg, indent=2))
    manifest.append({
        "run_id":      run_id,
        "phase":       "baseline",
        "group":       "baseline",
        "param_label": "default",
        "run_label":   f"baseline_{mo_abbrev}",
        "date":        date,
        "days":        days,
        "config_path": str(fname),
        "total_mw":    DEF["total_mw"],
        "flex_pct":    DEF["flex"],
        "lp_alpha":    DEF["alpha"],
        "deferral_h":  DEF["deferral"],
        "dc_grid":     DEF["dc_grid"],
    })
    run_id += 1

# ── Sensitivity ───────────────────────────────────────────────────────────
for date, days, mo_tag in SENS_MONTHS:
    for group, param_label, overrides in SENS_GROUPS:
        dc_grid  = overrides.get("dc_grid",   DEF["dc_grid"])
        total_mw = overrides.get("total_mw",  DEF["total_mw"])
        flex     = overrides.get("flex",      DEF["flex"])
        alpha    = overrides.get("alpha",     DEF["alpha"])
        deferral = overrides.get("deferral",  DEF["deferral"])

        out_root = f"outputs/CAS_STUDY_TX_2018/sensitivity/{group}/{param_label}"
        cfg   = make_config(date, days, dc_grid, total_mw,
                            flex, alpha, deferral, out_root)
        fname = CFG_DIR / f"tx_sens__{group}__{param_label}__{mo_tag}.json"
        fname.write_text(json.dumps(cfg, indent=2))
        manifest.append({
            "run_id":      run_id,
            "phase":       "sensitivity",
            "group":       group,
            "param_label": param_label,
            "run_label":   f"{group}/{param_label}/{mo_tag}",
            "date":        date,
            "days":        days,
            "config_path": str(fname),
            "total_mw":    total_mw,
            "flex_pct":    flex,
            "lp_alpha":    alpha,
            "deferral_h":  deferral,
            "dc_grid":     dc_grid,
        })
        run_id += 1

# ── Write manifest ─────────────────────────────────────────────────────────
manifest_path = REPO / "scripts" / "study" / "tx_manifest.csv"
with open(manifest_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=manifest[0].keys())
    w.writeheader()
    w.writerows(manifest)

# ── Summary ────────────────────────────────────────────────────────────────
baseline_n = sum(1 for r in manifest if r["phase"] == "baseline")
sens_n     = sum(1 for r in manifest if r["phase"] == "sensitivity")
print(f"Generated {len(manifest)} configs  ({baseline_n} baseline + {sens_n} sensitivity)")
print(f"Configs  → {CFG_DIR}")
print(f"Manifest → {manifest_path}")
print()
print("Baseline runs (array 0-3):")
for r in manifest:
    if r["phase"] == "baseline":
        print(f"  [{r['run_id']}] {r['run_label']}  ({r['date']}, {r['days']}d)")
print()
print("Sensitivity runs (array 0-23):")
for i, r in enumerate(r for r in manifest if r["phase"] == "sensitivity"):
    print(f"  [{i:2d}] {r['run_label']}")
print()
print("NOTE: Before submitting sensitivity jobs, pre-build the SCALED DC grid:")
print(f"  python scripts/main.py --config {CFG_DIR}/tx_sens__load_growth__amp_25pct__2018-01.json")
print("  (or: sbatch --array=0-0 scripts/study/run_tx_sensitivity.slurm)")
