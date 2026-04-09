#!/usr/bin/env python3
"""
generate_configs.py — Generate all config JSONs and manifest.csv for the CAS study.

Study design
------------
Baseline (4 runs):
  Jan, Apr, Jul, Oct  ×  default params (15% load growth, flex=30%, α=0.5, defer=12h)

Sensitivity (24 runs):
  Jan + Jul  ×  one parameter varied at a time, all others at default:
    Load growth : 25%
    Flex        : 20%, 40%, 50%
    Alpha       : 0.00, 0.25, 0.75, 1.00
    Deferral    : 4h, 8h, 18h, 24h

Usage:
    module load anaconda3/2024.10
    python scripts/study/generate_configs.py
"""

import csv, hashlib, json
from pathlib import Path

REPO     = Path(__file__).resolve().parents[2]
CFG_DIR  = REPO / "scripts" / "study" / "configs"
CFG_DIR.mkdir(parents=True, exist_ok=True)

# ── Grid demand → amplitude_mw ─────────────────────────────────────────────
def _avg_demand(fallback: float = 3809.0) -> float:
    """Average hourly demand from the January baseline run (MW)."""
    hs = (REPO / "outputs" / "SENSITIVITY_RTS_2020" / "annual_baseline"
          / "2020-01-01" / "baseline" / "hourly_summary.csv")
    try:
        import pandas as pd
        return float(pd.read_csv(hs)["Demand"].mean())
    except Exception:
        return fallback

AVG_DEMAND = _avg_demand()
N_BUSES    = 10
AMP_15     = round(0.15 * AVG_DEMAND / N_BUSES, 1)   # per-bus MW at 15% load growth
AMP_25     = round(0.25 * AVG_DEMAND / N_BUSES, 1)   # per-bus MW at 25% load growth

DC_BUSES = [
    "Abel","Adler","Attar","Attlee","Bach",
    "Balzac","Beethoven","Cabell","Caesar","Clark",
]

# ── Defaults ───────────────────────────────────────────────────────────────
DEF = dict(amp=AMP_15, dc_grid="RTS-GMLC-DC-15PCT",
           flex=30, alpha=0.5, deferral=12, max_iter=2)

# ── Month tables ───────────────────────────────────────────────────────────
BASELINE_MONTHS = [
    ("2020-01-01", 31, "Jan"),
    ("2020-04-01", 30, "Apr"),
    ("2020-07-01", 31, "Jul"),
    ("2020-10-01", 31, "Oct"),
]
SENS_MONTHS = [
    ("2020-01-01", 31, "2020-01"),
    ("2020-07-01", 31, "2020-07"),
]

# ── Sensitivity groups: (group, label, overrides-dict) ────────────────────
SENS_GROUPS = [
    ("load_growth", "amp_25pct",    dict(amp=AMP_25, dc_grid="RTS-GMLC-DC-25PCT")),
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

# ── Helpers ────────────────────────────────────────────────────────────────
def lp_hash(alpha, flex, deferral, max_iter) -> str:
    key = f"{alpha:.4f}_{flex}_{deferral}_{max_iter}"
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def make_config(date, days, amp, dc_grid, flex, alpha, deferral, max_iter, out_root) -> dict:
    return {
        "grid": {
            "base_grid":    "RTS-GMLC",
            "dc_grid":      dc_grid,
            "dc_buses":     DC_BUSES,
            "amplitude_mw": amp,
            "variation":    0.05,
            "period_hours": 24.0,
            "phase_hours":  0.0,
        },
        "simulation": {
            "solver":        "gurobi",
            "gurobi_module": "gurobi/10.0.1",
            "solver_args":   "Cuts=1 Presolve=1 Heuristics=0.03",
            "threads":       8,
            "ruc_horizon":   48,
            "sced_horizon":  4,
            "ruc_mipgap":    0.01,
            "reserve_factor":0.05,
            "output_detail": 2,
            "date":          date,
            "days":          days,
        },
        "cas": {
            "extra_capacity_pct":   flex,
            "flexible_ratio_pct":   flex,
            "renew_fraction":       0.1,
            "lp_alpha":             alpha,
            "lp_alpha_steps":       3,
            "deferral_window_hours":deferral,
            "lp_mode":              "joint",
            "headroom_pct":         flex,
        },
        "qp":       {"enabled": False, "perturb_scale": 1.05},
        "renew_opt":{"enabled": False, "apply_best":    False},
        "cas_iter": {"max_iter": 1,    "tol_co2_pct":   0.5},
        "sim_lp_iter":   {"mode": "single", "max_iterations": 1},
        "joint_lp_iter": {"max_iter": max_iter, "tol": 0.5},
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
    out_root = f"outputs/CAS_STUDY_RTS_2020/baseline"
    cfg   = make_config(date, days, DEF["amp"], DEF["dc_grid"],
                        DEF["flex"], DEF["alpha"], DEF["deferral"],
                        DEF["max_iter"], out_root)
    fname = CFG_DIR / f"baseline__{date}.json"
    fname.write_text(json.dumps(cfg, indent=2))
    manifest.append({
        "run_id":       run_id,
        "phase":        "baseline",
        "group":        "baseline",
        "param_label":  "default",
        "run_label":    f"baseline_{mo_abbrev}",
        "date":         date,
        "days":         days,
        "config_path":  str(fname),
        "amplitude_mw": DEF["amp"],
        "flex_pct":     DEF["flex"],
        "lp_alpha":     DEF["alpha"],
        "deferral_h":   DEF["deferral"],
        "max_iter":     DEF["max_iter"],
        "dc_grid":      DEF["dc_grid"],
    })
    run_id += 1

# ── Sensitivity ───────────────────────────────────────────────────────────
for date, days, mo_tag in SENS_MONTHS:
    for group, param_label, overrides in SENS_GROUPS:
        amp      = overrides.get("amp",     DEF["amp"])
        dc_grid  = overrides.get("dc_grid", DEF["dc_grid"])
        flex     = overrides.get("flex",    DEF["flex"])
        alpha    = overrides.get("alpha",   DEF["alpha"])
        deferral = overrides.get("deferral",DEF["deferral"])
        max_iter = overrides.get("max_iter",DEF["max_iter"])

        out_root = f"outputs/CAS_STUDY_RTS_2020/sensitivity/{group}/{param_label}"
        cfg   = make_config(date, days, amp, dc_grid,
                            flex, alpha, deferral, max_iter, out_root)
        fname = CFG_DIR / f"sens__{group}__{param_label}__{mo_tag}.json"
        fname.write_text(json.dumps(cfg, indent=2))
        manifest.append({
            "run_id":       run_id,
            "phase":        "sensitivity",
            "group":        group,
            "param_label":  param_label,
            "run_label":    f"{group}/{param_label}/{mo_tag}",
            "date":         date,
            "days":         days,
            "config_path":  str(fname),
            "amplitude_mw": amp,
            "flex_pct":     flex,
            "lp_alpha":     alpha,
            "deferral_h":   deferral,
            "max_iter":     max_iter,
            "dc_grid":      dc_grid,
        })
        run_id += 1

# ── Write manifest ─────────────────────────────────────────────────────────
manifest_path = REPO / "scripts" / "study" / "manifest.csv"
with open(manifest_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=manifest[0].keys())
    w.writeheader()
    w.writerows(manifest)

# ── Summary ────────────────────────────────────────────────────────────────
baseline_n = sum(1 for r in manifest if r["phase"] == "baseline")
sens_n     = sum(1 for r in manifest if r["phase"] == "sensitivity")
print(f"Generated {len(manifest)} configs  ({baseline_n} baseline + {sens_n} sensitivity)")
print(f"Configs → {CFG_DIR}")
print(f"Manifest → {manifest_path}")
print()
print(f"Average grid demand (Jan baseline): {AVG_DEMAND:.0f} MW")
print(f"  15% amplitude: {AMP_15} MW/bus  ({AMP_15*N_BUSES:.0f} MW total)")
print(f"  25% amplitude: {AMP_25} MW/bus  ({AMP_25*N_BUSES:.0f} MW total)")
print()
print("Baseline runs (array 0-3):")
for r in manifest:
    if r["phase"] == "baseline":
        print(f"  [{r['run_id']}] {r['run_label']}  ({r['date']}, {r['days']}d)")
print()
print("Sensitivity runs (array 0-23):")
for i, r in enumerate([r for r in manifest if r["phase"] == "sensitivity"]):
    print(f"  [{i:2d}] {r['run_label']}")
