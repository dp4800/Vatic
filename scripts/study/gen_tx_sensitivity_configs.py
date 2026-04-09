"""Generate sensitivity-sweep configs for TX annual study.

8 representative weeks (2 per season) × 3 sweeps × 2 grids = 176 configs.

Sweeps (matching RTS study):
  alpha    : lp_alpha in {0.0, 0.25, 0.75, 1.0}  (baseline 0.5)
  deferral : deferral_window_hours in {4, 8, 18, 24}  (baseline 12)
  flex     : flexible_ratio_pct in {20, 40, 50}  (baseline 30)

Note: load_growth sweep omitted — TX grids use dc_buses_mode='from_grid'
so amplitude is embedded in BusInjections CSVs, not a single scalar.

Run: python scripts/study/gen_tx_sensitivity_configs.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "scripts/study/configs"
CFG_DIR.mkdir(parents=True, exist_ok=True)

# 2 representative weeks per season
SENS_WEEKS = [
    ("week00", "2018-01-07"),  # Winter
    ("week01", "2018-01-21"),  # Winter
    ("week06", "2018-04-01"),  # Spring
    ("week07", "2018-04-15"),  # Spring
    ("week12", "2018-07-01"),  # Summer
    ("week13", "2018-07-15"),  # Summer
    ("week18", "2018-10-07"),  # Fall
    ("week19", "2018-10-21"),  # Fall
]

SWEEPS = [
    {
        "name": "alpha",
        "section": "cas",
        "key": "lp_alpha",
        "values": [
            ("alpha_0.00", 0.0),
            ("alpha_0.25", 0.25),
            ("alpha_0.75", 0.75),
            ("alpha_1.00", 1.0),
        ],
    },
    {
        "name": "deferral",
        "section": "cas",
        "key": "deferral_window_hours",
        "values": [
            ("deferral_4h",  4),
            ("deferral_8h",  8),
            ("deferral_18h", 18),
            ("deferral_24h", 24),
        ],
    },
    {
        "name": "flex",
        "section": "cas",
        "key": "flexible_ratio_pct",
        "values": [
            ("flex_20pct", 20),
            ("flex_40pct", 40),
            ("flex_50pct", 50),
        ],
    },
]

GRIDS = [
    {
        "tag":       "tx_2018",
        "base_grid": "Texas-7k",
        "dc_grid":   "Texas-7k-DC-REAL",
        "out_root":  "outputs/TX_2018_ANNUAL",  # shared with annual — reuses baseline/GM/247
    },
    {
        "tag":       "tx_2030",
        "base_grid": "Texas-7k_2030",
        "dc_grid":   "Texas-7k_2030-DC",
        "out_root":  "outputs/TX_2030_ANNUAL",  # shared with annual — reuses baseline/GM/247
    },
]

TEMPLATE = {
    "grid": {
        "base_grid":     None,
        "dc_grid":       None,
        "dc_buses_mode": "from_grid",
        "dc_buses":      [],
        "amplitude_mw":  0.0,
        "variation":     0.08,
        "period_hours":  24.0,
        "phase_hours":   4.0,
    },
    "simulation": {
        "solver":         "gurobi",
        "gurobi_module":  "gurobi/10.0.1",
        "solver_args":    "Cuts=1 Presolve=1 Heuristics=0.03",
        "threads":        8,
        "ruc_horizon":    48,
        "sced_horizon":   4,
        "ruc_mipgap":     0.01,
        "reserve_factor": 0.05,
        "output_detail":  2,
        "date":           None,
        "days":           7,
    },
    "cas": {
        "extra_capacity_pct":    30,
        "flexible_ratio_pct":    30,
        "renew_fraction":        0.1,
        "lp_alpha":              0.5,
        "lp_alpha_steps":        3,
        "deferral_window_hours": 12,
        "lp_mode":               "joint",
        "headroom_pct":          30,
        "run_247":               False,
    },
    "qp":           {"enabled": False, "perturb_scale": 1.05},
    "renew_opt":    {"enabled": False, "apply_best": False},
    "cas_iter":     {"max_iter": 1, "tol_co2_pct": 0.5},
    "sim_lp_iter":  {"mode": "single", "max_iterations": 1},
    "joint_lp_iter":{"max_iter": 2, "tol": 0.5},
    "analytics":    {"ncs_enabled": False, "carbon_weight": 0.5, "water_weight": 0.5},
    "out_root":     None,
}

generated = []
for grid in GRIDS:
    for sweep in SWEEPS:
        for val_tag, val in sweep["values"]:
            for week_id, date in SENS_WEEKS:
                cfg = {k: dict(v) if isinstance(v, dict) else v
                       for k, v in TEMPLATE.items()}
                cfg["grid"]       = dict(TEMPLATE["grid"])
                cfg["simulation"] = dict(TEMPLATE["simulation"])
                cfg["cas"]        = dict(TEMPLATE["cas"])

                cfg["grid"]["base_grid"] = grid["base_grid"]
                cfg["grid"]["dc_grid"]   = grid["dc_grid"]
                cfg["simulation"]["date"] = date
                cfg["cas"][sweep["key"]] = val
                # Share out_root with annual run so baseline/GM/247 are reused.
                # cas_lp_tag namespaces the LP output dirs within that shared root.
                cfg["out_root"]    = grid["out_root"]
                cfg["cas_lp_tag"]  = f"{sweep['name']}-{val_tag}"

                fname = CFG_DIR / (
                    f"{grid['tag']}_sens__{sweep['name']}__{val_tag}"
                    f"__{week_id}_{date}.json"
                )
                with open(fname, "w") as f:
                    json.dump(cfg, f, indent=2)
                generated.append(fname.name)

print(f"Generated {len(generated)} sensitivity configs:")
by_grid = {}
for name in generated:
    g = name.split("_sens__")[0]
    by_grid.setdefault(g, 0)
    by_grid[g] += 1
for g, n in by_grid.items():
    print(f"  {g}: {n} configs")
print(f"\nSample filenames:")
for name in generated[:4]:
    print(f"  {name}")
print(f"  ...")
for name in generated[-2:]:
    print(f"  {name}")
