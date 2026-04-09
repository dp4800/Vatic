"""Generate 24-week annual configs for TX-2018 and TX-2030 CAS studies.

Two per month (7th and 21st), 7-day runs, both grids.
Run: python scripts/study/gen_tx_annual_configs.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "scripts/study/configs"
CFG_DIR.mkdir(parents=True, exist_ok=True)

# 2 start dates per month (7th and 21st) for 2018
DATES = [
    "2018-01-07", "2018-01-21",
    "2018-02-04", "2018-02-18",
    "2018-03-04", "2018-03-18",
    "2018-04-01", "2018-04-15",
    "2018-05-06", "2018-05-20",
    "2018-06-03", "2018-06-17",
    "2018-07-01", "2018-07-15",
    "2018-08-05", "2018-08-19",
    "2018-09-02", "2018-09-16",
    "2018-10-07", "2018-10-21",
    "2018-11-04", "2018-11-18",
    "2018-12-02", "2018-12-16",
]
assert len(DATES) == 24

TX_2018_BUSES = [
    "RIO_MEDINA_2_1", "PECOS_2_1", "WILMER_4_1", "DYESS_AFB_6_1",
    "MONTGOMERY_3_1", "TOMBALL_10_1", "SILVERTON_1_2", "RICHARDSON_15_3",
    "ROUND_ROCK_11_1", "PARIS_7_4", "PAMPA_1_1",
]

TX_2030_BUSES = [
    "DALLAS_29_1", "CARROLLTON_10_1", "DALLAS_22_1", "RICHARDSON_8_1",
    "CARROLLTON_5_1", "PLANO_3_1", "DALLAS_43_1", "PLANO_5_2",
    "ADDISON_14_1", "ADDISON_20_1", "GARLAND_10_1", "RICHARDSON_6_2",
    "THE_COLONY_3_1", "WILMER_5_1", "CARROLLTON_13_1", "IRVING_3_1",
    "ALLEN_10_1", "RICHARDSON_18_2", "DALLAS_11_1", "DALLAS_8_1",
    "DALLAS_3_1", "COPPELL_6_1", "KELLER_4_2", "RICHARDSON_2_1",
    "RICHARDSON_14_1", "ADDISON_17_1", "ADDISON_7_1", "DALLAS_30_1",
    "FRISCO_9_2", "MCKINNEY_6_1", "DALLAS_41_2", "IRVING_1_1",
    "IRVING_5_1", "KENNEDALE_7_1", "RICHARDSON_5_1", "RICHARDSON_19_1",
    "FORT_WORTH_4_1", "WILMER_4_1", "WILMER_6_1", "DALLAS_24_2",
    "PLANO_9_1", "WHITNEY_5_2", "FORT_WORTH_16_1", "DENTON_2_6",
    "RED_OAK_2_1", "MANSFIELD_3_1", "GRAND_PRAIRIE_14_1", "VENUS_3_5",
    "HASLET_2_1", "RED_OAK_5_1", "PARIS_7_4", "IRVING_4_1",
    "CARROLLTON_8_1", "WACO_6_1", "TEMPLE_5_1", "WACO_10_1",
    "LACKLAND_A_F_B_12_2", "LACKLAND_A_F_B_3_1", "ATASCOSA_4_1",
    "SAN_ANTONIO_15_1", "SAN_ANTONIO_26_1", "RIO_MEDINA_2_1",
    "SAN_ANTONIO_8_1", "UNIVERSAL_CITY_9_1", "HELOTES_9_1",
    "ELMENDORF_13_2", "VICTORIA_7_1", "CASTROVILLE_2_1",
    "SAN_ANTONIO_37_1", "LACKLAND_A_F_B_11_1", "SAN_MARCOS_1_1",
    "SAN_ANTONIO_39_1", "CASTROVILLE_1_1", "CONVERSE_8_2",
    "AUSTIN_27_1", "AUSTIN_60_1", "PFLUGERVILLE_7_1", "AUSTIN_52_1",
    "DEL_VALLE_4_1", "ROUND_ROCK_7_1", "CEDAR_PARK_17_1", "AUSTIN_5_1",
    "ROUND_ROCK_11_1", "AUSTIN_28_1", "HUTTO_1_1", "TEMPLE_3_1",
    "MAXWELL_1_1", "ROUND_ROCK_13_1", "SAN_MARCOS_7_3", "AUSTIN_44_1",
    "ROCKDALE_6_1", "AUSTIN_2_1", "AUSTIN_32_1", "AUSTIN_13_2",
    "BRYAN_10_1", "SPRING_5_1", "TOMBALL_6_2", "SPRING_9_1",
    "KATY_73_1", "KATY_74_2", "HOUSTON_33_1", "SPRING_16_1",
    "HUMBLE_6_2", "BELLAIRE_10_1", "BELLAIRE_28_1", "BELLAIRE_9_1",
    "CYPRESS_31_1", "BELLAIRE_8_1", "HOUSTON_4_1", "HOCKLEY_3_1",
    "WILLIS_5_2", "TOMBALL_10_1", "HOUSTON_25_1", "HOUSTON_31_1",
    "BELLAIRE_41_1", "CYPRESS_39_1", "STAFFORD_18_1", "SPRING_21_1",
    "MONTGOMERY_3_1", "SILVERTON_1_2", "PAMPA_1_1", "AMARILLO_11_1",
    "AMARILLO_19_1", "SLATON_1_1", "SHEPPARD_AFB_2_1", "CHILDRESS_2_1",
    "WICHITA_FALLS_7_1", "MCALLEN_10_1", "MCALLEN_5_1", "HARLINGEN_5_1",
    "HARLINGEN_3_1", "LAREDO_25_1", "LAREDO_12_1", "CORPUS_CHRISTI_21_1",
    "EAGLE_PASS_2_1", "CUSHING_3_1", "TYLER_7_1", "FORT_STOCKTON_1_1",
    "PECOS_2_1", "GOODFELLOW_AFB_6_1", "BOYS_RANCH_2_1", "VAN_HORN_1_1",
    "DYESS_AFB_6_1", "DYESS_AFB_4_1",
]

GRIDS = [
    {
        "tag":        "tx_2018",
        "base_grid":  "Texas-7k",
        "dc_grid":    "Texas-7k-DC-REAL",
        "dc_buses":   TX_2018_BUSES,
        "amplitude":  200.0,
        "out_root":   "outputs/TX_2018_ANNUAL",
    },
    {
        "tag":        "tx_2030",
        "base_grid":  "Texas-7k_2030",
        "dc_grid":    "Texas-7k_2030-DC",
        "dc_buses":   TX_2030_BUSES,
        "amplitude":  200.0,
        "out_root":   "outputs/TX_2030_ANNUAL",
    },
]

TEMPLATE = {
    "grid": {
        "base_grid":     None,
        "dc_grid":       None,
        # dc_buses resolved automatically from the grid's existing BusInjections CSV
        # (real per-bus DC load values, not a synthetic sinusoidal amplitude)
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
    "qp": {"enabled": False, "perturb_scale": 1.05},
    "renew_opt": {"enabled": False, "apply_best": False},
    "cas_iter": {"max_iter": 1, "tol_co2_pct": 0.5},
    "sim_lp_iter": {"mode": "single", "max_iterations": 1},
    "joint_lp_iter": {"max_iter": 2, "tol": 0.5},
    "analytics": {"ncs_enabled": False, "carbon_weight": 0.5, "water_weight": 0.5},
    "out_root": None,
}

generated = []
for grid in GRIDS:
    for i, date in enumerate(DATES):
        cfg = {k: dict(v) if isinstance(v, dict) else v
               for k, v in TEMPLATE.items()}
        cfg["grid"] = dict(TEMPLATE["grid"])
        cfg["grid"]["base_grid"] = grid["base_grid"]
        cfg["grid"]["dc_grid"]   = grid["dc_grid"]
        # dc_buses resolved at runtime from BusInjections CSV — leave empty
        cfg["grid"]["dc_buses"]      = []
        cfg["grid"]["amplitude_mw"]  = 0.0
        cfg["simulation"] = dict(TEMPLATE["simulation"])
        cfg["simulation"]["date"] = date
        cfg["out_root"] = grid["out_root"]

        fname = CFG_DIR / f"{grid['tag']}_week{i:02d}_{date}.json"
        with open(fname, "w") as f:
            json.dump(cfg, f, indent=2)
        generated.append(fname.name)

print(f"Generated {len(generated)} configs:")
for name in generated:
    print(f"  {name}")
