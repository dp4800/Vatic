"""
scripts/study/generate_renew_configs.py
-----------------------------------------
Generate configs and grids for the renewable penetration sensitivity study.

Portfolio design (1D sweeps + combined points):
  Wind sweep  : +500, +1000, +2000, +3000 MW
  Solar sweep : +500, +1000, +2000, +3000 MW
  Battery     : +250, +500, +1000 MWh
  Combined    : +1500W+1500S,  +3000W+3000S

Months: Jan (2020-01-01, 31d) + Jul (2020-07-01, 31d)

Each config runs main.py which executes:
  baseline + GM-CAS + LP-CAS  (247-CAS skipped — circular when varying renewables)

Outputs:
  scripts/study/renew_configs/   — JSON config files
  scripts/study/renew_manifest.csv
"""

from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

REPO        = Path(__file__).resolve().parents[2]
SCRIPTS     = REPO / "scripts"
CONFIG_DIR  = SCRIPTS / "study" / "renew_configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST    = SCRIPTS / "study" / "renew_manifest.csv"

BASE_CONFIG = SCRIPTS / "study" / "configs" / "baseline__2020-01-01.json"

DATES = [
    {"date": "2020-01-01", "days": 31, "label": "Jan"},
    {"date": "2020-07-01", "days": 31, "label": "Jul"},
]

PORTFOLIOS = [
    # Wind sweep (solar=0, battery=0)
    {"group": "wind",    "label": "W500",        "wind": 500,  "solar": 0,    "batt": 0},
    {"group": "wind",    "label": "W1000",        "wind": 1000, "solar": 0,    "batt": 0},
    {"group": "wind",    "label": "W2000",        "wind": 2000, "solar": 0,    "batt": 0},
    {"group": "wind",    "label": "W3000",        "wind": 3000, "solar": 0,    "batt": 0},
    # Solar sweep (wind=0, battery=0)
    {"group": "solar",   "label": "S500",         "wind": 0,    "solar": 500,  "batt": 0},
    {"group": "solar",   "label": "S1000",        "wind": 0,    "solar": 1000, "batt": 0},
    {"group": "solar",   "label": "S2000",        "wind": 0,    "solar": 2000, "batt": 0},
    {"group": "solar",   "label": "S3000",        "wind": 0,    "solar": 3000, "batt": 0},
    # Battery sweep (wind=0, solar=0)
    {"group": "battery", "label": "B250",         "wind": 0,    "solar": 0,    "batt": 250},
    {"group": "battery", "label": "B500",         "wind": 0,    "solar": 0,    "batt": 500},
    {"group": "battery", "label": "B1000",        "wind": 0,    "solar": 0,    "batt": 1000},
    # Combined
    {"group": "combined","label": "W1500S1500",   "wind": 1500, "solar": 1500, "batt": 0},
    {"group": "combined","label": "W3000S3000",   "wind": 3000, "solar": 3000, "batt": 0},
    # Intermediate wind/solar — added to characterise non-monotone CO2 curve
    {"group": "wind",    "label": "W1500",        "wind": 1500, "solar": 0,    "batt": 0},
    {"group": "wind",    "label": "W2500",        "wind": 2500, "solar": 0,    "batt": 0},
    {"group": "solar",   "label": "S1500",        "wind": 0,    "solar": 1500, "batt": 0},
    {"group": "solar",   "label": "S2500",        "wind": 0,    "solar": 2500, "batt": 0},
]


def grid_name(p: dict) -> str:
    w = f"W{int(p['wind'])}" if p["wind"] > 0 else ""
    s = f"S{int(p['solar'])}" if p["solar"] > 0 else ""
    b = f"B{int(p['batt'])}"  if p["batt"]  > 0 else ""
    return f"RTS-GMLC-DC-15PCT-RENEW-{w}{s}{b}"


def build_grid(p: dict, force: bool = False) -> str:
    gname = grid_name(p)
    gdir  = REPO / "vatic" / "data" / "grids" / gname
    if gdir.exists() and not force:
        print(f"  [grid exists] {gname}")
        return gname

    cmd = [
        sys.executable,
        str(SCRIPTS / "add_renewables.py"),
        "--source-grid", "RTS-GMLC-DC-15PCT",
        "--output-grid", gname,
        "--wind-mw",     str(p["wind"]),
        "--solar-mw",    str(p["solar"]),
        "--battery-mwh", str(p["batt"]),
        "--force",
    ]
    print(f"  Building grid: {gname} ...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] {result.stderr[-500:]}")
        return None
    print("done")
    return gname


def make_config(p: dict, date_info: dict, grid: str, run_id: int) -> dict:
    base = json.loads(BASE_CONFIG.read_text())

    # Override grid
    base["grid"]["base_grid"]  = "RTS-GMLC-DC-15PCT"
    base["grid"]["dc_grid"]    = grid

    # Override date/days
    base["simulation"]["date"] = date_info["date"]
    base["simulation"]["days"] = date_info["days"]

    # Override output root
    label = f"{p['label']}/{date_info['date']}"
    base["out_root"] = f"outputs/RENEW_STUDY_RTS_2020/{p['group']}/{label}"

    # Disable 247-CAS (circular when varying renewables)
    if "cas" not in base:
        base["cas"] = {}
    base["cas"]["run_247"] = False

    return base


def main() -> None:
    base_cfg = json.loads(BASE_CONFIG.read_text())

    rows = []
    run_id = 0

    print("Building renewable grids...")
    for p in PORTFOLIOS:
        grid = build_grid(p)
        if grid is None:
            print(f"  [FAILED] {p['label']}")
            continue

        for d in DATES:
            cfg = make_config(p, d, grid, run_id)
            fname = f"renew__{p['group']}__{p['label']}__{d['date'][:7].replace('-','')}.json"
            cfg_path = CONFIG_DIR / fname
            cfg_path.write_text(json.dumps(cfg, indent=2))

            rows.append({
                "run_id":     run_id,
                "group":      p["group"],
                "label":      p["label"],
                "run_label":  f"{p['group']}/{p['label']}/{d['date']}",
                "date":       d["date"],
                "days":       d["days"],
                "wind_mw":    p["wind"],
                "solar_mw":   p["solar"],
                "battery_mwh": p["batt"],
                "dc_grid":    grid,
                "config_path": str(cfg_path),
            })
            run_id += 1

    # Write manifest
    import pandas as pd
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST, index=False)

    print(f"\nGenerated {len(rows)} configs → {CONFIG_DIR}")
    print(f"Manifest  → {MANIFEST}")
    print(f"\nNext:\n  sbatch scripts/study/run_renew.slurm")


if __name__ == "__main__":
    main()
