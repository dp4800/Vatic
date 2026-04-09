#!/usr/bin/env python3
"""
fix_uc_regional_load.py

Adds DC bus injections (BusInjections CSVs) into the regional Load CSVs so
that the unit commitment (RUC) stage can see the full system demand including
DC load.  Without this fix the RUC solves against non-DC load only, leaving
committed capacity ~22 GW short of actual demand in many hours.

Modifies in-place (with automatic .bak backups) for:
  - Texas-7k-DC-REAL
  - Texas-7k-DC-REAL-RENEW-{W10000,W20000,S10000,S20000,
                              W10000S6000B4000,W5000S3000B2000}

Usage:
  python scripts/study/fix_uc_regional_load.py          # preview only
  python scripts/study/fix_uc_regional_load.py --apply  # write changes
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GRIDS_DIR = Path(__file__).resolve().parents[2] / "vatic" / "data" / "grids"

TARGET_GRIDS = [
    "Texas-7k-DC-REAL",
    "Texas-7k-DC-REAL-RENEW-W10000",
    "Texas-7k-DC-REAL-RENEW-W20000",
    "Texas-7k-DC-REAL-RENEW-S10000",
    "Texas-7k-DC-REAL-RENEW-S20000",
    "Texas-7k-DC-REAL-RENEW-W10000S6000B4000",
    "Texas-7k-DC-REAL-RENEW-W5000S3000B2000",
]

DATE_COLS = ["Year", "Month", "Day", "Period"]
REGIONS   = ["Coast", "East", "Far_West", "North",
             "North_Central", "South", "South_Central", "West"]


def load_bus_area_map(grid_dir: Path) -> dict[str, str]:
    """Return {bus_name: area} from SourceData/bus.csv."""
    bus_csv = grid_dir / "TX_Data" / "SourceData" / "bus.csv"
    df = pd.read_csv(bus_csv, usecols=["Bus Name", "Area"])
    return dict(zip(df["Bus Name"], df["Area"]))


def compute_dc_by_region(
    inj_path: Path,
    bus_area: dict[str, str],
) -> pd.DataFrame:
    """
    Read a BusInjections CSV, group DC load by region, and return a DataFrame
    with columns [Year, Month, Day, Period, <region>, ...].
    """
    inj = pd.read_csv(inj_path)
    bus_cols = [c for c in inj.columns if c not in DATE_COLS]

    # Aggregate MW by region for each timestep
    region_df = inj[DATE_COLS].copy()
    for region in REGIONS:
        buses_in_region = [b for b in bus_cols if bus_area.get(b) == region]
        if buses_in_region:
            region_df[region] = inj[buses_in_region].sum(axis=1)
        else:
            region_df[region] = 0.0

    return region_df


def patch_load_csv(
    load_path: Path,
    dc_by_region: pd.DataFrame,
    label: str,
    apply: bool,
) -> dict:
    """
    Add dc_by_region to the Load CSV.  Returns summary stats dict.
    """
    load = pd.read_csv(load_path)
    regions_present = [c for c in load.columns if c in REGIONS]

    merged = load.merge(dc_by_region, on=DATE_COLS, suffixes=("", "_dc"))

    before_total = load[regions_present].sum(axis=1)
    for region in regions_present:
        dc_col = f"{region}_dc" if f"{region}_dc" in merged.columns else region
        if dc_col in merged.columns:
            merged[region] = merged[region] + merged[dc_col]

    after_total = merged[regions_present].sum(axis=1)

    # Monthly summary
    merged["_month"] = merged["Month"]
    monthly = merged.groupby("_month").apply(
        lambda g: pd.Series({
            "before_MW_mean": before_total[g.index].mean(),
            "after_MW_mean":  after_total[g.index].mean(),
            "dc_MW_mean":     (after_total - before_total)[g.index].mean(),
        })
    )

    stats = {
        "path": load_path,
        "rows": len(load),
        "regions": regions_present,
        "before_annual_mean_MW": before_total.mean(),
        "after_annual_mean_MW":  after_total.mean(),
        "dc_annual_mean_MW":     (after_total - before_total).mean(),
        "monthly": monthly,
    }

    if apply:
        bak_path = load_path.with_suffix(".csv.bak")
        if not bak_path.exists():
            load_path.rename(bak_path)
        else:
            # overwrite only the modified file, keep existing bak
            pass

        # Write only the original columns (drop _dc suffix cols and _month)
        out_cols = [c for c in merged.columns
                    if not c.endswith("_dc") and c != "_month"]
        merged[out_cols].to_csv(load_path, index=False)
        print(f"  [WRITTEN] {load_path.relative_to(GRIDS_DIR.parent.parent.parent)}")
    else:
        print(f"  [PREVIEW] {load_path.relative_to(GRIDS_DIR.parent.parent.parent)}")

    return stats


def print_summary(grid_name: str, da_stats: dict, rt_stats: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  Grid: {grid_name}")
    print(f"{'='*70}")
    for label, s in [("DAY_AHEAD", da_stats), ("REAL_TIME", rt_stats)]:
        print(f"  {label}:")
        print(f"    Rows       : {s['rows']}")
        print(f"    Regions    : {s['regions']}")
        print(f"    Annual mean before : {s['before_annual_mean_MW']:>10,.1f} MW")
        print(f"    Annual mean after  : {s['after_annual_mean_MW']:>10,.1f} MW")
        print(f"    DC added (mean)    : {s['dc_annual_mean_MW']:>10,.1f} MW")
        print(f"    Monthly DC breakdown (mean MW added per region-hour):")
        monthly = s["monthly"]
        print(f"      {'Month':>5}  {'Before':>10}  {'After':>10}  {'DC Added':>10}")
        for month, row in monthly.iterrows():
            print(f"      {month:>5}  {row['before_MW_mean']:>10,.1f}  "
                  f"{row['after_MW_mean']:>10,.1f}  {row['dc_MW_mean']:>10,.1f}")


def process_grid(grid_name: str, apply: bool) -> None:
    grid_dir = GRIDS_DIR / grid_name
    if not grid_dir.exists():
        print(f"WARNING: {grid_dir} not found — skipping")
        return

    bus_area = load_bus_area_map(grid_dir)
    ts_dir   = grid_dir / "TX_Data" / "timeseries_data_files"
    inj_dir  = ts_dir / "BusInjections"
    load_dir = ts_dir / "Load"

    # DAY_AHEAD
    da_dc    = compute_dc_by_region(inj_dir / "DAY_AHEAD_bus_injections.csv", bus_area)
    da_stats = patch_load_csv(load_dir / "DAY_AHEAD_regional_Load.csv",
                               da_dc, "DAY_AHEAD", apply)

    # REAL_TIME
    rt_dc    = compute_dc_by_region(inj_dir / "REAL_TIME_bus_injections.csv", bus_area)
    rt_stats = patch_load_csv(load_dir / "REAL_TIME_regional_Load.csv",
                               rt_dc, "REAL_TIME", apply)

    print_summary(grid_name, da_stats, rt_stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="Write changes (default: preview only)")
    args = parser.parse_args()

    mode = "APPLY" if args.apply else "PREVIEW (dry-run — pass --apply to write)"
    print(f"\nMode: {mode}")
    print(f"Grids to process: {len(TARGET_GRIDS)}\n")

    for grid in TARGET_GRIDS:
        process_grid(grid, args.apply)

    print(f"\n{'='*70}")
    if args.apply:
        print("  Done. Original files backed up as *.csv.bak")
        print("  Configs and SLURM scripts unchanged — same grid names.")
    else:
        print("  Dry-run complete. Run with --apply to write changes.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
