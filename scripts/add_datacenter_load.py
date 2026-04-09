#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
add_datacenter_load.py

Copy a VATIC grid and inject a sinusoidal load profile at one or more buses
to model data center workload addition. The resulting grid can be run directly
with `vatic-det --grid <output-grid>` and compared against the baseline to
study effects on LMPs, emissions, dispatch, and other outputs.

Load model
----------
Each specified bus receives an additional load injection:

    dc_load(t) = amplitude_mw * (1 + variation * sin(2π(t - phase_hours) / period_hours))

where t is elapsed hours from the first row of the dataset.

  amplitude_mw  : mean DC load in MW              (default: 100)
  variation     : fractional amplitude of swing    (default: 0.05, i.e. ±5%)
  period_hours  : length of one sinusoidal cycle   (default: 24, daily)
  phase_hours   : shift peak away from t=0         (default: 0)

Min load = amplitude*(1 - variation), max = amplitude*(1 + variation).

Implementation
--------------
Existing zone assignments and bus.csv are left completely unchanged. The
injection is stored in a new `BusInjections/` directory inside the grid's
timeseries_data_files folder:

    timeseries_data_files/
        Load/               (unchanged)
        BusInjections/
            DAY_AHEAD_bus_injections.csv
            REAL_TIME_bus_injections.csv

Each file mirrors the time-column format of the grid's existing load CSVs,
with one column per injected bus containing the sinusoidal MW values.

A small patch is applied to load_by_bus() in vatic/data/loaders.py so that
these injections are added to the appropriate bus entries after the normal
zone-proportional allocation, keeping all other buses untouched.

Usage
-----
    python scripts/add_datacenter_load.py \\
        --grid          RTS-GMLC          \\
        --output-grid   RTS-GMLC-DC       \\
        --buses         Abel Adams        \\
        --amplitude-mw  100               \\
        --variation     0.05              \\
        --period-hours  24                \\
        --phase-hours   0

Supported grids: RTS-GMLC, Texas-7k, Texas-7k_2030

After running:
    vatic-det --grid RTS-GMLC-DC --start-date 2020-01-01 --num-days 7 ...
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).resolve().parent
VATIC_ROOT  = SCRIPT_DIR.parent
GRIDS_DIR   = VATIC_ROOT / "vatic" / "data" / "grids"
INIT_DIR    = GRIDS_DIR / "initial-state"
LOADERS_PY  = VATIC_ROOT / "vatic" / "data" / "loaders.py"

# ---------------------------------------------------------------------------
# Grid registry
# ---------------------------------------------------------------------------

GRID_REGISTRY: dict[str, dict] = {
    "RTS-GMLC": {
        "data_dir": "RTS_Data",
        # DAY_AHEAD:  Year/Month/Day/Period(1-24), hourly
        # REAL_TIME:  Year/Month/Day/Period(1-288), 5-minute
        "da_fmt": "period",
        "rt_fmt": "period",
    },
    "Texas-7k_2030": {
        "data_dir": "TX2030_Data",
        # DAY_AHEAD:  Issue_time / Forecast_time columns
        # REAL_TIME:  Time column
        "da_fmt": "timestamps_da",
        "rt_fmt": "timestamps_rt",
    },
    "Texas-7k": {
        "data_dir": "TX_Data",
        # DAY_AHEAD:  Year/Month/Day/Period(0-23), hourly
        # REAL_TIME:  Year/Month/Day/Period(0-23), hourly
        "da_fmt": "period",
        "rt_fmt": "period",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_resolution_minutes(df: pd.DataFrame, fmt: str) -> float:
    """Infer the time resolution of a load CSV from its contents."""
    if fmt == "period":
        max_period = int(df["Period"].max())
        return 1440.0 / max_period          # 1440 / 24 = 60 min, / 288 = 5 min
    elif fmt == "timestamps_da":
        times = pd.to_datetime(df["Forecast_time"], utc=True)
        return float((times.iloc[1] - times.iloc[0]).total_seconds() / 60)
    elif fmt == "timestamps_rt":
        times = pd.to_datetime(df["Time"], utc=True)
        return float((times.iloc[1] - times.iloc[0]).total_seconds() / 60)
    else:
        raise ValueError(f"Unknown time format: {fmt!r}")


def sinusoidal_load(n: int, res_min: float, amplitude: float,
                    variation: float, period_h: float, phase_h: float) -> np.ndarray:
    """Sinusoidal injection array of length n.

        dc(t) = amplitude * (1 + variation * sin(2π(t - phase_h) / period_h))

    where t = row_index * res_min / 60 hours.
    """
    t = np.arange(n) * (res_min / 60.0)
    return amplitude * (1.0 + variation * np.sin(2.0 * np.pi * (t - phase_h) / period_h))


def build_injection_df(source_csv: Path, fmt: str,
                       bus_amplitudes: dict[str, float],
                       variation: float,
                       period_h: float, phase_h: float) -> pd.DataFrame:
    """Create an injection DataFrame that mirrors source_csv's time columns.

    Time columns are copied verbatim from the source load CSV; one column is
    added per bus containing the sinusoidal injection values.

    bus_amplitudes : dict mapping bus name -> mean DC load in MW (per bus).
    """
    src = pd.read_csv(source_csv, dtype=str)

    if fmt == "period":
        time_cols = ["Year", "Month", "Day", "Period"]
    elif fmt == "timestamps_da":
        time_cols = ["Issue_time", "Forecast_time"]
    elif fmt == "timestamps_rt":
        time_cols = ["Time"]
    else:
        raise ValueError(f"Unknown format: {fmt!r}")

    for col in time_cols:
        if col not in src.columns:
            raise KeyError(f"Expected time column '{col}' in {source_csv}")

    # Detect resolution from numeric version of the source
    src_numeric = pd.read_csv(source_csv, nrows=500)
    res_min = detect_resolution_minutes(src_numeric, fmt)
    n = len(src)

    inj = src[time_cols].copy()
    for bus, amplitude in bus_amplitudes.items():
        inj[bus] = sinusoidal_load(n, res_min, amplitude, variation, period_h, phase_h)

    return inj


def find_load_csvs(load_dir: Path) -> tuple[Path, Path]:
    da = list(load_dir.glob("DAY_AHEAD_*.csv"))
    rt = list(load_dir.glob("REAL_TIME_*.csv"))
    if len(da) != 1:
        raise FileNotFoundError(f"Expected 1 DAY_AHEAD_*.csv in {load_dir}, found {len(da)}")
    if len(rt) != 1:
        raise FileNotFoundError(f"Expected 1 REAL_TIME_*.csv in {load_dir}, found {len(rt)}")
    return da[0], rt[0]


# ---------------------------------------------------------------------------
# loaders.py patching
# ---------------------------------------------------------------------------

# The injection support block inserted into load_by_bus().
# 8-space indent throughout — matches the method body level in loaders.py.
_INJECTION_BLOCK = (
    "        # --- Bus-level load injections (added by add_datacenter_load.py) ---\n"
    "        inject_dir = Path(self.data_path, 'timeseries_data_files', 'BusInjections')\n"
    "        if inject_dir.is_dir():\n"
    "            try:\n"
    "                inj_fcsts = self.get_forecasts('BusInjections', start_date, end_date)\n"
    "                inj_actls = self.get_actuals(\n"
    "                    'BusInjections', start_date, end_date).resample('h').mean()\n"
    "            except (AssertionError, FileNotFoundError, KeyError, StopIteration):\n"
    "                inj_fcsts, inj_actls = None, None\n"
    "            if inj_fcsts is not None:\n"
    "                for _bus in inj_fcsts.columns:\n"
    "                    if ('fcst', _bus) in result.columns:\n"
    "                        result[('fcst', _bus)] = result[('fcst', _bus)] + inj_fcsts[_bus]\n"
    "            if inj_actls is not None:\n"
    "                for _bus in inj_actls.columns:\n"
    "                    if ('actl', _bus) in result.columns:\n"
    "                        result[('actl', _bus)] = result[('actl', _bus)] + inj_actls[_bus]\n"
    "        # --- end bus-level injections ---\n"
)

_RETURN_OLD = "        return pd.concat(site_dfs.values(), axis=1).sort_index(axis=1)"
_RETURN_NEW = (
    "        result = pd.concat(site_dfs.values(), axis=1).sort_index(axis=1)\n"
    + _INJECTION_BLOCK
    + "        return result"
)


def patch_loaders_py(loaders_path: Path) -> None:
    """Idempotently patch load_by_bus() to apply BusInjections after zone allocation.

    load_input() now uses prefix matching, so no per-grid registration is needed.
    This function only needs to run once regardless of how many derived grids exist.
    """
    src = loaders_path.read_text()

    if "# --- Bus-level load injections" not in src:
        if _RETURN_OLD not in src:
            print("    WARNING: Could not locate load_by_bus() return line — "
                  "please apply the injection patch manually.")
        else:
            src = src.replace(_RETURN_OLD, _RETURN_NEW, 1)
            loaders_path.write_text(src)
            print("    Patched load_by_bus() to apply BusInjections")
    else:
        print("    load_by_bus() injection patch already present — skipping")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--grid", required=True, choices=list(GRID_REGISTRY),
                   help="Source grid (e.g. RTS-GMLC).")
    p.add_argument("--output-grid", required=True,
                   help="Name for the new grid (e.g. RTS-GMLC-DC).")

    # Bus selection: explicit list OR auto-select all load buses
    bus_group = p.add_mutually_exclusive_group(required=True)
    bus_group.add_argument("--buses", nargs="+", metavar="BUS",
                           help="Bus names (as in bus.csv 'Bus Name' column) to inject load at.")
    bus_group.add_argument("--all-load-buses", action="store_true",
                           help="Inject into all buses that have non-zero MW Load in bus.csv.")

    # Amplitude: per-bus flat OR total distributed proportionally
    amp_group = p.add_mutually_exclusive_group()
    amp_group.add_argument("--amplitude-mw", type=float, default=None,
                           help="Mean DC load *per bus* in MW (flat, applied equally to every "
                                "selected bus). Mutually exclusive with --total-mw.")
    amp_group.add_argument("--total-mw", type=float, default=None,
                           help="Total DC load in MW distributed across selected buses "
                                "proportionally to each bus's existing MW Load in bus.csv. "
                                "Mutually exclusive with --amplitude-mw.")

    p.add_argument("--variation", type=float, default=0.05,
                   help="Fractional sinusoidal swing ±variation (default: 0.05).")
    p.add_argument("--period-hours", type=float, default=24.0,
                   help="Cycle length in hours (default: 24).")
    p.add_argument("--phase-hours", type=float, default=0.0,
                   help="Phase offset in hours from dataset start (default: 0).")
    p.add_argument("--vatic-root", type=Path, default=None,
                   help="VATIC repo root (auto-detected by default).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output grid without prompting.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.vatic_root:
        root = args.vatic_root.resolve()
        grids_dir  = root / "vatic" / "data" / "grids"
        init_dir   = grids_dir / "initial-state"
        loaders_py = root / "vatic" / "data" / "loaders.py"
    else:
        grids_dir, init_dir, loaders_py = GRIDS_DIR, INIT_DIR, LOADERS_PY

    source = args.grid
    output = args.output_grid
    # Match longest prefix first (Texas-7k_2030 before Texas-7k).
    cfg = next(v for k, v in GRID_REGISTRY.items() if source.startswith(k))

    if output == source:
        sys.exit("--output-grid must differ from --grid.")

    src_grid_dir = grids_dir / source
    src_data_dir = src_grid_dir / cfg["data_dir"]
    src_init_dir = init_dir / source

    if not src_data_dir.is_dir():
        sys.exit(f"Source grid data directory not found: {src_data_dir}")

    # -----------------------------------------------------------------------
    # 1. Copy grid directory
    # -----------------------------------------------------------------------
    dst_grid_dir = grids_dir / output
    dst_data_dir = dst_grid_dir / cfg["data_dir"]

    print(f"\n[1/4] Copying grid '{source}' → '{output}'")
    if dst_grid_dir.exists():
        if args.force:
            print(f"  '{dst_grid_dir}' already exists — overwriting (--force).")
            shutil.rmtree(dst_grid_dir)
        else:
            answer = input(f"  '{dst_grid_dir}' already exists. Overwrite? [y/N] ").strip().lower()
            if answer != "y":
                sys.exit("Aborted.")
            shutil.rmtree(dst_grid_dir)

    shutil.copytree(src_grid_dir, dst_grid_dir)
    print(f"  Copied → {dst_grid_dir}")

    # -----------------------------------------------------------------------
    # 2. Copy initial-state directory
    # -----------------------------------------------------------------------
    print(f"\n[2/4] Copying initial-state '{source}' → '{output}'")
    dst_init_dir = init_dir / output
    if src_init_dir.is_dir():
        if dst_init_dir.exists():
            shutil.rmtree(dst_init_dir)
        shutil.copytree(src_init_dir, dst_init_dir)
        print(f"  Copied → {dst_init_dir}")
    else:
        print(f"  No initial-state dir for '{source}' — skipping.")

    # -----------------------------------------------------------------------
    # 3. Validate buses and create BusInjections CSVs
    # -----------------------------------------------------------------------
    print(f"\n[3/4] Creating BusInjections timeseries")

    bus_csv = dst_data_dir / "SourceData" / "bus.csv"
    bus_df  = pd.read_csv(bus_csv)
    available = set(bus_df["Bus Name"].astype(str))

    # Resolve bus list
    if args.all_load_buses:
        load_mask = bus_df["MW Load"] > 0
        bus_list = bus_df.loc[load_mask, "Bus Name"].astype(str).tolist()
        print(f"  Auto-selected {len(bus_list)} buses with non-zero MW Load")
    else:
        bus_list = args.buses
        missing = [b for b in bus_list if b not in available]
        if missing:
            shutil.rmtree(dst_grid_dir)
            if dst_init_dir.exists():
                shutil.rmtree(dst_init_dir)
            sys.exit(
                f"Bus(es) not found in bus.csv: {missing}\n"
                f"Available buses: {sorted(available)}"
            )

    # Resolve per-bus amplitudes
    if args.total_mw is not None:
        # Distribute total_mw proportionally to each bus's existing MW Load
        bus_loads = bus_df.set_index("Bus Name")["MW Load"].astype(float)
        selected_loads = bus_loads.loc[bus_list]
        total_existing = selected_loads.sum()
        if total_existing == 0:
            sys.exit("Selected buses have zero total MW Load — cannot distribute proportionally.")
        bus_amplitudes = {
            bus: float(args.total_mw * selected_loads[bus] / total_existing)
            for bus in bus_list
        }
        total_dc = args.total_mw
        amp_description = f"proportional, total={total_dc:.1f} MW"
    else:
        # Flat amplitude per bus (default 100 MW if neither flag given)
        flat_amp = args.amplitude_mw if args.amplitude_mw is not None else 100.0
        bus_amplitudes = {bus: flat_amp for bus in bus_list}
        total_dc = flat_amp * len(bus_list)
        amp_description = f"flat {flat_amp} MW/bus, total={total_dc:.1f} MW"

    load_dir = dst_data_dir / "timeseries_data_files" / "Load"
    da_csv, rt_csv = find_load_csvs(load_dir)

    inject_dir = dst_data_dir / "timeseries_data_files" / "BusInjections"
    inject_dir.mkdir(exist_ok=True)

    da_inj = build_injection_df(
        da_csv, cfg["da_fmt"], bus_amplitudes,
        args.variation, args.period_hours, args.phase_hours,
    )
    rt_inj = build_injection_df(
        rt_csv, cfg["rt_fmt"], bus_amplitudes,
        args.variation, args.period_hours, args.phase_hours,
    )

    da_out = inject_dir / "DAY_AHEAD_bus_injections.csv"
    rt_out = inject_dir / "REAL_TIME_bus_injections.csv"
    da_inj.to_csv(da_out, index=False)
    rt_inj.to_csv(rt_out, index=False)

    # Report resolutions
    src_num = pd.read_csv(da_csv, nrows=500)
    da_res  = detect_resolution_minutes(src_num, cfg["da_fmt"])
    src_num = pd.read_csv(rt_csv, nrows=500)
    rt_res  = detect_resolution_minutes(src_num, cfg["rt_fmt"])

    print(f"  DAY_AHEAD injection: {len(da_inj)} rows @ {da_res:.0f} min → {da_out.name}")
    print(f"  REAL_TIME injection: {len(rt_inj)} rows @ {rt_res:.0f} min → {rt_out.name}")
    print(f"  Buses injected ({len(bus_list)}): {', '.join(bus_list)}")
    print(f"  Amplitude: {amp_description}  ±{args.variation*100:.1f}%  "
          f"period={args.period_hours}h, phase={args.phase_hours}h")

    # -----------------------------------------------------------------------
    # 4. Patch loaders.py
    # -----------------------------------------------------------------------
    print(f"\n[4/4] Patching loaders.py")
    patch_loaders_py(loaders_py)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    min_amp = min(bus_amplitudes.values())
    max_amp = max(bus_amplitudes.values())
    print(f"""
Done!  New grid: '{output}'

Bus injections (additive on top of existing load):
  Total DC load : {total_dc:.1f} MW  ({amp_description})
  Per-bus range : {min_amp:.2f} – {max_amp:.2f} MW
  Swing         : ±{args.variation*100:.1f}%
  Period        : {args.period_hours} h
  Phase         : {args.phase_hours} h from dataset start

Buses injected ({len(bus_list)}): {', '.join(bus_list)}

Injection files:
  {da_out}
  {rt_out}

No existing bus.csv, zone assignments, or load CSVs were modified.

Run:
  vatic-det --grid {output} --start-date <DATE> --num-days <N> [options]
""")


if __name__ == "__main__":
    main()
