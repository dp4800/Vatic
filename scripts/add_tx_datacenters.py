#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
add_tx_datacenters.py

Create a derived Texas-7k (or Texas-7k_2030) grid with data center load
injections placed at real Texas data center locations from a CSV file.

MW assignment
-------------
  1. Data centers with a listed MW Capacity keep that value exactly.
  2. Remaining MW  (total_mw − sum_of_known_mw)  is distributed across the
     data centers that have no MW listing:
       • Weighted by Square Footage for DCs that list sqft.
       • No-info DCs each receive the mean sqft weight (uniform allocation
         within this group, at the same per-DC average as the sqft group).
  3. Each DC is assigned to the nearest grid bus by Euclidean lat/lon
     distance.  Multiple DCs mapping to the same bus are summed.

This mirrors the methodology described in Rankin et al. (2024):
  "We weighted the load by square footage for the data centers that
   provided their square footage and spread the remaining load uniformly
   across the leftover data centers."

Load profile
------------
Injected load uses the same sinusoidal model as add_datacenter_load.py:

    load(t) = amplitude * (1 + variation * sin(2π(t − phase_h) / period_h))

where amplitude is the per-bus aggregate MW.

Usage
-----
    python scripts/add_tx_datacenters.py \\
        --grid        Texas-7k                          \\
        --output-grid Texas-7k-DC                       \\
        --dc-csv      inputs/TX_Data_Center_Info.csv    \\
        --total-mw    20300                             \\
        --variation   0.05                              \\
        --period-hours 24

Supported grids: Texas-7k, Texas-7k_2030
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

SCRIPT_DIR = Path(__file__).resolve().parent
VATIC_ROOT  = SCRIPT_DIR.parent
GRIDS_DIR   = VATIC_ROOT / "vatic" / "data" / "grids"
INIT_DIR    = GRIDS_DIR / "initial-state"
LOADERS_PY  = VATIC_ROOT / "vatic" / "data" / "loaders.py"

# ---------------------------------------------------------------------------
# Grid registry (Texas grids only)
# ---------------------------------------------------------------------------

GRID_REGISTRY: dict[str, dict] = {
    "Texas-7k_2030": {
        "data_dir": "TX2030_Data",
        "da_fmt":   "timestamps_da",
        "rt_fmt":   "timestamps_rt",
    },
    "Texas-7k": {
        "data_dir": "TX_Data",
        "da_fmt":   "period",
        "rt_fmt":   "period",
    },
}

# ---------------------------------------------------------------------------
# Time-series helpers (shared logic with add_datacenter_load.py)
# ---------------------------------------------------------------------------

def detect_resolution_minutes(df: pd.DataFrame, fmt: str) -> float:
    if fmt == "period":
        return 1440.0 / int(df["Period"].max())
    if fmt == "timestamps_da":
        times = pd.to_datetime(df["Forecast_time"], utc=True)
        return float((times.iloc[1] - times.iloc[0]).total_seconds() / 60)
    if fmt == "timestamps_rt":
        times = pd.to_datetime(df["Time"], utc=True)
        return float((times.iloc[1] - times.iloc[0]).total_seconds() / 60)
    raise ValueError(f"Unknown time format: {fmt!r}")


def sinusoidal_load(n: int, res_min: float, amplitude: float,
                    variation: float, period_h: float, phase_h: float) -> np.ndarray:
    t = np.arange(n) * (res_min / 60.0)
    return amplitude * (1.0 + variation * np.sin(2.0 * np.pi * (t - phase_h) / period_h))


def build_injection_df(source_csv: Path, fmt: str,
                       bus_amplitudes: dict[str, float],
                       variation: float, period_h: float, phase_h: float) -> pd.DataFrame:
    """Build a BusInjections DataFrame with the same time columns as source_csv."""
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

    src_numeric = pd.read_csv(source_csv, nrows=500)
    res_min = detect_resolution_minutes(src_numeric, fmt)
    n = len(src)

    wave_cols = {
        bus: sinusoidal_load(n, res_min, amplitude, variation, period_h, phase_h)
        for bus, amplitude in bus_amplitudes.items()
    }
    inj = pd.concat(
        [src[time_cols], pd.DataFrame(wave_cols, index=src.index)],
        axis=1,
    )
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
# MW assignment
# ---------------------------------------------------------------------------

def assign_dc_mw(dc_df: pd.DataFrame, total_mw: float) -> pd.Series:
    """Return a Series (index = dc_df.index) of MW load per data center.

    Rules
    -----
    1. DCs with a listed MW Capacity (including 0) keep that value.
    2. remaining = total_mw − sum(known MW)
       Among DCs without a listed MW:
         - Those with Square Footage receive a share weighted by their sqft.
         - Those with neither receive the *mean* sqft-group weight each
           (i.e., uniform allocation at the sqft-group average).
    """
    mw   = pd.to_numeric(dc_df["MW Capacity"],    errors="coerce")
    sqft = pd.to_numeric(dc_df["Square Footage"], errors="coerce")

    known_mask   = mw.notna()
    sqft_mask    = sqft.notna() & ~known_mask
    uniform_mask = ~known_mask & ~sqft.notna()

    known_sum = float(mw[known_mask].sum())
    remaining = max(total_mw - known_sum, 0.0)

    result = pd.Series(0.0, index=dc_df.index)
    result[known_mask] = mw[known_mask]

    if remaining > 0.0 and (sqft_mask.any() or uniform_mask.any()):
        # Impute mean sqft for no-info DCs so the two groups blend into a
        # single weighted pool (no-info DCs each get the average per-DC share).
        avg_sqft = float(sqft[sqft_mask].mean()) if sqft_mask.any() else 1.0

        weights = pd.Series(0.0, index=dc_df.index)
        weights[sqft_mask]    = sqft[sqft_mask].astype(float)
        weights[uniform_mask] = avg_sqft

        total_w = weights.sum()
        if total_w > 0.0:
            result[~known_mask] = remaining * weights[~known_mask] / total_w

    return result


# ---------------------------------------------------------------------------
# Nearest-bus matching
# ---------------------------------------------------------------------------

def nearest_bus(dc_lat: float, dc_lon: float,
                bus_lats: np.ndarray, bus_lons: np.ndarray) -> int:
    """Return the index in bus_lats/bus_lons closest to (dc_lat, dc_lon)."""
    dlat = bus_lats - dc_lat
    dlon = bus_lons - dc_lon
    return int(np.argmin(dlat * dlat + dlon * dlon))


# Known coordinate corrections keyed by exact DC name from the CSV.
# Use this for cases where the raw CSV has wrong coordinates.
_COORD_OVERRIDES: dict[str, tuple[float, float]] = {
    # "Project Ellen" has a missing minus sign AND slightly wrong value
    "Project Ellen": (26.2354, -97.69068627),
}


def aggregate_by_bus(dc_df: pd.DataFrame, mw_series: pd.Series,
                     bus_df: pd.DataFrame, top_k: int = 1,
                     branch_df: pd.DataFrame | None = None) -> dict[str, float]:
    """Map each DC to its top_k nearest buses and sum MW allocations per bus.

    When top_k > 1 each DC's MW is split across the k nearest buses.
    If branch_df is supplied the split uses capacity-weighted inverse-distance
    (weight ∝ line_capacity / distance²), which draws large DC loads toward
    high-capacity buses and avoids saturating low-voltage rural nodes.
    Without branch_df the split uses plain inverse-distance weighting.

    Returns dict {bus_name: total_mw} for buses that receive > 0 MW.
    Coordinate overrides in _COORD_OVERRIDES take precedence over CSV values.
    """
    bus_lats  = bus_df["lat"].to_numpy(dtype=float)
    bus_lons  = bus_df["lng"].to_numpy(dtype=float)
    bus_names = bus_df["Bus Name"].astype(str).tolist()
    bus_ids   = bus_df["Bus ID"].tolist()

    # Pre-compute per-bus total line capacity if branch data available
    if branch_df is not None:
        cap_map: dict = {}
        for bid in bus_ids:
            lines = branch_df[
                (branch_df["From Bus"] == bid) | (branch_df["To Bus"] == bid)
            ]
            cap_map[bid] = float(lines["Cont Rating"].sum())
        bus_caps = np.array([cap_map.get(bid, 1.0) for bid in bus_ids])
        # Replace zero-capacity buses with a small floor so they can still
        # receive a negligible share when they are the only nearby option.
        bus_caps = np.where(bus_caps > 0, bus_caps, 1.0)
    else:
        bus_caps = None

    k = max(1, min(top_k, len(bus_lats)))
    bus_mw: dict[str, float] = {}
    skipped = 0

    for idx, row in dc_df.iterrows():
        mw = float(mw_series.loc[idx])
        if mw <= 0.0:
            continue

        name = str(row["Name"])
        if name in _COORD_OVERRIDES:
            lat, lon = _COORD_OVERRIDES[name]
        else:
            lat = float(str(row["Latitude"]).rstrip(", "))
            lon = float(str(row["Logitude"]).rstrip(", "))

        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            skipped += 1
            continue

        dlat = bus_lats - lat
        dlon = bus_lons - lon
        sq_dist = dlat * dlat + dlon * dlon

        if k == 1:
            indices = [int(np.argmin(sq_dist))]
            weights = np.array([1.0])
        else:
            indices = list(np.argpartition(sq_dist, k)[:k])
            d = sq_dist[indices]
            # Exact coordinate match → assign 100% to that bus
            if np.any(d == 0.0):
                zero_i = next(i for i, v in zip(indices, d) if v == 0.0)
                indices = [zero_i]
                weights = np.array([1.0])
            elif bus_caps is not None:
                cap_k = bus_caps[indices]
                score = cap_k / d          # capacity-weighted inverse-distance
                weights = score / score.sum()
            else:
                inv = 1.0 / d
                weights = inv / inv.sum()

        for bi, w in zip(indices, weights):
            bname = bus_names[bi]
            bus_mw[bname] = bus_mw.get(bname, 0.0) + mw * float(w)

    if skipped:
        print(f"  WARNING: skipped {skipped} DC(s) with invalid coordinates")

    return bus_mw


# ---------------------------------------------------------------------------
# loaders.py patch (identical to add_datacenter_load.py)
# ---------------------------------------------------------------------------

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
    src = loaders_path.read_text()
    if "# --- Bus-level load injections" not in src:
        if _RETURN_OLD not in src:
            print("    WARNING: Could not locate load_by_bus() return line — "
                  "apply the injection patch manually.")
        else:
            src = src.replace(_RETURN_OLD, _RETURN_NEW, 1)
            loaders_path.write_text(src)
            print("    Patched load_by_bus() to apply BusInjections")
    else:
        print("    load_by_bus() injection patch already present — skipping")


# ---------------------------------------------------------------------------
# Per-bus cap + overflow redistribution
# ---------------------------------------------------------------------------

def redistribute_overflow(bus_mw: dict[str, float],
                          cap_map: dict,
                          bus_name_to_id: dict,
                          cap_fraction: float = 0.9,
                          max_iter: int = 50) -> dict[str, float]:
    """Cap per-bus injection at cap_fraction × line capacity.

    Any bus exceeding the cap has its excess redistributed proportionally to
    all other buses weighted by their remaining headroom.  Iterates until
    convergence (no bus exceeds the cap) or max_iter is reached.

    cap_fraction=0.0 disables capping (returns bus_mw unchanged).
    """
    if cap_fraction <= 0.0:
        return bus_mw

    bus_mw = dict(bus_mw)   # work on a copy

    for _ in range(max_iter):
        # Identify buses that exceed their cap
        overflow_total = 0.0
        capped: dict[str, float] = {}
        for bname, bmw in bus_mw.items():
            bid = bus_name_to_id.get(bname)
            cap = cap_map.get(bid, 0.0) * cap_fraction
            if cap > 0.0 and bmw > cap:
                overflow_total += bmw - cap
                capped[bname]   = cap

        if overflow_total == 0.0:
            break   # converged

        # Apply caps
        for bname, cval in capped.items():
            bus_mw[bname] = cval

        # Distribute overflow to buses with remaining headroom
        headroom: dict[str, float] = {}
        for bname, bmw in bus_mw.items():
            if bname in capped:
                continue
            bid = bus_name_to_id.get(bname)
            cap = cap_map.get(bid, 0.0) * cap_fraction
            if cap > 0.0 and bmw < cap:
                headroom[bname] = cap - bmw

        total_headroom = sum(headroom.values())
        if total_headroom == 0.0:
            # No room anywhere — just spread across all buses uniformly
            all_buses = [b for b in bus_mw if b not in capped]
            if all_buses:
                share = overflow_total / len(all_buses)
                for b in all_buses:
                    bus_mw[b] += share
            break

        for bname, room in headroom.items():
            bus_mw[bname] += overflow_total * room / total_headroom

    return bus_mw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--grid", required=True, choices=list(GRID_REGISTRY),
                   help="Source grid name.")
    p.add_argument("--output-grid", required=True,
                   help="Name for the new derived grid.")
    p.add_argument("--dc-csv", type=Path,
                   default=VATIC_ROOT / "inputs" / "TX_Data_Center_Info.csv",
                   help="Path to TX_Data_Center_Info.csv "
                        "(default: inputs/TX_Data_Center_Info.csv).")
    p.add_argument("--total-mw", type=float, required=True,
                   help="Total DC load in MW across all data centers.  DCs "
                        "with a listed MW keep their value; the remainder is "
                        "distributed among unlisted DCs (sqft-weighted where "
                        "available, else uniform).")
    p.add_argument("--variation", type=float, default=0.05,
                   help="Sinusoidal swing ±variation (default: 0.05).")
    p.add_argument("--period-hours", type=float, default=24.0,
                   help="Cycle length in hours (default: 24).")
    p.add_argument("--phase-hours", type=float, default=0.0,
                   help="Phase offset in hours from dataset start (default: 0).")
    p.add_argument("--top-k", type=int, default=3,
                   help="Distribute each DC's load across the k nearest buses "
                        "using capacity-weighted inverse-distance (default: 3). "
                        "Use 1 to revert to nearest-bus-only behaviour.")
    p.add_argument("--per-bus-cap", type=float, default=0.9,
                   help="After initial assignment, cap each bus at this "
                        "fraction of its total line capacity and redistribute "
                        "any excess to buses with remaining headroom "
                        "(default: 0.9).  Set to 0 to disable.")
    p.add_argument("--vatic-root", type=Path, default=None,
                   help="VATIC repo root (auto-detected by default).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.vatic_root:
        root       = args.vatic_root.resolve()
        grids_dir  = root / "vatic" / "data" / "grids"
        init_dir   = grids_dir / "initial-state"
        loaders_py = root / "vatic" / "data" / "loaders.py"
    else:
        grids_dir, init_dir, loaders_py = GRIDS_DIR, INIT_DIR, LOADERS_PY

    source = args.grid
    output = args.output_grid
    cfg    = next(v for k, v in GRID_REGISTRY.items() if source.startswith(k))

    if output == source:
        sys.exit("--output-grid must differ from --grid.")

    src_grid_dir = grids_dir / source
    src_data_dir = src_grid_dir / cfg["data_dir"]
    src_init_dir = init_dir / source

    if not src_data_dir.is_dir():
        sys.exit(f"Source grid data directory not found: {src_data_dir}")

    # -----------------------------------------------------------------------
    # 1. Load and process data center CSV
    # -----------------------------------------------------------------------
    print(f"\n[1/5] Loading data center CSV: {args.dc_csv}")
    dc_df = pd.read_csv(args.dc_csv, encoding="latin-1")
    print(f"  {len(dc_df)} data centers loaded")

    mw_raw  = pd.to_numeric(dc_df["MW Capacity"],    errors="coerce")
    sqft_raw = pd.to_numeric(dc_df["Square Footage"], errors="coerce")
    n_known   = int(mw_raw.notna().sum())
    n_sqft    = int((sqft_raw.notna() & mw_raw.isna()).sum())
    n_uniform = int((mw_raw.isna() & sqft_raw.isna()).sum())
    known_sum = float(mw_raw.fillna(0.0).sum())

    print(f"  Known MW:              {n_known:>4} DCs  ({known_sum:.1f} MW total)")
    print(f"  Sqft only (no MW):     {n_sqft:>4} DCs")
    print(f"  Neither:               {n_uniform:>4} DCs")
    print(f"  Target total:          {args.total_mw:.1f} MW")
    print(f"  To distribute (rest):  {max(args.total_mw - known_sum, 0):.1f} MW")

    dc_mw = assign_dc_mw(dc_df, args.total_mw)
    print(f"  Assigned MW — min: {dc_mw.min():.3f}  mean: {dc_mw.mean():.3f}  "
          f"max: {dc_mw.max():.3f}  sum: {dc_mw.sum():.1f}")

    # -----------------------------------------------------------------------
    # 2. Match DCs to nearest bus
    # -----------------------------------------------------------------------
    print(f"\n[2/5] Matching data centers to nearest bus")
    bus_csv    = src_data_dir / "SourceData" / "bus.csv"
    branch_csv = src_data_dir / "SourceData" / "branch.csv"
    bus_df    = pd.read_csv(bus_csv)
    branch_df = pd.read_csv(branch_csv) if branch_csv.exists() else None

    bus_mw = aggregate_by_bus(dc_df, dc_mw, bus_df, top_k=args.top_k,
                              branch_df=branch_df)
    n_buses_used = len(bus_mw)
    mode_str = (f"top_k={args.top_k}, capacity-weighted"
                if branch_df is not None and args.top_k > 1
                else f"top_k={args.top_k}")
    print(f"  {len(dc_df)} DCs → {n_buses_used} unique buses  "
          f"({mode_str}, total injected: {sum(bus_mw.values()):.1f} MW)")
    print(f"  Per-bus range: {min(bus_mw.values()):.2f} – {max(bus_mw.values()):.2f} MW")

    if args.per_bus_cap > 0.0 and branch_df is not None:
        bus_name_to_id = dict(zip(bus_df["Bus Name"].astype(str),
                                  bus_df["Bus ID"].tolist()))
        cap_map = {}
        for bid in bus_df["Bus ID"].tolist():
            lines = branch_df[
                (branch_df["From Bus"] == bid) | (branch_df["To Bus"] == bid)
            ]
            cap_map[bid] = float(lines["Cont Rating"].sum())

        bus_mw = redistribute_overflow(bus_mw, cap_map, bus_name_to_id,
                                       cap_fraction=args.per_bus_cap)
        n_over = sum(
            1 for bname, bmw in bus_mw.items()
            if (bid := bus_name_to_id.get(bname)) is not None
            and cap_map.get(bid, 0) > 0
            and bmw > cap_map[bid] * args.per_bus_cap
        )
        print(f"  After overflow redistribution (cap={args.per_bus_cap:.0%}): "
              f"total={sum(bus_mw.values()):.1f} MW, "
              f"max={max(bus_mw.values()):.1f} MW, "
              f"buses still over cap={n_over}")

    # Verify all bus names exist
    available = set(bus_df["Bus Name"].astype(str))
    missing = [b for b in bus_mw if b not in available]
    if missing:
        sys.exit(f"BUG: nearest-bus lookup returned unknown names: {missing}")

    # -----------------------------------------------------------------------
    # 3. Copy grid directory
    # -----------------------------------------------------------------------
    dst_grid_dir = grids_dir / output
    dst_data_dir = dst_grid_dir / cfg["data_dir"]

    print(f"\n[3/5] Copying grid '{source}' → '{output}'")
    if dst_grid_dir.exists():
        answer = input(f"  '{dst_grid_dir}' already exists. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            sys.exit("Aborted.")
        shutil.rmtree(dst_grid_dir)

    shutil.copytree(src_grid_dir, dst_grid_dir)
    print(f"  Copied → {dst_grid_dir}")

    dst_init_dir = init_dir / output
    if src_init_dir.is_dir():
        if dst_init_dir.exists():
            shutil.rmtree(dst_init_dir)
        shutil.copytree(src_init_dir, dst_init_dir)
        print(f"  Initial-state copied → {dst_init_dir}")
    else:
        print(f"  No initial-state dir for '{source}' — skipping.")

    # -----------------------------------------------------------------------
    # 4. Write BusInjections CSVs
    # -----------------------------------------------------------------------
    print(f"\n[4/5] Writing BusInjections timeseries")

    load_dir   = dst_data_dir / "timeseries_data_files" / "Load"
    da_csv, rt_csv = find_load_csvs(load_dir)

    inject_dir = dst_data_dir / "timeseries_data_files" / "BusInjections"
    inject_dir.mkdir(exist_ok=True)

    da_inj = build_injection_df(da_csv, cfg["da_fmt"], bus_mw,
                                args.variation, args.period_hours, args.phase_hours)
    rt_inj = build_injection_df(rt_csv, cfg["rt_fmt"], bus_mw,
                                args.variation, args.period_hours, args.phase_hours)

    da_out = inject_dir / "DAY_AHEAD_bus_injections.csv"
    rt_out = inject_dir / "REAL_TIME_bus_injections.csv"
    da_inj.to_csv(da_out, index=False)
    rt_inj.to_csv(rt_out, index=False)

    src_num = pd.read_csv(da_csv, nrows=500)
    da_res  = detect_resolution_minutes(src_num, cfg["da_fmt"])
    src_num = pd.read_csv(rt_csv, nrows=500)
    rt_res  = detect_resolution_minutes(src_num, cfg["rt_fmt"])

    print(f"  DAY_AHEAD: {len(da_inj):>6} rows @ {da_res:.0f} min  →  {da_out.name}")
    print(f"  REAL_TIME: {len(rt_inj):>6} rows @ {rt_res:.0f} min  →  {rt_out.name}")

    # -----------------------------------------------------------------------
    # 5. Patch loaders.py
    # -----------------------------------------------------------------------
    print(f"\n[5/5] Patching loaders.py")
    patch_loaders_py(loaders_py)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"""
Done!  New grid: '{output}'

Data center load injections (added on top of existing bus loads):
  Source CSV     : {args.dc_csv}
  Total DCs      : {len(dc_df)}  ({n_known} known-MW, {n_sqft} sqft-only, {n_uniform} unknown)
  Total MW       : {dc_mw.sum():.1f} MW  (target: {args.total_mw:.1f} MW)
  Buses injected : {n_buses_used}
  Per-bus range  : {min(bus_mw.values()):.2f} – {max(bus_mw.values()):.2f} MW
  Swing          : ±{args.variation*100:.1f}%
  Period         : {args.period_hours} h
  Phase          : {args.phase_hours} h from dataset start

Injection files:
  {da_out}
  {rt_out}

Run:
  vatic-det --grid {output} --start-date <DATE> --num-days <N> [options]
""")


if __name__ == "__main__":
    main()
