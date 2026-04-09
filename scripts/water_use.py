#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
water_use.py — Hourly water withdrawal and consumption from dispatch.

Combines EIA Form 860 Schedule 6 Part D cooling data with Vatic simulation
outputs to estimate water use at the generator, fuel-type, and system level.

Rate assignment hierarchy (per generator):
  1. EIA direct match by (Plant Code, Generator ID) if those columns exist in gen.csv
  2. EIA technology-weighted average for the matching (Unit Type, Fuel) class
  3. Literature fallback (Meldrum et al. 2013 / NETL medians by tech + cooling)

Usage
-----
    # Build rates once — writes gen_water_rates.csv:
    python scripts/water_use.py build-rates \\
        --eia-file inputs/Cooling_Boiler_Generator_Data_Texas_2024.csv \\
        --grid-dir vatic/data/grids/RTS-GMLC \\
        --out      outputs/gen_water_rates.csv

    # Estimate hourly water use from a simulation run:
    python scripts/water_use.py compute \\
        --sim-dir  outputs/2020-01-13/baseline \\
        --rates    outputs/gen_water_rates.csv \\
        --out-dir  outputs/2020-01-13/water

    # Both steps in one call (auto-builds rates if --rates file is absent):
    python scripts/water_use.py run \\
        --sim-dir  outputs/2020-01-13/baseline \\
        --grid-dir vatic/data/grids/RTS-GMLC \\
        --eia-file inputs/Cooling_Boiler_Generator_Data_Texas_2024.csv \\
        --out-dir  outputs/2020-01-13/water
"""

import argparse
import csv
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Grid registry — maps grid name prefix to its SourceData parent directory.
# ---------------------------------------------------------------------------
_GRID_REGISTRY: dict[str, str] = {
    "Texas-7k_2030": "TX2030_Data",
    "Texas-7k":      "TX_Data",
    "RTS-GMLC":      "RTS_Data",
}


def _data_dir_for(grid_dir: Path) -> str:
    """Return the SourceData parent directory name for a grid path."""
    name = grid_dir.name
    for prefix, data_dir in sorted(_GRID_REGISTRY.items(), key=lambda x: len(x[0]), reverse=True):
        if name.startswith(prefix):
            return data_dir
    return "RTS_Data"


def _gen_csv_for(grid_dir: Path) -> Path:
    return grid_dir / _data_dir_for(grid_dir) / "SourceData" / "gen.csv"

# ---------------------------------------------------------------------------
# Literature fallback rates (Meldrum et al. 2013; NETL 2010; EPRI 2011).
# Keys match actual gen.csv "Unit Type" (uppercase) and "Fuel" (as-is, lowered).
# Values: (withdrawal_gal_per_mwh, consumption_gal_per_mwh).
# Recirculating cooling tower assumed where cooling type is unknown.
# Hydro rates reflect reservoir evaporation (Fthenakis & Kim 2010).
# ---------------------------------------------------------------------------
_LITERATURE_RATES: dict[tuple[str, str], tuple[float, float]] = {
    # Consumption rates from Table AII (Macknick et al. [16]).
    # Withdrawal rates from Meldrum et al. 2013 / NETL medians.
    # Format: (withdrawal_gal_mwh, consumption_gal_mwh)
    # ── Thermal ─────────────────────────────────────────────────────────────
    ("STEAM",    "coal"):     (1_000.0,  530.0),   # Table AII: coal 530
    ("STEAM",    "ng"):       (  950.0,  210.0),   # Table AII: NG 210
    ("STEAM",    "oil"):      (1_000.0,  530.0),   # Table AII: oil 530
    ("STEAM",    "pc"):       (1_000.0,  530.0),   # Table AII: pet. coke 530
    ("NUCLEAR",  "nuclear"):  (1_101.0,  720.0),   # Table AII: nuclear 720
    ("CC",       "ng"):       (  255.0,  210.0),   # Table AII: NG 210
    ("CC",       "oil"):      (  255.0,  210.0),
    # Combustion turbines: open-cycle, no steam cooling
    ("CT",       "ng"):       (    0.0,    0.0),
    ("CT",       "oil"):      (    0.0,    0.0),
    ("CT",       "coal"):     (    0.0,    0.0),
    # Biopower (Table AII: 480)
    ("STEAM",    "biomass"):  (1_000.0,  480.0),
    # ── Renewables ──────────────────────────────────────────────────────────
    ("HYDRO",    "hydro"):    (4_491.0, 4_491.0),  # Table AII: hydro 4491
    ("ROR",      "hydro"):    (    0.0,    0.0),   # run-of-river: minimal
    ("PV",       "solar"):    (    0.0,    0.0),
    ("RTPV",     "solar"):    (    0.0,    0.0),
    ("WIND",     "wind"):     (    1.0,    1.0),   # Table AII: wind 1
    ("CSP",      "solar"):    (   81.0,   81.0),   # Table AII: solar 81
    # ── Storage / misc ──────────────────────────────────────────────────────
    ("STORAGE",  "storage"):  (    0.0,    0.0),   # Table AII: battery 0
    ("SYNC_COND","sync_cond"):(    0.0,    0.0),
}

# EIA "Generator Primary Technology" → (vatic_unit_type, vatic_fuel_lowercase).
# Used to join EIA aggregate rates back to Vatic generators.
# Fuel strings match gen.csv "Fuel" column values lowercased.
_EIA_TECH_TO_VATIC: dict[str, tuple[str, str]] = {
    "Conventional Steam Coal":                      ("STEAM",    "coal"),
    "Coal Integrated Gasification Combined Cycle":  ("STEAM",    "coal"),
    "Natural Gas Steam Turbine":                    ("STEAM",    "ng"),
    "Natural Gas Fired Combined Cycle":             ("CC",       "ng"),
    "Natural Gas Fired Combustion Turbine":         ("CT",       "ng"),
    "Petroleum Liquids":                            ("CT",       "oil"),
    "Petroleum Coke":                               ("STEAM",    "oil"),
    "Nuclear":                                      ("NUCLEAR",  "nuclear"),
    "Conventional Hydroelectric":                   ("HYDRO",    "hydro"),
    "Pumped Storage":                               ("STORAGE",  "storage"),
    "Solar Photovoltaic":                           ("PV",       "solar"),
    "Small Scale Solar Photovoltaic":               ("RTPV",     "solar"),
    "Onshore Wind Turbine":                         ("WIND",     "wind"),
    "Offshore Wind Turbine":                        ("WIND",     "wind"),
    "Solar Thermal without Energy Storage":         ("CSP",      "solar"),
    "Solar Thermal with Energy Storage":            ("CSP",      "solar"),
    "Wood/Wood Waste Biomass":                      ("STEAM",    "coal"),
    "Other Gases":                                  ("CT",       "ng"),
}


# ---------------------------------------------------------------------------
# EIA file parsing
# ---------------------------------------------------------------------------

def _load_eia_csv(eia_path: Path) -> pd.DataFrame:
    """
    Parse EIA Form 860 Schedule 6 Part D CSV.

    The file has a multi-line quoted header (title rows) before the actual
    column names.  We locate the header by scanning for the row that contains
    both "Plant Code" and "Utility ID", join it with the following lines to
    capture any quoted continuation, then read the data rows below.

    Returns a DataFrame with cleaned, numeric-coerced columns.
    """
    raw   = eia_path.read_text(encoding="latin-1")
    lines = raw.splitlines()

    # Locate the line that starts the column-name block
    hdr_line = next(
        (i for i, l in enumerate(lines)
         if "Plant Code" in l and ("Utility ID" in l or "Generator ID" in l)),
        None,
    )
    if hdr_line is None:
        raise ValueError(f"Cannot locate column header in {eia_path}")

    # Header may span up to 3 lines due to quoted embedded newlines
    hdr_text = lines[hdr_line]
    for extra in lines[hdr_line + 1 : hdr_line + 4]:
        hdr_text += chr(10) + extra
        if len(list(csv.reader(io.StringIO(hdr_text)))[0]) >= 60:
            break

    col_names = [c.strip().strip('"') for c in list(csv.reader(io.StringIO(hdr_text)))[0]]

    # Data rows start after the header block
    data_start = hdr_line + hdr_text.count(chr(10)) + 1
    data_text  = chr(10).join(lines[data_start:])

    # Parse — filter non-numeric leading field (footers, blank lines)
    parsed = [
        r for r in csv.reader(io.StringIO(data_text))
        if r and r[0].strip().lstrip('"').lstrip("'").isdigit()
    ]

    # Align column count: pad short rows, truncate long rows
    n       = len(col_names)
    aligned = [row[:n] + [""] * max(0, n - len(row)) for row in parsed]

    df = pd.DataFrame(aligned, columns=col_names)

    # Strip commas and coerce numeric columns
    _numeric = [
        "Water Withdrawal Volume (Million Gallons)",
        "Water Consumption Volume (Million Gallons)",
        "Water Withdrawal Intensity Rate (Gallons / MWh)",
        "Water Consumption Intensity Rate (Gallons / MWh)",
        "Net Generation from Steam Turbines (MWh)",
        "Net Generation Associated with Single Shaft Combined Cycle Units (MWh)",
        "Net Generation Associated with Combined Cycle Gas Turbines (MWh)",
        "Plant Code",
    ]
    for col in _numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )

    # Normalise cooling type: prefer 923 code, fall back to 860 Type 1
    ct_col = "923 Cooling Type" if "923 Cooling Type" in df.columns else "860 Cooling Type 1"
    df["_cooling_type"] = df[ct_col].str.strip()

    # Aggregate net generation across turbine sub-types
    gen_cols = [
        "Net Generation from Steam Turbines (MWh)",
        "Net Generation Associated with Single Shaft Combined Cycle Units (MWh)",
        "Net Generation Associated with Combined Cycle Gas Turbines (MWh)",
    ]
    df["_net_gen_mwh"] = sum(df[c].fillna(0.0) for c in gen_cols if c in df.columns)

    return df


# ---------------------------------------------------------------------------
# EIA aggregate rates by (EIA technology, cooling type)
# ---------------------------------------------------------------------------

def _build_eia_rates(eia_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute generation-weighted average water intensity rates by
    (Generator Primary Technology, cooling type).

    Returns a DataFrame with columns:
        eia_tech, cooling_type, wd_gal_mwh, wc_gal_mwh, gen_gwh, n_obs

    A sentinel row with cooling_type == "__all__" is added for each
    technology, giving the tech-wide weighted average across cooling types.
    """
    wd_col = "Water Withdrawal Intensity Rate (Gallons / MWh)"
    wc_col = "Water Consumption Intensity Rate (Gallons / MWh)"
    tc_col = "Generator Primary Technology"

    needed = [wd_col, wc_col, tc_col, "_net_gen_mwh", "_cooling_type"]
    missing = [c for c in needed if c not in eia_df.columns]
    if missing:
        raise ValueError(f"EIA DataFrame missing columns: {missing}")

    sub = eia_df[needed].copy()
    sub[wd_col] = pd.to_numeric(sub[wd_col], errors="coerce")
    sub[wc_col] = pd.to_numeric(sub[wc_col], errors="coerce")

    mask = (sub["_net_gen_mwh"] > 0) & (sub[wd_col] > 0)
    sub  = sub[mask]

    if sub.empty:
        warnings.warn("EIA data has no rows with positive generation and water rates.")
        return pd.DataFrame(
            columns=["eia_tech", "cooling_type", "wd_gal_mwh", "wc_gal_mwh", "gen_gwh", "n_obs"]
        )

    agg_rows = []

    def _wagg(grp: pd.DataFrame) -> tuple[float, float, float, int]:
        w  = grp["_net_gen_mwh"].values
        wd = np.average(grp[wd_col].values, weights=w)
        wc = np.average(grp[wc_col].values, weights=w)
        return float(wd), float(wc), float(w.sum()), len(grp)

    for (tech, cool), grp in sub.groupby([tc_col, "_cooling_type"]):
        wd, wc, gen, n = _wagg(grp)
        agg_rows.append(dict(eia_tech=tech, cooling_type=cool,
                             wd_gal_mwh=round(wd, 1), wc_gal_mwh=round(wc, 1),
                             gen_gwh=round(gen / 1e6, 4), n_obs=n))

    for tech, grp in sub.groupby(tc_col):
        wd, wc, gen, n = _wagg(grp)
        agg_rows.append(dict(eia_tech=tech, cooling_type="__all__",
                             wd_gal_mwh=round(wd, 1), wc_gal_mwh=round(wc, 1),
                             gen_gwh=round(gen / 1e6, 4), n_obs=n))

    return pd.DataFrame(agg_rows)


def _lookup_eia_rate(
    eia_rates:      pd.DataFrame,
    vatic_utype:    str,
    vatic_fuel_lc:  str,
    cooling_type:   str | None = None,
) -> tuple[float, float, str] | None:
    """
    Return (wd_gal_mwh, wc_gal_mwh, source_tag) from EIA aggregate table,
    or None if no match.
    """
    if eia_rates.empty:
        return None

    vatic_key = (vatic_utype.upper(), vatic_fuel_lc)
    eia_tech  = next(
        (k for k, v in _EIA_TECH_TO_VATIC.items() if v == vatic_key), None
    )
    if eia_tech is None:
        return None

    sub = eia_rates[eia_rates["eia_tech"] == eia_tech]
    if sub.empty:
        return None

    if cooling_type:
        row = sub[sub["cooling_type"] == cooling_type]
        if not row.empty:
            r = row.iloc[0]
            return (r["wd_gal_mwh"], r["wc_gal_mwh"], f"eia:{eia_tech}+{cooling_type}")

    row = sub[sub["cooling_type"] == "__all__"]
    if not row.empty:
        r = row.iloc[0]
        return (r["wd_gal_mwh"], r["wc_gal_mwh"], f"eia:{eia_tech}")

    return None


# ---------------------------------------------------------------------------
# Per-generator rate table
# ---------------------------------------------------------------------------

def build_water_rates(
    eia_path: Path,
    gen_df:   pd.DataFrame,
    out_csv:  Path | None = None,
) -> pd.DataFrame:
    """
    Assign water withdrawal and consumption rates to every generator in gen_df.

    Parameters
    ----------
    eia_path : path to EIA Form 860 Schedule 6 Part D CSV
    gen_df   : gen.csv loaded as a DataFrame
    out_csv  : if given, write the rate table here for later reuse

    Returns
    -------
    DataFrame with one row per generator:
        gen_uid, unit_type, fuel, eia_tech,
        withdrawal_gal_mwh, consumption_gal_mwh, rate_source
    """
    print(f"[WATER] Loading EIA data from {eia_path}")
    eia_df    = _load_eia_csv(eia_path)
    eia_rates = _build_eia_rates(eia_df)
    print(f"[WATER] EIA aggregate rows: {len(eia_rates)}")

    # Optional direct EIA match if gen.csv carries EIA Plant Code + Generator ID
    has_plant_code = "EIA Plant Code" in gen_df.columns
    direct_lookup: dict[tuple[int, str], tuple[float, float]] = {}
    if has_plant_code:
        wd_col = "Water Withdrawal Intensity Rate (Gallons / MWh)"
        wc_col = "Water Consumption Intensity Rate (Gallons / MWh)"
        valid  = eia_df[(eia_df[wd_col].notna()) & (eia_df["_net_gen_mwh"] > 0)]
        for _, erow in valid.iterrows():
            key = (int(erow["Plant Code"]), str(erow["Generator ID"]).strip())
            direct_lookup[key] = (float(erow[wd_col]), float(erow[wc_col]))

    records = []
    # Use iterrows() — preserves original column names (including those with spaces)
    for _, row in gen_df.iterrows():
        uid   = str(row.get("GEN UID",    "")).strip()
        utype = str(row.get("Unit Type",  "")).strip()
        fuel  = str(row.get("Fuel",       "")).strip()
        cat   = str(row.get("Category",   "")).strip()

        fuel_lc = fuel.lower()
        wd = wc = 0.0
        source  = "default:0"

        # ── 1. Direct EIA plant-level match ──────────────────────────────
        if has_plant_code:
            pc  = row.get("EIA Plant Code")
            gid = str(row.get("EIA Generator ID", "")).strip()
            if pc and (int(pc), gid) in direct_lookup:
                wd, wc = direct_lookup[(int(pc), gid)]
                source = f"eia:direct:{pc}:{gid}"

        # ── 2. EIA technology aggregate ───────────────────────────────────
        if source == "default:0":
            result = _lookup_eia_rate(eia_rates, utype, fuel_lc)
            if result is not None:
                wd, wc, source = result

        # ── 3. Literature fallback ────────────────────────────────────────
        if source == "default:0":
            # Try direct key first (works for grids with short-form unit types)
            lit_key = (utype.upper(), fuel_lc)
            if lit_key in _LITERATURE_RATES:
                wd, wc = _LITERATURE_RATES[lit_key]
                source = f"literature:{utype}:{fuel_lc}"
            else:
                # Translate full unit type name via _EIA_TECH_TO_VATIC mapping
                vatic_key = _EIA_TECH_TO_VATIC.get(utype)
                if vatic_key and vatic_key in _LITERATURE_RATES:
                    wd, wc = _LITERATURE_RATES[vatic_key]
                    source = f"literature:{vatic_key[0]}:{vatic_key[1]}"
                else:
                    # Partial keyword match on fuel (handles edge cases)
                    for (ut, fk), (wd_, wc_) in _LITERATURE_RATES.items():
                        if ut == utype.upper() and (fk in fuel_lc or fuel_lc in fk):
                            wd, wc = wd_, wc_
                            source = f"literature:{ut}:{fk}"
                            break

        eia_tech = next(
            (k for k, v in _EIA_TECH_TO_VATIC.items()
             if v == (utype.upper(), fuel_lc)), ""
        )

        records.append({
            "gen_uid":             uid,
            "unit_type":           utype,
            "fuel":                fuel,
            "category":            cat,
            "eia_tech":            eia_tech,
            "withdrawal_gal_mwh":  round(wd, 2),
            "consumption_gal_mwh": round(wc, 2),
            "rate_source":         source,
        })

    result = pd.DataFrame(records)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_csv, index=False)
        print(f"[WATER] Rates written → {out_csv}  ({len(result)} generators)")

    return result


# ---------------------------------------------------------------------------
# Hourly water use computation
# ---------------------------------------------------------------------------

def compute_water_use(
    thermal_df:  pd.DataFrame,
    renew_df:    pd.DataFrame,
    water_rates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute hourly water withdrawal and consumption from dispatch.

    Thermal generators: Dispatch (MW × 1 h = MWh) × rate.
    Hydro from renew_df: Output × reservoir evaporation rate.
    Wind / PV / other renewables: zero water (rate = 0 by default).

    Returns
    -------
    gen_hourly    : Date, Hour, Generator, fuel, unit_type, wd_gal, wc_gal
    fuel_hourly   : Date, Hour, fuel, wd_gal, wc_gal
    system_hourly : Date, Hour, total_wd_gal, total_wc_gal
    """
    rate_map = {
        row["gen_uid"]: (row["withdrawal_gal_mwh"], row["consumption_gal_mwh"])
        for _, row in water_rates.iterrows()
    }
    meta_map = {
        row["gen_uid"]: (row["fuel"], row["unit_type"])
        for _, row in water_rates.iterrows()
    }

    rows: list[dict] = []

    # Thermal dispatch
    for rec in thermal_df.itertuples(index=False):
        gen        = rec.Generator
        mwh        = float(rec.Dispatch)
        wd_r, wc_r = rate_map.get(gen, (0.0, 0.0))
        fuel, utype = meta_map.get(gen, ("", ""))
        rows.append({
            "Date":      rec.Date,
            "Hour":      rec.Hour,
            "Generator": gen,
            "fuel":      fuel,
            "unit_type": utype,
            "wd_gal":    mwh * wd_r,
            "wc_gal":    mwh * wc_r,
        })

    # Hydro (reservoir evaporation); other renewables have zero rates
    hydro_utypes = {"HYDRO", "HY", "ROR", "PS"}
    hydro_uids   = set(
        water_rates.loc[
            water_rates["unit_type"].str.upper().isin(hydro_utypes), "gen_uid"
        ]
    )
    for rec in renew_df.itertuples(index=False):
        gen = rec.Generator
        if gen not in hydro_uids:
            continue
        mwh        = float(rec.Output)
        wd_r, wc_r = rate_map.get(gen, (0.0, 0.0))
        fuel, utype = meta_map.get(gen, ("", ""))
        rows.append({
            "Date":      rec.Date,
            "Hour":      rec.Hour,
            "Generator": gen,
            "fuel":      fuel,
            "unit_type": utype,
            "wd_gal":    mwh * wd_r,
            "wc_gal":    mwh * wc_r,
        })

    gen_hourly = pd.DataFrame(rows)
    if gen_hourly.empty:
        empty = pd.DataFrame(columns=["Date", "Hour", "wd_gal", "wc_gal"])
        return gen_hourly, empty.copy(), empty.copy()

    fuel_hourly = (
        gen_hourly
        .groupby(["Date", "Hour", "fuel"])[["wd_gal", "wc_gal"]]
        .sum()
        .reset_index()
    )

    system_hourly = (
        gen_hourly
        .groupby(["Date", "Hour"])[["wd_gal", "wc_gal"]]
        .sum()
        .reset_index()
        .rename(columns={"wd_gal": "total_wd_gal", "wc_gal": "total_wc_gal"})
    )

    return gen_hourly, fuel_hourly, system_hourly


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run(
    sim_dir:        Path,
    grid_dir:       Path,
    eia_path:       Path,
    out_dir:        Path,
    rate_csv:       Path | None = None,
    water_rates_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: build rates → compute hourly water use → write CSVs.

    Outputs written to out_dir:
        gen_water_rates.csv     — rate table (reusable across runs)
        gen_water_hourly.csv    — per-generator per-hour
        fuel_water_hourly.csv   — per-fuel per-hour
        system_water_hourly.csv — system totals per-hour

    Parameters
    ----------
    water_rates_df : pre-loaded rate DataFrame (from a previous build_water_rates
                     call or pd.read_csv of rate_csv).  When supplied the rate
                     building / CSV-reading step is skipped entirely — useful when
                     calling run() in a loop over multiple simulations.

    Returns (gen_hourly, fuel_hourly, system_hourly).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rates — use pre-loaded DataFrame if available, else load from CSV / build
    if water_rates_df is not None and not water_rates_df.empty:
        print("[WATER] Reusing pre-loaded rate table")
        water_rates = water_rates_df
    elif rate_csv is not None and rate_csv.exists():
        print(f"[WATER] Reusing rate table from {rate_csv}")
        water_rates = pd.read_csv(rate_csv)
    else:
        gen_df      = pd.read_csv(_gen_csv_for(grid_dir))
        out_rate    = rate_csv or (out_dir / "gen_water_rates.csv")
        water_rates = build_water_rates(eia_path, gen_df, out_csv=out_rate)

    # Simulation outputs
    print(f"[WATER] Loading simulation outputs from {sim_dir}")
    thermal_df = pd.read_csv(sim_dir / "thermal_detail.csv")
    _renew_path = sim_dir / "renew_detail.csv"
    renew_df = (pd.read_csv(_renew_path)
                if _renew_path.stat().st_size > 0
                else pd.DataFrame(columns=["Date", "Hour", "Generator",
                                           "Output", "Curtailment"]))

    # Compute
    print("[WATER] Computing hourly water use …")
    gen_h, fuel_h, sys_h = compute_water_use(thermal_df, renew_df, water_rates)

    # Write
    gen_h.to_csv( out_dir / "gen_water_hourly.csv",    index=False)
    fuel_h.to_csv(out_dir / "fuel_water_hourly.csv",   index=False)
    sys_h.to_csv( out_dir / "system_water_hourly.csv", index=False)
    print(f"[WATER] Written → {out_dir}/{{gen,fuel,system}}_water_hourly.csv")

    # Summary
    if not sys_h.empty:
        total_wd = sys_h["total_wd_gal"].sum() / 1e6
        total_wc = sys_h["total_wc_gal"].sum() / 1e6
        n_hours  = sys_h["Hour"].nunique()
        print(f"\n── Water Use Summary ({n_hours} hours, {sim_dir.name}) ──")
        print(f"   Total withdrawal:   {total_wd:,.1f} Mgal")
        print(f"   Total consumption:  {total_wc:,.1f} Mgal")
        if not fuel_h.empty:
            print("\n   By fuel type (totals):")
            ft = (fuel_h.groupby("fuel")[["wd_gal", "wc_gal"]]
                  .sum()
                  .sort_values("wd_gal", ascending=False))
            ft["wd_Mgal"] = ft["wd_gal"] / 1e6
            ft["wc_Mgal"] = ft["wc_gal"] / 1e6
            print(ft[["wd_Mgal", "wc_Mgal"]].to_string(float_format=lambda v: f"{v:,.2f}"))

    return gen_h, fuel_h, sys_h


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--eia-file", type=Path,
        default=Path("inputs/Cooling_Boiler_Generator_Data_Texas_2024.csv"),
        metavar="FILE",
        help="EIA Form 860 Schedule 6 Part D CSV (default: %(default)s)",
    )
    p.add_argument(
        "--grid-dir", type=Path, default=Path("vatic/data/grids/RTS-GMLC"),
        help="Grid source directory (default: %(default)s). gen.csv is resolved via grid registry.",
    )


def main() -> None:
    top = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = top.add_subparsers(dest="cmd", required=True)

    # build-rates
    p_build = sub.add_parser("build-rates", help="Build gen_water_rates.csv from EIA data")
    _shared_args(p_build)
    p_build.add_argument(
        "--out", type=Path, default=Path("outputs/gen_water_rates.csv"),
        help="Output path for rate table (default: %(default)s)",
    )

    # compute
    p_comp = sub.add_parser("compute", help="Estimate hourly water use from dispatch")
    p_comp.add_argument("--sim-dir", type=Path, required=True,
                        help="Directory with thermal_detail.csv and renew_detail.csv")
    p_comp.add_argument("--rates",   type=Path, required=True,
                        help="gen_water_rates.csv from build-rates step")
    p_comp.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for water CSVs")

    # run (combined)
    p_run = sub.add_parser("run", help="Build rates and compute water use in one step")
    _shared_args(p_run)
    p_run.add_argument("--sim-dir", type=Path, required=True)
    p_run.add_argument("--out-dir", type=Path, required=True)
    p_run.add_argument(
        "--rates", type=Path, default=None,
        help="Reuse existing gen_water_rates.csv if present (skip rebuild)",
    )

    args = top.parse_args()

    if args.cmd == "build-rates":
        gen_df = pd.read_csv(_gen_csv_for(args.grid_dir))
        build_water_rates(args.eia_file, gen_df, out_csv=args.out)

    elif args.cmd == "compute":
        water_rates = pd.read_csv(args.rates)
        thermal_df  = pd.read_csv(args.sim_dir / "thermal_detail.csv")
        _rp = args.sim_dir / "renew_detail.csv"
        renew_df = (pd.read_csv(_rp)
                    if _rp.stat().st_size > 0
                    else pd.DataFrame(columns=["Date", "Hour", "Generator",
                                               "Output", "Curtailment"]))
        gen_h, fuel_h, sys_h = compute_water_use(thermal_df, renew_df, water_rates)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        gen_h.to_csv( args.out_dir / "gen_water_hourly.csv",    index=False)
        fuel_h.to_csv(args.out_dir / "fuel_water_hourly.csv",   index=False)
        sys_h.to_csv( args.out_dir / "system_water_hourly.csv", index=False)
        print(f"[WATER] Written → {args.out_dir}")

    elif args.cmd == "run":
        run(
            sim_dir  = args.sim_dir,
            grid_dir = args.grid_dir,
            eia_path = args.eia_file,
            out_dir  = args.out_dir,
            rate_csv = args.rates,
        )


if __name__ == "__main__":
    main()
