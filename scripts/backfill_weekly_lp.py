#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""Backfill and harmonize weekly_summary CSVs.

1. For sweep weeks missing sim-lp data, compute all metrics from raw outputs.
2. For ALL weeks, recompute CO₂ and Carbon intensity using canonical emission
   factors (thesis Table AI) to ensure consistency across all scenarios.

Usage:
    module load anaconda3/2024.10
    python scripts/backfill_weekly_lp.py [--study-dir outputs/TX_2018_ANNUAL]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from constants import EMISSION_FACTORS, FUEL_CATEGORY, WATER_WITHDRAWAL

# Scenarios expected in weekly_summary columns
SCENARIOS = ['baseline', 'sim-gm', 'sim-247', 'sim-lp']


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_gen_ef_map(grid_dir: Path) -> dict[str, float]:
    """Build generator UID → CO₂ emission factor (tCO₂/MWh) from gen.csv."""
    for subdir in ['TX_Data/SourceData', 'RTS_Data/SourceData']:
        gen_path = grid_dir / subdir / 'gen.csv'
        if gen_path.exists():
            break
    else:
        raise FileNotFoundError(f"No gen.csv found under {grid_dir}")

    gen = pd.read_csv(gen_path)
    uid_to_ef = {}
    for _, row in gen.iterrows():
        uid = str(row['GEN UID'])
        fuel = str(row.get('Fuel', ''))
        category = FUEL_CATEGORY.get(fuel, 'Natural Gas')
        uid_to_ef[uid] = EMISSION_FACTORS.get(category, 0.0)
    return uid_to_ef


def compute_co2(sim_dir: Path, uid_to_ef: dict[str, float]) -> tuple[float, float]:
    """Compute total CO₂ (kt) and carbon intensity (kg/MWh) from thermal_detail.

    Returns (total_co2_kt, carbon_intensity_kg_mwh).
    """
    td_path = sim_dir / 'thermal_detail.csv'
    hs_path = sim_dir / 'hourly_summary.csv'
    if not td_path.exists() or not hs_path.exists():
        return np.nan, np.nan

    td = pd.read_csv(td_path)
    td['co2_t'] = td['Generator'].map(uid_to_ef).fillna(0.0) * td['Dispatch']
    total_co2_t = td['co2_t'].sum()

    hs = pd.read_csv(hs_path)
    demand = hs['Demand'].sum()
    ci = (total_co2_t * 1000) / demand if demand > 0 else np.nan

    return round(total_co2_t * 1e-3, 4), round(ci, 4)


def compute_all_metrics(sim_dir: Path, water_dir: Path,
                        uid_to_ef: dict[str, float]) -> dict[str, float]:
    """Compute all weekly metrics for a scenario from raw VATIC outputs."""
    hs = pd.read_csv(sim_dir / 'hourly_summary.csv')

    total_cost = (hs['FixedCosts'] + hs['VariableCosts']).sum()
    renew_curtailed = hs['RenewablesCurtailment'].sum()
    renew_used = hs['RenewablesUsed'].sum()
    load_shedding = hs['LoadShedding'].sum()
    lolp = (hs['LoadShedding'] > 0).mean()

    co2_kt, ci = compute_co2(sim_dir, uid_to_ef)

    water_path = water_dir / 'system_water_hourly.csv'
    total_wd = np.nan
    total_wc = np.nan
    if water_path.exists():
        water = pd.read_csv(water_path)
        total_wd = water['total_wd_gal'].sum()
        if 'total_wc_gal' in water.columns:
            total_wc = water['total_wc_gal'].sum()

    return {
        'Total CO2':            co2_kt,
        'Operational cost':     round(total_cost * 1e-6, 4),
        'Water withdrawal':     round(total_wd * 1e-9, 4),
        'Water consumption':    round(total_wc * 1e-9, 4),
        'Renewables curtailed': round(renew_curtailed * 1e-3, 4),
        'Renewables used':      round(renew_used * 1e-3, 4),
        'Load shedding':        round(load_shedding, 4),
        'Carbon intensity':     ci,
        'LOLP':                 round(lolp, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Backfill
# ═══════════════════════════════════════════════════════════════════════════════

def backfill(study: Path, grid_dir: Path):
    uid_to_ef = load_gen_ef_map(grid_dir)
    patched_lp = 0
    patched_co2 = 0

    co2_metrics = {'Total CO2', 'Carbon intensity'}

    for week_dir in sorted(study.iterdir()):
        if not week_dir.is_dir():
            continue
        csv_paths = list(week_dir.glob('weekly_summary_*.csv'))
        if not csv_paths:
            continue
        csv_path = csv_paths[0]
        df = pd.read_csv(csv_path)
        changed = False

        # ── Phase 1: backfill missing sim-lp metrics ────────────────────────
        needs_lp = False
        if 'sim-lp' not in df.columns:
            needs_lp = True
        else:
            main_metrics = {'Total CO2', 'Operational cost', 'Renewables curtailed',
                            'Renewables used', 'Load shedding', 'Carbon intensity', 'LOLP'}
            metric_rows = df[df['metric'].isin(main_metrics)]
            needs_lp = metric_rows['sim-lp'].isna().any()

        if needs_lp:
            sim_lp_dir = week_dir / 'sim-lp'
            if not (sim_lp_dir / 'hourly_summary.csv').exists():
                print(f"  {week_dir.name}: no sim-lp output — skipping LP backfill")
            else:
                water_dir = week_dir / 'water' / 'sim-lp'
                metrics = compute_all_metrics(sim_lp_dir, water_dir, uid_to_ef)

                if 'sim-lp' not in df.columns:
                    df['sim-lp'] = np.nan

                for metric_name, value in metrics.items():
                    df.loc[df['metric'] == metric_name, 'sim-lp'] = value

                changed = True
                patched_lp += 1
                print(f"  {week_dir.name}: backfilled sim-lp metrics")

        # ── Phase 2: recompute CO₂ and CI for all scenarios ─────────────────
        for scenario in SCENARIOS:
            if scenario not in df.columns:
                continue
            sim_dir = week_dir / scenario
            co2_kt, ci = compute_co2(sim_dir, uid_to_ef)
            if np.isnan(co2_kt):
                continue
            df.loc[df['metric'] == 'Total CO2', scenario] = co2_kt
            df.loc[df['metric'] == 'Carbon intensity', scenario] = ci
            changed = True

        # ── Phase 3: backfill water consumption for all scenarios ──────────
        if 'Water consumption' not in df['metric'].values:
            # Insert row after Water withdrawal
            wd_idx = df.index[df['metric'] == 'Water withdrawal']
            if len(wd_idx):
                wd_row = df.loc[wd_idx[0]].copy()
                wd_row['metric'] = 'Water consumption'
                wd_row['unit'] = 'Bgal'
                for col in SCENARIOS + ['baseline']:
                    wd_row[col] = np.nan
                df = pd.concat([
                    df.iloc[:wd_idx[0] + 1],
                    pd.DataFrame([wd_row]),
                    df.iloc[wd_idx[0] + 1:]
                ], ignore_index=True)
                changed = True

        for scenario in SCENARIOS + ['baseline']:
            if scenario not in df.columns:
                continue
            water_path = week_dir / 'water' / scenario / 'system_water_hourly.csv'
            if not water_path.exists():
                continue
            wc_row = df[df['metric'] == 'Water consumption']
            if wc_row.empty or pd.notna(wc_row[scenario].values[0]):
                continue
            water = pd.read_csv(water_path)
            if 'total_wc_gal' in water.columns:
                total_wc = round(water['total_wc_gal'].sum() * 1e-9, 4)
                df.loc[df['metric'] == 'Water consumption', scenario] = total_wc
                changed = True

        if changed:
            # Ensure column order: ..., sim-lp, winner
            cols = list(df.columns)
            if 'winner' in cols and 'sim-lp' in cols:
                cols.remove('sim-lp')
                winner_idx = cols.index('winner')
                cols.insert(winner_idx, 'sim-lp')
                df = df[cols]

            df.to_csv(csv_path, index=False, float_format='%.4f')
            patched_co2 += 1

    print(f"\nDone: backfilled sim-lp for {patched_lp} weeks, "
          f"recomputed CO₂/CI for {patched_co2} weeks")


def main():
    parser = argparse.ArgumentParser(
        description='Backfill sim-lp and harmonize CO₂ emission factors')
    parser.add_argument('--study-dir', default='outputs/TX_2018_ANNUAL')
    parser.add_argument('--grid-dir',
                        default='vatic/data/grids/Texas-7k',
                        help='Path to grid directory with gen.csv')
    args = parser.parse_args()

    backfill(Path(args.study_dir), Path(args.grid_dir))


if __name__ == '__main__':
    main()
