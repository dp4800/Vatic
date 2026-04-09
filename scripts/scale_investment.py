#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
scale_investment.py — Scale existing wind/solar generators and add batteries for Texas-7k.

Unlike renew_invest._apply_investment_t7k which ADDS new generator rows,
this module MULTIPLIES the PMax of all existing wind and solar generators
by scale factors and proportionally scales their dispatch timeseries.
Battery capacity is added as new rows distributed across the top-N
highest-renewable buses, using an existing battery row as a template.

Key advantages over the "add" approach:
  - All existing generator siting/connectivity is preserved
  - Timeseries capacity factors remain consistent with location
  - Transmission expansion (thermal_rating_scale) covers scaled output
"""
from __future__ import annotations

import ast
import hashlib
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_SCALE_MARKER = ".scale_complete"   # written last; signals grid is fully built

# ── Paths ───────────────────────────────────────────────────────────────────────
REPO      = Path(__file__).resolve().parents[1]
GRIDS_DIR = REPO / "vatic" / "data" / "grids"
INIT_DIR  = GRIDS_DIR / "initial-state"

# ── Constants ───────────────────────────────────────────────────────────────────
BATTERY_N_BUSES = 20     # distribute new battery across top-N renewable buses
_T7K_DATA_DIR   = "TX_Data"


# ── ScaleVector ─────────────────────────────────────────────────────────────────

@dataclass
class ScaleVector:
    """Describes a renewable investment via scale factors on existing infrastructure."""
    wind_scale:  float   # multiply PMax of ALL existing wind generators by this
    solar_scale: float   # multiply PMax of ALL existing solar generators by this
    battery_mw:  float   # total new battery MW to add (0 = none)

    def vector_hash(self) -> str:
        """8-char SHA-256 of the rounded parameter vector."""
        key = (
            f"W{round(self.wind_scale,  4):.4f}"
            f"_S{round(self.solar_scale, 4):.4f}"
            f"_B{round(self.battery_mw,  0):.0f}"
        )
        return hashlib.sha256(key.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return {
            "wind_scale":  self.wind_scale,
            "solar_scale": self.solar_scale,
            "battery_mw":  self.battery_mw,
        }

    def added_mw(self, baseline_wind_mw: float, baseline_solar_mw: float) -> dict:
        """Added capacity relative to the unmodified baseline — used for CAPEX."""
        return {
            "wind_mw":    max(0.0, (self.wind_scale  - 1.0) * baseline_wind_mw),
            "solar_mw":   max(0.0, (self.solar_scale - 1.0) * baseline_solar_mw),
            "battery_mw": self.battery_mw,
        }


# ── Cost-curve helpers ───────────────────────────────────────────────────────────

def _scale_row_cost_curve(gen_df: pd.DataFrame, idx: int, k: float) -> None:
    """
    Scale the piecewise-linear cost curve of row idx by factor k (in-place).

    The MW breakpoints scale proportionally; $/MWh prices are unchanged.
    TCC_x (MW array) and TCC_y (cumulative $$ array) are rebuilt.
    """
    row = gen_df.iloc[idx]
    pmax_orig = float(row["PMax MW"])
    if pmax_orig <= 0:
        return

    # Scale MW Break 1..5 (these define piecewise segments; Break 1 = PMin)
    for i in range(1, 6):
        col = f"MW Break {i}"
        if col in gen_df.columns and pd.notna(row[col]):
            gen_df.at[idx, col] = round(float(row[col]) * k, 4)

    # Scale PMax and Part. Factor
    gen_df.at[idx, "PMax MW"]      = round(pmax_orig * k, 4)
    gen_df.at[idx, "Part. Factor"] = round(float(row.get("Part. Factor", pmax_orig)) * k, 4)

    # Scale PMin
    if "PMin MW" in gen_df.columns and pd.notna(row.get("PMin MW")):
        gen_df.at[idx, "PMin MW"] = round(float(row["PMin MW"]) * k, 4)

    # Rebuild TCC_x and TCC_y using new MW breakpoints
    try:
        tcc_x_orig = ast.literal_eval(str(row["TCC_x"]))
        if not isinstance(tcc_x_orig, list) or len(tcc_x_orig) < 2:
            return
        tcc_x_new = [v * k for v in tcc_x_orig]
        gen_df.at[idx, "TCC_x"] = str(tcc_x_new)

        # Prices ($/MWh) — unchanged
        num_segs = len(tcc_x_orig) - 1
        prices = []
        for i in range(1, num_segs + 1):
            col = f"MWh Price {i}"
            if col in gen_df.columns and pd.notna(row.get(col)):
                prices.append(float(row[col]))

        if len(prices) == num_segs:
            fixed = float(row.get("Fixed Cost($/hr)", 0.0)) if pd.notna(
                row.get("Fixed Cost($/hr)")) else 0.0
            tcc_y = [fixed]
            for i in range(num_segs):
                tcc_y.append(tcc_y[-1] + prices[i] * (tcc_x_new[i + 1] - tcc_x_new[i]))
            gen_df.at[idx, "TCC_y"] = str(tcc_y)
    except Exception:
        pass  # leave TCC as-is if parsing fails; simulation uses MW Break cols


def _scale_timeseries(ts_dir: Path, scale: float, glob: str) -> None:
    """Multiply all data columns (non-time) in CSV files matching glob by scale."""
    time_cols = {"Year", "Month", "Day", "Period"}
    for ts_path in ts_dir.glob(glob):
        df = pd.read_csv(ts_path)
        data_cols = [c for c in df.columns if c not in time_cols]
        df[data_cols] = df[data_cols] * scale
        df.to_csv(ts_path, index=False)


# ── Main grid-building function ─────────────────────────────────────────────────

def _apply_investment_scale(
    source_grid: str,
    sv: ScaleVector,
    output_grid: str,
) -> None:
    """
    Build a scaled copy of source_grid with the ScaleVector sv applied.

    Actions:
      1. All wind generator PMax × sv.wind_scale (plus timeseries and cost curves)
      2. All solar generator PMax × sv.solar_scale (plus timeseries and cost curves)
      3. sv.battery_mw of new battery capacity distributed across the top
         BATTERY_N_BUSES buses by combined post-scale wind+solar PMax

    Idempotent: returns immediately if output_grid already exists (cache hit).
    Thread-safe: if another process creates the grid concurrently, uses that copy.
    """
    dst_dir = GRIDS_DIR / output_grid
    marker  = dst_dir / _SCALE_MARKER

    if marker.exists():
        return  # fully built — fast cache hit

    if dst_dir.exists():
        # Dir was created by another concurrent process but may not be finished.
        # Wait up to 60 s for the marker to appear.
        for _ in range(120):
            if marker.exists():
                return
            time.sleep(0.5)
        return  # marker still absent after 60 s — assume complete

    src_dir = GRIDS_DIR / source_grid
    if not src_dir.exists():
        raise FileNotFoundError(f"Source grid not found: {src_dir}")

    # ── Copy grid (handle last-instant race on copytree) ─────────────────────
    try:
        shutil.copytree(src_dir, dst_dir)
    except FileExistsError:
        # Another process won the race — wait for its marker
        for _ in range(120):
            if marker.exists():
                return
            time.sleep(0.5)
        return

    # Copy initial-state if present (best-effort)
    src_init = INIT_DIR / source_grid
    dst_init = INIT_DIR / output_grid
    if src_init.is_dir():
        try:
            if dst_init.exists():
                shutil.rmtree(dst_init)
            shutil.copytree(src_init, dst_init)
        except Exception as e:
            print(f"  [scale] Warning: could not copy initial-state: {e}")

    # ── Paths within the new grid copy ────────────────────────────────────────
    dst_data  = dst_dir / _T7K_DATA_DIR
    gen_csv   = dst_data / "SourceData" / "gen.csv"
    wind_dir  = dst_data / "timeseries_data_files" / "WIND"
    solar_dir = dst_data / "timeseries_data_files" / "PV"

    gen_df = pd.read_csv(gen_csv)

    # ── Scale wind ────────────────────────────────────────────────────────────
    if sv.wind_scale != 1.0:
        wind_mask = gen_df["Fuel"] == "WND (Wind)"
        for idx in gen_df[wind_mask].index:
            _scale_row_cost_curve(gen_df, idx, sv.wind_scale)
        _scale_timeseries(wind_dir, sv.wind_scale, "DAY_AHEAD_*.csv")
        _scale_timeseries(wind_dir, sv.wind_scale, "REAL_TIME_*.csv")
        wind_total = gen_df[wind_mask]["PMax MW"].sum()
        print(f"  [scale] Wind ×{sv.wind_scale:.4f} → {wind_total:.0f} MW total")

    # ── Scale solar ───────────────────────────────────────────────────────────
    if sv.solar_scale != 1.0:
        solar_mask = gen_df["Fuel"] == "SUN (Solar)"
        for idx in gen_df[solar_mask].index:
            _scale_row_cost_curve(gen_df, idx, sv.solar_scale)
        _scale_timeseries(solar_dir, sv.solar_scale, "DAY_AHEAD_*.csv")
        _scale_timeseries(solar_dir, sv.solar_scale, "REAL_TIME_*.csv")
        solar_total = gen_df[solar_mask]["PMax MW"].sum()
        print(f"  [scale] Solar ×{sv.solar_scale:.4f} → {solar_total:.0f} MW total")

    # ── Add batteries ─────────────────────────────────────────────────────────
    if sv.battery_mw > 0:
        batt_template_rows = gen_df[gen_df["Unit Type"] == "Batteries"]
        if batt_template_rows.empty:
            print("  [scale] WARNING: No battery template found — skipping battery addition")
        else:
            template = batt_template_rows.iloc[0].copy()
            ref_pmax = max(float(template["PMax MW"]), 1.0)

            # Top BATTERY_N_BUSES buses by combined post-scale wind+solar PMax
            renew_mask = gen_df["Fuel"].isin(["WND (Wind)", "SUN (Solar)"])
            bus_pmax   = (
                gen_df[renew_mask]
                .groupby("Bus ID")["PMax MW"]
                .sum()
                .nlargest(BATTERY_N_BUSES)
            )
            chosen_buses = list(bus_pmax.index)
            batt_per_bus = sv.battery_mw / len(chosen_buses)
            batt_scale   = batt_per_bus / ref_pmax

            new_rows = []
            for i, bus_id in enumerate(chosen_buses, start=1):
                new_uid = f"NEWB_{int(bus_id)}_{i:03d}"
                nr = template.to_dict()
                nr["GEN UID"]          = new_uid
                nr["Bus ID"]           = int(bus_id)
                nr["Gen MW"]           = 0.0
                nr["PMax MW"]          = round(batt_per_bus, 4)
                nr["PMin MW"]          = 0.0
                nr["Part. Factor"]     = round(batt_per_bus, 4)
                nr["Ramp Rate MW/Min"] = round(batt_per_bus, 4)  # near-instant ramp
                nr["Fuel"]             = "Storage"  # triggers SOC model in loaders.py

                # Scale cost curve MW breakpoints
                for j in range(1, 6):
                    col = f"MW Break {j}"
                    if col in nr and pd.notna(nr.get(col)):
                        nr[col] = round(float(nr[col]) * batt_scale, 4)

                # Rebuild TCC_x, TCC_y for new battery size
                try:
                    tcc_x_orig = ast.literal_eval(str(template["TCC_x"]))
                    if isinstance(tcc_x_orig, list) and len(tcc_x_orig) >= 2:
                        tcc_x_new = [v * batt_scale for v in tcc_x_orig]
                        nr["TCC_x"] = str(tcc_x_new)
                        num_segs    = len(tcc_x_orig) - 1
                        prices = [
                            float(template[f"MWh Price {k+1}"])
                            for k in range(num_segs)
                            if pd.notna(template.get(f"MWh Price {k+1}"))
                        ]
                        if len(prices) == num_segs:
                            fixed = float(template.get("Fixed Cost($/hr)", 0.0))
                            tcc_y = [fixed]
                            for k in range(num_segs):
                                tcc_y.append(
                                    tcc_y[-1] + prices[k] * (tcc_x_new[k+1] - tcc_x_new[k])
                                )
                            nr["TCC_y"] = str(tcc_y)
                except Exception:
                    pass

                new_rows.append(nr)

            gen_df = pd.concat([gen_df, pd.DataFrame(new_rows)], ignore_index=True)
            batt_actual = gen_df[gen_df["Unit Type"] == "Batteries"]["PMax MW"].sum()
            print(f"  [scale] Battery: +{len(new_rows)} units × {batt_per_bus:.0f} MW "
                  f"= {sv.battery_mw:.0f} MW (total in grid: {batt_actual:.0f} MW)")

    gen_df.to_csv(gen_csv, index=False)

    wind_tot  = gen_df[gen_df["Fuel"] == "WND (Wind)"]["PMax MW"].sum()
    solar_tot = gen_df[gen_df["Fuel"] == "SUN (Solar)"]["PMax MW"].sum()
    batt_tot  = gen_df[gen_df["Unit Type"] == "Batteries"]["PMax MW"].sum()
    print(f"  [scale] Grid ready: {output_grid}")
    print(f"  [scale] Totals → Wind {wind_tot:.0f} MW | Solar {solar_tot:.0f} MW "
          f"| Battery {batt_tot:.0f} MW")

    # Mark grid as fully built so concurrent processes stop waiting
    marker.touch()
