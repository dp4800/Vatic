#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""
net_carbon_score.py — Environmental Score per microgrid.

Implements Ackon (2025) sections 3.4–3.5 (Net Carbon Score via Shapley
allocation) and extends it with a water dimension, producing a unified
Environmental Score for each K-means microgrid.

Pipeline
--------
  1. K-means partition of buses into K microgrids using lat/lon.
  2. Aggregate G[k,t], D[k,t], E[k,t] (generation, demand, CO2) per microgrid.
  3. Compute water withdrawal W[k,t] and consumption C[k,t] if water_rates
     are supplied (output of water_use.build_water_rates).
  4. Net Carbon Score (NCS) per microgrid via exact Shapley over 2^K coalitions:
       x[k,t]    = G[k,t] - D[k,t]               (net export, MW)
       e[k,t]    = E[k,t] / G[k,t]               (emission intensity, tCO2/MWh)
       h^B[k,t]  = e[k,t] / max_{j∈S} e[j,t]    (brownness, coalition-local)
       NCS_k(S,t) = x*(1-h^B) if exporting, x*h^B if importing
       φ_k(t)   = Shapley allocation from v(S,t) = Σ_{k∈S} NCS_k(S,t)
  5. Water intensity per microgrid:
       wd_intensity[k,t] = W[k,t] / G[k,t]       (gal/MWh, withdrawal)
       wc_intensity[k,t] = C[k,t] / G[k,t]       (gal/MWh, consumption)
  6. Environmental Score (ES):
       Normalise both carbon Shapley (higher=better) and water withdrawal
       intensity (lower=better) to [0,1] across microgrids, then combine:
         ES_k = carbon_weight * norm_φ_k
              + water_weight  * (1 - norm_wd_k)
       where norm uses the inter-microgrid range for each metric.
  7. Bootstrap S=500 resamples of hours for 95% CI on φ_k and ES_k.

Usage
-----
    python scripts/net_carbon_score.py \\
        --sim-dir   outputs/2020-01-13/baseline \\
        --grid-dir  vatic/data/grids/RTS-GMLC \\
        --rates     outputs/gen_water_rates.csv \\
        --K 16 --scenarios 500 \\
        --out       outputs/2020-01-13/environmental_score/baseline.csv
"""

import argparse
import warnings
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Emission factor fallback (metric tons CO2/MWh) by fuel keyword.
# Used when gen.csv heat rate or CO2 column is non-numeric / missing
# (e.g. Texas-7k which lacks HR_avg_0 / Emissions CO2 Lbs/MMBTU columns).
#
# Sources:
#   Solid fuels / oil: EPA eGRID 2023 Texas median lbs CO2/MWh (PLNT23 sheet,
#     PLCO2RTA column, filtered PSTATABB=='TX') ÷ 2204.62 lbs/tonne.
#     File: inputs/egrid2023_data_rev2.xlsx
#   Natural gas, petroleum coke: EIA CO2 emission coefficients table
#     (lbs CO2/MMBtu) × typical heat rate ÷ 2204.62.
#     File: inputs/co2_vol_mass.xlsx  (sheet CO2_factors_2022)
#   Biomass: eGRID 2023 TX BIOMASS median (combustion CO2 only; note US
#     policy often treats biogenic CO2 as carbon-neutral).
# ---------------------------------------------------------------------------
_EIA_FACTORS = {
    # Solid fuels
    "coal":           1.0785,  # eGRID 2023 TX COAL median
    "lignite":        1.0785,  # eGRID 2023 TX COAL median (lignite ≈ coal range)
    "subbituminous":  1.0785,  # eGRID 2023 TX COAL median
    "petroleum coke": 1.0212,  # EIA: 225.13 lbs/MMBtu × 10 MMBTU/MWh / 2204.62
    "pet coke":       1.0212,
    # Oil
    "oil":            0.7958,  # eGRID 2023 TX OIL median
    # Natural gas (fleet average; can't distinguish CC vs CT from fuel name)
    "natural gas":    0.4963,  # eGRID 2023 US GAS median
    "ng":             0.4963,
    "gas":            0.4963,
    # Biomass / waste
    "biomass":        0.0541,  # eGRID 2023 TX BIOMASS median
    "wood":           0.0541,
    # Zero-emission sources
    "nuclear":        0.0,
    "wind":           0.0,
    "solar":          0.0,
    "pv":             0.0,
    "rtpv":           0.0,
    "hydro":          0.0,
    "water":          0.0,
    "csp":            0.0,
    "storage":        0.0,
    "mwh":            0.0,
    "geothermal":     0.0,
}

_LBS_PER_TON   = 2000.0
_BTU_PER_MMBTU = 1_000_000.0

# Hydro unit types that appear in renew_detail.csv and carry water use
_HYDRO_UTYPES = {"HYDRO", "HY", "ROR", "PS"}


# ---------------------------------------------------------------------------
# Grid data helpers
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


def _load_grid_data(grid_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = grid_dir / _data_dir_for(grid_dir) / "SourceData"
    return pd.read_csv(src / "bus.csv"), pd.read_csv(src / "gen.csv")


def _emission_rates(gen_df: pd.DataFrame) -> dict[str, float]:
    """Return {GEN_UID: tons_CO2_per_MWh} for every generator.

    Uses iterrows() so that column names with spaces (e.g. "GEN UID",
    "Emissions CO2 Lbs/MMBTU") are preserved without pandas renaming them.
    """
    rates: dict[str, float] = {}
    for _, row in gen_df.iterrows():
        uid  = str(row.get("GEN UID", "")).strip()
        fuel = str(row.get("Fuel", "")).lower().strip()
        computed = False
        try:
            hr  = float(row.get("HR_avg_0", 0))                    # BTU / kWh
            co2 = float(row.get("Emissions CO2 Lbs/MMBTU", 0))    # lbs/MMBTU
            if hr > 0 and co2 > 0:
                # lbs/MMBTU * BTU/kWh * 1000 kWh/MWh / 1e6 BTU/MMBTU / 2000 lbs/ton
                rates[uid] = co2 * hr * 1000.0 / _BTU_PER_MMBTU / _LBS_PER_TON
                computed = True
        except (ValueError, TypeError):
            pass
        if not computed:
            rates[uid] = next(
                (v for k, v in _EIA_FACTORS.items() if k in fuel), 0.0
            )
    return rates


def _gen_to_bus(gen_df: pd.DataFrame, bus_df: pd.DataFrame) -> dict[str, str]:
    """Return {GEN_UID: bus_name}.

    Uses iterrows() so that column names with spaces are preserved.
    """
    id_to_name = dict(zip(bus_df["Bus ID"], bus_df["Bus Name"]))
    result = {}
    for _, row in gen_df.iterrows():
        uid    = str(row.get("GEN UID", "")).strip()
        bus_id = row.get("Bus ID")
        result[uid] = id_to_name.get(bus_id, "")
    return result


def _gen_unit_types(gen_df: pd.DataFrame) -> dict[str, str]:
    """Return {GEN_UID: unit_type} for hydro filtering."""
    result = {}
    for _, row in gen_df.iterrows():
        uid   = str(row.get("GEN UID", "")).strip()
        utype = str(row.get("Unit Type", "")).strip().upper()
        result[uid] = utype
    return result


# ---------------------------------------------------------------------------
# K-means partitioning
# ---------------------------------------------------------------------------

def _kmeans_partition(bus_df: pd.DataFrame, K: int, seed: int = 42) -> dict[str, int]:
    """Return {bus_name: cluster_id} from K-means on (lat, lng)."""
    coords = bus_df[["lat", "lng"]].values
    labels = KMeans(n_clusters=K, random_state=seed, n_init=20).fit_predict(coords)
    return dict(zip(bus_df["Bus Name"].tolist(), labels.tolist()))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _build_arrays(
    thermal_df:    pd.DataFrame,
    renew_df:      pd.DataFrame,
    bus_out_df:    pd.DataFrame,
    gen2bus:       dict[str, str],
    gen2utype:     dict[str, str],
    bus2mg:        dict[str, int],
    em_rates:      dict[str, float],
    water_rates:   pd.DataFrame | None,
    K:             int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple]]:
    """
    Build per-microgrid per-hour arrays.

    Returns
    -------
    G  : (K, T) generation MW
    D  : (K, T) demand MW
    E  : (K, T) CO2 emissions (metric tons)
    W  : (K, T) water withdrawal (gallons)
    C  : (K, T) water consumption (gallons)
    time_list : sorted list of (Date, Hour) tuples
    """
    times     = (thermal_df[["Date", "Hour"]]
                 .drop_duplicates()
                 .sort_values(["Date", "Hour"]))
    time_list = list(zip(times["Date"], times["Hour"]))
    t_map     = {key: idx for idx, key in enumerate(time_list)}
    T         = len(time_list)

    G = np.zeros((K, T))
    D = np.zeros((K, T))
    E = np.zeros((K, T))
    W = np.zeros((K, T))
    C = np.zeros((K, T))

    # Water rate lookup: {gen_uid: (wd_gal_mwh, wc_gal_mwh)}
    wr_map: dict[str, tuple[float, float]] = {}
    if water_rates is not None and not water_rates.empty:
        wr_map = {
            row["gen_uid"]: (row["withdrawal_gal_mwh"], row["consumption_gal_mwh"])
            for _, row in water_rates.iterrows()
        }

    # Thermal dispatch
    for rec in thermal_df.itertuples(index=False):
        t = t_map.get((rec.Date, rec.Hour))
        if t is None:
            continue
        gen = rec.Generator
        bus = gen2bus.get(gen, "")
        k   = bus2mg.get(bus)
        if k is None:
            continue
        mw      = float(rec.Dispatch)
        G[k, t] += mw
        E[k, t] += mw * em_rates.get(gen, 0.0)
        wd_r, wc_r = wr_map.get(gen, (0.0, 0.0))
        W[k, t] += mw * wd_r
        C[k, t] += mw * wc_r

    # Renewable dispatch (zero CO2; hydro may have water use)
    hydro_uids = {uid for uid, ut in gen2utype.items() if ut in _HYDRO_UTYPES}
    for rec in renew_df.itertuples(index=False):
        t = t_map.get((rec.Date, rec.Hour))
        if t is None:
            continue
        gen = rec.Generator
        bus = gen2bus.get(gen, "")
        k   = bus2mg.get(bus)
        if k is None:
            continue
        mw      = float(rec.Output)
        G[k, t] += mw
        if gen in hydro_uids:
            wd_r, wc_r = wr_map.get(gen, (0.0, 0.0))
            W[k, t] += mw * wd_r
            C[k, t] += mw * wc_r

    # Bus demand
    for rec in bus_out_df.itertuples(index=False):
        t = t_map.get((rec.Date, rec.Hour))
        if t is None:
            continue
        k = bus2mg.get(rec.Bus)
        if k is None:
            continue
        D[k, t] += float(rec.Demand)

    return G, D, E, W, C, time_list


# ---------------------------------------------------------------------------
# Shapley / NCS core
# ---------------------------------------------------------------------------

def _shapley_exact(
    G: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact Shapley decomposition using all 2^K coalition evaluations.

    Implements Ackon (2025) §3.4–3.5:
      x[k,t]    = G - D                            (net export)
      e[k,t]    = E / G                            (emission intensity, 0 when G=0)
      h^B[k,t]  = e[k] / max_{j∈S} e[j]           (brownness, within coalition S)
      NCS_k(S,t) = x*(1-h^B) if exporting, x*h^B if importing
      v(S,t)    = Σ_{k∈S} NCS_k(S,t)
      φ_k(t)   = Σ_{S∋k} [(|S|-1)!(K-|S|)!/K!] [v(S,t) - v(S/{k},t)]

    Returns
    -------
    phi : (K, T)  Shapley value per microgrid per hour
    ncs : (K, T)  Standalone NCS (grand-coalition brownness)
    """
    T = G.shape[1]
    x = G - D
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        e = np.where(G > 0, E / G, 0.0)

    # ── Precompute coalition metadata (one-time) ──────────────────────────────
    n_coal     = 1 << K
    masks      = np.arange(n_coal, dtype=np.int32)
    bits       = np.arange(K,      dtype=np.int32)
    membership = ((masks[:, None] >> bits[None, :]) & 1).astype(bool)  # (2^K, K)
    coal_size  = membership.sum(axis=1)

    K_fact    = factorial(K)
    shapley_w = np.array([
        factorial(int(s) - 1) * factorial(K - int(s)) / K_fact if s > 0 else 0.0
        for s in coal_size
    ])

    k_bits     = (1 << bits).astype(np.int32)
    contains_k = (masks[:, None] & k_bits[None, :]).astype(bool)  # (2^K, K)
    s_minus_k  = np.where(
        contains_k, masks[:, None] ^ k_bits[None, :], 0
    ).astype(np.int32)

    # ── Per-hour Shapley ──────────────────────────────────────────────────────
    phi = np.zeros((K, T))
    for t in range(T):
        e_t = e[:, t]
        x_t = x[:, t]

        e_in_coal = np.where(membership, e_t[None, :], -np.inf)  # (2^K, K)
        max_e     = np.max(e_in_coal, axis=1, keepdims=True)      # (2^K, 1)

        h_brown = np.where(
            membership,
            e_t[None, :] / np.where(max_e > 0, max_e, 1.0),
            0.0,
        )

        ncs_coal = np.where(
            x_t[None, :] > 0,
            x_t[None, :] * (1.0 - h_brown),
            x_t[None, :] * h_brown,
        )
        ncs_coal = np.where(membership, ncs_coal, 0.0)

        v     = ncs_coal.sum(axis=1)
        v_smk = v[s_minus_k]
        phi[:, t] = np.einsum(
            "m,mk,mk->k",
            shapley_w,
            contains_k.astype(np.float64),
            v[:, None] - v_smk,
        )

    # Standalone NCS (grand-coalition normalisation)
    max_e_all = e.max(axis=0, keepdims=True)
    h_all     = np.where(max_e_all > 0, e / max_e_all, 0.0)
    ncs       = np.where(x > 0, x * (1.0 - h_all), x * h_all)

    return phi, ncs


def _bootstrap_ci(
    arr:  np.ndarray,
    S:    int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap mean of arr[K, T] over S resamples of hours.
    Returns (mean, ci_low_2.5, ci_high_97.5) — each shape (K,).
    """
    K, T = arr.shape
    rng   = np.random.default_rng(seed)
    boots = np.stack(
        [arr[:, rng.integers(0, T, size=T)].mean(axis=1) for _ in range(S)]
    )
    return arr.mean(axis=1), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


# ---------------------------------------------------------------------------
# Environmental Score combination
# ---------------------------------------------------------------------------

def _environmental_score(
    phi:              np.ndarray,   # (K,) mean Shapley NCS — higher = better
    wd_intensity:     np.ndarray,   # (K,) mean water withdrawal gal/MWh — lower = better
    carbon_weight:    float = 0.5,
    water_weight:     float = 0.5,
) -> np.ndarray:
    """
    Combine carbon (NCS) and water dimensions into a single Environmental Score.

    Each metric is min-max normalised across microgrids to [0, 1]:
      - carbon:  higher Shapley NCS → higher norm score (good)
      - water:   lower withdrawal intensity → higher norm score (good)

    ES_k = carbon_weight * norm_carbon_k + water_weight * norm_water_k

    When all values are equal (zero range), the component contributes 0.5.

    Returns ES : (K,) in [0, 1]
    """
    def _norm(v: np.ndarray, higher_is_better: bool) -> np.ndarray:
        lo, hi = v.min(), v.max()
        if hi == lo:
            return np.full_like(v, 0.5, dtype=float)
        normed = (v - lo) / (hi - lo)
        return normed if higher_is_better else 1.0 - normed

    norm_c = _norm(phi,         higher_is_better=True)
    norm_w = _norm(wd_intensity, higher_is_better=False)

    return carbon_weight * norm_c + water_weight * norm_w


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    sim_dir:        Path,
    grid_dir:       Path,
    K:              int = 16,
    S:              int = 500,
    out_csv:        Path | None = None,
    out_hourly_csv: Path | None = None,
    water_rates:    pd.DataFrame | None = None,
    carbon_weight:  float = 0.5,
    water_weight:   float = 0.5,
    bus2mg:         dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Run the full Environmental Score pipeline.

    Parameters
    ----------
    sim_dir        : directory containing thermal_detail.csv, renew_detail.csv,
                     bus_detail.csv
    grid_dir       : grid root (RTS_Data/SourceData/bus.csv and gen.csv)
    K              : number of microgrids for K-means partition
    S              : bootstrap resamples for 95% CI
    out_csv        : write per-microgrid summary to this path if given
    out_hourly_csv : write hourly (microgrid, date, hour, phi, net_export_mw)
                     to this path if given — used by Policy Analysis Figure 1
    water_rates    : DataFrame from water_use.build_water_rates(); if None,
                     water dimension is skipped and only carbon score is computed
    carbon_weight  : weight for carbon NCS in combined Environmental Score
    water_weight   : weight for water intensity in combined Environmental Score
    bus2mg         : pre-computed {bus_name: cluster_id} from a prior call.
                     If supplied the K-means step is skipped entirely, so all
                     scenarios use an identical geographic partition (recommended).

    Returns
    -------
    DataFrame sorted by env_score descending, with columns:
        microgrid, mean_phi, phi_ci_lo, phi_ci_hi,
        mean_ncs, mean_wd_gal_mwh, mean_wc_gal_mwh,
        env_score, centroid_lat, centroid_lng, buses
    """
    has_water = water_rates is not None and not water_rates.empty

    print(f"[ENV] Loading grid  : {grid_dir}")
    bus_df, gen_df = _load_grid_data(grid_dir)

    if bus2mg is None:
        print(f"[ENV] K-means       : K={K}")
        bus2mg = _kmeans_partition(bus_df, K)
    else:
        print(f"[ENV] K-means       : K={K} (pre-computed partition reused)")
    em_rates  = _emission_rates(gen_df)
    gen2bus   = _gen_to_bus(gen_df, bus_df)
    gen2utype = _gen_unit_types(gen_df)

    print(f"[ENV] Loading sim   : {sim_dir}")
    thermal_df  = pd.read_csv(sim_dir / "thermal_detail.csv")
    renew_df    = pd.read_csv(sim_dir / "renew_detail.csv")
    bus_out_df  = pd.read_csv(sim_dir / "bus_detail.csv")

    print("[ENV] Aggregating arrays …")
    G, D, E, W, C, time_list = _build_arrays(
        thermal_df, renew_df, bus_out_df,
        gen2bus, gen2utype, bus2mg, em_rates, water_rates, K,
    )
    T = len(time_list)
    print(f"[ENV] T={T} hours  |  2^K={1 << K:,} coalitions")

    print("[ENV] Shapley (NCS, exact) …")
    phi, ncs = _shapley_exact(G, D, E, K)

    print(f"[ENV] Bootstrap CI  : S={S} resamples …")
    mean_phi, phi_lo, phi_hi = _bootstrap_ci(phi, S=S)

    # Water intensity per microgrid (gal/MWh, generation-weighted)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wd_intensity = np.where(G > 0, W / G, 0.0).mean(axis=1)  # (K,)
        wc_intensity = np.where(G > 0, C / G, 0.0).mean(axis=1)

    # Environmental Score
    if has_water:
        env_score = _environmental_score(
            mean_phi, wd_intensity, carbon_weight, water_weight
        )
        print(f"[ENV] Combined score: carbon={carbon_weight:.0%}  water={water_weight:.0%}")
    else:
        # Carbon-only: normalise NCS directly
        lo, hi = mean_phi.min(), mean_phi.max()
        env_score = ((mean_phi - lo) / (hi - lo)) if hi > lo else np.full(K, 0.5)
        print("[ENV] No water rates supplied — environmental score = normalised NCS only")

    # ── Result table ──────────────────────────────────────────────────────────
    merged = bus_df.copy()
    merged["microgrid"] = merged["Bus Name"].map(bus2mg)
    centroids = merged.groupby("microgrid")[["lat", "lng"]].mean()
    bus_lists = (
        merged.groupby("microgrid")["Bus Name"]
        .apply(lambda s: ",".join(sorted(s.tolist())))
    )

    result = pd.DataFrame({
        "microgrid":       np.arange(K),
        "mean_phi":        mean_phi,
        "phi_ci_lo":       phi_lo,
        "phi_ci_hi":       phi_hi,
        "mean_ncs":        ncs.mean(axis=1),
        "mean_wd_gal_mwh": wd_intensity,
        "mean_wc_gal_mwh": wc_intensity,
        "env_score":       env_score,
        "centroid_lat":    centroids["lat"].values,
        "centroid_lng":    centroids["lng"].values,
        "buses":           bus_lists.values,
    }).sort_values("env_score", ascending=False).reset_index(drop=True)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_csv, index=False)
        print(f"[ENV] Written → {out_csv}")

    if out_hourly_csv is not None:
        x_export = G - D  # (K, T) net export MW
        # Per-hour water intensity: gal/MWh generation-weighted
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            wd_intensity_h = np.where(G > 0, W / G, 0.0)  # (K, T)
            wc_intensity_h = np.where(G > 0, C / G, 0.0)  # (K, T)
        rows = []
        for t, (date_val, hour_val) in enumerate(time_list):
            for k in range(K):
                rows.append({
                    "microgrid":     k,
                    "date":          date_val,
                    "hour":          int(hour_val),
                    "phi":           float(phi[k, t]),
                    "net_export_mw": float(x_export[k, t]),
                    "wd_gal_mwh":    float(wd_intensity_h[k, t]),
                    "wc_gal_mwh":    float(wc_intensity_h[k, t]),
                })
        phi_df = pd.DataFrame(rows)
        out_hourly_csv.parent.mkdir(parents=True, exist_ok=True)
        phi_df.to_csv(out_hourly_csv, index=False)
        print(f"[ENV] Hourly phi → {out_hourly_csv}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sim-dir", type=Path, required=True,
        help="Directory containing thermal_detail.csv, renew_detail.csv, bus_detail.csv",
    )
    p.add_argument(
        "--grid-dir", type=Path, default=Path("vatic/data/grids/RTS-GMLC"),
        help="Grid source directory (default: %(default)s)",
    )
    p.add_argument(
        "--rates", type=Path, default=None,
        help="gen_water_rates.csv from water_use.py (enables water dimension)",
    )
    p.add_argument(
        "--K", type=int, default=16,
        help="Number of microgrids for K-means partition (default: %(default)s)",
    )
    p.add_argument(
        "--scenarios", type=int, default=500,
        help="Bootstrap resamples S for 95%% CI (default: %(default)s)",
    )
    p.add_argument(
        "--carbon-weight", type=float, default=0.5,
        help="Weight for carbon NCS in combined score (default: %(default)s)",
    )
    p.add_argument(
        "--water-weight", type=float, default=0.5,
        help="Weight for water intensity in combined score (default: %(default)s)",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path (default: <sim-dir>/environmental_score.csv)",
    )
    args = p.parse_args()

    water_rates = None
    if args.rates is not None:
        if args.rates.exists():
            water_rates = pd.read_csv(args.rates)
        else:
            print(f"[WARN] --rates file not found: {args.rates} — running carbon-only")

    out    = args.out or (args.sim_dir / "environmental_score.csv")
    result = run(
        sim_dir       = args.sim_dir,
        grid_dir      = args.grid_dir,
        K             = args.K,
        S             = args.scenarios,
        out_csv       = out,
        water_rates   = water_rates,
        carbon_weight = args.carbon_weight,
        water_weight  = args.water_weight,
    )

    print("\n── Environmental Score — per microgrid ──────────────────────────────────")
    print(result.to_string(
        columns=["microgrid", "mean_phi", "mean_wd_gal_mwh",
                 "mean_wc_gal_mwh", "env_score", "buses"],
        index=False,
        float_format=lambda v: f"{v:+.3f}" if abs(v) < 1e4 else f"{v:,.0f}",
    ))


if __name__ == "__main__":
    main()
