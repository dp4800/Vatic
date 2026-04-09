#!/usr/bin/env python3
# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""Texas-7k grid map: before and after data center load injection.

Thesis: Figure 5 — "Mapping of new data centers onto the Texas-7k grid system."

Side-by-side comparison showing transmission network, buses, generators,
and data center locations on the Texas-7k grid.

Usage:
    module load anaconda3/2024.10
    python scripts/tx_datacenter_map.py [--output PATH]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════

REPO     = Path(__file__).resolve().parents[1]
BASE_SRC = REPO / "vatic/data/grids/Texas-7k/TX_Data/SourceData"
DC_SRC   = REPO / "vatic/data/grids/Texas-7k-DC-REAL/TX_Data"
GEOJSON  = REPO / "vatic/data/ne_states_us.geojson"

# ═══════════════════════════════════════════════════════════════════════════════
# Styling
# ═══════════════════════════════════════════════════════════════════════════════

AREA_COLORS = {
    "Coast":         "#2196F3",
    "East":          "#4CAF50",
    "Far_West":      "#FF9800",
    "North":         "#9C27B0",
    "North_Central": "#F44336",
    "South":         "#00BCD4",
    "South_Central": "#795548",
    "West":          "#607D8B",
}

GEN_FUEL_COLORS = {
    "Nuclear":    "#7c3aed",   # purple
    "Coal":       "#1e293b",   # dark slate
    "Natural Gas":"#ea580c",   # deep orange
    "Wind":       "#16a34a",   # green
    "Solar":      "#eab308",   # gold
    "Hydro":      "#0ea5e9",   # sky blue
    "Other":      "#a1a1aa",   # zinc gray
}

FUEL_MAP = {
    "NUC (Nuclear)":                 "Nuclear",
    "SUB (Subbituminous Coal)":      "Coal",
    "LIG (Lignite Coal)":           "Coal",
    "NG (Natural Gas)":              "Natural Gas",
    "WND (Wind)":                    "Wind",
    "SUN (Solar)":                   "Solar",
    "WAT (Water)":                   "Hydro",
    "WDS (Wood/Wood Waste Solids)":  "Other",
    "PUR (Purchased Steam)":         "Other",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_bus():
    df = pd.read_csv(BASE_SRC / "bus.csv")
    return df[["Bus ID", "Bus Name", "lat", "lng", "Area", "MW Load", "BaseKV"]]


def load_branch():
    df = pd.read_csv(BASE_SRC / "branch.csv")
    return df[["From Bus", "To Bus", "Branch Device Type", "Cont Rating"]]


def load_gen():
    df = pd.read_csv(BASE_SRC / "gen.csv")
    return df[["Bus ID", "Fuel", "PMax MW"]].copy()


def load_dc_injections():
    """Return {bus_name: avg_mw} for DC injection buses."""
    da = pd.read_csv(
        DC_SRC / "timeseries_data_files/BusInjections/DAY_AHEAD_bus_injections.csv")
    bus_cols = [c for c in da.columns if c not in ("Year", "Month", "Day", "Period")]
    avg = da[bus_cols].mean()
    return avg.to_dict()


def load_texas_border():
    """Extract Texas state polygon from GeoJSON."""
    with open(GEOJSON) as f:
        geo = json.load(f)
    for feat in geo["features"]:
        if feat["properties"].get("name") == "Texas":
            geom = feat["geometry"]
            if geom["type"] == "Polygon":
                return [np.array(geom["coordinates"][0])]
            elif geom["type"] == "MultiPolygon":
                return [np.array(ring[0]) for ring in geom["coordinates"]]
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════════════════════════════════

def draw_texas(ax, polys):
    for poly in polys:
        ax.fill(poly[:, 0], poly[:, 1], color="#f5f5f0", alpha=0.6, zorder=0)
        ax.plot(poly[:, 0], poly[:, 1], color="#888888", linewidth=1.0, zorder=1)


LINE_VOLTAGE_STYLE = {
    345.0: {"color": "#9f1239", "linewidth": 1.0,  "alpha": 0.70, "zorder": 4, "label": "345 kV"},  # rose/maroon
    138.0: {"color": "#0d9488", "linewidth": 0.55, "alpha": 0.50, "zorder": 3, "label": "138 kV"},  # teal
     69.0: {"color": "#a8a29e", "linewidth": 0.35, "alpha": 0.45, "zorder": 2, "label": "69 kV"},   # warm gray
}


def draw_lines(ax, branch, bus_coords, bus_kv):
    """Draw transmission lines colored and sized by voltage level."""
    by_kv: dict[float, list] = {kv: [] for kv in LINE_VOLTAGE_STYLE}

    for _, row in branch.iterrows():
        fb, tb = int(row["From Bus"]), int(row["To Bus"])
        if fb in bus_coords and tb in bus_coords:
            kv = bus_kv.get(fb, 138.0)
            if kv not in by_kv:
                kv = 138.0
            by_kv[kv].append([bus_coords[fb], bus_coords[tb]])

    # Draw low voltage first, high voltage on top
    for kv in sorted(by_kv.keys()):
        segs = by_kv[kv]
        if not segs:
            continue
        style = LINE_VOLTAGE_STYLE[kv]
        lc = LineCollection(segs, colors=style["color"],
                            linewidths=style["linewidth"],
                            alpha=style["alpha"], zorder=style["zorder"])
        ax.add_collection(lc)


def draw_buses(ax, bus, alpha=0.25, size=0.4):
    """Draw all buses as small dots colored by area."""
    for area, color in AREA_COLORS.items():
        mask = bus["Area"] == area
        subset = bus[mask]
        ax.scatter(subset["lng"], subset["lat"],
                   s=size, c=color, alpha=alpha, zorder=3,
                   edgecolors="none", rasterized=True)


def draw_generators(ax, gen, bus):
    """Draw generators as circles sized by capacity, colored by fuel."""
    gen_merged = gen.merge(bus[["Bus ID", "lat", "lng"]], on="Bus ID", how="left")
    gen_merged["fuel_cat"] = gen_merged["Fuel"].map(FUEL_MAP).fillna("Other")

    # Aggregate by bus + fuel category
    agg = (gen_merged.groupby(["Bus ID", "lat", "lng", "fuel_cat"])
           .agg(total_mw=("PMax MW", "sum"))
           .reset_index())

    for fuel, color in GEN_FUEL_COLORS.items():
        mask = agg["fuel_cat"] == fuel
        subset = agg[mask]
        if subset.empty:
            continue
        sizes = np.clip(subset["total_mw"] / 12, 3, 80)
        ax.scatter(subset["lng"], subset["lat"],
                   s=sizes, c=color, alpha=0.6, zorder=5,
                   edgecolors="white", linewidths=0.2,
                   label=fuel)


def draw_dc_sites(ax, dc_mw, bus):
    """Draw data center sites as red diamonds sized by injection MW."""
    bus_name_to_coord = dict(zip(bus["Bus Name"], zip(bus["lng"], bus["lat"])))

    lngs, lats, mws = [], [], []
    for name, mw in dc_mw.items():
        if name in bus_name_to_coord:
            coord = bus_name_to_coord[name]
            lngs.append(coord[0])
            lats.append(coord[1])
            mws.append(mw)

    mws = np.array(mws)
    sizes = np.clip(mws / 5, 8, 150)

    ax.scatter(lngs, lats, s=sizes, c="#ec4899", marker="D",
               alpha=0.85, zorder=6, edgecolors="white", linewidths=0.4)


# ═══════════════════════════════════════════════════════════════════════════════
# Main figure
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_single(ax, polys, lines, bus, gen, bus_coords, bus_kv, lat_mid,
                 xlim, ylim, *, show_dc=False, dc_mw=None):
    """Draw one map panel."""
    draw_texas(ax, polys)
    draw_lines(ax, lines, bus_coords, bus_kv)
    draw_buses(ax, bus, alpha=0.35, size=0.6)
    draw_generators(ax, gen, bus)

    if show_dc and dc_mw is not None:
        draw_dc_sites(ax, dc_mw, bus)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(1.0 / np.cos(lat_mid))
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.5)


def _legend_handles(include_dc=False):
    """Build legend handles."""
    gen_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=7, label=f, markeredgecolor="white",
               markeredgewidth=0.3)
        for f, c in GEN_FUEL_COLORS.items()
    ]
    infra_handles = [
        Line2D([0], [0], color=s["color"],
               linewidth=max(s["linewidth"] * 3, 1.2),
               alpha=s["alpha"], label=s["label"])
        for s in LINE_VOLTAGE_STYLE.values()
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=4, alpha=0.5, label="Bus", markeredgecolor="none"),
    ]
    dc_handles = []
    if include_dc:
        dc_handles = [
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#ec4899",
                   markersize=8, label="Data Center",
                   markeredgecolor="white", markeredgewidth=0.3),
        ]
    return gen_handles + infra_handles + dc_handles


def plot_maps(out_dir: Path):
    bus    = load_bus()
    branch = load_branch()
    gen    = load_gen()
    dc_mw  = load_dc_injections()
    polys  = load_texas_border()

    lines = branch[branch["Branch Device Type"] == "Line"]
    bus_coords = dict(zip(bus["Bus ID"], zip(bus["lng"], bus["lat"])))
    bus_kv     = dict(zip(bus["Bus ID"], bus["BaseKV"]))

    all_lngs = np.concatenate([p[:, 0] for p in polys])
    all_lats = np.concatenate([p[:, 1] for p in polys])
    xlim = (all_lngs.min() - 0.3, all_lngs.max() + 0.3)
    ylim = (all_lats.min() - 0.3, all_lats.max() + 0.3)

    lat_mid = np.radians((ylim[0] + ylim[1]) / 2)
    dx = (xlim[1] - xlim[0]) * np.cos(lat_mid)
    dy = ylim[1] - ylim[0]
    fig_w = 11
    fig_h = fig_w * dy / dx

    common = dict(polys=polys, lines=lines, bus=bus, gen=gen,
                  bus_coords=bus_coords, bus_kv=bus_kv, lat_mid=lat_mid,
                  xlim=xlim, ylim=ylim)

    # ── Figure 1: Base grid (no data centers) ────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))
    _plot_single(ax1, **common, show_dc=False)
    ax1.set_title("Texas-7k Base Grid\n"
                  "6,717 buses  |  7,173 transmission lines  |  731 generators",
                  fontsize=13, fontweight="bold", pad=10)
    handles = _legend_handles(include_dc=False)
    fig1.legend(handles=handles, loc="lower center", ncol=6,
                fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01),
                handletextpad=0.5, columnspacing=1.0)
    fig1.tight_layout()
    p1 = out_dir / "tx_grid_base.png"
    fig1.savefig(p1, dpi=200, bbox_inches="tight")
    print(f"Saved: {p1}")
    plt.close(fig1)

    # ── Figure 2: Grid + data center load ────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    _plot_single(ax2, **common, show_dc=True, dc_mw=dc_mw)

    total_dc = sum(dc_mw.values())
    n_sites = len(dc_mw)
    ax2.set_title("Texas-7k + Data Center Load\n"
                  f"6,717 buses  |  7,173 lines  |  731 generators  |  "
                  f"{n_sites} DC sites ({total_dc:,.0f} MW avg.)",
                  fontsize=13, fontweight="bold", pad=10)
    ax2.annotate(
        f"{n_sites} data center buses\n"
        f"{total_dc:,.0f} MW avg. injection",
        xy=(0.98, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#ec4899", alpha=0.9),
        color="#333333")
    handles = _legend_handles(include_dc=True)
    fig2.legend(handles=handles, loc="lower center", ncol=6,
                fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01),
                handletextpad=0.5, columnspacing=1.0)
    fig2.tight_layout()
    p2 = out_dir / "tx_grid_datacenter.png"
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    print(f"Saved: {p2}")
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(
        description="Texas-7k grid map: before and after data center load")
    parser.add_argument("--output-dir",
                        default="/scratch/gpfs/SIRCAR/dp4800/vatic/outputs/"
                                "TX_2018_ANNUAL",
                        help="Output directory")
    args = parser.parse_args()
    plot_maps(Path(args.output_dir))


if __name__ == "__main__":
    main()
