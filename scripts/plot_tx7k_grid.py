# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""Plot Texas-7k grid: buses and transmission lines over Texas state outline.

Thesis: Figure 4 — "Vatic's representation of 2018 ERCOT grid conditions."

Plots all 6,717 buses, 9,140 transmission lines, and 731 generators over
the Texas state outline, color-coded by fuel type.

Usage:
    module load anaconda3/2024.10
    python scripts/plot_tx7k_grid.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
REPO      = Path(__file__).resolve().parents[1]
SRC       = REPO / "vatic/data/grids/Texas-7k-DC/TX_Data/SourceData"
GEOJSON   = REPO / "vatic/data/ne_states_us.geojson"
OUT       = REPO / "outputs/tx7k_grid_map.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────
bus    = pd.read_csv(SRC / "bus.csv")
branch = pd.read_csv(SRC / "branch.csv")
gen    = pd.read_csv(SRC / "gen.csv")

bus_pos = dict(zip(bus["Bus ID"], zip(bus["lng"], bus["lat"])))

# ── Texas polygon from GeoJSON ───────────────────────────────────────────────
with open(GEOJSON) as f:
    geojson = json.load(f)

tx_feature = next(
    feat for feat in geojson["features"]
    if feat["properties"].get("postal") == "TX"
)

def extract_rings(geometry):
    """Yield coordinate rings from Polygon or MultiPolygon."""
    if geometry["type"] == "Polygon":
        yield from geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        for poly in geometry["coordinates"]:
            yield from poly

# ── Generator fuel types for colouring buses ────────────────────────────────
gen_bus_fuel = {}
fuel_map = {
    "NUC": "Nuclear", "Nuclear": "Nuclear",
    "WND": "Wind",    "Wind":    "Wind",
    "SUN": "Solar",   "Solar":   "Solar",
    "LIG": "Coal",    "SUB":     "Coal",    "Coal": "Coal",
    "WAT": "Hydro",   "Hydro":   "Hydro",
    "WDS": "Other",   "WH":      "Other",
}
for _, row in gen.iterrows():
    fuel_raw = str(row.get("Fuel", row.get("Unit Type", ""))).split()[0]
    fuel = fuel_map.get(fuel_raw, "Gas")   # default → natural gas
    bid  = int(row["Bus ID"]) if "Bus ID" in gen.columns else None
    if bid and bid not in gen_bus_fuel:
        gen_bus_fuel[bid] = fuel

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 11), dpi=150)
ax.set_facecolor("#f5f5f0")
fig.patch.set_facecolor("#ffffff")

# Texas outline
for ring in extract_rings(tx_feature["geometry"]):
    xs, ys = zip(*ring)
    ax.fill(xs, ys, color="#e8e8e0", zorder=0)
    ax.plot(xs, ys, color="#aaaaaa", lw=0.7, zorder=1)

# Transmission lines — draw as LineCollection for speed
line_segs = []
for _, row in branch.iterrows():
    fb, tb = row["From Bus"], row["To Bus"]
    if fb in bus_pos and tb in bus_pos:
        line_segs.append([bus_pos[fb], bus_pos[tb]])

lc = LineCollection(line_segs, linewidths=0.25, colors="#7090b0", alpha=0.45, zorder=2)
ax.add_collection(lc)

# Buses — colour by generator type present, else light grey
fuel_colors = {
    "Gas":     "#e07030",
    "Coal":    "#505050",
    "Nuclear": "#9b59b6",
    "Wind":    "#27ae60",
    "Solar":   "#f1c40f",
    "Hydro":   "#2980b9",
    "Other":   "#95a5a6",
}
load_only_color = "#cccccc"

# Separate buses into generator buses and load-only buses
gen_buses  = {bid: fuel for bid, fuel in gen_bus_fuel.items() if bid in bus_pos}
load_buses = [bid for bid in bus["Bus ID"] if bid in bus_pos and bid not in gen_buses]

# Plot load-only buses (small, grey)
if load_buses:
    lx = [bus_pos[b][0] for b in load_buses]
    ly = [bus_pos[b][1] for b in load_buses]
    ax.scatter(lx, ly, s=0.6, c=load_only_color, zorder=3, linewidths=0)

# Plot generator buses by fuel type
for fuel, color in fuel_colors.items():
    buses_f = [bid for bid, f in gen_buses.items() if f == fuel]
    if not buses_f:
        continue
    xs = [bus_pos[b][0] for b in buses_f]
    ys = [bus_pos[b][1] for b in buses_f]
    size = 18 if fuel == "Nuclear" else 10 if fuel in ("Coal", "Hydro") else 8
    ax.scatter(xs, ys, s=size, c=color, zorder=4, linewidths=0.3,
               edgecolors="white", label=fuel)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], color="#7090b0", lw=1.2, alpha=0.7, label=f"Transmission line (n={len(line_segs):,})"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=load_only_color,
           markersize=5, label=f"Load bus (n={len(load_buses):,})"),
]
for fuel, color in fuel_colors.items():
    count = sum(1 for f in gen_buses.values() if f == fuel)
    if count:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=6, label=f"{fuel} generator (n={count})")
        )

ax.legend(handles=legend_elements, loc="lower left", fontsize=7.5,
          framealpha=0.9, edgecolor="#cccccc", frameon=True)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlim(-107.5, -93.0)
ax.set_ylim(25.5, 36.8)
ax.set_xlabel("Longitude", fontsize=9)
ax.set_ylabel("Latitude",  fontsize=9)
ax.set_title("Texas-7k Power Grid\n"
             f"{len(bus):,} buses · {len(branch):,} transmission lines · {len(gen):,} generators",
             fontsize=12, fontweight="bold", pad=10)
ax.tick_params(labelsize=8)
ax.grid(True, linestyle="--", linewidth=0.3, color="#cccccc", alpha=0.5)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved → {OUT}")
