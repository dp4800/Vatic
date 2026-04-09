"""
Plot average 24-hour demand load profile for each CAS mode vs no-CAS baseline.
3 subplots side-by-side (one per CAS mode), averaged over all available VATIC runs.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO     = Path(__file__).resolve().parents[2]
ANN_DIR  = REPO / "outputs" / "TX_2018_ANNUAL"
OUT_FILE = REPO / "outputs" / "cas_demand_profile.png"

MODES = {
    "sim-gm":  "Grid-Mix CAS",
    "sim-247": "24/7 CAS",
    "sim-lp":  "LP CAS (α=0.5)",
}
BASELINE = "baseline"

# ── Style (match previous CAS plots) ──────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "nocas": "#555555",   # dark grey — baseline (no CAS)
    "cas":   "#2196F3",   # blue — CAS shifted
}
MODE_COLORS = {
    "sim-gm":  "#FF9800",  # orange
    "sim-247": "#9C27B0",  # purple
    "sim-lp":  "#2196F3",  # blue
}

# ── Load data ─────────────────────────────────────────────────────────────────
def load_hourly_profile(mode: str) -> pd.Series:
    """Return average demand (MW) by hour-of-day across all available runs."""
    frames = []
    for path in sorted(ANN_DIR.glob(f"*/{mode}/hourly_summary.csv")):
        try:
            df = pd.read_csv(path, usecols=["Hour", "Demand"])
            if df.empty or df["Demand"].isna().all():
                continue
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise RuntimeError(f"No data found for mode '{mode}'")
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby("Hour")["Demand"].mean()


print("Loading demand profiles...")
baseline_profile = load_hourly_profile(BASELINE)
mode_profiles    = {m: load_hourly_profile(m) for m in MODES}

hours = np.arange(24)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
fig.suptitle("Average 24-Hour Demand Load Profile by CAS Mode", fontsize=13, fontweight="bold", y=1.01)

for ax, (mode_key, mode_label) in zip(axes, MODES.items()):
    profile  = mode_profiles[mode_key]
    color    = MODE_COLORS[mode_key]

    ax.plot(hours, baseline_profile.reindex(hours).values / 1e3,
            color=COLORS["nocas"], linewidth=2.0, linestyle="--",
            label="No CAS (baseline)", zorder=3)

    ax.plot(hours, profile.reindex(hours).values / 1e3,
            color=color, linewidth=2.5, linestyle="-",
            label=mode_label, zorder=4)

    # Shade the difference
    ax.fill_between(
        hours,
        baseline_profile.reindex(hours).values / 1e3,
        profile.reindex(hours).values / 1e3,
        where=(profile.reindex(hours).values >= baseline_profile.reindex(hours).values),
        alpha=0.15, color=color, label="_nolegend_",
    )
    ax.fill_between(
        hours,
        baseline_profile.reindex(hours).values / 1e3,
        profile.reindex(hours).values / 1e3,
        where=(profile.reindex(hours).values < baseline_profile.reindex(hours).values),
        alpha=0.15, color=color, label="_nolegend_",
    )

    ax.set_title(mode_label, fontsize=12, fontweight="bold", color=color)
    ax.set_xlabel("Hour of Day")
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax.set_xticklabels(["12am", "4am", "8am", "12pm", "4pm", "8pm", "11pm"], fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    ax.set_xlim(0, 23)

axes[0].set_ylabel("Average System Demand (GW)")

# ── Annotation: total runs used ────────────────────────────────────────────────
n_runs = sum(1 for _ in ANN_DIR.glob(f"*/{BASELINE}/hourly_summary.csv"))
fig.text(0.5, -0.03, f"Averaged over {n_runs} weekly VATIC runs (TX_2018_ANNUAL)",
         ha="center", fontsize=9, color="#666666")

plt.tight_layout()
fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
