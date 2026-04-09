# Generated and debugged with the assistance of Claude Code (claude.ai/code)
"""Shared constants for thesis analysis scripts.

Centralises emission factors, fuel mappings, CAS mode styling, and week
definitions that were previously duplicated across 12+ plotting and analysis
scripts.  Imported by individual scripts; never run directly.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Emission factors — thesis Table AI (tCO₂/MWh)
# ═══════════════════════════════════════════════════════════════════════════════

EMISSION_FACTORS = {
    'Coal':           1.0785,
    'Petroleum Coke': 1.0212,
    'Oil':            0.7958,
    'Natural Gas':    0.4963,
    'Biopower':       0.054,
    'Nuclear':        0.0,
    'Wind':           0.0,
    'Solar':          0.0,
    'Hydro':          0.0,
    'Battery':        0.0,
}

# Texas-7k gen.csv Fuel string → thesis emission category
FUEL_CATEGORY = {
    'NUC (Nuclear)':                            'Nuclear',
    'SUB (Subbituminous Coal)':                 'Coal',
    'LIG (Lignite Coal)':                       'Coal',
    'NG (Natural Gas)':                         'Natural Gas',
    'OG (Other Gas)':                           'Natural Gas',
    'PUR (Purchased Steam)':                    'Natural Gas',
    'WND (Wind)':                               'Wind',
    'SUN (Solar)':                              'Solar',
    'WAT (Water)':                              'Hydro',
    'PC (Petroleum Coke)':                      'Petroleum Coke',
    'WDS (Wood/Wood Waste Solids)':             'Biopower',
    'AB (Agricultural By-Products)':            'Biopower',
    'WH (Waste Heat)':                          'Natural Gas',
    'OTH (Other)':                              'Natural Gas',
    'MWH (Electricity use for Energy Storage)': 'Battery',
}

# Derived: raw fuel string → kg CO₂/MWh (for direct thermal_detail lookup)
EMISSION_FACTORS_BY_FUEL = {
    fuel: EMISSION_FACTORS.get(cat, 0.0) * 1000
    for fuel, cat in FUEL_CATEGORY.items()
}

# ═══════════════════════════════════════════════════════════════════════════════
# Water withdrawal rates — thesis Table AII (gal/MWh)
# ═══════════════════════════════════════════════════════════════════════════════

WATER_WITHDRAWAL = {
    'Coal':           530,
    'Petroleum Coke': 530,
    'Oil':            530,
    'Natural Gas':    210,
    'Biopower':       480,
    'Nuclear':        720,
    'Wind':           1,
    'Solar':          81,
    'Hydro':          4491,
    'Battery':        0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# CAS mode styling
# ═══════════════════════════════════════════════════════════════════════════════

CAS_MODES = {
    "sim-gm":  {"label": "Grid-Mix CAS",              "color": "#FF8C00",
                "marker": "o", "linestyle": "-", "linewidth": 1.8},
    "sim-247": {"label": "24/7 CAS",                   "color": "#8B5CF6",
                "marker": "s", "linestyle": "-", "linewidth": 1.8},
    "sim-lp":  {"label": r"LP CAS ($\alpha$=0.5)",     "color": "#3B82F6",
                "marker": "^", "linestyle": "-", "linewidth": 1.8},
}

BASELINE_STYLE = {
    "label": "No CAS (baseline)", "color": "#555555",
    "linestyle": "--", "linewidth": 2.2,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Week definitions — TX_2018_ANNUAL study
# ═══════════════════════════════════════════════════════════════════════════════

# 8 sensitivity-sweep weeks (2 per season)
SWEEP_WEEKS = [
    '2018-01-07', '2018-01-21',
    '2018-04-01', '2018-04-15',
    '2018-07-01', '2018-07-15',
    '2018-10-07', '2018-10-21',
]

# Full 24-week annual study (6 per season)
SEASONS = {
    "Winter": ["2018-01-07", "2018-01-21", "2018-02-04", "2018-02-18",
               "2018-12-02", "2018-12-16"],
    "Spring": ["2018-03-04", "2018-03-18", "2018-04-01", "2018-04-15",
               "2018-05-06", "2018-05-20"],
    "Summer": ["2018-06-03", "2018-06-17", "2018-07-01", "2018-07-15",
               "2018-08-05", "2018-08-19"],
    "Fall":   ["2018-09-02", "2018-09-16", "2018-10-07", "2018-10-21",
               "2018-11-04", "2018-11-18"],
}

SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
