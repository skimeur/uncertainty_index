"""
Configuration for the Uncertainty Index project.

Author: Eric Vansteenberghe
Reference:
    Vansteenberghe, E. (2026). "Uncertain and Asymmetric Forecasts."
    Banque de France Working Paper.

Theory
------
Normalized Uncertainty (NU) corrects raw forecast variance for the mechanical
widening that occurs when expectations drift from a policy anchor.  The raw
variance of a subjective probability distribution satisfies, to first order,
Var ~ a + b * |mu - mu*|, where mu* is the anchor.  NU divides out this
structural component:

    NU_{i,t} = sqrt(Var_{i,t}) / sqrt(a + b * |mu_{i,t} - mu*|)

With unit calibration (a = b = 1) the measure is sample-free and correlates
> 0.99 with in-sample estimates.  NU = 1 means all dispersion is explained
by distance from the anchor; NU > 1 signals excess uncertainty; NU < 1
signals compressed uncertainty.

Asymmetry Coherence (AC) extracts a directional-risk signal from the third
moment.  It pairs the median deviation from target with Bowley skewness,
weighting their average by a coherence term that rewards sign agreement:

    AC_t = ((Q_tilde + A_tilde) / 2) * ((1 + Q_tilde * A_tilde) / 2)

AC lies in (-1, 1): positive values indicate coherent upside risk, negative
values indicate coherent downside risk, and values near zero indicate no
directional signal.
"""
import os
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "SPF_individual_forecasts"
PANELS_DIR = DATA_DIR / "panels"
REALIZED_DIR = DATA_DIR / "realized"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

SPF_ZIP_URL = (
    "https://www.ecb.europa.eu/stats/prices/indic/forecast/"
    "shared/files/SPF_individual_forecasts.zip"
)

# ============================================================================
# Variables and horizons to process
# ============================================================================
VARIABLES = ["inflation", "gdp"]
HORIZONS = ["1Y", "2Y", "5Y"]

# ============================================================================
# Targets (policy anchors / structural benchmarks)
# ============================================================================
TARGETS = {
    "inflation": 2.0,       # ECB inflation target (%) — fixed
    "core": 2.0,            # Same as headline inflation — fixed
    "gdp": 1.3,             # Euro Area potential growth (%) — fallback, see GDP_TARGET_*
}

# ============================================================================
# Time-varying GDP target (default: linear interpolation)
# ============================================================================
# Euro-area potential growth declined from ~2.3 % in 1999 to ~1.0 % in 2026.
# A fixed target would create a spurious trend in NU because the denominator
# sqrt(1 + |mu - target|) would be artificially large in early periods.
# Linear interpolation removes this artefact.
GDP_TARGET_TIME_VARYING = True
GDP_TARGET_START_YEAR = 1999
GDP_TARGET_START_VALUE = 2.3     # % — pre-crisis potential estimate
GDP_TARGET_END_YEAR = 2026
GDP_TARGET_END_VALUE = 1.0       # % — current EC/ECB estimate

# Alternative: AMECO official potential growth estimates (more granular,
# captures the 2009-2014 potential growth collapse).  To use AMECO, set
# GDP_TARGET_MODE = "ameco" and provide the CSV at GDP_TARGET_AMECO_PATH.
GDP_TARGET_MODE = "linear"       # "linear" (default) or "ameco"
GDP_TARGET_AMECO_PATH = DATA_DIR / "realized" / "potential_growth_ameco.csv"

# ============================================================================
# NIU parameters (unit calibration — robust, sample-free)
#   NIU_{i,t} = sqrt(Var_{i,t}) / (a + b * |mu_{i,t} - target|)^p
# ============================================================================
NIU_A = 1.0
NIU_B = 1.0
NIU_P = 0.5

# ============================================================================
# AC parameters
# ============================================================================
BOWLEY_SMOOTHING_WINDOW = 2   # Rolling window for Bowley skewness (per forecaster)
LOESS_FRAC = 0.3              # Bandwidth for LOESS (sigma on realized series)

# ============================================================================
# Bin regime cutoff: ECB changed HICP bin definitions in 2024Q4
# ============================================================================
BIN_REGIME_CUTOFF = "2024-09-01"

# ============================================================================
# Section headers in the raw SPF CSV files
# ============================================================================
SECTION_HEADERS = {
    "inflation": "INFLATION EXPECTATIONS; YEAR-ON-YEAR CHANGE IN HICP",
    "core":      "CORE INFLATION EXPECTATIONS; YEAR-ON-YEAR CHANGE IN CORE",
    "gdp":       "GROWTH EXPECTATIONS; YEAR-ON-YEAR CHANGE IN REAL GDP",
}

# ============================================================================
# Column-name-to-label mappings (coded column → human-readable label)
# These cover ALL column names that appear across the full 1999–2026 history.
# ============================================================================

# --- Inflation / Core: pre-2024Q4 regime ---
INFLATION_COL_MAP_PRE = {
    "TN4_0":      "]-inf,-4]",
    "TN2_0":      "]-inf,-2]",
    "TN1_0":      "]-inf,-1]",
    "T0_0":       "]-inf,0]",
    "FN4_0TN3_6": "[-4,-3.6]",
    "FN3_5TN3_1": "[-3.5,-3.1]",
    "FN3_0TN2_6": "[-3,-2.6]",
    "FN2_5TN2_1": "[-2.5,-2.1]",
    "FN2_0TN1_6": "[-2,-1.6]",
    "FN1_5TN1_1": "[-1.5,-1.1]",
    "FN1_0TN0_6": "[-1,-0.6]",
    "FN0_5TN0_1": "[-0.5,-0.1]",
    "F0_0T0_4":   "[0,0.4]",
    "F0_5T0_9":   "[0.5,0.9]",
    "F1_0T1_4":   "[1,1.4]",
    "F1_5T1_9":   "[1.5,1.9]",
    "F2_0T2_4":   "[2,2.4]",
    "F2_5T2_9":   "[2.5,2.9]",
    "F3_0T3_4":   "[3,3.4]",
    "F3_5T3_9":   "[3.5,3.9]",
    "F4_0T4_4":   "[4.0,4.4]",
    "F4_5T4_9":   "[4.5,4.9]",
    "F3_5":       "[3.5,+inf[",
    "F4_0":       "[4,+inf[",
    "F5_0":       "[5,+inf[",
}

# --- Inflation / Core: post-2024Q4 regime ---
INFLATION_COL_MAP_POST = {
    "TN0_8":      "]-inf,-0.8]",
    "FN0_7TN0_3": "[-0.7,-0.3]",
    "FN0_2T0_2":  "[-0.2,0.2]",
    "F0_3T0_7":   "[0.3,0.7]",
    "F0_8T1_2":   "[0.8,1.2]",
    "F1_3T1_7":   "[1.3,1.7]",
    "F1_8T2_2":   "[1.8,2.2]",
    "F2_3T2_7":   "[2.3,2.7]",
    "F2_8T3_2":   "[2.8,3.2]",
    "F3_3T3_7":   "[3.3,3.7]",
    "F3_8T4_2":   "[3.8,4.2]",
    "F4_3T4_7":   "[4.3,4.7]",
    "F4_8":       "[4.8,+inf[",
}

# --- GDP Growth: all column names across history ---
GDP_COL_MAP = {
    "TN15_0":      "]-inf,-15]",
    "TN6_0":       "]-inf,-6]",
    "TN4_0":       "]-inf,-4]",
    "TN2_0":       "]-inf,-2]",
    "TN1_0":       "]-inf,-1]",
    "T0_0":        "]-inf,0]",
    "FN15_0TN13_1":"[-15,-13]",
    "FN13_0TN11_1":"[-13,-11]",
    "FN11_0TN9_1": "[-11,-9]",
    "FN9_0TN7_1":  "[-9,-7]",
    "FN7_0TN5_1":  "[-7,-5]",
    "FN6_0TN5_6":  "[-6,-5.5]",
    "FN5_5TN5_1":  "[-5.5,-5]",
    "FN5_0TN4_6":  "[-5,-4.5]",
    "FN4_5TN4_1":  "[-4.5,-4]",
    "FN5_0TN3_1":  "[-5,-3]",
    "FN3_0TN1_1":  "[-3,-1]",
    "FN4_0TN3_6":  "[-4,-3.5]",
    "FN3_5TN3_1":  "[-3.5,-3.1]",
    "FN3_0TN2_6":  "[-3,-2.5]",
    "FN2_5TN2_1":  "[-2.5,-2.1]",
    "FN2_0TN1_6":  "[-2,-1.5]",
    "FN1_5TN1_1":  "[-1.5,-1.1]",
    "FN1_0TN0_6":  "[-1,-0.5]",
    "FN0_5TN0_1":  "[-0.5,-0.1]",
    "F0_0T0_4":    "[0,0.5]",
    "F0_5T0_9":    "[0.5,1]",
    "F1_0T1_4":    "[1,1.5]",
    "F1_5T1_9":    "[1.5,2]",
    "F2_0T2_4":    "[2,2.5]",
    "F2_5T2_9":    "[2.5,3]",
    "F3_0T3_4":    "[3,3.5]",
    "F3_5T3_9":    "[3.5,4]",
    "F4_0T4_4":    "[4.0,4.5]",
    "F4_0T5_9":    "[4,6]",
    "F6_0T7_9":    "[6,8]",
    "F4_5T4_9":    "[4.5,5]",
    "F8_0T9_9":    "[8,10]",
    "F10_0":       "[10,+inf[",
    "F3_5":        "[3.5,+inf[",
    "F4_0":        "[4,+inf[",
    "F5_0":        "[5,+inf[",
}

# ============================================================================
# Canonical bin definitions for SPD moment computation
# Each entry: (label, lower_edge, upper_edge)
# The first/last bins use dynamic tails based on realized data range.
# ============================================================================

# Inflation PRE-2024Q4: 14 canonical bins after tail aggregation
INFLATION_BINS_PRE = [
    ("]-inf,-1]",   None, -1.0),
    ("[-1,-0.6]",   -1.0, -0.5),
    ("[-0.5,-0.1]", -0.5,  0.0),
    ("[0,0.4]",      0.0,  0.5),
    ("[0.5,0.9]",    0.5,  1.0),
    ("[1,1.4]",      1.0,  1.5),
    ("[1.5,1.9]",    1.5,  2.0),
    ("[2,2.4]",      2.0,  2.5),
    ("[2.5,2.9]",    2.5,  3.0),
    ("[3,3.4]",      3.0,  3.5),
    ("[3.5,3.9]",    3.5,  4.0),
    ("[4.0,4.4]",    4.0,  4.5),
    ("[4.5,4.9]",    4.5,  5.0),
    ("[5,+inf[",     5.0,  None),
]

# Inflation POST-2024Q4: 13 bins
INFLATION_BINS_POST = [
    ("]-inf,-0.8]",  None,  -0.75),
    ("[-0.7,-0.3]", -0.75, -0.25),
    ("[-0.2,0.2]",  -0.25,  0.25),
    ("[0.3,0.7]",    0.25,  0.75),
    ("[0.8,1.2]",    0.75,  1.25),
    ("[1.3,1.7]",    1.25,  1.75),
    ("[1.8,2.2]",    1.75,  2.25),
    ("[2.3,2.7]",    2.25,  2.75),
    ("[2.8,3.2]",    2.75,  3.25),
    ("[3.3,3.7]",    3.25,  3.75),
    ("[3.8,4.2]",    3.75,  4.25),
    ("[4.3,4.7]",    4.25,  4.75),
    ("[4.8,+inf[",   4.75,  None),
]

# GDP: 33 fine bins after disaggregation (matching existing code)
GDP_BINS = [
    ("]-inf,-15]",   None,  -15.0),
    ("[-15,-13]",   -15.0, -13.0),
    ("[-13,-11]",   -13.0, -11.0),
    ("[-11,-9]",    -11.0,  -9.0),
    ("[-9,-7]",      -9.0,  -7.0),
    ("[-7,-6]",      -7.0,  -6.0),
    ("[-6,-5.5]",    -6.0,  -5.5),
    ("[-5.5,-5]",    -5.5,  -5.0),
    ("[-5,-4.5]",    -5.0,  -4.5),
    ("[-4.5,-4]",    -4.5,  -4.0),
    ("[-4,-3.5]",    -4.0,  -3.5),
    ("[-3.5,-3.1]",  -3.5,  -3.0),
    ("[-3,-2.5]",    -3.0,  -2.5),
    ("[-2.5,-2.1]",  -2.5,  -2.0),
    ("[-2,-1.5]",    -2.0,  -1.5),
    ("[-1.5,-1.1]",  -1.5,  -1.0),
    ("[-1,-0.5]",    -1.0,  -0.5),
    ("[-0.5,-0.1]",  -0.5,   0.0),
    ("[0,0.5]",       0.0,   0.5),
    ("[0.5,1]",       0.5,   1.0),
    ("[1,1.5]",       1.0,   1.5),
    ("[1.5,2]",       1.5,   2.0),
    ("[2,2.5]",       2.0,   2.5),
    ("[2.5,3]",       2.5,   3.0),
    ("[3,3.5]",       3.0,   3.5),
    ("[3.5,4]",       3.5,   4.0),
    ("[4.0,4.5]",     4.0,   4.5),
    ("[4.5,5]",       4.5,   5.0),
    ("[5,5.5]",       5.0,   5.5),
    ("[5.5,6]",       5.5,   6.0),
    ("[6,8]",         6.0,   8.0),
    ("[8,10]",        8.0,  10.0),
    ("[10,+inf[",    10.0,  None),
]

# ============================================================================
# Pre-2024Q4 inflation: tail bins to aggregate into canonical bins
# ============================================================================
INFLATION_PRE_LEFT_SOURCES = [
    "]-inf,-4]", "[-4,-3.6]", "[-3.5,-3.1]", "[-3,-2.6]",
    "[-2.5,-2.1]", "[-2,-1.6]", "[-1.5,-1.1]",
    "]-inf,-2]", "]-inf,-1]", "]-inf,0]",
]
INFLATION_PRE_RIGHT_SOURCES = ["[3.5,+inf[", "[4,+inf[", "[5,+inf["]

# ============================================================================
# Formation date mapping: when the survey was actually conducted
# Q1 survey → January; Q2 → April; Q3 → July; Q4 → October
# ============================================================================
QUARTER_TO_FORMATION_MONTH = {1: 1, 2: 4, 3: 7, 4: 10}

# ============================================================================
# Figure settings
# ============================================================================
FIGURE_DPI = 300
FIGURE_FORMATS = ["png"]
FIGURE_STYLE = "seaborn-v0_8-whitegrid"

# ============================================================================
# Citation
# ============================================================================
CITATION = (
    'Vansteenberghe, E. (2026). "Uncertain and Asymmetric Forecasts." '
    "Banque de France Working Paper."
)
CITATION_SHORT = "Vansteenberghe (2026)"

# Colors for consistent plotting
COLORS = {
    "inflation": "#1f77b4",
    "core":      "#ff7f0e",
    "gdp":       "#2ca02c",
}

VARIABLE_LABELS = {
    "inflation": "HICP Inflation",
    "core": "Core Inflation",
    "gdp": "Real GDP Growth",
}

HORIZON_LABELS = {
    "CY": "Current Year",
    "1Y": "1-Year Ahead",
    "2Y": "2-Year Ahead",
    "5Y": "5-Year Ahead",
}
