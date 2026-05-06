"""
02_prepare_panels.py -- Parse raw ECB-SPF quarterly CSV files into clean panels.

For each variable (inflation, GDP, ...) and horizon (1Y, 2Y, 5Y), extracts the
relevant section from every quarterly CSV, maps coded column names to
human-readable bin labels, aggregates or disaggregates tail bins as needed,
and normalizes probabilities to 100 %.

Author: Eric Vansteenberghe
Reference:
    Vansteenberghe, E. (2026). "Uncertain and Asymmetric Forecasts."
    Banque de France Working Paper.

Output:
    data/panels/panel_{variable}_{horizon}.csv
    Columns: Date, FCT_SOURCE, POINT, [probability bin columns ...]
"""
import sys
import os
import re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================================================
# Helpers
# ============================================================================

def _list_raw_files():
    """List all quarterly CSV files, sorted by name."""
    files = sorted(config.RAW_DIR.glob("*.csv"))
    return files


def _parse_filename(path):
    """Extract year, quarter from filename like '2025Q4.csv'."""
    name = path.stem
    m = re.match(r"(\d{4})Q(\d)", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _formation_date(year, quarter):
    """Survey formation date: Q1→Jan, Q2→Apr, Q3→Jul, Q4→Oct."""
    month = config.QUARTER_TO_FORMATION_MONTH[quarter]
    return pd.Timestamp(year=year, month=month, day=1)


# ============================================================================
# Section extraction
# ============================================================================

def _extract_section(filepath, variable):
    """
    Extract rows belonging to a specific section from a multi-section SPF CSV.

    For inflation: it's the first section (before the first blank line after header).
    For GDP/core: search for the section header, then read until blank line.

    Returns a DataFrame with columns from the section header.
    """
    header_text = config.SECTION_HEADERS[variable]

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if variable == "inflation":
        # Inflation is the first section — read with header on line 2 (index 1)
        df = pd.read_csv(filepath, header=1, dtype=str)
        # Find first all-NaN row
        mask = df.apply(lambda row: row.isna().all() or (row == "").all(), axis=1)
        if mask.any():
            first_empty = mask.idxmax()
            df = df.iloc[:first_empty]
        return df

    # For other sections: find the header line, then parse
    start_idx = None
    for i, line in enumerate(lines):
        if header_text in line:
            start_idx = i
            break

    if start_idx is None:
        return pd.DataFrame()

    # Collect lines from start_idx+1 (column header) until blank line
    data_lines = []
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped == "" or stripped == ",":
            # Stop if we hit blank or another section header
            break
        # Also stop if we hit another section header row in the ECB file.
        first_cell = stripped.split(",", 1)[0].strip()
        if ";" in first_cell and first_cell != header_text:
            break
        data_lines.append(stripped)

    if not data_lines:
        return pd.DataFrame()

    # Parse: first line is column header
    from io import StringIO
    csv_text = "\n".join(data_lines)
    df = pd.read_csv(StringIO(csv_text), dtype=str)

    # Drop empty columns
    df = df.loc[:, df.columns.notna()]
    df = df.loc[:, df.columns != ""]
    empty_cols = [c for c in df.columns if c.strip() == ""]
    df.drop(columns=empty_cols, inplace=True, errors="ignore")

    return df


# ============================================================================
# Target period logic
# ============================================================================

def _get_target_periods_and_date(year, quarter, variable, horizon):
    """
    Determine the TARGET_PERIOD values to filter and the output Date.

    Returns: (list_of_target_periods, formation_date)
    """
    form_date = _formation_date(year, quarter)

    if variable in ("inflation", "core"):
        return _inflation_target(year, quarter, horizon), form_date
    elif variable == "gdp":
        return _gdp_target(year, quarter, horizon), form_date
    return [], form_date


def _select_preferred_targets(df, target_periods):
    """
    Resolve multiple matched TARGET_PERIOD rows for one forecaster.

    target_periods is ordered from preferred exact code to fallback code.
    When both are present in a file, keep the most-preferred row.
    """
    if df.empty or "FCT_SOURCE" not in df.columns or len(target_periods) <= 1:
        return df

    rank = {target: i for i, target in enumerate(target_periods)}
    out = df.copy()
    out["_target_rank"] = out["TARGET_PERIOD"].map(rank).fillna(len(target_periods))
    out.sort_values(["FCT_SOURCE", "_target_rank"], inplace=True)
    out = out.drop_duplicates(subset=["FCT_SOURCE"], keep="first")
    out.drop(columns=["_target_rank"], inplace=True)
    return out


def _inflation_target(year, quarter, horizon):
    """
    Inflation TARGET_PERIOD patterns.
    1Y: Q1→{year}Dec, Q2→{year+1}Mar, Q3→{year+1}Jun, Q4→{year+1}Sep
    CY: Q1→{year}Dec (annual estimate for current year)
    2Y: shift 1Y target by +1 year
    5Y: Q1/Q2→str(year+4), Q3/Q4→str(year+5)
    """
    if horizon == "1Y":
        targets = {
            1: [f"{year}Dec"],
            2: [f"{year+1}Mar"],
            3: [f"{year+1}Jun"],
            4: [f"{year+1}Sep"],
        }
    elif horizon == "CY":
        # Current year = annual target for survey year
        targets = {
            1: [str(year)],
            2: [str(year)],
            3: [str(year)],
            4: [str(year)],
        }
    elif horizon == "2Y":
        targets = {
            1: [f"{year+1}Dec", str(year + 1)],
            2: [f"{year+2}Mar", str(year + 2)],
            3: [f"{year+2}Jun", str(year + 2)],
            4: [f"{year+2}Sep", str(year + 2)],
        }
    elif horizon == "5Y":
        if quarter < 3:
            targets = {quarter: [str(year + 4)]}
        else:
            targets = {quarter: [str(year + 5)]}
    else:
        return []
    return targets.get(quarter, [])


def _gdp_target(year, quarter, horizon):
    """
    GDP TARGET_PERIOD patterns (quarterly targets like '2026Q2').
    1Y: Q1→{year}Q3, Q2→{year}Q4, Q3→{year+1}Q1, Q4→{year+1}Q2
    """
    if horizon == "1Y":
        targets = {
            1: [f"{year}Q3"],
            2: [f"{year}Q4"],
            3: [f"{year+1}Q1"],
            4: [f"{year+1}Q2"],
        }
    elif horizon == "CY":
        targets = {
            1: [str(year)],
            2: [str(year)],
            3: [str(year)],
            4: [str(year)],
        }
    elif horizon == "2Y":
        targets = {
            1: [f"{year+1}Q3", str(year + 1)],
            2: [f"{year+1}Q4", str(year + 2)],
            3: [f"{year+2}Q1", str(year + 2)],
            4: [f"{year+2}Q2", str(year + 2)],
        }
    elif horizon == "5Y":
        if quarter < 3:
            targets = {quarter: [str(year + 4)]}
        else:
            targets = {quarter: [str(year + 5)]}
    else:
        return []
    return targets.get(quarter, [])


# ============================================================================
# Column mapping and bin processing
# ============================================================================

def _get_col_map(variable):
    """Return the full column-name-to-label mapping for a variable."""
    if variable in ("inflation", "core"):
        # Merge pre and post maps
        m = {}
        m.update(config.INFLATION_COL_MAP_PRE)
        m.update(config.INFLATION_COL_MAP_POST)
        return m
    elif variable == "gdp":
        return config.GDP_COL_MAP
    return {}


def _aggregate_inflation_tails(df):
    """
    For pre-2024Q4 inflation: aggregate sub-bins into the 14 canonical bins.
    Collapse left-tail bins into ']-inf,-1]' and right-tail into '[5,+inf['.
    """
    left_sources = [c for c in config.INFLATION_PRE_LEFT_SOURCES if c in df.columns]
    right_sources = [c for c in config.INFLATION_PRE_RIGHT_SOURCES if c in df.columns]

    if left_sources:
        df["]-inf,-1]"] = df[left_sources].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    if right_sources:
        df["[5,+inf["] = df[right_sources].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    return df


def _disaggregate_gdp_bins(df):
    """
    GDP-specific: disaggregate aggregated bins into finer bins
    using proportional distribution (matching existing code logic).
    """
    def _spread(df, target_cols, source_col):
        """Spread source_col evenly across target_cols."""
        if source_col not in df.columns:
            return df
        n = len(target_cols)
        for c in target_cols:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = df[c].fillna(0) + df[source_col].fillna(0) / n
        return df

    # [-7,-5] → split into [-7,-6] + refine [-6,-5.5] and [-5.5,-5]
    if "[-7,-5]" in df.columns:
        if "[-7,-6]" not in df.columns:
            df["[-7,-6]"] = 0.0
        df["[-7,-6]"] = df["[-7,-5]"].fillna(0) / 2
        df["[-6,-5.5]"] = df["[-6,-5.5]"].fillna(0) + df["[-7,-5]"].fillna(0) / 4
        df["[-5.5,-5]"] = df["[-5.5,-5]"].fillna(0) + df["[-7,-5]"].fillna(0) / 4

    # [-5,-3] → spread to [-5,-4.5], [-4.5,-4], [-4,-3.5], [-3.5,-3.1]
    df = _spread(df,
                 ["[-5,-4.5]", "[-4.5,-4]", "[-4,-3.5]", "[-3.5,-3.1]"],
                 "[-5,-3]")

    # [-3,-1] → spread to [-3,-2.5], [-2.5,-2.1], [-2,-1.5], [-1.5,-1.1]
    df = _spread(df,
                 ["[-3,-2.5]", "[-2.5,-2.1]", "[-2,-1.5]", "[-1.5,-1.1]"],
                 "[-3,-1]")

    # ]-inf,-6] → spread to [-15,-13]..[-7,-6]
    df = _spread(df,
                 ["[-15,-13]", "[-13,-11]", "[-11,-9]", "[-9,-7]", "[-7,-6]"],
                 "]-inf,-6]")

    # ]-inf,-1] → spread to all left bins
    all_left = [
        "[-15,-13]", "[-13,-11]", "[-11,-9]", "[-9,-7]", "[-7,-6]",
        "[-6,-5.5]", "[-5.5,-5]", "[-5,-4.5]", "[-4.5,-4]",
        "[-4,-3.5]", "[-3.5,-3.1]", "[-3,-2.5]", "[-2.5,-2.1]",
        "[-2,-1.5]", "[-1.5,-1.1]",
    ]
    df = _spread(df, all_left, "]-inf,-1]")

    # ]-inf,0] → spread to all left bins + [-1,-0.5], [-0.5,-0.1]
    all_left_plus = all_left + ["[-1,-0.5]", "[-0.5,-0.1]"]
    df = _spread(df, all_left_plus, "]-inf,0]")

    # [4,+inf[ → spread to [4.0,4.5], [4.5,5], [8,10], [10,+inf[]
    df = _spread(df,
                 ["[4.0,4.5]", "[4.5,5]", "[8,10]", "[10,+inf["],
                 "[4,+inf[")

    # [4,6] → spread to [4.0,4.5], [4.5,5], [5,5.5], [5.5,6]
    for c in ["[5,5.5]", "[5.5,6]"]:
        if c not in df.columns:
            df[c] = 0.0
    df = _spread(df,
                 ["[4.0,4.5]", "[4.5,5]", "[5,5.5]", "[5.5,6]"],
                 "[4,6]")

    # [5,+inf[ → spread to [8,10], [10,+inf[]
    df = _spread(df,
                 ["[8,10]", "[10,+inf["],
                 "[5,+inf[")

    return df


def build_panel(variable, horizon):
    """
    Build a clean panel for a given variable and horizon.

    Parameters
    ----------
    variable : str, one of 'inflation', 'core', 'gdp'
    horizon  : str, one of 'CY', '1Y', '2Y', '5Y'

    Returns
    -------
    pd.DataFrame with columns: Date, FCT_SOURCE, POINT, [bin columns]
    """
    col_map = _get_col_map(variable)
    raw_files = _list_raw_files()

    if not raw_files:
        print(f"  No raw CSV files found in {config.RAW_DIR}. Run 01_download_data.py first.")
        return pd.DataFrame()

    frames = []

    for fpath in raw_files:
        year, quarter = _parse_filename(fpath)
        if year is None:
            continue

        target_periods, form_date = _get_target_periods_and_date(year, quarter, variable, horizon)
        if not target_periods:
            continue

        # Extract the relevant section
        df_section = _extract_section(fpath, variable)
        if df_section.empty:
            continue

        # Rename coded columns
        df_section.rename(columns=col_map, inplace=True)

        # Convert to numeric (except TARGET_PERIOD)
        for col in df_section.columns:
            if col not in ("TARGET_PERIOD", "FCT_SOURCE"):
                df_section[col] = pd.to_numeric(df_section[col], errors="coerce")
        if "FCT_SOURCE" in df_section.columns:
            df_section["FCT_SOURCE"] = pd.to_numeric(df_section["FCT_SOURCE"], errors="coerce")

        # Filter to target period(s)
        if "TARGET_PERIOD" in df_section.columns:
            mask = df_section["TARGET_PERIOD"].isin(target_periods)
            df_filtered = df_section.loc[mask].copy()
            df_filtered = _select_preferred_targets(df_filtered, target_periods)
        else:
            continue

        if df_filtered.empty:
            continue

        df_filtered["Date"] = form_date
        frames.append(df_filtered)

    if not frames:
        print(f"  No data found for {variable} {horizon}.")
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)

    # Drop TARGET_PERIOD (no longer needed)
    panel.drop(columns=["TARGET_PERIOD"], errors="ignore", inplace=True)

    # Drop all-NaN rows/columns
    panel.dropna(how="all", axis=1, inplace=True)
    panel.dropna(how="all", axis=0, inplace=True)

    # Variable-specific bin processing
    if variable in ("inflation", "core"):
        panel = _aggregate_inflation_tails(panel)
    elif variable == "gdp":
        panel = _disaggregate_gdp_bins(panel)

    # Identify the canonical bin columns for this variable
    bin_defs = _get_bin_defs(variable, panel)
    bin_labels = [b[0] for b in bin_defs]
    bin_cols = [c for c in bin_labels if c in panel.columns]

    # Keep only Date, FCT_SOURCE, POINT, and bin columns
    keep = ["Date", "FCT_SOURCE", "POINT"] + bin_cols
    keep = [c for c in keep if c in panel.columns]
    panel = panel[keep].copy()

    # Normalize probabilities row-wise
    if bin_cols:
        row_sums = panel[bin_cols].sum(axis=1).replace(0, np.nan)
        panel[bin_cols] = panel[bin_cols].div(row_sums, axis=0) * 100.0

    # Drop rows with no probability data
    panel.dropna(subset=bin_cols, how="all", inplace=True)

    panel.sort_values(["Date", "FCT_SOURCE"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    return panel


def _get_bin_defs(variable, panel):
    """
    Return the appropriate canonical bin definitions for a variable.
    For inflation, determine regime based on what columns are present.
    """
    if variable in ("inflation", "core"):
        # Check which regime columns are present
        post_labels = [b[0] for b in config.INFLATION_BINS_POST]
        pre_labels = [b[0] for b in config.INFLATION_BINS_PRE]
        # Return both — rows will have one or the other populated
        return config.INFLATION_BINS_PRE + config.INFLATION_BINS_POST
    elif variable == "gdp":
        return config.GDP_BINS
    return []


def prepare_all(variables=None, horizons=None):
    """Build and save panels for all requested variable/horizon combinations."""
    variables = variables or config.VARIABLES
    horizons = horizons or config.HORIZONS
    config.PANELS_DIR.mkdir(parents=True, exist_ok=True)

    for var in variables:
        for hor in horizons:
            print(f"Building panel: {var} / {hor} ...")
            panel = build_panel(var, hor)
            if panel.empty:
                print(f"  -> empty, skipping.")
                continue
            out_path = config.PANELS_DIR / f"panel_{var}_{hor}.csv"
            panel.to_csv(out_path, index=False)
            n_dates = panel["Date"].nunique()
            n_fct = panel["FCT_SOURCE"].nunique()
            print(f"  -> {len(panel)} rows, {n_dates} dates, {n_fct} forecasters → {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2:
        prepare_all(variables=[args[0]], horizons=[args[1]])
    elif len(args) == 1:
        prepare_all(variables=[args[0]])
    else:
        prepare_all()
