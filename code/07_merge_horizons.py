"""
07_merge_horizons.py -- Merge NU across horizons into a single time series.

Computes a weighted average of NU across the 1-year, 2-year, and 5-year-ahead
horizons for each variable.  Default weights (0.5, 0.3, 0.2) decline with
horizon, reflecting that shorter horizons have richer cross-sectional data
and are more responsive to current conditions.

Author: Eric Vansteenberghe
Reference:
    Vansteenberghe, E. (2026). "Uncertain and Asymmetric Forecasts."
    Banque de France Working Paper.

Output:
    data/results/merged_niu_{variable}.csv
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================================================
# Data loading
# ============================================================================

def _load_niu_horizons(variable):
    """
    Load aggregate NU for all horizons and align on a common date index.

    Returns a DataFrame with columns NIU_{horizon} indexed by Date.
    """
    frames = {}
    for hor in config.HORIZONS:
        path = config.RESULTS_DIR / f"aggregate_{variable}_{hor}_niu.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        frames[hor] = df.set_index("Date")[["NIU_mean", "NIU_std", "n_forecasters"]]

    if not frames:
        return pd.DataFrame()

    # Outer join to keep maximum date coverage
    combined = pd.DataFrame(index=sorted(
        set().union(*(f.index for f in frames.values()))
    ))
    for hor, df in frames.items():
        combined[f"NIU_{hor}"] = df["NIU_mean"]
        combined[f"std_{hor}"] = df["NIU_std"]
        combined[f"n_{hor}"] = df["n_forecasters"]

    combined.index.name = "Date"
    return combined


# ============================================================================
# Weighted average
# ============================================================================

def weighted_average(data, weights=None):
    """
    Weighted average of NU across horizons.

    Parameters
    ----------
    data : pd.DataFrame
        Aligned horizon data from ``_load_niu_horizons``.
    weights : dict, optional
        Horizon-to-weight mapping.  Default: {1Y: 0.5, 2Y: 0.3, 5Y: 0.2}.

    Returns
    -------
    pd.Series
        Merged NU index.
    """
    if weights is None:
        weights = {"1Y": 0.5, "2Y": 0.3, "5Y": 0.2}

    result = pd.Series(0.0, index=data.index, dtype=float)
    total_w = pd.Series(0.0, index=data.index, dtype=float)

    for h in config.HORIZONS:
        col = f"NIU_{h}"
        if col in data.columns:
            w = weights.get(h, 0.0)
            mask = data[col].notna()
            result[mask] += w * data.loc[mask, col]
            total_w[mask] += w

    result = result.divide(total_w.where(total_w > 0))

    # NaN where no horizon has data
    any_data = data[[f"NIU_{h}" for h in config.HORIZONS
                      if f"NIU_{h}" in data.columns]].notna().any(axis=1)
    result[~any_data] = np.nan

    return result


# ============================================================================
# Processing
# ============================================================================

def process_variable(variable):
    """Compute merged NU for one variable and save to CSV."""
    print(f"\nMerging horizons for {variable} ...")
    data = _load_niu_horizons(variable)
    if data.empty:
        print(f"  No data for {variable}.")
        return

    merged = pd.DataFrame(index=data.index)
    merged.index.name = "Date"

    # Keep individual horizons for reference
    for h in config.HORIZONS:
        col = f"NIU_{h}"
        if col in data.columns:
            merged[col] = data[col]

    merged["NIU_merged"] = weighted_average(data)
    merged["A_weighted_avg"] = merged["NIU_merged"]

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / f"merged_niu_{variable}.csv"
    merged.to_csv(out_path)
    print(f"  Saved: {out_path}")

    return merged


def process_all(variables=None):
    """Run merging for all variables."""
    variables = variables or config.VARIABLES

    for var in variables:
        process_variable(var)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        process_all(variables=[args[0]])
    else:
        process_all()
