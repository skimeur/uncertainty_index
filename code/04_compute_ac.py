"""
04_compute_ac.py -- Compute Asymmetry Coherence (AC) from individual NU results.

AC measures directional risk by combining the median deviation from target
with Bowley skewness, weighted by a coherence term:

    AC_t = ((Q_tilde + A_tilde) / 2) * ((1 + Q_tilde * A_tilde) / 2)

The GDP target is time-varying (linear interpolation by default) to account
for the secular decline in euro-area potential growth.

Author: Eric Vansteenberghe
Reference:
    Vansteenberghe, E. (2026). "Uncertain and Asymmetric Forecasts."
    Banque de France Working Paper.

Output:
    data/results/individual_{var}_{hor}_ac.csv
    data/results/aggregate_{var}_{hor}_ac.csv

The public AC series is the cross-sectional mean of individual forecaster
scores, matching the aggregation used for NIU.  The aggregate cross-sectional
formula is still exported as a reference diagnostic.
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from code.utils import iqr_scale, get_target


def compute_individual_ac(variable, horizon):
    """
    Compute individual-level AC components.

    Reads the individual NIU file (which contains Bowley_Skewness, Q2_median_spd).
    Adds smoothed Bowley, individual Q_norm, A_norm, ACI, coherence.
    """
    indiv_path = config.RESULTS_DIR / f"individual_{variable}_{horizon}_niu.csv"
    if not indiv_path.exists():
        print(f"  Individual NIU not found: {indiv_path}. Run 03_compute_niu.py first.")
        return pd.DataFrame()

    df = pd.read_csv(indiv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    window = config.BOWLEY_SMOOTHING_WINDOW

    # Compute time-varying target per row
    df["_target"] = df["Date"].apply(lambda d: get_target(variable, d))

    # Smooth Bowley skewness per forecaster (rolling window)
    df.sort_values(["FCT_SOURCE", "Date"], inplace=True)
    df["Bowley_smoothed"] = (
        df.groupby("FCT_SOURCE")["Bowley_Skewness"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )

    # Individual-level normalized components (using time-varying target)
    scale_med = iqr_scale(df["Q2_median_spd"] - df["_target"])
    scale_skw = iqr_scale(df["Bowley_smoothed"])

    df["Q2_norm"] = np.tanh((df["Q2_median_spd"] - df["_target"]) / scale_med)
    df["A_norm"] = np.tanh(df["Bowley_smoothed"] / scale_skw)

    # Individual ACI
    df["ACI"] = (
        0.5 * (df["Q2_norm"] + df["A_norm"])
        * 0.5 * (1.0 + df["Q2_norm"] * df["A_norm"])
    )
    df["coherence"] = 0.5 * (1.0 + df["Q2_norm"] * df["A_norm"])

    # Clean up temporary column
    df.drop(columns=["_target"], inplace=True, errors="ignore")

    df.sort_values(["Date", "FCT_SOURCE"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def aggregate_ac(individual_df, variable):
    """
    Aggregate to date-level AC time series.

    The released AC series is the mean of individual forecaster ACI scores,
    with mean +/- 1 standard error bands.  For reference, we also export the
    cross-sectional formula built from the median of medians and the mean
    Bowley skewness.
    """
    if individual_df.empty:
        return pd.DataFrame()

    # Cross-sectional statistics per date
    date_stats = individual_df.groupby("Date").agg(
        Q_median=("Q2_median_spd", "median"),      # median of individual medians
        A_mean=("Bowley_smoothed", "mean"),        # mean of individual Bowley
        AC_mean=("ACI", "mean"),                   # released AC = mean forecaster ACI
        AC_median=("ACI", "median"),
        AC_std=("ACI", "std"),
        coherence_mean=("coherence", "mean"),
        NIU_mean=("NIU", "mean"),
        n_forecasters=("FCT_SOURCE", "count"),
    ).reset_index()

    # Mean +/- 1 standard error for individual forecaster AC
    se = date_stats["AC_std"] / np.sqrt(date_stats["n_forecasters"])
    date_stats["AC_se_lo"] = date_stats["AC_mean"] - se
    date_stats["AC_se_hi"] = date_stats["AC_mean"] + se

    # Backward-compatible aliases for older downstream code.
    date_stats["ACI_mean"] = date_stats["AC_mean"]
    date_stats["ACI_median"] = date_stats["AC_median"]
    date_stats["ACI_std"] = date_stats["AC_std"]
    date_stats["ACI_se_lo"] = date_stats["AC_se_lo"]
    date_stats["ACI_se_hi"] = date_stats["AC_se_hi"]

    # Compute time-varying targets for aggregate AC
    target_series = date_stats["Date"].apply(lambda d: get_target(variable, d))

    # Compute the cross-sectional formula as a reference diagnostic.
    q_med_idx = date_stats.set_index("Date")["Q_median"]
    a_mean_idx = date_stats.set_index("Date")["A_mean"]
    target_idx = pd.Series(target_series.values, index=q_med_idx.index)

    Q_dev = q_med_idx - target_idx
    iqr_q = iqr_scale(Q_dev)
    iqr_a = iqr_scale(a_mean_idx)
    Q_norm = np.tanh(Q_dev / iqr_q)
    A_norm = np.tanh(a_mean_idx / iqr_a)
    AC = 0.5 * (Q_norm + A_norm) * 0.5 * (1.0 + Q_norm * A_norm)
    coherence = 0.5 * (1.0 + Q_norm * A_norm)
    ac_df = pd.DataFrame({
        "Q_norm_formula": Q_norm,
        "A_norm_formula": A_norm,
        "AC_formula": AC,
        "coherence_formula": coherence,
    })
    ac_df = ac_df.reset_index()

    # Merge
    result = date_stats.merge(ac_df, on="Date", how="left")
    result.sort_values("Date", inplace=True)
    return result


def process_all(variables=None, horizons=None):
    """Compute and save AC for all requested variable/horizon combinations."""
    variables = variables or config.VARIABLES
    horizons = horizons or config.HORIZONS
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for var in variables:
        for hor in horizons:
            print(f"Computing AC: {var} / {hor} ...")
            indiv = compute_individual_ac(var, hor)
            if indiv.empty:
                print(f"  -> empty, skipping.")
                continue

            # Save individual
            indiv_path = config.RESULTS_DIR / f"individual_{var}_{hor}_ac.csv"
            indiv.to_csv(indiv_path, index=False)
            print(f"  -> {len(indiv)} individual rows → {indiv_path}")

            # Aggregate
            agg = aggregate_ac(indiv, var)
            agg_path = config.RESULTS_DIR / f"aggregate_{var}_{hor}_ac.csv"
            agg.to_csv(agg_path, index=False)
            print(f"  -> {len(agg)} dates → {agg_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2:
        process_all(variables=[args[0]], horizons=[args[1]])
    elif len(args) == 1:
        process_all(variables=[args[0]])
    else:
        process_all()
