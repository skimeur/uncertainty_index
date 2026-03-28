"""
03_compute_niu.py -- Compute Normalized Uncertainty (NU) from individual SPD panels.

For each forecaster x survey round, computes SPD moments (mean, variance),
quartiles, Bowley skewness, NU, normalized entropy, and informativeness.
Results are then aggregated across forecasters per survey date.

The GDP target is time-varying (linear interpolation by default) to account
for the secular decline in euro-area potential growth.

Author: Eric Vansteenberghe
Reference:
    Vansteenberghe, E. (2026). "Uncertain and Asymmetric Forecasts."
    Banque de France Working Paper.

Output:
    data/results/individual_{var}_{hor}_niu.csv
    data/results/aggregate_{var}_{hor}_niu.csv
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from code.utils import (
    compute_bin_edges, compute_spd_moments, find_percentile,
    bowley_skewness, compute_niu, normalized_entropy,
    informativeness, bins_filled, get_target,
)


def _get_bin_config(variable, date):
    """
    Return (bin_labels, bin_edges) appropriate for the variable and date.
    For inflation/core: regime depends on date vs cutoff.
    """
    cutoff = pd.Timestamp(config.BIN_REGIME_CUTOFF)

    if variable in ("inflation", "core"):
        if date >= cutoff:
            bin_defs = config.INFLATION_BINS_POST
        else:
            bin_defs = config.INFLATION_BINS_PRE
    elif variable == "gdp":
        bin_defs = config.GDP_BINS
    else:
        return [], []

    # Dynamic tails: use reasonable defaults
    tail_min_map = {"inflation": -5.0, "core": -5.0, "gdp": -20.0}
    tail_max_map = {"inflation": 12.0, "core": 12.0, "gdp": 15.0}
    tail_min = tail_min_map.get(variable, -20.0)
    tail_max = tail_max_map.get(variable, 20.0)

    labels, edges = compute_bin_edges(bin_defs, tail_min=tail_min, tail_max=tail_max)
    return labels, edges


def compute_individual_niu(variable, horizon):
    """
    Compute individual-level NIU and SPD statistics.

    Returns DataFrame with columns:
        Date, FCT_SOURCE, POINT, Mean_spd, Variance_spd, sigma_spd,
        Q1, Q2_median_spd, Q3, Bowley_Skewness, NIU,
        bins_filled, entropy_norm, I_informativeness
    """
    panel_path = config.PANELS_DIR / f"panel_{variable}_{horizon}.csv"
    if not panel_path.exists():
        print(f"  Panel not found: {panel_path}. Run 02_prepare_panels.py first.")
        return pd.DataFrame()

    df = pd.read_csv(panel_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Process each row
    results = []
    for _, row in df.iterrows():
        date = row["Date"]
        fct = row.get("FCT_SOURCE", np.nan)
        point = row.get("POINT", np.nan)

        # Time-varying target for GDP, fixed for others
        target = get_target(variable, date)

        labels, edges = _get_bin_config(variable, date)
        if not labels:
            continue

        # Extract probabilities for this row's bin regime
        probs = []
        for lbl in labels:
            val = row.get(lbl, 0.0)
            probs.append(float(val) if pd.notna(val) else 0.0)
        probs = np.array(probs)

        total = probs.sum()
        if total <= 0:
            continue

        # Normalize to 100%
        probs = probs / total * 100.0

        # Moments
        mu, var = compute_spd_moments(probs, edges)
        sigma = np.sqrt(var) if np.isfinite(var) and var >= 0 else np.nan

        # Quartiles
        cdf = np.cumsum(probs)
        q1 = find_percentile(25.0, edges, cdf)
        q2 = find_percentile(50.0, edges, cdf)
        q3 = find_percentile(75.0, edges, cdf)

        # Bowley skewness
        bow = bowley_skewness(q1, q2, q3)

        # NIU
        niu = compute_niu(var, mu, target, a=config.NIU_A, b=config.NIU_B, p=config.NIU_P)

        # Entropy and informativeness
        K = len(edges)
        bf = bins_filled(probs)
        ent = normalized_entropy(probs, K)
        info = informativeness(probs, K)

        results.append({
            "Date": date,
            "FCT_SOURCE": fct,
            "POINT": point,
            "Mean_spd": mu,
            "Variance_spd": var,
            "sigma_spd": sigma,
            "Q1": q1,
            "Q2_median_spd": q2,
            "Q3": q3,
            "Bowley_Skewness": bow,
            "NIU": niu,
            "bins_filled": bf,
            "entropy_norm": ent,
            "I_informativeness": info,
        })

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out.sort_values(["Date", "FCT_SOURCE"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def aggregate_niu(individual_df):
    """
    Aggregate individual NIU to date-level time series.

    Returns DataFrame with:
        Date, NIU_mean, NIU_se_lo, NIU_se_hi, NIU_median,
        Variance_mean, sigma_mean, Mean_spd_mean,
        n_forecasters
    """
    if individual_df.empty:
        return pd.DataFrame()

    agg = individual_df.groupby("Date").agg(
        NIU_mean=("NIU", "mean"),
        NIU_median=("NIU", "median"),
        NIU_std=("NIU", "std"),
        Variance_mean=("Variance_spd", "mean"),
        sigma_mean=("sigma_spd", "mean"),
        Mean_spd_mean=("Mean_spd", "mean"),
        POINT_mean=("POINT", "mean"),
        n_forecasters=("FCT_SOURCE", "count"),
    ).reset_index()

    # Mean +/- 1 standard error
    se = agg["NIU_std"] / np.sqrt(agg["n_forecasters"])
    agg["NIU_se_lo"] = agg["NIU_mean"] - se
    agg["NIU_se_hi"] = agg["NIU_mean"] + se

    agg.sort_values("Date", inplace=True)
    return agg


def process_all(variables=None, horizons=None):
    """Compute and save NIU for all requested variable/horizon combinations."""
    variables = variables or config.VARIABLES
    horizons = horizons or config.HORIZONS
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for var in variables:
        for hor in horizons:
            print(f"Computing NIU: {var} / {hor} ...")
            indiv = compute_individual_niu(var, hor)
            if indiv.empty:
                print(f"  -> empty, skipping.")
                continue

            # Save individual
            indiv_path = config.RESULTS_DIR / f"individual_{var}_{hor}_niu.csv"
            indiv.to_csv(indiv_path, index=False)
            print(f"  -> {len(indiv)} individual rows → {indiv_path}")

            # Aggregate
            agg = aggregate_niu(indiv)
            agg_path = config.RESULTS_DIR / f"aggregate_{var}_{hor}_niu.csv"
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
