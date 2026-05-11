"""
09_decomposition.py -- Variance decomposition: individual uncertainty vs.
disagreement, for every (variable, horizon) combination.

Theory
------
For each survey round, the variance of the cross-sectional *average* SPD
admits the textbook decomposition

    Var(average SPD)_t  =  E_i[ Var(SPD_{i,t}) ]   +   Var_i[ E(SPD_{i,t}) ]
    --------------------    ------------------------    -------------------
       LHS (pooled         Average individual           Disagreement
       uncertainty)        SPD variance                 (variance of means)

The left-hand side mixes two distinct economic channels:

    * Average individual SPD variance -- the typical forecaster's own
      subjective uncertainty.
    * Disagreement -- how different forecasters' point views are from
      each other.

This step materializes the decomposition per (variable, horizon) and
saves both a CSV of the components and the publication-style stacked
figures used in the working paper.

Reference figures from the paper:
    * SPD_Variance_Decomposition_MeanDisagreement_Stacked.png
    * Variance_Decomposition_Stacked_Theoretical.png
    * Variance_Decomposition_Stacked.png
    * growth_decomposition_alt.png

Outputs (per variable x horizon)
--------------------------------
    data/results/decomposition_{variable}_{horizon}.csv
        Date, var_of_avg_spd, avg_indiv_var, disagreement,
        disagreement_theo, rhs_sum, gap, n_forecasters

    figures/decomposition_{variable}_{horizon}.png
        Stacked area: avg individual variance + disagreement (variance of
        SPD means), with the variance of the averaged SPD overlaid as
        a black line.

    figures/decomposition_theo_{variable}_{horizon}.png
        Identical stack but using the residual ``disagreement_theo`` so
        that the components sum exactly to the LHS (no gap).

Author: Eric Vansteenberghe
Reference:
    Vansteenberghe, E. (2026). "Uncertain and Asymmetric Forecasts."
    Banque de France Working Paper.
"""
import sys
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from steps.utils import (
    compute_bin_edges, compute_spd_moments, figure_note,
)


# ============================================================================
# Helpers
# ============================================================================

def _bin_config_for_date(variable, date):
    """
    Return (labels, edges) matching the bin regime active at ``date``.
    Mirrors steps/03_compute_niu.py:_get_bin_config so the variance of the
    averaged SPD is computed on the same support as individual variances.
    """
    cutoff = pd.Timestamp(config.BIN_REGIME_CUTOFF)

    if variable in ("inflation", "core"):
        bin_defs = (
            config.INFLATION_BINS_POST if date >= cutoff
            else config.INFLATION_BINS_PRE
        )
    elif variable == "gdp":
        bin_defs = config.GDP_BINS
    else:
        return [], []

    tail_min_map = {"inflation": -5.0, "core": -5.0, "gdp": -20.0}
    tail_max_map = {"inflation": 12.0, "core": 12.0, "gdp": 15.0}
    tail_min = tail_min_map.get(variable, -20.0)
    tail_max = tail_max_map.get(variable, 20.0)

    return compute_bin_edges(bin_defs, tail_min=tail_min, tail_max=tail_max)


def _variance_of_average_spd(panel_df, variable):
    """
    Build the cross-sectional average SPD per Date and compute its
    variance using the bin regime active at that date.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Output of steps/02_prepare_panels.py.  Long format with bin
        probability columns and ``Date``/``FCT_SOURCE`` identifiers.
    variable : str

    Returns
    -------
    pd.DataFrame with columns Date, var_of_avg_spd, mean_of_avg_spd.
    """
    panel_df = panel_df.copy()
    panel_df["Date"] = pd.to_datetime(panel_df["Date"])

    meta_cols = {"Date", "FCT_SOURCE", "POINT"}
    bin_cols = [c for c in panel_df.columns if c not in meta_cols]

    # Fill NaN with 0 inside bins, then average across forecasters per Date
    panel_df[bin_cols] = panel_df[bin_cols].fillna(0.0)
    avg_spd = panel_df.groupby("Date")[bin_cols].mean()

    rows = []
    for date, row in avg_spd.iterrows():
        labels, edges = _bin_config_for_date(variable, date)
        if not labels:
            continue

        probs = np.array(
            [float(row.get(lbl, 0.0)) for lbl in labels], dtype=float
        )
        total = probs.sum()
        if total <= 0:
            continue
        probs = probs / total * 100.0

        mu, var = compute_spd_moments(probs, edges)
        rows.append({
            "Date": date,
            "var_of_avg_spd": var,
            "mean_of_avg_spd": mu,
        })

    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def compute_decomposition(variable, horizon):
    """
    Compute the variance decomposition for one (variable, horizon).

    The three components come from two complementary sources:
        * Var(avg SPD)      -- recomputed from the panel CSV (average
                                across forecasters of bin probabilities).
        * E_i[ Var(SPD_i) ] -- mean of the individual SPD variances
                                already produced by step ``niu``.
        * Var_i[ E(SPD_i) ] -- population variance (ddof=0) of individual
                                SPD means produced by step ``niu``.

    Returns
    -------
    pd.DataFrame indexed by Date with columns:
        var_of_avg_spd, avg_indiv_var, disagreement,
        disagreement_theo, rhs_sum, gap, n_forecasters.
    """
    panel_path = config.PANELS_DIR / f"panel_{variable}_{horizon}.csv"
    indiv_path = config.RESULTS_DIR / f"individual_{variable}_{horizon}_niu.csv"

    if not panel_path.exists():
        print(f"  Panel not found: {panel_path}. Run step 'panels' first.")
        return pd.DataFrame()
    if not indiv_path.exists():
        print(f"  Individual NIU not found: {indiv_path}. Run step 'niu' first.")
        return pd.DataFrame()

    panel = pd.read_csv(panel_path)
    indiv = pd.read_csv(indiv_path, parse_dates=["Date"])

    # Cross-sectional Var of individual means + mean of individual variances
    agg = (
        indiv.dropna(subset=["Mean_spd", "Variance_spd"])
        .groupby("Date")
        .agg(
            avg_indiv_var=("Variance_spd", "mean"),
            disagreement=("Mean_spd", lambda s: s.var(ddof=0)),
            n_forecasters=("FCT_SOURCE", "count"),
        )
        .reset_index()
    )

    var_avg = _variance_of_average_spd(panel, variable)
    if var_avg.empty:
        return pd.DataFrame()

    out = pd.merge(var_avg, agg, on="Date", how="inner")

    out["rhs_sum"] = out["avg_indiv_var"] + out["disagreement"]
    out["gap"] = out["var_of_avg_spd"] - out["rhs_sum"]
    out["disagreement_theo"] = (
        out["var_of_avg_spd"] - out["avg_indiv_var"]
    ).clip(lower=0.0)

    out.sort_values("Date", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


# ============================================================================
# Plots
# ============================================================================

def _plot_stack(
    decomp,
    variable,
    horizon,
    out_path,
    use_theoretical=False,
):
    """
    Stacked area chart of the variance decomposition.

    When ``use_theoretical`` is True, the disagreement layer is the
    residual ``disagreement_theo``, so the two stacks sum *exactly* to
    the variance of the averaged SPD (no gap visible).  When False, the
    disagreement layer is the cross-sectional variance of the individual
    SPD means; any residual ``gap`` -- typically tiny -- shows as a thin
    sliver between the top of the stack and the black line.
    """
    if decomp.empty:
        return

    dates = pd.to_datetime(decomp["Date"]).to_numpy()
    avg_var = decomp["avg_indiv_var"].to_numpy()
    if use_theoretical:
        disagree = decomp["disagreement_theo"].to_numpy()
        title_suffix = (
            "Avg Individual Variance + Disagreement (identity)"
        )
        disagree_label = "Disagreement (residual; identity)"
    else:
        disagree = decomp["disagreement"].to_numpy()
        title_suffix = (
            "Avg Individual Variance + Disagreement (Var of means)"
        )
        disagree_label = "Disagreement (Var of SPD means)"
    lhs = decomp["var_of_avg_spd"].to_numpy()

    var_label = config.VARIABLE_LABELS.get(variable, variable)
    hor_label = config.HORIZON_LABELS.get(horizon, horizon)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.stackplot(
        dates,
        avg_var,
        disagree,
        labels=["Average individual SPD variance", disagree_label],
        colors=["#2ca02c", "#d62728"],
        alpha=0.45,
    )

    ax.plot(
        dates,
        lhs,
        color="black",
        linewidth=2.4,
        label="Variance of averaged SPD",
    )

    ax.set_title(
        f"Variance decomposition -- {var_label}, {hor_label}\n"
        f"{title_suffix}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Variance (pp$^2$)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    note = figure_note(variable, horizon)
    ax.annotate(
        note, xy=(0, -0.18), xycoords="axes fraction",
        fontsize=7, color="gray", va="top", style="italic",
    )

    fig.subplots_adjust(bottom=0.22)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ============================================================================
# Top-level orchestrator
# ============================================================================

def process_all(variables=None, horizons=None):
    """Compute the decomposition and save CSVs + figures for every combo."""
    variables = variables or config.VARIABLES
    horizons = horizons or config.HORIZONS
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for var in variables:
        for hor in horizons:
            print(f"Decomposition: {var} / {hor} ...")
            decomp = compute_decomposition(var, hor)
            if decomp.empty:
                print("  -> empty, skipping.")
                continue

            csv_path = config.RESULTS_DIR / f"decomposition_{var}_{hor}.csv"
            decomp.to_csv(csv_path, index=False)

            mae = decomp["gap"].abs().mean()
            rmse = np.sqrt((decomp["gap"] ** 2).mean())
            corr = decomp[["var_of_avg_spd", "rhs_sum"]].corr().iloc[0, 1]
            print(
                f"  -> {len(decomp)} dates, "
                f"Corr[LHS,RHS]={corr:0.3f}, "
                f"MAE(gap)={mae:0.4g}, RMSE(gap)={rmse:0.4g} -> {csv_path.name}"
            )

            _plot_stack(
                decomp, var, hor,
                config.FIGURES_DIR / f"decomposition_{var}_{hor}.png",
                use_theoretical=False,
            )
            _plot_stack(
                decomp, var, hor,
                config.FIGURES_DIR / f"decomposition_theo_{var}_{hor}.png",
                use_theoretical=True,
            )


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2:
        process_all(variables=[args[0]], horizons=[args[1]])
    elif len(args) == 1:
        process_all(variables=[args[0]])
    else:
        process_all()
