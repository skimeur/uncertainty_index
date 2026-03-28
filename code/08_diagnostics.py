"""
08_diagnostics.py -- Diagnostic figures for the Uncertainty Index project.

Diagnostics:
    1. variance_diagnostic  -- Cross-sectional variance decomposition over time.
       Panels show the distribution of individual forecast variance (mean,
       median, P10-P90 range), the average number of bins filled, and the
       number of participating forecasters.  These figures reveal whether
       changes in aggregate NU are driven by tail behaviour, broader use of
       the probability bins, or panel composition effects.

    2. merged_publication    -- Publication-ready merged NU plots.  Each figure
       shows the weighted-average NU index in the foreground with individual
       horizon series (1Y, 2Y, 5Y) drawn faintly in the background.  These
       are the main summary figures for the project.

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
from code.utils import figure_note


# ============================================================================
# 1. Variance Diagnostic
# ============================================================================

def variance_diagnostic():
    """
    Cross-sectional variance decomposition over time.

    For each variable and selected horizons, produces a three-panel figure:
      - Panel 1: Individual variance (mean, median, P10-P90 envelope).
      - Panel 2: Average number of bins with non-zero probability.
      - Panel 3: Number of forecasters per survey round.

    These panels help diagnose whether aggregate NU movements stem from
    genuine changes in forecaster uncertainty or from compositional shifts.
    """
    print("\n=== Variance Diagnostic: Cross-Sectional Decomposition ===")

    for variable in config.VARIABLES:
        for hor in ["5Y", "1Y"]:
            indiv_path = config.RESULTS_DIR / f"individual_{variable}_{hor}_niu.csv"
            if not indiv_path.exists():
                continue

            df = pd.read_csv(indiv_path, parse_dates=["Date"])

            stats = df.groupby("Date").agg(
                n_forecasters=("FCT_SOURCE", "count"),
                var_mean=("Variance_spd", "mean"),
                var_median=("Variance_spd", "median"),
                var_p90=("Variance_spd", lambda x: x.quantile(0.90)),
                var_p10=("Variance_spd", lambda x: x.quantile(0.10)),
                bins_filled_mean=("bins_filled", "mean"),
                niu_mean=("NIU", "mean"),
            ).reset_index()

            fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

            color = config.COLORS[variable]
            var_label = config.VARIABLE_LABELS[variable]
            hor_label = config.HORIZON_LABELS[hor]

            # Panel 1: Variance
            ax = axes[0]
            ax.fill_between(stats["Date"], stats["var_p10"], stats["var_p90"],
                            alpha=0.15, color=color, label="P10-P90")
            ax.plot(stats["Date"], stats["var_mean"], color=color,
                    linewidth=2, label="Mean variance")
            ax.plot(stats["Date"], stats["var_median"], color=color,
                    linewidth=1.5, linestyle="--", label="Median variance")
            ax.set_title(f"Variance Diagnostic -- {var_label}, {hor_label}")
            ax.set_ylabel("Individual Variance")
            ax.legend(loc="upper left", fontsize=8)

            # Panel 2: Bins filled
            ax = axes[1]
            ax.plot(stats["Date"], stats["bins_filled_mean"],
                    color=color, linewidth=2)
            ax.set_ylabel("Avg. Bins Filled")
            ax.set_title("Average Number of Bins with Non-Zero Probability")

            # Panel 3: Number of forecasters
            ax = axes[2]
            ax.bar(stats["Date"], stats["n_forecasters"],
                   width=60, color=color, alpha=0.5)
            ax.set_ylabel("N Forecasters")
            ax.set_title("Number of Forecasters per Survey Round")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

            note = figure_note(variable, hor)
            ax.annotate(note, xy=(0, -0.35), xycoords="axes fraction",
                        fontsize=7, color="gray", va="top", style="italic")

            fig.tight_layout()
            fig.subplots_adjust(bottom=0.10)
            config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(config.FIGURES_DIR / f"diagnostic_variance_{variable}_{hor}.png",
                        dpi=config.FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: diagnostic_variance_{variable}_{hor}.png")


# ============================================================================
# 2. Merged NU Publication Plots
# ============================================================================

def merged_publication_plots():
    """
    Publication-ready merged NU figure per variable.

    Each figure shows individual horizon NU series (1Y, 2Y, 5Y) drawn
    faintly in the background and the weighted-average merged NU index
    in the foreground.  The horizontal reference line at NU = 1 marks the
    boundary between excess and compressed uncertainty.
    """
    print("\n=== Merged NU Publication Plots ===")

    for variable in config.VARIABLES:
        merged_path = config.RESULTS_DIR / f"merged_niu_{variable}.csv"
        if not merged_path.exists():
            print(f"  No merged data for {variable}. Run 07_merge_horizons.py first.")
            continue

        merged = pd.read_csv(merged_path, parse_dates=["Date"], index_col="Date")
        var_label = config.VARIABLE_LABELS[variable]
        color = config.COLORS[variable]

        col = "NIU_merged" if "NIU_merged" in merged.columns else "A_weighted_avg"
        if col not in merged.columns:
            continue

        horizon_colors = {"1Y": "#aec7e8", "2Y": "#ffbb78", "5Y": "#ff9896"}

        fig, ax = plt.subplots(figsize=(10, 5))

        # Background: individual horizons
        for h in config.HORIZONS:
            hcol = f"NIU_{h}"
            if hcol in merged.columns:
                ax.plot(merged.index, merged[hcol],
                        color=horizon_colors.get(h, "#cccccc"),
                        linewidth=1.0, alpha=0.5,
                        label=f"{config.HORIZON_LABELS[h]}")

        # Foreground: merged series
        ax.plot(merged.index, merged[col],
                color=color, linewidth=2.5,
                label="Merged NU")

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(f"Normalized Uncertainty Index -- {var_label}")
        ax.set_ylabel("NU")
        ax.legend(loc="upper left", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        note = (
            f"Source: ECB Survey of Professional Forecasters.  "
            f'Methodology: Vansteenberghe (2026), '
            f'"Uncertain and Asymmetric Forecasts", Banque de France WP.\n'
            f"Weighted average of 1Y (50%), 2Y (30%), 5Y (20%) horizon NU.\n"
            f"Dates = survey formation date.  "
            f"Faint lines = individual horizon NU."
        )
        ax.annotate(note, xy=(0, -0.18), xycoords="axes fraction",
                    fontsize=7, color="gray", va="top", style="italic")

        fig.subplots_adjust(bottom=0.28)
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / f"niu_index_{variable}.png",
                    dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: niu_index_{variable}.png")


# ============================================================================
# Main
# ============================================================================

DIAGNOSTICS = {
    "variance_diagnostic": variance_diagnostic,
    "merged_publication": merged_publication_plots,
}


def run_all():
    """Run all standard diagnostics."""
    for name, fn in DIAGNOSTICS.items():
        fn()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        for name in args:
            if name in DIAGNOSTICS:
                DIAGNOSTICS[name]()
            else:
                print(f"Unknown diagnostic: {name}. "
                      f"Available: {list(DIAGNOSTICS.keys())}")
    else:
        run_all()
