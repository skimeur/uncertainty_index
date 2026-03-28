"""
06_plot_results.py -- Generate publication-ready figures for NU and AC indicators.

Figures produced:
    1. NU time series (mean +/- 1 SE band) per variable x horizon
    2. AC time series (mean +/- 1 SE band) per variable x horizon
    3. NU vs raw variance comparison (normalization effect)
    4. Multi-horizon overlay (1Y, 2Y, 5Y) per variable
    5. Multi-variable panel (inflation vs GDP) per horizon

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
# Style setup
# ============================================================================

def _setup_style():
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
    })


def _save_fig(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in config.FIGURE_FORMATS:
        path = config.FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def _add_note(ax, variable, horizon):
    note = figure_note(variable, horizon)
    ax.annotate(
        note, xy=(0, -0.18), xycoords="axes fraction",
        fontsize=7, color="gray", va="top",
        style="italic",
    )


# ============================================================================
# Individual figure types
# ============================================================================

def plot_niu_timeseries(variable, horizon):
    """NIU time series with mean +/- 1 SE band."""
    path = config.RESULTS_DIR / f"aggregate_{variable}_{horizon}_niu.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, parse_dates=["Date"])

    _setup_style()
    fig, ax = plt.subplots()

    ax.fill_between(df["Date"], df["NIU_se_lo"], df["NIU_se_hi"],
                     alpha=0.25, color=config.COLORS[variable],
                     label="Mean +/- 1 SE")
    ax.plot(df["Date"], df["NIU_mean"], color=config.COLORS[variable],
            label="Mean NIU", linewidth=2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    var_label = config.VARIABLE_LABELS[variable]
    hor_label = config.HORIZON_LABELS[horizon]
    ax.set_title(f"Normalized Uncertainty — {var_label}, {hor_label}")
    ax.set_ylabel("NIU")
    ax.set_xlabel("")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _add_note(ax, variable, horizon)

    fig.subplots_adjust(bottom=0.25)
    _save_fig(fig, f"niu_{variable}_{horizon}")


def plot_ac_timeseries(variable, horizon):
    """AC time series with mean +/- 1 SE band."""
    path = config.RESULTS_DIR / f"aggregate_{variable}_{horizon}_ac.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, parse_dates=["Date"])

    _setup_style()
    fig, ax = plt.subplots()

    color_ac = config.COLORS[variable]
    mean_col = "AC_mean" if "AC_mean" in df.columns else "ACI_mean"
    lo_col = "AC_se_lo" if "AC_se_lo" in df.columns else "ACI_se_lo"
    hi_col = "AC_se_hi" if "AC_se_hi" in df.columns else "ACI_se_hi"

    ax.fill_between(df["Date"], df[lo_col], df[hi_col],
                     alpha=0.25, color=color_ac, label="Mean +/- 1 SE")
    ax.plot(df["Date"], df[mean_col], color=color_ac, linewidth=2,
            label="Mean AC")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Asymmetry Coherence (AC)")
    ax.set_ylim(-1, 1)

    var_label = config.VARIABLE_LABELS[variable]
    hor_label = config.HORIZON_LABELS[horizon]
    ax.set_title(f"Asymmetry Coherence — {var_label}, {hor_label}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")

    _add_note(ax, variable, horizon)
    fig.subplots_adjust(bottom=0.25)
    _save_fig(fig, f"ac_{variable}_{horizon}")


def plot_niu_vs_raw_variance(variable, horizon):
    """Compare NIU (normalized) vs raw variance to show normalization effect."""
    path = config.RESULTS_DIR / f"aggregate_{variable}_{horizon}_niu.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, parse_dates=["Date"])

    _setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    color = config.COLORS[variable]

    ax1.plot(df["Date"], df["Variance_mean"], color=color, linewidth=2)
    ax1.set_ylabel("Raw Variance (mean)")
    ax1.set_title("Raw Variance")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2.fill_between(df["Date"], df["NIU_se_lo"], df["NIU_se_hi"],
                      alpha=0.25, color=color)
    ax2.plot(df["Date"], df["NIU_mean"], color=color, linewidth=2)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("NIU (mean)")
    ax2.set_title("Normalized Uncertainty (distance-adjusted)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    var_label = config.VARIABLE_LABELS[variable]
    hor_label = config.HORIZON_LABELS[horizon]
    fig.suptitle(f"Normalization Effect — {var_label}, {hor_label}", fontsize=14, y=1.01)

    _add_note(ax2, variable, horizon)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    _save_fig(fig, f"niu_vs_variance_{variable}_{horizon}")


def plot_multi_horizon(variable):
    """Overlay NIU at different horizons for one variable."""
    _setup_style()
    fig, ax = plt.subplots()

    horizon_colors = {"1Y": "#1f77b4", "2Y": "#ff7f0e", "5Y": "#d62728"}
    found = False

    for hor in config.HORIZONS:
        path = config.RESULTS_DIR / f"aggregate_{variable}_{hor}_niu.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        ax.plot(df["Date"], df["NIU_mean"],
                color=horizon_colors.get(hor, "gray"),
                label=config.HORIZON_LABELS[hor], linewidth=1.8)
        found = True

    if not found:
        plt.close(fig)
        return

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    var_label = config.VARIABLE_LABELS[variable]
    ax.set_title(f"NIU Across Horizons — {var_label}")
    ax.set_ylabel("NIU (mean)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _add_note(ax, variable, "1Y")

    fig.subplots_adjust(bottom=0.25)
    _save_fig(fig, f"niu_multi_horizon_{variable}")


def plot_multi_variable(horizon):
    """Overlay NIU across different variables for one horizon."""
    _setup_style()
    fig, ax = plt.subplots()
    found = False

    for var in config.VARIABLES:
        path = config.RESULTS_DIR / f"aggregate_{var}_{horizon}_niu.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        ax.plot(df["Date"], df["NIU_mean"],
                color=config.COLORS[var],
                label=config.VARIABLE_LABELS[var], linewidth=1.8)
        found = True

    if not found:
        plt.close(fig)
        return

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    hor_label = config.HORIZON_LABELS[horizon]
    ax.set_title(f"NIU Across Variables — {hor_label}")
    ax.set_ylabel("NIU (mean)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    note = (
        "Source: ECB Survey of Professional Forecasters.  "
        f'Methodology: Vansteenberghe (2026), '
        f'"Uncertain and Asymmetric Forecasts", Banque de France WP.\n'
        "Dates = survey formation date."
    )
    ax.annotate(note, xy=(0, -0.18), xycoords="axes fraction",
                fontsize=7, color="gray", va="top", style="italic")

    fig.subplots_adjust(bottom=0.25)
    _save_fig(fig, f"niu_multi_variable_{horizon}")


def plot_ac_multi_variable(horizon):
    """Overlay AC across different variables for one horizon."""
    _setup_style()
    fig, ax = plt.subplots()
    found = False

    for var in config.VARIABLES:
        path = config.RESULTS_DIR / f"aggregate_{var}_{horizon}_ac.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        mean_col = "AC_mean" if "AC_mean" in df.columns else "ACI_mean"
        ax.plot(df["Date"], df[mean_col],
                color=config.COLORS[var],
                label=config.VARIABLE_LABELS[var], linewidth=1.8)
        found = True

    if not found:
        plt.close(fig)
        return

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    hor_label = config.HORIZON_LABELS[horizon]
    ax.set_title(f"Asymmetry Coherence Across Variables — {hor_label}")
    ax.set_ylabel("AC")
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    note = (
        "Source: ECB Survey of Professional Forecasters.  "
        f'Methodology: Vansteenberghe (2026), '
        f'"Uncertain and Asymmetric Forecasts", Banque de France WP.\n'
        "Dates = survey formation date."
    )
    ax.annotate(note, xy=(0, -0.18), xycoords="axes fraction",
                fontsize=7, color="gray", va="top", style="italic")

    fig.subplots_adjust(bottom=0.25)
    _save_fig(fig, f"ac_multi_variable_{horizon}")


# ============================================================================
# Main
# ============================================================================

def generate_all(variables=None, horizons=None):
    """Generate all figures."""
    variables = variables or config.VARIABLES
    horizons = horizons or config.HORIZONS

    print("Generating figures ...")

    # Per variable x horizon
    for var in variables:
        for hor in horizons:
            plot_niu_timeseries(var, hor)
            plot_ac_timeseries(var, hor)
            plot_niu_vs_raw_variance(var, hor)

    # Multi-horizon (per variable)
    for var in variables:
        plot_multi_horizon(var)

    # Multi-variable (per horizon)
    for hor in horizons:
        plot_multi_variable(hor)
        plot_ac_multi_variable(hor)

    print("All figures generated.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2:
        generate_all(variables=[args[0]], horizons=[args[1]])
    elif len(args) == 1:
        generate_all(variables=[args[0]])
    else:
        generate_all()
