"""
10_tail_proba.py -- Tail probabilities: P(X <= low) and P(X >= high),
their directional asymmetry dP = P_high - P_low, plus side-by-side
comparison with the Asymmetry Coherence (AC) series.

Motivation
----------
The cumulative-probability tails are a classical asymmetry measure used
in ECB communication.  For each forecaster's SPD and at each survey
round, we compute:

    P_low_t  = P(X <= low_threshold_t)   (downside-scenario probability)
    P_high_t = P(X >= high_threshold_t)  (upside-scenario probability)
    dP_t     = P_high_t - P_low_t        (signed asymmetry, in pp)

Threshold resolution is mode-driven via ``config.TAIL_THRESHOLDS``:

    * mode="absolute" -- fixed thresholds in the variable's units.
      Inflation/core default to (1%, 3%): the ECB 2% target +/- 1pp.

    * mode="target_relative" -- thresholds drift with the same target
      the AC step uses (``utils.get_target(variable, date)``).
      GDP defaults to (mu*_t - 1pp, mu*_t + 1pp); with the linear
      potential-growth interpolation the cutoffs drift from
      (1.3%, 3.3%) in 1999 to (0%, 2%) in 2026.  This keeps dP and
      AC anchored on the same notion of "normal" growth.

dP has the same sign convention as AC: positive when the upside tail
dominates, negative when the downside tail dominates, zero when the two
tails balance.  Comparing the two series isolates regimes where AC's
coherence weighting (level x skewness) and the raw probability gap
agree, and regimes where they diverge.

Two cross-sectional aggregates are reported:

    1. Mean across forecasters of P_low_i, P_high_i, dP_i with
       ``mean +/- 1 SE`` bands -- the headline series.
    2. The same quantities computed directly on the cross-sectional
       average SPD.  By linearity of the binned cumulative operator,
       this equals the mean of individual probabilities -- shown as a
       thin dashed line on the figure as a sanity check.

Outputs (per variable x horizon)
--------------------------------
    data/results/individual_{variable}_{horizon}_tailproba.csv
    data/results/aggregate_{variable}_{horizon}_tailproba.csv

    figures/tailproba_{variable}_{horizon}.png
        Both tails (P_low, P_high) with confidence bands, plus the
        avg-SPD probabilities as thin dashed lines.

    figures/ac_vs_tailproba_{variable}_{horizon}.png
        Two stacked subplots sharing the x-axis:
            top   -- AC mean +/- 1 SE (from step ``ac``)
            bottom -- P_low and P_high with confidence bands

    figures/delta_p_{variable}_{horizon}.png
        Stand-alone time series of dP = P_high - P_low with the
        ``mean +/- 1 SE`` band and the avg-SPD overlay.

    figures/ac_vs_dp_{variable}_{horizon}.png
        Twin-axis overlay: AC on the left axis and dP on the right
        axis, with their zero lines aligned.  Survey dates where AC
        and dP disagree on sign (both meaningfully non-zero) are
        highlighted with a faint vertical grey band.

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
    compute_bin_edges, cumulative_proba, figure_note, get_target,
)


# ============================================================================
# Bin-regime helper (mirrors steps 03 / 09)
# ============================================================================

def _bin_config_for_date(variable, date):
    """Return (labels, edges) for the regime active at ``date``."""
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


# ============================================================================
# Threshold resolution
# ============================================================================

def _resolve_thresholds(variable, date):
    """
    Return ``(low_t, high_t)`` for ``variable`` at ``date``.

    * mode="absolute"        -> constant ``(spec["low"], spec["high"])``.
    * mode="target_relative" -> ``(mu*_t + low_offset, mu*_t + high_offset)``
                                 where ``mu*_t = get_target(variable, date)``
                                 is the same target used by the AC step.
    """
    spec = config.TAIL_THRESHOLDS.get(variable)
    if spec is None:
        return None, None
    mode = spec.get("mode", "absolute")
    if mode == "absolute":
        return float(spec["low"]), float(spec["high"])
    if mode == "target_relative":
        target = float(get_target(variable, date))
        return (
            target + float(spec["low_offset"]),
            target + float(spec["high_offset"]),
        )
    raise ValueError(f"Unknown TAIL_THRESHOLDS mode {mode!r} for {variable!r}")


def _signed(offset):
    """Return (sign_str, abs_offset) for compact legend formatting."""
    if offset >= 0:
        return "+", float(offset)
    return "-", float(-offset)


def _tail_legend_label(variable, tail):
    """
    Matplotlib math-mode label for one tail (low or high).

        absolute        -> "$P(X \\leq 1\\%)$"
        target_relative -> "$P(X \\leq \\mu^*_t - 1\\,\\mathrm{pp})$"
    """
    spec = config.TAIL_THRESHOLDS[variable]
    mode = spec.get("mode", "absolute")
    op = r"\leq" if tail == "low" else r"\geq"
    if mode == "absolute":
        val = spec["low" if tail == "low" else "high"]
        return f"$P(X {op} {val:g}\\%)$"
    sign, mag = _signed(spec[f"{tail}_offset"])
    return f"$P(X {op} \\mu^*_t {sign} {mag:g}\\,\\mathrm{{pp}})$"


def _dp_legend_label(variable):
    """Matplotlib math-mode label for the dP series."""
    spec = config.TAIL_THRESHOLDS[variable]
    mode = spec.get("mode", "absolute")
    if mode == "absolute":
        return (
            f"$\\Delta P = P(X \\geq {spec['high']:g}\\%)"
            f" - P(X \\leq {spec['low']:g}\\%)$"
        )
    sgn_lo, mag_lo = _signed(spec["low_offset"])
    sgn_hi, mag_hi = _signed(spec["high_offset"])
    return (
        f"$\\Delta P = P(X \\geq \\mu^*_t {sgn_hi} {mag_hi:g}\\,\\mathrm{{pp}})"
        f" - P(X \\leq \\mu^*_t {sgn_lo} {mag_lo:g}\\,\\mathrm{{pp}})$"
    )


def _threshold_subtitle(variable):
    """Short caption describing the active threshold rule."""
    spec = config.TAIL_THRESHOLDS[variable]
    if spec.get("mode", "absolute") == "absolute":
        return f"thresholds: {spec['low']:g}% / {spec['high']:g}%"
    sgn_lo, mag_lo = _signed(spec["low_offset"])
    sgn_hi, mag_hi = _signed(spec["high_offset"])
    return (
        f"thresholds: $\\mu^*_t$ {sgn_lo} {mag_lo:g}pp  /  "
        f"$\\mu^*_t$ {sgn_hi} {mag_hi:g}pp  (time-varying, same target as AC)"
    )


# ============================================================================
# Computations
# ============================================================================

def compute_individual_tail_proba(variable, horizon):
    """
    Per-forecaster cumulative probabilities at each survey round.

    Thresholds are resolved per Date via ``_resolve_thresholds`` and
    therefore drift over time when ``mode="target_relative"``.

    Returns DataFrame with columns:
        Date, FCT_SOURCE, POINT, P_low, P_high, dP,
        low_threshold, high_threshold
    """
    panel_path = config.PANELS_DIR / f"panel_{variable}_{horizon}.csv"
    if not panel_path.exists():
        print(f"  Panel not found: {panel_path}. Run step 'panels' first.")
        return pd.DataFrame()

    df = pd.read_csv(panel_path)
    df["Date"] = pd.to_datetime(df["Date"])

    rows = []
    for _, r in df.iterrows():
        date = r["Date"]
        low_t, high_t = _resolve_thresholds(variable, date)
        if low_t is None:
            continue
        labels, edges = _bin_config_for_date(variable, date)
        if not labels:
            continue

        probs = np.array(
            [float(r.get(lbl, 0.0)) if pd.notna(r.get(lbl, 0.0)) else 0.0
             for lbl in labels],
            dtype=float,
        )
        total = probs.sum()
        if total <= 0:
            continue
        probs = probs / total * 100.0

        p_low = cumulative_proba(probs, edges, low_t, "lower")
        p_high = cumulative_proba(probs, edges, high_t, "upper")
        if np.isfinite(p_low) and np.isfinite(p_high):
            dp = p_high - p_low
        else:
            dp = np.nan

        rows.append({
            "Date": date,
            "FCT_SOURCE": r.get("FCT_SOURCE", np.nan),
            "POINT": r.get("POINT", np.nan),
            "P_low": p_low,
            "P_high": p_high,
            "dP": dp,
            "low_threshold": low_t,
            "high_threshold": high_t,
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.sort_values(["Date", "FCT_SOURCE"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def compute_avg_spd_tail_proba(variable, horizon):
    """
    Cumulative probabilities computed on the cross-sectional average SPD.

    By linearity of the binned cumulative operator this equals the mean
    of individual probabilities -- the column is kept as a sanity check
    and as a "smoothed" alternative when individual SPDs are sparse.

    Thresholds are resolved per Date via ``_resolve_thresholds``.

    Returns DataFrame with columns:
        Date, P_low_avg_spd, P_high_avg_spd, dP_avg_spd
    """
    panel_path = config.PANELS_DIR / f"panel_{variable}_{horizon}.csv"
    if not panel_path.exists():
        return pd.DataFrame()

    panel = pd.read_csv(panel_path)
    panel["Date"] = pd.to_datetime(panel["Date"])

    meta_cols = {"Date", "FCT_SOURCE", "POINT"}
    bin_cols = [c for c in panel.columns if c not in meta_cols]
    panel[bin_cols] = panel[bin_cols].fillna(0.0)
    avg_spd = panel.groupby("Date")[bin_cols].mean()

    rows = []
    for date, row in avg_spd.iterrows():
        low_t, high_t = _resolve_thresholds(variable, date)
        if low_t is None:
            continue
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

        p_low = cumulative_proba(probs, edges, low_t, "lower")
        p_high = cumulative_proba(probs, edges, high_t, "upper")
        rows.append({
            "Date": date,
            "P_low_avg_spd": p_low,
            "P_high_avg_spd": p_high,
            "dP_avg_spd": (
                p_high - p_low
                if np.isfinite(p_low) and np.isfinite(p_high)
                else np.nan
            ),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def aggregate_tail_proba(individual_df):
    """
    Aggregate to date-level: mean, std, SE bands, and n_forecasters.

    Returns DataFrame indexed by Date with columns:
        Date, P_low_mean, P_low_se_lo, P_low_se_hi,
        P_high_mean, P_high_se_lo, P_high_se_hi,
        n_forecasters.
    """
    if individual_df.empty:
        return pd.DataFrame()

    agg = (
        individual_df.dropna(subset=["P_low", "P_high", "dP"])
        .groupby("Date")
        .agg(
            P_low_mean=("P_low", "mean"),
            P_low_std=("P_low", "std"),
            P_high_mean=("P_high", "mean"),
            P_high_std=("P_high", "std"),
            dP_mean=("dP", "mean"),
            dP_std=("dP", "std"),
            low_threshold=("low_threshold", "first"),
            high_threshold=("high_threshold", "first"),
            n_forecasters=("FCT_SOURCE", "count"),
        )
        .reset_index()
    )

    se_low = agg["P_low_std"] / np.sqrt(agg["n_forecasters"])
    se_high = agg["P_high_std"] / np.sqrt(agg["n_forecasters"])
    se_dp = agg["dP_std"] / np.sqrt(agg["n_forecasters"])
    agg["P_low_se_lo"] = agg["P_low_mean"] - se_low
    agg["P_low_se_hi"] = agg["P_low_mean"] + se_low
    agg["P_high_se_lo"] = agg["P_high_mean"] - se_high
    agg["P_high_se_hi"] = agg["P_high_mean"] + se_high
    agg["dP_se_lo"] = agg["dP_mean"] - se_dp
    agg["dP_se_hi"] = agg["dP_mean"] + se_dp

    agg.sort_values("Date", inplace=True)
    agg.reset_index(drop=True, inplace=True)
    return agg


# ============================================================================
# Plots
# ============================================================================

LOW_COLOR = "#1f77b4"   # blue:   downside-scenario probability
HIGH_COLOR = "#d62728"  # red:    upside-scenario probability
DP_COLOR = "#9467bd"    # purple: directional asymmetry dP = P_high - P_low

# Sign-disagreement thresholds for the AC-vs-dP overlay (avoid noise
# around zero where both series are essentially flat).
_AC_NOISE = 0.02       # |AC|   below this counts as "near zero"
_DP_NOISE = 1.0        # |dP|   in pp below this counts as "near zero"


def _plot_tail_proba(agg, variable, horizon):
    """Single-panel figure: both tails with bands + avg-SPD overlay."""
    if agg.empty:
        return

    var_label = config.VARIABLE_LABELS.get(variable, variable)
    hor_label = config.HORIZON_LABELS.get(horizon, horizon)
    low_lbl = _tail_legend_label(variable, "low")
    high_lbl = _tail_legend_label(variable, "high")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(agg["Date"], agg["P_low_se_lo"], agg["P_low_se_hi"],
                    alpha=0.20, color=LOW_COLOR)
    ax.plot(agg["Date"], agg["P_low_mean"], color=LOW_COLOR, linewidth=2.0,
            label=f"{low_lbl} - mean of individuals")

    ax.fill_between(agg["Date"], agg["P_high_se_lo"], agg["P_high_se_hi"],
                    alpha=0.20, color=HIGH_COLOR)
    ax.plot(agg["Date"], agg["P_high_mean"], color=HIGH_COLOR, linewidth=2.0,
            label=f"{high_lbl} - mean of individuals")

    if "P_low_avg_spd" in agg.columns:
        ax.plot(agg["Date"], agg["P_low_avg_spd"], color=LOW_COLOR,
                linestyle="--", linewidth=1.0, alpha=0.85,
                label=f"{low_lbl} - on average SPD")
    if "P_high_avg_spd" in agg.columns:
        ax.plot(agg["Date"], agg["P_high_avg_spd"], color=HIGH_COLOR,
                linestyle="--", linewidth=1.0, alpha=0.85,
                label=f"{high_lbl} - on average SPD")

    ax.set_title(
        f"Tail probabilities -- {var_label}, {hor_label}\n"
        f"{_threshold_subtitle(variable)}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, linestyle="--", alpha=0.4)

    note = figure_note(variable, horizon)
    ax.annotate(note, xy=(0, -0.20), xycoords="axes fraction",
                fontsize=7, color="gray", va="top", style="italic")
    fig.subplots_adjust(bottom=0.25)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / f"tailproba_{variable}_{horizon}.png"
    fig.savefig(out_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _plot_ac_vs_tails(agg, variable, horizon):
    """
    2x1 comparison figure: AC on top, tail probabilities on bottom.

    The two subplots share the x-axis so disagreement episodes between
    the AC directional signal and the cumulative tail probabilities are
    easy to spot visually.
    """
    ac_path = config.RESULTS_DIR / f"aggregate_{variable}_{horizon}_ac.csv"
    if not ac_path.exists():
        print(f"  No AC results for {variable}/{horizon}, "
              f"skipping comparison plot.")
        return
    if agg.empty:
        return

    ac = pd.read_csv(ac_path, parse_dates=["Date"])
    if ac.empty:
        return

    ac_col = "AC_mean" if "AC_mean" in ac.columns else "AC"
    if ac_col not in ac.columns:
        print(f"  AC column missing in {ac_path.name}, skipping.")
        return

    var_label = config.VARIABLE_LABELS.get(variable, variable)
    hor_label = config.HORIZON_LABELS.get(horizon, horizon)
    var_color = config.COLORS.get(variable, "#333333")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # --- Top: AC
    ax = axes[0]
    if "AC_se_lo" in ac.columns and "AC_se_hi" in ac.columns:
        ax.fill_between(ac["Date"], ac["AC_se_lo"], ac["AC_se_hi"],
                        alpha=0.20, color=var_color)
    ax.plot(ac["Date"], ac[ac_col], color=var_color, linewidth=2.0,
            label="AC (mean +/- 1 SE)")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(
        f"AC vs. tail probabilities -- {var_label}, {hor_label}\n"
        f"{_threshold_subtitle(variable)}"
    )
    ax.set_ylabel("Asymmetry Coherence (AC)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- Bottom: tail probabilities
    ax = axes[1]
    low_lbl = _tail_legend_label(variable, "low")
    high_lbl = _tail_legend_label(variable, "high")
    ax.fill_between(agg["Date"], agg["P_low_se_lo"], agg["P_low_se_hi"],
                    alpha=0.20, color=LOW_COLOR)
    ax.plot(agg["Date"], agg["P_low_mean"], color=LOW_COLOR, linewidth=2.0,
            label=low_lbl)
    ax.fill_between(agg["Date"], agg["P_high_se_lo"], agg["P_high_se_hi"],
                    alpha=0.20, color=HIGH_COLOR)
    ax.plot(agg["Date"], agg["P_high_mean"], color=HIGH_COLOR, linewidth=2.0,
            label=high_lbl)
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, linestyle="--", alpha=0.4)

    note = figure_note(variable, horizon)
    fig.text(0.02, 0.005, note, ha="left", fontsize=7,
             color="gray", style="italic", wrap=True)

    fig.tight_layout(rect=(0, 0.03, 1, 1))

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / f"ac_vs_tailproba_{variable}_{horizon}.png"
    fig.savefig(out_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _plot_delta_p(agg, variable, horizon):
    """Stand-alone time series of dP = P_high - P_low with band."""
    if agg.empty:
        return

    var_label = config.VARIABLE_LABELS.get(variable, variable)
    hor_label = config.HORIZON_LABELS.get(horizon, horizon)
    dp_lbl = _dp_legend_label(variable)

    fig, ax = plt.subplots(figsize=(10, 5))

    if "dP_se_lo" in agg.columns and "dP_se_hi" in agg.columns:
        ax.fill_between(agg["Date"], agg["dP_se_lo"], agg["dP_se_hi"],
                        alpha=0.22, color=DP_COLOR)
    ax.plot(
        agg["Date"], agg["dP_mean"], color=DP_COLOR, linewidth=2.0,
        label=f"{dp_lbl}  -  mean of individuals",
    )
    if "dP_avg_spd" in agg.columns:
        ax.plot(agg["Date"], agg["dP_avg_spd"], color=DP_COLOR,
                linestyle="--", linewidth=1.0, alpha=0.85,
                label="$\\Delta P$ on average SPD")

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(
        f"Tail-probability asymmetry $\\Delta P$ -- {var_label}, {hor_label}\n"
        f"{_threshold_subtitle(variable)}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("$\\Delta P$ (percentage points)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, linestyle="--", alpha=0.4)

    note = figure_note(variable, horizon)
    ax.annotate(note, xy=(0, -0.20), xycoords="axes fraction",
                fontsize=7, color="gray", va="top", style="italic")
    fig.subplots_adjust(bottom=0.25)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / f"delta_p_{variable}_{horizon}.png"
    fig.savefig(out_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _plot_ac_vs_delta_p(agg, variable, horizon):
    """
    Twin-axis overlay: AC on the left, dP on the right.

    Both axes are centered on zero so the natural sign comparison is
    immediate; survey dates where AC and dP disagree on sign (both
    meaningfully non-zero per ``_AC_NOISE`` / ``_DP_NOISE``) are
    highlighted with a faint vertical grey band.
    """
    ac_path = config.RESULTS_DIR / f"aggregate_{variable}_{horizon}_ac.csv"
    if not ac_path.exists() or agg.empty:
        return
    ac = pd.read_csv(ac_path, parse_dates=["Date"])
    if ac.empty:
        return
    ac_col = "AC_mean" if "AC_mean" in ac.columns else "AC"
    if ac_col not in ac.columns:
        return

    merged = pd.merge(
        ac[[c for c in ["Date", ac_col, "AC_se_lo", "AC_se_hi"] if c in ac.columns]],
        agg[["Date", "dP_mean", "dP_se_lo", "dP_se_hi"]],
        on="Date", how="inner",
    )
    if merged.empty:
        return

    var_label = config.VARIABLE_LABELS.get(variable, variable)
    hor_label = config.HORIZON_LABELS.get(horizon, horizon)
    var_color = config.COLORS.get(variable, "#333333")

    disagree_mask = (
        (merged[ac_col].abs() > _AC_NOISE)
        & (merged["dP_mean"].abs() > _DP_NOISE)
        & (np.sign(merged[ac_col]) != np.sign(merged["dP_mean"]))
    )
    n_disagree = int(disagree_mask.sum())
    n_total = len(merged)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    # Sign-disagreement shading: ~3-month-wide band per quarterly date
    half_band = pd.Timedelta(days=45)
    for ts in merged.loc[disagree_mask, "Date"]:
        ax1.axvspan(ts - half_band, ts + half_band,
                    alpha=0.18, color="dimgray", zorder=0,
                    linewidth=0)

    # --- AC on the left axis ---
    if "AC_se_lo" in merged.columns and "AC_se_hi" in merged.columns:
        ax1.fill_between(merged["Date"], merged["AC_se_lo"], merged["AC_se_hi"],
                         alpha=0.22, color=var_color)
    ax1.plot(merged["Date"], merged[ac_col], color=var_color, linewidth=2.0,
             label="AC (left axis)")
    ax1.axhline(0.0, color=var_color, linestyle="--", linewidth=0.8, alpha=0.4)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Asymmetry Coherence (AC)", color=var_color)
    ax1.tick_params(axis="y", labelcolor=var_color)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.grid(True, linestyle="--", alpha=0.35)

    # --- dP on the right axis ---
    ax2 = ax1.twinx()
    ax2.fill_between(merged["Date"], merged["dP_se_lo"], merged["dP_se_hi"],
                     alpha=0.17, color=DP_COLOR)
    ax2.plot(
        merged["Date"], merged["dP_mean"], color=DP_COLOR, linewidth=2.0,
        label=f"{_dp_legend_label(variable)} (right axis)",
    )
    ax2.set_ylabel("$\\Delta P$ (percentage points)", color=DP_COLOR)
    ax2.tick_params(axis="y", labelcolor=DP_COLOR)

    # Align both zero lines: make each y-limit symmetric around zero
    ac_max = float(np.nanmax(np.abs([
        merged[ac_col].min(), merged[ac_col].max(),
        merged.get("AC_se_lo", merged[ac_col]).min(),
        merged.get("AC_se_hi", merged[ac_col]).max(),
    ])))
    dp_max = float(np.nanmax(np.abs([
        merged["dP_se_lo"].min(), merged["dP_se_hi"].max(),
    ])))
    pad = 1.10
    if ac_max > 0:
        ax1.set_ylim(-ac_max * pad, ac_max * pad)
    if dp_max > 0:
        ax2.set_ylim(-dp_max * pad, dp_max * pad)

    ax1.set_title(
        f"AC vs $\\Delta P$  --  {var_label}, {hor_label}\n"
        f"{_threshold_subtitle(variable)}   "
        f"|  sign disagreement: {n_disagree}/{n_total} dates shaded"
    )

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    note = figure_note(variable, horizon)
    ax1.annotate(note, xy=(0, -0.18), xycoords="axes fraction",
                 fontsize=7, color="gray", va="top", style="italic")
    fig.subplots_adjust(bottom=0.25)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / f"ac_vs_dp_{variable}_{horizon}.png"
    fig.savefig(out_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}  "
          f"(sign disagreement on {n_disagree}/{n_total} dates)")


# ============================================================================
# Top-level orchestrator
# ============================================================================

def process_all(variables=None, horizons=None):
    """Compute and save tail probabilities + figures for every combo."""
    variables = variables or config.VARIABLES
    horizons = horizons or config.HORIZONS
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for var in variables:
        spec = config.TAIL_THRESHOLDS.get(var)
        if spec is None:
            print(f"  No TAIL_THRESHOLDS configured for {var}, skipping.")
            continue
        mode = spec.get("mode", "absolute")
        rule = (
            f"abs(low={spec['low']:g}, high={spec['high']:g})"
            if mode == "absolute"
            else (
                f"target-relative(low={spec['low_offset']:+g}pp, "
                f"high={spec['high_offset']:+g}pp)"
            )
        )

        for hor in horizons:
            print(f"Tail probabilities: {var} / {hor}  [{rule}] ...")

            indiv = compute_individual_tail_proba(var, hor)
            if indiv.empty:
                print("  -> empty, skipping.")
                continue

            indiv_path = (
                config.RESULTS_DIR
                / f"individual_{var}_{hor}_tailproba.csv"
            )
            indiv.to_csv(indiv_path, index=False)

            agg = aggregate_tail_proba(indiv)
            avg = compute_avg_spd_tail_proba(var, hor)
            if not avg.empty:
                agg = agg.merge(avg, on="Date", how="left")

            agg_path = (
                config.RESULTS_DIR
                / f"aggregate_{var}_{hor}_tailproba.csv"
            )
            agg.to_csv(agg_path, index=False)
            print(f"  -> {len(agg)} dates -> {agg_path.name}")

            _plot_tail_proba(agg, var, hor)
            _plot_ac_vs_tails(agg, var, hor)
            _plot_delta_p(agg, var, hor)
            _plot_ac_vs_delta_p(agg, var, hor)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2:
        process_all(variables=[args[0]], horizons=[args[1]])
    elif len(args) == 1:
        process_all(variables=[args[0]])
    else:
        process_all()
