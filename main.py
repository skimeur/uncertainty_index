"""
main.py -- Uncertainty Index Pipeline Orchestrator

Computes Normalized Uncertainty (NU) and Asymmetry Coherence (AC) indicators
from ECB Survey of Professional Forecasters (SPF) individual probability
distributions, for HICP inflation and real GDP growth at the 1-year, 2-year,
and 5-year-ahead horizons.

Author: Eric Vansteenberghe
Reference:
    Vansteenberghe, E. (2026). "Uncertain and Asymmetric Forecasts."
    Banque de France Working Paper.

Theory
------
Normalized Uncertainty (NU) corrects raw forecast variance for the mechanical
widening that occurs when expectations drift from a policy anchor.  The raw
variance of a subjective probability distribution (SPD) satisfies, to first
order, Var ~ a + b * |mu - mu*|, where mu* is the anchor (e.g. the ECB's
2 % inflation target).  NU divides out this structural component:

    NU_{i,t} = sqrt(Var_{i,t}) / sqrt(a + b * |mu_{i,t} - mu*|)

With unit calibration (a = b = 1) the measure is sample-free and correlates
> 0.99 with in-sample estimates.  NU = 1 means all dispersion is explained by
distance from the anchor; NU > 1 signals excess uncertainty; NU < 1 signals
compressed uncertainty.

Asymmetry Coherence (AC) extracts a directional-risk signal from the third
moment of the SPD.  It pairs Bowley skewness (a robust quantile-based
asymmetry measure) with the median's deviation from target, weighting their
average by a coherence term that rewards sign agreement:

    AC_t = ((Q_tilde + A_tilde) / 2) * ((1 + Q_tilde * A_tilde) / 2)

where Q_tilde and A_tilde are tanh-normalized (IQR-scaled) versions of the
median deviation and mean Bowley skewness.  AC lies in (-1, 1): positive
values indicate coherent upside risk; negative values indicate coherent
downside risk.

Pipeline steps
--------------
    download    Download ECB-SPF individual forecasts
    realized    Download realized macro series (HICP, GDP)
    panels      Parse raw CSV files into clean variable x horizon panels
    niu         Compute Normalized Uncertainty at individual and aggregate level
    ac          Compute Asymmetry Coherence at individual and aggregate level
    merge       Merge horizons into a single NU index per variable
    plots       Generate publication-ready figures
    diagnostics Run variance diagnostics and merged publication plots

Usage:
    python main.py              # full pipeline
    python main.py download     # single step
    python main.py niu ac plots # multiple steps
"""
import sys
import os
import importlib

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def _import(module_path):
    """Import a module by dotted path, handling numeric prefixes."""
    return importlib.import_module(module_path)


def run_download():
    mod = _import("code.01_download_data")
    mod.download_spf()


def run_realized():
    mod = _import("code.05_download_realized")
    mod.download_all()


def run_panels():
    mod = _import("code.02_prepare_panels")
    mod.prepare_all()


def run_niu():
    mod = _import("code.03_compute_niu")
    mod.process_all()


def run_ac():
    mod = _import("code.04_compute_ac")
    mod.process_all()


def run_plots():
    mod = _import("code.06_plot_results")
    mod.generate_all()


def run_merge():
    mod = _import("code.07_merge_horizons")
    mod.process_all()


def run_diagnostics():
    mod = _import("code.08_diagnostics")
    mod.run_all()


STEPS = {
    "download": run_download,
    "realized": run_realized,
    "panels": run_panels,
    "niu": run_niu,
    "ac": run_ac,
    "merge": run_merge,
    "plots": run_plots,
    "diagnostics": run_diagnostics,
}


def main():
    args = sys.argv[1:]

    if args:
        for step_name in args:
            if step_name in STEPS:
                print(f"\n{'='*60}")
                print(f"  Step: {step_name}")
                print(f"{'='*60}")
                STEPS[step_name]()
            else:
                print(f"Unknown step: {step_name}. Available: {list(STEPS.keys())}")
                sys.exit(1)
    else:
        # Full pipeline
        print("=" * 60)
        print("  Uncertainty Index — Full Pipeline")
        print(f"  {config.CITATION}")
        print("=" * 60)

        for step_name, step_fn in STEPS.items():
            print(f"\n{'='*60}")
            print(f"  Step: {step_name}")
            print(f"{'='*60}")
            step_fn()

        print(f"\n{'='*60}")
        print("  Pipeline complete.")
        print(f"  Results in: {config.RESULTS_DIR}")
        print(f"  Figures in: {config.FIGURES_DIR}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
