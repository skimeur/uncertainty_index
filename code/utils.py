"""
Shared utility functions for the Uncertainty Index project.

Provides SPD moment computation, percentile interpolation, Normalized
Uncertainty (NU), Asymmetry Coherence (AC), entropy measures, and
time-varying target helpers.

Author: Eric Vansteenberghe
Reference:
    Vansteenberghe, E. (2026). "Uncertain and Asymmetric Forecasts."
    Banque de France Working Paper.
"""
import numpy as np
import pandas as pd


def compute_bin_edges(bin_defs, tail_min=-20.0, tail_max=20.0):
    """
    Convert canonical bin definitions [(label, lo, hi), ...] into
    concrete (lo, hi) edges, replacing None tails with tail_min/tail_max.

    Returns: (labels, edges)
        labels: list of str
        edges: list of (float, float)
    """
    labels = []
    edges = []
    for label, lo, hi in bin_defs:
        lo_val = tail_min if lo is None else lo
        hi_val = tail_max if hi is None else hi
        labels.append(label)
        edges.append((lo_val, hi_val))
    return labels, edges


def midpoints_from_edges(edges):
    """Return array of midpoints for each (lo, hi) bin edge pair."""
    return np.array([(lo + hi) / 2.0 for lo, hi in edges], dtype=float)


def normalize_probs(probs):
    """Normalize probability vector to sum to 100. Returns NaN array if sum <= 0."""
    probs = np.where(np.isfinite(probs), probs, 0.0)
    total = probs.sum()
    if total <= 0:
        return np.full_like(probs, np.nan)
    return probs / total * 100.0


def compute_spd_moments(probs_pct, edges):
    """
    Compute mean and variance from a discrete SPD (probability histogram).

    Parameters
    ----------
    probs_pct : array-like, probabilities in % (should sum to ~100)
    edges : list of (lo, hi) tuples

    Returns
    -------
    mean, variance : float
    """
    probs_pct = np.asarray(probs_pct, dtype=float)
    probs_pct = np.where(np.isfinite(probs_pct), probs_pct, 0.0)
    total = probs_pct.sum()
    if total <= 0:
        return np.nan, np.nan

    mid = midpoints_from_edges(edges)
    mu = np.dot(mid, probs_pct) / total
    var = np.dot((mid - mu) ** 2, probs_pct) / total
    return float(mu), float(var)


def find_percentile(percentile, edges, cumulative_probs):
    """
    Linearly interpolate a percentile from binned CDF.

    Parameters
    ----------
    percentile : float (0–100 scale)
    edges : list of (lower, upper) tuples
    cumulative_probs : array of cumulative probabilities (0–100 scale)

    Returns
    -------
    float or None
    """
    for i, cum_prob in enumerate(cumulative_probs):
        if cum_prob >= percentile:
            lower_edge, upper_edge = edges[i]
            previous_prob = 0.0 if i == 0 else cumulative_probs[i - 1]
            denom = cum_prob - previous_prob
            if denom <= 0:
                return None
            w = (percentile - previous_prob) / denom
            return lower_edge + w * (upper_edge - lower_edge)
    return None


def bowley_skewness(q1, q2, q3):
    """
    Bowley's quantile-based skewness: (Q3 + Q1 - 2*Q2) / (Q3 - Q1).
    Returns NaN if Q3 == Q1 or any input is None/NaN.
    """
    if q1 is None or q2 is None or q3 is None:
        return np.nan
    if not (np.isfinite(q1) and np.isfinite(q2) and np.isfinite(q3)):
        return np.nan
    if q3 == q1:
        return np.nan
    return ((q3 - q2) - (q2 - q1)) / (q3 - q1)


def compute_niu(variance, mean, target, a=1.0, b=1.0, p=0.5):
    """
    Normalized Uncertainty:
        NIU = sqrt(Var) / (a + b * |mean - target|)^p

    Unit calibration (a=1, b=1, p=0.5):
        NIU = sqrt(Var) / sqrt(1 + |mean - target|)
    """
    if not np.isfinite(variance) or not np.isfinite(mean) or variance < 0:
        return np.nan
    d = abs(mean - target)
    denom = (a + b * d) ** p
    if denom <= 0:
        return np.nan
    return np.sqrt(variance) / denom


def normalized_entropy(probs_pct, K):
    """
    Normalized Shannon entropy: h(p) = H(p) / log(K).
    probs_pct should sum to ~100 (will be normalized to probabilities).
    """
    probs = np.asarray(probs_pct, dtype=float)
    probs = np.where(np.isfinite(probs), probs, 0.0)
    total = probs.sum()
    if total <= 0 or K <= 1:
        return np.nan
    pvec = probs / total
    ppos = pvec[pvec > 0]
    H = float(-np.sum(ppos * np.log(ppos)))
    return H / np.log(K)


def informativeness(probs_pct, K):
    """
    Informativeness I_{i,t}: 0 if only one bin filled, else normalized entropy.
    """
    probs = np.asarray(probs_pct, dtype=float)
    probs = np.where(np.isfinite(probs), probs, 0.0)
    total = probs.sum()
    if total <= 0:
        return 0.0
    pvec = probs / total
    S = int(np.sum(pvec > 0))
    if S <= 1:
        return 0.0
    return normalized_entropy(probs_pct, K)


def bins_filled(probs_pct):
    """Count the number of bins with non-zero probability."""
    probs = np.asarray(probs_pct, dtype=float)
    probs = np.where(np.isfinite(probs), probs, 0.0)
    return int(np.sum(probs > 0))


def iqr_scale(series):
    """Compute IQR of a series, returning 1.0 if zero or insufficient data."""
    x = series.dropna()
    if len(x) < 4:
        return 1.0
    s = float(x.quantile(0.75) - x.quantile(0.25))
    return s if s != 0 else 1.0


def compute_ac_series(q_median_series, bowley_mean_series, target):
    """
    Compute Asymmetry Coherence (AC) time series from cross-sectional
    median-of-medians (Q_t) and mean-of-Bowley (A_t).

    Parameters
    ----------
    q_median_series : pd.Series indexed by Date
    bowley_mean_series : pd.Series indexed by Date
    target : float (mu*)

    Returns
    -------
    pd.DataFrame with columns: Q_norm, A_norm, AC, coherence
    """
    Q_dev = q_median_series - target
    iqr_q = iqr_scale(Q_dev)
    iqr_a = iqr_scale(bowley_mean_series)

    Q_norm = np.tanh(Q_dev / iqr_q)
    A_norm = np.tanh(bowley_mean_series / iqr_a)

    AC = 0.5 * (Q_norm + A_norm) * 0.5 * (1.0 + Q_norm * A_norm)
    coherence = 0.5 * (1.0 + Q_norm * A_norm)

    return pd.DataFrame({
        "Q_norm": Q_norm,
        "A_norm": A_norm,
        "AC": AC,
        "coherence": coherence,
    })


_AMECO_CACHE = None

def _load_ameco():
    """Load and cache the AMECO potential growth data."""
    global _AMECO_CACHE
    if _AMECO_CACHE is not None:
        return _AMECO_CACHE
    import config as _cfg
    path = _cfg.GDP_TARGET_AMECO_PATH
    if not path.exists():
        print(f"  WARNING: AMECO file not found at {path}, falling back to linear.")
        _AMECO_CACHE = {}
        return _AMECO_CACHE
    df = pd.read_csv(path, comment="#")
    _AMECO_CACHE = dict(zip(df["Year"].astype(int), df["potential_growth_pct"].astype(float)))
    return _AMECO_CACHE


def get_target(variable, date=None, mode=None):
    """
    Return the policy anchor / structural benchmark for a variable.

    Parameters
    ----------
    variable : str
        One of 'inflation', 'core', 'gdp'.
    date : datetime-like, optional
        Survey date.  Required for the time-varying GDP target.
    mode : str, optional
        Override for GDP target mode ('linear' or 'ameco').

    Returns
    -------
    float
        Target value (%).

    Notes
    -----
    - Inflation / core: fixed at 2.0 % (ECB mandate).
    - GDP: by default, a time-varying linear interpolation from 2.3 %
      (1999) to 1.0 % (2026) tracks the secular decline in euro-area
      potential growth.  Set ``GDP_TARGET_MODE = "ameco"`` in
      ``config.py`` to use AMECO official estimates instead.
    """
    import config as _cfg

    if variable == "gdp" and getattr(_cfg, "GDP_TARGET_TIME_VARYING", False) and date is not None:
        t = pd.Timestamp(date)
        effective_mode = mode or getattr(_cfg, "GDP_TARGET_MODE", "linear")

        if effective_mode == "ameco":
            ameco = _load_ameco()
            if ameco:
                year = t.year
                if year in ameco:
                    return ameco[year]
                # Interpolate between nearest years
                years = sorted(ameco.keys())
                if year <= years[0]:
                    return ameco[years[0]]
                if year >= years[-1]:
                    return ameco[years[-1]]
                # Linear interpolation between surrounding years
                for i in range(len(years) - 1):
                    if years[i] <= year <= years[i + 1]:
                        frac = (year - years[i]) / (years[i + 1] - years[i])
                        return ameco[years[i]] + frac * (ameco[years[i + 1]] - ameco[years[i]])

        # Default: linear interpolation
        year_frac = t.year + (t.month - 1) / 12.0
        y0 = _cfg.GDP_TARGET_START_YEAR
        y1 = _cfg.GDP_TARGET_END_YEAR
        v0 = _cfg.GDP_TARGET_START_VALUE
        v1 = _cfg.GDP_TARGET_END_VALUE
        if year_frac <= y0:
            return v0
        if year_frac >= y1:
            return v1
        return v0 + (v1 - v0) * (year_frac - y0) / (y1 - y0)
    return _cfg.TARGETS[variable]


def formation_date(year, quarter):
    """
    Return the formation date (when the survey was conducted).
    Q1 → January; Q2 → April; Q3 → July; Q4 → October.
    """
    month_map = {1: 1, 2: 4, 3: 7, 4: 10}
    return pd.Timestamp(year=year, month=month_map[quarter], day=1)


def figure_note(variable, horizon, formation_dt=None):
    """
    Generate a standard figure note explaining the date convention
    and citing the working paper.
    """
    var_labels = {
        "inflation": "HICP inflation",
        "core": "core inflation",
        "gdp": "real GDP growth",
    }
    horizon_labels = {
        "CY": "current-year",
        "1Y": "one-year-ahead",
        "2Y": "two-year-ahead",
        "5Y": "five-year-ahead",
    }
    var_str = var_labels.get(variable, variable)
    hor_str = horizon_labels.get(horizon, horizon)
    note = (
        f"Source: ECB Survey of Professional Forecasters.  "
        f"Methodology: Vansteenberghe (2026), "
        f'"Uncertain and Asymmetric Forecasts", Banque de France WP.\n'
        f"Dates = survey formation date.  "
        f"Forecasters estimated {var_str} at the {hor_str} horizon.  "
        f"Band = mean +/- 1 standard error."
    )
    return note
