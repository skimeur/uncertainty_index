# Uncertainty Index: Normalized Uncertainty and Asymmetry Coherence from the ECB Survey of Professional Forecasters

**Author:** Eric Vansteenberghe

> **Citation (required):**
>
> Vansteenberghe, E. (2026). "Uncertain and Asymmetric Forecasts." *Banque de France Working Paper*.
>
> ```bibtex
> @article{vansteenberghe2026,
>   title   = {Uncertain and Asymmetric Forecasts},
>   author  = {Vansteenberghe, Eric},
>   journal = {Banque de France Working Paper},
>   year    = {2026}
> }
> ```

This repository computes **Normalized Uncertainty (NU)** and **Asymmetry Coherence (AC)** indicators from ECB Survey of Professional Forecasters (SPF) individual probability distributions, for HICP inflation and real GDP growth at the 1-year, 2-year, and 5-year-ahead horizons. It then merges these into a single cross-horizon **NIU index** per variable. All methodology follows Vansteenberghe (2026).

---

## 1. Theoretical Foundations

### 1.1 Normalized Uncertainty (NU)

Raw forecast variance is a poor measure of genuine uncertainty. When professional forecasters expect inflation (or GDP growth) far from a structural anchor such as the ECB's 2% inflation target, their subjective probability distributions (SPDs) mechanically widen, even if genuine belief imprecision is unchanged. A forecaster who expects 5% inflation faces more structural dispersion than one who expects 2%, simply because of the distance from the anchor. Raw variance therefore conflates two distinct sources of dispersion:

1. **Structural dispersion** -- the part mechanically explained by distance from the anchor.
2. **Genuine uncertainty** -- the residual, reflecting true belief imprecision.

In the ECB-SPF data, 42% of raw variance variation is explained by distance from target alone. Failing to correct for this contaminates any uncertainty measure with first-moment movements.

**Proposition 1** in Vansteenberghe (2026) proves that the conditional variance satisfies a local affine relationship:

```
Var(X_{t+1} | I_t) = a + b * |mu_t - mu*|
```

where `mu_t` is the forecast mean and `mu*` the policy anchor. The coefficient `b > 0` arises from two micro-founded channels:

- **(1) Bayesian learning about a latent target.** Extending the Stock-Watson unobserved-components stochastic-volatility (UCSV) framework, the paper shows that when forecasters learn about a latent policy target through noisy signals, the posterior variance of the target -- and hence of the forecast -- increases with the distance between the point estimate and the anchor. This captures the intuition that forecasters are more uncertain about where the economy is heading when it is far from its normal state.

- **(2) Policy-response uncertainty (Bernoulli correction).** When the central bank may or may not intervene to correct a deviation, forecasters face a discrete mixture of scenarios (correction vs. no correction). This Bernoulli structure adds a variance component proportional to the squared deviation, which, locally, contributes to the linear relationship in Proposition 1.

NU removes this structural contamination by dividing out the predicted variance:

```
NU_{i,t} = sqrt(Var_{i,t}) / sqrt(a + b * |mu_{i,t} - mu*|)
```

**Unit calibration** (`a = 1, b = 1`) yields a sample-free, fully reproducible formula:

```
NU_{i,t} = sqrt(Var_{i,t}) / sqrt(1 + |mu_{i,t} - mu*|)
```

This calibration correlates above 0.99 with in-sample OLS estimates on ECB-SPF data, making it the natural choice: it requires no estimation, is immune to sample selection, and can be applied to any new survey round without re-fitting.

**Interpretation:**

| NU value | Meaning |
|----------|---------|
| `NU = 1` | All observed dispersion is structural; no excess uncertainty. |
| `NU > 1` | Genuine excess uncertainty beyond what distance from target explains. |
| `NU < 1` | Compressed uncertainty -- less dispersion than the structural model predicts. |

The aggregate NU for a survey round is the cross-sectional mean of individual NU values, with confidence bands at +/- 1 standard error.

### 1.2 Asymmetry Coherence (AC)

AC is a signal-extraction device that combines first-moment (level) and third-moment (skewness) information to quantify the direction and coherence of forecast risks. It operationalizes the "balance of risks" concept central to central bank communication.

**Proposition 2** in Vansteenberghe (2026) proves that the conditional third central moment satisfies:

```
mu_3(X_{t+1} | I_t) = c * (mu_t - mu*)
```

where the net sign `c` is determined by three micro-founded channels:

- **(A) Innovation asymmetry.** Under diagnostic expectations or asymmetric shock distributions, innovations to the forecast process exhibit skewness aligned with the deviation from target. This channel is *coherent*: when expectations are above target, skewness is positive.

- **(B) Asymmetric learning drift.** When the learning process itself drifts asymmetrically (e.g., forecasters update more aggressively in one direction), the third moment inherits the sign of the deviation. This channel is also *coherent*.

- **(C) Policy-correction asymmetry.** If the probability or magnitude of policy correction is asymmetric, it introduces skewness that opposes the deviation from target. This channel is *anti-coherent*: when expectations are above target, the prospect of corrective policy pulls the distribution's tail downward.

The net sign is `c = s_A + s_B - s_C`, where `s_A`, `s_B`, `s_C` denote the signs of each channel. When coherent channels dominate, `c > 0` and risks are aligned with the deviation. When the anti-coherent channel dominates, `c < 0` and risks are tilted against the deviation.

**Construction.** AC is built in five steps:

1. **Bowley skewness** -- a robust, quantile-based asymmetry measure (bounded in [-1, 1], resistant to sparse bins) computed for each individual SPD:
   ```
   B_{i,t} = (Q3 + Q1 - 2*Q2) / (Q3 - Q1)
   ```

2. **Individual directional components.**
   For each forecaster `i` and date `t`, the survey median `Q2_{i,t}` is centered on the target `mu*`, Bowley skewness is smoothed with a short rolling window, and both components are scaled with an IQR-based hyperbolic tangent transformation:
   ```
   Q_tilde_{i,t} = tanh((Q2_{i,t} - mu*) / IQR(Q2 - mu*))
   A_tilde_{i,t} = tanh(B_{i,t} / IQR(B))
   ```

3. **Individual AC score.**
   ```
   AC_{i,t} = ((Q_tilde_{i,t} + A_tilde_{i,t}) / 2) * ((1 + Q_tilde_{i,t} * A_tilde_{i,t}) / 2)
              |-------- directional signal --------|   |------ coherence weight ------|
   ```

4. **Cross-sectional aggregation.**
   The released AC series is the cross-sectional mean of individual `AC_{i,t}` values:
   ```
   AC_t = mean_i(AC_{i,t})
   ```

5. **Confidence bands.**
   The pipeline reports `AC_t +/- 1` standard error across forecasters, matching the aggregation used for NU.

The first factor in `AC_{i,t}` is the directional signal: the average of the level and skewness components. The second factor is a coherence weight in `[0, 1]` that amplifies when both components share the same sign and attenuates when they disagree.

**Interpretation:**

| AC value | Meaning |
|----------|---------|
| `AC > 0` | Coherent upside risk: expectations above target and positively skewed. |
| `AC < 0` | Coherent downside risk: expectations below target and negatively skewed. |
| `AC near 0` | No directional signal, or first and third moments disagree. |

AC lies in `(-1, 1)`. All confidence bands are mean +/- 1 standard error.

---

## 2. Variables, Horizons, and Targets

| Variable | Horizons | Target (`mu*`) | Notes |
|----------|----------|----------------|-------|
| **HICP Inflation** | 1Y, 2Y, 5Y | 2.0% | ECB price-stability mandate; fixed over the sample. |
| **Real GDP Growth** | 1Y, 2Y, 5Y | Time-varying | Linear interpolation from 2.3% (1999) to 1.0% (2026), reflecting the secular decline in euro area potential growth. |

The GDP growth target can alternatively be set to official AMECO potential-growth estimates by setting `GDP_TARGET_MODE = "ameco"` in `config.py`.

### Merged NIU Index

For each variable, the pipeline produces a single summary uncertainty index by taking a cross-horizon weighted average:

```
NIU_merged = 0.50 * NIU(1Y) + 0.30 * NIU(2Y) + 0.20 * NIU(5Y)
```

This weights the near-term horizon most heavily while retaining information from longer horizons. If one horizon is unavailable on a given date, the remaining weights are renormalized over the available horizons.

### Variance Decomposition: Individual Uncertainty vs Disagreement

For each `(variable, horizon)`, the pipeline materializes the textbook variance identity used in Vansteenberghe (2026):

```
Var(average SPD)_t  =  E_i[ Var(SPD_{i,t}) ]   +   Var_i[ E(SPD_{i,t}) ]
                       --------------------        ----------------------
                       Average individual          Disagreement
                       SPD variance                (variance of point means)
```

- **`var_of_avg_spd`** -- variance of the cross-sectional average SPD (built by averaging individual bin probabilities, then computing variance on the active bin regime).
- **`avg_indiv_var`** -- mean across forecasters of each forecaster's own SPD variance.
- **`disagreement`** -- population variance (`ddof=0`) of forecasters' SPD means.
- **`disagreement_theo`** -- residual that closes the identity exactly: `var_of_avg_spd − avg_indiv_var`. By construction the stacked components sum to the LHS.
- **`gap`** -- `var_of_avg_spd − (avg_indiv_var + disagreement)`; typically `1e-16` (machine epsilon).

Outputs per `(variable, horizon)`:

| File | Description |
|------|-------------|
| `data/results/decomposition_{var}_{hor}.csv` | All four series + diagnostics, indexed by survey date. |
| `figures/decomposition_{var}_{hor}.png` | Stacked area: avg individual SPD variance + disagreement (Var of SPD means), with `var_of_avg_spd` overlaid. |
| `figures/decomposition_theo_{var}_{hor}.png` | Same stack but using `disagreement_theo`, so the stack sums exactly to the LHS line. |

### Cumulative Tail Probabilities

A complementary, distribution-free asymmetry signal: for each forecaster and each survey round, compute

```
P_low_t  = P(X <= low_threshold_t)
P_high_t = P(X >= high_threshold_t)
```

Thresholds are configured in `config.TAIL_THRESHOLDS` with one of two modes:

| Variable | Mode | Threshold rule | Rationale |
|----------|------|----------------|-----------|
| Inflation | `absolute` | `low = 1.0%`, `high = 3.0%` | Symmetric +/- 1pp around the fixed ECB 2% target |
| Core      | `absolute` | `low = 1.0%`, `high = 3.0%` | Same as headline inflation |
| GDP       | `target_relative` | `low = mu*_t - 1pp`, `high = mu*_t + 1pp` | Time-varying around the same target the AC step uses (`utils.get_target`).  Anchors both AC and dP on the same notion of "normal" growth. |

In `target_relative` mode the cutoffs drift with `mu*_t`: for GDP they move from `(1.3%, 3.3%)` in 1999 to `(0%, 2%)` in 2026 (linear interpolation). Per-date thresholds are recorded as `low_threshold` / `high_threshold` columns in both individual and aggregate CSVs.

To switch a variable from one mode to the other, just edit the entry in `config.TAIL_THRESHOLDS` -- e.g. change `"inflation"` from `{"mode": "absolute", ...}` to `{"mode": "target_relative", "low_offset": -1.0, "high_offset": 1.0}`.

Bin probabilities are interpolated linearly within each bin (uniform-within-bin assumption). Because the binned cumulative operator is linear, the cross-sectional mean of individual `P_low_i` exactly equals `P_low` computed on the cross-sectional average SPD; the pipeline saves both columns for verification (`*_mean` vs `*_avg_spd`).

A directional asymmetry is derived from the two tails:

```
dP_t = P_high_t - P_low_t
```

`dP` shares AC's sign convention (positive when upside risk dominates), making the two signals directly comparable. Like the individual tails, `dP` is computed per forecaster and aggregated as `mean +/- 1 SE`. By linearity it also coincides with `dP` computed on the average SPD.

Outputs per `(variable, horizon)`:

| File | Description |
|------|-------------|
| `data/results/individual_{var}_{hor}_tailproba.csv` | Per-forecaster `P_low`, `P_high`, `dP`. |
| `data/results/aggregate_{var}_{hor}_tailproba.csv` | `mean +/- 1 SE` bands for `P_low`, `P_high`, `dP`, plus `*_avg_spd` columns and the active thresholds. |
| `figures/tailproba_{var}_{hor}.png` | Both tails on one axis with confidence bands; thin dashed lines show the avg-SPD probabilities as a sanity check. |
| `figures/ac_vs_tailproba_{var}_{hor}.png` | 2x1 stacked comparison: AC (top, with band) vs. `P_low` / `P_high` (bottom). Useful to spot regimes where AC and the cumulative tails disagree on the direction of risk. |
| `figures/delta_p_{var}_{hor}.png` | Stand-alone time series of `dP = P_high - P_low` with `mean +/- 1 SE` band and avg-SPD overlay. |
| `figures/ac_vs_dp_{var}_{hor}.png` | Twin-axis overlay of AC (left axis) and `dP` (right axis), with both zero lines aligned. Survey dates where AC and `dP` disagree on sign (with both meaningfully non-zero) are highlighted with a faint vertical grey band; the count is reported in the title. |

---

## 3. Data Source

**ECB Survey of Professional Forecasters (SPF):** individual probability distributions, collected quarterly since 1999.

- Portal: https://www.ecb.europa.eu/stats/prices/indic/forecast/html/index.en.html
- Individual forecasts ZIP: https://www.ecb.europa.eu/stats/prices/indic/forecast/shared/files/SPF_individual_forecasts.zip

The pipeline downloads and processes these files automatically.

---

## 4. Date Convention

All dates represent the **survey formation date** -- when the ECB surveyed professional forecasters -- not the target period being forecasted. This convention is noted on all figures.

| Quarter | Formation Month |
|---------|-----------------|
| Q1 | January |
| Q2 | April |
| Q3 | July |
| Q4 | October |

---

## 5. Quick Start

```bash
pip install -r requirements.txt
python main.py
```

This runs the full pipeline. To run individual steps:

```bash
python main.py download        # Step 1: Download SPF data from ECB
python main.py realized        # Step 2: Download realized HICP and GDP series via API
python main.py panels          # Step 3: Parse raw CSVs into clean panels
python main.py niu             # Step 4: Compute Normalized Uncertainty
python main.py ac              # Step 5: Compute Asymmetry Coherence
python main.py merge           # Step 6: Merge horizons into summary NIU index
python main.py decomposition   # Step 7: Decompose Var(avg SPD) = Avg(Var_i) + Disagreement
python main.py tail_proba      # Step 8: Cumulative tail probabilities + AC-vs-tails comparison
python main.py plots           # Step 9: Generate publication-ready figures
python main.py diagnostics     # Step 10: Variance diagnostics and data-quality checks
```

### Offline / firewalled environment

If your network blocks `www.ecb.europa.eu`, `data-api.ecb.europa.eu`, or
`ec.europa.eu` (typical on corporate intranets), enable offline mode and
place the inputs manually. Offline mode is controlled by either:

- the environment variable `UNCERTAINTY_OFFLINE=1` (recommended — no edit
  to `config.py`, no risk of committing a local override), or
- editing `config.py` and setting `OFFLINE_MODE = True`.

When offline mode is on, no network calls are attempted. Each download
step looks for local files and either reuses them or prints a manual-
download hint pointing at the expected path.

**Files to provide manually** (download from another machine, then copy
into the indicated paths under your project folder):

| Required by | Source URL | Local path |
|---|---|---|
| `download` | https://www.ecb.europa.eu/stats/prices/indic/forecast/shared/files/SPF_individual_forecasts.zip | unzip into `data/SPF_individual_forecasts/*.csv` |
| `realized` (optional) | ECB Data Portal → series `ICP.M.U2.N.000000.4.ANR` (HICP YoY) | `data/realized/inflation.csv` with columns `Date,inflation` |
| `realized` (optional) | ECB Data Portal → `MNA.Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY` *or* Eurostat `NAMQ_10_GDP` (geo=EA20, na_item=B1GQ, unit=CLV_PCH_SM, s_adj=SCA) | `data/realized/gdp.csv` with columns `Date,gdp_growth` |

The `realized` step is optional: no other step currently consumes those
files, so the pipeline will run end-to-end without them. Only `download`
is hard-required.

Typical workflow on a firewalled machine:

```bash
# Windows PowerShell example
$env:UNCERTAINTY_OFFLINE = "1"
# ... copy SPF CSVs into data\SPF_individual_forecasts\ ...
python main.py panels niu ac merge plots diagnostics
```

```bash
# macOS / Linux
export UNCERTAINTY_OFFLINE=1
python main.py panels niu ac merge plots diagnostics
```

If you skip the `download` and `realized` steps explicitly (as above),
no network access is attempted at all and `UNCERTAINTY_OFFLINE` is not
strictly needed — but setting it is the safest default in case you
accidentally include a download step.

---

## 6. Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| `download` | `01_download_data.py` | Downloads and unzips the ECB-SPF individual forecasts archive. |
| `realized` | `05_download_realized.py` | Downloads realized HICP inflation and GDP growth series via Eurostat/ECB APIs. |
| `panels` | `02_prepare_panels.py` | Parses the raw quarterly CSV files into clean panel datasets (one per variable-horizon). |
| `niu` | `03_compute_niu.py` | Computes individual SPD moments (mean, variance, quantiles, Bowley skewness, entropy) and Normalized Uncertainty for each forecaster-round. |
| `ac` | `04_compute_ac.py` | Computes individual forecaster AC scores and aggregates them into the cross-sectional mean AC series. |
| `merge` | `07_merge_horizons.py` | Produces the cross-horizon weighted-average NIU index (50/30/20 weights). |
| `decomposition` | `09_decomposition.py` | Decomposes the variance of the cross-sectional average SPD into average individual SPD variance + cross-forecaster disagreement, per variable x horizon. |
| `tail_proba` | `10_tail_proba.py` | Cumulative tail probabilities `P(X <= low)` and `P(X >= high)` per variable x horizon, plus an AC-vs-tails comparison figure. |
| `plots` | `06_plot_results.py` | Generates publication-ready time-series figures with confidence bands. |
| `diagnostics` | `08_diagnostics.py` | Runs variance diagnostics and data-quality checks (see Section 8). |

---

## 7. Project Structure

```
uncertainty_index/
├── main.py                          # Pipeline orchestrator (run all or individual steps)
├── config.py                        # All configurable parameters (edit this file)
├── requirements.txt                 # Python dependencies
├── steps/
│   ├── __init__.py
│   ├── 01_download_data.py          # Download + unzip SPF data
│   ├── 02_prepare_panels.py         # Parse raw CSVs into clean panels
│   ├── 03_compute_niu.py            # Normalized Uncertainty computation
│   ├── 04_compute_ac.py             # Asymmetry Coherence computation
│   ├── 05_download_realized.py      # Download realized macro series via API
│   ├── 06_plot_results.py           # Publication-ready figures
│   ├── 07_merge_horizons.py         # Cross-horizon weighted average
│   ├── 08_diagnostics.py            # Variance diagnostics and data-quality checks
│   ├── 09_decomposition.py          # Variance decomposition (disagreement vs individual uncertainty)
│   ├── 10_tail_proba.py             # Cumulative tail probabilities + AC-vs-tails comparison
│   └── utils.py                     # Shared computation functions (SPD moments, quantiles)
├── data/
│   ├── SPF_individual_forecasts/    # Raw quarterly CSVs from ECB
│   ├── panels/                      # Prepared panel datasets
│   ├── realized/                    # Realized macro series (HICP, GDP)
│   └── results/                     # NU, AC, and merged NIU output CSVs
└── figures/                         # Publication-ready PNG figures
```

---

## 8. Diagnostics

The diagnostics step (`python main.py diagnostics`) produces three sets of checks that help verify whether observed variance trends reflect genuine shifts in uncertainty rather than survey artifacts:

1. **Raw variance decomposition.** For each variable-horizon, reports the mean, median, and P10--P90 range of raw individual SPD variance over time. This shows whether variance movements are driven by the bulk of the cross-section or by outliers.

2. **Average bins filled.** Tracks the average number of non-zero probability bins per forecaster over time. A decline in bins filled could indicate that forecasters are using fewer bins (a survey-design artifact) rather than genuinely narrowing their distributions.

3. **Number of forecasters per round.** Reports the cross-sectional sample size for each survey round. Thin samples reduce the reliability of cross-sectional aggregates and may produce spurious volatility in NU and AC.

---

## 9. Configuration

All parameters are centralized in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGETS` | `{"inflation": 2.0, "gdp": 1.3}` | Policy anchors for NU normalization. |
| `GDP_TARGET_TIME_VARYING` | `True` | Use time-varying GDP target (linear interpolation). |
| `GDP_TARGET_MODE` | `"linear"` | GDP target source: `"linear"` (interpolation 2.3% to 1.0%) or `"ameco"` (official estimates). |
| `GDP_TARGET_START_VALUE` / `END_VALUE` | `2.3` / `1.0` | Endpoints for linear interpolation of GDP potential growth. |
| `NIU_A`, `NIU_B` | `1.0`, `1.0` | Unit calibration parameters for NU denominator. |
| `NIU_P` | `0.5` | Exponent in NU denominator (`sqrt` by default). |
| `BOWLEY_SMOOTHING_WINDOW` | `2` | Rolling window for individual Bowley skewness. |
| `BIN_REGIME_CUTOFF` | `"2024-09-01"` | Date when ECB changed HICP bin definitions. |
| `TAIL_THRESHOLDS` | `{inflation: abs 1%/3%, gdp: target +/- 1pp}` | Mode-driven cutoffs for the cumulative tail probabilities (see Section 2). |
| `VARIABLES` | `["inflation", "gdp"]` | Variables to process. |
| `HORIZONS` | `["1Y", "2Y", "5Y"]` | Forecast horizons to process. |
| `FIGURE_DPI` | `300` | Resolution for saved figures. |

---

## 10. Confidence Bands

All aggregate time series (NU, AC, merged NIU) are reported with confidence bands constructed as the cross-sectional mean +/- 1 standard error. This reflects sampling uncertainty from the finite pool of survey respondents in each round.

---

## License and Citation

The code in this repository is released under the MIT License. Raw and realized macroeconomic data remain subject to the terms of their original providers (ECB, Eurostat, and AMECO where applicable).

If you use this code, data, or methodology, please cite:

> Vansteenberghe, E. (2026). "Uncertain and Asymmetric Forecasts." *Banque de France Working Paper*.
