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
python main.py download      # Step 1: Download SPF data from ECB
python main.py realized      # Step 2: Download realized HICP and GDP series via API
python main.py panels        # Step 3: Parse raw CSVs into clean panels
python main.py niu           # Step 4: Compute Normalized Uncertainty
python main.py ac            # Step 5: Compute Asymmetry Coherence
python main.py merge         # Step 6: Merge horizons into summary NIU index
python main.py plots         # Step 7: Generate publication-ready figures
python main.py diagnostics   # Step 8: Variance diagnostics and data-quality checks
```

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
| `plots` | `06_plot_results.py` | Generates publication-ready time-series figures with confidence bands. |
| `diagnostics` | `08_diagnostics.py` | Runs variance diagnostics and data-quality checks (see Section 8). |

---

## 7. Project Structure

```
uncertainty_index/
├── main.py                          # Pipeline orchestrator (run all or individual steps)
├── config.py                        # All configurable parameters (edit this file)
├── requirements.txt                 # Python dependencies
├── code/
│   ├── __init__.py
│   ├── 01_download_data.py          # Download + unzip SPF data
│   ├── 02_prepare_panels.py         # Parse raw CSVs into clean panels
│   ├── 03_compute_niu.py            # Normalized Uncertainty computation
│   ├── 04_compute_ac.py             # Asymmetry Coherence computation
│   ├── 05_download_realized.py      # Download realized macro series via API
│   ├── 06_plot_results.py           # Publication-ready figures
│   ├── 07_merge_horizons.py         # Cross-horizon weighted average
│   ├── 08_diagnostics.py            # Variance diagnostics and data-quality checks
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
