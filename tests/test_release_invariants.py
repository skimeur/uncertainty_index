import importlib
import unittest

import numpy as np
import pandas as pd


merge_horizons = importlib.import_module("steps.07_merge_horizons")
compute_ac = importlib.import_module("steps.04_compute_ac")
prepare_panels = importlib.import_module("steps.02_prepare_panels")
decomposition = importlib.import_module("steps.09_decomposition")
tail_proba = importlib.import_module("steps.10_tail_proba")
utils = importlib.import_module("steps.utils")


class ReleaseInvariantTests(unittest.TestCase):
    def test_panel_target_selection_prefers_exact_horizon_code(self):
        df = pd.DataFrame(
            {
                "TARGET_PERIOD": ["2000", "2000Dec", "2000", "2000Dec"],
                "FCT_SOURCE": [1, 1, 2, 2],
                "POINT": [1.5, 1.7, 1.2, 1.4],
            }
        )

        out = prepare_panels._select_preferred_targets(df, ["2000Dec", "2000"])

        self.assertEqual(len(out), 2)
        self.assertEqual(list(out.sort_values("FCT_SOURCE")["TARGET_PERIOD"]), ["2000Dec", "2000Dec"])

    def test_weighted_average_reweights_available_horizons(self):
        data = pd.DataFrame(
            {
                "NIU_1Y": [1.0, 1.0, np.nan],
                "NIU_2Y": [3.0, np.nan, 2.0],
                "NIU_5Y": [np.nan, 5.0, np.nan],
            },
            index=pd.to_datetime(["2000-01-01", "2000-04-01", "2000-07-01"]),
        )

        out = merge_horizons.weighted_average(
            data,
            weights={"1Y": 0.5, "2Y": 0.3, "5Y": 0.2},
        )

        self.assertAlmostEqual(out.iloc[0], (1.0 * 0.5 + 3.0 * 0.3) / 0.8)
        self.assertAlmostEqual(out.iloc[1], (1.0 * 0.5 + 5.0 * 0.2) / 0.7)
        self.assertAlmostEqual(out.iloc[2], 2.0)

    def test_aggregate_ac_releases_mean_forecaster_score(self):
        individual = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-01-01", "2024-04-01", "2024-04-01"]
                ),
                "FCT_SOURCE": [1, 2, 3, 1, 2],
                "Q2_median_spd": [2.2, 2.4, 2.3, 1.8, 1.9],
                "Bowley_smoothed": [0.1, 0.2, 0.15, -0.1, -0.2],
                "ACI": [0.2, 0.5, 0.4, -0.1, -0.2],
                "coherence": [0.8, 0.9, 0.85, 0.75, 0.7],
                "NIU": [1.0, 1.1, 0.9, 1.2, 1.0],
            }
        )

        agg = compute_ac.aggregate_ac(individual, "inflation")
        jan = agg.loc[agg["Date"] == pd.Timestamp("2024-01-01")].iloc[0]

        expected_mean = np.mean([0.2, 0.5, 0.4])
        expected_std = np.std([0.2, 0.5, 0.4], ddof=1)
        expected_se = expected_std / np.sqrt(3)

        self.assertIn("AC_mean", agg.columns)
        self.assertIn("AC_formula", agg.columns)
        self.assertAlmostEqual(jan["AC_mean"], expected_mean)
        self.assertAlmostEqual(jan["AC_se_lo"], expected_mean - expected_se)
        self.assertAlmostEqual(jan["AC_se_hi"], expected_mean + expected_se)
        self.assertAlmostEqual(jan["ACI_mean"], jan["AC_mean"])

    def test_tail_threshold_modes_resolve_correctly(self):
        """``_resolve_thresholds`` returns fixed values in absolute mode
        and ``mu*_t + offset`` in target-relative mode (matching the AC
        target).  This is the contract relied on by the release CSVs."""
        import config

        infl_spec = config.TAIL_THRESHOLDS["inflation"]
        gdp_spec = config.TAIL_THRESHOLDS["gdp"]
        self.assertEqual(infl_spec["mode"], "absolute")
        self.assertEqual(gdp_spec["mode"], "target_relative")

        # Inflation: constant across dates
        for date in (pd.Timestamp("1999-01-01"), pd.Timestamp("2026-04-01")):
            lo, hi = tail_proba._resolve_thresholds("inflation", date)
            self.assertEqual((lo, hi), (1.0, 3.0))

        # GDP: must equal (target + low_offset, target + high_offset),
        # i.e. drift with the linear potential-growth interpolation.
        for date in (
            pd.Timestamp("1999-01-01"),
            pd.Timestamp("2010-07-01"),
            pd.Timestamp("2026-01-01"),
        ):
            lo, hi = tail_proba._resolve_thresholds("gdp", date)
            target = utils.get_target("gdp", date)
            self.assertAlmostEqual(lo, target - 1.0, places=10)
            self.assertAlmostEqual(hi, target + 1.0, places=10)

        # And the endpoints match the configured start/end values
        # (target = 2.3 in 1999, 1.0 in 2026 → thresholds 1.3/3.3 and 0/2)
        lo_99, hi_99 = tail_proba._resolve_thresholds(
            "gdp", pd.Timestamp("1999-01-01")
        )
        self.assertAlmostEqual(lo_99, 1.3, places=10)
        self.assertAlmostEqual(hi_99, 3.3, places=10)
        lo_26, hi_26 = tail_proba._resolve_thresholds(
            "gdp", pd.Timestamp("2026-01-01")
        )
        self.assertAlmostEqual(lo_26, 0.0, places=10)
        self.assertAlmostEqual(hi_26, 2.0, places=10)

    def test_cumulative_proba_uniform_within_bin(self):
        # Three identical bins on [0,1], [1,2], [2,3] with 1/3 mass each.
        edges = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
        probs = [100 / 3, 100 / 3, 100 / 3]

        # Threshold exactly at a bin edge -> two full bins below
        self.assertAlmostEqual(
            utils.cumulative_proba(probs, edges, 2.0, "lower"),
            2 / 3 * 100, places=6,
        )
        # Threshold mid-bin -> one full bin + half of the second
        self.assertAlmostEqual(
            utils.cumulative_proba(probs, edges, 1.5, "lower"),
            (1 / 3 + 1 / 3 * 0.5) * 100, places=6,
        )
        # Upper symmetry: P(X >= 1.5) = 1 - 0.5 over uniform = same as lower's complement
        self.assertAlmostEqual(
            utils.cumulative_proba(probs, edges, 1.5, "upper"),
            (1 / 3 + 1 / 3 * 0.5) * 100, places=6,
        )
        # Threshold below the support: P(X<=T)=0, P(X>=T)=100
        self.assertAlmostEqual(
            utils.cumulative_proba(probs, edges, -1.0, "lower"), 0.0, places=6,
        )
        self.assertAlmostEqual(
            utils.cumulative_proba(probs, edges, -1.0, "upper"), 100.0, places=6,
        )

    def test_cumulative_proba_is_linear_across_forecasters(self):
        """Mean across forecasters of P_low_i must equal P_low on the
        cross-sectional average SPD.  This identity is the basis for the
        ``P_*_avg_spd`` sanity-check column in the released CSV."""
        edges = [(-1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
        # Two forecasters with very different beliefs
        p1 = [10.0, 30.0, 40.0, 20.0]
        p2 = [0.0, 60.0, 30.0, 10.0]

        mean_of_indiv = (
            utils.cumulative_proba(p1, edges, 1.5, "lower")
            + utils.cumulative_proba(p2, edges, 1.5, "lower")
        ) / 2.0

        avg_probs = [(a + b) / 2 for a, b in zip(p1, p2)]
        on_avg_spd = utils.cumulative_proba(avg_probs, edges, 1.5, "lower")

        self.assertAlmostEqual(mean_of_indiv, on_avg_spd, places=10)

    def test_tail_proba_release_csvs_match_avg_spd_identity(self):
        """Released aggregate CSVs must satisfy the linearity identity:
        cross-sectional mean of individual tail probabilities equals the
        tail probability computed directly on the average SPD."""
        import config

        for var in config.VARIABLES:
            if var not in config.TAIL_THRESHOLDS:
                continue
            for hor in config.HORIZONS:
                path = (
                    config.RESULTS_DIR
                    / f"aggregate_{var}_{hor}_tailproba.csv"
                )
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                for col_mean, col_avg in (
                    ("P_low_mean", "P_low_avg_spd"),
                    ("P_high_mean", "P_high_avg_spd"),
                    ("dP_mean", "dP_avg_spd"),
                ):
                    if col_mean not in df.columns or col_avg not in df.columns:
                        continue
                    diff = (df[col_mean] - df[col_avg]).abs().max()
                    self.assertLess(
                        float(diff), 1e-9,
                        msg=f"{var}/{hor}: {col_mean} != {col_avg} (max diff {diff})",
                    )
                # dP must equal P_high - P_low up to float roundoff
                if {"dP_mean", "P_high_mean", "P_low_mean"} <= set(df.columns):
                    gap = (
                        df["dP_mean"]
                        - (df["P_high_mean"] - df["P_low_mean"])
                    ).abs().max()
                    self.assertLess(float(gap), 1e-9)

                # Per-date thresholds must match _resolve_thresholds for
                # each Date in the CSV.  Catches drift between the AC
                # target and the tail-proba target after edits to either.
                if {"low_threshold", "high_threshold"} <= set(df.columns):
                    df_dt = df.copy()
                    df_dt["Date"] = pd.to_datetime(df_dt["Date"])
                    for _, row in df_dt.iterrows():
                        lo, hi = tail_proba._resolve_thresholds(
                            var, row["Date"]
                        )
                        self.assertAlmostEqual(
                            float(row["low_threshold"]), float(lo), places=10,
                        )
                        self.assertAlmostEqual(
                            float(row["high_threshold"]), float(hi), places=10,
                        )

    def test_decomposition_identity_holds_on_release(self):
        """Var(avg SPD) == AvgIndivVar + Disagreement must hold to 1e-10 on
        the released CSVs (panels x niu). This is the headline invariant
        of the decomposition step."""
        import config

        for var in config.VARIABLES:
            for hor in config.HORIZONS:
                path = config.RESULTS_DIR / f"decomposition_{var}_{hor}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                gap = (
                    df["var_of_avg_spd"]
                    - df["avg_indiv_var"]
                    - df["disagreement"]
                ).abs()
                self.assertLess(
                    float(gap.max()),
                    1e-10,
                    msg=f"Decomposition identity violated for {var} {hor}",
                )
                # Residual disagreement must be non-negative (clipped at 0)
                self.assertTrue((df["disagreement_theo"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
