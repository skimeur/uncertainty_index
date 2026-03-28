import importlib
import unittest

import numpy as np
import pandas as pd


merge_horizons = importlib.import_module("code.07_merge_horizons")
compute_ac = importlib.import_module("code.04_compute_ac")
prepare_panels = importlib.import_module("code.02_prepare_panels")


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


if __name__ == "__main__":
    unittest.main()
