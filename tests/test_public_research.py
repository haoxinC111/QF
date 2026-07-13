from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_quant.public_research import (
    PublicStrategyConfig,
    _build_features,
    _calculate_metrics,
    _eastmoney_secid,
    load_membership,
    members_at,
)


class PublicResearchTests(unittest.TestCase):
    def test_membership_uses_exclusive_opt_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "membership.csv"
            pd.DataFrame(
                [
                    {
                        "symbol": "SH600000",
                        "name": "甲",
                        "opt-in": "2020-01-01",
                        "opt-out": "2020-06-15",
                    },
                    {
                        "symbol": "SZ000001",
                        "name": "乙",
                        "opt-in": "2020-06-15",
                        "opt-out": "",
                    },
                ]
            ).to_csv(path, index=False)
            membership = load_membership(path)
            on_boundary = set(members_at(membership, "2020-06-15")["symbol"])
            self.assertEqual(on_boundary, {"SZ000001"})

    def test_market_code_mapping(self) -> None:
        self.assertEqual(_eastmoney_secid("SH600000"), "1.600000")
        self.assertEqual(_eastmoney_secid("SZ000001"), "0.000001")
        with self.assertRaises(ValueError):
            _eastmoney_secid("US.AAPL")

    def test_features_do_not_use_future_prices(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=300)
        close = pd.Series(np.linspace(10.0, 20.0, len(dates)))
        bars = pd.DataFrame(
            {
                "date": dates,
                "symbol": "SH600000",
                "name": "测试",
                "open": close,
                "close": close,
                "amount": 100_000_000.0,
            }
        )
        signal_date = dates[270]
        original = _build_features(bars, 60).set_index("date").loc[signal_date]
        changed = bars.copy()
        changed.loc[changed["date"] > signal_date, ["open", "close"]] *= 50.0
        recomputed = _build_features(changed, 60).set_index("date").loc[signal_date]
        for column in ["mom_12_1", "mom_6_1", "trend", "volatility"]:
            self.assertAlmostEqual(float(original[column]), float(recomputed[column]), places=12)

    def test_metrics_cagr_and_drawdown(self) -> None:
        curve = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
                "nav": [1.0, 0.8, 1.21],
            }
        )
        metrics = _calculate_metrics(curve)
        self.assertAlmostEqual(float(metrics["cagr"]), 0.10, places=2)
        self.assertAlmostEqual(float(metrics["max_drawdown"]), -0.20, places=12)

    def test_public_config_rejects_infeasible_stock_cap(self) -> None:
        with self.assertRaises(ValueError):
            PublicStrategyConfig(top_n=10, max_stock_weight=0.05).validate()


if __name__ == "__main__":
    unittest.main()
