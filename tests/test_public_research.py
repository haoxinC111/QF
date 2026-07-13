from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare_quant.public_research import (
    PublicStrategyConfig,
    _build_features,
    _calculate_metrics,
    _eastmoney_secid,
    _rebalance_public_positions,
    load_membership,
    members_at,
    seal_public_cache,
    verify_public_cache,
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

    def test_suspended_position_is_never_scaled_to_fund_buys(self) -> None:
        config = PublicStrategyConfig()
        outcome = _rebalance_public_positions(
            current_positions={"LOCKED": 0.60, "OLD": 0.35},
            requested_target={"NEW": 0.95},
            tradable_symbols={"OLD", "NEW"},
            nav_open=1.0,
            when=pd.Timestamp("2024-01-02"),
            config=config,
        )
        self.assertAlmostEqual(outcome.positions["LOCKED"], 0.60, places=12)
        self.assertGreaterEqual(outcome.cash, 0.0)
        self.assertLess(outcome.executable_scale, 1.0)
        self.assertAlmostEqual(outcome.locked_value, 0.60, places=12)
        self.assertAlmostEqual(
            outcome.cash + sum(outcome.positions.values()) + outcome.cost,
            1.0,
            places=12,
        )

    def test_public_cache_fingerprint_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            membership_path = root / "csi300.csv"
            pd.DataFrame(
                [
                    {
                        "symbol": "SH600000",
                        "name": "测试",
                        "opt-in": "2020-01-01",
                        "opt-out": "",
                    },
                    {
                        "symbol": "SH600001",
                        "name": "缺失行情",
                        "opt-in": "2020-01-01",
                        "opt-out": "",
                    },
                ]
            ).to_csv(membership_path, index=False)
            cache = root / "cache"
            bars = cache / "bars"
            bars.mkdir(parents=True)
            for symbol in ["SH600000", "SH000300", "SH510300"]:
                (bars / f"{symbol}.csv.gz").write_bytes(f"sealed-{symbol}".encode())

            manifest = seal_public_cache(membership_path, cache)
            self.assertEqual(manifest["available_count"], 3)
            verified = verify_public_cache(membership_path, cache)
            self.assertTrue(verified["verification"]["verified"])

            (bars / "SH600000.csv.gz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "完整性校验失败"):
                verify_public_cache(membership_path, cache)
            with self.assertRaisesRegex(ValueError, "已经封存"):
                seal_public_cache(membership_path, cache)

            (bars / "SH600000.csv.gz").write_bytes(b"sealed-SH600000")
            (bars / "SH600001.csv.gz").write_bytes(b"new-unsealed-file")
            with self.assertRaisesRegex(ValueError, "文件集合不一致"):
                verify_public_cache(membership_path, cache)
            resumable = verify_public_cache(
                membership_path,
                cache,
                _allow_unsealed_files=True,
            )
            self.assertEqual(
                resumable["verification"]["unsealed_paths"],
                ["bars/SH600001.csv.gz"],
            )

    def test_public_config_requires_separate_regime_and_benchmark(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须分离"):
            PublicStrategyConfig(
                regime_symbol="SH000300",
                performance_benchmark_symbol="SH000300",
            ).validate()


if __name__ == "__main__":
    unittest.main()
