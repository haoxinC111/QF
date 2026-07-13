from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare_quant.backtest import Backtester
from ashare_quant.config import AppConfig
from ashare_quant.data import make_demo_bundle
from ashare_quant.factors import MultiFactorStrategy, _capped_allocation


class AllocationTests(unittest.TestCase):
    def test_cap_and_total_are_respected(self) -> None:
        raw = pd.Series([10.0, 3.0, 1.0, 0.5], index=list("ABCD"))
        result = _capped_allocation(raw, total=0.60, cap=0.20)
        self.assertAlmostEqual(float(result.sum()), 0.60, places=10)
        self.assertLessEqual(float(result.max()), 0.20 + 1e-12)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = make_demo_bundle(seed=11, start="2016-01-04", end="2020-12-31", symbols=30)
        base = AppConfig()
        cls.config = replace(
            base,
            backtest=replace(
                base.backtest,
                start_date="2018-01-01",
                end_date="2019-12-31",
                initial_cash=800_000.0,
            ),
            strategy=replace(
                base.strategy,
                top_n=12,
                max_stock_weight=0.10,
                min_avg_amount_million=20.0,
            ),
        )
        cls.result = Backtester(cls.bundle, cls.config).run()

    def test_signal_is_executed_later(self) -> None:
        trades = self.result.trades
        self.assertFalse(trades.empty)
        self.assertTrue((pd.to_datetime(trades["date"]) > pd.to_datetime(trades["signal_date"])).all())

    def test_buys_use_board_lots(self) -> None:
        buys = self.result.trades.loc[self.result.trades["side"].eq("BUY")]
        self.assertTrue(((buys["shares"] % self.config.execution.lot_size).abs() < 1e-8).all())

    def test_fee_model(self) -> None:
        trades = self.result.trades
        self.assertTrue((trades["commission"] >= self.config.execution.minimum_commission).all())
        sells = trades.loc[trades["side"].eq("SELL")]
        self.assertFalse(sells.empty)
        expected = sells["notional"] * self.config.execution.stamp_duty_sell
        self.assertTrue(((sells["stamp_duty"] - expected).abs() < 1e-8).all())

    def test_target_weight_constraints(self) -> None:
        selections = self.result.selections
        self.assertLessEqual(
            float(selections["target_weight"].max()),
            self.config.strategy.max_stock_weight + 1e-12,
        )
        sums = selections.groupby("signal_date")["target_weight"].sum()
        self.assertTrue((sums <= self.config.strategy.risk_on_exposure + 1e-12).all())

    def test_no_future_price_leakage(self) -> None:
        signal_date = pd.Timestamp("2018-06-29")
        original = MultiFactorStrategy(self.bundle, self.config.strategy).generate(signal_date)
        changed_bundle = make_demo_bundle(
            seed=11, start="2016-01-04", end="2020-12-31", symbols=30
        )
        future = changed_bundle.bars["date"] > signal_date
        changed_bundle.bars.loc[future, ["open", "high", "low", "close"]] *= 50.0
        changed = MultiFactorStrategy(changed_bundle, self.config.strategy).generate(signal_date)
        self.assertEqual(original.weights.keys(), changed.weights.keys())
        for symbol in original.weights:
            self.assertAlmostEqual(original.weights[symbol], changed.weights[symbol], places=12)

    def test_limit_up_is_retried(self) -> None:
        bundle = make_demo_bundle(seed=17, start="2016-01-04", end="2019-06-30", symbols=30)
        strategy = MultiFactorStrategy(bundle, self.config.strategy)
        signal_date = pd.Timestamp("2018-01-31")
        plan = strategy.generate(signal_date)
        self.assertTrue(plan.weights)
        blocked = next(iter(plan.weights))
        execution_date = bundle.calendar[bundle.calendar.searchsorted(signal_date, side="right")]
        mask = bundle.bars["date"].eq(execution_date) & bundle.bars["symbol"].eq(blocked)
        bundle.bars.loc[mask, "open"] = bundle.bars.loc[mask, "up_limit"]
        result = Backtester(
            bundle,
            replace(self.config, backtest=replace(self.config.backtest, end_date="2018-03-31")),
        ).run()
        rejected = result.orders.loc[
            result.orders["symbol"].eq(blocked)
            & result.orders["reason"].eq("open_at_up_limit")
        ]
        later_fill = result.trades.loc[
            result.trades["symbol"].eq(blocked)
            & (pd.to_datetime(result.trades["date"]) > execution_date)
        ]
        self.assertFalse(rejected.empty)
        self.assertFalse(later_fill.empty)


if __name__ == "__main__":
    unittest.main()
