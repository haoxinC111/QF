from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare_quant.backtest import Backtester, Lot, Position
from ashare_quant.config import AppConfig
from ashare_quant.data import (
    ACTION_COLUMNS,
    CACHE_SCHEMA_VERSION,
    MarketDataBundle,
    TushareDownloader,
    make_demo_bundle,
)
from ashare_quant.factors import (
    MultiFactorStrategy,
    _capped_allocation,
    _group_capped_allocation,
)
from ashare_quant.report import (
    calculate_metrics,
    industry_exposure_table,
    style_exposure_table,
)
from ashare_quant.research import (
    run_cost_stress,
    run_factor_ablation,
    run_rolling_oos,
)
from ashare_quant.provenance import (
    build_reproducibility_manifest,
    build_file_inventory,
    inventory_sha256,
    record_experiment,
    verify_artifact_manifest,
    write_artifact_manifest,
)


def research_config(end_date: str = "2019-12-31") -> AppConfig:
    base = AppConfig()
    return replace(
        base,
        backtest=replace(
            base.backtest,
            start_date="2018-01-01",
            end_date=end_date,
            initial_cash=800_000.0,
        ),
        strategy=replace(
            base.strategy,
            top_n=12,
            max_stock_weight=0.10,
            min_avg_amount_million=20.0,
        ),
    )


class AllocationAndConfigTests(unittest.TestCase):
    def test_cap_and_total_are_respected(self) -> None:
        raw = pd.Series([10.0, 3.0, 1.0, 0.5], index=list("ABCD"))
        result = _capped_allocation(raw, total=0.60, cap=0.20)
        self.assertAlmostEqual(float(result.sum()), 0.60, places=10)
        self.assertLessEqual(float(result.max()), 0.20 + 1e-12)

    def test_stock_and_industry_caps_are_respected_together(self) -> None:
        raw = pd.Series([9.0, 5.0, 3.0, 2.0, 1.0], index=list("ABCDE"))
        industries = pd.Series(["X", "X", "Y", "Y", "Z"], index=raw.index)
        result = _group_capped_allocation(
            raw, total=0.75, stock_cap=0.20, groups=industries, group_cap=0.30
        )
        self.assertAlmostEqual(float(result.sum()), 0.75, places=10)
        self.assertLessEqual(float(result.max()), 0.20 + 1e-12)
        self.assertLessEqual(
            float(result.groupby(industries).sum().max()), 0.30 + 1e-12
        )

    def test_historical_fee_boundaries(self) -> None:
        execution = AppConfig().execution
        self.assertEqual(execution.fee_on("2022-04-28").transfer_fee_rate, 0.00002)
        self.assertEqual(execution.fee_on("2022-04-29").transfer_fee_rate, 0.00001)
        self.assertEqual(execution.fee_on("2023-08-27").stamp_duty_sell, 0.001)
        self.assertEqual(execution.fee_on("2023-08-28").stamp_duty_sell, 0.0005)

    def test_total_return_benchmark_cannot_be_regime_index(self) -> None:
        base = AppConfig()
        invalid = replace(
            base,
            data=replace(
                base.data,
                regime_index=base.data.benchmark_index,
                benchmark_is_total_return=True,
            ),
        )
        with self.assertRaisesRegex(ValueError, "不能同时作为"):
            invalid.validate()

    def test_lot_ledger_enforces_t_plus_one(self) -> None:
        position = Position()
        day_one = pd.Timestamp("2024-01-02")
        day_two = pd.Timestamp("2024-01-03")
        position.add(Lot(100, day_one, day_two))
        self.assertEqual(position.sellable_shares(day_one), 0)
        self.assertEqual(position.remove_sellable(100, day_one), 0)
        self.assertEqual(position.remove_sellable(100, day_two), 100)


class ProvenanceTests(unittest.TestCase):
    def test_experiment_registry_is_idempotent_and_refuses_corruption(self) -> None:
        reproducibility = {
            "run_fingerprint_sha256": "run-001",
            "created_at_utc": "2026-07-13T00:00:00+00:00",
            "config": {"sha256": "config-001"},
            "source": {"sha256": "source-001"},
            "git": {"commit": "abc123", "dirty": False},
            "data": {"data_fingerprint_sha256": "data-001"},
        }
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "experiments.jsonl"
            metrics = Path(directory) / "metrics.json"
            metrics.write_text('{"cagr": 0.1}\n', encoding="utf-8")
            for _ in range(2):
                record_experiment(
                    registry,
                    reproducibility,
                    experiment_type="unit_test",
                    protocol={"untouched_holdout_certified": False},
                    artifacts=["metrics.json"],
                )
            self.assertEqual(len(registry.read_text().splitlines()), 1)
            record = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual(record["artifacts"][0]["path"], "metrics.json")
            self.assertEqual(len(record["artifacts"][0]["sha256"]), 64)
            registry.write_text("not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                record_experiment(
                    registry,
                    {**reproducibility, "run_fingerprint_sha256": "run-002"},
                    experiment_type="unit_test",
                    protocol={},
                    artifacts=[],
                )

    def test_artifact_manifest_detects_tampering_and_unsealed_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "metrics.json"
            second = root / "nested" / "curve.csv"
            second.parent.mkdir()
            first.write_text('{"cagr": 0.1}\n', encoding="utf-8")
            second.write_text("date,nav\n2025-01-01,1.0\n", encoding="utf-8")
            manifest = write_artifact_manifest(root, [first, second])

            verified = verify_artifact_manifest(manifest, strict=True)
            self.assertTrue(verified["verified"])
            self.assertEqual(verified["file_count"], 2)
            self.assertEqual(verified["unsealed_paths"], [])

            first.write_text('{"cagr": 0.2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "完整性校验失败"):
                verify_artifact_manifest(manifest)
            first.write_text('{"cagr": 0.1}\n', encoding="utf-8")
            (root / "unsealed.txt").write_text("extra", encoding="utf-8")
            self.assertEqual(
                verify_artifact_manifest(manifest)["unsealed_paths"],
                ["unsealed.txt"],
            )
            with self.assertRaisesRegex(ValueError, "未封存文件"):
                verify_artifact_manifest(manifest, strict=True)

    def test_reproducibility_records_racer_distributions_and_module_provider(self) -> None:
        reproducibility = build_reproducibility_manifest({}, project_root=ROOT)
        dependencies = reproducibility["runtime"]["dependencies"]
        self.assertIn("akracer", dependencies)
        self.assertIn("mini-racer", dependencies)
        self.assertIn("py-mini-racer", dependencies)
        module = reproducibility["runtime"]["modules"]["py_mini_racer"]
        self.assertIn("providers", module)
        self.assertIn("sha256", module)


class DataValidationTests(unittest.TestCase):
    def test_nan_price_is_rejected(self) -> None:
        bundle = make_demo_bundle(seed=1, start="2016-01-04", end="2018-12-31", symbols=25)
        bundle.bars.loc[bundle.bars.index[0], "open"] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            bundle.prepare()

    def test_invalid_ohlc_is_rejected(self) -> None:
        bundle = make_demo_bundle(seed=2, start="2016-01-04", end="2018-12-31", symbols=25)
        bundle.bars.loc[bundle.bars.index[0], "high"] = 0.01
        with self.assertRaisesRegex(ValueError, "OHLC"):
            bundle.prepare()

    def test_missing_member_bars_are_rejected(self) -> None:
        bundle = make_demo_bundle(seed=3, start="2016-01-04", end="2018-12-31", symbols=25)
        missing = bundle.membership["symbol"].iloc[0]
        bundle.bars = bundle.bars.loc[~bundle.bars["symbol"].eq(missing)]
        with self.assertRaisesRegex(ValueError, "缺少行情"):
            bundle.prepare()

    def test_benchmark_gap_is_rejected(self) -> None:
        bundle = make_demo_bundle(seed=4, start="2016-01-04", end="2018-12-31", symbols=25)
        bundle.benchmark = bundle.benchmark.drop(bundle.benchmark.index[100])
        with self.assertRaisesRegex(ValueError, "基准行情缺少"):
            bundle.prepare()

    def test_regime_gap_is_rejected(self) -> None:
        bundle = make_demo_bundle(seed=14, start="2016-01-04", end="2018-12-31", symbols=25)
        bundle.regime = bundle.regime.drop(bundle.regime.index[100])
        with self.assertRaisesRegex(ValueError, "择时指数行情缺少"):
            bundle.prepare()

    def test_regime_signal_does_not_use_performance_benchmark(self) -> None:
        bundle = make_demo_bundle(seed=15, start="2016-01-04", end="2018-12-31", symbols=25)
        bundle.benchmark["close"] = np.linspace(100.0, 300.0, len(bundle.benchmark))
        bundle.regime["close"] = np.linspace(300.0, 100.0, len(bundle.regime))
        prepared = bundle.prepare()
        strategy = MultiFactorStrategy(prepared, AppConfig().strategy)
        regime, exposure = strategy._regime_at(prepared.calendar[-1])
        self.assertEqual(regime, "RISK_OFF")
        self.assertEqual(exposure, AppConfig().strategy.risk_off_exposure)

    def test_membership_snapshot_gap_is_rejected(self) -> None:
        bundle = make_demo_bundle(seed=5, start="2016-01-04", end="2018-12-31", symbols=25)
        month = pd.Period("2017-06", freq="M")
        bundle.membership = bundle.membership.loc[
            ~bundle.membership["date"].dt.to_period("M").between(month, month + 2)
        ]
        with self.assertRaisesRegex(ValueError, "成分快照"):
            bundle.prepare()

    def test_missing_industry_history_is_rejected(self) -> None:
        bundle = make_demo_bundle(seed=6, start="2016-01-04", end="2018-12-31", symbols=25)
        missing = bundle.membership["symbol"].iloc[0]
        bundle.industry_membership = bundle.industry_membership.loc[
            ~bundle.industry_membership["symbol"].eq(missing)
        ]
        with self.assertRaisesRegex(ValueError, "行业成员表缺少"):
            bundle.prepare()

    def test_numeric_provider_dates_are_not_parsed_as_1970(self) -> None:
        bundle = make_demo_bundle(seed=8, start="2016-01-04", end="2018-12-31", symbols=25)
        bundle.securities["list_date"] = 20100101
        bundle.industry_membership["in_date"] = 20100101
        prepared = bundle.prepare()
        self.assertEqual(prepared.securities["list_date"].dt.year.unique().tolist(), [2010])
        self.assertEqual(
            prepared.industry_membership["in_date"].dt.year.unique().tolist(), [2010]
        )

    def test_industry_intervals_must_cover_member_snapshots(self) -> None:
        bundle = make_demo_bundle(seed=9, start="2016-01-04", end="2018-12-31", symbols=25)
        symbol = bundle.membership["symbol"].iloc[0]
        bundle.industry_membership.loc[
            bundle.industry_membership["symbol"].eq(symbol), "in_date"
        ] = pd.Timestamp("2018-01-01")
        with self.assertRaisesRegex(ValueError, "未覆盖"):
            bundle.prepare()

    def test_legacy_bar_without_size_data_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "600000_SH.csv.gz"
            pd.DataFrame({"date": ["2018-01-02", "2018-01-03"]}).to_csv(
                path, index=False, compression="gzip"
            )
            membership = pd.DataFrame(
                {"date": pd.to_datetime(["2018-01-31"]), "symbol": ["600000.SH"]}
            )
            usable = TushareDownloader._cached_bar_usable(
                path,
                "600000.SH",
                pd.Timestamp("2018-01-01"),
                pd.Timestamp("2018-02-01"),
                membership,
            )
            self.assertFalse(usable)

    def test_strict_cache_fingerprint_detects_tampering_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                root / "membership.csv.gz",
                root / "benchmark.csv.gz",
                root / "regime.csv.gz",
                root / "corporate_actions.csv.gz",
                root / "securities.csv.gz",
                root / "industry_membership.csv.gz",
                root / "calendar.csv.gz",
                root / "bars" / "600000_SH.csv.gz",
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"sealed-{path.name}".encode())
            files = build_file_inventory(root, paths)
            base = AppConfig()
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "provider": base.data.provider,
                        "universe_index": base.data.universe_index,
                        "regime_index": base.data.regime_index,
                        "benchmark_index": base.data.benchmark_index,
                        "industry_standard": base.data.industry_standard,
                        "industry_level": base.data.industry_level,
                        "requested_start": "2016-01-01",
                        "requested_end": "2025-12-31",
                        "files": files,
                        "data_fingerprint_sha256": inventory_sha256(files),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "身份与当前配置不匹配"):
                MarketDataBundle.from_cache(
                    root,
                    strict=False,
                    expected_config=replace(
                        base,
                        data=replace(base.data, universe_index="000905.SH"),
                    ),
                )
            with self.assertRaisesRegex(ValueError, "缓存日期范围不足"):
                MarketDataBundle.from_cache(
                    root,
                    strict=False,
                    expected_config=replace(
                        base,
                        backtest=replace(base.backtest, end_date="2026-12-31"),
                    ),
                )
            paths[-1].write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "完整性校验失败"):
                MarketDataBundle.from_cache(root, strict=False)
            paths[-1].write_bytes(f"sealed-{paths[-1].name}".encode())
            extra = root / "bars" / "600001_SH.csv.gz"
            extra.write_bytes(b"unsealed-extra")
            with self.assertRaisesRegex(ValueError, "文件集合不一致"):
                MarketDataBundle.from_cache(root, strict=False)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = make_demo_bundle(
            seed=11, start="2016-01-04", end="2020-12-31", symbols=30
        )
        cls.config = research_config()
        cls.result = Backtester(cls.bundle, cls.config).run()

    def test_signal_is_executed_later(self) -> None:
        trades = self.result.trades
        self.assertFalse(trades.empty)
        self.assertTrue(
            (pd.to_datetime(trades["date"]) > pd.to_datetime(trades["signal_date"])).all()
        )

    def test_buys_use_board_lots(self) -> None:
        buys = self.result.trades.loc[self.result.trades["side"].eq("BUY")]
        self.assertTrue(
            ((buys["shares"] % self.config.execution.lot_size).abs() < 1e-8).all()
        )

    def test_fee_model_uses_trade_date(self) -> None:
        trades = self.result.trades
        self.assertTrue(
            (trades["commission"] >= self.config.execution.minimum_commission).all()
        )
        sells = trades.loc[trades["side"].eq("SELL")]
        self.assertFalse(sells.empty)
        expected = sells.apply(
            lambda row: row["notional"]
            * self.config.execution.fee_on(pd.Timestamp(row["date"])).stamp_duty_sell,
            axis=1,
        )
        self.assertTrue(((sells["stamp_duty"] - expected).abs() < 1e-8).all())

    def test_target_weight_constraints(self) -> None:
        selections = self.result.selections
        self.assertLessEqual(
            float(selections["target_weight"].max()),
            self.config.strategy.max_stock_weight + 1e-12,
        )
        sums = selections.groupby("signal_date")["target_weight"].sum()
        self.assertTrue((sums <= self.config.strategy.risk_on_exposure + 1e-12).all())
        industry_sums = selections.groupby(
            ["signal_date", "industry_code"]
        )["target_weight"].sum()
        self.assertLessEqual(
            float(industry_sums.max()),
            self.config.strategy.max_industry_weight + 1e-12,
        )

    def test_rank_buffer_uses_actual_holdings(self) -> None:
        buffered = self.result.selections.loc[
            self.result.selections["selection_reason"].eq("HOLD_BUFFER")
        ]
        self.assertFalse(buffered.empty)
        self.assertTrue(buffered["was_held"].all())
        self.assertTrue((buffered["rank"] > self.config.strategy.top_n).any())

    def test_industry_lookup_is_point_in_time(self) -> None:
        symbol = self.bundle.industry_membership["symbol"].iloc[0]
        changed = make_demo_bundle(
            seed=11, start="2016-01-04", end="2020-12-31", symbols=30
        )
        changed.industry_membership = changed.industry_membership.loc[
            ~changed.industry_membership["symbol"].eq(symbol)
        ]
        changed.industry_membership = pd.concat(
            [
                changed.industry_membership,
                pd.DataFrame(
                    [
                        {
                            "symbol": symbol,
                            "industry_code": "OLD.SI",
                            "industry_name": "旧行业",
                            "in_date": "2010-01-01",
                            "out_date": "2018-06-30",
                        },
                        {
                            "symbol": symbol,
                            "industry_code": "NEW.SI",
                            "industry_name": "新行业",
                            "in_date": "2018-07-01",
                            "out_date": pd.NaT,
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        strategy = MultiFactorStrategy(changed, self.config.strategy)
        self.assertEqual(strategy._industry_at(symbol, pd.Timestamp("2018-06-29"))[0], "OLD.SI")
        self.assertEqual(strategy._industry_at(symbol, pd.Timestamp("2018-07-02"))[0], "NEW.SI")

    def test_asset_identity(self) -> None:
        curve = self.result.equity_curve
        reconstructed = curve["cash"] + curve["holdings_value"] + curve["dividend_receivables"]
        self.assertTrue(((curve["nav"] - reconstructed).abs() < 1e-7).all())

    def test_order_shares_are_frozen_at_signal_close(self) -> None:
        first_signal = pd.Timestamp(self.result.selections["signal_date"].min())
        selections = self.result.selections.loc[
            pd.to_datetime(self.result.selections["signal_date"]).eq(first_signal)
        ]
        orders = self.result.orders.loc[
            pd.to_datetime(self.result.orders["signal_date"]).eq(first_signal)
        ]
        row = selections.iloc[0]
        expected = np.floor(
            self.config.backtest.initial_cash
            * row["target_weight"]
            / row["signal_close"]
            / self.config.execution.lot_size
        ) * self.config.execution.lot_size
        target = orders.loc[orders["symbol"].eq(row["symbol"]), "target_shares"].iloc[0]
        self.assertEqual(float(target), float(expected))

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
        blocked = next(iter(plan.weights))
        execution_date = bundle.calendar[bundle.calendar.searchsorted(signal_date, side="right")]
        mask = bundle.bars["date"].eq(execution_date) & bundle.bars["symbol"].eq(blocked)
        bundle.bars.loc[mask, "open"] = bundle.bars.loc[mask, "up_limit"]
        bundle.bars.loc[mask, "high"] = bundle.bars.loc[mask, ["high", "open"]].max(axis=1)
        result = Backtester(bundle, research_config("2018-03-31")).run()
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

    def test_execution_day_st_is_rejected(self) -> None:
        bundle = make_demo_bundle(seed=19, start="2016-01-04", end="2019-06-30", symbols=30)
        signal_date = pd.Timestamp("2018-01-31")
        plan = MultiFactorStrategy(bundle, self.config.strategy).generate(signal_date)
        blocked = next(iter(plan.weights))
        execution_date = bundle.calendar[bundle.calendar.searchsorted(signal_date, side="right")]
        mask = bundle.bars["date"].eq(execution_date) & bundle.bars["symbol"].eq(blocked)
        bundle.bars.loc[mask, "is_st"] = True
        result = Backtester(bundle, research_config("2018-03-31")).run()
        rejected = result.orders.loc[
            result.orders["symbol"].eq(blocked)
            & result.orders["reason"].eq("st_on_execution")
        ]
        bought_that_day = result.trades.loc[
            result.trades["symbol"].eq(blocked)
            & result.trades["side"].eq("BUY")
            & pd.to_datetime(result.trades["date"]).eq(execution_date)
        ]
        self.assertFalse(rejected.empty)
        self.assertTrue(bought_that_day.empty)

    def test_matched_benchmark_metrics_are_reported(self) -> None:
        metrics = calculate_metrics(self.result, self.config)
        self.assertIn("matched_benchmark_cagr", metrics)
        self.assertIn("benchmark_max_drawdown", metrics)
        self.assertIsNotNone(metrics["matched_benchmark_cagr"])

    def test_deterministic_golden_metrics(self) -> None:
        metrics = calculate_metrics(self.result, self.config)
        self.assertAlmostEqual(float(metrics["final_nav"]), 870961.5596035972, places=4)
        self.assertAlmostEqual(float(metrics["cagr"]), 0.04196309744833271, places=10)
        self.assertAlmostEqual(
            float(metrics["max_drawdown"]), -0.0908917222450416, places=10
        )
        self.assertAlmostEqual(float(metrics["sharpe"]), 0.4919462883545638, places=10)
        self.assertAlmostEqual(
            float(metrics["annual_turnover"]), 1.9195314006881268, places=10
        )
        self.assertAlmostEqual(float(metrics["total_fees"]), 5630.1728582604455, places=4)
        self.assertEqual(int(metrics["filled_trade_count"]), 280)

    def test_exposure_tables_are_reported(self) -> None:
        industries = industry_exposure_table(self.result.selections)
        styles = style_exposure_table(self.result.selections)
        self.assertFalse(industries.empty)
        self.assertFalse(styles.empty)
        self.assertIn("stock_book_weight", industries)
        self.assertIn("z_size", styles)


class ResearchSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = make_demo_bundle(
            seed=41, start="2016-01-04", end="2020-12-31", symbols=24
        )
        base = research_config("2020-12-31")
        cls.config = replace(
            base,
            backtest=replace(base.backtest, start_date="2017-01-03"),
            strategy=replace(base.strategy, top_n=10, max_stock_weight=0.10),
        )

    def test_factor_ablation_has_full_and_removed_case(self) -> None:
        result = run_factor_ablation(self.bundle, self.config, factors=["liquidity"])
        self.assertEqual(set(result["variant"]), {"full", "without_liquidity"})
        self.assertTrue(result["cagr"].notna().all())

    def test_cost_stress_records_assumptions(self) -> None:
        result = run_cost_stress(
            self.bundle,
            self.config,
            slippage_bps=[10.0],
            commission_multipliers=[1.0],
        )
        self.assertEqual(set(result["slippage_bps"]), {5.0, 10.0})
        self.assertTrue((result["commission_multiplier"] == 1.0).all())

    def test_rolling_oos_uses_non_overlapping_windows(self) -> None:
        result = run_rolling_oos(
            self.bundle, self.config, train_years=2, test_years=1
        )
        self.assertGreaterEqual(len(result), 1)
        starts = pd.to_datetime(result["test_start"])
        ends = pd.to_datetime(result["test_end"])
        self.assertTrue((starts.iloc[1:].to_numpy() > ends.iloc[:-1].to_numpy()).all())


class CorporateActionTests(unittest.TestCase):
    def _selected_bundle(self, seed: int = 23):
        config = research_config("2018-03-30")
        bundle = make_demo_bundle(seed=seed, start="2016-01-04", end="2018-04-30", symbols=30)
        signal_date = pd.Timestamp("2018-01-31")
        plan = MultiFactorStrategy(bundle, config.strategy).generate(signal_date)
        return bundle, config, next(iter(plan.weights))

    def test_cash_and_stock_dividends_enter_ledger(self) -> None:
        bundle, config, symbol = self._selected_bundle(29)
        bundle.corporate_actions = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "record_date": "2018-02-02",
                    "ex_date": "2018-02-05",
                    "pay_date": "2018-02-07",
                    "stock_list_date": "2018-02-08",
                    "cash_dividend": 0.10,
                    "stock_dividend": 0.10,
                }
            ],
            columns=ACTION_COLUMNS,
        )
        result = Backtester(bundle, config).run()
        events = result.corporate_events.loc[result.corporate_events["symbol"].eq(symbol)]
        self.assertIn("CASH_DIVIDEND_RECEIVABLE", set(events["event"]))
        self.assertIn("CASH_DIVIDEND_PAID", set(events["event"]))
        self.assertIn("STOCK_DIVIDEND", set(events["event"]))
        paid = events.loc[events["event"].eq("CASH_DIVIDEND_PAID"), "amount"].sum()
        self.assertGreater(paid, 0)

    def test_delisted_position_is_not_carried_forever(self) -> None:
        bundle, config, symbol = self._selected_bundle(31)
        execution_date = pd.Timestamp("2018-02-01")
        delist_date = pd.Timestamp("2018-02-06")
        bundle.securities.loc[bundle.securities["symbol"].eq(symbol), "delist_date"] = delist_date
        bundle.securities.loc[bundle.securities["symbol"].eq(symbol), "list_status"] = "D"
        bundle.bars = bundle.bars.loc[
            ~(bundle.bars["symbol"].eq(symbol) & (bundle.bars["date"] > delist_date))
        ]
        result = Backtester(bundle, config).run()
        bought = result.trades.loc[
            result.trades["symbol"].eq(symbol)
            & result.trades["side"].eq("BUY")
            & pd.to_datetime(result.trades["date"]).eq(execution_date)
        ]
        events = result.corporate_events.loc[
            result.corporate_events["symbol"].eq(symbol)
            & result.corporate_events["event"].eq("DELISTED")
        ]
        self.assertFalse(bought.empty)
        self.assertFalse(events.empty)
        self.assertNotIn(symbol, result.final_positions)
        self.assertGreater(result.equity_curve["cumulative_delist_writeoff"].iloc[-1], 0)

    def test_stale_position_is_explicitly_reported(self) -> None:
        bundle, config, symbol = self._selected_bundle(37)
        config = replace(
            config,
            backtest=replace(config.backtest, maximum_stale_trading_days=3),
        )
        execution_date = pd.Timestamp("2018-02-01")
        bundle.bars = bundle.bars.loc[
            ~(bundle.bars["symbol"].eq(symbol) & (bundle.bars["date"] > execution_date))
        ]
        result = Backtester(bundle, config).run()
        self.assertTrue(any(symbol in warning and "陈旧价格" in warning for warning in result.warnings))
        self.assertGreater(result.equity_curve["stale_positions"].max(), 0)


if __name__ == "__main__":
    unittest.main()
