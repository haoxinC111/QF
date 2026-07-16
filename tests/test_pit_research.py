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

from ashare_quant import cli as cli_module
from ashare_quant.config import AppConfig
from ashare_quant.data import MarketDataBundle, make_demo_bundle
from ashare_quant.pit_data import (
    PointInTimeDataBundle,
    normalize_fundamental_source,
    normalize_valuations,
)
from ashare_quant.pit_research import (
    PIT_COMPOSITE_NAME,
    build_pit_factor_panel,
    calculate_factor_exposures,
    calculate_factor_ic,
    calculate_quantile_returns,
    compute_pit_factor_values,
    pit_factor_registry_frame,
    run_pit_factor_cost_stress,
    summarize_factor_ic,
    summarize_quantile_returns,
    write_pit_factor_research,
)
from ashare_quant.provenance import verify_artifact_manifest


def _make_research_fixture(
    root: Path,
) -> tuple[MarketDataBundle, PointInTimeDataBundle, AppConfig]:
    market = make_demo_bundle(
        seed=17,
        start="2017-01-02",
        end="2022-12-30",
        symbols=20,
    )
    symbols = sorted(market.bars["symbol"].unique())
    fundamental_rows: list[dict[str, object]] = []
    for year in range(2016, 2022):
        announcement = pd.Timestamp(year=year + 1, month=3, day=31)
        for index, symbol in enumerate(symbols):
            scale = index / max(1, len(symbols) - 1)
            fundamental_rows.append(
                {
                    "ts_code": symbol,
                    "ann_date": announcement.strftime("%Y%m%d"),
                    "end_date": f"{year}1231",
                    "update_flag": "0",
                    "roe": 5.0 + 20.0 * scale + (year - 2016) * 0.1,
                    "roa": 2.0 + 8.0 * scale,
                    "grossprofit_margin": 15.0 + 30.0 * scale,
                    "debt_to_assets": 75.0 - 30.0 * scale,
                    "ocf_to_or": 3.0 + 20.0 * scale,
                    "or_yoy": -5.0 + 25.0 * scale,
                    "netprofit_yoy": -10.0 + 35.0 * scale,
                }
            )
    fundamentals = normalize_fundamental_source(
        "fina_indicator",
        pd.DataFrame(fundamental_rows),
        market.calendar,
        lag_trading_days=1,
    )

    bar_lookup = market.bars.set_index(["date", "symbol"])
    valuation_rows: list[dict[str, object]] = []
    for date in sorted(market.membership["date"].unique()):
        stamp = pd.Timestamp(date)
        for index, symbol in enumerate(symbols):
            bar = bar_lookup.loc[(stamp, symbol)]
            scale = index / max(1, len(symbols) - 1)
            valuation_rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": stamp.strftime("%Y%m%d"),
                    "turnover_rate": 1.0 + scale,
                    "pe_ttm": 30.0 - 18.0 * scale,
                    "pb": 4.0 - 2.5 * scale,
                    "ps_ttm": 6.0 - 4.0 * scale,
                    "dv_ttm": 0.5 + 3.0 * scale,
                    "total_mv": float(bar["total_mv"]),
                    "circ_mv": float(bar["circ_mv"]),
                }
            )
    valuations = normalize_valuations(
        pd.DataFrame(valuation_rows), market.calendar, lag_trading_days=0
    )

    base_cache = root / "base"
    pit_cache = root / "pit"
    base_cache.mkdir(parents=True)
    pit_cache.mkdir(parents=True)
    base_manifest = {
        "schema_version": 4,
        "provider": "synthetic_demo",
        "data_fingerprint_sha256": "base-fingerprint-alpha2",
    }
    pit_manifest = {
        "schema_version": 1,
        "provider": "synthetic_demo",
        "data_fingerprint_sha256": "pit-fingerprint-alpha2",
        "base_data_fingerprint_sha256": "base-fingerprint-alpha2",
    }
    (base_cache / "manifest.json").write_text(
        json.dumps(base_manifest), encoding="utf-8"
    )
    (pit_cache / "manifest.json").write_text(
        json.dumps(pit_manifest), encoding="utf-8"
    )
    pit = PointInTimeDataBundle(
        fundamentals=fundamentals,
        valuations=valuations,
        calendar=market.calendar,
        manifest=pit_manifest,
    ).prepare()
    base = AppConfig()
    config = replace(
        base,
        data=replace(
            base.data,
            provider="synthetic_demo",
            cache_dir=str(base_cache),
        ),
        point_in_time=replace(
            base.point_in_time,
            enabled=True,
            cache_dir=str(pit_cache),
        ),
        backtest=replace(
            base.backtest,
            start_date="2018-01-01",
            end_date="2021-12-31",
        ),
    )
    return market, pit, config


class PITFactorDefinitionTests(unittest.TestCase):
    def test_registry_is_research_only_and_contains_fixed_composite(self) -> None:
        registry = pit_factor_registry_frame()
        self.assertIn(PIT_COMPOSITE_NAME, set(registry["factor"]))
        self.assertEqual(set(registry["lifecycle_status"]), {"research_only"})
        self.assertEqual(float(registry["production_weight"].sum()), 0.0)

    def test_factor_transforms_reject_nonpositive_valuation_multiples(self) -> None:
        snapshot = pd.DataFrame(
            {
                "symbol": ["600000.SH", "600001.SH"],
                "roe_pct": [12.0, 8.0],
                "debt_to_assets_pct": [40.0, 70.0],
                "pe_ttm": [10.0, -5.0],
                "pb": [2.0, 0.0],
            }
        )
        values = compute_pit_factor_values(
            snapshot,
            [
                "roe_quality",
                "low_leverage_quality",
                "earnings_yield",
                "book_to_price",
            ],
        )
        self.assertAlmostEqual(float(values.iloc[0]["earnings_yield"]), 0.1)
        self.assertTrue(pd.isna(values.iloc[1]["earnings_yield"]))
        self.assertTrue(pd.isna(values.iloc[1]["book_to_price"]))
        self.assertEqual(
            values["low_leverage_quality"].tolist(), [-40.0, -70.0]
        )


class PITFactorPanelTests(unittest.TestCase):
    def test_future_pit_revisions_cannot_change_past_factor_panel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            market, pit, _ = _make_research_fixture(Path(directory))
            kwargs = {
                "start_date": "2018-01-01",
                "end_date": "2019-12-31",
                "factor_names": ["roe_quality", "earnings_yield"],
                "horizons": (21,),
                "minimum_factors_per_symbol": 2,
            }
            baseline = build_pit_factor_panel(market, pit, **kwargs)
            changed_fundamentals = pit.fundamentals.copy()
            future = changed_fundamentals["available_date"].gt("2020-01-01")
            changed_fundamentals.loc[future, "value"] *= 1000.0
            changed_valuations = pit.valuations.copy()
            future_valuation = changed_valuations["available_date"].gt(
                "2020-01-01"
            )
            changed_valuations.loc[future_valuation, "pe_ttm"] = 0.01
            perturbed = PointInTimeDataBundle(
                fundamentals=changed_fundamentals,
                valuations=changed_valuations,
                calendar=pit.calendar,
                manifest=pit.manifest,
            ).prepare()
            replay = build_pit_factor_panel(market, perturbed, **kwargs)
            compared = [
                "signal_date",
                "symbol",
                "roe_quality",
                "earnings_yield",
                "z_roe_quality",
                "z_earnings_yield",
                f"z_{PIT_COMPOSITE_NAME}",
            ]
            pd.testing.assert_frame_equal(
                baseline[compared], replay[compared], check_exact=True
            )

    def test_panel_uses_month_end_members_and_future_total_returns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            market, pit, _ = _make_research_fixture(Path(directory))
            panel = build_pit_factor_panel(
                market,
                pit,
                start_date="2018-01-01",
                end_date="2018-12-31",
                factor_names=["roe_quality", "earnings_yield"],
                horizons=(21, 63),
                minimum_factors_per_symbol=2,
            )
            self.assertEqual(panel["signal_date"].nunique(), 12)
            self.assertEqual(panel.groupby("signal_date")["symbol"].nunique().min(), 20)
            self.assertTrue(panel["forward_return_21d"].notna().all())
            self.assertTrue(
                panel[f"z_{PIT_COMPOSITE_NAME}"].notna().all()
            )
            self.assertTrue(
                panel["roe_quality_available_date"].le(
                    panel["signal_date"]
                ).all()
            )
            self.assertTrue(
                panel["roe_quality_source_row_sha256"].str.fullmatch(
                    r"[0-9a-f]{64}"
                ).all()
            )


class PITFactorDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        rows: list[dict[str, object]] = []
        for month in range(1, 5):
            date = pd.Timestamp(year=2020, month=month, day=28)
            for index in range(20):
                score = (index - 9.5) / 5.0
                rows.append(
                    {
                        "signal_date": date,
                        "symbol": f"{600000 + index:06d}.SH",
                        "universe_size": 20,
                        "industry_code": f"IND{index % 4}",
                        "log_total_market_value": float(index % 5),
                        "z_roe_quality": score,
                        f"z_{PIT_COMPOSITE_NAME}": score,
                        "forward_return_21d": score * 0.01,
                    }
                )
        self.panel = pd.DataFrame(rows)

    def test_ic_and_quantile_spread_detect_monotonic_signal(self) -> None:
        ic = calculate_factor_ic(
            self.panel,
            ["roe_quality"],
            [21],
            minimum_observations=10,
        )
        summary = summarize_factor_ic(ic)
        composite = summary.loc[
            summary["factor"].eq(PIT_COMPOSITE_NAME)
        ].iloc[0]
        self.assertAlmostEqual(float(composite["mean_ic"]), 1.0)
        quantiles = calculate_quantile_returns(
            self.panel, ["roe_quality"], [21], quantiles=5
        )
        quantile_summary = summarize_quantile_returns(quantiles, quantiles=5)
        composite_spread = quantile_summary.loc[
            quantile_summary["factor"].eq(PIT_COMPOSITE_NAME),
            "top_minus_bottom_mean_return",
        ].iloc[0]
        self.assertGreater(float(composite_spread), 0.0)

    def test_exposure_diagnostics_and_cost_ledger_are_explicit(self) -> None:
        exposure = calculate_factor_exposures(
            self.panel, ["roe_quality"]
        )
        self.assertEqual(len(exposure), 8)
        self.assertTrue(exposure["industry_r_squared"].between(0, 1).all())
        cost = run_pit_factor_cost_stress(
            self.panel,
            ["roe_quality"],
            horizon=21,
            quantiles=5,
            cost_bps=(20.0,),
        )
        first = cost.loc[cost["factor"].eq(PIT_COMPOSITE_NAME)].iloc[0]
        self.assertAlmostEqual(
            float(first["estimated_cost"]),
            float(first["two_sided_weight_turnover"]) * 20.0 / 10_000.0,
        )
        missing_outcome = self.panel.copy()
        first_date = missing_outcome["signal_date"].min()
        top_index = missing_outcome.loc[
            missing_outcome["signal_date"].eq(first_date),
            f"z_{PIT_COMPOSITE_NAME}",
        ].idxmax()
        missing_outcome.loc[top_index, "forward_return_21d"] = np.nan
        guarded = run_pit_factor_cost_stress(
            missing_outcome,
            ["roe_quality"],
            horizon=21,
            quantiles=5,
            cost_bps=(20.0,),
        )
        guarded_first = guarded.loc[
            guarded["factor"].eq(PIT_COMPOSITE_NAME)
        ].iloc[0]
        self.assertEqual(int(guarded_first["holdings"]), 4)
        self.assertEqual(int(guarded_first["realized_outcomes"]), 3)
        self.assertAlmostEqual(float(guarded_first["outcome_coverage"]), 0.75)


class PITFactorResearchWriterTests(unittest.TestCase):
    def test_writer_binds_both_data_fingerprints_and_seals_every_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market, pit, config = _make_research_fixture(root)
            output = root / "research"
            written = write_pit_factor_research(
                market,
                pit,
                config,
                output,
                factor_names=[
                    "roe_quality",
                    "roa_quality",
                    "cashflow_quality",
                    "low_leverage_quality",
                    "earnings_yield",
                ],
                horizons=(21,),
                quantiles=5,
                minimum_factors_per_symbol=3,
                minimum_ic_observations=10,
                cost_bps=(5.0, 20.0),
                train_years=1,
                test_years=1,
            )
            verification = verify_artifact_manifest(
                written["artifacts"], strict=True
            )
            self.assertEqual(verification["unsealed_paths"], [])
            manifest = json.loads(written["manifest"].read_text("utf-8"))
            reproducibility = json.loads(
                written["reproducibility"].read_text("utf-8")
            )
            governance = json.loads(
                written["governance"].read_text("utf-8")
            )
            self.assertEqual(
                manifest["base_data_fingerprint_sha256"],
                "base-fingerprint-alpha2",
            )
            self.assertEqual(
                manifest["pit_data_fingerprint_sha256"],
                "pit-fingerprint-alpha2",
            )
            self.assertEqual(
                reproducibility["pit_data"]["data_fingerprint_sha256"],
                "pit-fingerprint-alpha2",
            )
            self.assertFalse(governance["production_default_changed"])
            self.assertEqual(
                governance["promotion_decision"], "nonproduction_data_only"
            )
            cost_stress = pd.read_csv(written["cost_stress"])
            self.assertEqual(set(cost_stress["return_label"]), {"next_signal_date"})
            with self.assertRaisesRegex(FileExistsError, "非空"):
                write_pit_factor_research(
                    market,
                    pit,
                    config,
                    output,
                    factor_names=["roe_quality", "earnings_yield"],
                    horizons=(21,),
                    minimum_factors_per_symbol=2,
                    minimum_ic_observations=10,
                    train_years=1,
                )

    def test_cli_exposes_research_only_command(self) -> None:
        args = cli_module._parser().parse_args(
            [
                "pit-research",
                "--config",
                "config.yaml",
                "--factors",
                "roe_quality,earnings_yield",
                "--horizons",
                "21",
            ]
        )
        self.assertEqual(args.command, "pit-research")
        self.assertEqual(args.minimum_factors_per_symbol, 4)
        self.assertEqual(args.cost_bps, "5,10,20")


if __name__ == "__main__":
    unittest.main()
