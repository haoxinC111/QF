from __future__ import annotations

import json
import inspect
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare_quant import cli as cli_module
from ashare_quant.pit_data import PointInTimeDataBundle
from ashare_quant.pit_research import (
    PIT_COMPOSITE_NAME,
    write_pit_factor_research,
)
from ashare_quant.pit_shadow import (
    COVERAGE_MATCHED_PRICE_ARM,
    HYBRID_ARM,
    PIT_ONLY_ARM,
    PIT_SHADOW_ARMS,
    PITShadowScorePanel,
    PITShadowStrategy,
    build_pit_shadow_score_panel,
    validate_alpha2_research_bundle,
    write_pit_shadow_research,
)
from ashare_quant.provenance import verify_artifact_manifest
from test_pit_research import _make_research_fixture


class PITShadowScoreTests(unittest.TestCase):
    def test_score_panel_has_no_outcome_and_future_rows_cannot_change_past(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            market, pit, config = _make_research_fixture(Path(directory))
            config = replace(
                config,
                backtest=replace(
                    config.backtest,
                    start_date="2018-01-01",
                    end_date="2019-12-31",
                ),
            )
            baseline = build_pit_shadow_score_panel(market, pit, config)
            self.assertFalse(
                any("forward_return" in column for column in baseline.columns)
            )
            changed_fundamentals = pit.fundamentals.copy()
            future = changed_fundamentals["available_date"].gt("2020-01-01")
            changed_fundamentals.loc[future, "value"] *= 10_000.0
            changed_valuations = pit.valuations.copy()
            future_valuation = changed_valuations["available_date"].gt("2020-01-01")
            changed_valuations.loc[future_valuation, "pe_ttm"] = 0.001
            perturbed = PointInTimeDataBundle(
                fundamentals=changed_fundamentals,
                valuations=changed_valuations,
                calendar=pit.calendar,
                manifest=pit.manifest,
            ).prepare()
            replay = build_pit_shadow_score_panel(market, perturbed, config)
            compared = [
                "signal_date",
                "symbol",
                "available_factor_count",
                PIT_COMPOSITE_NAME,
                f"z_{PIT_COMPOSITE_NAME}",
            ]
            pd.testing.assert_frame_equal(
                baseline[compared], replay[compared], check_exact=True
            )

    def test_score_panel_rejects_duplicate_and_future_identity(self) -> None:
        frame = pd.DataFrame(
            {
                "signal_date": ["2020-01-31"],
                "as_of_date": ["2020-02-03"],
                "symbol": ["600000.SH"],
                "universe_size": [1],
                "available_factor_count": [4],
                f"z_{PIT_COMPOSITE_NAME}": [1.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "晚于信号日"):
            PITShadowScorePanel(frame)
        valid = frame.assign(as_of_date="2020-01-31")
        with self.assertRaisesRegex(ValueError, "重复"):
            PITShadowScorePanel(pd.concat([valid, valid], ignore_index=True))
        with self.assertRaisesRegex(ValueError, "未来收益标签"):
            PITShadowScorePanel(valid.assign(forward_return_21d=0.10))
        with self.assertRaisesRegex(ValueError, "非有限"):
            PITShadowScorePanel(
                valid.assign(**{f"z_{PIT_COMPOSITE_NAME}": float("inf")})
            )
        with self.assertRaisesRegex(ValueError, "无效日期"):
            PITShadowScorePanel(valid.assign(valuation_available_date="not-a-date"))

    def test_all_shadow_arms_use_only_coverage_eligible_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            market, pit, config = _make_research_fixture(Path(directory))
            config = replace(
                config,
                backtest=replace(
                    config.backtest,
                    start_date="2018-01-01",
                    end_date="2018-12-31",
                ),
            )
            frame = build_pit_shadow_score_panel(market, pit, config)
            first_date = pd.Timestamp(frame["signal_date"].min())
            first_symbols = frame.loc[
                frame["signal_date"].eq(first_date), "symbol"
            ].head(3)
            frame.loc[
                frame["signal_date"].eq(first_date)
                & frame["symbol"].isin(first_symbols),
                f"z_{PIT_COMPOSITE_NAME}",
            ] = pd.NA
            panel = PITShadowScorePanel(frame)
            for arm in [COVERAGE_MATCHED_PRICE_ARM, PIT_ONLY_ARM, HYBRID_ARM]:
                strategy = PITShadowStrategy(market, config, panel, arm=arm)
                plan = strategy.generate(first_date)
                if plan.selection.empty:
                    continue
                self.assertTrue(plan.selection["pit_coverage_eligible"].all())
                self.assertFalse(plan.selection["symbol"].isin(first_symbols).any())
                self.assertEqual(set(plan.selection["shadow_arm"]), {arm})
                self.assertTrue(
                    plan.selection["pit_score_signal_date"].eq(first_date).all()
                )

                formula_symbols = (
                    panel.at(first_date)
                    .dropna(subset=[f"z_{PIT_COMPOSITE_NAME}"])["symbol"]
                    .head(3)
                    .tolist()
                )
                candidates = pd.DataFrame(
                    {
                        "symbol": formula_symbols,
                        "score_pre_neutral": [1.0, 2.0, 4.0][: len(formula_symbols)],
                    }
                )
                overlaid = strategy._apply_research_score_overlay(
                    candidates, first_date
                )
                if arm == COVERAGE_MATCHED_PRICE_ARM:
                    expected = overlaid["z_price_shadow"]
                    expected_weight = 0.0
                elif arm == PIT_ONLY_ARM:
                    expected = overlaid["z_pit_shadow"]
                    expected_weight = 1.0
                else:
                    expected = (
                        0.75 * overlaid["z_price_shadow"]
                        + 0.25 * overlaid["z_pit_shadow"]
                    )
                    expected_weight = 0.25
                self.assertTrue(
                    overlaid["score_pre_neutral"].sub(expected).abs().lt(1e-12).all()
                )
                self.assertTrue(overlaid["shadow_pit_weight"].eq(expected_weight).all())


class PITShadowWriterTests(unittest.TestCase):
    def test_writer_runs_four_strict_ledger_arms_and_seals_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market, pit, config = _make_research_fixture(root)
            config = replace(
                config,
                backtest=replace(
                    config.backtest,
                    start_date="2018-01-01",
                    end_date="2019-12-31",
                ),
            )
            alpha2_dir = root / "alpha2"
            write_pit_factor_research(
                market,
                pit,
                config,
                alpha2_dir,
                horizons=(21,),
                minimum_ic_observations=10,
                cost_bps=(5.0,),
                train_years=1,
                test_years=1,
            )
            output = root / "alpha3"
            written = write_pit_shadow_research(
                market,
                pit,
                config,
                output,
                alpha2_research_dir=alpha2_dir,
                cost_bps=(5.0,),
            )
            verification = verify_artifact_manifest(written["artifacts"], strict=True)
            self.assertEqual(verification["unsealed_paths"], [])
            comparison = pd.read_csv(written["comparison"])
            self.assertEqual(set(comparison["arm"]), set(PIT_SHADOW_ARMS))
            manifest = json.loads(written["manifest"].read_text("utf-8"))
            governance = json.loads(written["governance"].read_text("utf-8"))
            reproducibility = json.loads(written["reproducibility"].read_text("utf-8"))
            self.assertFalse(manifest["production_strategy_changed"])
            self.assertEqual(manifest["candidate_production_weight"], 0.0)
            self.assertEqual(
                governance["promotion_decision"], "nonproduction_data_only"
            )
            self.assertEqual(
                reproducibility["pit_data"]["data_fingerprint_sha256"],
                "pit-fingerprint-alpha2",
            )
            for arm in PIT_SHADOW_ARMS:
                metrics = json.loads(written[f"{arm}_metrics"].read_text("utf-8"))
                if arm == "production_baseline":
                    self.assertEqual(
                        metrics["effective_alpha_lifecycle_status"], "promoted"
                    )
                else:
                    self.assertEqual(
                        metrics["effective_alpha_lifecycle_status"],
                        "research_only",
                    )
            with self.assertRaisesRegex(FileExistsError, "非空"):
                write_pit_shadow_research(
                    market,
                    pit,
                    config,
                    output,
                    alpha2_research_dir=alpha2_dir,
                    cost_bps=(5.0,),
                )
            with self.assertRaisesRegex(ValueError, "有限非负数"):
                write_pit_shadow_research(
                    market,
                    pit,
                    config,
                    root / "invalid-cost",
                    alpha2_research_dir=alpha2_dir,
                    cost_bps=(float("nan"),),
                )
            pit_manifest_path = Path(config.point_in_time.cache_dir) / "manifest.json"
            pit_manifest = json.loads(pit_manifest_path.read_text("utf-8"))
            pit_manifest["provider"] = "unexpected-provider"
            pit_manifest_path.write_text(json.dumps(pit_manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provider"):
                write_pit_shadow_research(
                    market,
                    pit,
                    config,
                    root / "provider-mismatch",
                    alpha2_research_dir=alpha2_dir,
                    cost_bps=(5.0,),
                )

    def test_alpha2_bundle_rejects_unsealed_or_nonfixed_research(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market, pit, config = _make_research_fixture(root)
            alpha2_dir = root / "alpha2"
            write_pit_factor_research(
                market,
                pit,
                config,
                alpha2_dir,
                factor_names=["roe_quality", "earnings_yield"],
                horizons=(21,),
                minimum_factors_per_symbol=2,
                minimum_ic_observations=10,
                cost_bps=(5.0,),
                train_years=1,
                test_years=1,
            )
            with self.assertRaisesRegex(ValueError, "固定候选契约"):
                validate_alpha2_research_bundle(
                    alpha2_dir,
                    base_data_fingerprint="base-fingerprint-alpha2",
                    pit_data_fingerprint="pit-fingerprint-alpha2",
                )
            (alpha2_dir / "unknown.txt").write_text("unsealed", "utf-8")
            with self.assertRaisesRegex(ValueError, "未封存"):
                validate_alpha2_research_bundle(
                    alpha2_dir,
                    base_data_fingerprint="base-fingerprint-alpha2",
                    pit_data_fingerprint="pit-fingerprint-alpha2",
                )

    def test_cli_exposes_fixed_alpha3_shadow_protocol(self) -> None:
        args = cli_module._parser().parse_args(
            [
                "pit-shadow",
                "--config",
                "config.yaml",
                "--alpha2-research",
                "results/alpha2",
            ]
        )
        self.assertEqual(args.command, "pit-shadow")
        self.assertEqual(args.cost_bps, "5,10,20")
        self.assertFalse(hasattr(args, "hybrid_pit_weight"))
        self.assertNotIn(
            "blend_weight", inspect.signature(PITShadowStrategy).parameters
        )


if __name__ == "__main__":
    unittest.main()
