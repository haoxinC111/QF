from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ashare_quant.cli as cli_module
from ashare_quant import __version__
from ashare_quant.config import AppConfig, CONFIG_SCHEMA_VERSION
from ashare_quant.pit_data import (
    FUNDAMENTAL_SOURCE_FIELDS,
    PointInTimeDataBundle,
    TusharePointInTimeDownloader,
    availability_dates,
    normalize_fundamental_source,
    normalize_valuations,
    verify_pit_cache,
    write_pit_cache,
)
from ashare_quant.provenance import build_file_inventory, inventory_sha256


class PointInTimeDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calendar = pd.bdate_range("2019-12-20", "2020-06-30")
        self.raw_income = pd.DataFrame(
            [
                {
                    "ts_code": "600000.SH",
                    "ann_date": "20200331",
                    "f_ann_date": "20200331",
                    "end_date": "20191231",
                    "report_type": "1",
                    "comp_type": "1",
                    "update_flag": "0",
                    "revenue": 100.0,
                    "n_income_attr_p": 10.0,
                },
                {
                    "ts_code": "600000.SH",
                    "ann_date": "20200331",
                    "f_ann_date": "20200415",
                    "end_date": "20191231",
                    "report_type": "1",
                    "comp_type": "1",
                    "update_flag": "1",
                    "revenue": 110.0,
                    "n_income_attr_p": 11.0,
                },
                {
                    "ts_code": "600000.SH",
                    "ann_date": "20200430",
                    "f_ann_date": "20200430",
                    "end_date": "20200331",
                    "report_type": "1",
                    "comp_type": "1",
                    "update_flag": "0",
                    "revenue": 30.0,
                    "n_income_attr_p": 3.0,
                },
            ]
        )
        self.fundamentals = normalize_fundamental_source(
            "income", self.raw_income, self.calendar, 1
        )
        self.valuations = normalize_valuations(
            pd.DataFrame(
                [
                    {
                        "ts_code": "600000.SH",
                        "trade_date": "20200331",
                        "turnover_rate": 1.2,
                        "pe_ttm": 8.0,
                        "pb": 0.9,
                        "ps_ttm": 1.1,
                        "dv_ttm": 2.0,
                        "total_mv": 1000.0,
                        "circ_mv": 800.0,
                    },
                    {
                        "ts_code": "600000.SH",
                        "trade_date": "20200401",
                        "turnover_rate": 1.3,
                        "pe_ttm": 8.2,
                        "pb": 0.92,
                        "ps_ttm": 1.12,
                        "dv_ttm": 2.1,
                        "total_mv": 1020.0,
                        "circ_mv": 810.0,
                    },
                ]
            ),
            self.calendar,
            0,
        )
        self.bundle = PointInTimeDataBundle(
            fundamentals=self.fundamentals,
            valuations=self.valuations,
            calendar=self.calendar,
        ).prepare()

    def test_fundamental_announcement_is_not_visible_on_same_day(self) -> None:
        same_day = self.bundle.fundamental_snapshot("2020-03-31")
        next_day = self.bundle.fundamental_snapshot("2020-04-01")
        self.assertTrue(same_day.empty)
        self.assertEqual(float(next_day.iloc[0]["revenue"]), 100.0)

    def test_revision_only_changes_snapshot_after_its_available_date(self) -> None:
        before = self.bundle.visible_fundamentals("2020-04-15")
        after = self.bundle.visible_fundamentals("2020-04-16")
        before_revenue = before.loc[
            before["metric"].eq("revenue"), "value"
        ].iloc[0]
        after_revenue = after.loc[
            after["metric"].eq("revenue"), "value"
        ].iloc[0]
        self.assertEqual(float(before_revenue), 100.0)
        self.assertEqual(float(after_revenue), 110.0)

    def test_revision_sequence_ignores_source_rows_without_supported_values(
        self,
    ) -> None:
        raw = pd.DataFrame(
            [
                {
                    "ts_code": "600000.SH",
                    "ann_date": "20200331",
                    "end_date": "20191231",
                    "report_type": "1",
                    "comp_type": "1",
                    "update_flag": "0",
                    "revenue": None,
                },
                {
                    "ts_code": "600000.SH",
                    "ann_date": "20200415",
                    "end_date": "20191231",
                    "report_type": "1",
                    "comp_type": "1",
                    "update_flag": "1",
                    "revenue": 110.0,
                },
            ]
        )
        normalized = normalize_fundamental_source(
            "income", raw, self.calendar, 1
        )
        self.assertEqual(normalized["revision_sequence"].unique().tolist(), [1])
        replace(self.bundle, fundamentals=normalized).prepare(strict=True)

    def test_late_old_period_revision_does_not_replace_newer_period(self) -> None:
        snapshot = self.bundle.fundamental_snapshot("2020-05-04")
        self.assertEqual(float(snapshot.iloc[0]["revenue"]), 30.0)

    def test_future_value_mutation_cannot_change_past_snapshot(self) -> None:
        original = self.bundle.fundamental_snapshot("2020-04-01")
        changed = self.bundle.fundamentals.copy()
        changed.loc[
            changed["available_date"].gt("2020-04-01"), "value"
        ] *= 1_000.0
        perturbed = replace(self.bundle, fundamentals=changed)
        replay = perturbed.fundamental_snapshot("2020-04-01")
        pd.testing.assert_frame_equal(original, replay)

    def test_prepare_rejects_forged_early_available_date(self) -> None:
        forged = self.bundle.fundamentals.copy()
        forged.loc[forged.index[0], "available_date"] = forged.loc[
            forged.index[0], "announcement_date"
        ]
        with self.assertRaisesRegex(ValueError, "滞后规则"):
            replace(self.bundle, fundamentals=forged).prepare()

    def test_prepare_rejects_unknown_metric_unit_contract(self) -> None:
        malformed = self.bundle.fundamentals.copy()
        malformed.loc[malformed.index[0], "unit"] = "mystery_unit"
        with self.assertRaisesRegex(ValueError, "未知来源"):
            replace(self.bundle, fundamentals=malformed).prepare()

    def test_valuation_snapshot_obeys_date_and_staleness(self) -> None:
        snapshot = self.bundle.valuation_snapshot("2020-04-01")
        self.assertEqual(float(snapshot.iloc[0]["pe_ttm"]), 8.2)
        stale = self.bundle.valuation_snapshot(
            "2020-04-20", maximum_age_days=5
        )
        self.assertTrue(stale.empty)

    def test_combined_snapshot_joins_only_information_visible_then(self) -> None:
        snapshot = self.bundle.snapshot(
            "2020-04-01",
            maximum_fundamental_age_days=550,
            maximum_valuation_age_days=10,
        )
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(float(snapshot.iloc[0]["revenue"]), 100.0)
        self.assertEqual(float(snapshot.iloc[0]["pe_ttm"]), 8.2)
        self.assertEqual(
            snapshot.iloc[0]["valuation_date"], pd.Timestamp("2020-04-01")
        )

    def test_weekend_announcement_maps_to_next_trading_day(self) -> None:
        calendar = pd.to_datetime(["2020-04-03", "2020-04-06", "2020-04-07"])
        mapped = availability_dates(["2020-04-04"], calendar, 1)
        self.assertEqual(mapped.iloc[0], pd.Timestamp("2020-04-06"))

    def test_cache_round_trip_seals_base_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_manifest = root / "base_manifest.json"
            base_manifest.write_text(
                json.dumps({"data_fingerprint_sha256": "base-data-001"}),
                encoding="utf-8",
            )
            cache = (root / "pit").resolve()
            manifest_path = write_pit_cache(
                self.bundle,
                cache,
                provider="tushare",
                requested_start="2019-01-01",
                requested_end="2020-06-30",
                expected_symbols=["600000.SH"],
                base_manifest_path=base_manifest,
            )
            manifest = verify_pit_cache(
                cache, base_manifest_path=base_manifest
            )
            restored = PointInTimeDataBundle.from_cache(
                cache, base_manifest_path=base_manifest
            )
            self.assertTrue(manifest["verification"]["verified"])
            self.assertEqual(manifest_path, cache / "manifest.json")
            self.assertEqual(
                restored.manifest["base_data_fingerprint_sha256"],
                "base-data-001",
            )
            pd.testing.assert_frame_equal(
                self.bundle.fundamental_snapshot("2020-04-01"),
                restored.fundamental_snapshot("2020-04-01"),
            )

    def test_same_logical_snapshot_has_deterministic_data_fingerprint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = []
            for name in ["first", "second"]:
                path = write_pit_cache(
                    self.bundle,
                    root / name,
                    provider="synthetic",
                    requested_start="2019-01-01",
                    requested_end="2020-06-30",
                    expected_symbols=["600000.SH"],
                )
                manifests.append(json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(
                manifests[0]["data_fingerprint_sha256"],
                manifests[1]["data_fingerprint_sha256"],
            )

    def test_cache_detects_data_and_base_manifest_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_manifest = root / "base_manifest.json"
            base_manifest.write_text(
                json.dumps({"data_fingerprint_sha256": "base-data-001"}),
                encoding="utf-8",
            )
            cache = root / "pit"
            write_pit_cache(
                self.bundle,
                cache,
                provider="tushare",
                requested_start="2019-01-01",
                requested_end="2020-06-30",
                expected_symbols=["600000.SH"],
                base_manifest_path=base_manifest,
            )
            valuation_path = next((cache / "valuations").glob("*.csv.gz"))
            original = valuation_path.read_bytes()
            valuation_path.write_bytes(original + b"tampered")
            with self.assertRaisesRegex(ValueError, "完整性校验失败"):
                PointInTimeDataBundle.from_cache(cache)
            valuation_path.write_bytes(original)
            base_manifest.write_text(
                json.dumps({"data_fingerprint_sha256": "base-data-002"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "manifest 已变化"):
                PointInTimeDataBundle.from_cache(
                    cache, base_manifest_path=base_manifest
                )

    def test_cache_rejects_unsealed_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "pit"
            write_pit_cache(
                self.bundle,
                cache,
                provider="synthetic",
                requested_start="2019-01-01",
                requested_end="2020-06-30",
                expected_symbols=["600000.SH"],
            )
            (cache / "unsealed.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "文件集合不一致"):
                PointInTimeDataBundle.from_cache(cache)

    def test_cache_rejects_sealed_partition_with_wrong_symbol_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = (Path(directory) / "pit").resolve()
            manifest_path = write_pit_cache(
                self.bundle,
                cache,
                provider="synthetic",
                requested_start="2019-01-01",
                requested_end="2020-06-30",
                expected_symbols=["600000.SH"],
            )
            original = cache / "fundamentals" / "600000_SH.csv.gz"
            renamed = cache / "fundamentals" / "600001_SH.csv.gz"
            os.replace(original, renamed)
            paths = [
                path
                for path in cache.rglob("*")
                if path.is_file() and path != manifest_path
            ]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"] = build_file_inventory(cache, paths)
            manifest["data_fingerprint_sha256"] = inventory_sha256(
                manifest["files"]
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "证券分区"):
                PointInTimeDataBundle.from_cache(cache)

    def test_cache_overwrite_refuses_unknown_user_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "pit"
            write_pit_cache(
                self.bundle,
                cache,
                provider="synthetic",
                requested_start="2019-01-01",
                requested_end="2020-06-30",
            )
            (cache / "keep-me.txt").write_text("user data", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "未知文件"):
                write_pit_cache(
                    self.bundle,
                    cache,
                    provider="synthetic",
                    requested_start="2019-01-01",
                    requested_end="2020-06-30",
                    overwrite=True,
                )

    def test_cache_coverage_is_enforced_by_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "pit"
            write_pit_cache(
                self.bundle,
                cache,
                provider="tushare",
                requested_start="2010-01-01",
                requested_end="2025-12-31",
                expected_symbols=["600000.SH", "600001.SH"],
            )
            config = replace(
                AppConfig(),
                point_in_time=replace(
                    AppConfig().point_in_time,
                    minimum_symbol_coverage=0.75,
                ),
            )
            with self.assertRaisesRegex(ValueError, "证券覆盖不足"):
                PointInTimeDataBundle.from_cache(
                    cache, expected_config=config
                )

    def test_downloader_uses_endpoint_specific_fundamental_fields(self) -> None:
        calls: dict[str, dict[str, object]] = {}

        class EmptyClient:
            def call(self, endpoint: str, **kwargs: object) -> pd.DataFrame:
                calls[endpoint] = dict(kwargs)
                return pd.DataFrame()

        config = replace(
            AppConfig(),
            point_in_time=replace(AppConfig().point_in_time, enabled=True),
        )
        downloader = TusharePointInTimeDownloader(config, client=EmptyClient())
        result = downloader._fetch_fundamentals(
            "600000.SH",
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-06-30"),
            self.calendar,
        )
        self.assertTrue(result.empty)
        income_fields = str(calls["income"]["fields"])
        indicator_fields = str(calls["fina_indicator"]["fields"])
        self.assertIn("f_ann_date", income_fields)
        self.assertNotIn("f_ann_date", indicator_fields)
        self.assertEqual(calls["income"]["report_type"], "1")
        self.assertNotIn("report_type", calls["fina_indicator"])
        self.assertEqual(
            tuple(indicator_fields.split(","))[
                : len(FUNDAMENTAL_SOURCE_FIELDS["fina_indicator"])
            ],
            FUNDAMENTAL_SOURCE_FIELDS["fina_indicator"],
        )

    def test_downloader_seals_restartable_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_cache = root / "base"
            base_cache.mkdir()
            base = AppConfig()
            membership = pd.DataFrame({"symbol": ["600000.SH"]})
            membership_path = base_cache / "membership.csv.gz"
            membership.to_csv(
                membership_path, index=False, compression="gzip"
            )
            base_paths = [membership_path]
            for relative in [
                "benchmark.csv.gz",
                "regime.csv.gz",
                "corporate_actions.csv.gz",
                "securities.csv.gz",
                "industry_membership.csv.gz",
                "calendar.csv.gz",
                "bars/600000_SH.csv.gz",
            ]:
                path = base_cache / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"sealed test fixture")
                base_paths.append(path)
            files = build_file_inventory(base_cache, base_paths)
            (base_cache / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "provider": base.data.provider,
                        "universe_index": base.data.universe_index,
                        "regime_index": base.data.regime_index,
                        "benchmark_index": base.data.benchmark_index,
                        "industry_standard": base.data.industry_standard,
                        "industry_level": base.data.industry_level,
                        "requested_start": "2010-01-01",
                        "requested_end": "2025-12-31",
                        "files": files,
                        "data_fingerprint_sha256": inventory_sha256(files),
                    }
                ),
                encoding="utf-8",
            )
            config = replace(
                base,
                data=replace(base.data, cache_dir=str(base_cache)),
                point_in_time=replace(
                    base.point_in_time,
                    enabled=True,
                    cache_dir=str(root / "pit"),
                ),
                backtest=replace(
                    base.backtest,
                    start_date="2020-03-01",
                    end_date="2020-06-30",
                ),
            )
            downloader = TusharePointInTimeDownloader(config, client=object())
            market = SimpleNamespace(membership=membership)
            with (
                mock.patch(
                    "ashare_quant.pit_data.MarketDataBundle.from_cache",
                    return_value=market,
                ),
                mock.patch.object(
                    downloader, "_fetch_calendar", return_value=self.calendar
                ),
                mock.patch.object(
                    downloader,
                    "_fetch_fundamentals",
                    return_value=self.fundamentals,
                ),
                mock.patch.object(
                    downloader,
                    "_fetch_valuations",
                    return_value=self.valuations,
                ),
            ):
                real_replace = os.replace
                failed = False

                def fail_sealed_swap(
                    source: str | Path, destination: str | Path
                ) -> None:
                    nonlocal failed
                    if (
                        not failed
                        and Path(source).resolve() == (root / "pit.sealed").resolve()
                        and Path(destination).resolve() == (root / "pit").resolve()
                    ):
                        failed = True
                        raise OSError("simulated directory swap failure")
                    real_replace(source, destination)

                with mock.patch(
                    "ashare_quant.pit_data.os.replace",
                    side_effect=fail_sealed_swap,
                ):
                    with self.assertRaisesRegex(OSError, "simulated"):
                        downloader.download()
                self.assertTrue(
                    (root / "pit" / "download_state.json").is_file()
                )
                restored = downloader.download()
            self.assertTrue((root / "pit" / "manifest.json").is_file())
            self.assertFalse((root / "pit" / "download_state.json").exists())
            self.assertTrue(restored.manifest["verification"]["verified"])
            (base_cache / "benchmark.csv.gz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "完整性校验失败"):
                PointInTimeDataBundle.from_cache(
                    root / "pit",
                    expected_config=config,
                    base_manifest_path=base_cache / "manifest.json",
                )

    def test_cli_snapshot_writes_sha256_metadata_and_refuses_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "snapshot.csv"
            config = replace(
                AppConfig(),
                point_in_time=replace(
                    AppConfig().point_in_time,
                    enabled=True,
                    cache_dir=str(Path(directory) / "pit"),
                ),
            )
            bundle = replace(
                self.bundle,
                manifest={
                    "data_fingerprint_sha256": "pit-fingerprint",
                    "base_data_fingerprint_sha256": "base-fingerprint",
                },
            )
            args = cli_module._parser().parse_args(
                [
                    "pit-snapshot",
                    "--config",
                    "config.yaml",
                    "--date",
                    "2020-04-01",
                    "--output",
                    str(output),
                ]
            )
            with (
                mock.patch.object(cli_module, "_load_config", return_value=config),
                mock.patch.object(cli_module, "_pit_bundle", return_value=bundle),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(cli_module._dispatch(args), 0)
                with self.assertRaises(FileExistsError):
                    cli_module._dispatch(args)
            metadata = json.loads(
                output.with_name(output.name + ".manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["pit_data_fingerprint_sha256"], "pit-fingerprint")
            self.assertEqual(metadata["snapshot_rows"], 1)
            self.assertEqual(len(metadata["snapshot_sha256"]), 64)


class V2ConfigMigrationTests(unittest.TestCase):
    def test_package_and_lock_versions_match(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
        lock = tomllib.loads((ROOT / "uv.lock").read_text("utf-8"))
        locked = next(
            package
            for package in lock["package"]
            if package["name"] == "ashare-multifactor-backtest"
        )
        self.assertEqual(project["project"]["version"], __version__)
        self.assertEqual(locked["version"], __version__)

    def test_v1_yaml_is_migrated_with_pit_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.yaml"
            path.write_text("backtest:\n  end_date: 2025-12-31\n", encoding="utf-8")
            config = AppConfig.from_yaml(path)
        self.assertEqual(config.schema_version, CONFIG_SCHEMA_VERSION)
        self.assertFalse(config.point_in_time.enabled)

    def test_future_config_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.yaml"
            path.write_text("schema_version: 99\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不受支持"):
                AppConfig.from_yaml(path)

    def test_unknown_top_level_config_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "typo.yaml"
            path.write_text("stratgey: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "未知顶层字段"):
                AppConfig.from_yaml(path)

    def test_unknown_nested_config_key_is_reported_as_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "typo.yaml"
            path.write_text(
                "point_in_time:\n  fundamental_lag_days: 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "配置字段无效"):
                AppConfig.from_yaml(path)

    def test_point_in_time_cache_path_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resolved = AppConfig().resolve_paths(directory)
            self.assertTrue(Path(resolved.point_in_time.cache_dir).is_absolute())
            self.assertEqual(
                Path(resolved.point_in_time.cache_dir),
                Path(directory).resolve() / "data/pit_cache",
            )

    def test_point_in_time_rejects_ambiguous_scalar_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            enabled = Path(directory) / "enabled.yaml"
            enabled.write_text(
                "point_in_time:\n  enabled: 'false'\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "必须是布尔值"):
                AppConfig.from_yaml(enabled)
            lag = Path(directory) / "lag.yaml"
            lag.write_text(
                "point_in_time:\n  fundamental_lag_trading_days: 1.5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "整数参数类型错误"):
                AppConfig.from_yaml(lag)


if __name__ == "__main__":
    unittest.main()
