from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare_quant.config import AppConfig
from ashare_quant.pit_data import (
    PointInTimeDataBundle,
    require_pit_research_eligible,
)
from ashare_quant.pit_lake import (
    build_pit_cache_from_archive,
    remap_archive_path,
    verify_archive_pit_cache,
)
from ashare_quant.provenance import build_file_inventory, inventory_sha256


class ArchiveFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.catalog = root / "catalog" / "archive.duckdb"
        self.schemas = root / "catalog" / "schema_registry"
        self.catalog.parent.mkdir(parents=True)
        self.rows: list[tuple[object, ...]] = []

    def add(
        self,
        api_name: str,
        frame: pd.DataFrame,
        params: dict[str, str],
        snapshot_id: str = "fixture",
    ) -> None:
        fingerprint = (api_name.encode("utf-8").hex() + "0" * 64)[:64]
        partition = "_".join(f"{key}={value}" for key, value in params.items())
        relative = (
            Path("bronze")
            / "test_provider"
            / api_name
            / f"{api_name}_{partition}_{snapshot_id}.parquet"
        )
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        schema_dir = self.schemas / api_name
        schema_dir.mkdir(parents=True, exist_ok=True)
        (schema_dir / f"{fingerprint}.json").write_text(
            json.dumps(
                {
                    "endpoint": api_name,
                    "fingerprint": fingerprint,
                    "columns": list(frame.columns),
                }
            ),
            encoding="utf-8",
        )
        task_id = f"{len(self.rows) + 1:064x}"
        remote = Path("/another/machine/project/data_lake") / relative
        self.rows.append(
            (
                task_id,
                api_name,
                json.dumps(params),
                "success",
                len(frame),
                str(remote),
                fingerprint,
                "a" * 64,
            )
        )

    def seal(self) -> None:
        with sqlite3.connect(self.catalog) as connection:
            connection.execute(
                """
                CREATE TABLE archive_tasks (
                    task_id TEXT PRIMARY KEY,
                    api_name TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    bronze_path TEXT NOT NULL,
                    schema_fingerprint TEXT NOT NULL,
                    raw_sha256 TEXT NOT NULL
                )
                """
            )
            connection.executemany(
                "INSERT INTO archive_tasks VALUES (?,?,?,?,?,?,?,?)",
                self.rows,
            )


def _make_archive(root: Path, *, strict: bool = False) -> ArchiveFixture:
    archive = ArchiveFixture(root)
    b0 = "p0_B0_reference_test" if strict else "fixture"
    b1 = "p0_B1_market_test" if strict else "fixture"
    b3 = "p0_B3_financial_test" if strict else "fixture"
    calendar = pd.DataFrame(
        {
            "exchange": "SSE",
            "cal_date": pd.bdate_range("2019-04-01", "2020-05-01").strftime(
                "%Y%m%d"
            ),
            "is_open": 1,
            "pretrade_date": "",
        }
    )
    archive.add("trade_cal", calendar, {"exchange": "SSE"}, b0)
    archive.add(
        "daily_basic",
        pd.DataFrame(
            [
                {
                    "ts_code": "600000.SH",
                    "trade_date": "20200401",
                    "turnover_rate": 1.2,
                    "pe_ttm": 8.0,
                    "pb": 0.9,
                    "ps_ttm": 1.1,
                    "dv_ttm": 2.0,
                    "total_mv": 1000.0,
                    "circ_mv": 800.0,
                }
            ]
        ),
        {"trade_date": "20200401"},
        b1,
    )
    common = {
        "ts_code": "600000.SH",
        "ann_date": "20200331",
        "f_ann_date": "20200331",
        "end_date": "20191231",
        "report_type": "1",
        "comp_type": "1",
        "update_flag": "0",
    }
    archive.add(
        "income_vip" if strict else "income",
        pd.DataFrame([{**common, "revenue": 100.0, "n_income_attr_p": 10.0}]),
        {"end_date": "20191231"},
        b3,
    )
    archive.add(
        "balancesheet_vip" if strict else "balancesheet",
        pd.DataFrame([{**common, "total_assets": 500.0, "total_liab": 200.0}]),
        {"end_date": "20191231"},
        b3,
    )
    archive.add(
        "cashflow_vip" if strict else "cashflow",
        pd.DataFrame([{**common, "n_cashflow_act": 15.0}]),
        {"end_date": "20191231"},
        b3,
    )
    archive.add(
        "fina_indicator_vip" if strict else "fina_indicator",
        pd.DataFrame(
            [
                {
                    "ts_code": "600000.SH",
                    "ann_date": "20200331",
                    "end_date": "20191231",
                    "update_flag": "0",
                    "roe": 12.0,
                }
            ]
        ),
        {"end_date": "20191231"},
        b3,
    )
    archive.seal()
    return archive


def _config(cache: Path) -> AppConfig:
    base = AppConfig()
    return replace(
        base,
        backtest=replace(
            base.backtest,
            start_date="2020-04-01",
            end_date="2020-04-03",
        ),
        point_in_time=replace(
            base.point_in_time,
            enabled=True,
            cache_dir=str(cache),
            history_years=1,
        ),
    )


def _write_base_cache(root: Path) -> Path:
    root.mkdir(parents=True)
    membership = root / "membership.csv.gz"
    pd.DataFrame(
        {"date": ["2020-04-01"], "symbol": ["600000.SH"]}
    ).to_csv(
        membership, index=False, compression="gzip"
    )
    paths = [membership]
    for relative in [
        "benchmark.csv.gz",
        "regime.csv.gz",
        "corporate_actions.csv.gz",
        "securities.csv.gz",
        "industry_membership.csv.gz",
        "calendar.csv.gz",
        "bars/600000_SH.csv.gz",
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sealed")
        paths.append(path)
    files = build_file_inventory(root, paths)
    base = AppConfig()
    (root / "manifest.json").write_text(
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
    return root


def _write_batch_reports(archive: ArchiveFixture, reports: Path) -> None:
    batches = {
        "B0_reference": ("p0_B0_reference_test", {"trade_cal"}),
        "B1_market": ("p0_B1_market_test", {"daily_basic"}),
        "B3_financial": (
            "p0_B3_financial_test",
            {
                "income_vip",
                "balancesheet_vip",
                "cashflow_vip",
                "fina_indicator_vip",
            },
        ),
    }
    for batch, (snapshot, apis) in batches.items():
        root = reports / "batches" / batch
        root.mkdir(parents=True)
        rows = [row for row in archive.rows if row[1] in apis]
        tasks = [
            {
                "task_id": row[0],
                "api_name": row[1],
                "status": row[3],
                "row_count": row[4],
                "bronze_path": row[5],
                "schema_fingerprint": row[6],
                "raw_sha256": row[7],
            }
            for row in rows
        ]
        manifest = {
            "snapshot_id": snapshot,
            "total_tasks": len(tasks),
            "by_status": {"success": len(tasks)},
            "tasks": tasks,
        }
        (root / "batch_manifest.jsonl").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )
        (root / "batch_decision.json").write_text(
            json.dumps(
                {
                    "batch": batch,
                    "snapshot_id": snapshot,
                    "decision": "pass",
                    "gates": {"all_tasks_terminal": True, "no_retryable_left": True},
                }
            ),
            encoding="utf-8",
        )
        checksum_lines = []
        for row in rows:
            path, relative = remap_archive_path(str(row[5]), archive.root)
            checksum_lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n"
            )
        (root / "checksums.sha256").write_text(
            "".join(checksum_lines), encoding="utf-8"
        )


class PointInTimeLakeTests(unittest.TestCase):
    def test_remaps_foreign_data_lake_path_and_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, relative = remap_archive_path(
                "/foreign/project/data_lake/bronze/provider/file.parquet", root
            )
            self.assertEqual(
                path, (root / "bronze/provider/file.parquet").resolve()
            )
            self.assertEqual(relative, "bronze/provider/file.parquet")
            with self.assertRaisesRegex(ValueError, "不安全"):
                remap_archive_path("data_lake/bronze/../secret", root)

    def test_fixture_build_round_trip_is_explicitly_research_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _make_archive(root / "lake")
            output = root / "pit"
            config = _config(output)
            manifest = build_pit_cache_from_archive(
                config,
                archive.root,
                catalog_path=archive.catalog,
                schema_registry=archive.schemas,
                output_dir=output,
                fixture_mode=True,
                bucket_count=2,
            )
            self.assertFalse(manifest["research_eligible"])
            self.assertEqual(manifest["data_quality"]["expected_symbols"], 1)
            bundle = PointInTimeDataBundle.from_cache(
                output, expected_config=config
            )
            self.assertTrue(bundle.fundamental_snapshot("2020-03-31").empty)
            snapshot = bundle.fundamental_snapshot("2020-04-01")
            self.assertEqual(float(snapshot.iloc[0]["revenue"]), 100.0)
            with self.assertRaisesRegex(ValueError, "不能生成 Alpha"):
                require_pit_research_eligible(bundle.manifest)

    def test_same_sources_produce_deterministic_data_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _make_archive(root / "lake")
            fingerprints = []
            for name in ("first", "second"):
                output = root / name
                config = _config(output)
                manifest = build_pit_cache_from_archive(
                    config,
                    archive.root,
                    catalog_path=archive.catalog,
                    schema_registry=archive.schemas,
                    output_dir=output,
                    fixture_mode=True,
                    bucket_count=2,
                )
                fingerprints.append(manifest["data_fingerprint_sha256"])
            self.assertEqual(fingerprints[0], fingerprints[1])

    def test_source_replay_detects_bronze_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _make_archive(root / "lake")
            output = root / "pit"
            config = _config(output)
            build_pit_cache_from_archive(
                config,
                archive.root,
                catalog_path=archive.catalog,
                schema_registry=archive.schemas,
                output_dir=output,
                fixture_mode=True,
            )
            source = next((archive.root / "bronze").rglob("daily_basic/*.parquet"))
            source.write_bytes(source.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "源 Bronze 已变化"):
                verify_archive_pit_cache(
                    config,
                    output,
                    fixture_mode=True,
                    archive_root=archive.root,
                    catalog_path=archive.catalog,
                    schema_registry=archive.schemas,
                )

    def test_registered_schema_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _make_archive(root / "lake")
            schema = next((archive.schemas / "daily_basic").glob("*.json"))
            payload = json.loads(schema.read_text(encoding="utf-8"))
            payload["columns"] = list(reversed(payload["columns"]))
            schema.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "注册 Schema"):
                build_pit_cache_from_archive(
                    _config(root / "pit"),
                    archive.root,
                    catalog_path=archive.catalog,
                    schema_registry=archive.schemas,
                    output_dir=root / "pit",
                    fixture_mode=True,
                )

    def test_catalog_row_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _make_archive(root / "lake")
            with sqlite3.connect(archive.catalog) as connection:
                connection.execute(
                    "UPDATE archive_tasks SET row_count=row_count+1 "
                    "WHERE api_name='daily_basic'"
                )
            with self.assertRaisesRegex(ValueError, "行数"):
                build_pit_cache_from_archive(
                    _config(root / "pit"),
                    archive.root,
                    catalog_path=archive.catalog,
                    schema_registry=archive.schemas,
                    output_dir=root / "pit",
                    fixture_mode=True,
                )

    def test_strict_mode_requires_batch_evidence_before_reading_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _make_archive(root / "lake")
            with self.assertRaisesRegex(FileNotFoundError, "批次证据"):
                build_pit_cache_from_archive(
                    _config(root / "pit"),
                    archive.root,
                    catalog_path=archive.catalog,
                    schema_registry=archive.schemas,
                    reports_root=root / "missing-reports",
                    output_dir=root / "pit",
                    fixture_mode=False,
                )

    def test_strict_build_binds_batches_base_identity_and_research_eligibility(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _make_archive(root / "lake", strict=True)
            reports = archive.root / "reports"
            _write_batch_reports(archive, reports)
            base_cache = _write_base_cache(root / "base")
            output = root / "pit"
            config = replace(
                _config(output),
                data=replace(AppConfig().data, cache_dir=str(base_cache)),
            )
            manifest = build_pit_cache_from_archive(
                config,
                archive.root,
                catalog_path=archive.catalog,
                schema_registry=archive.schemas,
                reports_root=reports,
                output_dir=output,
                fixture_mode=False,
                bucket_count=2,
            )
            self.assertTrue(manifest["research_eligible"])
            self.assertEqual(
                set(manifest["archive_bridge"]["batch_snapshots"]),
                {"B0_reference", "B1_market", "B3_financial"},
            )
            require_pit_research_eligible(manifest)
            verification = verify_archive_pit_cache(
                config,
                output,
                archive_root=archive.root,
                catalog_path=archive.catalog,
                schema_registry=archive.schemas,
                reports_root=reports,
            )
            self.assertEqual(verification["source_files_verified"], 6)


if __name__ == "__main__":
    unittest.main()
