from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ashare_quant.archive.batch import batch_decision_gates
from ashare_quant.config import AppConfig
from ashare_quant.pit_acceptance import (
    ACCEPTANCE_REPORT_FILENAME,
    require_pit_acceptance,
    run_pit_acceptance,
    verify_pit_acceptance_receipt,
)
from ashare_quant.provenance import (
    ARTIFACT_MANIFEST_FILENAME,
    verify_artifact_manifest,
)
from test_pit_lake import (
    _config,
    _make_archive,
    _write_base_cache,
    _write_batch_reports,
)


class PointInTimeAcceptanceTests(unittest.TestCase):
    def _strict_setup(
        self, root: Path, *, maximum_valuation_age_days: int | None = None
    ) -> tuple[object, AppConfig]:
        archive = _make_archive(root / "lake", strict=True)
        _write_batch_reports(archive, archive.root / "reports")
        base_cache = _write_base_cache(root / "base")
        config = replace(
            _config(root / "pit"),
            data=replace(AppConfig().data, cache_dir=str(base_cache)),
        )
        if maximum_valuation_age_days is not None:
            config = replace(
                config,
                point_in_time=replace(
                    config.point_in_time,
                    maximum_valuation_age_days=maximum_valuation_age_days,
                ),
            )
        return archive, config

    def test_fixture_acceptance_is_sealed_but_engineering_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _make_archive(root / "lake")
            config = _config(root / "pit")
            output = root / "acceptance"
            report = run_pit_acceptance(
                config,
                archive.root,
                catalog_path=archive.catalog,
                schema_registry=archive.schemas,
                output_dir=output,
                fixture_mode=True,
                bucket_count=2,
            )
            self.assertEqual(report["decision"], "engineering_only")
            self.assertFalse(report["research_eligible"])
            self.assertTrue(report["temporal_audit"]["passed"])
            verification = verify_artifact_manifest(
                output / ARTIFACT_MANIFEST_FILENAME, strict=True
            )
            self.assertEqual(verification["file_count"], 3)
            with self.assertRaisesRegex(ValueError, "决策无效"):
                verify_pit_acceptance_receipt(
                    output / ACCEPTANCE_REPORT_FILENAME,
                    json.loads(
                        (root / "pit" / "manifest.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                )

    def test_strict_acceptance_pass_receipt_is_bound_to_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, config = self._strict_setup(root)
            output = root / "acceptance"
            report = run_pit_acceptance(
                config,
                archive.root,
                catalog_path=archive.catalog,
                schema_registry=archive.schemas,
                reports_root=archive.root / "reports",
                output_dir=output,
                bucket_count=2,
            )
            manifest = json.loads(
                (root / "pit" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["decision"], "pass")
            self.assertTrue(manifest["archive_bridge"]["acceptance_required"])
            with self.assertRaisesRegex(ValueError, "要求 Alpha5"):
                require_pit_acceptance(manifest, None)
            legacy = dict(manifest)
            legacy_bridge = dict(manifest["archive_bridge"])
            legacy_bridge["schema_version"] = 1
            legacy_bridge.pop("acceptance_required")
            legacy["archive_bridge"] = legacy_bridge
            with self.assertRaisesRegex(ValueError, "要求 Alpha5"):
                require_pit_acceptance(legacy, None)
            require_pit_acceptance(
                manifest, output / ACCEPTANCE_REPORT_FILENAME
            )
            downgraded = dict(manifest)
            downgraded["archive_bridge"] = {
                **manifest["archive_bridge"],
                "acceptance_required": False,
            }
            with self.assertRaisesRegex(ValueError, "门禁标志"):
                require_pit_acceptance(
                    downgraded, output / ACCEPTANCE_REPORT_FILENAME
                )
            verified = verify_pit_acceptance_receipt(
                output / ACCEPTANCE_REPORT_FILENAME, manifest
            )
            self.assertEqual(
                verified["pit_identity"]["data_fingerprint_sha256"],
                manifest["data_fingerprint_sha256"],
            )

    def test_missing_batch_evidence_writes_blocked_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, config = self._strict_setup(root)
            (
                archive.root
                / "reports/batches/B3_financial/batch_decision.json"
            ).unlink()
            output = root / "blocked"
            report = run_pit_acceptance(
                config,
                archive.root,
                catalog_path=archive.catalog,
                schema_registry=archive.schemas,
                reports_root=archive.root / "reports",
                output_dir=output,
                bucket_count=2,
            )
            self.assertEqual(report["decision"], "blocked")
            self.assertEqual(report["stages"][0]["status"], "fail")
            self.assertEqual(report["stages"][1]["status"], "skipped")
            verify_artifact_manifest(
                output / ARTIFACT_MANIFEST_FILENAME, strict=True
            )

    def test_source_mutation_blocks_reused_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, config = self._strict_setup(root)
            first = root / "first"
            run_pit_acceptance(
                config,
                archive.root,
                catalog_path=archive.catalog,
                schema_registry=archive.schemas,
                reports_root=archive.root / "reports",
                output_dir=first,
                bucket_count=2,
            )
            source = next(
                (archive.root / "bronze").rglob("daily_basic/*.parquet")
            )
            source.write_bytes(source.read_bytes() + b"tamper")
            second = root / "second"
            report = run_pit_acceptance(
                config,
                archive.root,
                catalog_path=archive.catalog,
                schema_registry=archive.schemas,
                reports_root=archive.root / "reports",
                output_dir=second,
                bucket_count=2,
            )
            self.assertEqual(report["decision"], "blocked")
            self.assertEqual(report["stages"][0]["details"]["action"], "reused")
            self.assertEqual(report["stages"][1]["status"], "fail")
            self.assertIn(
                "源 Bronze 已变化",
                report["stages"][1]["details"]["message"],
            )

    def test_time_sliced_coverage_blocks_stale_valuation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, config = self._strict_setup(
                root, maximum_valuation_age_days=0
            )
            report = run_pit_acceptance(
                config,
                archive.root,
                catalog_path=archive.catalog,
                schema_registry=archive.schemas,
                reports_root=archive.root / "reports",
                output_dir=root / "acceptance",
                bucket_count=2,
            )
            self.assertEqual(report["decision"], "blocked")
            self.assertEqual(report["stages"][2]["status"], "fail")
            self.assertEqual(
                report["temporal_audit"]["dates_below_threshold"],
                ["2020-04-03"],
            )

    def test_receipt_tamper_and_wrong_cache_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, config = self._strict_setup(root)
            output = root / "acceptance"
            run_pit_acceptance(
                config,
                archive.root,
                catalog_path=archive.catalog,
                schema_registry=archive.schemas,
                reports_root=archive.root / "reports",
                output_dir=output,
                bucket_count=2,
            )
            manifest = json.loads(
                (root / "pit" / "manifest.json").read_text(encoding="utf-8")
            )
            wrong = dict(manifest)
            wrong["data_fingerprint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "指纹不一致"):
                verify_pit_acceptance_receipt(
                    output / ACCEPTANCE_REPORT_FILENAME, wrong
                )
            path = output / ACCEPTANCE_REPORT_FILENAME
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["conclusion"] = "tampered"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "完整性校验失败"):
                verify_pit_acceptance_receipt(path, manifest)

    def test_nonempty_acceptance_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _make_archive(root / "lake")
            output = root / "acceptance"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "必须不存在或为空"):
                run_pit_acceptance(
                    _config(root / "pit"),
                    archive.root,
                    catalog_path=archive.catalog,
                    schema_registry=archive.schemas,
                    output_dir=output,
                    fixture_mode=True,
                )
            self.assertEqual(
                (output / "existing.txt").read_text(encoding="utf-8"), "keep"
            )

    def test_quarantined_task_can_never_pass_batch_decision(self) -> None:
        gates = batch_decision_gates(
            {"success": 999, "confirmed_empty": 0, "quarantined": 1}, 1000
        )
        self.assertTrue(gates["all_tasks_terminal"])
        self.assertTrue(gates["success_rate_ge_99.5%"])
        self.assertFalse(gates["no_quarantined"])
        self.assertFalse(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
