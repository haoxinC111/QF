"""Phase-A sample download runner and report generation."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .config import ArchiveConfig
from .pipeline import ArchivePipeline, EndpointSpec, build_manifest, write_manifest
from .provider import ArchiveProvider
from .registry import EndpointInventory
from .reports import (
    write_capacity_estimate,
    write_coverage_report,
    write_cross_source_validation,
)
from .state import TaskStateDB

logger = logging.getLogger(__name__)


# Phase A sample scope.
SAMPLE_TRADE_DATES = ["20250106", "20250107", "20250108", "20250109", "20250110"]
SAMPLE_SYMBOLS = [
    "000001.SZ",
    "000002.SZ",
    "000063.SZ",
    "000333.SZ",
    "000538.SZ",
    "000568.SZ",
    "000651.SZ",
    "000725.SZ",
    "000768.SZ",
    "000858.SZ",
    "000895.SZ",
    "002001.SZ",
    "002007.SZ",
    "002024.SZ",
    "002027.SZ",
    "002142.SZ",
    "002230.SZ",
    "002236.SZ",
    "002304.SZ",
    "002352.SZ",
    "002415.SZ",
    "002460.SZ",
    "002475.SZ",
    "002594.SZ",
    "002714.SZ",
    "002812.SZ",
    "300003.SZ",
    "300014.SZ",
    "300015.SZ",
    "300033.SZ",
    "300059.SZ",
    "300122.SZ",
    "300124.SZ",
    "300142.SZ",
    "300274.SZ",
    "300408.SZ",
    "300413.SZ",
    "300433.SZ",
    "300498.SZ",
    "300750.SZ",
    "600000.SH",
    "600009.SH",
    "600016.SH",
    "600028.SH",
    "600030.SH",
    "600031.SH",
    "600036.SH",
    "600048.SH",
    "600050.SH",
    "600104.SH",
]
SAMPLE_FINANCIAL_QUARTERS = ["20240331", "20240630", "20240930", "20241231"]


def _expand_sample_specs(
    inventory: EndpointInventory,
) -> list[EndpointSpec]:
    """Build concrete Phase-A tasks from inventory templates."""
    specs: list[EndpointSpec] = []
    for ep in inventory.list_by_priority(priorities=["P0", "P1"]):
        if not ep.enabled:
            continue
        base = ep.to_spec()
        if ep.api_name in ("trade_cal",):
            specs.append(base)
        elif ep.api_name == "stock_basic":
            for status in ("L", "D", "P"):
                params = dict(base.params_template)
                params["list_status"] = status
                specs.append(base.__class__(**{**base.__dict__, "params_template": params}))
        elif ep.primary_split == "trade_date":
            for trade_date in SAMPLE_TRADE_DATES:
                params = dict(base.params_template)
                params["trade_date"] = trade_date
                specs.append(base.__class__(**{**base.__dict__, "params_template": params}))
        elif ep.primary_split == "end_date":
            for end_date in SAMPLE_FINANCIAL_QUARTERS:
                params = dict(base.params_template)
                params["end_date"] = end_date
                # For financial endpoints, also restrict to sample symbols.
                if ep.fallback_split == "ts_code":
                    params["ts_code"] = ",".join(SAMPLE_SYMBOLS[:10])
                specs.append(base.__class__(**{**base.__dict__, "params_template": params}))
        elif ep.primary_split == "period":
            for quarter in SAMPLE_FINANCIAL_QUARTERS:
                params = dict(base.params_template)
                params["period"] = quarter
                specs.append(base.__class__(**{**base.__dict__, "params_template": params}))
        elif ep.api_name == "index_weight":
            params = dict(base.params_template)
            params["index_code"] = "399300.SZ"
            params["trade_date"] = SAMPLE_TRADE_DATES[-1]
            specs.append(base.__class__(**{**base.__dict__, "params_template": params}))
        else:
            specs.append(base)
    return specs


def run_phase_a_sample(
    config: ArchiveConfig,
    provider: ArchiveProvider,
    inventory: EndpointInventory,
    db: TaskStateDB,
    *,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Run the Phase A sample download and generate all required reports."""
    config.ensure_dirs()
    snapshot_id = f"phase_a_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    pipeline = ArchivePipeline(config, provider, db, snapshot_id=snapshot_id)

    specs = _expand_sample_specs(inventory)
    logger.info("Phase A 准备 %d 个任务", len(specs))

    started = time.perf_counter()
    result = pipeline.run_tasks(specs, skip_existing=True)
    elapsed = time.perf_counter() - started

    manifest = build_manifest(config, db, snapshot_id, git_commit=git_commit)
    manifest_path = write_manifest(config, manifest)

    # Reports.
    coverage_path = write_coverage_report(config, db, snapshot_id)
    capacity_path = write_capacity_estimate(config, db, snapshot_id)
    cross_path = write_cross_source_validation(config, db, snapshot_id)

    summary = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "tasks_completed": result.tasks_completed,
        "tasks_failed": result.tasks_failed,
        "tasks_empty": result.tasks_empty,
        "rows_total": result.rows_total,
        "elapsed_seconds": round(elapsed, 2),
        "errors": result.errors,
        "manifest_path": str(manifest_path),
        "coverage_report": str(coverage_path),
        "capacity_estimate": str(capacity_path),
        "cross_source_validation": str(cross_path),
    }
    summary_path = config.reports_dir / "phase_a_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Phase A 完成: %s", summary_path)
    return summary
