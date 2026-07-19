"""Phase A reporting: coverage, capacity, and cross-source validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ArchiveConfig
from .state import TaskStateDB, TaskStatus


def _load_bronze_rows(db: TaskStateDB) -> list[dict[str, Any]]:
    tasks = [t for t in db.list_tasks() if t.status == TaskStatus.SUCCESS and t.bronze_path]
    return [
        {
            "api_name": t.api_name,
            "dataset": t.dataset,
            "priority": t.priority,
            "row_count": t.row_count,
            "bronze_path": t.bronze_path,
            "raw_sha256": t.raw_sha256,
            "schema_fingerprint": t.schema_fingerprint,
        }
        for t in tasks
    ]


def write_coverage_report(
    config: ArchiveConfig,
    db: TaskStateDB,
    snapshot_id: str,
) -> Path:
    rows = _load_bronze_rows(db)
    by_api: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_api.setdefault(row["api_name"], {"tasks": 0, "rows": 0}).update(
            {
                "tasks": by_api[row["api_name"]]["tasks"] + 1,
                "rows": by_api[row["api_name"]]["rows"] + row["row_count"],
            }
        )

    status_counts = db.count_by_status()
    report = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "provider": config.provider.name,
        "by_api": by_api,
        "status_counts": status_counts,
        "sample_scope": {
            "trade_dates": 5,
            "symbols": 50,
            "financial_quarters": 4,
        },
    }
    path = config.reports_dir / "coverage_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Phase A 覆盖报告\n",
        f"- 快照: `{snapshot_id}`\n",
        f"- 来源: `{config.provider.name}`\n",
        "\n## 接口覆盖\n",
        "| 接口 | 成功任务数 | 总行数 |\n",
        "|---|---|---|\n",
    ]
    for api_name, info in sorted(by_api.items()):
        md.append(f"| {api_name} | {info['tasks']} | {info['rows']:,} |\n")
    md.append("\n## 状态统计\n")
    for status, count in sorted(status_counts.items()):
        md.append(f"- {status}: {count}\n")
    md_path = config.reports_dir / "coverage_report.md"
    md_path.write_text("".join(md), encoding="utf-8")
    return md_path


def write_capacity_estimate(
    config: ArchiveConfig,
    db: TaskStateDB,
    snapshot_id: str,
) -> Path:
    rows = _load_bronze_rows(db)
    total_rows = sum(r["row_count"] for r in rows)
    # Roughly assume 250 trading days/year and full-market symbol count.
    total_bronze_bytes = 0
    for r in rows:
        p = Path(r["bronze_path"])
        if p.exists():
            total_bronze_bytes += p.stat().st_size

    report = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "sample_rows": total_rows,
        "sample_bronze_bytes": total_bronze_bytes,
        "extrapolation_notes": [
            "P0 全市场日线按 250 交易日/年 × 约 5500 只估算",
            "财务表按 4 季度/年 × 约 5500 只估算",
            "实际容量必须以持续抽样为准",
        ],
    }
    path = config.reports_dir / "capacity_estimate.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Phase A 容量估算\n",
        f"- 样本行数: {total_rows:,}\n",
        f"- 样本 Bronze 字节: {total_bronze_bytes:,}\n",
        "\n## 外推说明\n",
    ]
    for note in report["extrapolation_notes"]:
        md.append(f"- {note}\n")
    md_path = config.reports_dir / "capacity_estimate.md"
    md_path.write_text("".join(md), encoding="utf-8")
    return md_path


def write_cross_source_validation(
    config: ArchiveConfig,
    db: TaskStateDB,
    snapshot_id: str,
) -> Path:
    """Placeholder for cross-source validation; Phase A only records intent."""
    rows = _load_bronze_rows(db)
    report = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "status": "pending_real_data",
        "message": (
            "Phase A 仅完成单来源样本；跨源核验需要第二来源或官方缓存，"
            "待真实数据下载后补充。"
        ),
        "sample_tasks": len(rows),
        "planned_symbols": 100,
        "planned_dates": 20,
    }
    path = config.reports_dir / "cross_source_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Phase A 跨来源抽样核验\n",
        f"- 状态: {report['status']}\n",
        f"- 说明: {report['message']}\n",
        f"- 已采样任务: {report['sample_tasks']}\n",
        f"- 计划抽样: {report['planned_symbols']} 只股票 × {report['planned_dates']} 个交易日\n",
    ]
    md_path = config.reports_dir / "cross_source_validation.md"
    md_path.write_text("".join(md), encoding="utf-8")
    return md_path
