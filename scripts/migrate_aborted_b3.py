#!/usr/bin/env python
"""受控状态迁移: 将 aborted B3 snapshot 的非终态任务标记为 aborted_prestart。

背景(2026-07-21 用户指令): snapshot p0_B3_financial_20260721_071141 在未经
确认的情况下被启动后中止(启动期 429 致死)。其全部任务与已落盘文件保留不删,
snapshot 永不续跑;后续 B3 使用新 snapshot(新主键口径,新 task_id)。

迁移规则:
  - 归属判定: 重算 task_id(provider, api, params, fields, snapshot_id) 精确匹配,
    不靠时间窗或路径猜测;
  - running / retryable_error -> aborted_prestart(非终态清理,绝不标 success);
  - success / confirmed_empty 等终态行不改状态,仅 metadata 标注
    snapshot_status=aborted_prestart(审计可追溯);
  - 全部改动写入审计 JSONL(data_lake/catalog/migrations/),逐行 before/after。

Usage:
    uv run --no-sync python scripts/migrate_aborted_b3.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ashare_quant.archive.config import ArchiveConfig  # noqa: E402
from ashare_quant.archive.pipeline import task_id  # noqa: E402
from ashare_quant.archive.state import TaskStateDB, TaskStatus  # noqa: E402

ABORTED_SNAPSHOT = "p0_B3_financial_20260721_071141"
MIGRATION_ID = "20260721_aborted_b3_prestart"
REASON = (
    "B3 未经确认被启动后于启动期中止(429);按用户指令保留全部数据,"
    "snapshot 标记 aborted_prestart,永不续跑,后续使用新 snapshot"
)
NON_TERMINAL = {TaskStatus.RUNNING, TaskStatus.RETRYABLE_ERROR}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.archive.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = ArchiveConfig.from_yaml(REPO_ROOT / args.config)
    db = TaskStateDB(config.catalog_dir / "archive.duckdb")

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    audit_dir = config.catalog_dir / "migrations"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"{MIGRATION_ID}.jsonl"

    matched, migrated, annotated = 0, 0, 0
    audit_records = []
    for task in db.list_tasks():
        # 精确归属: 用 aborted snapshot 重算 task_id。
        recomputed = task_id(
            config.provider.name, task.api_name, task.params, task.fields, ABORTED_SNAPSHOT
        )
        if recomputed != task.task_id:
            continue
        matched += 1
        before = {"status": task.status.value, "metadata": dict(task.metadata)}
        changed = False
        if task.status in NON_TERMINAL:
            task.status = TaskStatus.ABORTED_PRESTART
            task.metadata["aborted_from_status"] = before["status"]
            changed = True
            migrated += 1
        else:
            annotated += 1
        task.metadata["snapshot_status"] = "aborted_prestart"
        task.metadata["aborted_snapshot"] = ABORTED_SNAPSHOT
        task.metadata["migration_id"] = MIGRATION_ID
        task.metadata["migrated_at_utc"] = now
        if changed or True:
            audit_records.append(
                {
                    "migration_id": MIGRATION_ID,
                    "task_id": task.task_id,
                    "api_name": task.api_name,
                    "params": task.params,
                    "before": before,
                    "after": {"status": task.status.value, "metadata_added": [
                        "snapshot_status", "aborted_snapshot", "migration_id", "migrated_at_utc",
                    ] + (["aborted_from_status"] if changed else [])},
                    "reason": REASON,
                    "at_utc": now,
                    "dry_run": args.dry_run,
                }
            )
        if not args.dry_run:
            db.upsert(task)

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    if not args.dry_run:
        with audit_path.open("a", encoding="utf-8") as f:
            for rec in audit_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "migration_id": MIGRATION_ID,
        "mode": mode,
        "snapshot": ABORTED_SNAPSHOT,
        "matched_rows": matched,
        "status_migrated": migrated,
        "terminal_annotated": annotated,
        "audit_file": str(audit_path) if not args.dry_run else None,
        "at_utc": now,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
