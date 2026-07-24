#!/usr/bin/env python
"""受控状态迁移: 768 个 context 漂移孤儿任务 -> orphaned_context_drift。

背景(2026-07-22 用户指令): 断点续跑 B3 时 build_context 按新交易日
(20260720 -> 20260721)重建,params 漂移导致 task_id 全部重算,产生了
768 个不属于原 manifest 的 balancesheet_vip 任务(767 success + 1 running)。
按指令: 数据库记录与 Raw/Bronze 文件保留不删、不覆盖;统一标记
ORPHANED_CONTEXT_DRIFT(不用 aborted_prestart),research_eligible=false,
记录原状态、原/新 context SHA、原因与迁移时间。它们不在原 manifest 的
task_id 集合内,天然从 manifest/decision/fixtures/研究选择器中排除。

归属判定(三重精确匹配,不靠时间窗猜测):
  1. 重算 task_id(provider, api, params, fields, snapshot) == 行 task_id;
  2. task_id 不在原 manifest 的 46,960 个 task_id 内;
  3. 计数与用户确认的 768 一致,不符即 fail-closed。

Usage:
    uv run --no-sync python scripts/migrate_orphaned_context_drift.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ashare_quant.archive.config import ArchiveConfig  # noqa: E402
from ashare_quant.archive.pipeline import task_id  # noqa: E402
from ashare_quant.archive.state import TaskStateDB, TaskStatus  # noqa: E402

SNAPSHOT = "p0_B3_financial_20260721_092031"
BATCH = "B3_financial"
MIGRATION_ID = "20260722_orphaned_context_drift"
REASON = (
    "断点续跑时 build_context 按新交易日重建(20260720->20260721),params 漂移、"
    "task_id 重算,任务不属于原 manifest;按用户指令保留记录与文件,"
    "标记 orphaned_context_drift 且 research_eligible=false"
)
ORIGINAL_CONTEXT_SHA256 = "93deb5837aa01a772ebf5cf55fd373ee88cd907eb981fc1826a2cf22ac7d291b"
EXPECTED_COUNT = 768


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.archive.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = ArchiveConfig.from_yaml(REPO_ROOT / args.config)
    db = TaskStateDB(config.catalog_dir / "archive.duckdb")
    reports_dir = REPO_ROOT / "data_lake" / "reports" / "batches" / BATCH

    # 原 manifest(物化于漂移前,mtime 2026-07-22 01:30)与其 SHA。
    manifest_path = reports_dir / "batch_manifest.jsonl"
    manifest_sha_before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    assert manifest["snapshot_id"] == SNAPSHOT, "manifest snapshot 不符"
    manifest_ids = {t["task_id"] for t in manifest["tasks"]}
    assert len(manifest_ids) == 46960, f"原 manifest 任务数异常: {len(manifest_ids)}"

    # 漂移 context 的 SHA(2026-07-22 02:06 由误启动的续跑写入)。
    drifted_sha = ""
    for p in sorted(reports_dir.glob("context_*.json")):
        sha = json.loads(p.read_text(encoding="utf-8")).get("context_sha256", "")
        if sha and sha != ORIGINAL_CONTEXT_SHA256:
            drifted_sha = sha
    assert drifted_sha, "未找到漂移 context 文件"

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    audit_dir = config.catalog_dir / "migrations"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"{MIGRATION_ID}.jsonl"

    orphans = []
    for task in db.list_tasks():
        recomputed = task_id(config.provider.name, task.api_name, task.params, task.fields, SNAPSHOT)
        if recomputed != task.task_id:
            continue  # 不属于本 snapshot
        if task.task_id in manifest_ids:
            continue  # 属于原 manifest 的合法任务
        orphans.append(task)

    if len(orphans) != EXPECTED_COUNT:
        print(json.dumps({
            "error": "孤儿计数与用户确认值不符, fail-closed",
            "expected": EXPECTED_COUNT, "actual": len(orphans),
            "by_api": {a: sum(1 for t in orphans if t.api_name == a) for a in {t.api_name for t in orphans}},
        }, ensure_ascii=False, indent=2))
        return 1

    audit_records = []
    for task in orphans:
        before = {"status": task.status.value, "row_count": task.row_count,
                  "raw_sha256": task.raw_sha256}
        task.metadata["orphaned_from_status"] = before["status"]
        task.metadata["original_context_sha256"] = ORIGINAL_CONTEXT_SHA256
        task.metadata["drifted_context_sha256"] = drifted_sha
        task.metadata["research_eligible"] = False
        task.metadata["migration_id"] = MIGRATION_ID
        task.metadata["migrated_at_utc"] = now
        task.status = TaskStatus.ORPHANED_CONTEXT_DRIFT
        audit_records.append({
            "migration_id": MIGRATION_ID,
            "task_id": task.task_id,
            "api_name": task.api_name,
            "params": task.params,
            "before": before,
            "after": {"status": task.status.value, "research_eligible": False},
            "original_context_sha256": ORIGINAL_CONTEXT_SHA256,
            "drifted_context_sha256": drifted_sha,
            "reason": REASON,
            "at_utc": now,
            "dry_run": args.dry_run,
        })
        if not args.dry_run:
            db.upsert(task)

    manifest_sha_after = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if not args.dry_run:
        with audit_path.open("a", encoding="utf-8") as f:
            for rec in audit_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "migration_id": MIGRATION_ID,
        "mode": "DRY-RUN" if args.dry_run else "APPLY",
        "orphans_migrated": len(orphans),
        "by_status_before": {s: sum(1 for r in audit_records if r["before"]["status"] == s)
                             for s in {r["before"]["status"] for r in audit_records}},
        "original_context_sha256": ORIGINAL_CONTEXT_SHA256[:16],
        "drifted_context_sha256": drifted_sha[:16],
        "manifest_sha256_before": manifest_sha_before,
        "manifest_sha256_after": manifest_sha_after,
        "manifest_untouched": manifest_sha_before == manifest_sha_after,
        "audit_file": str(audit_path) if not args.dry_run else None,
        "at_utc": now,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
