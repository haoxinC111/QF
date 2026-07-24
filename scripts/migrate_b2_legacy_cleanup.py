#!/usr/bin/env python
"""B2 历史遗留清理(2026-07-23 用户图片方案,方案 1 审计迁移)。

1. 189 个 index_weight 僵尸 running(2026-07-19 撞名事件遗留,旧宇宙
   end_date=20260716 的 2026 年段): 逐任务验证存在同 index_code、
   区间完整覆盖 [20260101-20260716] 的新一代终态后继任务(success/bisected),
   标记 SUPERSEDED_LEGACY_COLLISION + research_eligible=false,
   原状态与后继映射写入迁移 JSONL;任何一个找不到覆盖后继即整体中止不写。
2. 1 个 index_basic suspect_truncated(无参全量单拉恰 8000 行截断):
   标记 SUPERSEDED_TRUNCATED_CAP,绑定 17 个已验证拆分后继任务
   (16 success 合计 11,798 行 + 1 OTHERS empty)。
3. 1 个 index_daily 僵尸 running(003198.CJ@2025)不在本脚本处理,
   由 run_b2_legacy_replay.py 与 23 个 retryable 一起精确重放。

Usage:
    uv run --no-sync python scripts/migrate_b2_legacy_cleanup.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ashare_quant.archive.config import ArchiveConfig  # noqa: E402
from ashare_quant.archive.state import TaskStateDB, TaskStatus  # noqa: E402

MIGRATION_ID = "b2_legacy_cleanup_20260723"
OUT = REPO_ROOT / "data_lake" / "catalog" / "migrations" / "20260723_b2_legacy_cleanup.jsonl"

TERMINAL_CARRIERS = (TaskStatus.SUCCESS, TaskStatus.BISECTED)


def main() -> int:
    config = ArchiveConfig.from_yaml(REPO_ROOT / "config.archive.yaml")
    db = TaskStateDB(config.catalog_dir / "archive.duckdb")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tasks = db.list_tasks()

    zombies = [t for t in tasks
               if t.status == TaskStatus.RUNNING and t.api_name == "index_weight"]
    assert len(zombies) == 189, f"僵尸数量预期 189,实际 {len(zombies)}"

    # --- 预检: 每个僵尸必须有覆盖后继,全部满足才落库(fail-closed) ---
    plan = []
    for z in zombies:
        code = z.params["index_code"]
        z_start, z_end = z.params["start_date"], z.params["end_date"]
        covering = [
            t for t in tasks
            if t.api_name == "index_weight"
            and t.status in TERMINAL_CARRIERS
            and t.params.get("index_code") == code
            and t.params.get("start_date", "99999999") <= z_start
            and t.params.get("end_date", "00000000") >= z_end
        ]
        if not covering:
            print(f"FAIL-CLOSED: {code} [{z_start}-{z_end}] 无覆盖后继,整体中止")
            return 1
        # 覆盖后继自身若是 bisected,数据由其子任务承载;记录直接后继
        plan.append((z, covering))

    # --- index_basic suspect: 绑定 17 段后继 ---
    suspect = [t for t in tasks
               if t.status == TaskStatus.SUSPECT_TRUNCATED and t.api_name == "index_basic"]
    assert len(suspect) == 1, f"index_basic suspect 预期 1,实际 {len(suspect)}"
    segments = [t for t in tasks
                if t.api_name == "index_basic"
                and t.status in (TaskStatus.SUCCESS, TaskStatus.CONFIRMED_EMPTY)
                and t.task_id != suspect[0].task_id]
    seg_rows = sum(t.row_count for t in segments if t.status == TaskStatus.SUCCESS)
    assert len(segments) == 17 and seg_rows == 11798, \
        f"index_basic 后继段校验失败: {len(segments)} 段 {seg_rows} 行"

    # --- 落库 + 审计 ---
    records = []
    for z, covering in plan:
        before = {"status": z.status.value, "row_count": z.row_count}
        z.status = TaskStatus.SUPERSEDED_LEGACY_COLLISION
        z.metadata["supersede_reason"] = (
            "2026-07-19 撞名事件僵尸 running: 旧宇宙 end_date=20260716 的 2026 年段,"
            "universe 重建后未重跑;区间已被新一代同名指数任务完全覆盖")
        z.metadata["superseded_by_task_ids"] = sorted(t.task_id for t in covering)
        z.metadata["research_eligible"] = False
        z.metadata["migration_id"] = MIGRATION_ID
        z.metadata["migrated_at_utc"] = now
        db.upsert(z)
        records.append({"task_id": z.task_id, "api_name": z.api_name, "params": z.params,
                        "before": before, "after": {"status": z.status.value},
                        "superseded_by": sorted(t.task_id for t in covering), "at_utc": now})

    s = suspect[0]
    before = {"status": s.status.value, "row_count": s.row_count}
    s.status = TaskStatus.SUPERSEDED_TRUNCATED_CAP
    s.metadata["supersede_reason"] = (
        "无参全量单拉恰 8000 行静默截断(2026-07-17 人工确认,分市场求和>=10740);"
        "数据由 market+CSI category 17 段拆分任务承载")
    s.metadata["superseded_by_task_ids"] = sorted(t.task_id for t in segments)
    s.metadata["research_eligible"] = False
    s.metadata["migration_id"] = MIGRATION_ID
    s.metadata["migrated_at_utc"] = now
    db.upsert(s)
    records.append({"task_id": s.task_id, "api_name": s.api_name, "params": s.params,
                    "before": before, "after": {"status": s.status.value},
                    "superseded_by": sorted(t.task_id for t in segments), "at_utc": now})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                   encoding="utf-8")
    print(f"migrated: {len(plan)} zombies + 1 suspect, audit: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
