#!/usr/bin/env python
"""B2_repair v2: 以 fields="" + row_cap=7000 动态二分重抓 index_weight。

v1 教训(2026-07-20 实测): 显式 fields 会让网关只返回月末权重,丢掉月中
临时调样快照;fields="" 在大区间返回全粒度,且月级小区间可避开畸形
schema(仅 con_code 一列,只出现在大区间响应)。

范围(--task-list,默认 repair_task_list_v2.json, 1,171 原任务 / 1,193 个根 spec):
  - 1,169 个被 7000 行硬上限静默截断的任务: 年段 fields="" + 动态二分;
  - 2 个畸形 schema 任务: 按月拆分(24 个月级 spec)fields="" 获取。

原则:
  - 不改原 B2 与 repair v1 的任何状态与文件;
  - v2 写入独立 snapshot (p0_B2_repair_*),artifact 目录 B2_repair_v2;
  - supersedes 两级: 原任务 -> v2 根; v1 根 -> v2 根。研究层只引用 v2。

Usage:
    python scripts/run_b2_repair.py --v2 [--snapshot p0_B2_repair_YYYYMMDD_HHMMSS]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

for _var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_var, None)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ashare_quant.archive.batch import write_batch_artifacts  # noqa: E402
from ashare_quant.archive.config import ArchiveConfig  # noqa: E402
from ashare_quant.archive.pipeline import ArchivePipeline, task_id  # noqa: E402
from ashare_quant.archive.provider import TushareCompatibleHttpProvider  # noqa: E402
from ashare_quant.archive.registry import default_inventory  # noqa: E402
from ashare_quant.archive.state import TaskStateDB  # noqa: E402

EXPECTED_ROW_CAP = 7000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="v2 模式(fields=空,月拆畸形任务)")
    parser.add_argument("--snapshot", default=None, help="复用已有 repair snapshot(断点续跑)")
    parser.add_argument("--config", default="config.archive.yaml")
    parser.add_argument("--task-list", default=None)
    parser.add_argument("--artifact-batch", default=None)
    args = parser.parse_args()

    artifact_batch = args.artifact_batch or ("B2_repair_v2" if args.v2 else "B2_repair")
    artifact_dir = REPO_ROOT / "data_lake" / "reports" / "batches" / artifact_batch
    artifact_dir.mkdir(parents=True, exist_ok=True)
    task_list_path = Path(args.task_list) if args.task_list else artifact_dir / (
        "repair_task_list_v2.json" if args.v2 else "repair_task_list.json"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(REPO_ROOT / "data_lake" / "reports" / f"{artifact_batch}.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)

    repair_items = json.loads(task_list_path.read_text(encoding="utf-8"))["tasks"]
    log.info("修复清单(%s): %d 条", task_list_path.name, len(repair_items))

    config = ArchiveConfig.from_yaml(REPO_ROOT / args.config)
    config.validate_for_run()
    config.ensure_dirs()

    provider = TushareCompatibleHttpProvider(
        url_env=config.provider.base_url_env,
        token_env=config.provider.token_env,
        forbid_token_env=config.provider.forbid_token_env,
        source_provider=config.provider.name,
        allowed_hosts=config.provider.allowed_hosts,
        api_key_env=config.provider.api_key_env,
        api_key_header=config.provider.api_key_header,
    )
    inventory = default_inventory()
    ep = inventory.endpoints["index_weight"]
    # 防配置回退: fields 必须为空(全粒度),row_cap 必须登记。
    expected_fields = ""
    assert ep.fields == expected_fields, f"index_weight fields 应为空(全粒度): {ep.fields!r}"
    assert ep.row_cap == EXPECTED_ROW_CAP, f"index_weight row_cap 回退: {ep.row_cap!r}"

    snapshot_id = args.snapshot or time.strftime("p0_B2_repair_%Y%m%d_%H%M%S", time.gmtime())
    db = TaskStateDB(config.catalog_dir / "archive.duckdb")

    # 展开: 每个原任务 1 个或多个根 spec(畸形任务按月拆)。
    item_specs: list[tuple[dict, list]] = []
    for item in repair_items:
        params_list = item.get("params_list") or [item["params"]]
        specs = []
        for params in params_list:
            spec = ep.to_spec()
            spec.params_template = dict(params)
            specs.append(spec)
        item_specs.append((item, specs))
    all_specs = [s for _, specs in item_specs for s in specs]
    log.info("根 spec 总数: %d", len(all_specs))

    pipeline = ArchivePipeline(config, provider, db, snapshot_id=snapshot_id, symbol_universe=[])
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.perf_counter()
    result = pipeline.run_tasks(all_specs, skip_existing=True)
    elapsed = time.perf_counter() - started
    log.info(
        "修复执行完成: completed=%d failed=%d empty=%d rows=%d 耗时=%.0fs",
        result.tasks_completed, result.tasks_failed, result.tasks_empty, result.rows_total, elapsed,
    )

    # supersedes: 原任务 -> v2 根(列表); 若条目带 v1_root_task_id,同时登记 v1 -> v2。
    # 增量模式: 若 supersedes.json 已存在(补丁批次),合并而非覆盖。
    sup_path = artifact_dir / "supersedes.json"
    supersedes_map = {}
    v1_map = {}
    if sup_path.exists():
        prev = json.loads(sup_path.read_text(encoding="utf-8"))
        supersedes_map.update(prev.get("map", {}))
        v1_map.update(prev.get("v1_superseded", {}))
    for item, specs in item_specs:
        root_ids = [
            task_id(config.provider.name, spec.api_name, dict(spec.params_template), spec.fields, snapshot_id)
            for spec in specs
        ]
        original_id = item["original_task_id"]
        prev_entry = supersedes_map.get(original_id, {})
        merged_roots = list(prev_entry.get("repair_root_task_ids", [])) + [
            r for r in root_ids if r not in prev_entry.get("repair_root_task_ids", [])
        ]
        supersedes_map[original_id] = {
            "repair_root_task_ids": merged_roots,
            "repair_snapshot": snapshot_id,
            "reason": prev_entry.get("reason", item["reason"]),
        }
        for spec, rid in zip(specs, root_ids):
            root_task = db.get(rid)
            if root_task is not None:
                root_task.metadata["supersedes"] = original_id
                root_task.metadata["supersede_reason"] = item["reason"]
                db.upsert(root_task)
        original_task = db.get(original_id)
        if original_task is not None:
            original_task.metadata["superseded_by"] = merged_roots[0] if len(merged_roots) == 1 else merged_roots
            original_task.metadata["supersede_snapshot"] = snapshot_id
            original_task.metadata["supersede_reason"] = item["reason"]
            db.upsert(original_task)
        v1_root = item.get("v1_root_task_id")
        if v1_root:
            v1_map[v1_root] = root_ids
            v1_task = db.get(v1_root)
            if v1_task is not None:
                v1_task.metadata["superseded_by"] = root_ids[0] if len(root_ids) == 1 else root_ids
                v1_task.metadata["supersede_snapshot"] = snapshot_id
                v1_task.metadata["supersede_reason"] = "v1_explicit_fields_month_end_granularity"
                db.upsert(v1_task)
    (artifact_dir / "supersedes.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "repair_snapshot": snapshot_id,
                "source_snapshot": "p0_B2_universe_20260717_022655",
                "note": "研究层读取 index_weight 时,凡 original_task_id 出现在本表中,必须用 v2 repair 快照数据替代;v1_superseded 中的 v1 根任务同样被替代",
                "map": supersedes_map,
                "v1_superseded": v1_map,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    log.info("supersedes 映射: 原任务 %d 条, v1 根 %d 条", len(supersedes_map), len(v1_map))

    decision = write_batch_artifacts(
        config, db, artifact_batch, snapshot_id, all_specs, result, started_at, elapsed
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["decision"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
