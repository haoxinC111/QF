#!/usr/bin/env python
"""B4 repair: 主批次完成后执行(2026-07-23 用户图片方案)。

前置约束: 主 worker 已停止;B4 停在原批次,不进 B1 repair。

步骤(--steps 逗号选择,默认全部按序):
  sharefloat  share_float 截断收复:
              - 528 个恰好 6000 行的任务: 全历史窗口(start/end)重抓,
                row_cap=6000 触发管线递归二分(不重叠窗口,不用 ann_date——
                网关忽略该参数,实测单日子查询仍返 6000),叶子必须 <6000;
                单日仍撞帽的转 quarantined 并报错;
              - 原任务保留并标记 SUPERSEDED_TRUNCATED_CAP(绝不删改文件),
                生成 repair manifest、supersedes 映射与 SHA、审计 JSONL;
              - 352 个 5000-5999 行任务: 两半窗口对账(主键集合+行数+SHA),
                一致保留(metadata 标注),不一致按截断同法收复。
  holdertrade stk_holdertrade 18 个 schema 变体: 仅当主键完整且缺失列确认为
              可选列时按 nullable union 登记完整指纹回收;否则保持隔离。
  transient   最终 retryable 按冻结 manifest+原参数精确重放,最多两轮,
              不重建 context;两轮后仍有则暂停汇报。
  empty       动态全量 empty 复核: 两次皆空才保持 confirmed_empty。
  accept      全量验收: retryable/running/pending=0、quarantine=0(或仅余已
              报告的解释项)、无 success 恰满 cap(2000/6000)、叶子<6000、
              Raw/Bronze/SHA 一致、candidate_empty=0、主键完整。

Usage:
    set -a; . ./.env; set +a && uv run --no-sync python scripts/run_b4_repair.py \
        [--steps sharefloat,holdertrade,transient,empty,accept]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import zstandard

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ashare_quant.archive.config import ArchiveConfig  # noqa: E402
from ashare_quant.archive.pipeline import ArchivePipeline, task_id  # noqa: E402
from ashare_quant.archive.provider import TushareCompatibleHttpProvider  # noqa: E402
from ashare_quant.archive.registry import default_inventory  # noqa: E402
from ashare_quant.archive.state import TaskStateDB, TaskStatus  # noqa: E402

SNAPSHOT = "p0_B4_events_20260722_141540"
SHARE_FLOAT_CAP = 6000
SHARE_FLOAT_EARLIEST = "20050101"
LATEST_TRADE_DATE = "20260721"  # 冻结 context(ceed5b1f…)
REPORT_DIR = REPO_ROOT / "data_lake" / "reports" / "batches" / "B4_repair"

log = logging.getLogger("run_b4_repair")


def _provider(config: ArchiveConfig) -> TushareCompatibleHttpProvider:
    return TushareCompatibleHttpProvider(
        url_env=config.provider.base_url_env,
        token_env=config.provider.token_env,
        forbid_token_env=config.provider.forbid_token_env,
        source_provider=config.provider.name,
        allowed_hosts=config.provider.allowed_hosts,
        api_key_env=config.provider.api_key_env,
        api_key_header=config.provider.api_key_header,
    )


def _owned(config, t) -> bool:
    return task_id(config.provider.name, t.api_name, t.params, t.fields, SNAPSHOT) == t.task_id


def _frozen_ids() -> set[str]:
    path = REPO_ROOT / "data_lake" / "reports" / "batches" / "B4_events" / f"frozen_specs_{SNAPSHOT}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        task_id(payload["provider"], e["api_name"], e["params"], e["fields"], SNAPSHOT)
        for e in payload["specs"]
    }


def _read_raw_pk_set(raw_path: str, pk_cols: list[str]) -> tuple[set, int, str]:
    raw = zstandard.ZstdDecompressor().stream_reader(open(raw_path, "rb"))
    data = json.loads(raw.read().decode())
    payload = data.get("data", data)
    cols, items = payload["fields"], payload["items"]
    idx = [cols.index(c) for c in pk_cols]
    keys = {tuple(r[i] for i in idx) for r in items}
    sha = hashlib.sha256(
        json.dumps(sorted(items), ensure_ascii=False, default=str).encode()
    ).hexdigest()
    return keys, len(items), sha


def step_sharefloat(config, db, provider, inventory, artifact_dir: Path) -> dict:
    ep = inventory.endpoints["share_float"]
    assert ep.row_cap == SHARE_FLOAT_CAP, f"share_float row_cap 应为 {SHARE_FLOAT_CAP}(先改 registry): {ep.row_cap!r}"
    frozen = _frozen_ids()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    capped = [
        t for t in db.list_tasks()
        if t.api_name == "share_float" and t.status == TaskStatus.SUCCESS
        and t.row_count == SHARE_FLOAT_CAP and t.task_id in frozen and _owned(config, t)
    ]
    borderline = [
        t for t in db.list_tasks()
        if t.api_name == "share_float" and t.status == TaskStatus.SUCCESS
        and 5000 <= t.row_count < SHARE_FLOAT_CAP and t.task_id in frozen and _owned(config, t)
    ]
    log.info("share_float 截断 %d 个,边界(5000-5999) %d 个", len(capped), len(borderline))

    # --- 352 边界任务对账: 两半窗口 PK 集合+行数+SHA ---
    # 三类结果: match(一致保留) / mismatch(数据不一致,进修复) /
    # inconclusive(请求失败或半窗撞帽,无法判定——绝非"不一致",保留原任务,
    # 待风暴平息后复核一轮,仍无法判定则保留并在报告中列清单)。
    pk_cols = list(ep.primary_key)

    def reconcile_one(t) -> str:
        code = t.params["ts_code"]
        halves = [
            {"ts_code": code, "start_date": SHARE_FLOAT_EARLIEST, "end_date": "20151231"},
            {"ts_code": code, "start_date": "20160101", "end_date": LATEST_TRADE_DATE},
        ]
        split_keys, split_rows, split_items = set(), 0, []
        for params in halves:
            resp = provider.request("share_float", params)
            if not resp.is_success:
                return "inconclusive"
            if resp.row_count >= SHARE_FLOAT_CAP:  # 半窗撞帽,无法判定
                return "inconclusive"
            idx = [resp.columns.index(c) for c in pk_cols]
            split_keys |= {tuple(r[j] for j in idx) for r in resp.items}
            split_rows += resp.row_count
            split_items += resp.items
        orig_keys, orig_rows, orig_sha = _read_raw_pk_set(t.raw_path, pk_cols)
        split_sha = hashlib.sha256(
            json.dumps(sorted(split_items), ensure_ascii=False, default=str).encode()
        ).hexdigest()
        if orig_keys == split_keys and orig_rows == split_rows and orig_sha == split_sha:
            return "match"
        return "mismatch"

    mismatched, inconclusive = [], []
    reconciled = 0
    pending = list(borderline)
    for attempt in (1, 2):
        still = []
        for i, t in enumerate(pending):
            verdict = reconcile_one(t)
            if verdict == "match":
                reconciled += 1
                t.metadata["window_reconcile"] = {
                    "method": "half_window_pk_rows_sha", "result": "match", "at_utc": now,
                }
                db.upsert(t)
            elif verdict == "mismatch":
                mismatched.append(t)
            else:
                still.append(t)
            if (i + 1) % 50 == 0:
                log.info("边界对账第 %d 轮进度 %d/%d (一致 %d, 不一致 %d, 待定 %d)",
                         attempt, i + 1, len(pending), reconciled, len(mismatched), len(still))
        pending = still
        if not pending:
            break
        if attempt == 1:
            log.info("%d 个对账待定(瞬时错误),等待 10 分钟后复核一轮", len(pending))
            time.sleep(600)
    inconclusive = pending  # 两轮后仍待定: 保留原任务,报告清单
    log.info("边界对账: 一致保留 %d, 不一致进修复 %d, 待定保留 %d",
             reconciled, len(mismatched), len(inconclusive))

    # --- 截断+不一致: 全历史窗口重抓(row_cap=6000 触发管线递归二分) ---
    to_repair = capped + mismatched
    specs = []
    for t in to_repair:
        spec = ep.to_spec()
        spec.params_template = {
            "ts_code": t.params["ts_code"],
            "start_date": SHARE_FLOAT_EARLIEST,
            "end_date": LATEST_TRADE_DATE,
        }
        specs.append(spec)
    started = time.perf_counter()
    result = None
    if specs:
        pipeline = ArchivePipeline(config, provider, db, snapshot_id=SNAPSHOT, symbol_universe=[])
        result = pipeline.run_tasks(specs, skip_existing=True)
    elapsed = time.perf_counter() - started
    log.info("share_float 重抓: %s 耗时 %.0fs",
             {"completed": result.tasks_completed if result else 0,
              "failed": result.tasks_failed if result else 0}, elapsed)

    # --- 原任务标 SUPERSEDED_TRUNCATED_CAP + 映射/审计 ---
    new_ids = sorted(
        task_id(config.provider.name, "share_float", dict(s.params_template), s.fields, SNAPSHOT)
        for s in specs
    )
    repair_manifest_sha = hashlib.sha256("\n".join(new_ids).encode()).hexdigest()
    audit_records, supersedes_map = [], {}
    for t, s in zip(to_repair, specs, strict=True):
        new_id = task_id(config.provider.name, "share_float", dict(s.params_template), s.fields, SNAPSHOT)
        before = {"status": t.status.value, "row_count": t.row_count, "raw_sha256": t.raw_sha256}
        t.status = TaskStatus.SUPERSEDED_TRUNCATED_CAP
        t.metadata["supersede_reason"] = (
            f"响应恰满真实 cap {SHARE_FLOAT_CAP}(静默截断),由不重叠日期窗口二分任务集承载"
            if t in capped else "边界任务两半窗口对账不一致,按截断处理"
        )
        t.metadata["superseded_by_task_id"] = new_id
        t.metadata["superseded_by_repair_manifest_sha256"] = repair_manifest_sha
        t.metadata["research_eligible"] = False
        t.metadata["migration_id"] = "b4_repair_sharefloat_cap"
        t.metadata["migrated_at_utc"] = now
        db.upsert(t)
        supersedes_map[t.task_id] = {
            "superseded_by": new_id,
            "repair_manifest_sha256": repair_manifest_sha,
            "reason": "truncated_at_real_cap_6000" if t in capped else "borderline_reconcile_mismatch",
        }
        audit_records.append({"task_id": t.task_id, "params": t.params, "before": before,
                              "after": {"status": t.status.value, "superseded_by": new_id}, "at_utc": now})
    (artifact_dir / "sharefloat_supersede_audit.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audit_records), encoding="utf-8")
    (artifact_dir / "supersedes_sharefloat.json").write_text(json.dumps({
        "schema_version": 1, "repair_snapshot": SNAPSHOT,
        "note": "share_float 恰满 6000 cap 的原任务由全历史窗口+递归二分任务集替代;研究层不得使用被 supersede 的任务",
        "repair_manifest_sha256": repair_manifest_sha, "new_root_task_count": len(new_ids),
        "map": supersedes_map,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- 单日仍撞帽检查: success 恰满 6000 或无法再拆的 SUSPECT 叶子 -> 隔离并报错 ---
    stuck = [
        t for t in db.list_tasks()
        if t.api_name == "share_float"
        and t.status in (TaskStatus.SUCCESS, TaskStatus.SUSPECT_TRUNCATED)
        and task_id(config.provider.name, t.api_name, t.params, t.fields, SNAPSHOT) == t.task_id
        and t.task_id not in supersedes_map
        and (t.status == TaskStatus.SUSPECT_TRUNCATED or t.row_count == SHARE_FLOAT_CAP)
    ]
    for t in stuck:
        t.status = TaskStatus.QUARANTINED
        t.last_error = f"单日/末段窗口仍恰满 cap {SHARE_FLOAT_CAP}或无法再拆,按用户指令隔离"
        t.metadata["migration_id"] = "b4_repair_sharefloat_cap"
        t.metadata["migrated_at_utc"] = now
        db.upsert(t)
    suspect = [
        t for t in db.list_tasks(status=TaskStatus.SUSPECT_TRUNCATED)
        if t.api_name == "share_float"
    ]
    return {
        "capped_repaired": len(capped), "borderline_reconciled": reconciled,
        "borderline_mismatched": len(mismatched),
        "borderline_inconclusive_kept": [t.task_id for t in inconclusive],
        "refetch": {"completed": result.tasks_completed if result else 0,
                    "failed": result.tasks_failed if result else 0},
        "stuck_quarantined": [t.task_id for t in stuck],
        "suspect_truncated_left": len(suspect),
        "repair_manifest_sha256": repair_manifest_sha,
    }


def step_holdertrade(config, db, provider, inventory, artifact_dir: Path) -> dict:
    """18 个 schema 变体: 主键完整+缺列可选 -> nullable union 登记回收;否则保持隔离。"""
    ep = inventory.endpoints["stk_holdertrade"]
    registry_dir = config.catalog_dir / "schema_registry" / "stk_holdertrade"
    registered_cols: set[str] = set()
    for fp in registry_dir.glob("*.json"):
        registered_cols |= set(json.loads(fp.read_text(encoding="utf-8")).get("columns", []))
    pk_cols = set(ep.primary_key)

    variant, keep = [], []
    for t in db.list_tasks():
        if t.api_name != "stk_holdertrade" or t.status != TaskStatus.QUARANTINED or not _owned(config, t):
            continue
        observed = (t.metadata or {}).get("schema_drift", {})
        obs_cols = set(observed.get("columns", []))
        missing = registered_cols - obs_cols
        if obs_cols and pk_cols <= obs_cols and missing and not (missing & pk_cols):
            variant.append((t, observed, sorted(missing)))
        else:
            keep.append(t)
    log.info("holdertrade 变体分类: 可回收 %d, 保持隔离 %d", len(variant), len(keep))

    recovered = {"completed": 0, "failed": 0, "empty": 0}
    if variant:
        fps = {v[1]["observed"] for v in variant}
        assert len(fps) == 1, f"变体指纹不唯一: {fps}"
        fp = fps.pop()
        fp_path = registry_dir / f"{fp}.json"
        if not fp_path.exists():
            fp_path.write_text(json.dumps({
                "endpoint": "stk_holdertrade", "fingerprint": fp,
                "columns": variant[0][1]["columns"], "snapshot_id": SNAPSHOT, "row_count": 0,
                "variant_note": f"nullable union schema: 缺可选列 {variant[0][2]}(主键完整),2026-07-23 分类登记;读取层按 {len(registered_cols)} 列 union 补 null",
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            log.info("变体指纹已登记: %s (缺列 %s)", fp[:12], variant[0][2])
        specs = []
        for t, _, _ in variant:
            spec = ep.to_spec()
            spec.params_template = dict(t.params)
            specs.append(spec)
        pipeline = ArchivePipeline(config, provider, db, snapshot_id=SNAPSHOT, symbol_universe=[])
        result = pipeline.run_tasks(specs, skip_existing=True)
        recovered = {"completed": result.tasks_completed, "failed": result.tasks_failed,
                     "empty": result.tasks_empty}
    remaining = [t.task_id for t in db.list_tasks()
                 if t.api_name == "stk_holdertrade" and t.status == TaskStatus.QUARANTINED and _owned(config, t)]
    (artifact_dir / "holdertrade_quarantine_remaining.json").write_text(
        json.dumps({"kept_quarantined": [t.task_id for t in keep], "remaining": remaining},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return {"variant_recoverable": len(variant), "kept_quarantined": [t.task_id for t in keep],
            "recovered": recovered, "remaining": len(remaining)}


def step_transient(config, db, provider, inventory, artifact_dir: Path) -> dict:
    """最终 retryable 精确重放(原参数身份校验),最多两轮。

    范围: 本 snapshot 全部 owned retryable——冻结清单内的原任务按
    「冻结 manifest 归属+原参数」校验;repair 产生的窗口子任务(不在冻结
    清单内)同样按 db 行自身 params/fields 重算 task_id 校验后重放。"""

    def candidates():
        return [
            t for t in db.list_tasks()
            if t.status == TaskStatus.RETRYABLE_ERROR and _owned(config, t)
        ]

    rounds = []
    for round_no in (1, 2):
        cands = candidates()
        specs = []
        for t in cands:
            spec = inventory.endpoints[t.api_name].to_spec()
            spec.params_template = dict(t.params)
            assert spec.fields == t.fields
            assert task_id(config.provider.name, t.api_name, spec.params_template, spec.fields, SNAPSHOT) == t.task_id
            specs.append(spec)
        result = None
        if specs:
            pipeline = ArchivePipeline(config, provider, db, snapshot_id=SNAPSHOT, symbol_universe=[])
            result = pipeline.run_tasks(specs, skip_existing=True)
        remaining = candidates()
        rounds.append({"round": round_no, "attempted": len(specs),
                       "completed": result.tasks_completed if result else 0,
                       "remaining": len(remaining)})
        log.info("transient 第 %d 轮: %s", round_no, rounds[-1])
        if not remaining:
            break
        if round_no == 1:
            log.info("等待 10 分钟后第二轮…")
            time.sleep(600)
    left = candidates()
    (artifact_dir / "transient_retry.json").write_text(json.dumps({
        "rounds": rounds,
        "left": [{"task_id": t.task_id, "api_name": t.api_name, "error": (t.last_error or "")[:80]} for t in left],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"rounds": rounds, "left": len(left)}


def step_empty(config, db, provider, artifact_dir: Path) -> dict:
    """动态全量 empty 复核: 本 snapshot 全部 owned confirmed_empty(含 repair
    产生的窗口子任务),两次皆空才保持,有数据的转 retryable 由后续重抓。"""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    empties = [
        t for t in db.list_tasks()
        if t.status == TaskStatus.CONFIRMED_EMPTY and _owned(config, t)
    ]
    log.info("confirmed_empty 复核: %d 个", len(empties))
    still_empty, refetch = 0, []
    for i, t in enumerate(empties):
        resp = provider.request(t.api_name, dict(t.params), fields=t.fields)
        if resp.is_success and resp.row_count > 0:
            refetch.append(t)
        else:
            still_empty += 1
            t.metadata["delayed_recheck_at_utc"] = now
            t.metadata["delayed_recheck_result"] = "still_empty"
            db.upsert(t)
        if (i + 1) % 500 == 0:
            log.info("复核进度 %d/%d (转重抓 %d)", i + 1, len(empties), len(refetch))
    for t in refetch:
        t.status = TaskStatus.RETRYABLE_ERROR
        t.last_error = "delayed_empty_recheck: 复查有数据,转重抓"
        t.metadata["migration_id"] = "b4_repair_empty_recheck"
        t.metadata["migrated_at_utc"] = now
        db.upsert(t)
    (artifact_dir / "empty_recheck_audit.jsonl").write_text(
        "".join(json.dumps({"task_id": t.task_id, "api_name": t.api_name, "params": t.params,
                            "action": "to_retryable", "at_utc": now}, ensure_ascii=False) + "\n"
                for t in refetch), encoding="utf-8")
    return {"checked": len(empties), "still_empty": still_empty, "refetched": len(refetch)}


def step_accept(config, db, artifact_dir: Path) -> dict:
    frozen = _frozen_ids()
    tasks = [t for t in db.list_tasks() if _owned(config, t)]
    repair_owned = [t for t in tasks if t.task_id not in frozen]  # 二分/重抓产生的新任务
    by_status: dict[str, int] = {}
    for t in tasks:
        by_status[t.status.value] = by_status.get(t.status.value, 0) + 1
    retryable = by_status.get("retryable_error", 0)
    running_pending = by_status.get("running", 0) + by_status.get("pending", 0)
    quarantined = by_status.get("quarantined", 0)
    suspect = by_status.get("suspect_truncated", 0)
    cap_full = [
        t.task_id for t in tasks
        if t.status == TaskStatus.SUCCESS and t.row_count in (2000, SHARE_FLOAT_CAP, 7000)
    ]
    sha_bad = missing = 0
    for t in tasks:
        if t.status != TaskStatus.SUCCESS:
            continue
        if not t.raw_path or not Path(t.raw_path).exists() or not t.bronze_path or not Path(t.bronze_path).exists():
            missing += 1
            continue
        if hashlib.sha256(Path(t.raw_path).read_bytes()).hexdigest() != t.raw_sha256:
            sha_bad += 1
    candidate_empty = sum(
        1 for t in tasks
        if t.status == TaskStatus.CONFIRMED_EMPTY
        and not (t.metadata or {}).get("delayed_recheck_at_utc")
    )
    gates = {
        "no_retryable_left": retryable == 0,
        "no_running_pending": running_pending == 0,
        "no_quarantine_left": quarantined == 0,
        "no_suspect_truncated": suspect == 0,
        "no_success_at_cap": len(cap_full) == 0,
        "raw_bronze_sha_consistent": sha_bad == 0 and missing == 0,
        "candidate_empty_zero": candidate_empty == 0,
    }
    decision = {
        "batch": "B4_repair", "source_snapshot": SNAPSHOT,
        "by_status": by_status, "repair_new_tasks": len(repair_owned),
        "cap_full_success": cap_full[:10], "raw_sha_bad": sha_bad, "files_missing": missing,
        "candidate_empty": candidate_empty, "gates": gates,
        "decision": "pass" if all(gates.values()) else "fail",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (artifact_dir / "repair_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("验收: %s", json.dumps(gates, ensure_ascii=False))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.archive.yaml")
    parser.add_argument("--steps", default="sharefloat,holdertrade,transient,empty,accept")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(REPO_ROOT / "data_lake" / "reports" / "B4_repair.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    config = ArchiveConfig.from_yaml(REPO_ROOT / args.config)
    config.validate_for_run()
    config.ensure_dirs()
    provider = _provider(config)
    inventory = default_inventory()
    db = TaskStateDB(config.catalog_dir / "archive.duckdb")

    summary = {}
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if "sharefloat" in steps:
        summary["sharefloat"] = step_sharefloat(config, db, provider, inventory, REPORT_DIR)
    if "holdertrade" in steps:
        summary["holdertrade"] = step_holdertrade(config, db, provider, inventory, REPORT_DIR)
    if "transient" in steps:
        summary["transient"] = step_transient(config, db, provider, inventory, REPORT_DIR)
        if summary["transient"]["left"] > 0:
            log.error("两轮后仍剩 %d 个 retryable,暂停汇报", summary["transient"]["left"])
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
            return 1
    if "empty" in steps:
        summary["empty"] = step_empty(config, db, provider, REPORT_DIR)
    if "accept" in steps:
        summary["accept"] = step_accept(config, db, REPORT_DIR)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    decision = summary.get("accept", {}).get("decision")
    return 0 if decision in (None, "pass") else 1


if __name__ == "__main__":
    sys.exit(main())
