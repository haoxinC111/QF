#!/usr/bin/env python
"""B3 repair: 主批次完成后执行(2026-07-21 用户图片方案)。

前置约束: 主 worker 已停止(本脚本独立进程运行); B3 停在原批次,不进 B4。

步骤(--steps 逗号选择,默认全部按序执行):
  audit     fina_audit 按 symbol 全历史修复(5,866 任务,单码查询,预检已验证
            6 只不同上市状态股票全历史可用,最大 39 行远低于 row_cap)。
            32 个年段失败任务标记 SUPERSEDED_INVALID_PARTITION(绝不标 success),
            生成 supersedes 映射与新 manifest SHA,旧 manifest 不静默修改。
  disclosure disclosure_date 325 个隔离按 schema 指纹分类:
            851fe39e(仅缺可选列 modify_date)→ 登记 nullable 变体指纹后重跑回收;
            其他异常(缺主键/类型冲突/响应异常)重试后仍异常则保留隔离并报告。
  empty     全部 confirmed_empty 延迟复查一次: 仍空才保持 confirmed_empty
            (metadata 记复查时间); 有数据的受控迁移为 retryable 后由管线重抓。
            重点复核 express_vip 的 ~1,642 个空。
  accept    验收: fina_audit 全部终态、未解决 retryable=0、无法解释 quarantine=0、
            Raw/Bronze/SHA 全量一致、无 cap 截断。写 repair decision。

Usage:
    uv run --no-sync python scripts/run_b3_repair.py \
        --snapshot p0_B3_financial_20260721_092031 [--steps audit,disclosure,empty,accept]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ashare_quant.archive.config import ArchiveConfig  # noqa: E402
from ashare_quant.archive.pipeline import ArchivePipeline, task_id  # noqa: E402
from ashare_quant.archive.provider import TushareCompatibleHttpProvider  # noqa: E402
from ashare_quant.archive.registry import default_inventory  # noqa: E402
from ashare_quant.archive.state import TaskStateDB, TaskStatus  # noqa: E402

BRONZE = REPO_ROOT / "data_lake" / "bronze" / "research_proxy_unverified"
# disclosure_date 缺 modify_date 可选列变体的**完整**指纹(2026-07-22 修正:
# 曾误用 12 字符前缀登记,pipeline 按完整指纹比对导致回收失败一轮;
# 完整值取自 324 个隔离任务 metadata.schema_drift.observed,全部一致)。
DISCLOSURE_VARIANT_FP = "851fe39e5b6486034313f2b2944dfe008c8c3a60003e6a21c3d6286fa4088fd4"
EXPECTED_DISCLOSURE_VARIANT_COLS = ["ts_code", "ann_date", "end_date", "pre_date", "actual_date"]

# 瞬时 retryable 精确重放(2026-07-22 用户指令): 收官时统计的 112 个
# 非 fina_audit 瞬时错误(429/500/timeout)的端点分布,重放前硬断言。
TRANSIENT_EXPECTED_DIST = {
    "cashflow_vip": 43, "express_vip": 40, "fina_mainbz_vip": 10,
    "income_vip": 6, "forecast_vip": 5, "fina_indicator_vip": 4,
    "disclosure_date": 4,
}
TRANSIENT_EXPECTED_TOTAL = 112

log = logging.getLogger("run_b3_repair")


def _stock_universe() -> list[str]:
    listed = pd.read_parquet(BRONZE / "stock_basic" / "stock_basic_list_status=L_phase_a_20260716_135101.parquet")
    delisted = pd.read_parquet(BRONZE / "stock_basic" / "stock_basic_list_status=D_phase_a_20260716_135101.parquet")
    return sorted(set(listed["ts_code"]) | set(delisted["ts_code"]))


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


def _manifest_task_ids(snapshot: str) -> list[str]:
    """读取已物化 manifest(原 46,960 任务),禁止重建。"""
    path = REPO_ROOT / "data_lake" / "reports" / "batches" / "B3_financial" / "batch_manifest.jsonl"
    manifest = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert manifest["snapshot_id"] == snapshot, "manifest snapshot 不符"
    return [t["task_id"] for t in manifest["tasks"]]


def select_transient_retry_candidates(config, db, snapshot: str, manifest_ids: set[str]) -> list:
    """瞬时 retryable 候选: 排除 fina_audit,必须属于原 manifest 且归属本 snapshot。"""
    rows = []
    for t in db.list_tasks():
        if t.status != TaskStatus.RETRYABLE_ERROR or t.api_name == "fina_audit":
            continue
        if t.task_id not in manifest_ids:
            continue
        if task_id(config.provider.name, t.api_name, t.params, t.fields, snapshot) != t.task_id:
            continue
        rows.append(t)
    return rows


def step_transient_retry(config, db, provider, inventory, snapshot: str,
                         artifact_dir: Path, *, first_round: bool) -> dict:
    """112 个瞬时 retryable 的精确任务重放。

    直接读取原任务的 task_id + params + fields + snapshot(不调用
    build_context、不重新生成参数),逐条校验重算 task_id 与原值一致。
    """
    manifest_ids = set(_manifest_task_ids(snapshot))
    candidates = select_transient_retry_candidates(config, db, snapshot, manifest_ids)
    dist: dict[str, int] = {}
    for t in candidates:
        dist[t.api_name] = dist.get(t.api_name, 0) + 1
    if first_round:
        # 重放前断言(用户指令): 恰好 112、端点分布一致、全部属于原 manifest、
        # 32 个 fina_audit 错误分区任务不在其中(选择器已排除,再断言兜底)。
        assert len(candidates) == TRANSIENT_EXPECTED_TOTAL, (
            f"瞬时 retryable 计数断言失败: 期望 {TRANSIENT_EXPECTED_TOTAL}, 实际 {len(candidates)}; dist={dist}")
        assert dist == TRANSIENT_EXPECTED_DIST, (
            f"端点分布断言失败: 期望 {TRANSIENT_EXPECTED_DIST}, 实际 {dist}")
        assert all(t.api_name != "fina_audit" for t in candidates)
        assert all(t.task_id in manifest_ids for t in candidates)

    specs = []
    for t in candidates:
        ep = inventory.endpoints[t.api_name]
        spec = ep.to_spec()
        spec.params_template = dict(t.params)  # 原参数(如 end_date=20260720),不重建
        assert spec.fields == t.fields, f"fields 漂移: {t.task_id[:12]}"
        assert task_id(config.provider.name, t.api_name, spec.params_template,
                       spec.fields, snapshot) == t.task_id, f"task_id 不一致: {t.task_id[:12]}"
        specs.append(spec)
    log.info("瞬时 retryable 精确重放: %d 个任务%s", len(specs), "(首轮)" if first_round else "(第二轮)")
    started = time.perf_counter()
    result = None
    if specs:
        pipeline = ArchivePipeline(config, provider, db, snapshot_id=snapshot, symbol_universe=[])
        result = pipeline.run_tasks(specs, skip_existing=True)
    elapsed = time.perf_counter() - started

    manifest_ids2 = set(_manifest_task_ids(snapshot))
    remaining = select_transient_retry_candidates(config, db, snapshot, manifest_ids2)
    report = {
        "round": 1 if first_round else 2,
        "attempted": len(specs),
        "completed": result.tasks_completed if result else 0,
        "failed": result.tasks_failed if result else 0,
        "remaining": len(remaining),
        "remaining_detail": [
            {"task_id": t.task_id, "api_name": t.api_name,
             "error": (t.last_error or "")[:80]} for t in remaining
        ],
        "elapsed_seconds": round(elapsed, 1),
    }
    (artifact_dir / f"transient_retry_round{report['round']}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("重放结果: %s", {k: v for k, v in report.items() if k != "remaining_detail"})
    return report


def step_audit(config, db, provider, inventory, snapshot: str, artifact_dir: Path) -> dict:
    """fina_audit symbol 修复 + 32 年段任务 supersedes 标记。"""
    ep = inventory.endpoints["fina_audit"]
    assert ep.split_unit == "symbol", f"fina_audit split_unit 应为 symbol(防回退): {ep.split_unit!r}"
    assert ep.row_cap == 7000, f"fina_audit row_cap 回退: {ep.row_cap!r}"

    symbols = _stock_universe()
    specs = []
    for code in symbols:
        spec = ep.to_spec()
        spec.params_template = {"ts_code": code}
        specs.append(spec)
    log.info("fina_audit symbol 任务: %d", len(specs))

    pipeline = ArchivePipeline(config, provider, db, snapshot_id=snapshot, symbol_universe=[])
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.perf_counter()
    result = pipeline.run_tasks(specs, skip_existing=True)
    elapsed = time.perf_counter() - started
    log.info("fina_audit 修复完成: completed=%d failed=%d empty=%d rows=%d 耗时=%.0fs",
             result.tasks_completed, result.tasks_failed, result.tasks_empty, result.rows_total, elapsed)

    # 32 个年段任务 -> SUPERSEDED_INVALID_PARTITION(审计 + supersedes 映射)。
    new_ids = sorted(
        task_id(config.provider.name, "fina_audit", dict(s.params_template), s.fields, snapshot)
        for s in specs
    )
    manifest_sha = hashlib.sha256("\n".join(new_ids).encode("utf-8")).hexdigest()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    audit_records, supersedes_map = [], {}
    for t in db.list_tasks():
        if t.api_name != "fina_audit" or t.status != TaskStatus.RETRYABLE_ERROR:
            continue
        # 年段任务形态: params 仅含 start_date/end_date(无 ts_code)
        if "ts_code" in t.params:
            continue
        if task_id(config.provider.name, t.api_name, t.params, t.fields, snapshot) != t.task_id:
            continue
        before = {"status": t.status.value, "last_error": t.last_error}
        t.status = TaskStatus.SUPERSEDED_INVALID_PARTITION
        t.metadata["supersede_reason"] = "split_unit=year 与网关能力不匹配(全市场/年段查询 HTTP 500),由 symbol 拆分全历史任务集替代"
        t.metadata["superseded_by_manifest_sha256"] = manifest_sha
        t.metadata["migration_id"] = "b3_repair_fina_audit_symbol"
        t.metadata["migrated_at_utc"] = now
        db.upsert(t)
        supersedes_map[t.task_id] = {
            "superseded_by_batch": "b3_repair_fina_audit_symbol",
            "coverage": "fina_audit 按 symbol 全历史任务集(见 manifest sha)",
            "new_manifest_sha256": manifest_sha,
            "reason": "invalid_partition_year_split_gateway_500",
        }
        audit_records.append({"task_id": t.task_id, "params": t.params, "before": before,
                              "after": {"status": t.status.value}, "at_utc": now})
    (artifact_dir / "fina_audit_year_tasks_audit.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audit_records), encoding="utf-8")
    (artifact_dir / "supersedes_fina_audit.json").write_text(json.dumps({
        "schema_version": 1,
        "repair_snapshot": snapshot,
        "note": "fina_audit 年段任务由 symbol 全历史任务集替代;研究层不得使用年段任务(无数据),按 ts_code 读取 symbol 任务产物",
        "new_manifest_sha256": manifest_sha,
        "new_task_count": len(new_ids),
        "map": supersedes_map,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("年段任务迁移 %d 个,新 manifest sha=%s", len(audit_records), manifest_sha[:16])
    return {"tasks": len(specs), "completed": result.tasks_completed, "failed": result.tasks_failed,
            "empty": result.tasks_empty, "year_tasks_migrated": len(audit_records),
            "new_manifest_sha256": manifest_sha, "started_at": started_at, "elapsed": elapsed}


def step_disclosure(config, db, provider, inventory, snapshot: str, artifact_dir: Path) -> dict:
    """disclosure_date 隔离按指纹分类回收。

    分类依据 metadata.schema_drift.observed 的**完整**指纹(2026-07-22 修正:
    曾误用 12 字符前缀登记导致回收失败;变体登记文件已按完整指纹重写)。
    候选同时纳入 retryable 的 disclosure 任务(瞬时错误一并重试)。
    """
    ep = inventory.endpoints["disclosure_date"]

    def _observed_fp(t) -> str:
        return (t.metadata or {}).get("schema_drift", {}).get("observed", "")

    def _owned(t) -> bool:
        return task_id(config.provider.name, t.api_name, t.params, t.fields, snapshot) == t.task_id

    quarantined = [
        t for t in db.list_tasks()
        if t.api_name == "disclosure_date" and t.status == TaskStatus.QUARANTINED
        and _observed_fp(t) == DISCLOSURE_VARIANT_FP and _owned(t)
    ]
    other = [
        t for t in db.list_tasks()
        if t.api_name == "disclosure_date" and t.status == TaskStatus.QUARANTINED
        and _observed_fp(t) != DISCLOSURE_VARIANT_FP and _owned(t)
    ]
    retryable = [
        t for t in db.list_tasks()
        if t.api_name == "disclosure_date" and t.status == TaskStatus.RETRYABLE_ERROR and _owned(t)
    ]
    log.info("disclosure 隔离分类: 缺可选列变体 %d, 其他异常 %d, 瞬时 retryable %d",
             len(quarantined), len(other), len(retryable))

    variant_path = config.catalog_dir / "schema_registry" / "disclosure_date" / f"{DISCLOSURE_VARIANT_FP}.json"
    if quarantined and not variant_path.exists():
        variant_path.parent.mkdir(parents=True, exist_ok=True)
        variant_path.write_text(json.dumps({
            "endpoint": "disclosure_date",
            "fingerprint": DISCLOSURE_VARIANT_FP,
            "columns": EXPECTED_DISCLOSURE_VARIANT_COLS,
            "snapshot_id": snapshot,
            "row_count": 0,
            "variant_note": "nullable union schema: 披露计划从未修改的股票网关不返回 modify_date 列(可选列缺省),数据有效,2026-07-22 按隔离任务 schema_drift.observed 完整指纹登记;读取层按 6 列 union(modify_date 补 null)",
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("变体指纹已登记: %s", variant_path.name)

    recovered = {"completed": 0, "failed": 0, "empty": 0}
    rerun = quarantined + retryable
    if rerun:
        specs = []
        for t in rerun:
            spec = ep.to_spec()
            spec.params_template = dict(t.params)
            specs.append(spec)
        pipeline = ArchivePipeline(config, provider, db, snapshot_id=snapshot, symbol_universe=[])
        result = pipeline.run_tasks(specs, skip_existing=True)
        recovered = {"completed": result.tasks_completed, "failed": result.tasks_failed, "empty": result.tasks_empty}
        log.info("disclosure 回收: %s", recovered)

    # 其他异常类: 重试由管线完成;仍隔离的保留并列入报告。
    remaining = [
        t.task_id for t in db.list_tasks()
        if t.api_name == "disclosure_date" and t.status == TaskStatus.QUARANTINED
        and task_id(config.provider.name, t.api_name, t.params, t.fields, snapshot) == t.task_id
    ]
    (artifact_dir / "disclosure_quarantine_remaining.json").write_text(
        json.dumps({"remaining_quarantined": remaining, "other_anomaly_count": len(other)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return {"variant_class": len(quarantined), "other_class": len(other),
            "retryable_class": len(retryable),
            "recovered": recovered, "remaining_quarantined": len(remaining)}


def step_empty_recheck(config, db, provider, inventory, snapshot: str, artifact_dir: Path) -> dict:
    """confirmed_empty 延迟复查: 两次皆空才保持,有数据的受控转 retryable 重抓。"""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    empties = [
        t for t in db.list_tasks()
        if t.status == TaskStatus.CONFIRMED_EMPTY
        and task_id(config.provider.name, t.api_name, t.params, t.fields, snapshot) == t.task_id
    ]
    log.info("confirmed_empty 复查: %d 个", len(empties))
    still_empty, refetch = 0, []
    audit_records = []
    for i, t in enumerate(empties):
        resp = provider.request(t.api_name, dict(t.params), fields=t.fields)
        if resp.is_success and resp.row_count > 0:
            refetch.append(t)
            audit_records.append({"task_id": t.task_id, "api_name": t.api_name, "params": t.params,
                                  "recheck_rows": resp.row_count, "action": "to_retryable", "at_utc": now})
        else:
            still_empty += 1
            t.metadata["delayed_recheck_at_utc"] = now
            t.metadata["delayed_recheck_result"] = "still_empty"
            db.upsert(t)
        if (i + 1) % 200 == 0:
            log.info("复查进度 %d/%d (转重抓 %d)", i + 1, len(empties), len(refetch))
    # 有数据的: 受控迁移 retryable(审计)后由管线正式重抓落盘。
    for t in refetch:
        t.status = TaskStatus.RETRYABLE_ERROR
        t.last_error = "delayed_empty_recheck: 复查有数据,转重抓"
        t.metadata["migration_id"] = "b3_repair_empty_recheck"
        t.metadata["migrated_at_utc"] = now
        db.upsert(t)
    (artifact_dir / "empty_recheck_audit.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audit_records), encoding="utf-8")
    recovered = {"completed": 0, "failed": 0}
    if refetch:
        by_api = {}
        for t in refetch:
            by_api.setdefault(t.api_name, []).append(t)
        specs = []
        for api, tasks in by_api.items():
            ep = inventory.endpoints[api]
            for t in tasks:
                spec = ep.to_spec()
                spec.params_template = dict(t.params)
                specs.append(spec)
        pipeline = ArchivePipeline(config, provider, db, snapshot_id=snapshot, symbol_universe=[])
        result = pipeline.run_tasks(specs, skip_existing=True)
        recovered = {"completed": result.tasks_completed, "failed": result.tasks_failed}
    log.info("empty 复查: 仍空 %d, 转重抓 %d, 重抓成功 %d", still_empty, len(refetch), recovered["completed"])
    return {"checked": len(empties), "still_empty": still_empty,
            "refetched": len(refetch), "refetch_completed": recovered["completed"]}


def step_accept(config, db, snapshot: str, artifact_dir: Path) -> dict:
    """验收: fina_audit 终态、retryable=0、无法解释隔离=0、Raw/Bronze/SHA、无 cap。"""
    tasks = [
        t for t in db.list_tasks()
        if task_id(config.provider.name, t.api_name, t.params, t.fields, snapshot) == t.task_id
    ]
    by_status: dict[str, int] = {}
    for t in tasks:
        by_status[t.status.value] = by_status.get(t.status.value, 0) + 1
    audit_terminal = {"success", "confirmed_empty"}
    fina_audit_nonterminal = [
        t.task_id for t in tasks
        if t.api_name == "fina_audit" and t.status.value not in audit_terminal | {"superseded_invalid_partition"}
    ]
    retryable = by_status.get("retryable_error", 0)
    quarantined = by_status.get("quarantined", 0)
    running = by_status.get("running", 0) + by_status.get("pending", 0)
    cap_hits = sum(1 for t in tasks if t.row_count in (6000, 7000, 9000, 10000, 12000))
    # Raw/Bronze/SHA 全量一致(success 任务)
    sha_bad, missing = 0, 0
    for t in tasks:
        if t.status != TaskStatus.SUCCESS:
            continue
        if not t.raw_path or not Path(t.raw_path).exists() or not t.bronze_path or not Path(t.bronze_path).exists():
            missing += 1
            continue
        if hashlib.sha256(Path(t.raw_path).read_bytes()).hexdigest() != t.raw_sha256:
            sha_bad += 1
    gates = {
        "fina_audit_all_terminal": len(fina_audit_nonterminal) == 0,
        "no_retryable_left": retryable == 0,
        "no_unexplained_quarantine": quarantined == 0,
        "no_running_pending": running == 0,
        "raw_bronze_sha_consistent": sha_bad == 0 and missing == 0,
        "no_cap_truncation": cap_hits == 0,
    }
    decision = {
        "batch": "B3_repair",
        "source_snapshot": snapshot,
        "by_status": by_status,
        "fina_audit_nonterminal": fina_audit_nonterminal[:10],
        "raw_sha_bad": sha_bad, "files_missing": missing, "cap_hits": cap_hits,
        "gates": gates,
        "decision": "pass" if all(gates.values()) else "fail",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (artifact_dir / "repair_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("验收: %s", json.dumps(gates, ensure_ascii=False))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--config", default="config.archive.yaml")
    parser.add_argument("--steps", default="transient,audit,disclosure,empty,accept")
    parser.add_argument("--artifact-batch", default="B3_repair")
    args = parser.parse_args()

    artifact_dir = REPO_ROOT / "data_lake" / "reports" / "batches" / args.artifact_batch
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(REPO_ROOT / "data_lake" / "reports" / f"{args.artifact_batch}.log", encoding="utf-8"),
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
    if "transient" in steps:
        # 最多两轮(用户指令): 仍剩瞬时 retryable 则暂停汇报,不无限重试,
        # 清零后才继续后续 repair 步骤。
        cleared = False
        for round_no in (1, 2):
            report = step_transient_retry(config, db, provider, inventory,
                                          args.snapshot, artifact_dir,
                                          first_round=(round_no == 1))
            summary[f"transient_round{round_no}"] = report
            if report["remaining"] == 0:
                cleared = True
                break
            if round_no == 1:
                log.info("仍有 %d 个瞬时 retryable,等待 10 分钟后第二轮…", report["remaining"])
                time.sleep(600)
        if not cleared:
            last = summary["transient_round2"]
            log.error("两轮后仍剩 %d 个瞬时 retryable,暂停汇报,不继续后续步骤", last["remaining"])
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
            return 1
    if "audit" in steps:
        summary["audit"] = step_audit(config, db, provider, inventory, args.snapshot, artifact_dir)
    if "disclosure" in steps:
        summary["disclosure"] = step_disclosure(config, db, provider, inventory, args.snapshot, artifact_dir)
    if "empty" in steps:
        summary["empty"] = step_empty_recheck(config, db, provider, inventory, args.snapshot, artifact_dir)
    if "accept" in steps:
        summary["accept"] = step_accept(config, db, args.snapshot, artifact_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    decision = summary.get("accept", {}).get("decision")
    return 0 if decision in (None, "pass") else 1


if __name__ == "__main__":
    sys.exit(main())
