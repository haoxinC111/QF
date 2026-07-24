"""P0 full-archive batch runner (B0..B4).

Expands each batch's endpoints into concrete EndpointSpec tasks according to
their split strategy, executes them through the ArchivePipeline (resumable,
rate-limited), and writes the per-batch artifact set:

    batch_manifest.jsonl      (appended per batch)
    checksums.sha256          (raw+bronze files of this batch snapshot)
    coverage_report.md
    schema_report.json
    failure_queue.jsonl
    batch_decision.json

Batches stop on the first non-pass decision; B3 is gated on the financial
cross-source validation having passed (market-only pass allows B0..B2).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from .config import ArchiveConfig
from .pipeline import ArchivePipeline, EndpointSpec, build_manifest
from .provider import ArchiveProvider
from .registry import EndpointInventory, InventoryEndpoint
from .state import TaskStateDB, TaskStatus

logger = logging.getLogger(__name__)

BATCH_ARTIFACT_DIR = "batches"

# index_basic 分段枚举:全量单拉被行数上限截断(全字段响应约 5000 行上限,
# 实测 market=CSI 全字段被截在 4879 而 category 细分求和 ~8800)。market 段
# 枚举不含 CSI —— CSI 必须按 category 细分(单列 category 均低于上限)。
# CICC/OTHERS 当前近空,保留段以监控未来增量。
INDEX_BASIC_MARKETS = ("SSE", "SZSE", "CNI", "SW", "CSI", "CICC", "OTHERS")
# 全字段 index_basic 响应的保守截断告警阈值:达到即按 category 细分。
INDEX_BASIC_ROW_ALERT = 4900
CSI_INDEX_CATEGORIES = (
    "规模指数",
    "行业指数",
    "主题指数",
    "策略指数",
    "风格指数",
    "债券指数",
    "多资产指数",
    "基金指数",
    "期货指数",
    "综合指数",
    "其他指数",
)

# B2 主力指数宇宙(分层归档,用户 2026-07-17 拍板):交易所官方 + CSI
# 规模/行业;主题/策略/债券等冷门类别由后续增补批次覆盖。
MAIN_INDEX_SEGMENTS: tuple[dict[str, str], ...] = (
    {"market": "SSE"},
    {"market": "SZSE"},
    {"market": "CSI", "category": "规模指数"},
    {"market": "CSI", "category": "行业指数"},
)


def quarter_end(year: int, quarter: int) -> str:
    return {1: f"{year}0331", 2: f"{year}0630", 3: f"{year}0930", 4: f"{year}1231"}[quarter]


def iter_months(start_yyyymmdd: str, end_yyyymmdd: str) -> list[tuple[str, str]]:
    """(start_date, end_date) pairs covering each month in the window."""
    import datetime as _dt

    y, m = int(start_yyyymmdd[:4]), int(start_yyyymmdd[4:6])
    y_end, m_end = int(end_yyyymmdd[:4]), int(end_yyyymmdd[4:6])
    out = []
    while (y, m) <= (y_end, m_end):
        month_start = _dt.date(y, m, 1)
        if m == 12:
            next_month = _dt.date(y + 1, 1, 1)
        else:
            next_month = _dt.date(y, m + 1, 1)
        month_end = next_month - _dt.timedelta(days=1)
        s = max(month_start.strftime("%Y%m%d"), start_yyyymmdd)
        e = min(month_end.strftime("%Y%m%d"), end_yyyymmdd)
        out.append((s, e))
        y, m = next_month.year, next_month.month
    return out


def iter_years(start_yyyymmdd: str, end_yyyymmdd: str) -> list[tuple[str, str]]:
    y0, y1 = int(start_yyyymmdd[:4]), int(end_yyyymmdd[:4])
    return [
        (max(f"{y}0101", start_yyyymmdd), min(f"{y}1231", end_yyyymmdd))
        for y in range(y0, y1 + 1)
    ]


def iter_quarters(start_yyyymmdd: str, end_yyyymmdd: str) -> list[str]:
    periods = []
    for y in range(int(start_yyyymmdd[:4]), int(end_yyyymmdd[:4]) + 1):
        for q in (1, 2, 3, 4):
            p = quarter_end(y, q)
            if start_yyyymmdd <= p <= end_yyyymmdd:
                periods.append(p)
    return periods


class BatchContext:
    """Shared expansion inputs for one batch run."""

    def __init__(
        self,
        *,
        universe: list[str],
        trade_dates: list[str],
        latest_trade_date: str,
        latest_report_period: str,
        index_codes: list[str] | None = None,
        index_codes_main: list[str] | None = None,
        context_sha256: str = "",
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        self.universe = universe
        self.trade_dates = trade_dates
        self.latest_trade_date = latest_trade_date
        self.latest_report_period = latest_report_period
        self.index_codes = index_codes or []
        self.index_codes_main = index_codes_main or []
        # 上下文构建来源(本地封存文件 SHA 或 API 参数)与整体 SHA,
        # 用于审计 context 出自哪些封存数据(2026-07-21 用户指令)。
        self.context_sha256 = context_sha256
        self.sources = sources or []


def _spec(ep: InventoryEndpoint, params: dict[str, Any]) -> EndpointSpec:
    base = ep.to_spec()
    base.params_template = params
    return base


def expand_endpoint(ep: InventoryEndpoint, ctx: BatchContext) -> list[EndpointSpec]:
    """Expand one endpoint into concrete tasks per its split strategy."""
    earliest = ep.earliest_date or "19901219"
    end = ctx.latest_trade_date
    params0 = dict(ep.params or {})

    if ep.api_name == "stock_basic":
        return [_spec(ep, {"list_status": s}) for s in ("L", "D", "P")]

    if ep.api_name == "trade_cal":
        return [_spec(ep, {**params0, "start_date": "19901219", "end_date": end})]

    if ep.split_unit == "snapshot":
        return [_spec(ep, params0)]

    if ep.split_unit == "index_basic_segments":
        # index_basic 全字段响应存在行数上限(实测 market=CSI 被截在 4879 行,
        # category 细分求和 ~8800)。CSI 只按 category 细分展开(每段均低于
        # 上限),其余 market 单段展开;空段(CICC/OTHERS)归档为 confirmed_empty。
        segments: list[dict[str, Any]] = [
            {"market": m} for m in INDEX_BASIC_MARKETS if m != "CSI"
        ]
        segments += [{"market": "CSI", "category": c} for c in CSI_INDEX_CATEGORIES]
        return [_spec(ep, {**params0, **seg}) for seg in segments]

    if ep.split_unit == "trade_date":
        dates = [d for d in ctx.trade_dates if earliest <= d <= end]
        return [_spec(ep, {**params0, "trade_date": d}) for d in dates]

    if ep.split_unit == "month":
        return [
            _spec(ep, {**params0, "start_date": s, "end_date": e})
            for s, e in iter_months(earliest, end)
        ]

    if ep.split_unit == "year":
        return [
            _spec(ep, {**params0, "start_date": s, "end_date": e})
            for s, e in iter_years(earliest, end)
        ]

    if ep.split_unit == "quarter":
        periods = iter_quarters(earliest, ctx.latest_report_period)
        return [_spec(ep, {**params0, "period": p}) for p in periods]

    if ep.split_unit == "symbol":
        specs: list[EndpointSpec] = []
        for code in ctx.universe:
            params = {**params0, "ts_code": code}
            if ep.primary_split == "period":
                # Financial endpoints: attach the full window so a row-cap hit
                # can be recovered by automatic date-range bisection.
                params.setdefault("start_date", f"{earliest[:4]}0101")
                params.setdefault("end_date", end)
            specs.append(_spec(ep, params))
        return specs

    if ep.split_unit == "symbol_chunk":
        chunk = 200
        return [
            _spec(ep, {**params0, "ts_code": ",".join(ctx.universe[i : i + chunk])})
            for i in range(0, len(ctx.universe), chunk)
        ]

    if ep.split_unit == "index_year":
        param_key = "index_code" if ep.api_name == "index_weight" else "ts_code"
        specs = []
        for code in ctx.index_codes:
            for s, e in iter_years(earliest, end):
                specs.append(_spec(ep, {**params0, param_key: code, "start_date": s, "end_date": e}))
        return specs

    if ep.split_unit == "index_year_main":
        # 分层归档的主力宇宙(SSE/SZSE 官方 + CSI 规模/行业,用户 2026-07-17
        # 拍板):主题/策略/债券等冷门类别进后续增补批次,不阻塞 B3/B4。
        param_key = "index_code" if ep.api_name == "index_weight" else "ts_code"
        specs = []
        for code in ctx.index_codes_main:
            for s, e in iter_years(earliest, end):
                specs.append(_spec(ep, {**params0, param_key: code, "start_date": s, "end_date": e}))
        return specs

    if ep.split_unit == "index_month":
        specs = []
        for code in ctx.index_codes:
            for s, e in iter_months(earliest, end):
                specs.append(_spec(ep, {**params0, "index_code": code, "start_date": s, "end_date": e}))
        return specs

    raise ValueError(f"未知 split_unit: {ep.api_name} {ep.split_unit}")


def expand_batch(
    inventory: EndpointInventory,
    batch_id: str,
    ctx: BatchContext,
) -> list[EndpointSpec]:
    specs: list[EndpointSpec] = []
    for ep in inventory.list_by_batch(batch_id):
        if not ep.enabled:
            continue
        ep_specs = expand_endpoint(ep, ctx)
        logger.info("展开 %s -> %d 个任务", ep.api_name, len(ep_specs))
        specs.extend(ep_specs)
    return specs


def batch_decision_gates(
    by_status: dict[str, int], total: int
) -> dict[str, bool]:
    """Return fail-closed research gates for one sealed batch snapshot."""
    success = by_status.get("success", 0)
    confirmed_empty = by_status.get("confirmed_empty", 0)
    # 父任务触发行数上限被拆分且子任务全部终态解决,数据由子任务承载(已解决终态)
    bisected = by_status.get("bisected", 0)
    # 已被替代任务集承载数据的任务,同属已解决终态:
    # 拆分方式无效 / 恰满真实 cap 被静默截断 / 撞名事件僵尸 running
    superseded = by_status.get("superseded_invalid_partition", 0)
    superseded += by_status.get("superseded_truncated_cap", 0)
    superseded += by_status.get("superseded_legacy_collision", 0)
    return {
        "all_tasks_terminal": (
            success + confirmed_empty + bisected + superseded + by_status.get("quarantined", 0)
        )
        == total,
        "no_suspect_truncated": by_status.get("suspect_truncated", 0) == 0,
        "no_retryable_left": by_status.get("retryable_error", 0) == 0,
        # Quarantine is terminal for the downloader state machine, but never a
        # valid terminal state for a research-ready batch.  Without this gate a
        # single missing required partition could hide inside the 99.5% rate.
        "no_quarantined": by_status.get("quarantined", 0) == 0,
        "no_denied": by_status.get("denied", 0) == 0,
        "no_invalid_params": by_status.get("invalid_params", 0) == 0,
        "success_rate_ge_99.5%": (
            (success + bisected) / max(1, total - confirmed_empty)
        )
        >= 0.995,
    }


def write_batch_artifacts(
    config: ArchiveConfig,
    db: TaskStateDB,
    batch_id: str,
    snapshot_id: str,
    specs: list[EndpointSpec],
    result: Any,
    started_at: str,
    elapsed: float,
    *,
    context_sha256: str = "",
) -> dict[str, Any]:
    """Write the six per-batch artifacts and return the decision dict."""
    reports_dir = config.reports_dir / BATCH_ARTIFACT_DIR / batch_id
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Task ids belonging to this batch snapshot (shared state db).
    from .pipeline import task_id

    batch_task_ids = {
        task_id(config.provider.name, s.api_name, dict(s.params_template), s.fields, snapshot_id)
        for s in specs
    }
    batch_tasks = [t for t in db.list_tasks() if t.task_id in batch_task_ids]

    # 1. batch_manifest.jsonl
    manifest = build_manifest(config, db, snapshot_id, tasks=batch_tasks)
    # 记录本批次封存 context 的 SHA,供断点续跑 fail-closed 校验(2026-07-22)。
    manifest["context_sha256"] = context_sha256
    manifest_path = reports_dir / "batch_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False) + "\n")

    # 2. checksums.sha256 for this batch snapshot's raw+bronze files.
    checksum_lines: list[str] = []
    for root in (config.raw_dir / snapshot_id, config.bronze_dir):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and snapshot_id in path.name:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                checksum_lines.append(f"{digest}  {path.relative_to(config.archive_root)}\n")
    (reports_dir / "checksums.sha256").write_text("".join(checksum_lines), encoding="utf-8")

    # 3. schema_report.json (fingerprints registered under this snapshot).
    schema_root = config.catalog_dir / "schema_registry"
    schema_entries = []
    if schema_root.exists():
        for ep_dir in sorted(schema_root.iterdir()):
            if not ep_dir.is_dir():
                continue
            for fp in sorted(ep_dir.glob("*.json")):
                payload = json.loads(fp.read_text(encoding="utf-8"))
                if payload.get("snapshot_id") == snapshot_id:
                    schema_entries.append(
                        {
                            "endpoint": ep_dir.name,
                            "fingerprint": payload.get("fingerprint"),
                            "columns": payload.get("columns"),
                            "row_count": payload.get("row_count"),
                        }
                    )
    (reports_dir / "schema_report.json").write_text(
        json.dumps({"batch": batch_id, "snapshot_id": snapshot_id, "schemas": schema_entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 4. failure_queue.jsonl — only tasks belonging to this batch snapshot.
    failures = [
        t
        for status in (TaskStatus.RETRYABLE_ERROR, TaskStatus.SUSPECT_TRUNCATED, TaskStatus.DENIED, TaskStatus.INVALID_PARAMS, TaskStatus.QUARANTINED)
        for t in db.list_tasks(status=status)
        if t.task_id in batch_task_ids
    ]
    failure_rows = [
        {
            "task_id": t.task_id,
            "api_name": t.api_name,
            "params": t.params,
            "status": t.status.value,
            "attempts": t.attempts,
            "last_error": t.last_error,
        }
        for t in failures
    ]
    with (reports_dir / "failure_queue.jsonl").open("w", encoding="utf-8") as f:
        for row in failure_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 5. coverage_report.md
    by_status = manifest["by_status"]
    coverage_md = [
        f"# 批次 {batch_id} 覆盖报告",
        "",
        f"- snapshot: {snapshot_id}",
        f"- 开始(UTC): {started_at}  耗时: {elapsed:.0f}s",
        f"- 任务总数: {manifest['total_tasks']}",
        f"- 状态分布: {json.dumps(by_status, ensure_ascii=False)}",
        f"- 总行数: {result.rows_total:,}",
        "",
        "| endpoint | 任务数 |",
        "|---|---|",
    ]
    per_api: dict[str, int] = {}
    for s in specs:
        per_api[s.api_name] = per_api.get(s.api_name, 0) + 1
    coverage_md += [f"| {api} | {n} |" for api, n in sorted(per_api.items())]
    (reports_dir / "coverage_report.md").write_text("\n".join(coverage_md) + "\n", encoding="utf-8")

    # 6. batch_decision.json
    total = manifest["total_tasks"]
    unresolved = by_status.get("suspect_truncated", 0) + by_status.get("retryable_error", 0)
    gates = batch_decision_gates(by_status, total)
    decision = {
        "batch": batch_id,
        "snapshot_id": snapshot_id,
        "decision": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "unresolved_failure_count": unresolved,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (reports_dir / "batch_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return decision


def run_batch(
    config: ArchiveConfig,
    provider: ArchiveProvider,
    inventory: EndpointInventory,
    db: TaskStateDB,
    batch_id: str,
    ctx: Any | None,
    *,
    snapshot_id: str | None = None,
    resume_specs: list[EndpointSpec] | None = None,
    context_sha256: str = "",
) -> dict[str, Any]:
    """Execute one batch end-to-end and write its artifact set.

    resume_specs 提供时(断点续跑)跳过 expand_batch,直接使用从封存
    manifest+db 行精确回放的任务集;此时 ctx 可为 None(symbol_universe
    仅用于整市场任务的符号二分,回放任务均为显式单码,传空安全)。
    """
    snapshot_id = snapshot_id or time.strftime(f"p0_{batch_id}_%Y%m%d_%H%M%S", time.gmtime())
    specs = resume_specs if resume_specs is not None else expand_batch(inventory, batch_id, ctx)
    logger.info("批次 %s: %d 个任务, snapshot=%s", batch_id, len(specs), snapshot_id)

    pipeline = ArchivePipeline(
        config,
        provider,
        db,
        snapshot_id=snapshot_id,
        symbol_universe=ctx.universe if ctx is not None else [],
    )
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.perf_counter()
    result = pipeline.run_tasks(specs, skip_existing=True)
    elapsed = time.perf_counter() - started

    decision = write_batch_artifacts(
        config, db, batch_id, snapshot_id, specs, result, started_at, elapsed,
        context_sha256=context_sha256,
    )
    logger.info("批次 %s decision=%s", batch_id, decision["decision"])
    return decision


def save_frozen_specs(
    path: Path,
    *,
    batch_id: str,
    snapshot_id: str,
    provider_name: str,
    context_record: dict[str, Any],
    specs: list[EndpointSpec],
) -> dict[str, Any]:
    """物化并封存批次任务清单(2026-07-22 用户指令)。

    启动前把 expand 结果连同 context 记录整体冻结;启动/恢复一律从本
    文件回放,禁止按最新交易日重建(见 ORPHANED_CONTEXT_DRIFT 事件)。
    manifest_sha256 覆盖排序后的全部 task_id,加载时逐条重算校验。
    """
    from .pipeline import task_id

    entries = [
        {"api_name": s.api_name, "params": dict(s.params_template), "fields": s.fields}
        for s in specs
    ]
    ids = sorted(
        task_id(provider_name, e["api_name"], e["params"], e["fields"], snapshot_id)
        for e in entries
    )
    manifest_sha = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    payload = {
        "schema_version": 1,
        "batch": batch_id,
        "snapshot_id": snapshot_id,
        "provider": provider_name,
        "context": context_record,
        "context_sha256": context_record.get("context_sha256", ""),
        "task_count": len(entries),
        "manifest_sha256": manifest_sha,
        "specs": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("冻结任务清单: %s (%d 任务, manifest sha=%s)", path, len(entries), manifest_sha[:16])
    return payload


def load_frozen_specs(path: Path, config: ArchiveConfig, inventory: EndpointInventory) -> tuple[str, list[EndpointSpec]]:
    """从冻结任务清单回放(fail-closed): 校验 snapshot/provider/manifest SHA。"""
    from .pipeline import task_id

    if not path.exists():
        raise FileNotFoundError(f"冻结任务清单不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    snapshot_id = payload["snapshot_id"]
    if payload.get("provider") != config.provider.name:
        raise ValueError(
            f"冻结清单 provider 不符: {payload.get('provider')} != {config.provider.name}"
        )
    entries = payload.get("specs", [])
    ids = sorted(
        task_id(config.provider.name, e["api_name"], e["params"], e["fields"], snapshot_id)
        for e in entries
    )
    manifest_sha = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if manifest_sha != payload.get("manifest_sha256"):
        raise ValueError(
            f"冻结清单 manifest SHA 不符(疑似篡改): 文件={str(payload.get('manifest_sha256'))[:12]} 重算={manifest_sha[:12]}"
        )
    specs: list[EndpointSpec] = []
    for e in entries:
        ep = inventory.endpoints[e["api_name"]]
        spec = ep.to_spec()
        spec.params_template = dict(e["params"])
        if spec.fields != e["fields"]:
            raise ValueError(f"冻结清单 fields 与注册表漂移: {e['api_name']}")
        specs.append(spec)
    logger.info("冻结清单回放: %d 任务, snapshot=%s", len(specs), snapshot_id)
    return snapshot_id, specs


def load_resume_specs(
    config: ArchiveConfig,
    db: TaskStateDB,
    inventory: EndpointInventory,
    batch_id: str,
    snapshot_id: str,
) -> list[EndpointSpec]:
    """断点续跑的 fail-closed 任务回放(2026-07-22 用户指令)。

    必须加载已物化的 manifest 与封存的 context 并校验 SHA;
    任务参数逐行取自数据库(manifest task_id 精确匹配),禁止调用
    build_context、禁止按最新交易日重建——context 漂移会产生不属于
    原 manifest 的孤儿任务(参见 ORPHANED_CONTEXT_DRIFT 事件)。
    任何校验失败直接抛异常(fail-closed),不降级。
    """
    reports_dir = config.reports_dir / BATCH_ARTIFACT_DIR / batch_id
    manifest_path = reports_dir / "batch_manifest.jsonl"
    if not manifest_path.exists():
        # 已物化 manifest 缺失时,回退到启动前冻结的任务清单(同样 fail-closed,
        # 内部校验 provider/manifest SHA);两者皆无则拒绝运行。
        frozen_path = reports_dir / f"frozen_specs_{snapshot_id}.json"
        if frozen_path.exists():
            frozen_snapshot, specs = load_frozen_specs(frozen_path, config, inventory)
            if frozen_snapshot != snapshot_id:
                raise ValueError(
                    f"断点续跑失败: 冻结清单 snapshot {frozen_snapshot} 与请求 {snapshot_id} 不符"
                )
            return specs
        raise FileNotFoundError(
            f"断点续跑失败: 已物化 manifest 不存在 {manifest_path} 且无冻结清单;"
            f"禁止根据最新交易日重建,请检查批次 {batch_id} 是否曾正常收官"
        )
    manifest_lines = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matching = [m for m in manifest_lines if m.get("snapshot_id") == snapshot_id]
    if not matching:
        raise ValueError(
            f"断点续跑失败: manifest 中没有 snapshot {snapshot_id} 的记录"
        )
    manifest = matching[0]  # 首条为原始物化 manifest
    manifest_ids = [t["task_id"] for t in manifest["tasks"]]

    # 封存 context 校验: manifest 记录的 SHA 优先;老 manifest 无该字段时,
    # 仅当 context 文件唯一才可信,否则 fail-closed(可能存在漂移 context)。
    expected_sha = manifest.get("context_sha256") or ""
    context_files = sorted(reports_dir.glob("context_*.json"))
    if expected_sha:
        context_path = reports_dir / f"context_{expected_sha[:12]}.json"
        if not context_path.exists():
            raise FileNotFoundError(
                f"断点续跑失败: 封存 context 文件缺失 {context_path}"
            )
        actual = json.loads(context_path.read_text(encoding="utf-8")).get("context_sha256", "")
        if actual != expected_sha:
            raise ValueError(
                f"断点续跑失败: context SHA 不符 manifest={expected_sha[:12]} 文件={actual[:12]}"
            )
    elif len(context_files) == 1:
        expected_sha = json.loads(context_files[0].read_text(encoding="utf-8")).get("context_sha256", "")
    else:
        raise ValueError(
            f"断点续跑失败: manifest 未记录 context_sha256 且存在 "
            f"{len(context_files)} 个 context 文件,无法确定封存版本"
        )

    rows = {t.task_id: t for t in db.list_tasks()}
    missing = [tid for tid in manifest_ids if tid not in rows]
    if missing:
        raise ValueError(
            f"断点续跑失败: {len(missing)} 个 manifest 任务在数据库中无记录(首: {missing[0][:12]})"
        )
    from .pipeline import task_id

    specs: list[EndpointSpec] = []
    for tid in manifest_ids:
        row = rows[tid]
        ep = inventory.endpoints[row.api_name]
        spec = ep.to_spec()
        if spec.fields != row.fields:
            raise ValueError(
                f"断点续跑失败: 任务 {tid[:12]} fields 漂移 registry={spec.fields!r} db={row.fields!r}"
            )
        recomputed = task_id(config.provider.name, row.api_name, dict(row.params), row.fields, snapshot_id)
        if recomputed != tid:
            raise ValueError(
                f"断点续跑失败: 任务 {tid[:12]} 参数与 task_id 不一致(疑似参数被篡改)"
            )
        spec.params_template = dict(row.params)
        specs.append(spec)
    logger.info(
        "断点续跑: 回放 manifest %d 任务, 封存 context=%s…", len(specs), expected_sha[:12]
    )
    return specs
