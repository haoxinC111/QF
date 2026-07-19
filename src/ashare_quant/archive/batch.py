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
    ) -> None:
        self.universe = universe
        self.trade_dates = trade_dates
        self.latest_trade_date = latest_trade_date
        self.latest_report_period = latest_report_period
        self.index_codes = index_codes or []
        self.index_codes_main = index_codes_main or []


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
    return {
        "all_tasks_terminal": (
            success + confirmed_empty + by_status.get("quarantined", 0)
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
            success / max(1, total - confirmed_empty)
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
    ctx: BatchContext,
    *,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Execute one batch end-to-end and write its artifact set."""
    snapshot_id = snapshot_id or time.strftime(f"p0_{batch_id}_%Y%m%d_%H%M%S", time.gmtime())
    specs = expand_batch(inventory, batch_id, ctx)
    logger.info("批次 %s: %d 个任务, snapshot=%s", batch_id, len(specs), snapshot_id)

    pipeline = ArchivePipeline(
        config,
        provider,
        db,
        snapshot_id=snapshot_id,
        symbol_universe=ctx.universe,
    )
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.perf_counter()
    result = pipeline.run_tasks(specs, skip_existing=True)
    elapsed = time.perf_counter() - started

    decision = write_batch_artifacts(
        config, db, batch_id, snapshot_id, specs, result, started_at, elapsed
    )
    logger.info("批次 %s decision=%s", batch_id, decision["decision"])
    return decision
