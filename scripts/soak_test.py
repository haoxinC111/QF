#!/usr/bin/env python
"""1000-request soak test for the archive pipeline.

Runs >= 1000 real requests across multiple endpoints and partitions through
the production ArchivePipeline (single worker, one global 75/min token
bucket), then reports stability metrics and gates the P0 full archive.

Reads QF_ARCHIVE_API_URL / QF_ARCHIVE_API_TOKEN from the environment and
the local config.archive.yaml (authorization already confirmed).

Outputs (under data_lake/reports/):
    soak_test_report.json
    soak_test_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_var, None)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ashare_quant.archive.config import ArchiveConfig  # noqa: E402
from ashare_quant.archive.pipeline import ArchivePipeline, EndpointSpec, task_id  # noqa: E402
from ashare_quant.archive.provider import TushareCompatibleHttpProvider  # noqa: E402
from ashare_quant.archive.state import TaskStateDB, TaskStatus  # noqa: E402
from ashare_quant.archive.storage import load_bronze_parquet, load_raw_json_zst, sha256_file  # noqa: E402

REPORTS = REPO_ROOT / "data_lake" / "reports"
SNAPSHOT = time.strftime("soak_test_%Y%m%d_%H%M%S", time.gmtime())


class InstrumentedProvider(TushareCompatibleHttpProvider):
    """Records every underlying HTTP attempt (one record per try)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attempts: list[dict] = []

    def request(self, api_name, params=None, *, fields=""):  # noqa: D102
        resp = super().request(api_name, params, fields=fields)
        self.attempts.append(
            {
                "api_name": api_name,
                "http_status": resp.http_status,
                "status": resp.status,
                "elapsed_seconds": round(resp.elapsed_seconds, 4),
            }
        )
        return resp


def build_plan(provider, trade_dates: list[str], target: int) -> list[EndpointSpec]:
    """Multi-endpoint, multi-partition plan of roughly ``target`` tasks."""
    specs: list[EndpointSpec] = []

    def spec(api, dataset, params, pk, split, fallback):
        specs.append(
            EndpointSpec(
                api_name=api,
                dataset=dataset,
                priority="P0",
                primary_key=pk,
                primary_split=split,
                fallback_split=fallback,
                all_fields=True,
                fields="",
                params_template=params,
            )
        )

    n_daily = min(170, len(trade_dates))
    daily_dates = trade_dates[:n_daily]
    basic_dates = trade_dates[n_daily : n_daily * 2]
    limit_dates = trade_dates[n_daily * 2 : n_daily * 3]
    moneyflow_dates = trade_dates[n_daily * 3 : n_daily * 4]
    suspend_dates = trade_dates[n_daily * 4 : n_daily * 4 + 110]

    for d in daily_dates:
        spec("daily", "market_daily", {"trade_date": d}, ["ts_code", "trade_date"], "trade_date", "ts_code")
    for d in basic_dates:
        spec("daily_basic", "market_daily_basic", {"trade_date": d}, ["ts_code", "trade_date"], "trade_date", "ts_code")
    for d in limit_dates:
        spec("stk_limit", "market_stk_limit", {"trade_date": d}, ["ts_code", "trade_date"], "trade_date", "ts_code")
    for d in moneyflow_dates:
        spec("moneyflow", "market_moneyflow", {"trade_date": d}, ["ts_code", "trade_date"], "trade_date", "ts_code")
    for d in suspend_dates:
        spec("suspend_d", "market_suspend", {"trade_date": d}, ["ts_code", "trade_date"], "trade_date", "ts_code")

    # Per-symbol partition: adj_factor for a slice of the universe.
    resp = provider.request("stock_basic", {"list_status": "L"}, fields="ts_code")
    codes = sorted(r[0] for r in resp.items)
    step = max(1, len(codes) // 200)
    for code in codes[::step][:200]:
        spec("adj_factor", "market_adj_factor", {"ts_code": code}, ["ts_code", "trade_date"], "ts_code", None)

    # Index daily by month partitions.
    for month in range(1, 13):
        spec(
            "index_daily",
            "index_daily",
            {"ts_code": "000300.SH", "start_date": f"2025{month:02d}01", "end_date": f"2025{month:02d}28"},
            ["ts_code", "trade_date"],
            "start_date",
            None,
        )

    if len(specs) < target:
        extra = trade_dates[n_daily * 4 + 100 :]
        for d in extra[: target - len(specs)]:
            spec("daily", "market_daily", {"trade_date": d}, ["ts_code", "trade_date"], "trade_date", "ts_code")
    return specs[: max(target, 1000)]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1020)
    args = parser.parse_args()

    config = ArchiveConfig.from_yaml(REPO_ROOT / "config.archive.yaml")
    config.validate_for_run(batch_id="soak_test")
    config.ensure_dirs()

    provider = InstrumentedProvider()
    db = TaskStateDB(config.catalog_dir / "tasks.db")

    # Trade dates: 2023-2025 open days from the SSE calendar endpoint.
    # Trade dates: 2022-2025 open days from the SSE calendar endpoint.
    # 2023-2025 alone is only ~727 days, short of the 680+110 daily-date
    # slices the plan needs for >=1000 tasks; 2022-2025 gives ~970.
    cal = provider.request("trade_cal", {"exchange": "SSE", "start_date": "20220101", "end_date": "20251231", "is_open": "1"})
    trade_dates = sorted(r[cal.columns.index("cal_date")] for r in cal.items)
    print(f"2022-2025 交易日: {len(trade_dates)} 天", flush=True)

    specs = build_plan(provider, trade_dates, args.target)
    print(f"soak 计划任务数: {len(specs)}, snapshot={SNAPSHOT}", flush=True)

    pipeline = ArchivePipeline(config, provider, db, snapshot_id=SNAPSHOT)
    started = time.perf_counter()
    result = pipeline.run_tasks(specs, skip_existing=False)
    elapsed = time.perf_counter() - started

    # ---- metrics ----------------------------------------------------------
    attempts = provider.attempts
    total_requests = len(attempts)
    status_429 = sum(1 for a in attempts if a["http_status"] == 429)
    status_5xx = sum(1 for a in attempts if a["http_status"] and a["http_status"] >= 500)
    non_retryable = sum(
        1 for a in attempts if a["status"] in ("denied", "invalid_params", "not_found", "incompatible")
    )
    ok_lat = [a["elapsed_seconds"] for a in attempts if a["status"] == "success"]

    tasks = db.list_tasks()
    # Precise snapshot filter: task ids embed provider+params+snapshot.
    expected_ids = {
        task_id(config.provider.name, s.api_name, dict(s.params_template), s.fields, SNAPSHOT)
        for s in specs
    }
    soak_tasks = [t for t in tasks if t.task_id in expected_ids]
    success_tasks = [t for t in soak_tasks if t.status == TaskStatus.SUCCESS]
    terminal_ok = {TaskStatus.SUCCESS, TaskStatus.CONFIRMED_EMPTY}
    unfinished = [t for t in soak_tasks if t.status not in terminal_ok]
    suspect = [t for t in soak_tasks if t.status == TaskStatus.SUSPECT_TRUNCATED]

    retry_hist: dict[int, int] = {}
    for t in soak_tasks:
        retry_hist[t.attempts] = retry_hist.get(t.attempts, 0) + 1

    # Raw/Bronze row consistency + SHA256 re-verify for every stored partition.
    row_mismatch = 0
    sha_mismatch = 0
    checked = 0
    for t in success_tasks:
        raw_path = Path(t.raw_path)
        bronze_path = Path(t.bronze_path)
        try:
            # Same unwrapping as the provider: items live inside the "data"
            # envelope for this gateway (flat payloads also accepted).
            payload = json.loads(load_raw_json_zst(raw_path).decode("utf-8"))
            body = payload.get("data") if isinstance(payload.get("data"), dict) else payload
            raw_items = len(body["items"])
            bronze_rows = len(load_bronze_parquet(bronze_path))
            if raw_items != bronze_rows or bronze_rows != t.row_count:
                row_mismatch += 1
            if sha256_file(raw_path) != t.raw_sha256:
                sha_mismatch += 1
            checked += 1
        except Exception as exc:  # noqa: BLE001
            row_mismatch += 1
            print(f"  校验失败 {raw_path}: {exc}", flush=True)

    # Orphan tmp files across the lake.
    orphans = [str(p) for p in config.archive_root.rglob("*.tmp")]

    # Token leak scan over reports/catalog/manifests (values, never names).
    token = os.environ.get("QF_ARCHIVE_API_TOKEN", "").strip()
    official = os.environ.get("TUSHARE_TOKEN", "").strip()
    leaks: list[str] = []
    scan_roots = [config.catalog_dir, config.reports_dir]
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix in (".db", ".parquet"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if token and token in text:
                leaks.append(f"{path}: QF token")
            if official and official in text:
                leaks.append(f"{path}: TUSHARE_TOKEN")

    first_attempt_success = sum(1 for t in success_tasks if t.attempts == 1)
    post_retry_rate = len(success_tasks) / max(1, len(soak_tasks))

    gates = {
        "post_retry_success_ge_99.5%": post_retry_rate >= 0.995,
        "unclassified_errors_zero": non_retryable == 0,
        "token_leaks_zero": len(leaks) == 0,
        "orphan_tmp_zero": len(orphans) == 0,
        "unresolved_truncation_zero": len(suspect) == 0,
        "sha256_all_pass": sha_mismatch == 0,
        "raw_bronze_rows_consistent": row_mismatch == 0,
    }
    soak_pass = all(gates.values())

    report = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snapshot_id": SNAPSHOT,
        "rate_limit": {"calls_per_minute": config.rate_limit.calls_per_minute, "workers": 1},
        "planned_tasks": len(specs),
        "total_http_requests": total_requests,
        "elapsed_seconds": round(elapsed, 1),
        "effective_rpm": round(total_requests / (elapsed / 60), 2) if elapsed else None,
        "tasks": {
            "success": len(success_tasks),
            "confirmed_empty": sum(1 for t in soak_tasks if t.status == TaskStatus.CONFIRMED_EMPTY),
            "first_attempt_success": first_attempt_success,
            "first_attempt_success_rate": round(first_attempt_success / max(1, len(soak_tasks)), 6),
            "post_retry_success_rate": round(post_retry_rate, 6),
            "unfinished": len(unfinished),
            "unfinished_ids": [t.task_id[:16] for t in unfinished][:20],
            "suspect_truncated": len(suspect),
            "retry_histogram": {str(k): v for k, v in sorted(retry_hist.items())},
        },
        "http": {
            "429_count": status_429,
            "5xx_count": status_5xx,
            "non_retryable_errors": non_retryable,
            "latency_seconds": {
                "p50": percentile(ok_lat, 0.50),
                "p95": percentile(ok_lat, 0.95),
                "p99": percentile(ok_lat, 0.99),
            },
        },
        "integrity": {
            "partitions_checked": checked,
            "raw_bronze_row_mismatch": row_mismatch,
            "sha256_mismatch": sha_mismatch,
            "orphan_tmp_files": orphans[:20],
            "token_leaks": leaks[:20],
        },
        "rows_total": result.rows_total,
        "gates": gates,
        "pass": soak_pass,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS / "soak_test_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Soak 稳定性测试报告",
        "",
        f"- snapshot: {SNAPSHOT}",
        f"- 计划任务: {len(specs)} / 实际 HTTP 请求: {total_requests}",
        f"- 耗时: {report['elapsed_seconds']}s (有效速率 {report['effective_rpm']} req/min, 单 worker, 75/min 令牌桶)",
        f"- 首次成功率: {report['tasks']['first_attempt_success_rate']:.4%}",
        f"- 重试后成功率: {report['tasks']['post_retry_success_rate']:.4%}",
        f"- 429: {status_429} / 5xx: {status_5xx} / 不可重试错误: {non_retryable}",
        f"- 延迟 p50/p95/p99: {report['http']['latency_seconds']}",
        f"- 重试分布: {report['tasks']['retry_histogram']}",
        f"- Raw-Bronze 行数不一致: {row_mismatch} / SHA256 不一致: {sha_mismatch} (校验 {checked} 分区)",
        f"- 未完成任务: {len(unfinished)} / suspect_truncated: {len(suspect)}",
        f"- orphan 临时文件: {len(orphans)} / Token 泄漏: {len(leaks)}",
        "",
        "## 门禁",
        "",
    ]
    md += [f"- {'✅' if ok else '❌'} {name}" for name, ok in gates.items()]
    md += ["", f"## 结论: {'**PASS**' if soak_pass else '**FAIL**'}"]
    (REPORTS / "soak_test_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({"pass": soak_pass, "requests": total_requests, "post_retry_rate": report["tasks"]["post_retry_success_rate"]}))
    print(f"报告: {json_path}")
    return 0 if soak_pass else 1


if __name__ == "__main__":
    sys.exit(main())
