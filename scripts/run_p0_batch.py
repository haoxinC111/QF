#!/usr/bin/env python
"""Drive one P0 full-archive batch (B0..B4) end-to-end.

Usage:
    python scripts/run_p0_batch.py --batch B0_reference

Context comes from the archived Phase-A bronze (stock_basic universe) and a
live trade_cal pull.  B3_financial refuses to start unless the financial
cross-source validation report says pass (per the Phase A.1 gate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

for _var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_var, None)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ashare_quant.archive.batch import (  # noqa: E402
    CSI_INDEX_CATEGORIES,
    INDEX_BASIC_MARKETS,
    INDEX_BASIC_ROW_ALERT,
    MAIN_INDEX_SEGMENTS,
    BatchContext,
    run_batch,
)
from ashare_quant.archive.config import ArchiveConfig  # noqa: E402
from ashare_quant.archive.provider import TushareCompatibleHttpProvider  # noqa: E402
from ashare_quant.archive.registry import P0_BATCH_ORDER, default_inventory  # noqa: E402
from ashare_quant.archive.state import TaskStateDB  # noqa: E402

BRONZE = REPO_ROOT / "data_lake" / "bronze" / "research_proxy_unverified"
REPORTS = REPO_ROOT / "data_lake" / "reports"


def latest_report_period(today: str) -> str:
    """Latest quarter whose statutory disclosure window has closed."""
    y, md = int(today[:4]), today[4:]
    # Q1->0430, Q2->0831, Q3->1031, Q4->next 0430.
    if md <= "0430":
        return f"{y - 1}0930"
    if md <= "0831":
        return f"{y}0331"
    if md <= "1031":
        return f"{y}0630"
    return f"{y}0930"


def _is_transient(resp) -> bool:
    """429/限流/网络类错误可重试;权限/参数类错误立即上交。"""
    if resp.http_status == 429:
        return True
    msg = resp.message or ""
    return any(k in msg for k in ("429", "频率受限", "超限", "网络错误", "超时", "服务端错误"))


def _request_with_retry(provider: TushareCompatibleHttpProvider, api: str, params: dict):
    """启动期直连拉取的重试包装(2026-07-21 教训: 启动撞 429 曾直接致死)。

    - 优先遵守网关 Retry-After 头;
    - 否则指数退避(5s 起,×2,封顶 300s)+ 0~50% jitter;
    - 429 与网络/超时/5xx 均可重试,最多 8 次。
    """
    log = logging.getLogger(__name__)
    resp = None
    for attempt in range(1, 9):
        resp = provider.request(api, params)
        if resp.is_success or resp.status == "empty":
            return resp
        if not _is_transient(resp):
            return resp  # 非瞬时错误交由调用方判定
        if resp.retry_after_seconds is not None:
            wait = resp.retry_after_seconds + random.uniform(0, 1)
        else:
            wait = min(300.0, 5.0 * (2 ** (attempt - 1))) + random.uniform(0, 0.5 * min(300.0, 5.0 * (2 ** (attempt - 1))))
        log.warning("%s %s 瞬时错误(%s,attempt %d/8),%.1fs 后重试", api, params, resp.message, attempt, wait)
        time.sleep(wait)
    return resp


def _sealed_bronze_file(api_dir: Path, prefix: str) -> Path | None:
    """在已封存 bronze 目录中按文件名前缀找最新一个 parquet。"""
    matches = sorted(api_dir.glob(f"{prefix}*.parquet"))
    return matches[-1] if matches else None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _segment_prefix(seg: dict) -> str:
    """index_basic 分段参数 -> 封存分区文件名前缀(与 B2 分区命名一致)。"""
    return "index_basic_" + "_".join(f"{k}={v}" for k, v in sorted(seg.items()))


def build_context(provider: TushareCompatibleHttpProvider) -> BatchContext:
    """构建批次上下文: 优先读已封存 B0/B2 bronze 并记录来源 SHA;
    仅本地缺失/过期时才走 API(带 Retry-After/指数退避重试)。"""
    log = logging.getLogger(__name__)
    sources: list[dict] = []
    today = time.strftime("%Y%m%d", time.gmtime())

    listed_path = BRONZE / "stock_basic" / "stock_basic_list_status=L_phase_a_20260716_135101.parquet"
    delisted_path = BRONZE / "stock_basic" / "stock_basic_list_status=D_phase_a_20260716_135101.parquet"
    listed = pd.read_parquet(listed_path)
    delisted = pd.read_parquet(delisted_path)
    universe = sorted(set(listed["ts_code"]) | set(delisted["ts_code"]))
    for p in (listed_path, delisted_path):
        sources.append({"kind": "sealed_file", "api": "stock_basic", "path": str(p), "sha256": _sha256_file(p)})

    # trade_cal: 本地封存日历覆盖到今日及以后才可信,否则 API 兜底。
    trade_dates = None
    cal_path = _sealed_bronze_file(BRONZE / "trade_cal", "trade_cal_exchange=SSE_is_open=1_")
    if cal_path is not None:
        cal = pd.read_parquet(cal_path)
        if str(cal["cal_date"].max()) >= today:
            trade_dates = sorted(str(d) for d in cal["cal_date"])
            sources.append({"kind": "sealed_file", "api": "trade_cal", "path": str(cal_path), "sha256": _sha256_file(cal_path)})
        else:
            log.warning("本地 trade_cal 仅覆盖至 %s < %s,回退 API", cal["cal_date"].max(), today)
    if trade_dates is None:
        params = {"exchange": "SSE", "start_date": "19901219", "end_date": "20261231", "is_open": "1"}
        resp = _request_with_retry(provider, "trade_cal", params)
        if not resp.is_success:
            raise RuntimeError(f"trade_cal 拉取失败: {resp.message}")
        trade_dates = sorted(r[resp.columns.index("cal_date")] for r in resp.items)
        sources.append({"kind": "api", "api": "trade_cal", "params": params})
    completed = [d for d in trade_dates if d < today]
    latest_complete = completed[-1]

    def _index_codes_for(segments: list[dict], label: str) -> list[str]:
        codes: list[str] = []
        seen: set[str] = set()
        for seg in segments:
            path = _sealed_bronze_file(BRONZE / "index_basic", _segment_prefix(seg))
            rows: list[str] | None = None
            if path is not None:
                df = pd.read_parquet(path, columns=["ts_code"])
                if len(df) >= INDEX_BASIC_ROW_ALERT:
                    raise RuntimeError(
                        f"封存 index_basic {seg} 分区 {len(df)} 行达到告警阈值,疑似截断,拒绝使用"
                    )
                rows = [str(c) for c in df["ts_code"]]
                sources.append({"kind": "sealed_file", "api": "index_basic", "params": seg, "path": str(path), "sha256": _sha256_file(path)})
            else:
                resp = _request_with_retry(provider, "index_basic", seg)
                if not resp.is_success and resp.status != "empty":
                    raise RuntimeError(f"index_basic {seg} 拉取失败: {resp.message}")
                if resp.status == "empty":
                    sources.append({"kind": "api", "api": "index_basic", "params": seg, "result": "legal_empty"})
                    continue  # 合法空段(如 CICC/OTHERS 当前无指数)
                if len(resp.items) >= INDEX_BASIC_ROW_ALERT:
                    raise RuntimeError(
                        f"index_basic {seg} 返回 {len(resp.items)} 行达到告警阈值,疑似截断,需进一步细分"
                    )
                rows = [str(r[resp.columns.index("ts_code")]) for r in resp.items]
                sources.append({"kind": "api", "api": "index_basic", "params": seg, "result": f"{len(rows)} rows"})
                time.sleep(1)
            for code in rows:
                if code not in seen:
                    seen.add(code)
                    codes.append(code)
        log.info("%s: %d 只", label, len(codes))
        return codes

    # index_basic 全字段响应存在行数上限(实测 market=CSI 被截在 4879 行,
    # 而 category 细分求和 ~8800)。CSI 无条件按 category 细分;其他 market
    # 段达到保守告警阈值时显式报错,绝不静默接受疑似截断的宇宙。
    idx_segments: list[dict] = []
    for market in INDEX_BASIC_MARKETS:
        if market == "CSI":
            idx_segments.extend({"market": "CSI", "category": c} for c in CSI_INDEX_CATEGORIES)
        else:
            idx_segments.append({"market": market})
    idx_codes = _index_codes_for(idx_segments, "指数全量宇宙(分段合并)")

    # 主力宇宙:B2 index_daily/index_weight 展开范围(分层归档,主力先行)。
    main_codes = _index_codes_for([dict(s) for s in MAIN_INDEX_SEGMENTS], "主力指数宇宙")

    context_sha = hashlib.sha256(
        json.dumps(
            {
                "sources": sources,
                "universe": universe,
                "latest_trade_date": latest_complete,
                "latest_report_period": latest_report_period(today),
                "index_codes": idx_codes,
                "index_codes_main": main_codes,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    log.info("context_sha256=%s (来源 %d 项,本地 %d 项)", context_sha, len(sources), sum(1 for s in sources if s["kind"] == "sealed_file"))

    return BatchContext(
        universe=universe,
        trade_dates=trade_dates,
        latest_trade_date=latest_complete,
        latest_report_period=latest_report_period(today),
        index_codes=idx_codes,
        index_codes_main=main_codes,
        context_sha256=context_sha,
        sources=sources,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, choices=P0_BATCH_ORDER)
    parser.add_argument("--config", default="config.archive.yaml")
    parser.add_argument(
        "--snapshot",
        default=None,
        help="Reuse an existing batch snapshot id (resume after fixing failures; "
        "task ids stay stable so completed tasks are skipped).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(REPO_ROOT / "data_lake" / "reports" / f"{args.batch}.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)

    if args.batch == "B3_financial":
        summary_path = REPORTS / "cross_source_summary.json"
        financial = json.loads(summary_path.read_text(encoding="utf-8")).get("financial", {})
        if not financial.get("pass"):
            log.error("财务跨源核验未通过，禁止启动 B3_financial")
            return 2

    config = ArchiveConfig.from_yaml(REPO_ROOT / args.config)
    config.validate_for_run(batch_id=args.batch)
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
    db = TaskStateDB(config.catalog_dir / "archive.duckdb")

    log.info("构建批次上下文…")
    batch_artifact_dir = REPORTS / "batches" / args.batch
    batch_artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        # 断点续跑(2026-07-22 用户指令): 必须加载封存 context + 已物化
        # manifest 精确回放,禁止调用 build_context 按最新交易日重建——
        # context 漂移会产生孤儿任务(ORPHANED_CONTEXT_DRIFT 事件)。
        from ashare_quant.archive.batch import load_resume_specs

        specs = load_resume_specs(config, db, inventory, args.batch, args.snapshot)
        decision = run_batch(
            config, provider, inventory, db, args.batch, None,
            snapshot_id=args.snapshot, resume_specs=specs,
        )
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 0 if decision["decision"] == "pass" else 1

    ctx = build_context(provider)
    log.info(
        "股票全集 %d 只, 最新完整交易日 %s, 最新报告期 %s",
        len(ctx.universe), ctx.latest_trade_date, ctx.latest_report_period,
    )
    # 记录 context 来源与 SHA(审计: 本次运行的宇宙/日历出自哪些封存数据)。
    context_record = {
        "batch": args.batch,
        "context_sha256": ctx.context_sha256,
        "universe_size": len(ctx.universe),
        "latest_trade_date": ctx.latest_trade_date,
        "latest_report_period": ctx.latest_report_period,
        "index_codes": len(ctx.index_codes),
        "index_codes_main": len(ctx.index_codes_main),
        "sources": ctx.sources,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    context_path = batch_artifact_dir / f"context_{ctx.context_sha256[:12]}.json"
    context_path.write_text(json.dumps(context_record, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("context 记录: %s", context_path)

    decision = run_batch(config, provider, inventory, db, args.batch, ctx, snapshot_id=args.snapshot,
                         context_sha256=ctx.context_sha256)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["decision"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
