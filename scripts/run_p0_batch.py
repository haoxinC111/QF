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
import json
import logging
import os
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


def build_context(provider: TushareCompatibleHttpProvider) -> BatchContext:
    listed = pd.read_parquet(
        BRONZE / "stock_basic" / "stock_basic_list_status=L_phase_a_20260716_135101.parquet"
    )
    delisted = pd.read_parquet(
        BRONZE / "stock_basic" / "stock_basic_list_status=D_phase_a_20260716_135101.parquet"
    )
    universe = sorted(set(listed["ts_code"]) | set(delisted["ts_code"]))

    resp = provider.request(
        "trade_cal", {"exchange": "SSE", "start_date": "19901219", "end_date": "20261231", "is_open": "1"}
    )
    if not resp.is_success:
        raise RuntimeError(f"trade_cal 拉取失败: {resp.message}")
    trade_dates = sorted(r[resp.columns.index("cal_date")] for r in resp.items)
    today = time.strftime("%Y%m%d", time.gmtime())
    completed = [d for d in trade_dates if d < today]
    latest_complete = completed[-1]

    idx_codes: list[str] = []
    seen_codes: set[str] = set()
    # index_basic 全字段响应存在行数上限(实测 market=CSI 被截在 4879 行,
    # 而 category 细分求和 ~8800)。CSI 无条件按 category 细分;其他 market
    # 段达到保守告警阈值时显式报错,绝不静默接受疑似截断的宇宙。
    for market in INDEX_BASIC_MARKETS:
        if market == "CSI":
            segments = [{"market": "CSI", "category": c} for c in CSI_INDEX_CATEGORIES]
        else:
            segments = [{"market": market}]
        for seg in segments:
            resp = provider.request("index_basic", seg)
            if not resp.is_success and resp.status != "empty":
                raise RuntimeError(f"index_basic {seg} 拉取失败: {resp.message}")
            if resp.status == "empty":
                continue  # 合法空段(如 CICC/OTHERS 当前无指数)
            if len(resp.items) >= INDEX_BASIC_ROW_ALERT:
                raise RuntimeError(
                    f"index_basic {seg} 返回 {len(resp.items)} 行达到告警阈值,疑似截断,需进一步细分"
                )
            for row in resp.items:
                code = row[resp.columns.index("ts_code")]
                if code not in seen_codes:
                    seen_codes.add(code)
                    idx_codes.append(code)
            time.sleep(1)
    logging.getLogger(__name__).info("指数全量宇宙(分段合并): %d 只", len(idx_codes))

    # 主力宇宙:B2 index_daily/index_weight 展开范围(分层归档,主力先行)。
    main_codes: list[str] = []
    seen_main: set[str] = set()
    for seg in MAIN_INDEX_SEGMENTS:
        resp = provider.request("index_basic", dict(seg))
        if not resp.is_success:
            raise RuntimeError(f"index_basic 主力段 {seg} 拉取失败: {resp.message}")
        for row in resp.items:
            code = row[resp.columns.index("ts_code")]
            if code not in seen_main:
                seen_main.add(code)
                main_codes.append(code)
        time.sleep(1)
    logging.getLogger(__name__).info("主力指数宇宙: %d 只 (段: %s)", len(main_codes), list(MAIN_INDEX_SEGMENTS))

    return BatchContext(
        universe=universe,
        trade_dates=trade_dates,
        latest_trade_date=latest_complete,
        latest_report_period=latest_report_period(today),
        index_codes=idx_codes,
        index_codes_main=main_codes,
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
    ctx = build_context(provider)
    log.info(
        "股票全集 %d 只, 最新完整交易日 %s, 最新报告期 %s",
        len(ctx.universe), ctx.latest_trade_date, ctx.latest_report_period,
    )

    decision = run_batch(config, provider, inventory, db, args.batch, ctx, snapshot_id=args.snapshot)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["decision"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
