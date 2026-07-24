#!/usr/bin/env python
"""B4_events 预检探针(2026-07-22 用户图片方案,启动 B4 前必须完成)。

探针项(只读网关,不落盘 Raw/Bronze,不写任务库):
  1. canary: 逐端点验证实际支持的查询参数与 fields=""(显式 fields 对比);
  2. columns/PK: 响应列清单,校验注册主键存在,寻找修订字段;
  3. 宽区间 vs 拆分区间一致性 + 真实 row cap 迹象;
  4. 空响应(退市股)、schema 变体指纹、PIT ann_date 非空。

输出: data_lake/reports/batches/B4_events/preflight/*.json

Usage:
    set -a; . ./.env; set +a && uv run --no-sync python scripts/preflight_b4.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ashare_quant.archive.config import ArchiveConfig  # noqa: E402
from ashare_quant.archive.provider import TushareCompatibleHttpProvider  # noqa: E402
from ashare_quant.archive.registry import default_inventory  # noqa: E402
from ashare_quant.archive.schema import schema_fingerprint  # noqa: E402

OUT = REPO_ROOT / "data_lake" / "reports" / "batches" / "B4_events" / "preflight"
BRONZE = REPO_ROOT / "data_lake" / "bronze" / "research_proxy_unverified"
HEAVY_STOCKS = ["000001.SZ", "600000.SH", "000002.SZ"]


def _request_with_retry(provider, api: str, params: dict, fields: str = ""):
    """与 run_p0_batch 同策略: Retry-After 优先,指数退避+jitter,最多 8 次。"""
    delay = 5.0
    for attempt in range(8):
        resp = provider.request(api, params, fields=fields)
        transient = resp.status == "transient_error" or any(
            k in resp.message for k in ("429", "频率受限", "超限", "网络错误", "超时", "服务端错误")
        )
        if not transient:
            return resp
        wait = resp.retry_after_seconds if resp.retry_after_seconds else delay
        wait = min(wait, 300.0) * (1 + random.random() * 0.5)
        print(f"  [retry {attempt + 1}/8] {api} {resp.message[:40]} wait {wait:.0f}s", flush=True)
        time.sleep(wait)
        delay = min(delay * 2, 300.0)
    return resp


def _brief(resp) -> dict:
    fp = schema_fingerprint(resp.columns) if resp.columns else ""
    return {
        "status": resp.status, "rows": resp.row_count, "columns": resp.columns,
        "fingerprint": fp, "message": resp.message[:120],
    }


def _rows_key(resp) -> str:
    payload = json.dumps(resp.items, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config = ArchiveConfig.from_yaml(REPO_ROOT / "config.archive.yaml")
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
    eps = {n: ep for n, ep in inventory.endpoints.items() if getattr(ep, "batch", None) == "B4_events"}

    listed = pd.read_parquet(BRONZE / "stock_basic" / "stock_basic_list_status=L_phase_a_20260716_135101.parquet")
    delisted = pd.read_parquet(BRONZE / "stock_basic" / "stock_basic_list_status=D_phase_a_20260716_135101.parquet")
    universe = sorted(set(listed["ts_code"]) | set(delisted["ts_code"]))
    delisted_sample = sorted(delisted["ts_code"])[:2]

    report: dict[str, dict] = {}

    for name, ep in eps.items():
        print(f"=== {name} ===", flush=True)
        entry: dict = {"probes": {}}

        def probe(label: str, params: dict, fields: str = ""):
            resp = _request_with_retry(provider, name, params, fields)
            entry["probes"][label] = {"params": params, "fields": fields, **_brief(resp)}
            print(f"  {label}: {resp.status} rows={resp.row_count} cols={len(resp.columns)}", flush=True)
            return resp

        # 1. canary: 注册 probe_params + fields=""
        canary = probe("canary_registry_params", dict(ep.probe_params or {}))

        # 2. fields 显式对比(fields="" 支持性)
        if canary.is_success and canary.columns:
            explicit = ",".join(canary.columns)
            r2 = _request_with_retry(provider, name, dict(ep.probe_params or {}), explicit)
            entry["fields_explicit_equal"] = (
                r2.is_success and r2.columns == canary.columns and _rows_key(r2) == _rows_key(canary)
            )
            print(f"  fields 显式对比: {entry['fields_explicit_equal']}", flush=True)

        # 3. 端点特定探针
        if name == "repurchase":
            # 宽区间(全历史,观察 cap 迹象) vs 年度拆分求和(2024 样本逐月 vs 全年)
            wide = probe("wide_2015_2026", {"start_date": "20150101", "end_date": "20260721"})
            y2024 = probe("year_2024", {"start_date": "20240101", "end_date": "20241231"})
            months_total, months_keys = 0, set()
            for m in range(1, 13):
                s = f"2024{m:02d}01"
                e = f"2024{m:02d}31"
                r = _request_with_retry(provider, name, {"start_date": s, "end_date": e})
                months_total += r.row_count
                for row in r.items:
                    months_keys.add(json.dumps(row, ensure_ascii=False, default=str))
            entry["split_vs_full_2024"] = {
                "year_rows": y2024.row_count, "months_rows": months_total,
                "year_rows_key": _rows_key(y2024),
                "months_distinct": len(months_keys),
                "count_match": y2024.row_count == months_total == len(months_keys),
            }
            entry["wide_rows"] = wide.row_count
            print(f"  2024 全年={y2024.row_count} 逐月合计={months_total} 宽区间={wide.row_count}", flush=True)
        elif name in ("share_float", "top10_holders"):
            # 多码 chunk 网关支持性(B3 教训: 部分端点拒绝多码)
            chunk5 = ",".join(universe[:5])
            probe("chunk_5_codes", {"ts_code": chunk5})
            singles_ok, single_rows = 0, 0
            for code in universe[:5]:
                r = _request_with_retry(provider, name, {"ts_code": code})
                singles_ok += 1 if r.status in ("success", "empty") else 0
                single_rows += r.row_count
            entry["single_code_total_rows_first5"] = single_rows
            entry["single_code_ok_first5"] = singles_ok
        else:
            # 单码全历史: 重仓股最大行数 + 退市股空响应
            max_rows = 0
            for code in HEAVY_STOCKS:
                r = probe(f"heavy_{code}", {"ts_code": code})
                max_rows = max(max_rows, r.row_count)
            entry["max_rows_heavy"] = max_rows
            for code in delisted_sample:
                probe(f"delisted_{code}", {"ts_code": code})

        # 4. schema 指纹集合 + PK 校验 + ann_date 非空抽查
        fps = {p["fingerprint"] for p in entry["probes"].values() if p["fingerprint"]}
        entry["fingerprints"] = sorted(fps)
        success_cols = next((p["columns"] for p in entry["probes"].values() if p["status"] == "success"), [])
        entry["pk_registered"] = list(ep.primary_key)
        entry["pk_missing_in_response"] = [c for c in ep.primary_key if c not in success_cols]
        entry["revision_field_candidates"] = [
            c for c in success_cols if any(k in c for k in ("update", "change", "ann", "modify"))
        ]
        if "ann_date" in success_cols:
            # items 未存,逐条响应已释放;ann 非空在 B4 批次中由 bronze 层复核
            entry["ann_date_in_columns"] = True
        report[name] = entry

    out = OUT / "b4_canary.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n探针完成: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
