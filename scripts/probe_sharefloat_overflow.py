#!/usr/bin/env python
"""share_float 15 只截断股票限时溢出探测(2026-07-23 用户图片方案)。

探测顺序(全部只读网关请求,不写库):
  A. 基线: 全历史 {ts_code},确认恰 6000 行 + 原始响应 count/has_more 字段值
  B. 分页参数: offset/limit、page/page_size、page_num 等常见变体——
     有效判据: ①不同页主键集合不重叠 ②能取到末页且末页<6000(或越界页为空)
     ③同一请求重复 2 次结果一致 ④各页合计行数>6000(确实突破截断)
  C. 服务端过滤组合: ann_date 单日、float_date 参数、start/end+ann_date——
     有效判据: 结果集与基线确实不同(子集且行数<6000)
  注意: 不用已截断结果中的 holder_name 列表拆分(无法证明覆盖未返回的持有人)。

产出: data_lake/reports/batches/B4_repair/sharefloat_overflow_probe.json

Usage:
    set -a; . ./.env; set +a && uv run --no-sync python scripts/probe_sharefloat_overflow.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ashare_quant.archive.config import ArchiveConfig  # noqa: E402
from ashare_quant.archive.provider import TushareCompatibleHttpProvider  # noqa: E402

STOCKS = ["001248.SZ", "001359.SZ", "300614.SZ", "301583.SZ", "601083.SH",
          "603257.SH", "603352.SH", "603391.SH", "688472.SH", "688552.SH",
          "688727.SH", "688796.SH", "688805.SH", "688807.SH", "688809.SH"]
CANARY = "001248.SZ"
OUT = REPO_ROOT / "data_lake" / "reports" / "batches" / "B4_repair" / "sharefloat_overflow_probe.json"

log = print


def pk_set(columns, items):
    i_code, i_date = columns.index("ts_code"), columns.index("float_date")
    return {(r[i_code], r[i_date]) for r in items}


def req(provider, params) -> dict:
    """原始请求(瞬时错误最多重试 8 轮,指数退避),返回 {ok, rows, cols, items, count, has_more, msg}。"""
    resp = None
    for attempt in range(8):
        resp = provider.request("share_float", params)
        if resp.is_success:
            break
        time.sleep(min(60, 2 ** attempt * 5))
    raw = {}
    try:
        raw = json.loads(resp.raw_payload.decode() if isinstance(resp.raw_payload, bytes) else resp.raw_payload)
    except Exception:
        pass
    data = raw.get("data", {}) if isinstance(raw, dict) else {}
    return {
        "ok": resp.is_success, "rows": resp.row_count, "cols": resp.columns,
        "items": resp.items, "count": data.get("count"), "has_more": data.get("has_more"),
        "msg": resp.message[:80] if resp.message else "",
    }


def probe_pagination(provider, code: str) -> dict:
    """B: offset/limit 与 page 变体。"""
    result = {"tested": [], "valid": None}
    variants = [
        ("offset_limit", [{"limit": "5000", "offset": str(o)} for o in (0, 5000, 10000, 15000)]),
        ("page_pagesize", [{"page": str(p), "page_size": "5000"} for p in (1, 2, 3, 4)]),
        ("pagenum_pagesize", [{"page_num": str(p), "page_size": "5000"} for p in (1, 2, 3, 4)]),
    ]
    for name, pages in variants:
        page_keys, page_rows, stable = [], [], True
        ok_all = True
        for extra in pages:
            params = {"ts_code": code, **extra}
            r1 = req(provider, params)
            time.sleep(0.3)
            r2 = req(provider, params)
            if not (r1["ok"] and r2["ok"]):
                ok_all = False
                break
            if r1["rows"] != r2["rows"] or pk_set(r1["cols"], r1["items"]) != pk_set(r2["cols"], r2["items"]):
                stable = False
            page_keys.append(pk_set(r1["cols"], r1["items"]))
            page_rows.append(r1["rows"])
        if not ok_all:
            result["tested"].append({"variant": name, "verdict": "request_failed"})
            continue
        # 判据①: 页间主键不重叠
        union, overlap = set(), 0
        for ks in page_keys:
            overlap += len(union & ks)
            union |= ks
        non_overlap = overlap == 0
        # 判据②: 存在末页(某页 < 5000 或为 0)
        has_last = any(r < 5000 for r in page_rows)
        # 判据④: 合计确实突破 6000
        total = sum(page_rows)
        breaks_cap = total > 6000 and len(union) > 6000
        verdict = "valid" if (non_overlap and has_last and stable and breaks_cap) else "invalid"
        result["tested"].append({
            "variant": name, "page_rows": page_rows, "unique_pk": len(union),
            "non_overlap": non_overlap, "has_last_page": has_last, "stable": stable,
            "breaks_cap": breaks_cap, "verdict": verdict,
        })
        if verdict == "valid":
            result["valid"] = {"variant": name, "total_rows": total}
            break
        log.info = getattr(log, "info", print)
    return result


def probe_filters(provider, code: str, baseline_keys: set) -> dict:
    """C: ann_date / float_date 过滤组合;结果集确实变化才算有效。"""
    result = {"tested": [], "valid": None}
    tests = [
        ("ann_date_single", {"ann_date": "20210302"}),
        ("float_date_param", {"float_date": "20210302"}),
        ("range_plus_ann_date", {"start_date": "20050101", "end_date": "20260721", "ann_date": "20210302"}),
    ]
    for name, extra in tests:
        r = req(provider, {"ts_code": code, **extra})
        if not r["ok"]:
            result["tested"].append({"filter": name, "verdict": "request_failed", "msg": r["msg"]})
            continue
        keys = pk_set(r["cols"], r["items"])
        changed = keys != baseline_keys
        result["tested"].append({
            "filter": name, "rows": r["rows"], "result_changed": changed,
            "verdict": "valid" if (changed and r["rows"] < 6000) else "invalid",
        })
    return result


def main() -> int:
    started = time.time()
    config = ArchiveConfig.from_yaml(REPO_ROOT / "config.archive.yaml")
    config.validate_for_run()
    provider = TushareCompatibleHttpProvider(
        url_env=config.provider.base_url_env, token_env=config.provider.token_env,
        forbid_token_env=config.provider.forbid_token_env, source_provider=config.provider.name,
        allowed_hosts=config.provider.allowed_hosts, api_key_env=config.provider.api_key_env,
        api_key_header=config.provider.api_key_header)

    report: dict = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "stocks": {}, "method_found": None}

    # A. 金丝雀基线 + 全方法探测
    log(f"[A] canary {CANARY} baseline")
    base = req(provider, {"ts_code": CANARY})
    base_keys = pk_set(base["cols"], base["items"])
    report["stocks"][CANARY] = {
        "baseline_rows": base["rows"], "count_field": base["count"], "has_more": base["has_more"],
    }
    log(f"    rows={base['rows']} count={base['count']} has_more={base['has_more']}")

    log("[B] pagination variants")
    pag = probe_pagination(provider, CANARY)
    report["stocks"][CANARY]["pagination"] = pag
    for t in pag["tested"]:
        log(f"    {t}")

    if pag["valid"]:
        report["method_found"] = pag["valid"]
        log(f"[B] VALID pagination: {pag['valid']}")
    else:
        log("[C] server-side filter combos")
        filt = probe_filters(provider, CANARY, base_keys)
        report["stocks"][CANARY]["filters"] = filt
        for t in filt["tested"]:
            log(f"    {t}")

    # 其余 14 只: 只跑基线确认 6000 截断 + count/has_more 证据(限时)
    for code in STOCKS[1:]:
        if time.time() - started > 3000:  # 50 分钟硬上限,留 10 分钟写报告
            report["timeboxed"] = True
            break
        r = req(provider, {"ts_code": code})
        report["stocks"][code] = {
            "baseline_rows": r["rows"], "count_field": r["count"], "has_more": r["has_more"],
        }
        log(f"[A] {code}: rows={r['rows']} count={r['count']} has_more={r['has_more']}")

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["elapsed_min"] = round((time.time() - started) / 60, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"probe written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
