#!/usr/bin/env python
"""Market cross-source validation: archive proxy vs independent public sources.

Compares OHLC / pre_close / vol / amount between the Tushare-compatible
gateway (the archive source) and independent public sources (Eastmoney
primary, Sina fallback) for a seeded sample of 100 stocks x 20 common
trade dates, covering SSE / SZSE / ST / delisted names.

Units (normalised before comparison):
    prices  : CNY (raw, unadjusted) — both sources
    vol     : 手 (100 shares) — proxy native; sina 股 / 100
    amount  : 千元 — proxy native; eastmoney/sina 元 / 1000

Reads QF_ARCHIVE_API_URL / QF_ARCHIVE_API_TOKEN from the environment.
Never writes the token anywhere.

Outputs (under data_lake/reports/):
    market_cross_source_details.csv
    cross_source_summary.json
    cross_source_validation.md
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# akshare's internal requests have no default timeout and can hang forever on
# flaky hosts; give every HTTP call a sane default (explicit timeouts win).
_orig_session_request = requests.sessions.Session.request


def _request_with_default_timeout(self, method, url, **kwargs):
    kwargs.setdefault("timeout", 30)
    return _orig_session_request(self, method, url, **kwargs)


requests.sessions.Session.request = _request_with_default_timeout

# The local HTTP proxy (127.0.0.1:7897) is unreliable for public finance
# endpoints; go direct for both the archive gateway and the public sources.
for _var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_var, None)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ashare_quant.archive.provider import TushareCompatibleHttpProvider  # noqa: E402
from ashare_quant.archive.throttle import RateLimitedClient, RetryPolicy, TokenBucket  # noqa: E402

SEED = 20260716
WINDOW_START = "20250616"
WINDOW_END = "20250715"
N_DATES = 20
BRONZE = REPO_ROOT / "data_lake" / "bronze" / "research_proxy_unverified"
REPORTS = REPO_ROOT / "data_lake" / "reports"

PRICE_TOL_ABS = 0.011       # CNY
VOL_TOL_REL = 2e-3          # 手
VOL_TOL_ABS = 2.0
AMT_TOL_REL = 5e-3          # 千元
AMT_TOL_ABS = 50.0


def load_stock_basic() -> tuple[pd.DataFrame, pd.DataFrame]:
    listed = pd.read_parquet(
        BRONZE / "stock_basic" / "stock_basic_list_status=L_phase_a_20260716_135101.parquet"
    )
    delisted = pd.read_parquet(
        BRONZE / "stock_basic" / "stock_basic_list_status=D_phase_a_20260716_135101.parquet"
    )
    return listed, delisted


def sample_universe(listed: pd.DataFrame, delisted: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED)

    def pick(df: pd.DataFrame, n: int) -> list[str]:
        codes = sorted(df["ts_code"].unique())
        return rng.sample(codes, min(n, len(codes)))

    sh_main = listed[listed["ts_code"].str.match(r"60[0135]\d{3}\.SH")]
    sh_star = listed[listed["ts_code"].str.match(r"688\d{3}\.SH")]
    sz_main = listed[listed["ts_code"].str.match(r"00[0123]\d{3}\.SZ")]
    sz_gem = listed[listed["ts_code"].str.match(r"30[01]\d{3}\.SZ")]
    st_mask = listed["name"].str.contains("ST", case=False, na=False)
    st_all = listed[st_mask]

    recent_delist = delisted[
        (delisted["delist_date"] >= "20250801")
        & delisted["ts_code"].str.endswith((".SH", ".SZ"))
    ]

    picks: dict[str, list[str]] = {
        "sh_main": pick(sh_main, 25),
        "sh_star": pick(sh_star, 10),
        "sz_main": pick(sz_main, 25),
        "sz_gem": pick(sz_gem, 10),
        "st": pick(st_all, 10),
        "delisted": pick(recent_delist, 20),
    }
    rows = []
    seen: set[str] = set()
    for bucket, codes in picks.items():
        for code in codes:
            if code in seen:
                continue
            seen.add(code)
            rows.append({"ts_code": code, "bucket": bucket})
    return pd.DataFrame(rows)


def common_trade_dates() -> list[str]:
    cal = pd.read_parquet(
        BRONZE / "trade_cal" / "trade_cal_exchange=SSE_is_open=1_phase_a_20260716_135101.parquet"
    )
    window = cal[(cal["cal_date"] >= WINDOW_START) & (cal["cal_date"] <= WINDOW_END)]
    dates = sorted(window["cal_date"].unique())
    return dates[-N_DATES:]


def fetch_proxy_daily(
    client: RateLimitedClient,
    provider: TushareCompatibleHttpProvider,
    dates: list[str],
) -> pd.DataFrame:
    frames = []
    for trade_date in dates:
        resp = client.call(provider.request, "daily", {"trade_date": trade_date})
        if not resp.is_success:
            raise RuntimeError(f"代理 daily {trade_date} 拉取失败: {resp.status} {resp.message}")
        df = pd.DataFrame(resp.items, columns=resp.columns)
        frames.append(df)
        print(f"  proxy daily {trade_date}: {len(df)} rows", flush=True)
    out = pd.concat(frames, ignore_index=True)
    return out[["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"]]


def explain_preclose_mismatches(
    details_df: pd.DataFrame,
    client: RateLimitedClient,
    provider: TushareCompatibleHttpProvider,
    cal_start: str,
) -> int:
    """Reclassify pre_close mismatches that are explained by corporate actions.

    On ex-dividend days Tushare ``pre_close`` is the adjusted previous close,
    while a raw-kline shift gives the unadjusted one.  Both are correct; the
    proxy's own adj_factor change is the evidence.  Such cells are excluded
    from the mismatch count (match=None) and annotated instead.
    """
    mism = details_df[
        (details_df["status"] == "compared") & (details_df["pre_close_match"] == False)  # noqa: E712
    ]
    explained = 0
    for ts_code in mism["ts_code"].unique():
        resp = client.call(
            provider.request,
            "adj_factor",
            {"ts_code": ts_code, "start_date": cal_start, "end_date": WINDOW_END},
        )
        if not resp.is_success or not resp.items:
            continue
        fac = pd.DataFrame(resp.items, columns=resp.columns).sort_values("trade_date")
        prev = fac["adj_factor"].astype(float).shift(1)
        changed = set(fac.loc[fac["adj_factor"].astype(float) != prev, "trade_date"])
        mask = (
            (details_df["ts_code"] == ts_code)
            & (details_df["pre_close_match"] == False)  # noqa: E712
            & details_df["trade_date"].isin(changed)
        )
        details_df.loc[mask, "pre_close_match"] = None
        details_df.loc[mask, "pre_close_note"] = "corporate_action_ex_dividend"
        explained += int(mask.sum())
    return explained


def _fetch_eastmoney(symbol6: str, start: str, end: str) -> pd.DataFrame | None:
    import akshare as ak

    df = ak.stock_zh_a_hist(symbol=symbol6, period="daily", start_date=start, end_date=end, adjust="")
    if df is None or df.empty:
        return None
    out = pd.DataFrame(
        {
            "trade_date": df["日期"].astype(str).str.replace("-", ""),
            "open": df["开盘"].astype(float),
            "high": df["最高"].astype(float),
            "low": df["最低"].astype(float),
            "close": df["收盘"].astype(float),
            "vol": df["成交量"].astype(float),          # 手
            "amount": df["成交额"].astype(float) / 1000.0,  # 元 -> 千元
        }
    ).sort_values("trade_date").reset_index(drop=True)
    out["pre_close"] = out["close"].shift(1)
    return out


def _fetch_sina(ts_code: str, start: str, end: str) -> pd.DataFrame | None:
    import akshare as ak

    symbol = ("sh" if ts_code.endswith(".SH") else "sz") + ts_code[:6]
    df = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust="")
    if df is None or df.empty:
        return None
    out = pd.DataFrame(
        {
            "trade_date": df["date"].astype(str).str.replace("-", ""),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "vol": df["volume"].astype(float) / 100.0,      # 股 -> 手
            "amount": df["amount"].astype(float) / 1000.0,  # 元 -> 千元
        }
    ).sort_values("trade_date").reset_index(drop=True)
    out["pre_close"] = out["close"].shift(1)
    return out


def _fetch_tencent(ts_code: str, start: str, end: str) -> pd.DataFrame | None:
    """Tencent daily kline — the only public source found serving delisted names.

    Array layout: [date, open, close, high, low, volume(手)]; no amount field.
    """
    symbol = ("sh" if ts_code.endswith(".SH") else "sz") + ts_code[:6]
    symbol6 = ts_code[:6]
    start_dash = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    end_dash = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    resp = requests.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        params={"param": f"{symbol},day,{start_dash},{end_dash},640,"},
        timeout=30,
    )
    node = resp.json().get("data", {}).get(symbol, {})
    rows = node.get("day") or node.get("qfqday") or []
    if not rows:
        return None
    # Tencent returns STAR-market (688/689) volume in shares, others in 手.
    vol_divisor = 100.0 if symbol6.startswith(("688", "689")) else 1.0
    out = pd.DataFrame(
        {
            "trade_date": [str(x[0]).replace("-", "") for x in rows],
            "open": [float(x[1]) for x in rows],
            "high": [float(x[3]) for x in rows],
            "low": [float(x[4]) for x in rows],
            "close": [float(x[2]) for x in rows],
            "vol": [float(x[5]) / vol_divisor for x in rows],  # -> 手
            "amount": [float("nan")] * len(rows),  # tencent 基础K线无成交额
        }
    ).sort_values("trade_date").reset_index(drop=True)
    out["pre_close"] = out["close"].shift(1)
    return out


def fetch_independent(ts_code: str, cal_start: str, cal_end: str) -> tuple[pd.DataFrame | None, str]:
    """Eastmoney primary, Sina fallback, Tencent last resort. Returns (df, source)."""
    symbol6 = ts_code[:6]
    try:
        df = _fetch_eastmoney(symbol6, cal_start, cal_end)
        if df is not None and not df.empty:
            return df, "eastmoney"
    except Exception as exc:  # noqa: BLE001
        print(f"  eastmoney {ts_code} 失败: {exc}", flush=True)
    try:
        df = _fetch_sina(ts_code, cal_start, cal_end)
        if df is not None and not df.empty:
            return df, "sina"
    except Exception as exc:  # noqa: BLE001
        print(f"  sina {ts_code} 失败: {exc}", flush=True)
    try:
        df = _fetch_tencent(ts_code, cal_start, cal_end)
        if df is not None and not df.empty:
            return df, "tencent"
    except Exception as exc:  # noqa: BLE001
        print(f"  tencent {ts_code} 失败: {exc}", flush=True)
    return None, "unavailable"


def _close(a: float, b: float, tol_abs: float, tol_rel: float) -> bool:
    return abs(a - b) <= max(tol_abs, tol_rel * max(abs(a), abs(b)))


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    listed, delisted = load_stock_basic()
    universe = sample_universe(listed, delisted)
    dates = common_trade_dates()
    print(f"样本 {len(universe)} 只, 窗口 {dates[0]}..{dates[-1]} 共 {len(dates)} 个交易日", flush=True)

    print("== 拉取代理 daily 全市场横截面 ==", flush=True)
    provider = TushareCompatibleHttpProvider()
    client = RateLimitedClient(TokenBucket(75), RetryPolicy(max_attempts=5))
    proxy = fetch_proxy_daily(client, provider, dates)
    proxy = proxy[proxy["ts_code"].isin(universe["ts_code"])]

    # Pre-window cushion so the independent source can derive pre_close.
    cal_start = (pd.Timestamp(dates[0]) - pd.Timedelta(days=15)).strftime("%Y%m%d")
    details: list[dict] = []
    source_counts: dict[str, int] = {}
    compared_stocks = 0

    print("== 拉取独立源逐股 K 线并比对 ==", flush=True)
    for idx, row in enumerate(universe.itertuples(), 1):
        ts_code, bucket = row.ts_code, row.bucket
        indep, source = fetch_independent(ts_code, cal_start, WINDOW_END)
        print(f"  [{idx}/{len(universe)}] {ts_code} ({bucket}) <- {source}", flush=True)
        source_counts[source] = source_counts.get(source, 0) + 1
        if indep is None:
            for trade_date in dates:
                details.append(
                    {"ts_code": ts_code, "bucket": bucket, "trade_date": trade_date,
                     "source": source, "status": "source_unavailable"}
                )
            continue
        indep = indep[indep["trade_date"].isin(dates)]
        sub_proxy = proxy[proxy["ts_code"] == ts_code].set_index("trade_date")
        sub_indep = indep.set_index("trade_date")
        compared_stocks += 1
        for trade_date in dates:
            p = sub_proxy.loc[trade_date] if trade_date in sub_proxy.index else None
            q = sub_indep.loc[trade_date] if trade_date in sub_indep.index else None
            rec: dict = {"ts_code": ts_code, "bucket": bucket, "trade_date": trade_date, "source": source}
            if p is None and q is None:
                rec["status"] = "both_absent"  # suspended in both: agreement
            elif p is None or q is None:
                rec["status"] = "missing_one_side"
                rec["missing_side"] = "proxy" if p is None else "independent"
            else:
                rec["status"] = "compared"
                all_match = True
                for field, (ta, tr) in {
                    "open": (PRICE_TOL_ABS, 0), "high": (PRICE_TOL_ABS, 0),
                    "low": (PRICE_TOL_ABS, 0), "close": (PRICE_TOL_ABS, 0),
                    "pre_close": (PRICE_TOL_ABS, 0),
                    "vol": (VOL_TOL_ABS, VOL_TOL_REL),
                    "amount": (AMT_TOL_ABS, AMT_TOL_REL),
                }.items():
                    a, b = float(p[field]), float(q[field])
                    if pd.isna(b):
                        # Source lacks this field (e.g. tencent amount, first-row
                        # pre_close): excluded from the match-rate denominator.
                        rec[f"{field}_match"] = None
                        continue
                    match = _close(a, b, ta, tr)
                    rec[f"proxy_{field}"] = a
                    rec[f"indep_{field}"] = b
                    rec[f"{field}_diff"] = a - b
                    rec[f"{field}_match"] = match
                    all_match = all_match and match
                rec["row_match"] = all_match
            details.append(rec)
        time.sleep(0.3)  # be polite to public sources

    details_df = pd.DataFrame(details)

    # Corporate-action pass: ex-dividend pre_close differences are expected.
    corp_explained = explain_preclose_mismatches(details_df, client, provider, cal_start)

    # Recompute row-level match treating excluded (None) cells as pass.
    match_cols = [
        f"{f}_match"
        for f in ("open", "high", "low", "close", "pre_close", "vol", "amount")
    ]

    def _row_ok(rec: pd.Series) -> bool:
        vals = [rec[c] for c in match_cols if not pd.isna(rec[c])]
        return bool(all(vals))

    compared_mask = details_df["status"] == "compared"
    details_df.loc[compared_mask, "row_match"] = details_df[compared_mask].apply(_row_ok, axis=1)

    csv_path = REPORTS / "market_cross_source_details.csv"
    details_df.to_csv(csv_path, index=False)

    compared = details_df[details_df["status"] == "compared"]
    field_rates = {}
    for field in ("open", "high", "low", "close", "pre_close", "vol", "amount"):
        col = compared[f"{field}_match"].dropna()
        field_rates[field] = {
            "compared": int(len(col)),
            "matched": int(col.sum()),
            "match_rate": round(float(col.mean()), 6) if len(col) else None,
        }
    row_match_rate = float(compared["row_match"].mean()) if len(compared) else 0.0
    missing = details_df[details_df["status"] == "missing_one_side"]
    unavailable = details_df[details_df["status"] == "source_unavailable"]["ts_code"].nunique()

    price_ok = all(
        (field_rates[f]["match_rate"] or 0) >= 0.995
        for f in ("open", "high", "low", "close", "pre_close")
    )
    va_ok = all((field_rates[f]["match_rate"] or 0) >= 0.99 for f in ("vol", "amount"))
    coverage_ok = compared_stocks >= 80
    validation_pass = bool(price_ok and va_ok and coverage_ok and row_match_rate >= 0.99)

    summary = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "window": {"start": dates[0], "end": dates[-1], "trade_dates": len(dates)},
        "sample_size": int(len(universe)),
        "buckets": universe["bucket"].value_counts().to_dict(),
        "units": {"price": "CNY raw", "vol": "手(100股)", "amount": "千元"},
        "tolerances": {
            "price_abs": PRICE_TOL_ABS, "vol_rel": VOL_TOL_REL,
            "amount_rel": AMT_TOL_REL,
        },
        "independent_sources": source_counts,
        "stocks_compared": compared_stocks,
        "stocks_source_unavailable": int(unavailable),
        "cells_compared": int(len(compared)),
        "both_absent_cells": int((details_df["status"] == "both_absent").sum()),
        "missing_one_side_cells": int(len(missing)),
        "pre_close_corporate_action_explained": int(corp_explained),
        "field_match_rates": field_rates,
        "row_match_rate": round(row_match_rate, 6),
        "pass": validation_pass,
        "pass_criteria": "价格字段一致率≥99.5%, vol/amount≥99%, 行一致率≥99%, 覆盖≥80只",
        "details_csv": str(csv_path),
    }
    summary_path = REPORTS / "cross_source_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# 市场数据跨源核验报告",
        "",
        f"- 生成时间(UTC): {summary['generated_at_utc']}",
        f"- 随机种子: {SEED}",
        f"- 样本: {len(universe)} 只（{summary['buckets']}）",
        f"- 窗口: {dates[0]} ~ {dates[-1]}，{len(dates)} 个共同交易日",
        "- 单位: 价格=元(原始价), vol=手, amount=千元",
        "- 独立源: " + json.dumps(source_counts, ensure_ascii=False),
        "",
        "## 字段一致率",
        "",
        "| 字段 | 比对单元格 | 一致 | 一致率 |",
        "|---|---|---|---|",
    ]
    for field, stat in field_rates.items():
        rate = f"{stat['match_rate']:.4%}" if stat["match_rate"] is not None else "n/a"
        md_lines.append(f"| {field} | {stat['compared']} | {stat['matched']} | {rate} |")
    md_lines += [
        "",
        f"- 行一致率: {row_match_rate:.4%}",
        f"- 双缺(停牌一致)单元格: {summary['both_absent_cells']}",
        f"- 单侧缺失单元格: {summary['missing_one_side_cells']}",
        f"- 独立源不可用股票数: {unavailable}",
        f"- pre_close 除权除息差异(经 adj_factor 佐证,不计入不一致): {corp_explained}",
        "",
        f"## 结论: {'**PASS**' if validation_pass else '**FAIL**'}",
        "",
        f"判定标准: {summary['pass_criteria']}",
    ]
    md_path = REPORTS / "cross_source_validation.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("stocks_compared", "row_match_rate", "pass")}, ensure_ascii=False))
    print(f"报告: {csv_path} / {summary_path} / {md_path}")
    return 0 if validation_pass else 1


if __name__ == "__main__":
    sys.exit(main())
