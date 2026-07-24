#!/usr/bin/env python3
"""archive → v4 基础行情缓存兼容层构建器(驱动)。

用法:
  uv run --no-sync python scripts/build_base_cache_from_archive.py \
    --config config.yaml \
    --archive-root ../a_share_quant/data_lake \
    --catalog data_lake/catalog/archive.duckdb

前置门禁(G1..G8)任一失败即停止,缺口报告写 --report(默认
results/base_cache_build_gap_report.json),退出码 2;全部门禁通过才构建
v4 缓存并用真实消费者 MarketDataBundle.from_cache(strict) 回读验证。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ashare_quant.base_cache_bridge import (  # noqa: E402
    MAPPING_VERSION,
    GateResult,
    build_bars_for_symbol,
    build_corporate_actions,
    build_index_frame,
    build_industry_membership,
    build_securities,
    load_success_tasks,
    membership_from_index_weight,
    month_end_dates,
    read_bronze,
    select_generations,
    sha256_payload,
    utc_now_iso,
    write_deterministic_csv_gz,
)
from ashare_quant.config import AppConfig  # noqa: E402
from ashare_quant.data import MarketDataBundle  # noqa: E402
from ashare_quant.provenance import (  # noqa: E402
    build_file_inventory,
    inventory_sha256,
    sha256_file,
)

# 交付状态库快照钉定。历史值:
#   2026-07-23 首轮交付 32ce478375271345b90087a9d02354fb2d0471a257b03e7c4d2e727c1b60e077
#   (该快照含 index_member_all/namechange 静默截断,G7/G8 fail-closed);
#   当前值为 2026-07-24 B0 截断修复后的快照。
DELIVERED_CATALOG_SHA256 = (
    "1d6acee0f1182253f966047b0adfcf9c6c09ddfa2036b7b172963fc33edd1d11"
)
REQUIRED_APIS = {
    "trade_cal", "stock_basic", "index_weight", "index_daily", "daily",
    "adj_factor", "stk_limit", "daily_basic", "dividend", "namechange",
    "index_member_all", "index_classify",
}
# 验收证据(只读引用,SHA 写入 provenance)
B1_ACCEPTANCE = "reports/batches/B1_B2_pre_cleanup_acceptance.json"
B2_ACCEPTANCE = "reports/batches/B2_final_strict_acceptance.json"


def _date_str(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", default=None, help="默认 <config.data.cache_dir>")
    parser.add_argument("--report", default="results/base_cache_build_gap_report.json")
    parser.add_argument("--catalog-sha256", default=DELIVERED_CATALOG_SHA256)
    parser.add_argument("--force", action="store_true", help="覆盖已有缓存目录")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = AppConfig.from_yaml(config_path).resolve_paths(config_path.parent)
    archive_root = Path(args.archive_root).resolve()
    catalog = Path(args.catalog).resolve()
    output = Path(args.output).resolve() if args.output else Path(config.data.cache_dir).resolve()
    report_path = Path(args.report).resolve()

    start = pd.Timestamp(config.backtest.start_date)
    end = pd.Timestamp(config.backtest.end_date)
    warmup_start = start - pd.Timedelta(config.data.warmup_calendar_days, unit="D")
    membership_start = start - pd.Timedelta(62, unit="D")

    gates: list[GateResult] = []
    provenance: dict = {"mapping_version": MAPPING_VERSION}

    # ---------- G1: 状态库快照指纹 ----------
    catalog_sha = sha256_file(catalog)
    gates.append(GateResult(
        "G1_catalog_pin", catalog_sha == args.catalog_sha256.lower(),
        f"catalog sha256={catalog_sha}",
        {"expected": args.catalog_sha256.lower(), "actual": catalog_sha},
    ))
    provenance["catalog_sha256"] = catalog_sha

    # ---------- 加载与代际去重(只 success;其余状态天然排除) ----------
    tasks = load_success_tasks(catalog, REQUIRED_APIS)
    selection = select_generations(tasks)
    chosen = selection.tasks
    provenance["dropped_duplicate_generations"] = selection.dropped_duplicates
    provenance["selected_task_count"] = len(chosen)

    def by_api(api: str):
        return [t for (a, _), t in chosen.items() if a == api]

    # ---------- G2: 必需端点存在 ----------
    missing_apis = sorted(api for api in REQUIRED_APIS if not by_api(api))
    gates.append(GateResult(
        "G2_required_endpoints", not missing_apis,
        f"缺失端点: {missing_apis}" if missing_apis else "12 个必需端点齐备",
        {"missing": missing_apis},
    ))

    # ---------- G3: 交易日历覆盖 ----------
    cal_frames = [read_bronze(t, archive_root) for t in by_api("trade_cal")]
    cal_raw = pd.concat(cal_frames, ignore_index=True) if cal_frames else pd.DataFrame()
    calendar = pd.DatetimeIndex([])
    if {"cal_date", "is_open"}.issubset(cal_raw.columns):
        open_days = cal_raw[cal_raw["is_open"].astype(str).isin({"1", "1.0"})]
        # 多个 success 分区(phase_a + B0)直接 concat 会按分区重复同一交易日,
        # 必须先按 cal_date 去重再建 DatetimeIndex,否则窗口交易日计数翻倍。
        calendar = pd.DatetimeIndex(
            pd.to_datetime(open_days["cal_date"].astype(str), format="%Y%m%d").unique()
        ).sort_values()
    window_days = calendar[(calendar >= warmup_start) & (calendar <= end)]
    cal_ok = len(window_days) > 0 and calendar.min() <= warmup_start and calendar.max() >= end
    gates.append(GateResult(
        "G3_calendar_coverage", bool(cal_ok),
        f"窗口交易日 {len(window_days)} 天, 日历 {calendar.min()} → {calendar.max()}" if len(calendar) else "交易日历为空",
        {"window_trading_days": len(window_days)},
    ))

    # ---------- G4: 行情端点窗口日覆盖 ----------
    g4_detail = {}
    g4_ok = True
    daily_dates: set[str] = set()
    for api in ("daily", "adj_factor", "daily_basic", "stk_limit"):
        dates = {
            str(t.params.get("trade_date", ""))
            for t in by_api(api)
            if t.params.get("trade_date")
        }
        if api == "daily":
            daily_dates = dates
        expected = {d.strftime("%Y%m%d") for d in window_days}
        missing = sorted(expected - dates)
        g4_detail[api] = {"covered": len(expected & dates), "missing_count": len(missing), "missing_sample": missing[:10]}
        if missing:
            g4_ok = False
    gates.append(GateResult(
        "G4_market_daily_coverage", g4_ok,
        "daily/adj_factor/daily_basic/stk_limit 窗口日覆盖" + ("完整" if g4_ok else "存在缺口"),
        g4_detail,
    ))

    # ---------- G5: PIT 成分宇宙(月末快照完整性) ----------
    # 与下载器 `_fetch_membership` 的逐月查询对齐:月末快照按月取,
    # end 所在月取该月完整月末快照(可晚于 backtest end,与下载器行为一致)。
    month_ends_all = month_end_dates(calendar)
    membership_end = max(d for d in month_ends_all if (d.year, d.month) <= (end.year, end.month))
    iw_frames = [read_bronze(t, archive_root, ["index_code", "trade_date", "con_code", "weight"]) for t in by_api("index_weight")]
    membership, anomalies = membership_from_index_weight(
        iw_frames, config.data.universe_index, membership_start, membership_end, calendar
    )
    expected_months = set()
    cursor = pd.Timestamp(membership_start.year, membership_start.month, 1)
    while cursor <= end:
        expected_months.add((cursor.year, cursor.month))
        cursor += pd.DateOffset(months=1)
    snapshot_months = {(d.year, d.month) for d in membership["date"]} if not membership.empty else set()
    missing_months = sorted(expected_months - snapshot_months)
    g5_ok = not anomalies and not missing_months and not membership.empty
    gates.append(GateResult(
        "G5_membership_integrity", g5_ok,
        f"月末快照 {len(snapshot_months)}/{len(expected_months)} 个月, 成分异常 {len(anomalies)}",
        {"missing_months": [f"{y}-{m:02d}" for y, m in missing_months], "anomalies": anomalies},
    ))
    members = set(membership["symbol"].astype(str)) if not membership.empty else set()

    # ---------- G6: 证券主表覆盖 ----------
    sb_frames = [read_bronze(t, archive_root) for t in by_api("stock_basic")]
    securities = build_securities(sb_frames) if sb_frames else pd.DataFrame()
    sec_symbols = set(securities["symbol"].astype(str)) if not securities.empty else set()
    missing_master = sorted(members - sec_symbols)
    gates.append(GateResult(
        "G6_securities_master", not missing_master and not securities.empty,
        f"证券主表 {len(sec_symbols)} 只, 成员缺失 {len(missing_master)}",
        {"missing_members": missing_master[:50]},
    ))

    # ---------- G7: 申万行业成员覆盖(历史区间) ----------
    ima_tasks = by_api("index_member_all")
    # 修复后权威分区为 64 叶子(32 申万 L1 × is_new Y/N),必须对全部 success
    # 叶子 concat;只读 tasks[0] 会漏掉绝大多数成员。
    ima = pd.concat(
        [read_bronze(t, archive_root) for t in ima_tasks], ignore_index=True
    ) if ima_tasks else pd.DataFrame()
    industry = build_industry_membership(ima, members) if not ima.empty else pd.DataFrame()
    ind_symbols = set(industry["symbol"].astype(str)) if not industry.empty else set()
    missing_industry = sorted(members - ind_symbols)
    # 行数恰整千的全表单拉任务按疑似截断处理(本轮实测: 3000 行恰整,106/342 成员缺失)
    cap_suspect = [
        {"api_name": t.api_name, "row_count": t.row_count, "task_id": t.task_id}
        for t in ima_tasks if t.row_count > 0 and t.row_count % 1000 == 0
    ]
    g7_ok = not missing_industry and not cap_suspect and not industry.empty
    gates.append(GateResult(
        "G7_industry_membership", g7_ok,
        f"行业成员 {len(ind_symbols)}/{len(members)} 只, 疑似截断任务 {len(cap_suspect)}",
        {
            "missing_members": missing_industry,
            "cap_suspect_tasks": cap_suspect,
            "note": "行数恰整千的全量任务按疑似截断处理;权威分区为 64 叶子(32 申万 L1 × is_new Y/N)全量 concat",
        },
    ))

    # ---------- G8: 更名记录完整性(is_st 语义) ----------
    nc_tasks = by_api("namechange")
    # 修复后权威分区为全宇宙 5,864 个逐 ts_code 任务,必须全部 concat。
    nc = pd.concat(
        [read_bronze(t, archive_root) for t in nc_tasks], ignore_index=True
    ) if nc_tasks else pd.DataFrame()
    nc_codes = set(nc["ts_code"].astype(str)) if not nc.empty else set()
    st_now = securities[
        securities["name"].astype(str).str.contains("ST|退", case=False, na=False)
    ]
    st_members = sorted(set(st_now["symbol"].astype(str)) & members)
    nc_st_codes = set(nc.loc[nc["name"].astype(str).str.contains("ST|退", case=False, na=False), "ts_code"].astype(str)) if not nc.empty else set()
    # 证据精化豁免:namechange 端点只记录更名事件,吸收合并退市不是更名。
    # 当前名带「退」但无 ST/退 更名记录的成员,若同时满足 (a) master 有
    # delist_date (b) 其全部更名记录已在并集,则「(退)」是退市标记而非历史名,
    # 豁免并全量记录(B0 修复验收门 9 同口径,如 601989.SH 中国重工)。
    delist_map = dict(zip(securities["symbol"].astype(str), securities["delist_date"])) if not securities.empty else {}
    st_without_record: list[str] = []
    st_exemptions: list[dict] = []
    for s in st_members:
        if s in nc_st_codes:
            continue
        code_records = nc.loc[nc["ts_code"].astype(str) == s] if not nc.empty else pd.DataFrame()
        has_delist = pd.notna(delist_map.get(s)) and str(delist_map.get(s)) not in ("", "None")
        if has_delist and not code_records.empty:
            st_exemptions.append({
                "symbol": s,
                "delist_date": str(delist_map.get(s)),
                "union_records": int(len(code_records)),
                "rule": "退市非更名:delist_date 存在且全量更名记录已在并集",
            })
        else:
            st_without_record.append(s)
    nc_cap_suspect = [
        {"api_name": t.api_name, "row_count": t.row_count, "task_id": t.task_id}
        for t in nc_tasks if t.row_count > 0 and t.row_count % 1000 == 0
    ]
    g8_ok = not st_without_record and not nc_cap_suspect and not nc.empty
    gates.append(GateResult(
        "G8_namechange_integrity", g8_ok,
        f"更名记录 {len(nc)} 行覆盖 {len(nc_codes)} 只; 当前 ST 成员缺记录 {len(st_without_record)}; "
        f"退市标记豁免 {len(st_exemptions)}; 疑似截断 {len(nc_cap_suspect)}",
        {
            "st_members_without_record": st_without_record,
            "st_delist_exemptions": st_exemptions,
            "cap_suspect_tasks": nc_cap_suspect,
            "note": "行数恰整千的全量任务按疑似截断处理;退市标记豁免须有 delist_date 且全量更名记录在并集",
        },
    ))

    # ---------- 门禁汇总 ----------
    failed = [g for g in gates if not g.ok]
    report = {
        "generated_at_utc": utc_now_iso(),
        "mapping_version": MAPPING_VERSION,
        "archive_root": str(archive_root),
        "catalog": str(catalog),
        "catalog_sha256": catalog_sha,
        "window": {"warmup_start": _date_str(warmup_start), "end": _date_str(end), "membership_start": _date_str(membership_start)},
        "gates": [
            {"gate": g.gate, "ok": g.ok, "detail": g.detail, "payload": g.payload}
            for g in gates
        ],
        "duplicate_generations_dropped": len(selection.dropped_duplicates),
        "verdict": "pass" if not failed else "blocked",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for g in gates:
        print(f"[{'PASS' if g.ok else 'FAIL'}] {g.gate}: {g.detail}")
    if failed:
        print(f"\n⛔ {len(failed)} 个前置门禁失败,按契约停止构建。缺口报告: {report_path}")
        return 2

    # ---------- 构建 v4 缓存 ----------
    if output.exists() and any(output.iterdir()) and not args.force:
        print(f"输出目录非空, 用 --force 覆盖: {output}")
        return 2
    output.mkdir(parents=True, exist_ok=True)

    inputs_used: list[dict] = []

    def track(tasks_used):
        for t in tasks_used:
            inputs_used.append({
                "task_id": t.task_id, "api_name": t.api_name, "snapshot": t.snapshot,
                "params": t.params, "row_count": t.row_count,
                "bronze_relative_path": t.bronze_path.split("data_lake/")[-1],
            })

    track(by_api("trade_cal"))
    track(by_api("index_weight"))
    track(by_api("stock_basic"))
    track(by_api("index_member_all"))

    write_deterministic_csv_gz(membership, output / "membership.csv.gz")
    write_deterministic_csv_gz(securities, output / "securities.csv.gz")
    write_deterministic_csv_gz(industry, output / "industry_membership.csv.gz")
    write_deterministic_csv_gz(pd.DataFrame({"date": [_date_str(d) for d in window_days]}), output / "calendar.csv.gz")

    # 逐日行情拼装(窗口内全部交易日,四表按日对齐)
    member_list = sorted(members)
    daily_tasks = {t.params["trade_date"]: t for t in by_api("daily") if t.params.get("trade_date")}
    adj_tasks = {t.params["trade_date"]: t for t in by_api("adj_factor") if t.params.get("trade_date")}
    basic_tasks = {t.params["trade_date"]: t for t in by_api("daily_basic") if t.params.get("trade_date")}
    limit_tasks = {t.params["trade_date"]: t for t in by_api("stk_limit") if t.params.get("trade_date")}
    per_symbol: dict[str, dict[str, list[pd.DataFrame]]] = {
        s: {"daily": [], "adj": [], "basic": [], "limit": []} for s in member_list
    }
    dates_sorted = sorted(
        d for d in daily_dates if warmup_start <= pd.Timestamp(d) <= end
    )
    for i, d in enumerate(dates_sorted, 1):
        day_daily = read_bronze(daily_tasks[d], archive_root)
        day_adj = read_bronze(adj_tasks[d], archive_root, ["ts_code", "trade_date", "adj_factor"])
        day_basic = read_bronze(basic_tasks[d], archive_root, ["ts_code", "trade_date", "total_mv", "circ_mv"])
        day_limit = read_bronze(limit_tasks[d], archive_root, ["ts_code", "trade_date", "up_limit", "down_limit"]) if d in limit_tasks else None
        for frame, key in ((day_daily, "daily"), (day_adj, "adj"), (day_basic, "basic")):
            sub = frame[frame["ts_code"].astype(str).isin(members)]
            for symbol, group in sub.groupby("ts_code"):
                per_symbol[str(symbol)][key].append(group)
        if day_limit is not None:
            sub = day_limit[day_limit["ts_code"].astype(str).isin(members)]
            for symbol, group in sub.groupby("ts_code"):
                per_symbol[str(symbol)]["limit"].append(group)
        if i % 100 == 0:
            print(f"行情拼装 {i}/{len(dates_sorted)} 交易日")
    for api in ("daily", "adj_factor", "daily_basic", "stk_limit"):
        track(by_api(api))

    names_by_symbol = {s: g for s, g in nc.groupby("ts_code")} if not nc.empty else {}
    track(by_api("namechange"))
    bar_dir = output / "bars"
    for i, symbol in enumerate(member_list, 1):
        parts = per_symbol[symbol]
        if not parts["daily"]:
            print(f"⚠ {symbol} 窗口内无行情, 跳过(与下载器语义一致)")
            continue
        daily = pd.concat(parts["daily"], ignore_index=True)
        adj = pd.concat(parts["adj"], ignore_index=True) if parts["adj"] else pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
        basic = pd.concat(parts["basic"], ignore_index=True) if parts["basic"] else pd.DataFrame(columns=["ts_code", "trade_date", "total_mv", "circ_mv"])
        limit = pd.concat(parts["limit"], ignore_index=True) if parts["limit"] else pd.DataFrame(columns=["ts_code", "trade_date", "up_limit", "down_limit"])
        names = names_by_symbol.get(symbol, pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]))
        frame = build_bars_for_symbol(symbol, daily, adj, limit, basic, names)
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        write_deterministic_csv_gz(frame, bar_dir / f"{symbol.replace('.', '_')}.csv.gz")
        if i % 50 == 0:
            print(f"bars 写出 {i}/{len(member_list)}")

    # 公司行动
    div_tasks = [t for t in by_api("dividend") if str(t.params.get("ts_code", "")) in members]
    div_frames = [read_bronze(t, archive_root) for t in div_tasks]
    actions = build_corporate_actions(div_frames, members, warmup_start, end)
    track(div_tasks)
    write_deterministic_csv_gz(actions, output / "corporate_actions.csv.gz")

    # 基准与择时指数
    id_tasks = by_api("index_daily")
    track(id_tasks)
    id_frames = [read_bronze(t, archive_root) for t in id_tasks]
    benchmark = build_index_frame(id_frames, config.data.benchmark_index)
    benchmark = benchmark[(benchmark["date"] >= warmup_start) & (benchmark["date"] <= end)]
    regime = build_index_frame(id_frames, config.data.regime_index)
    regime = regime[(regime["date"] >= warmup_start) & (regime["date"] <= end)]
    for name, frame in (("benchmark", benchmark), ("regime", regime)):
        if frame.empty:
            print(f"⛔ {name} 指数行情为空, 停止")
            return 2
        out = frame.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        write_deterministic_csv_gz(out, output / f"{name}.csv.gz")

    # ---------- provenance 与 manifest ----------
    for item in inputs_used:
        path = archive_root / item["bronze_relative_path"]
        item["bronze_sha256"] = sha256_file(path)
    inputs_doc = {
        "mapping_version": MAPPING_VERSION,
        "generated_at_utc": utc_now_iso(),
        "catalog_sha256": catalog_sha,
        "acceptance_evidence": {
            "B1": {"path": B1_ACCEPTANCE, "sha256": sha256_file(archive_root / B1_ACCEPTANCE)},
            "B2": {"path": B2_ACCEPTANCE, "sha256": sha256_file(archive_root / B2_ACCEPTANCE)},
        },
        "selected_task_set_sha256": sha256_payload(sorted(item["task_id"] for item in inputs_used)),
        "inputs": sorted(inputs_used, key=lambda x: x["task_id"]),
    }
    inputs_path = output / "archive_inputs_provenance.json"
    inputs_path.write_text(json.dumps(inputs_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cache_inputs = [
        output / "membership.csv.gz",
        output / "securities.csv.gz",
        output / "industry_membership.csv.gz",
        output / "corporate_actions.csv.gz",
        output / "benchmark.csv.gz",
        output / "regime.csv.gz",
        output / "calendar.csv.gz",
        *sorted(bar_dir.glob("*.csv.gz")),
    ]
    files = build_file_inventory(output, cache_inputs)
    manifest = {
        "schema_version": 4,
        # v4 身份契约字段:与 config.data.provider 保持一致(绑定语义不变);
        # 真实来源在 archive_provenance 段完整披露。
        "provider": config.data.provider,
        "universe_index": config.data.universe_index,
        "regime_index": config.data.regime_index,
        "benchmark_index": config.data.benchmark_index,
        "industry_standard": config.data.industry_standard,
        "industry_level": config.data.industry_level,
        "requested_start": _date_str(warmup_start),
        "requested_end": _date_str(end),
        "created_at_utc": utc_now_iso(),
        "symbols": len(member_list),
        "files": files,
        "data_fingerprint_sha256": inventory_sha256(files),
        "archive_provenance": {
            "mapping_version": MAPPING_VERSION,
            "catalog_sha256": catalog_sha,
            "acceptance_evidence": inputs_doc["acceptance_evidence"],
            "selected_task_set_sha256": inputs_doc["selected_task_set_sha256"],
            "inputs_provenance_file": "archive_inputs_provenance.json",
            "inputs_provenance_sha256": sha256_file(inputs_path),
            "generated_at_utc": inputs_doc["generated_at_utc"],
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ---------- 真实消费者回读验证 ----------
    bundle = MarketDataBundle.from_cache(output, strict=True, expected_config=config)
    print(
        f"\n✅ v4 基础行情缓存构建并回读验证通过: {len(member_list)} 只历史成分, "
        f"bars {len(bundle.bars):,} 行, 指纹 {manifest['data_fingerprint_sha256']}"
    )
    print(f"manifest: {output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
