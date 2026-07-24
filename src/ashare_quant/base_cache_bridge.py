"""archive → v4 基础行情缓存兼容层构建器。

把 P0 全量归档(B0/B1/B2 Bronze,只读)离线转换为研究侧 v4 基础行情缓存
(`data/cache/manifest.json` + 各 csv.gz),供 strict `pit-lake-build` 的
`_verify_base_market_cache` 绑定。不访问网络、不改写归档任何字节。

设计契约(2026-07-23 用户指令):
1. 不改 strict PIT 对 `data/cache/manifest.json` 的绑定契约(v4 schema、
   身份字段、requested 区间、文件清单指纹语义全部沿用 data.py 原定义)。
2. B1 Bronze 提供 daily/adj_factor/stk_limit/daily_basic;B2 Bronze 提供
   历史 index_weight/index_daily;B0 提供 trade_cal/stock_basic/namechange/
   index_member_all/index_classify。成分宇宙按**生效日期**构建 PIT
   membership,禁止用当前成分回填历史。
3. manifest 扩展 `archive_provenance` 段:catalog 快照 SHA、B1/B2 验收
   证据 SHA、任务集合 SHA、输入文件清单 SHA、映射版本、生成时间。
4. 只消费 `status='success'` 的任务;quarantined/orphaned/aborted/
   superseded 及 research_eligible=false 数据天然排除。同一逻辑分区存在
   多代 success(修复重抓)时按快照代际优先级确定性去重,全部决策记录到
   provenance。
5. 确定性:所有 csv.gz 以 gzip mtime=0 写入、行序固定,同输入重跑得到
   相同 data_fingerprint_sha256。
6. 前置门禁(G1..G7)任一失败即停止并输出机器可读缺口报告,禁止静默
   使用 fixture、当前成分或重新下载的数据凑数。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .data import ACTION_COLUMNS, INDUSTRY_COLUMNS, SECURITY_COLUMNS
from .pit_lake import remap_archive_path

MAPPING_VERSION = "archive-to-v4-base-cache/1.0.0"

# 快照代际优先级(数值越小越优先):修复代 > 主批次代 > Phase A 样例代。
# 修复代在同一逻辑分区上以更准确的网关认知(fields="" 全粒度、不重叠拆分)
# 重抓;同类型内按快照串(内嵌时间戳)新者优先。
_SNAPSHOT_RANK = (
    ("p0_B2_repair", 0),
    ("p0_B1_market", 1),
    ("p0_B0_reference", 1),
    ("p0_B2_universe", 2),
    ("phase_a", 9),
)

# gzip 确定性:mtime 固定为 0,同字节输入产出同字节文件。
_GZIP_MTIME = 0


@dataclass(frozen=True)
class BronzeTask:
    task_id: str
    api_name: str
    params: dict[str, Any]
    row_count: int
    bronze_path: str
    raw_sha256: str
    snapshot: str


@dataclass
class GateResult:
    gate: str
    ok: bool
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionResult:
    tasks: dict[tuple[str, str], BronzeTask]  # (api, canonical_params) -> task
    dropped_duplicates: list[dict[str, Any]]


def canonical_params(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def snapshot_of(bronze_path: str) -> str:
    name = Path(bronze_path).name
    for marker in ("_p0_", "_phase_a"):
        idx = name.find(marker)
        if idx >= 0:
            return name[idx + 1 :].removesuffix(".parquet")
    return ""


def _snapshot_priority(snapshot: str) -> tuple[int, str]:
    for prefix, rank in _SNAPSHOT_RANK:
        if snapshot.startswith(prefix):
            # 同 rank 内快照串大者(时间戳新)优先:返回反向排序键
            return (rank, "".join(chr(255 - ord(c)) for c in snapshot))
    return (5, "".join(chr(255 - ord(c)) for c in snapshot))


def load_success_tasks(catalog_path: Path, api_names: set[str]) -> list[BronzeTask]:
    """只读加载 success 任务;其他状态天然排除(研究排除契约)。"""
    connection = sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT task_id, api_name, params_json, row_count, bronze_path, "
            "raw_sha256 FROM archive_tasks WHERE status = 'success'"
        ).fetchall()
    finally:
        connection.close()
    tasks: list[BronzeTask] = []
    for row in rows:
        api_name = str(row["api_name"])
        if api_name not in api_names:
            continue
        tasks.append(
            BronzeTask(
                task_id=str(row["task_id"]),
                api_name=api_name,
                params=json.loads(row["params_json"] or "{}"),
                row_count=int(row["row_count"]),
                bronze_path=str(row["bronze_path"] or ""),
                raw_sha256=str(row["raw_sha256"] or ""),
                snapshot=snapshot_of(str(row["bronze_path"] or "")),
            )
        )
    return tasks


def select_generations(tasks: list[BronzeTask]) -> SelectionResult:
    """同一逻辑分区多代 success 共存时按快照代际确定性去重。

    被丢弃的重复代(含封存前截断但未被迁移的遗留 success)全部记录,
    供 provenance 与缺口报告审计。
    """
    grouped: dict[tuple[str, str], list[BronzeTask]] = {}
    for task in tasks:
        key = (task.api_name, canonical_params(task.params))
        grouped.setdefault(key, []).append(task)
    chosen: dict[tuple[str, str], BronzeTask] = {}
    dropped: list[dict[str, Any]] = []
    for key, group in grouped.items():
        group.sort(key=lambda t: (_snapshot_priority(t.snapshot), t.task_id))
        chosen[key] = group[0]
        for loser in group[1:]:
            dropped.append(
                {
                    "api_name": loser.api_name,
                    "params": loser.params,
                    "task_id": loser.task_id,
                    "snapshot": loser.snapshot,
                    "row_count": loser.row_count,
                    "kept_snapshot": group[0].snapshot,
                    "kept_task_id": group[0].task_id,
                }
            )
    return SelectionResult(tasks=chosen, dropped_duplicates=dropped)


def read_bronze(task: BronzeTask, archive_root: Path, columns: list[str] | None = None) -> pd.DataFrame:
    path, _ = remap_archive_path(task.bronze_path, archive_root)
    if not path.is_file():
        raise FileNotFoundError(f"Bronze 分区缺失: {path}")
    return pd.read_parquet(path, columns=columns)


def write_deterministic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    """gzip mtime=0 + 固定行序的确定性 csv.gz(重跑同指纹的前置条件)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw_fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, mtime=_GZIP_MTIME) as fh:
            fh.write(payload)
    temporary.replace(path)


def month_end_dates(calendar: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """每个自然月的最后一个交易日(复刻下载器默认 fields 的月末快照语义)。"""
    series = pd.Series(calendar)
    grouped = series.groupby([series.dt.year, series.dt.month]).max()
    return set(pd.DatetimeIndex(grouped))


def membership_from_index_weight(
    frames: list[pd.DataFrame],
    universe_index: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    min_constituents: int = 250,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """按生效日期构建 PIT 成分宇宙(禁止当前成分回填)。

    只保留每月最后一个交易日的快照(与下载器默认 fields 的月末口径一致),
    并对每个快照做完整性检查:成分数异常(低于 min_constituents 或低于
    相邻快照中位数的 95%)即判定该快照不完整,由调用方 fail-closed。
    """
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "index_weight"]), []
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[raw["index_code"].astype(str) == universe_index].copy()
    raw["date"] = pd.to_datetime(raw["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    raw = raw.loc[(raw["date"] >= window_start) & (raw["date"] <= window_end)]
    ends = month_end_dates(calendar)
    raw = raw.loc[raw["date"].isin(ends)]
    anomalies: list[dict[str, Any]] = []
    counts = raw.groupby("date")["con_code"].nunique()
    if not counts.empty:
        median = counts.median()
        for date, count in counts.items():
            if count < min_constituents or count < median * 0.95:
                anomalies.append(
                    {"date": str(date.date()), "constituents": int(count), "median": float(median)}
                )
    result = raw.rename(columns={"con_code": "symbol", "weight": "index_weight"})[
        ["date", "symbol", "index_weight"]
    ]
    result = result.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"])
    return result.reset_index(drop=True), anomalies


def fallback_limit_rate(symbol: str, is_st: bool) -> float:
    """与 data.py 下载器完全一致的涨跌停回退速率。"""
    code, _, exchange = symbol.partition(".")
    if exchange == "BJ" or code.startswith(("4", "8", "92")):
        return 0.30
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if is_st:
        return 0.05
    return 0.10


def historical_names(dates: pd.Series, names: pd.DataFrame) -> pd.Series:
    """按更名区间回填历史名称(与 data.py `_historical_names` 同语义)。"""
    result = pd.Series("", index=dates.index, dtype="object")
    if names.empty:
        return result
    for row in names.itertuples(index=False):
        start_raw = getattr(row, "start_date", None)
        end_raw = getattr(row, "end_date", None)
        # 注意:不能写 pd.Timestamp.min.normalize()——min 是 1677-09-21 00:12:43,
        # normalize 到当日 00:00 会低于 datetime64[ns] 下界而溢出回绕到 2262 年,
        # 导致 between 恒为 False(start_date 为空的"自始有效"记录被静默丢弃,
        # 这是 data.py `_historical_names` 的潜在 bug,已在验收报告中披露)。
        start = pd.Timestamp.min if pd.isna(start_raw) else pd.to_datetime(str(start_raw))
        end = pd.Timestamp.max if pd.isna(end_raw) else pd.to_datetime(str(end_raw))
        mask = dates.between(start, end)
        result.loc[mask] = str(getattr(row, "name", ""))
    return result


def build_bars_for_symbol(
    symbol: str,
    daily: pd.DataFrame,
    adj_factor: pd.DataFrame,
    stk_limit: pd.DataFrame,
    daily_basic: pd.DataFrame,
    names: pd.DataFrame,
) -> pd.DataFrame:
    """单符号 v4 bars 行集,变换口径与 data.py `_fetch_symbol` 完全一致。"""
    frame = daily.merge(adj_factor, on=["ts_code", "trade_date"], how="left")
    if frame["adj_factor"].isna().any():
        missing = frame.loc[frame["adj_factor"].isna(), "trade_date"].astype(str).tolist()
        raise ValueError(f"{symbol} 缺少 {len(missing)} 个交易日的复权因子,首个 {missing[0]}")
    frame = frame.merge(stk_limit, on=["ts_code", "trade_date"], how="left")
    frame = frame.merge(daily_basic, on=["ts_code", "trade_date"], how="left")
    if frame[["total_mv", "circ_mv"]].isna().any().any():
        count = int(frame[["total_mv", "circ_mv"]].isna().any(axis=1).sum())
        raise ValueError(f"{symbol} 缺少 {count} 个交易日的市值数据")
    frame = frame.rename(
        columns={
            "ts_code": "symbol",
            "trade_date": "date",
            "pre_close": "prev_close",
            "vol": "volume",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d")
    frame = frame.sort_values("date")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0) * 100.0
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0) * 1000.0
    frame["name"] = historical_names(frame["date"], names)
    frame["is_st"] = frame["name"].fillna("").str.contains(r"ST|退", case=False, regex=True)
    rates = frame["is_st"].map(lambda flag: fallback_limit_rate(symbol, bool(flag)))
    missing_up = frame["up_limit"].isna()
    missing_down = frame["down_limit"].isna()
    frame.loc[missing_up, "up_limit"] = (
        frame.loc[missing_up, "prev_close"].astype(float) * (1.0 + rates[missing_up])
    ).round(2)
    frame.loc[missing_down, "down_limit"] = (
        frame.loc[missing_down, "prev_close"].astype(float) * (1.0 - rates[missing_down])
    ).round(2)
    return frame[
        [
            "date", "symbol", "name", "open", "high", "low", "close", "prev_close",
            "volume", "amount", "adj_factor", "up_limit", "down_limit",
            "is_st", "total_mv", "circ_mv",
        ]
    ].reset_index(drop=True)


def build_corporate_actions(dividend_frames: list[pd.DataFrame], members: set[str],
                            start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """与 data.py `_fetch_actions` 同语义:仅 div_proc=实施,窗口内除权日。"""
    if not dividend_frames:
        return pd.DataFrame(columns=ACTION_COLUMNS)
    raw = pd.concat(dividend_frames, ignore_index=True)
    raw = raw.loc[raw["div_proc"].astype(str) == "实施"].copy()
    raw["ex_date"] = pd.to_datetime(raw["ex_date"].astype(str), format="%Y%m%d", errors="coerce")
    raw = raw.loc[raw["ex_date"].between(start, end)]
    if raw.empty:
        return pd.DataFrame(columns=ACTION_COLUMNS)
    raw = raw.loc[raw["ts_code"].astype(str).isin(members)]
    raw = raw.rename(
        columns={
            "ts_code": "symbol",
            "div_listdate": "stock_list_date",
            "cash_div": "cash_dividend",
            "stk_div": "stock_dividend",
        }
    )
    for column in ["record_date", "pay_date", "stock_list_date"]:
        raw[column] = pd.to_datetime(raw[column].astype(str), format="%Y%m%d", errors="coerce")
    for column in ["cash_dividend", "stock_dividend"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0.0)
    result = raw[ACTION_COLUMNS].sort_values(["ex_date", "symbol"]).drop_duplicates(
        ["symbol", "ex_date"], keep="last"
    )
    return result.reset_index(drop=True)


def build_securities(frames: list[pd.DataFrame]) -> pd.DataFrame:
    raw = pd.concat(frames, ignore_index=True)
    # Bronze 同时带 ts_code(000001.SZ)与 symbol(000001,6 位纯数字)两列;
    # v4 契约的 symbol 是带后缀的 ts_code,必须显式用 ts_code 覆盖,不能改名复用
    # (改名会撞已存在的 6 位 symbol 列,直接把 6 位码当 symbol 会让成员匹配全灭)。
    raw = raw.drop(columns=["symbol"], errors="ignore").rename(columns={"ts_code": "symbol"})
    for column in SECURITY_COLUMNS:
        if column not in raw.columns:
            raw[column] = pd.NA
    return raw[SECURITY_COLUMNS].drop_duplicates("symbol", keep="last").reset_index(drop=True)


def build_industry_membership(member_all: pd.DataFrame, members: set[str]) -> pd.DataFrame:
    raw = member_all.loc[member_all["ts_code"].astype(str).isin(members)].copy()
    if "out_date" not in raw.columns:
        raw["out_date"] = pd.NA
    result = raw.rename(
        columns={"ts_code": "symbol", "l1_code": "industry_code", "l1_name": "industry_name"}
    )
    result = result[INDUSTRY_COLUMNS].drop_duplicates()
    # is_new=Y/N 双叶子会对同一 (symbol, industry_code, in_date) 各给一行
    # (一行 out_date 为空、一行 out_date 非空)。按 B0 修复验收口径
    # 「历史有效日期(out_date 非空)优先」裁决,与交付的去重并集一致。
    result["_out_filled"] = result["out_date"].notna() & (result["out_date"].astype(str) != "")
    result = result.sort_values("_out_filled", ascending=False)
    result = result.drop_duplicates(["symbol", "industry_code", "in_date"], keep="first")
    result = result.drop(columns=["_out_filled"])
    return result.sort_values(["symbol", "in_date"]).reset_index(drop=True)


def build_index_frame(frames: list[pd.DataFrame], ts_code: str) -> pd.DataFrame:
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[raw["ts_code"].astype(str) == ts_code].copy()
    raw = raw.rename(columns={"trade_date": "date", "pre_close": "prev_close"})
    raw["date"] = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d")
    return raw.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
