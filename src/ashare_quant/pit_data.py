from __future__ import annotations

import json
import gzip
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .config import AppConfig, PointInTimeDataConfig
from .data import CACHE_SCHEMA_VERSION, MarketDataBundle, RateLimitedTushare
from .provenance import (
    build_file_inventory,
    inventory_sha256,
    payload_sha256,
    sha256_file,
    verify_file_inventory,
)


LOGGER = logging.getLogger(__name__)
PIT_CACHE_SCHEMA_VERSION = 1
CANONICAL_STATEMENT_REPORT_TYPE = "1"

FUNDAMENTAL_COLUMNS = [
    "symbol",
    "source",
    "metric",
    "unit",
    "period_end",
    "announcement_date",
    "available_date",
    "revision_sequence",
    "report_type",
    "company_type",
    "source_update_flag",
    "source_row_sha256",
    "value",
]

VALUATION_COLUMNS = [
    "date",
    "available_date",
    "symbol",
    "turnover_rate_pct",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dividend_yield_ttm_pct",
    "total_market_value_10k_cny",
    "float_market_value_10k_cny",
]

VALUATION_VALUE_COLUMNS = VALUATION_COLUMNS[3:]

# Provider fields remain isolated here. The canonical metric names and units are
# stable even if an adapter has to change later.
FUNDAMENTAL_SOURCE_SPECS: dict[str, dict[str, tuple[str, str]]] = {
    "income": {
        "basic_eps": ("basic_eps", "CNY_per_share"),
        "total_revenue": ("total_revenue", "CNY"),
        "revenue": ("revenue", "CNY"),
        "operate_profit": ("operating_profit", "CNY"),
        "total_profit": ("total_profit", "CNY"),
        "n_income": ("net_income", "CNY"),
        "n_income_attr_p": ("net_income_parent", "CNY"),
    },
    "balancesheet": {
        "money_cap": ("cash", "CNY"),
        "accounts_receiv": ("accounts_receivable", "CNY"),
        "inventories": ("inventory", "CNY"),
        "total_cur_assets": ("current_assets", "CNY"),
        "fix_assets": ("fixed_assets", "CNY"),
        "total_assets": ("total_assets", "CNY"),
        "total_cur_liab": ("current_liabilities", "CNY"),
        "total_liab": ("total_liabilities", "CNY"),
        "total_hldr_eqy_exc_min_int": ("equity_parent", "CNY"),
    },
    "cashflow": {
        "n_cashflow_act": ("operating_cash_flow", "CNY"),
        "n_cashflow_inv_act": ("investing_cash_flow", "CNY"),
        "n_cash_flows_fnc_act": ("financing_cash_flow", "CNY"),
        "c_pay_acq_const_fiolta": ("capital_expenditure", "CNY"),
    },
    "fina_indicator": {
        "roe": ("roe_pct", "percent"),
        "roa": ("roa_pct", "percent"),
        "grossprofit_margin": ("gross_margin_pct", "percent"),
        "netprofit_margin": ("net_margin_pct", "percent"),
        "debt_to_assets": ("debt_to_assets_pct", "percent"),
        "current_ratio": ("current_ratio", "ratio"),
        "quick_ratio": ("quick_ratio", "ratio"),
        "assets_turn": ("asset_turnover", "ratio"),
        "ocf_to_or": ("operating_cash_to_revenue_pct", "percent"),
        "ocf_to_opincome": (
            "operating_cash_to_operating_profit_pct",
            "percent",
        ),
        "profit_dedt": ("deducted_net_income", "CNY"),
        "or_yoy": ("revenue_yoy_pct", "percent"),
        "netprofit_yoy": ("net_income_yoy_pct", "percent"),
    },
}

# Tushare statement endpoints do not expose an identical metadata schema.
# Request only fields documented by each endpoint and let the normalizer fill
# canonical optional metadata with empty strings where a source has none.
FUNDAMENTAL_SOURCE_FIELDS: dict[str, tuple[str, ...]] = {
    "income": (
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "update_flag",
    ),
    "balancesheet": (
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "update_flag",
    ),
    "cashflow": (
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "update_flag",
    ),
    "fina_indicator": ("ts_code", "ann_date", "end_date", "update_flag"),
}

_SOURCE_ID_FIELDS = [
    "ts_code",
    "ann_date",
    "f_ann_date",
    "end_date",
    "report_type",
    "comp_type",
    "update_flag",
]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_partitioned_csv(
    paths: Iterable[Path], columns: Iterable[str]
) -> pd.DataFrame:
    """Read canonical partitions without concatenating all-NA empty frames."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True)


def _read_symbol_partitions(
    paths: Iterable[Path],
    symbols: Iterable[str],
    columns: Iterable[str],
) -> pd.DataFrame:
    expected = {
        str(symbol).replace(".", "_") + ".csv.gz": str(symbol)
        for symbol in symbols
    }
    path_list = list(paths)
    actual_names = {path.name for path in path_list}
    if actual_names != set(expected):
        raise ValueError(
            "PIT 证券分区与 manifest 不一致: "
            f"缺少={sorted(set(expected) - actual_names)[:5]}，"
            f"多出={sorted(actual_names - set(expected))[:5]}"
        )
    frames: list[pd.DataFrame] = []
    for path in path_list:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        if "symbol" not in frame:
            raise ValueError(f"PIT 分区缺少 symbol 字段: {path.name}")
        actual_symbols = set(frame["symbol"].dropna().astype(str))
        if actual_symbols != {expected[path.name]}:
            raise ValueError(
                f"PIT 分区证券身份错误: {path.name}={sorted(actual_symbols)}"
            )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True)


def require_pit_research_eligible(manifest: Mapping[str, Any]) -> None:
    """Reject engineering fixtures while preserving legacy/direct PIT caches."""
    if manifest.get("research_eligible") is False:
        mode = manifest.get("archive_bridge", {}).get("mode", "fixture")
        raise ValueError(
            "PIT 缓存仅用于工程复现，不能生成 Alpha 证据: "
            f"archive_bridge.mode={mode}"
        )


def _verify_base_market_cache(
    manifest_path: Path, config: AppConfig
) -> set[str]:
    """Verify the linked v4 inventory and return its historical members."""
    expected_manifest = (
        Path(config.data.cache_dir).resolve() / "manifest.json"
    )
    if manifest_path != expected_manifest:
        raise ValueError(
            "PIT 基础行情 manifest 路径与 data.cache_dir 不一致"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("基础行情 manifest.json 不是有效 JSON") from exc
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"基础行情缓存必须是 v{CACHE_SCHEMA_VERSION}，"
            f"实际为 {payload.get('schema_version')!r}"
        )
    expected_identity = {
        "provider": config.data.provider,
        "universe_index": config.data.universe_index,
        "regime_index": config.data.regime_index,
        "benchmark_index": config.data.benchmark_index,
        "industry_standard": config.data.industry_standard,
        "industry_level": config.data.industry_level,
    }
    mismatches = {
        name: {"cache": payload.get(name), "config": value}
        for name, value in expected_identity.items()
        if payload.get(name) != value
    }
    if mismatches:
        raise ValueError(f"基础行情缓存身份与配置不一致: {mismatches}")
    required_start = pd.Timestamp(config.backtest.start_date) - pd.Timedelta(
        config.data.warmup_calendar_days, unit="D"
    )
    required_end = pd.Timestamp(config.backtest.end_date)
    try:
        cached_start = pd.Timestamp(payload["requested_start"])
        cached_end = pd.Timestamp(payload["requested_end"])
    except (KeyError, ValueError) as exc:
        raise ValueError("基础行情 manifest 缺少有效请求日期范围") from exc
    if cached_start > required_start or cached_end < required_end:
        raise ValueError(
            "基础行情缓存日期范围不足: "
            f"缓存={cached_start.date()}..{cached_end.date()}，"
            f"需要={required_start.date()}..{required_end.date()}"
        )
    verification = verify_file_inventory(
        manifest_path.parent, payload.get("files", [])
    )
    if verification["inventory_sha256"] != payload.get(
        "data_fingerprint_sha256"
    ):
        raise ValueError("基础行情文件清单指纹与 manifest 不一致")
    membership_path = manifest_path.parent / "membership.csv.gz"
    recorded = {
        str(item.get("path"))
        for item in payload.get("files", [])
        if isinstance(item, Mapping)
    }
    required_files = {
        "membership.csv.gz",
        "benchmark.csv.gz",
        "regime.csv.gz",
        "corporate_actions.csv.gz",
        "securities.csv.gz",
        "industry_membership.csv.gz",
        "calendar.csv.gz",
    }
    missing_files = required_files.difference(recorded)
    has_bars = any(
        path.startswith("bars/") and path.endswith(".csv.gz")
        for path in recorded
    )
    if missing_files or not has_bars or not membership_path.is_file():
        raise ValueError(
            "基础行情封存集合不完整: "
            f"缺少={sorted(missing_files)}，bars={has_bars}"
        )
    membership = pd.read_csv(membership_path, usecols=["symbol"])
    symbols = set(membership["symbol"].dropna().astype(str))
    if not symbols:
        raise ValueError("基础行情历史成分证券集合为空")
    return symbols


def _provider_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(text, errors="coerce", format="mixed").dt.normalize()


def availability_dates(
    source_dates: Iterable[pd.Timestamp | str | None],
    calendar: Iterable[pd.Timestamp | str],
    lag_trading_days: int,
) -> pd.Series:
    """Map observations to their first conservatively usable trading date.

    A zero-lag close-derived observation can be used on the same trading date.
    A fundamental announcement with lag one becomes visible on the first trading
    date strictly after its announcement. Larger lags move forward from there.
    """
    if lag_trading_days < 0:
        raise ValueError("lag_trading_days 不能为负")
    trading_days = pd.DatetimeIndex(pd.to_datetime(list(calendar))).normalize()
    trading_days = trading_days.unique().sort_values()
    if trading_days.empty:
        raise ValueError("可见日期换算需要非空交易日历")

    result: list[pd.Timestamp | pd.NaT] = []
    for value in source_dates:
        stamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(stamp):
            result.append(pd.NaT)
            continue
        stamp = pd.Timestamp(stamp).normalize()
        if lag_trading_days == 0:
            location = int(trading_days.searchsorted(stamp, side="left"))
        else:
            location = int(trading_days.searchsorted(stamp, side="right"))
            location += lag_trading_days - 1
        result.append(
            trading_days[location] if location < len(trading_days) else pd.NaT
        )
    return pd.Series(result, dtype="datetime64[ns]")


def _effective_announcement_dates(frame: pd.DataFrame) -> pd.Series:
    candidates = pd.DataFrame(index=frame.index)
    for column in ["ann_date", "f_ann_date"]:
        candidates[column] = (
            _provider_dates(frame[column])
            if column in frame
            else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        )
    # Using the later provider date is deliberately conservative when scheduled
    # and actual announcement fields disagree.
    return candidates.max(axis=1)


def _row_sha256(row: pd.Series, source: str, fields: Iterable[str]) -> str:
    payload: dict[str, Any] = {"source": source}
    for field_name in fields:
        value = row.get(field_name)
        payload[field_name] = None if pd.isna(value) else str(value)
    return payload_sha256(payload)


def normalize_fundamental_source(
    source: str,
    raw: pd.DataFrame,
    calendar: Iterable[pd.Timestamp | str],
    lag_trading_days: int = 1,
) -> pd.DataFrame:
    """Normalize one provider statement table into immutable long-form records."""
    if source not in FUNDAMENTAL_SOURCE_SPECS:
        raise ValueError(f"未知财报来源: {source}")
    if raw.empty:
        return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
    required = {"ts_code", "end_date"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{source} 缺少字段: {sorted(missing)}")

    frame = raw.copy()
    frame["symbol"] = frame["ts_code"].astype(str)
    frame["period_end"] = _provider_dates(frame["end_date"])
    frame["announcement_date"] = _effective_announcement_dates(frame)
    frame["available_date"] = availability_dates(
        frame["announcement_date"], calendar, lag_trading_days
    ).to_numpy()
    frame["report_type"] = (
        frame.get("report_type", pd.Series("", index=frame.index))
        .astype("string")
        .fillna("")
        .astype(str)
    )
    frame["company_type"] = (
        frame.get("comp_type", pd.Series("", index=frame.index))
        .astype("string")
        .fillna("")
        .astype(str)
    )
    frame["source_update_flag"] = (
        frame.get("update_flag", pd.Series("", index=frame.index))
        .astype("string")
        .fillna("")
        .astype(str)
    )
    identity_fields = [
        *_SOURCE_ID_FIELDS,
        *FUNDAMENTAL_SOURCE_SPECS[source],
    ]
    frame["source_row_sha256"] = frame.apply(
        lambda row: _row_sha256(row, source, identity_fields), axis=1
    )
    frame = frame.dropna(
        subset=["period_end", "announcement_date", "available_date"]
    )
    frame = frame.drop_duplicates("source_row_sha256", keep="last")
    frame = frame.sort_values(
        [
            "symbol",
            "period_end",
            "report_type",
            "company_type",
            "announcement_date",
            "source_update_flag",
            "source_row_sha256",
        ]
    )
    has_supported_value = pd.Series(False, index=frame.index)
    for provider_field in FUNDAMENTAL_SOURCE_SPECS[source]:
        if provider_field not in frame:
            continue
        values = pd.to_numeric(frame[provider_field], errors="coerce")
        has_supported_value |= values.notna() & np.isfinite(values)
    frame = frame.loc[has_supported_value].copy()
    if frame.empty:
        return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
    revision_group = [
        "symbol",
        "period_end",
        "report_type",
        "company_type",
    ]
    frame["revision_sequence"] = frame.groupby(
        revision_group, dropna=False
    ).cumcount() + 1

    records: list[pd.DataFrame] = []
    for provider_field, (metric, unit) in FUNDAMENTAL_SOURCE_SPECS[
        source
    ].items():
        if provider_field not in frame:
            continue
        values = pd.to_numeric(frame[provider_field], errors="coerce")
        valid = values.notna() & np.isfinite(values)
        if not valid.any():
            continue
        selected = frame.loc[
            valid,
            [
                "symbol",
                "period_end",
                "announcement_date",
                "available_date",
                "revision_sequence",
                "report_type",
                "company_type",
                "source_update_flag",
                "source_row_sha256",
            ],
        ].copy()
        selected.insert(1, "source", source)
        selected.insert(2, "metric", metric)
        selected.insert(3, "unit", unit)
        selected["value"] = values.loc[valid].astype(float)
        records.append(selected)
    if not records:
        return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
    return (
        pd.concat(records, ignore_index=True)[FUNDAMENTAL_COLUMNS]
        .sort_values(
            [
                "symbol",
                "metric",
                "period_end",
                "available_date",
                "revision_sequence",
            ]
        )
        .reset_index(drop=True)
    )


def normalize_valuations(
    raw: pd.DataFrame,
    calendar: Iterable[pd.Timestamp | str],
    lag_trading_days: int = 0,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=VALUATION_COLUMNS)
    required = {"ts_code", "trade_date"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"daily_basic 缺少字段: {sorted(missing)}")
    mapping = {
        "turnover_rate": "turnover_rate_pct",
        "pe_ttm": "pe_ttm",
        "pb": "pb",
        "ps_ttm": "ps_ttm",
        "dv_ttm": "dividend_yield_ttm_pct",
        "total_mv": "total_market_value_10k_cny",
        "circ_mv": "float_market_value_10k_cny",
    }
    frame = pd.DataFrame(
        {
            "date": _provider_dates(raw["trade_date"]),
            "symbol": raw["ts_code"].astype(str),
        }
    )
    frame["available_date"] = availability_dates(
        frame["date"], calendar, lag_trading_days
    ).to_numpy()
    for provider_field, canonical in mapping.items():
        frame[canonical] = pd.to_numeric(
            raw.get(provider_field, np.nan), errors="coerce"
        )
    frame = frame.dropna(subset=["date", "available_date"])
    frame = frame.sort_values(["symbol", "date"]).drop_duplicates(
        ["symbol", "date"], keep="last"
    )
    return frame[VALUATION_COLUMNS].reset_index(drop=True)


@dataclass
class PointInTimeDataBundle:
    fundamentals: pd.DataFrame
    valuations: pd.DataFrame
    calendar: pd.DatetimeIndex
    fundamental_lag_trading_days: int = 1
    valuation_lag_trading_days: int = 0
    manifest: dict[str, Any] = field(default_factory=dict)

    def prepare(self, strict: bool = True) -> "PointInTimeDataBundle":
        fundamentals = self.fundamentals.copy()
        valuations = self.valuations.copy()
        if fundamentals.empty:
            fundamentals = pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        if valuations.empty:
            valuations = pd.DataFrame(columns=VALUATION_COLUMNS)
        missing_fundamentals = set(FUNDAMENTAL_COLUMNS).difference(
            fundamentals.columns
        )
        missing_valuations = set(VALUATION_COLUMNS).difference(valuations.columns)
        if missing_fundamentals:
            raise ValueError(
                f"fundamentals 缺少字段: {sorted(missing_fundamentals)}"
            )
        if missing_valuations:
            raise ValueError(
                f"valuations 缺少字段: {sorted(missing_valuations)}"
            )

        calendar = pd.DatetimeIndex(pd.to_datetime(self.calendar)).normalize()
        calendar = calendar.unique().sort_values()
        if calendar.empty:
            raise ValueError("PIT 数据必须包含交易日历")

        for column in ["period_end", "announcement_date", "available_date"]:
            fundamentals[column] = pd.to_datetime(
                fundamentals[column], errors="coerce"
            ).dt.normalize()
        for column in ["date", "available_date"]:
            valuations[column] = pd.to_datetime(
                valuations[column], errors="coerce"
            ).dt.normalize()
        for column in ["symbol", "source", "metric", "unit"]:
            fundamentals[column] = fundamentals[column].astype(str)
        for column in [
            "report_type",
            "company_type",
            "source_update_flag",
            "source_row_sha256",
        ]:
            fundamentals[column] = (
                fundamentals[column]
                .astype("string")
                .fillna("")
                .astype(str)
            )
        fundamentals["revision_sequence"] = pd.to_numeric(
            fundamentals["revision_sequence"], errors="coerce"
        )
        fundamentals["value"] = pd.to_numeric(
            fundamentals["value"], errors="coerce"
        )
        valuations["symbol"] = valuations["symbol"].astype(str)
        for column in VALUATION_VALUE_COLUMNS:
            valuations[column] = pd.to_numeric(
                valuations[column], errors="coerce"
            )

        if strict:
            if fundamentals.empty:
                raise ValueError("PIT 财报数据不能为空")
            if valuations.empty:
                raise ValueError("PIT 估值数据不能为空")
            if fundamentals[
                ["period_end", "announcement_date", "available_date"]
            ].isna().any().any():
                raise ValueError("财报记录包含无效日期")
            if valuations[["date", "available_date"]].isna().any().any():
                raise ValueError("估值记录包含无效日期")
            blank_labels = pd.Series(False, index=fundamentals.index)
            for column in ["symbol", "source", "metric", "unit"]:
                blank_labels |= fundamentals[column].str.strip().isin(
                    {"", "nan", "None"}
                )
            if blank_labels.any():
                raise ValueError("财报记录包含空证券、来源、指标或单位")
            symbol_pattern = r"\d{6}\.(?:SH|SZ|BJ)"
            if not fundamentals["symbol"].str.fullmatch(symbol_pattern).all():
                raise ValueError("财报记录包含无效 A 股证券代码")
            if not valuations["symbol"].str.fullmatch(symbol_pattern).all():
                raise ValueError("估值记录包含无效 A 股证券代码")
            canonical_metric_units = {
                (source, metric, unit)
                for source, metrics in FUNDAMENTAL_SOURCE_SPECS.items()
                for metric, unit in metrics.values()
            }
            actual_metric_units = set(
                fundamentals[["source", "metric", "unit"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            unsupported = sorted(
                actual_metric_units.difference(canonical_metric_units)
            )
            if unsupported:
                raise ValueError(
                    "财报记录包含未知来源、指标或单位组合: "
                    + repr(unsupported[:5])
                )
            if (
                fundamentals["period_end"]
                > fundamentals["announcement_date"]
            ).any():
                raise ValueError("财报公告日期不能早于报告期结束日")
            if not np.isfinite(
                fundamentals["value"].to_numpy(dtype=float)
            ).all():
                raise ValueError("财报指标值必须是有限数")
            revisions = fundamentals["revision_sequence"].to_numpy(dtype=float)
            if (
                not np.isfinite(revisions).all()
                or (revisions < 1).any()
                or (np.floor(revisions) != revisions).any()
            ):
                raise ValueError("revision_sequence 必须是从 1 开始的整数")
            valid_hash = fundamentals["source_row_sha256"].str.fullmatch(
                r"[0-9a-f]{64}"
            )
            if not valid_hash.all():
                raise ValueError("source_row_sha256 必须是 64 位十六进制摘要")
            duplicate_key = [
                "symbol",
                "source",
                "metric",
                "period_end",
                "report_type",
                "company_type",
                "source_row_sha256",
            ]
            if fundamentals.duplicated(duplicate_key).any():
                raise ValueError("财报长表存在重复指标记录")

            expected_fundamental = availability_dates(
                fundamentals["announcement_date"],
                calendar,
                self.fundamental_lag_trading_days,
            )
            if not expected_fundamental.reset_index(drop=True).equals(
                fundamentals["available_date"].reset_index(drop=True)
            ):
                raise ValueError("财报 available_date 不符合公告日交易日滞后规则")
            expected_valuation = availability_dates(
                valuations["date"],
                calendar,
                self.valuation_lag_trading_days,
            )
            if not expected_valuation.reset_index(drop=True).equals(
                valuations["available_date"].reset_index(drop=True)
            ):
                raise ValueError("估值 available_date 不符合交易日滞后规则")
            if valuations.duplicated(["symbol", "date"]).any():
                raise ValueError("估值数据存在重复证券交易日")
            finite_valuations = valuations[VALUATION_VALUE_COLUMNS].replace(
                [np.inf, -np.inf], np.nan
            )
            if not finite_valuations.notna().any(axis=1).all():
                raise ValueError("估值记录不能全部为空")
            for column in [
                "total_market_value_10k_cny",
                "float_market_value_10k_cny",
            ]:
                values = valuations[column].dropna()
                if (values <= 0).any():
                    raise ValueError(f"{column} 必须为正数")

            revision_rows = fundamentals[
                [
                    "symbol",
                    "source",
                    "period_end",
                    "report_type",
                    "company_type",
                    "source_row_sha256",
                    "revision_sequence",
                ]
            ].drop_duplicates()
            group_key = [
                "symbol",
                "source",
                "period_end",
                "report_type",
                "company_type",
            ]
            for _, group in revision_rows.groupby(group_key, dropna=False):
                actual = sorted(group["revision_sequence"].astype(int).unique())
                if actual != list(range(1, len(actual) + 1)):
                    raise ValueError("同一财报期的修订序号必须连续")

        fundamentals = fundamentals.sort_values(
            [
                "symbol",
                "metric",
                "period_end",
                "available_date",
                "revision_sequence",
            ]
        ).reset_index(drop=True)
        valuations = valuations.sort_values(["symbol", "date"]).reset_index(
            drop=True
        )
        return PointInTimeDataBundle(
            fundamentals=fundamentals[FUNDAMENTAL_COLUMNS],
            valuations=valuations[VALUATION_COLUMNS],
            calendar=calendar,
            fundamental_lag_trading_days=self.fundamental_lag_trading_days,
            valuation_lag_trading_days=self.valuation_lag_trading_days,
            manifest=dict(self.manifest),
        )

    def visible_fundamentals(
        self,
        when: pd.Timestamp | str,
        symbols: Iterable[str] | None = None,
        maximum_age_days: int | None = None,
    ) -> pd.DataFrame:
        """Return the latest visible revision for every report period and metric."""
        date = pd.Timestamp(when).normalize()
        frame = self.fundamentals.loc[
            self.fundamentals["available_date"].le(date)
        ].copy()
        if symbols is not None:
            wanted = set(map(str, symbols))
            frame = frame.loc[frame["symbol"].isin(wanted)]
        if maximum_age_days is not None:
            frame = frame.loc[
                (date - frame["period_end"]).dt.days.le(maximum_age_days)
            ]
        if frame.empty:
            return frame.reset_index(drop=True)
        frame = frame.sort_values(
            [
                "symbol",
                "source",
                "metric",
                "period_end",
                "report_type",
                "company_type",
                "available_date",
                "revision_sequence",
                "source_row_sha256",
            ]
        )
        key = [
            "symbol",
            "source",
            "metric",
            "period_end",
            "report_type",
            "company_type",
        ]
        return frame.groupby(key, as_index=False, sort=False).tail(1).reset_index(
            drop=True
        )

    def fundamental_snapshot(
        self,
        when: pd.Timestamp | str,
        symbols: Iterable[str] | None = None,
        maximum_age_days: int | None = None,
    ) -> pd.DataFrame:
        visible = self.visible_fundamentals(
            when,
            symbols=symbols,
            maximum_age_days=maximum_age_days,
        )
        if visible.empty:
            return pd.DataFrame(columns=["symbol", "as_of_date"])
        latest = visible.sort_values(
            [
                "symbol",
                "metric",
                "period_end",
                "available_date",
                "revision_sequence",
            ]
        ).groupby(["symbol", "metric"], as_index=False, sort=False).tail(1)
        snapshot = latest.pivot(index="symbol", columns="metric", values="value")
        snapshot.columns.name = None
        snapshot = snapshot.reset_index()
        snapshot.insert(1, "as_of_date", pd.Timestamp(when).normalize())
        return snapshot.sort_values("symbol").reset_index(drop=True)

    def valuation_snapshot(
        self,
        when: pd.Timestamp | str,
        symbols: Iterable[str] | None = None,
        maximum_age_days: int | None = None,
    ) -> pd.DataFrame:
        date = pd.Timestamp(when).normalize()
        frame = self.valuations.loc[
            self.valuations["available_date"].le(date)
        ].copy()
        if symbols is not None:
            frame = frame.loc[frame["symbol"].isin(set(map(str, symbols)))]
        if maximum_age_days is not None:
            frame = frame.loc[(date - frame["date"]).dt.days.le(maximum_age_days)]
        if frame.empty:
            return frame.reset_index(drop=True)
        return (
            frame.sort_values(["symbol", "date", "available_date"])
            .groupby("symbol", as_index=False, sort=False)
            .tail(1)
            .sort_values("symbol")
            .reset_index(drop=True)
        )

    def snapshot(
        self,
        when: pd.Timestamp | str,
        symbols: Iterable[str] | None = None,
        *,
        maximum_fundamental_age_days: int | None = None,
        maximum_valuation_age_days: int | None = None,
    ) -> pd.DataFrame:
        """Build one wide, point-in-time feature snapshot for inspection/replay."""
        as_of = pd.Timestamp(when).normalize()
        fundamentals = self.fundamental_snapshot(
            as_of,
            symbols=symbols,
            maximum_age_days=maximum_fundamental_age_days,
        )
        valuations = self.valuation_snapshot(
            as_of,
            symbols=symbols,
            maximum_age_days=maximum_valuation_age_days,
        ).rename(
            columns={
                "date": "valuation_date",
                "available_date": "valuation_available_date",
            }
        )
        if fundamentals.empty and valuations.empty:
            return pd.DataFrame(columns=["symbol", "as_of_date"])
        if fundamentals.empty:
            fundamentals = valuations[["symbol"]].copy()
            fundamentals.insert(1, "as_of_date", as_of)
        if valuations.empty:
            return fundamentals.sort_values("symbol").reset_index(drop=True)
        return (
            fundamentals.merge(valuations, on="symbol", how="outer")
            .assign(as_of_date=lambda frame: frame["as_of_date"].fillna(as_of))
            .sort_values("symbol")
            .reset_index(drop=True)
        )

    def audit(self, expected_symbols: Iterable[str] = ()) -> dict[str, Any]:
        expected = set(map(str, expected_symbols))
        fundamental_symbols = set(self.fundamentals["symbol"].astype(str))
        valuation_symbols = set(self.valuations["symbol"].astype(str))
        denominator = max(len(expected), 1)
        return {
            "fundamental_rows": int(len(self.fundamentals)),
            "valuation_rows": int(len(self.valuations)),
            "fundamental_symbols": len(fundamental_symbols),
            "valuation_symbols": len(valuation_symbols),
            "expected_symbols": len(expected),
            "fundamental_symbol_coverage": (
                len(expected.intersection(fundamental_symbols)) / denominator
                if expected
                else 1.0
            ),
            "valuation_symbol_coverage": (
                len(expected.intersection(valuation_symbols)) / denominator
                if expected
                else 1.0
            ),
            "missing_fundamental_symbols": sorted(
                expected.difference(fundamental_symbols)
            ),
            "missing_valuation_symbols": sorted(
                expected.difference(valuation_symbols)
            ),
            "metrics": sorted(self.fundamentals["metric"].unique().tolist()),
            "first_period_end": (
                str(self.fundamentals["period_end"].min().date())
                if not self.fundamentals.empty
                else None
            ),
            "last_available_date": (
                str(
                    max(
                        self.fundamentals["available_date"].max(),
                        self.valuations["available_date"].max(),
                    ).date()
                )
                if not self.fundamentals.empty and not self.valuations.empty
                else None
            ),
        }

    @classmethod
    def from_cache(
        cls,
        cache_dir: str | Path,
        *,
        strict: bool = True,
        expected_config: AppConfig | PointInTimeDataConfig | None = None,
        base_manifest_path: str | Path | None = None,
    ) -> "PointInTimeDataBundle":
        root = Path(cache_dir).resolve()
        manifest_path = root / "manifest.json"
        calendar_path = root / "calendar.csv.gz"
        fundamental_paths = sorted((root / "fundamentals").glob("*.csv.gz"))
        valuation_paths = sorted((root / "valuations").glob("*.csv.gz"))
        if (
            not manifest_path.is_file()
            or not calendar_path.is_file()
            or not fundamental_paths
            or not valuation_paths
        ):
            raise FileNotFoundError(f"PIT v1 缓存不完整: {root}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("PIT manifest.json 不是有效 JSON") from exc
        if manifest.get("schema_version") != PIT_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"PIT 缓存版本为 {manifest.get('schema_version')!r}，"
                f"当前要求 v{PIT_CACHE_SCHEMA_VERSION}"
            )
        if (
            manifest.get("statement_report_type")
            != CANONICAL_STATEMENT_REPORT_TYPE
        ):
            raise ValueError(
                "PIT 财报口径不一致；当前要求 Tushare 合并报表类型 "
                f"{CANONICAL_STATEMENT_REPORT_TYPE}"
            )

        verification = verify_file_inventory(root, manifest.get("files", []))
        if verification["inventory_sha256"] != manifest.get(
            "data_fingerprint_sha256"
        ):
            raise ValueError("PIT 数据集合指纹与 manifest 不一致")
        consumed = {
            path.resolve().relative_to(root).as_posix()
            for path in [calendar_path, *fundamental_paths, *valuation_paths]
        }
        recorded = {
            str(item.get("path"))
            for item in manifest.get("files", [])
            if isinstance(item, Mapping)
        }
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        archive_bridge = manifest.get("archive_bridge")
        if archive_bridge is not None:
            if not isinstance(archive_bridge, Mapping):
                raise ValueError("PIT archive_bridge 必须是对象")
            lineage_name = str(archive_bridge.get("lineage_file", ""))
            if lineage_name != "archive_lineage.json.gz":
                raise ValueError("PIT 归档桥接血缘文件名不受支持")
            lineage_path = root / lineage_name
            if not lineage_path.is_file():
                raise FileNotFoundError(f"PIT 缺少归档血缘文件: {lineage_path}")
            try:
                with gzip.open(lineage_path, "rt", encoding="utf-8") as handle:
                    lineage = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("PIT 归档血缘文件无效") from exc
            if not isinstance(lineage, Mapping):
                raise ValueError("PIT 归档血缘顶层必须是对象")
            task_set_sha256 = payload_sha256(lineage.get("selected_tasks", []))
            bridge_version = int(archive_bridge.get("schema_version", 0))
            bridge_mode = str(archive_bridge.get("mode", ""))
            acceptance_required = bridge_mode == "strict" and bridge_version >= 2
            if (
                task_set_sha256 != lineage.get("selected_task_set_sha256")
                or task_set_sha256
                != archive_bridge.get("selected_task_set_sha256")
                or bool(lineage.get("research_eligible"))
                != bool(manifest.get("research_eligible"))
                or lineage.get("mode") != bridge_mode
                or bool(manifest.get("research_eligible"))
                != (bridge_mode == "strict")
                or bool(archive_bridge.get("acceptance_required"))
                != acceptance_required
            ):
                raise ValueError("PIT 归档血缘、模式、任务集合或研究资格不一致")
            consumed.add(lineage_name)
        if recorded != consumed or actual != recorded:
            raise ValueError(
                "PIT 缓存读取文件、封存文件与实际文件集合不一致；"
                f"未封存={sorted(actual - recorded)[:5]}，"
                f"未使用={sorted(recorded - consumed)[:5]}"
            )

        config = (
            expected_config.point_in_time
            if isinstance(expected_config, AppConfig)
            else expected_config
        )
        if config is not None:
            expected_identity = {
                "provider": config.provider,
                "fundamental_lag_trading_days": (
                    config.fundamental_lag_trading_days
                ),
                "valuation_lag_trading_days": config.valuation_lag_trading_days,
            }
            mismatches = {
                key: {"cache": manifest.get(key), "config": value}
                for key, value in expected_identity.items()
                if manifest.get(key) != value
            }
            if mismatches:
                raise ValueError(f"PIT 缓存身份与配置不一致: {mismatches}")
            if isinstance(expected_config, AppConfig):
                required_start = pd.Timestamp(
                    expected_config.backtest.start_date
                ) - pd.DateOffset(years=config.history_years)
                required_end = pd.Timestamp(expected_config.backtest.end_date)
                if (
                    pd.Timestamp(manifest["requested_start"]) > required_start
                    or pd.Timestamp(manifest["requested_end"]) < required_end
                ):
                    raise ValueError("PIT 缓存日期范围不足")

        if base_manifest_path is not None:
            base_path = Path(base_manifest_path).resolve()
            if not base_path.is_file():
                raise FileNotFoundError(f"缺少基础行情 manifest: {base_path}")
            base_payload = json.loads(base_path.read_text(encoding="utf-8"))
            if manifest.get("base_manifest_sha256") != sha256_file(base_path):
                raise ValueError("基础行情 manifest 已变化，PIT 快照身份失效")
            if manifest.get("base_data_fingerprint_sha256") != base_payload.get(
                "data_fingerprint_sha256"
            ):
                raise ValueError("基础行情数据指纹与 PIT 快照不一致")
            if isinstance(expected_config, AppConfig):
                base_symbols = _verify_base_market_cache(
                    base_path, expected_config
                )
                pit_symbols = set(map(str, manifest.get("symbols", [])))
                if pit_symbols != base_symbols:
                    raise ValueError(
                        "PIT 证券全集与基础行情历史成分不一致: "
                        f"缺少={sorted(base_symbols - pit_symbols)[:5]}，"
                        f"多出={sorted(pit_symbols - base_symbols)[:5]}"
                    )

        manifest_symbols = list(map(str, manifest.get("symbols", [])))
        fundamentals = _read_symbol_partitions(
            fundamental_paths, manifest_symbols, FUNDAMENTAL_COLUMNS
        )
        valuations = _read_symbol_partitions(
            valuation_paths, manifest_symbols, VALUATION_COLUMNS
        )
        calendar = pd.DatetimeIndex(
            pd.to_datetime(pd.read_csv(calendar_path)["date"])
        )
        manifest = {**manifest, "verification": verification}
        bundle = cls(
            fundamentals=fundamentals,
            valuations=valuations,
            calendar=calendar,
            fundamental_lag_trading_days=int(
                manifest["fundamental_lag_trading_days"]
            ),
            valuation_lag_trading_days=int(
                manifest["valuation_lag_trading_days"]
            ),
            manifest=manifest,
        ).prepare(strict=strict)
        expected_symbols = manifest.get("symbols", [])
        audit = bundle.audit(expected_symbols)
        if config is not None and (
            audit["fundamental_symbol_coverage"]
            < config.minimum_symbol_coverage
            or audit["valuation_symbol_coverage"]
            < config.minimum_symbol_coverage
        ):
            raise ValueError(
                "PIT 数据证券覆盖不足: "
                f"财报={audit['fundamental_symbol_coverage']:.2%}，"
                f"估值={audit['valuation_symbol_coverage']:.2%}"
            )
        bundle.manifest["data_quality"] = audit
        return bundle


def _clear_known_cache(root: Path) -> None:
    if not root.exists():
        return
    allowed_names = {
        "manifest.json",
        "calendar.csv.gz",
        "download_state.json",
        "archive_lineage.json.gz",
    }
    unexpected: list[str] = []
    files = [path for path in root.rglob("*") if path.is_file()]
    for path in files:
        relative = path.relative_to(root)
        allowed = (
            relative.as_posix() in allowed_names
            or (
                relative.parent.as_posix() in {"fundamentals", "valuations"}
                and relative.name.endswith(".csv.gz")
            )
        )
        if not allowed:
            unexpected.append(relative.as_posix())
    if unexpected:
        raise ValueError(
            "拒绝清理包含未知文件的 PIT 目录: " + ", ".join(unexpected[:5])
        )
    for path in files:
        path.unlink()
    for path in sorted(
        [value for value in root.rglob("*") if value.is_dir()],
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        path.rmdir()


def write_pit_cache(
    bundle: PointInTimeDataBundle,
    cache_dir: str | Path,
    *,
    provider: str,
    requested_start: pd.Timestamp | str,
    requested_end: pd.Timestamp | str,
    expected_symbols: Iterable[str] = (),
    base_manifest_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    root = Path(cache_dir).resolve()
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"PIT 缓存已存在，拒绝覆盖: {root}")
        _clear_known_cache(root)
    root.mkdir(parents=True, exist_ok=True)
    prepared = bundle.prepare(strict=True)
    symbols = sorted(
        set(map(str, expected_symbols))
        | set(prepared.fundamentals["symbol"])
        | set(prepared.valuations["symbol"])
    )
    if not symbols:
        raise ValueError("PIT 缓存没有证券")
    fundamental_paths: list[Path] = []
    valuation_paths: list[Path] = []
    for symbol in symbols:
        filename = symbol.replace(".", "_") + ".csv.gz"
        fundamental_path = root / "fundamentals" / filename
        valuation_path = root / "valuations" / filename
        _atomic_csv(
            prepared.fundamentals.loc[
                prepared.fundamentals["symbol"].eq(symbol),
                FUNDAMENTAL_COLUMNS,
            ],
            fundamental_path,
        )
        _atomic_csv(
            prepared.valuations.loc[
                prepared.valuations["symbol"].eq(symbol), VALUATION_COLUMNS
            ],
            valuation_path,
        )
        fundamental_paths.append(fundamental_path)
        valuation_paths.append(valuation_path)
    calendar_path = root / "calendar.csv.gz"
    _atomic_csv(pd.DataFrame({"date": prepared.calendar}), calendar_path)

    base_manifest_sha256 = None
    base_data_fingerprint = None
    if base_manifest_path is not None:
        base_path = Path(base_manifest_path).resolve()
        base_payload = json.loads(base_path.read_text(encoding="utf-8"))
        base_manifest_sha256 = sha256_file(base_path)
        base_data_fingerprint = base_payload.get("data_fingerprint_sha256")
    files = build_file_inventory(
        root, [calendar_path, *fundamental_paths, *valuation_paths]
    )
    audit = prepared.audit(symbols)
    manifest = {
        "schema_version": PIT_CACHE_SCHEMA_VERSION,
        "provider": provider,
        "statement_report_type": CANONICAL_STATEMENT_REPORT_TYPE,
        "requested_start": str(pd.Timestamp(requested_start).date()),
        "requested_end": str(pd.Timestamp(requested_end).date()),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "fundamental_lag_trading_days": (
            prepared.fundamental_lag_trading_days
        ),
        "valuation_lag_trading_days": prepared.valuation_lag_trading_days,
        "base_manifest_sha256": base_manifest_sha256,
        "base_data_fingerprint_sha256": base_data_fingerprint,
        "symbols": symbols,
        "data_quality": audit,
        "files": files,
        "data_fingerprint_sha256": inventory_sha256(files),
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest, manifest_path)
    return manifest_path


def verify_pit_cache(
    cache_dir: str | Path,
    *,
    expected_config: AppConfig | PointInTimeDataConfig | None = None,
    base_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    bundle = PointInTimeDataBundle.from_cache(
        cache_dir,
        strict=True,
        expected_config=expected_config,
        base_manifest_path=base_manifest_path,
    )
    return bundle.manifest


class TusharePointInTimeDownloader:
    """Download a separate, restartable PIT sidecar without mutating v4 bars."""

    def __init__(self, config: AppConfig, client: Any | None = None) -> None:
        config.validate()
        if not config.point_in_time.enabled:
            raise ValueError(
                "PIT 数据未启用；请在 point_in_time.enabled 设置为 true"
            )
        self.config = config
        if client is None:
            token = os.getenv(config.data.token_env, "").strip()
            if not token:
                raise RuntimeError(
                    f"缺少环境变量 {config.data.token_env}；PIT 下载需要 Tushare Token"
                )
            client = RateLimitedTushare(
                token=token,
                calls_per_minute=config.data.calls_per_minute,
                retries=config.data.retries,
            )
        self.client = client
        self.root = Path(config.point_in_time.cache_dir).resolve()
        self.base_manifest_path = (
            Path(config.data.cache_dir).resolve() / "manifest.json"
        )

    def download(self) -> PointInTimeDataBundle:
        self._recover_interrupted_swap()
        pit = self.config.point_in_time
        if self.root.joinpath("manifest.json").is_file() and not pit.refresh:
            LOGGER.info("命中完整 PIT 缓存，跳过下载")
            return PointInTimeDataBundle.from_cache(
                self.root,
                strict=True,
                expected_config=self.config,
                base_manifest_path=self.base_manifest_path,
            )
        if pit.refresh:
            _clear_known_cache(self.root)

        market = MarketDataBundle.from_cache(
            self.config.data.cache_dir,
            strict=self.config.data.strict_validation,
            expected_config=self.config,
        )
        start = pd.Timestamp(self.config.backtest.start_date) - pd.DateOffset(
            years=pit.history_years
        )
        end = pd.Timestamp(self.config.backtest.end_date)
        symbols = sorted(market.membership["symbol"].astype(str).unique())
        calendar = self._fetch_calendar(start, end + pd.Timedelta(14, unit="D"))
        state_path = self.root / "download_state.json"
        identity = {
            "schema_version": PIT_CACHE_SCHEMA_VERSION,
            "provider": pit.provider,
            "statement_report_type": CANONICAL_STATEMENT_REPORT_TYPE,
            "requested_start": str(start.date()),
            "requested_end": str(end.date()),
            "symbols": symbols,
            "fundamental_lag_trading_days": pit.fundamental_lag_trading_days,
            "valuation_lag_trading_days": pit.valuation_lag_trading_days,
            "base_manifest_sha256": sha256_file(self.base_manifest_path),
        }
        identity_hash = payload_sha256(identity)
        completed: set[str] = set()
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("identity_sha256") != identity_hash:
                raise ValueError(
                    "未完成 PIT 下载的身份与当前配置不同；请设置 "
                    "point_in_time.refresh=true 后重新开始"
                )
            completed = set(map(str, state.get("completed_symbols", [])))
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                {
                    "identity_sha256": identity_hash,
                    "identity": identity,
                    "completed_symbols": [],
                },
                state_path,
            )

        fundamental_dir = self.root / "fundamentals"
        valuation_dir = self.root / "valuations"
        for number, symbol in enumerate(symbols, start=1):
            filename = symbol.replace(".", "_") + ".csv.gz"
            fundamental_path = fundamental_dir / filename
            valuation_path = valuation_dir / filename
            reusable = (
                symbol in completed
                and fundamental_path.is_file()
                and valuation_path.is_file()
            )
            if reusable:
                LOGGER.info("[%d/%d] 复用 PIT %s", number, len(symbols), symbol)
                continue
            LOGGER.info("[%d/%d] 下载 PIT %s", number, len(symbols), symbol)
            fundamentals = self._fetch_fundamentals(
                symbol, start, end, calendar
            )
            valuations = self._fetch_valuations(symbol, start, end, calendar)
            _atomic_csv(fundamentals, fundamental_path)
            _atomic_csv(valuations, valuation_path)
            completed.add(symbol)
            _atomic_json(
                {
                    "identity_sha256": identity_hash,
                    "identity": identity,
                    "completed_symbols": sorted(completed),
                },
                state_path,
            )

        fundamental_paths = [
            fundamental_dir / (symbol.replace(".", "_") + ".csv.gz")
            for symbol in symbols
        ]
        valuation_paths = [
            valuation_dir / (symbol.replace(".", "_") + ".csv.gz")
            for symbol in symbols
        ]
        bundle = PointInTimeDataBundle(
            fundamentals=_read_partitioned_csv(
                fundamental_paths, FUNDAMENTAL_COLUMNS
            ),
            valuations=_read_partitioned_csv(
                valuation_paths, VALUATION_COLUMNS
            ),
            calendar=calendar,
            fundamental_lag_trading_days=pit.fundamental_lag_trading_days,
            valuation_lag_trading_days=pit.valuation_lag_trading_days,
        ).prepare(strict=True)
        audit = bundle.audit(symbols)
        if (
            audit["fundamental_symbol_coverage"] < pit.minimum_symbol_coverage
            or audit["valuation_symbol_coverage"] < pit.minimum_symbol_coverage
        ):
            raise RuntimeError(
                "PIT 下载覆盖不足: "
                f"财报={audit['fundamental_symbol_coverage']:.2%}，"
                f"估值={audit['valuation_symbol_coverage']:.2%}"
            )

        # The completed files are already in place. Keep the state marker until
        # the replacement cache has been sealed successfully, so a failed seal
        # remains restartable.
        temporary_root = self.root.with_name(self.root.name + ".sealed")
        if temporary_root.exists():
            _clear_known_cache(temporary_root)
        write_pit_cache(
            bundle,
            temporary_root,
            provider=pit.provider,
            requested_start=start,
            requested_end=end,
            expected_symbols=symbols,
            base_manifest_path=self.base_manifest_path,
            overwrite=False,
        )
        PointInTimeDataBundle.from_cache(
            temporary_root,
            strict=True,
            expected_config=self.config,
            base_manifest_path=self.base_manifest_path,
        )
        backup_root = self.root.with_name(self.root.name + ".unsealed")
        if backup_root.exists():
            _clear_known_cache(backup_root)
            backup_root.rmdir()
        os.replace(self.root, backup_root)
        try:
            os.replace(temporary_root, self.root)
            result = PointInTimeDataBundle.from_cache(
                self.root,
                strict=True,
                expected_config=self.config,
                base_manifest_path=self.base_manifest_path,
            )
        except Exception:
            if self.root.exists():
                _clear_known_cache(self.root)
                self.root.rmdir()
            if backup_root.exists():
                os.replace(backup_root, self.root)
            raise
        _clear_known_cache(backup_root)
        backup_root.rmdir()
        return result

    def _recover_interrupted_swap(self) -> None:
        backup_root = self.root.with_name(self.root.name + ".unsealed")
        if not backup_root.exists():
            return
        if not self.root.exists():
            LOGGER.warning("恢复上次中断的 PIT 缓存交换")
            os.replace(backup_root, self.root)
            return
        if self.root.joinpath("manifest.json").is_file():
            _clear_known_cache(backup_root)
            backup_root.rmdir()
            return
        raise ValueError(
            "同时发现未封存 PIT 根目录和交换备份，拒绝猜测；"
            f"请人工检查 {self.root} 与 {backup_root}"
        )

    def _fetch_calendar(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DatetimeIndex:
        frame = self.client.call(
            "trade_cal",
            exchange="SSE",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            is_open="1",
            fields="cal_date,is_open",
        )
        if frame.empty:
            raise RuntimeError("PIT 交易日历为空")
        return pd.DatetimeIndex(
            pd.to_datetime(frame["cal_date"].astype(str), format="%Y%m%d")
        ).sort_values()

    def _fetch_fundamentals(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for source, metrics in FUNDAMENTAL_SOURCE_SPECS.items():
            fields = ",".join(
                dict.fromkeys([*FUNDAMENTAL_SOURCE_FIELDS[source], *metrics])
            )
            query: dict[str, object] = {
                "ts_code": symbol,
                "start_date": start.strftime("%Y%m%d"),
                "end_date": end.strftime("%Y%m%d"),
                "fields": fields,
            }
            if source != "fina_indicator":
                # Type 1 is Tushare's latest consolidated statement scope. An
                # explicit query makes the cache contract independent of any
                # future provider-default change; update_flag still preserves
                # returned revisions within this scope.
                query["report_type"] = CANONICAL_STATEMENT_REPORT_TYPE
            raw = self.client.call(source, **query)
            if raw.empty:
                continue
            normalized = normalize_fundamental_source(
                source,
                raw,
                calendar,
                self.config.point_in_time.fundamental_lag_trading_days,
            )
            if not normalized.empty:
                frames.append(normalized)
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        )

    def _fetch_valuations(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        raw = self.client.call(
            "daily_basic",
            ts_code=symbol,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            fields=(
                "ts_code,trade_date,turnover_rate,pe_ttm,pb,ps_ttm,dv_ttm,"
                "total_mv,circ_mv"
            ),
        )
        return normalize_valuations(
            raw,
            calendar,
            self.config.point_in_time.valuation_lag_trading_days,
        )


__all__ = [
    "CANONICAL_STATEMENT_REPORT_TYPE",
    "FUNDAMENTAL_COLUMNS",
    "FUNDAMENTAL_SOURCE_FIELDS",
    "FUNDAMENTAL_SOURCE_SPECS",
    "PIT_CACHE_SCHEMA_VERSION",
    "VALUATION_COLUMNS",
    "PointInTimeDataBundle",
    "TusharePointInTimeDownloader",
    "availability_dates",
    "normalize_fundamental_source",
    "normalize_valuations",
    "require_pit_research_eligible",
    "verify_pit_cache",
    "write_pit_cache",
]
