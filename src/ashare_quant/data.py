from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import AppConfig, DataConfig
from .provenance import build_file_inventory, inventory_sha256, verify_file_inventory

LOGGER = logging.getLogger(__name__)
CACHE_SCHEMA_VERSION = 4

BAR_COLUMNS = {
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "amount",
    "adj_factor",
    "up_limit",
    "down_limit",
    "is_st",
    "total_mv",
    "circ_mv",
}

ACTION_COLUMNS = [
    "symbol",
    "record_date",
    "ex_date",
    "pay_date",
    "stock_list_date",
    "cash_dividend",
    "stock_dividend",
]

SECURITY_COLUMNS = [
    "symbol",
    "name",
    "industry",
    "market",
    "list_status",
    "list_date",
    "delist_date",
]

INDUSTRY_COLUMNS = [
    "symbol",
    "industry_code",
    "industry_name",
    "in_date",
    "out_date",
]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _provider_dates(values: pd.Series) -> pd.Series:
    """Parse provider YYYYMMDD values safely even when CSV inferred them as numbers."""
    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(text, errors="coerce", format="mixed").dt.normalize()


@dataclass
class MarketDataBundle:
    bars: pd.DataFrame
    membership: pd.DataFrame
    benchmark: pd.DataFrame
    calendar: pd.DatetimeIndex
    corporate_actions: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=ACTION_COLUMNS)
    )
    securities: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=SECURITY_COLUMNS)
    )
    industry_membership: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=INDUSTRY_COLUMNS)
    )
    regime: pd.DataFrame = field(default_factory=pd.DataFrame)

    def prepare(self, strict: bool = True) -> "MarketDataBundle":
        bars = self.bars.copy()
        membership = self.membership.copy()
        benchmark = self.benchmark.copy()
        regime = self.regime.copy() if isinstance(self.regime, pd.DataFrame) else pd.DataFrame()
        if regime.empty:
            if strict:
                raise ValueError(
                    "缺少独立择时指数 regime；v1.4 不允许用业绩基准静默替代"
                )
            # Non-strict ad-hoc analysis can opt into the legacy fallback explicitly.
            regime = benchmark.copy()
        actions = self.corporate_actions.copy()
        securities = self.securities.copy()
        industries = self.industry_membership.copy()

        missing = BAR_COLUMNS.difference(bars.columns)
        if missing:
            raise ValueError(f"bars 缺少字段: {sorted(missing)}")
        for frame in (bars, membership, benchmark, regime):
            if "date" not in frame.columns:
                raise ValueError("数据表缺少 date 字段")
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

        numeric_bar_columns = [
            "open",
            "high",
            "low",
            "close",
            "prev_close",
            "volume",
            "amount",
            "adj_factor",
            "up_limit",
            "down_limit",
            "total_mv",
            "circ_mv",
        ]
        for column in numeric_bar_columns:
            bars[column] = pd.to_numeric(bars[column], errors="coerce")
        if bars["is_st"].dtype == object:
            bars["is_st"] = (
                bars["is_st"]
                .fillna(False)
                .map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})
            )
        else:
            bars["is_st"] = bars["is_st"].fillna(False).astype(bool)
        bars["symbol"] = bars["symbol"].astype(str)
        membership["symbol"] = membership["symbol"].astype(str)
        membership["index_weight"] = pd.to_numeric(
            membership.get("index_weight", 0.0), errors="coerce"
        ).fillna(0.0)
        for label, frame in (("benchmark", benchmark), ("regime", regime)):
            if "close" not in frame.columns:
                raise ValueError(f"{label} 缺少 close 字段")
            for column in ["open", "high", "low", "close", "prev_close"]:
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if actions.empty:
            actions = pd.DataFrame(columns=ACTION_COLUMNS)
        missing_actions = set(ACTION_COLUMNS).difference(actions.columns)
        if missing_actions:
            raise ValueError(f"corporate_actions 缺少字段: {sorted(missing_actions)}")
        actions["symbol"] = actions["symbol"].astype(str)
        for column in ["record_date", "ex_date", "pay_date", "stock_list_date"]:
            actions[column] = pd.to_datetime(actions[column], errors="coerce").dt.normalize()
        for column in ["cash_dividend", "stock_dividend"]:
            actions[column] = pd.to_numeric(actions[column], errors="coerce").fillna(0.0)

        if securities.empty:
            securities = pd.DataFrame(columns=SECURITY_COLUMNS)
        missing_securities = set(SECURITY_COLUMNS).difference(securities.columns)
        if missing_securities:
            raise ValueError(f"securities 缺少字段: {sorted(missing_securities)}")
        securities["symbol"] = securities["symbol"].astype(str)
        for column in ["list_date", "delist_date"]:
            securities[column] = _provider_dates(securities[column])

        if industries.empty:
            industries = pd.DataFrame(columns=INDUSTRY_COLUMNS)
        missing_industries = set(INDUSTRY_COLUMNS).difference(industries.columns)
        if missing_industries:
            raise ValueError(f"industry_membership 缺少字段: {sorted(missing_industries)}")
        industries["symbol"] = industries["symbol"].astype(str)
        industries["industry_code"] = industries["industry_code"].astype(str)
        industries["industry_name"] = industries["industry_name"].astype(str)
        for column in ["in_date", "out_date"]:
            industries[column] = _provider_dates(industries[column])

        bars = bars.sort_values(["symbol", "date"]).drop_duplicates(
            ["symbol", "date"], keep="last"
        )
        membership = membership.sort_values(["date", "symbol"]).drop_duplicates(
            ["date", "symbol"], keep="last"
        )
        benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last")
        regime = regime.sort_values("date").drop_duplicates("date", keep="last")
        actions = actions.sort_values(["ex_date", "symbol"]).drop_duplicates(
            ["symbol", "ex_date"], keep="last"
        )
        securities = securities.sort_values("symbol").drop_duplicates("symbol", keep="last")
        industries = industries.sort_values(
            ["symbol", "in_date", "out_date", "industry_code"], na_position="last"
        ).drop_duplicates(
            ["symbol", "industry_code", "in_date", "out_date"], keep="last"
        )

        required_numeric = bars[numeric_bar_columns]
        if strict and not np.isfinite(required_numeric.to_numpy(dtype=float)).all():
            bad_columns = [column for column in numeric_bar_columns if bars[column].isna().any()]
            raise ValueError(f"行情存在 NaN 或无穷值: {bad_columns}")
        positive_columns = [
            "open",
            "high",
            "low",
            "close",
            "prev_close",
            "adj_factor",
            "up_limit",
            "down_limit",
            "total_mv",
            "circ_mv",
        ]
        if (bars[positive_columns] <= 0).any().any():
            raise ValueError("价格和复权因子必须为正数")
        if strict:
            invalid_ohlc = (
                (bars["high"] < bars[["open", "close"]].max(axis=1) - 1e-8)
                | (bars["low"] > bars[["open", "close"]].min(axis=1) + 1e-8)
                | (bars["high"] < bars["low"])
            )
            if invalid_ohlc.any():
                sample = bars.loc[invalid_ohlc, ["date", "symbol"]].iloc[0].to_dict()
                raise ValueError(f"OHLC 关系非法，示例: {sample}")
            if (bars[["volume", "amount"]] < 0).any().any():
                raise ValueError("成交量和成交额不能为负")
            if (bars["up_limit"] < bars["down_limit"]).any():
                raise ValueError("涨停价不能低于跌停价")
            if not np.isfinite(benchmark["close"].to_numpy(dtype=float)).all():
                raise ValueError("基准收盘价存在 NaN 或无穷值")
            if (benchmark["close"] <= 0).any():
                raise ValueError("基准收盘价必须为正数")
            if not np.isfinite(regime["close"].to_numpy(dtype=float)).all():
                raise ValueError("择时指数收盘价存在 NaN 或无穷值")
            if (regime["close"] <= 0).any():
                raise ValueError("择时指数收盘价必须为正数")
            if actions["ex_date"].isna().any():
                raise ValueError("公司行动必须包含有效 ex_date")
            if (actions[["cash_dividend", "stock_dividend"]] < 0).any().any():
                raise ValueError("分红送股比例不能为负")
            if not industries.empty:
                if industries["in_date"].isna().any():
                    raise ValueError("行业成员记录必须包含有效 in_date")
                invalid_labels = (
                    industries["industry_code"].str.strip().isin({"", "nan", "None"})
                    | industries["industry_name"].str.strip().isin({"", "nan", "None"})
                )
                if invalid_labels.any():
                    raise ValueError("行业成员记录必须包含有效行业代码和名称")
                invalid_interval = industries["out_date"].notna() & (
                    industries["out_date"] < industries["in_date"]
                )
                if invalid_interval.any():
                    raise ValueError("行业成员记录的 out_date 不能早于 in_date")
        if bars.empty or membership.empty or benchmark.empty or regime.empty:
            raise ValueError("行情、成分、业绩基准或择时指数数据不能为空")

        calendar = pd.DatetimeIndex(pd.to_datetime(self.calendar)).normalize().unique().sort_values()
        if calendar.empty:
            calendar = pd.DatetimeIndex(benchmark["date"].unique()).sort_values()
        if strict:
            member_symbols = set(membership["symbol"])
            bar_symbols = set(bars["symbol"])
            missing_bar_symbols = sorted(member_symbols.difference(bar_symbols))
            if missing_bar_symbols:
                raise ValueError(
                    "历史成分缺少行情文件: " + ", ".join(missing_bar_symbols[:10])
                )
            if securities.empty:
                raise ValueError("缺少证券主表 securities，请重新下载 v4 数据缓存")
            missing_master = sorted(member_symbols.difference(set(securities["symbol"])))
            if missing_master:
                raise ValueError("证券主表缺少历史成分: " + ", ".join(missing_master[:10]))
            if industries.empty:
                raise ValueError("缺少行业成员表 industry_membership，请重新下载 v4 数据缓存")
            missing_industry = sorted(member_symbols.difference(set(industries["symbol"])))
            if missing_industry:
                raise ValueError("行业成员表缺少历史成分: " + ", ".join(missing_industry[:10]))
            uncovered: list[str] = []
            industry_groups = {
                symbol: group for symbol, group in industries.groupby("symbol", sort=False)
            }
            for symbol, member_group in membership.groupby("symbol", sort=False):
                dates = pd.DatetimeIndex(member_group["date"].unique())
                covered = np.zeros(len(dates), dtype=bool)
                for interval in industry_groups[str(symbol)].itertuples(index=False):
                    interval_end = (
                        pd.Timestamp.max.normalize()
                        if pd.isna(interval.out_date)
                        else pd.Timestamp(interval.out_date)
                    )
                    covered |= (dates >= pd.Timestamp(interval.in_date)) & (
                        dates <= interval_end
                    )
                if not covered.all():
                    first_gap = dates[np.flatnonzero(~covered)[0]]
                    uncovered.append(f"{symbol}@{first_gap.date()}")
            if uncovered:
                raise ValueError(
                    "行业成员区间未覆盖历史指数成分快照: "
                    + ", ".join(uncovered[:10])
                )
            snapshot_sizes = membership.groupby("date")["symbol"].nunique()
            if (snapshot_sizes < 2).any():
                raise ValueError("指数成分快照数量异常")
            if not set(benchmark["date"]).issubset(set(calendar)):
                raise ValueError("基准行情包含交易日历之外的日期")
            expected_benchmark_dates = set(
                calendar[(calendar >= benchmark["date"].min()) & (calendar <= benchmark["date"].max())]
            )
            missing_benchmark_dates = sorted(expected_benchmark_dates.difference(set(benchmark["date"])))
            if missing_benchmark_dates:
                raise ValueError(
                    f"基准行情缺少 {len(missing_benchmark_dates)} 个交易日，"
                    f"首个缺口为 {missing_benchmark_dates[0].date()}"
                )
            if not set(regime["date"]).issubset(set(calendar)):
                raise ValueError("择时指数行情包含交易日历之外的日期")
            expected_regime_dates = set(
                calendar[(calendar >= regime["date"].min()) & (calendar <= regime["date"].max())]
            )
            missing_regime_dates = sorted(expected_regime_dates.difference(set(regime["date"])))
            if missing_regime_dates:
                raise ValueError(
                    f"择时指数行情缺少 {len(missing_regime_dates)} 个交易日，"
                    f"首个缺口为 {missing_regime_dates[0].date()}"
                )
            snapshot_dates = pd.DatetimeIndex(sorted(membership["date"].unique()))
            if len(snapshot_dates) > 1 and snapshot_dates.to_series().diff().dt.days.max() > 62:
                raise ValueError("历史指数成分快照存在超过 62 天的断档")
        return MarketDataBundle(
            bars=bars.reset_index(drop=True),
            membership=membership.reset_index(drop=True),
            benchmark=benchmark.reset_index(drop=True),
            calendar=calendar,
            corporate_actions=actions.reset_index(drop=True),
            securities=securities.reset_index(drop=True),
            industry_membership=industries.reset_index(drop=True),
            regime=regime.reset_index(drop=True),
        )

    @classmethod
    def from_cache(
        cls,
        cache_dir: str | Path,
        strict: bool = True,
        expected_config: AppConfig | DataConfig | None = None,
    ) -> "MarketDataBundle":
        root = Path(cache_dir)
        manifest_path = root / "manifest.json"
        membership_path = root / "membership.csv.gz"
        benchmark_path = root / "benchmark.csv.gz"
        regime_path = root / "regime.csv.gz"
        actions_path = root / "corporate_actions.csv.gz"
        securities_path = root / "securities.csv.gz"
        industries_path = root / "industry_membership.csv.gz"
        calendar_path = root / "calendar.csv.gz"
        bar_paths = sorted((root / "bars").glob("*.csv.gz"))
        if (
            not manifest_path.exists()
            or not membership_path.exists()
            or not benchmark_path.exists()
            or not regime_path.exists()
            or not actions_path.exists()
            or not securities_path.exists()
            or not industries_path.exists()
            or not calendar_path.exists()
            or not bar_paths
        ):
            raise FileNotFoundError(
                f"v4 缓存不完整: {root}。请设置 refresh=true 后重新运行 download。"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"缓存 manifest.json 不是有效 JSON: {root}") from exc
        if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"缓存结构版本为 {manifest.get('schema_version')!r}，"
                f"当前要求 v{CACHE_SCHEMA_VERSION}；请设置 refresh=true 后重新下载。"
            )
        if expected_config is not None:
            data_config = (
                expected_config.data
                if isinstance(expected_config, AppConfig)
                else expected_config
            )
            expected_fields = {
                "provider": data_config.provider,
                "universe_index": data_config.universe_index,
                "regime_index": data_config.regime_index,
                "benchmark_index": data_config.benchmark_index,
                "industry_standard": data_config.industry_standard,
                "industry_level": data_config.industry_level,
            }
            mismatches = {
                name: {"cache": manifest.get(name), "config": value}
                for name, value in expected_fields.items()
                if manifest.get(name) != value
            }
            if mismatches:
                raise ValueError(f"缓存数据身份与当前配置不匹配: {mismatches}")
            if isinstance(expected_config, AppConfig):
                required_start = pd.Timestamp(
                    expected_config.backtest.start_date
                ) - pd.Timedelta(data_config.warmup_calendar_days, unit="D")
                required_end = pd.Timestamp(expected_config.backtest.end_date)
                try:
                    cached_start = pd.Timestamp(manifest["requested_start"])
                    cached_end = pd.Timestamp(manifest["requested_end"])
                except (KeyError, ValueError) as exc:
                    raise ValueError("缓存 manifest 缺少有效请求日期范围") from exc
                if cached_start > required_start or cached_end < required_end:
                    raise ValueError(
                        "缓存日期范围不足: "
                        f"缓存={cached_start.date()}..{cached_end.date()}，"
                        f"需要={required_start.date()}..{required_end.date()}"
                    )
        verification = verify_file_inventory(root, manifest.get("files", []))
        expected = str(manifest.get("data_fingerprint_sha256", ""))
        if verification["inventory_sha256"] != expected:
            raise ValueError("缓存文件清单指纹与 manifest.json 不一致")
        recorded_paths = {
            str(item["path"])
            for item in manifest.get("files", [])
            if isinstance(item, dict) and "path" in item
        }
        consumed_paths = {
            path.resolve().relative_to(root.resolve()).as_posix()
            for path in [
                membership_path,
                benchmark_path,
                regime_path,
                actions_path,
                securities_path,
                industries_path,
                calendar_path,
                *bar_paths,
            ]
        }
        if recorded_paths != consumed_paths:
            unsealed = sorted(consumed_paths.difference(recorded_paths))
            unused = sorted(recorded_paths.difference(consumed_paths))
            raise ValueError(
                "缓存清单与实际读取文件集合不一致；"
                f"未封存={unsealed[:5]}，未使用={unused[:5]}"
            )
        bars = pd.concat((pd.read_csv(path) for path in bar_paths), ignore_index=True)
        membership = pd.read_csv(membership_path)
        benchmark = pd.read_csv(benchmark_path)
        regime = pd.read_csv(regime_path)
        actions = pd.read_csv(actions_path)
        securities = pd.read_csv(securities_path)
        industries = pd.read_csv(industries_path)
        calendar_frame = pd.read_csv(calendar_path)
        calendar = pd.DatetimeIndex(pd.to_datetime(calendar_frame["date"]))
        return cls(
            bars=bars,
            membership=membership,
            benchmark=benchmark,
            calendar=calendar,
            corporate_actions=actions,
            securities=securities,
            industry_membership=industries,
            regime=regime,
        ).prepare(strict=strict)


class RateLimitedTushare:
    def __init__(self, token: str, calls_per_minute: int, retries: int) -> None:
        try:
            import tushare as ts
        except ImportError as exc:
            raise RuntimeError("未安装 tushare，请先执行 pip install -r requirements.txt") from exc
        ts.set_token(token)
        self.pro = ts.pro_api(token)
        self.min_interval = 60.0 / max(1, calls_per_minute)
        self.retries = max(1, retries)
        self._last_call = 0.0

    def call(self, method: str, **kwargs: Any) -> pd.DataFrame:
        operation: Callable[..., pd.DataFrame] = getattr(self.pro, method)
        last_error: Exception | None = None
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                result = operation(**kwargs)
                self._last_call = time.monotonic()
                if result is None:
                    return pd.DataFrame()
                return result
            except Exception as exc:  # provider errors are not typed consistently
                self._last_call = time.monotonic()
                last_error = exc
                if attempt + 1 < self.retries:
                    delay = min(30.0, 2.0**attempt)
                    LOGGER.warning("Tushare %s 调用失败，%.1f 秒后重试: %s", method, delay, exc)
                    time.sleep(delay)
        raise RuntimeError(f"Tushare {method} 连续失败 {self.retries} 次") from last_error


class TushareDownloader:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        token = os.getenv(config.data.token_env, "").strip()
        if not token:
            raise RuntimeError(
                f"缺少环境变量 {config.data.token_env}。请把 Tushare Token 放入该环境变量，"
                "不要写入配置文件或提交到版本库。"
            )
        self.client = RateLimitedTushare(
            token=token,
            calls_per_minute=config.data.calls_per_minute,
            retries=config.data.retries,
        )
        self.root = Path(config.data.cache_dir)

    def download(self) -> MarketDataBundle:
        start = pd.Timestamp(self.config.backtest.start_date)
        end = pd.Timestamp(self.config.backtest.end_date)
        warmup_start = start - pd.Timedelta(
            self.config.data.warmup_calendar_days, unit="D"
        )
        manifest_path = self.root / "manifest.json"

        if not self.config.data.refresh and self._manifest_covers(manifest_path, warmup_start, end):
            LOGGER.info("命中完整本地缓存，跳过下载")
            return MarketDataBundle.from_cache(
                self.root,
                strict=self.config.data.strict_validation,
                expected_config=self.config,
            )

        self.root.mkdir(parents=True, exist_ok=True)
        LOGGER.info("下载交易日历")
        calendar = self._fetch_calendar(warmup_start, end)
        LOGGER.info("下载历史指数成分")
        membership = self._fetch_membership(start - pd.Timedelta(62, unit="D"), end)
        if membership.empty:
            raise RuntimeError("未取得指数成分数据，请检查指数代码和 Tushare 权限")
        _atomic_csv(membership, self.root / "membership.csv.gz")
        LOGGER.info("下载证券主表与退市日期")
        securities = self._fetch_securities()
        _atomic_csv(securities, self.root / "securities.csv.gz")

        symbols = sorted(membership["symbol"].unique())
        LOGGER.info("下载历史申万行业成员区间")
        industry_membership = self._fetch_industry_membership(symbols)
        _atomic_csv(industry_membership, self.root / "industry_membership.csv.gz")
        bar_dir = self.root / "bars"
        action_dir = self.root / "actions"
        bar_dir.mkdir(parents=True, exist_ok=True)
        action_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("需下载 %d 只历史成分股行情", len(symbols))
        for number, symbol in enumerate(symbols, start=1):
            target = bar_dir / f"{symbol.replace('.', '_')}.csv.gz"
            reuse_bar = (
                target.exists()
                and not self.config.data.refresh
                and self._cached_bar_usable(target, symbol, warmup_start, end, membership)
            )
            if reuse_bar:
                LOGGER.info("[%d/%d] 复用 %s", number, len(symbols), symbol)
            else:
                LOGGER.info("[%d/%d] 下载 %s", number, len(symbols), symbol)
                frame = self._fetch_symbol(symbol, warmup_start, end)
                if frame.empty:
                    LOGGER.warning("%s 在请求区间没有行情", symbol)
                    continue
                _atomic_csv(frame, target)
            action_target = action_dir / f"{symbol.replace('.', '_')}.csv.gz"
            actions = self._fetch_actions(symbol, warmup_start, end)
            _atomic_csv(actions, action_target)

        action_paths = sorted(action_dir.glob("*.csv.gz"))
        action_frames = [pd.read_csv(path) for path in action_paths]
        corporate_actions = (
            pd.concat(action_frames, ignore_index=True)
            if action_frames
            else pd.DataFrame(columns=ACTION_COLUMNS)
        )
        if not corporate_actions.empty:
            corporate_actions = corporate_actions.loc[
                corporate_actions["symbol"].isin(symbols)
            ]
        _atomic_csv(corporate_actions, self.root / "corporate_actions.csv.gz")

        LOGGER.info("下载全收益业绩基准行情")
        benchmark = self._fetch_benchmark(warmup_start, end)
        _atomic_csv(benchmark, self.root / "benchmark.csv.gz")
        LOGGER.info("下载价格择时指数行情")
        regime = self._fetch_regime(warmup_start, end)
        _atomic_csv(regime, self.root / "regime.csv.gz")
        _atomic_csv(pd.DataFrame({"date": calendar}), self.root / "calendar.csv.gz")
        cache_inputs = [
            self.root / "membership.csv.gz",
            self.root / "securities.csv.gz",
            self.root / "industry_membership.csv.gz",
            self.root / "corporate_actions.csv.gz",
            self.root / "benchmark.csv.gz",
            self.root / "regime.csv.gz",
            self.root / "calendar.csv.gz",
            *sorted(bar_dir.glob("*.csv.gz")),
        ]
        files = build_file_inventory(self.root, cache_inputs)
        _atomic_json(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "provider": "tushare",
                "universe_index": self.config.data.universe_index,
                "regime_index": self.config.data.regime_index,
                "benchmark_index": self.config.data.benchmark_index,
                "industry_standard": self.config.data.industry_standard,
                "industry_level": self.config.data.industry_level,
                "requested_start": warmup_start.strftime("%Y-%m-%d"),
                "requested_end": end.strftime("%Y-%m-%d"),
                "created_at_utc": pd.Timestamp.utcnow().isoformat(),
                "symbols": len(symbols),
                "files": files,
                "data_fingerprint_sha256": inventory_sha256(files),
            },
            manifest_path,
        )
        return MarketDataBundle.from_cache(
            self.root,
            strict=self.config.data.strict_validation,
            expected_config=self.config,
        )

    @staticmethod
    def _cached_bar_usable(
        path: Path,
        symbol: str,
        requested_start: pd.Timestamp,
        requested_end: pd.Timestamp,
        membership: pd.DataFrame,
    ) -> bool:
        try:
            cached = pd.read_csv(path, usecols=["date", "total_mv", "circ_mv"])
            dates = pd.to_datetime(cached["date"])
            if dates.empty:
                return False
            if cached[["total_mv", "circ_mv"]].isna().any().any():
                return False
            member_dates = membership.loc[membership["symbol"].eq(symbol), "date"]
            if member_dates.empty:
                return False
            first_needed = max(
                requested_start,
                pd.Timestamp(member_dates.min()) - pd.Timedelta(400, unit="D"),
            )
            last_needed = min(
                requested_end,
                pd.Timestamp(member_dates.max()) + pd.Timedelta(10, unit="D"),
            )
            start_ok = dates.min() <= first_needed + pd.Timedelta(10, unit="D")
            end_ok = dates.max() >= last_needed - pd.Timedelta(10, unit="D")
            return bool(start_ok and end_ok)
        except (OSError, ValueError, KeyError):
            return False

    def _manifest_covers(self, path: Path, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return (
                payload.get("schema_version") == CACHE_SCHEMA_VERSION
                and
                payload.get("provider") == "tushare"
                and payload.get("universe_index") == self.config.data.universe_index
                and payload.get("regime_index") == self.config.data.regime_index
                and payload.get("benchmark_index") == self.config.data.benchmark_index
                and payload.get("industry_standard") == self.config.data.industry_standard
                and payload.get("industry_level") == self.config.data.industry_level
                and pd.Timestamp(payload["requested_start"]) <= start
                and pd.Timestamp(payload["requested_end"]) >= end
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            return False

    def _fetch_securities(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        fields = "ts_code,name,industry,market,list_status,list_date,delist_date"
        for status in ["L", "D", "P"]:
            frame = self.client.call(
                "stock_basic", exchange="", list_status=status, fields=fields
            )
            if not frame.empty:
                frames.append(frame)
        if not frames:
            raise RuntimeError("证券主表为空")
        result = pd.concat(frames, ignore_index=True).rename(columns={"ts_code": "symbol"})
        for column in SECURITY_COLUMNS:
            if column not in result.columns:
                result[column] = np.nan
        return result[SECURITY_COLUMNS].drop_duplicates("symbol", keep="last")

    def _fetch_calendar(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        frame = self.client.call(
            "trade_cal",
            exchange="SSE",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            is_open="1",
            fields="cal_date,is_open",
        )
        if frame.empty:
            raise RuntimeError("交易日历为空")
        return pd.DatetimeIndex(pd.to_datetime(frame["cal_date"], format="%Y%m%d")).sort_values()

    def _fetch_membership(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for month in pd.period_range(start, end, freq="M"):
            frame = self.client.call(
                "index_weight",
                index_code=self.config.data.universe_index,
                start_date=month.start_time.strftime("%Y%m%d"),
                end_date=month.end_time.strftime("%Y%m%d"),
            )
            if not frame.empty:
                frames.append(frame[["trade_date", "con_code", "weight"]])
        if not frames:
            return pd.DataFrame(columns=["date", "symbol", "index_weight"])
        result = pd.concat(frames, ignore_index=True).rename(
            columns={"trade_date": "date", "con_code": "symbol", "weight": "index_weight"}
        )
        result["date"] = pd.to_datetime(result["date"], format="%Y%m%d")
        return result.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"])

    def _fetch_industry_membership(self, symbols: list[str]) -> pd.DataFrame:
        level = self.config.data.industry_level.upper()
        level_key = level.lower()
        classifications = self.client.call(
            "index_classify",
            level=level,
            src=self.config.data.industry_standard,
        )
        if classifications.empty or "index_code" not in classifications.columns:
            raise RuntimeError("申万行业分类为空，请检查 Tushare 权限")

        frames: list[pd.DataFrame] = []
        code_argument = f"{level_key}_code"
        for code in sorted(classifications["index_code"].dropna().astype(str).unique()):
            for is_new in ["Y", "N"]:
                frame = self.client.call(
                    "index_member_all", **{code_argument: code, "is_new": is_new}
                )
                if not frame.empty:
                    frames.append(frame)
        if not frames:
            raise RuntimeError("申万行业成员数据为空，请检查 Tushare 权限")

        raw = pd.concat(frames, ignore_index=True)
        wanted = set(map(str, symbols))
        raw = raw.loc[raw["ts_code"].astype(str).isin(wanted)].copy()

        # 某些历史证券不会出现在分类批量结果中；只对缺口做按证券补查。
        missing = sorted(wanted.difference(set(raw["ts_code"].astype(str))))
        for symbol in missing:
            for is_new in ["Y", "N"]:
                frame = self.client.call(
                    "index_member_all", ts_code=symbol, is_new=is_new
                )
                if not frame.empty:
                    raw = pd.concat([raw, frame], ignore_index=True)

        code_column = f"{level_key}_code"
        name_column = f"{level_key}_name"
        required = {"ts_code", code_column, name_column, "in_date"}
        absent = required.difference(raw.columns)
        if absent:
            raise RuntimeError(f"申万行业成员接口缺少字段: {sorted(absent)}")
        if "out_date" not in raw.columns:
            raw["out_date"] = np.nan
        result = raw.rename(
            columns={
                "ts_code": "symbol",
                code_column: "industry_code",
                name_column: "industry_name",
            }
        )
        result = result.loc[result["symbol"].astype(str).isin(wanted), INDUSTRY_COLUMNS]
        if result.empty:
            raise RuntimeError("历史成分股未匹配到申万行业成员记录")
        missing_after = sorted(wanted.difference(set(result["symbol"].astype(str))))
        if missing_after:
            raise RuntimeError(
                "申万行业成员缺少历史成分: " + ", ".join(missing_after[:10])
            )
        return result.drop_duplicates().reset_index(drop=True)

    def _fetch_symbol(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        common = {
            "ts_code": symbol,
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
        }
        daily = self.client.call(
            "daily",
            **common,
            fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
        )
        if daily.empty:
            return pd.DataFrame()
        factors = self.client.call(
            "adj_factor", **common, fields="ts_code,trade_date,adj_factor"
        )
        if factors.empty:
            raise RuntimeError(f"{symbol} 复权因子为空")
        limits = self.client.call(
            "stk_limit", **common, fields="ts_code,trade_date,up_limit,down_limit"
        )
        daily_basic = self.client.call(
            "daily_basic", **common, fields="ts_code,trade_date,total_mv,circ_mv"
        )
        if daily_basic.empty:
            raise RuntimeError(f"{symbol} 每日市值数据为空")
        names = self.client.call(
            "namechange", ts_code=symbol, fields="ts_code,name,start_date,end_date"
        )

        frame = daily.merge(factors, on=["ts_code", "trade_date"], how="left")
        if frame["adj_factor"].isna().any():
            missing_dates = frame.loc[frame["adj_factor"].isna(), "trade_date"].astype(str).tolist()
            raise RuntimeError(
                f"{symbol} 缺少 {len(missing_dates)} 个交易日的复权因子，"
                f"首个缺口为 {missing_dates[0]}"
            )
        if not limits.empty:
            frame = frame.merge(limits, on=["ts_code", "trade_date"], how="left")
        else:
            frame["up_limit"] = np.nan
            frame["down_limit"] = np.nan
        frame = frame.merge(daily_basic, on=["ts_code", "trade_date"], how="left")
        if frame[["total_mv", "circ_mv"]].isna().any().any():
            missing_count = int(frame[["total_mv", "circ_mv"]].isna().any(axis=1).sum())
            raise RuntimeError(f"{symbol} 缺少 {missing_count} 个交易日的市值数据")
        frame = frame.rename(
            columns={
                "ts_code": "symbol",
                "trade_date": "date",
                "pre_close": "prev_close",
                "vol": "volume",
            }
        )
        frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
        frame = frame.sort_values("date")
        frame["adj_factor"] = pd.to_numeric(frame["adj_factor"], errors="raise")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0) * 100.0
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0) * 1000.0
        frame["name"] = self._historical_names(frame["date"], names)
        frame["is_st"] = frame["name"].fillna("").str.contains(r"ST|退", case=False, regex=True)

        fallback_rate = frame.apply(
            lambda row: self._fallback_limit_rate(symbol, bool(row["is_st"])), axis=1
        )
        missing_up = frame["up_limit"].isna()
        missing_down = frame["down_limit"].isna()
        frame.loc[missing_up, "up_limit"] = np.round(
            frame.loc[missing_up, "prev_close"] * (1.0 + fallback_rate[missing_up]), 2
        )
        frame.loc[missing_down, "down_limit"] = np.round(
            frame.loc[missing_down, "prev_close"] * (1.0 - fallback_rate[missing_down]), 2
        )
        return frame[
            [
                "date",
                "symbol",
                "name",
                "open",
                "high",
                "low",
                "close",
                "prev_close",
                "volume",
                "amount",
                "adj_factor",
                "up_limit",
                "down_limit",
                "is_st",
                "total_mv",
                "circ_mv",
            ]
        ].reset_index(drop=True)

    def _fetch_actions(
        self, symbol: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        fields = (
            "ts_code,div_proc,record_date,ex_date,pay_date,div_listdate,"
            "cash_div,stk_div,imp_ann_date"
        )
        frame = self.client.call("dividend", ts_code=symbol, fields=fields)
        if frame.empty:
            return pd.DataFrame(columns=ACTION_COLUMNS)
        frame = frame.loc[frame["div_proc"].astype(str).eq("实施")].copy()
        frame["ex_date"] = pd.to_datetime(frame["ex_date"], format="%Y%m%d", errors="coerce")
        frame = frame.loc[frame["ex_date"].between(start, end)]
        if frame.empty:
            return pd.DataFrame(columns=ACTION_COLUMNS)
        frame = frame.rename(
            columns={
                "ts_code": "symbol",
                "div_listdate": "stock_list_date",
                "cash_div": "cash_dividend",
                "stk_div": "stock_dividend",
            }
        )
        for column in ["record_date", "pay_date", "stock_list_date"]:
            frame[column] = pd.to_datetime(
                frame[column], format="%Y%m%d", errors="coerce"
            )
        for column in ["cash_dividend", "stock_dividend"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        return frame[ACTION_COLUMNS].sort_values("ex_date").drop_duplicates(
            ["symbol", "ex_date"], keep="last"
        )

    @staticmethod
    def _historical_names(dates: pd.Series, names: pd.DataFrame) -> pd.Series:
        result = pd.Series("", index=dates.index, dtype="object")
        if names.empty:
            return result
        for row in names.itertuples(index=False):
            start_raw = getattr(row, "start_date", None)
            end_raw = getattr(row, "end_date", None)
            start = pd.Timestamp.min.normalize() if pd.isna(start_raw) else pd.to_datetime(str(start_raw))
            end = pd.Timestamp.max.normalize() if pd.isna(end_raw) else pd.to_datetime(str(end_raw))
            mask = dates.between(start, end)
            result.loc[mask] = str(getattr(row, "name", ""))
        return result

    @staticmethod
    def _fallback_limit_rate(symbol: str, is_st: bool) -> float:
        code, _, exchange = symbol.partition(".")
        if exchange == "BJ" or code.startswith(("4", "8", "92")):
            return 0.30
        if code.startswith(("300", "301", "688", "689")):
            return 0.20
        if is_st:
            return 0.05
        return 0.10

    def _fetch_benchmark(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return self._fetch_index(
            self.config.data.benchmark_index, start, end, label="业绩基准"
        )

    def _fetch_regime(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return self._fetch_index(
            self.config.data.regime_index, start, end, label="择时指数"
        )

    def _fetch_index(
        self,
        ts_code: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        label: str,
    ) -> pd.DataFrame:
        frame = self.client.call(
            "index_daily",
            ts_code=ts_code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            fields="ts_code,trade_date,open,high,low,close,pre_close,pct_chg",
        )
        if frame.empty:
            raise RuntimeError(f"{label}行情为空: {ts_code}")
        return (
            frame.rename(columns={"trade_date": "date", "pre_close": "prev_close"})
            .assign(date=lambda value: pd.to_datetime(value["date"], format="%Y%m%d"))
            .sort_values("date")
            .reset_index(drop=True)
        )


def make_demo_bundle(
    seed: int = 7,
    start: str = "2017-01-02",
    end: str = "2025-12-31",
    symbols: int = 50,
) -> MarketDataBundle:
    """Create deterministic, non-investable data for an offline smoke test."""
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range(start, end)
    count = len(calendar)
    market = rng.normal(0.00018, 0.009, count)
    factor_strength = np.linspace(-0.00012, 0.00028, symbols)
    frames: list[pd.DataFrame] = []

    for index in range(symbols):
        code = f"{600000 + index:06d}.SH" if index % 2 == 0 else f"{index + 1:06d}.SZ"
        idiosyncratic = rng.normal(0.0, 0.011 + index / symbols * 0.004, count)
        returns = 0.65 * market + factor_strength[index] + idiosyncratic
        close = 10.0 * np.exp(np.cumsum(returns))
        prev_close = np.r_[close[0] / (1.0 + returns[0]), close[:-1]]
        overnight = rng.normal(0.0, 0.0025, count)
        open_price = prev_close * np.exp(overnight)
        high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.0, 0.012, count))
        low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.0, 0.012, count))
        volume = rng.lognormal(mean=16.4, sigma=0.35, size=count)
        amount = volume * (open_price + close) / 2.0
        total_shares = 300_000_000.0 * np.exp(2.0 * index / max(1, symbols - 1))
        frame = pd.DataFrame(
            {
                "date": calendar,
                "symbol": code,
                "name": f"DEMO{index:02d}",
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "prev_close": prev_close,
                "volume": volume,
                "amount": amount,
                "adj_factor": 1.0,
                "up_limit": np.round(prev_close * 1.10, 2),
                "down_limit": np.round(prev_close * 0.90, 2),
                "is_st": False,
                "total_mv": close * total_shares / 10_000.0,
                "circ_mv": close * total_shares * 0.70 / 10_000.0,
            }
        )
        frames.append(frame)

    bars = pd.concat(frames, ignore_index=True)
    month_ends = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).max().tolist()
    membership = pd.DataFrame(
        [
            {"date": snapshot, "symbol": symbol, "index_weight": 100.0 / symbols}
            for snapshot in month_ends
            for symbol in sorted(bars["symbol"].unique())
        ]
    )
    benchmark_close = 1000.0 * np.exp(np.cumsum(market))
    benchmark = pd.DataFrame(
        {
            "date": calendar,
            "open": np.r_[benchmark_close[0], benchmark_close[:-1]],
            "high": benchmark_close * 1.005,
            "low": benchmark_close * 0.995,
            "close": benchmark_close,
            "prev_close": np.r_[benchmark_close[0], benchmark_close[:-1]],
        }
    )
    securities = pd.DataFrame(
        {
            "symbol": sorted(bars["symbol"].unique()),
            "name": [f"DEMO{index:02d}" for index in range(symbols)],
            "industry": [f"IND{index % 8}" for index in range(symbols)],
            "market": "DEMO",
            "list_status": "L",
            "list_date": pd.Timestamp(start) - pd.Timedelta(1000, unit="D"),
            "delist_date": pd.NaT,
        }
    )
    actions = pd.DataFrame(columns=ACTION_COLUMNS)
    industry_membership = pd.DataFrame(
        {
            "symbol": sorted(bars["symbol"].unique()),
            "industry_code": [f"801{index % 8:03d}.SI" for index in range(symbols)],
            "industry_name": [f"IND{index % 8}" for index in range(symbols)],
            "in_date": pd.Timestamp(start) - pd.Timedelta(1000, unit="D"),
            "out_date": pd.NaT,
        }
    )
    return MarketDataBundle(
        bars=bars,
        membership=membership,
        benchmark=benchmark,
        calendar=calendar,
        corporate_actions=actions,
        securities=securities,
        industry_membership=industry_membership,
        regime=benchmark.copy(),
    ).prepare()
