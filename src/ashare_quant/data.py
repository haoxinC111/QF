from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import AppConfig

LOGGER = logging.getLogger(__name__)

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
}


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


@dataclass
class MarketDataBundle:
    bars: pd.DataFrame
    membership: pd.DataFrame
    benchmark: pd.DataFrame
    calendar: pd.DatetimeIndex

    def prepare(self) -> "MarketDataBundle":
        bars = self.bars.copy()
        membership = self.membership.copy()
        benchmark = self.benchmark.copy()

        missing = BAR_COLUMNS.difference(bars.columns)
        if missing:
            raise ValueError(f"bars 缺少字段: {sorted(missing)}")
        for frame in (bars, membership, benchmark):
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

        bars = bars.sort_values(["symbol", "date"]).drop_duplicates(
            ["symbol", "date"], keep="last"
        )
        membership = membership.sort_values(["date", "symbol"]).drop_duplicates(
            ["date", "symbol"], keep="last"
        )
        benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last")

        if (bars[["open", "close", "adj_factor"]] <= 0).any().any():
            raise ValueError("价格和复权因子必须为正数")
        if bars.empty or membership.empty or benchmark.empty:
            raise ValueError("行情、成分或基准数据不能为空")

        calendar = pd.DatetimeIndex(pd.to_datetime(self.calendar)).normalize().unique().sort_values()
        if calendar.empty:
            calendar = pd.DatetimeIndex(benchmark["date"].unique()).sort_values()
        return MarketDataBundle(
            bars=bars.reset_index(drop=True),
            membership=membership.reset_index(drop=True),
            benchmark=benchmark.reset_index(drop=True),
            calendar=calendar,
        )

    @classmethod
    def from_cache(cls, cache_dir: str | Path) -> "MarketDataBundle":
        root = Path(cache_dir)
        membership_path = root / "membership.csv.gz"
        benchmark_path = root / "benchmark.csv.gz"
        bar_paths = sorted((root / "bars").glob("*.csv.gz"))
        if not membership_path.exists() or not benchmark_path.exists() or not bar_paths:
            raise FileNotFoundError(
                f"缓存不完整: {root}。请先运行 download，或运行 all 自动下载。"
            )
        bars = pd.concat((pd.read_csv(path) for path in bar_paths), ignore_index=True)
        membership = pd.read_csv(membership_path)
        benchmark = pd.read_csv(benchmark_path)
        calendar_path = root / "calendar.csv.gz"
        if calendar_path.exists():
            calendar_frame = pd.read_csv(calendar_path)
            calendar = pd.DatetimeIndex(pd.to_datetime(calendar_frame["date"]))
        else:
            calendar = pd.DatetimeIndex(pd.to_datetime(benchmark["date"]))
        return cls(bars, membership, benchmark, calendar).prepare()


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
        warmup_start = start - pd.Timedelta(days=self.config.data.warmup_calendar_days)
        manifest_path = self.root / "manifest.json"

        if not self.config.data.refresh and self._manifest_covers(manifest_path, warmup_start, end):
            LOGGER.info("命中完整本地缓存，跳过下载")
            return MarketDataBundle.from_cache(self.root)

        self.root.mkdir(parents=True, exist_ok=True)
        LOGGER.info("下载交易日历")
        calendar = self._fetch_calendar(warmup_start, end)
        LOGGER.info("下载历史指数成分")
        membership = self._fetch_membership(start - pd.Timedelta(days=62), end)
        if membership.empty:
            raise RuntimeError("未取得指数成分数据，请检查指数代码和 Tushare 权限")
        _atomic_csv(membership, self.root / "membership.csv.gz")

        symbols = sorted(membership["symbol"].unique())
        bar_dir = self.root / "bars"
        bar_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("需下载 %d 只历史成分股行情", len(symbols))
        for number, symbol in enumerate(symbols, start=1):
            target = bar_dir / f"{symbol.replace('.', '_')}.csv.gz"
            if (
                target.exists()
                and not self.config.data.refresh
                and self._cached_bar_usable(target, symbol, warmup_start, end, membership)
            ):
                LOGGER.info("[%d/%d] 复用 %s", number, len(symbols), symbol)
                continue
            LOGGER.info("[%d/%d] 下载 %s", number, len(symbols), symbol)
            frame = self._fetch_symbol(symbol, warmup_start, end)
            if frame.empty:
                LOGGER.warning("%s 在请求区间没有行情", symbol)
                continue
            _atomic_csv(frame, target)

        LOGGER.info("下载基准指数行情")
        benchmark = self._fetch_benchmark(warmup_start, end)
        _atomic_csv(benchmark, self.root / "benchmark.csv.gz")
        _atomic_csv(pd.DataFrame({"date": calendar}), self.root / "calendar.csv.gz")
        _atomic_json(
            {
                "schema_version": 1,
                "provider": "tushare",
                "universe_index": self.config.data.universe_index,
                "benchmark_index": self.config.data.benchmark_index,
                "requested_start": warmup_start.strftime("%Y-%m-%d"),
                "requested_end": end.strftime("%Y-%m-%d"),
                "created_at_utc": pd.Timestamp.utcnow().isoformat(),
                "symbols": len(symbols),
            },
            manifest_path,
        )
        return MarketDataBundle.from_cache(self.root)

    @staticmethod
    def _cached_bar_usable(
        path: Path,
        symbol: str,
        requested_start: pd.Timestamp,
        requested_end: pd.Timestamp,
        membership: pd.DataFrame,
    ) -> bool:
        try:
            dates = pd.to_datetime(pd.read_csv(path, usecols=["date"])["date"])
            if dates.empty:
                return False
            member_dates = membership.loc[membership["symbol"].eq(symbol), "date"]
            if member_dates.empty:
                return False
            first_needed = max(requested_start, pd.Timestamp(member_dates.min()) - pd.Timedelta(days=400))
            last_needed = min(requested_end, pd.Timestamp(member_dates.max()) + pd.Timedelta(days=10))
            start_ok = dates.min() <= first_needed + pd.Timedelta(days=10)
            end_ok = dates.max() >= last_needed - pd.Timedelta(days=10)
            return bool(start_ok and end_ok)
        except (OSError, ValueError, KeyError):
            return False

    def _manifest_covers(self, path: Path, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return (
                payload.get("provider") == "tushare"
                and payload.get("universe_index") == self.config.data.universe_index
                and payload.get("benchmark_index") == self.config.data.benchmark_index
                and pd.Timestamp(payload["requested_start"]) <= start
                and pd.Timestamp(payload["requested_end"]) >= end
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            return False

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
        limits = self.client.call(
            "stk_limit", **common, fields="ts_code,trade_date,up_limit,down_limit"
        )
        names = self.client.call(
            "namechange", ts_code=symbol, fields="ts_code,name,start_date,end_date"
        )

        frame = daily.merge(factors, on=["ts_code", "trade_date"], how="left")
        if not limits.empty:
            frame = frame.merge(limits, on=["ts_code", "trade_date"], how="left")
        else:
            frame["up_limit"] = np.nan
            frame["down_limit"] = np.nan
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
        frame["adj_factor"] = pd.to_numeric(frame["adj_factor"], errors="coerce").ffill().bfill()
        frame["adj_factor"] = frame["adj_factor"].fillna(1.0)
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
            ]
        ].reset_index(drop=True)

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
        frame = self.client.call(
            "index_daily",
            ts_code=self.config.data.benchmark_index,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            fields="ts_code,trade_date,open,high,low,close,pre_close,pct_chg",
        )
        if frame.empty:
            raise RuntimeError("基准指数行情为空")
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
    return MarketDataBundle(bars, membership, benchmark, calendar).prepare()
