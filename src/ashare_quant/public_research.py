from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .factors import _group_capped_allocation, _winsorized_zscore


LOGGER = logging.getLogger(__name__)
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
BAR_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude_pct",
    "pct_change",
    "change",
    "turnover_pct",
]


@dataclass(frozen=True)
class PublicDownloadConfig:
    start_date: str = "2012-01-01"
    end_date: str = "2025-12-31"
    source: str = "sina"
    workers: int = 6
    retries: int = 7
    timeout_seconds: float = 30.0
    request_pause_seconds: float = 0.15


@dataclass(frozen=True)
class PublicStrategyConfig:
    name: str = "baseline_v1_2_public"
    start_date: str = "2013-01-01"
    end_date: str = "2025-12-31"
    top_n: int = 20
    exit_rank: int = 35
    selection_buffer: bool = True
    min_history_days: int = 252
    min_avg_amount: float = 20_000_000.0
    min_price: float = 2.0
    stock_trend_filter: bool = True
    benchmark_ma_days: int = 200
    volatility_lookback: int = 60
    risk_on_exposure: float = 0.95
    risk_off_exposure: float = 0.30
    max_stock_weight: float = 0.08
    winsor_quantile: float = 0.05
    weight_mom_12_1: float = 0.35
    weight_mom_6_1: float = 0.20
    weight_trend: float = 0.15
    weight_low_volatility: float = 0.20
    weight_liquidity: float = 0.10
    commission_rate: float = 0.00025
    slippage_bps: float = 5.0
    annual_cash_rate: float = 0.015

    def validate(self) -> None:
        if pd.Timestamp(self.start_date) >= pd.Timestamp(self.end_date):
            raise ValueError("公开回测 start_date 必须早于 end_date")
        if self.top_n < 2 or self.exit_rank < self.top_n:
            raise ValueError("top_n 至少为 2，exit_rank 不得小于 top_n")
        if self.top_n * self.max_stock_weight + 1e-12 < self.risk_on_exposure:
            raise ValueError("top_n × max_stock_weight 无法容纳 risk_on_exposure")
        if not 0 <= self.risk_off_exposure <= self.risk_on_exposure <= 1:
            raise ValueError("风险仓位必须满足 0 <= risk_off <= risk_on <= 1")
        if self.factor_weights.sum() <= 0:
            raise ValueError("至少需要一个正的因子权重")

    @property
    def factor_weights(self) -> pd.Series:
        return pd.Series(
            {
                "mom_12_1": self.weight_mom_12_1,
                "mom_6_1": self.weight_mom_6_1,
                "trend": self.weight_trend,
                "low_volatility": self.weight_low_volatility,
                "liquidity": self.weight_liquidity,
            },
            dtype=float,
        )


@dataclass
class PublicBacktestResult:
    config: PublicStrategyConfig
    equity_curve: pd.DataFrame
    selections: pd.DataFrame
    rebalances: pd.DataFrame
    data_quality: dict[str, object]


def load_membership(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"symbol", "name", "opt-in", "opt-out"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"历史成分文件缺少字段: {sorted(missing)}")
    frame = frame.loc[:, ["symbol", "name", "opt-in", "opt-out"]].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["opt-in"] = pd.to_datetime(frame["opt-in"], errors="raise").dt.normalize()
    frame["opt-out"] = pd.to_datetime(frame["opt-out"], errors="coerce").dt.normalize()
    invalid = frame["opt-out"].notna() & frame["opt-out"].le(frame["opt-in"])
    if invalid.any():
        raise ValueError("历史成分存在 opt-out 不晚于 opt-in 的记录")
    return frame.sort_values(["symbol", "opt-in"]).reset_index(drop=True)


def members_at(membership: pd.DataFrame, when: pd.Timestamp | str) -> pd.DataFrame:
    when = pd.Timestamp(when).normalize()
    return membership.loc[
        membership["opt-in"].le(when)
        & (membership["opt-out"].isna() | membership["opt-out"].gt(when))
    ].drop_duplicates("symbol", keep="last")


def _eastmoney_secid(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol.startswith("SH"):
        return f"1.{symbol[2:]}"
    if symbol.startswith("SZ") or symbol.startswith("BJ"):
        return f"0.{symbol[2:]}"
    raise ValueError(f"不支持的证券代码: {symbol}")


def _request_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    retries: int,
    timeout: float,
    pause: float,
) -> pd.DataFrame:
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": "2",
        "secid": _eastmoney_secid(symbol),
        "beg": pd.Timestamp(start_date).strftime("%Y%m%d"),
        "end": pd.Timestamp(end_date).strftime("%Y%m%d"),
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ashare-quant-public-research/1.0)",
        "Referer": "https://quote.eastmoney.com/",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                EASTMONEY_KLINE_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data")
            if not data or not data.get("klines"):
                raise RuntimeError(f"{symbol} 返回空行情")
            rows = [value.split(",") for value in data["klines"]]
            frame = pd.DataFrame(rows, columns=BAR_COLUMNS)
            frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
            for column in BAR_COLUMNS[1:]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame.insert(1, "symbol", symbol)
            frame.insert(2, "name", str(data.get("name") or symbol))
            frame = frame.drop_duplicates("date", keep="last").sort_values("date")
            if frame[["open", "close", "high", "low"]].isna().any().any():
                raise RuntimeError(f"{symbol} OHLC 含非法值")
            if (frame[["open", "close", "high", "low"]] <= 0).any().any():
                raise RuntimeError(f"{symbol} OHLC 含非正数")
            return frame.reset_index(drop=True)
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 >= retries:
                break
            delay = min(12.0, 0.8 * (2**attempt)) + random.random() * 0.5
            time.sleep(delay)
        finally:
            if pause > 0:
                time.sleep(pause)
    raise RuntimeError(f"下载 {symbol} 失败，已重试 {retries} 次: {last_error}")


def _request_sina_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    retries: int,
    timeout: float,
    pause: float,
) -> pd.DataFrame:
    """Fetch the compact Sina history blob and apply its point-in-date HFQ factors."""
    try:
        from akshare.stock.cons import hk_js_decode
        from py_mini_racer import py_mini_racer
    except ImportError as exc:
        raise RuntimeError(
            "新浪公开源需要可选依赖，请先运行 pip install 'akshare>=1.18,<2'"
        ) from exc

    sina_symbol = symbol.lower()
    history_url = (
        "https://finance.sina.com.cn/realstock/company/"
        f"{sina_symbol}/hisdata_klc2/klc_kl.js"
    )
    factor_url = f"https://finance.sina.com.cn/realstock/company/{sina_symbol}/hfq.js"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ashare-quant-public-research/1.0)",
        "Referer": f"https://finance.sina.com.cn/realstock/company/{sina_symbol}/nc.shtml",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(history_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            encoded = response.text.split("=", maxsplit=1)[1].split(";", maxsplit=1)[0]
            runtime = py_mini_racer.MiniRacer()
            runtime.eval(hk_js_decode)
            decoded = runtime.call("d", encoded.replace('"', ""))
            frame = pd.DataFrame(decoded)
            if frame.empty:
                raise RuntimeError(f"{symbol} 返回空行情")
            frame["date"] = (
                pd.to_datetime(frame["date"], errors="raise")
                .dt.tz_localize(None)
                .dt.normalize()
                .astype("datetime64[ns]")
            )
            for column in ["open", "high", "low", "close", "volume", "amount"]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.loc[:, ["date", "open", "close", "high", "low", "volume", "amount"]]
            frame = frame.sort_values("date").drop_duplicates("date", keep="last")

            # The CSI price index has no corporate-action factor. Stocks and ETFs must
            # have a factor so splits/dividends are included in total-return ratios.
            if symbol != "SH000300":
                factor_response = requests.get(factor_url, headers=headers, timeout=timeout)
                factor_response.raise_for_status()
                factor_text = factor_response.text.split("=", maxsplit=1)[1].split("\n", maxsplit=1)[0]
                factor_payload = json.loads(factor_text.replace("'", '"'))
                factor = pd.DataFrame(factor_payload.get("data", []))
                if factor.empty:
                    raise RuntimeError(f"{symbol} 缺少新浪后复权因子")
                factor = factor.rename(columns={"d": "date", "f": "hfq_factor"})
                if not {"date", "hfq_factor"}.issubset(factor.columns):
                    raise RuntimeError(f"{symbol} 新浪后复权因子结构异常")
                factor = factor.loc[:, ["date", "hfq_factor"]]
                factor["date"] = (
                    pd.to_datetime(factor["date"], errors="raise")
                    .dt.normalize()
                    .astype("datetime64[ns]")
                )
                factor["hfq_factor"] = pd.to_numeric(factor["hfq_factor"], errors="coerce")
                factor = factor.sort_values("date").drop_duplicates("date", keep="last")
                frame = pd.merge_asof(frame, factor, on="date", direction="backward")
                if frame["hfq_factor"].isna().any():
                    raise RuntimeError(f"{symbol} 早期行情缺少后复权因子覆盖")
                for column in ["open", "close", "high", "low"]:
                    frame[column] = frame[column] * frame["hfq_factor"]
                frame = frame.drop(columns="hfq_factor")

            frame = frame.loc[
                frame["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
            ].copy()
            if frame.empty:
                raise RuntimeError(f"{symbol} 在请求区间内没有行情")
            frame.insert(1, "symbol", symbol)
            frame.insert(2, "name", symbol)
            frame["amplitude_pct"] = np.nan
            frame["pct_change"] = frame["close"].pct_change(fill_method=None) * 100.0
            frame["change"] = frame["close"].diff()
            frame["turnover_pct"] = np.nan
            frame = frame.loc[:, ["date", "symbol", "name", *BAR_COLUMNS[1:]]]
            if frame[["open", "close", "high", "low"]].isna().any().any():
                raise RuntimeError(f"{symbol} OHLC 含非法值")
            if (frame[["open", "close", "high", "low"]] <= 0).any().any():
                raise RuntimeError(f"{symbol} OHLC 含非正数")
            return frame.reset_index(drop=True)
        except (requests.RequestException, ValueError, RuntimeError, IndexError, KeyError) as exc:
            last_error = exc
            if attempt + 1 >= retries:
                break
            delay = min(15.0, 1.0 * (2**attempt)) + random.random() * 0.8
            time.sleep(delay)
        finally:
            if pause > 0:
                time.sleep(pause)
    raise RuntimeError(f"下载 {symbol} 失败，已重试 {retries} 次: {last_error}")


def _bar_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / "bars" / f"{symbol}.csv.gz"


def _valid_cached_bars(path: Path, start_date: str, end_date: str) -> bool:
    if not path.exists() or path.stat().st_size < 100:
        return False
    try:
        frame = pd.read_csv(path, usecols=["date", "symbol", "open", "close"])
        if frame.empty or frame[["open", "close"]].isna().any().any():
            return False
        dates = pd.to_datetime(frame["date"], errors="raise")
        requested_start = pd.Timestamp(start_date)
        requested_end = pd.Timestamp(end_date)
        # Newly listed/delisted stocks need not span the complete requested interval.
        return bool(dates.min() <= requested_end and dates.max() >= requested_start)
    except (OSError, ValueError, KeyError):
        return False


def download_public_history(
    membership_path: str | Path,
    cache_dir: str | Path,
    config: PublicDownloadConfig | None = None,
    *,
    force: bool = False,
) -> dict[str, object]:
    config = config or PublicDownloadConfig()
    if config.source not in {"sina", "eastmoney"}:
        raise ValueError("公开行情 source 仅支持 sina 或 eastmoney")
    if config.workers < 1 or config.retries < 1 or config.timeout_seconds <= 0:
        raise ValueError("workers/retries/timeout_seconds 必须为正数")
    membership = load_membership(membership_path)
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    if start >= end:
        raise ValueError("下载 start_date 必须早于 end_date")
    cache_dir = Path(cache_dir).resolve()
    bars_dir = cache_dir / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)

    relevant = membership.loc[
        membership["opt-in"].le(end)
        & (membership["opt-out"].isna() | membership["opt-out"].ge(start))
    ]
    symbols = sorted(relevant["symbol"].unique())
    # ETF is the dividend-adjusted investable benchmark; index is the regime signal.
    requested = symbols + ["SH000300", "SH510300"]
    pending = [
        symbol
        for symbol in requested
        if force or not _valid_cached_bars(_bar_path(cache_dir, symbol), config.start_date, config.end_date)
    ]
    LOGGER.info(
        "公开数据计划: %d 个历史成分，%d 个待下载，%d 个已缓存",
        len(symbols),
        len(pending),
        len(requested) - len(pending),
    )

    failed: dict[str, str] = {}
    completed = 0

    def worker(symbol: str) -> tuple[str, int]:
        fetcher = _request_sina_bars if config.source == "sina" else _request_bars
        frame = fetcher(
            symbol,
            config.start_date,
            config.end_date,
            retries=config.retries,
            timeout=config.timeout_seconds,
            pause=config.request_pause_seconds,
        )
        path = _bar_path(cache_dir, symbol)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False, compression="gzip", date_format="%Y-%m-%d")
        temporary.replace(path)
        return symbol, len(frame)

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(worker, symbol): symbol for symbol in pending}
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                _, rows = future.result()
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    LOGGER.info("行情下载进度: %d/%d（最近 %s, %d 行）", completed, len(pending), symbol, rows)
            except Exception as exc:  # noqa: BLE001 - preserve per-symbol failure and continue
                failed[symbol] = str(exc)
                LOGGER.warning("%s", failed[symbol])

    manifest = {
        "source": f"{config.source} public daily history via HTTPS",
        "source_url": (
            "https://finance.sina.com.cn/realstock/company/"
            if config.source == "sina"
            else EASTMONEY_KLINE_URL
        ),
        "membership_source": str(Path(membership_path).resolve()),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "download_config": asdict(config),
        "historical_constituent_count": len(symbols),
        "requested_count": len(requested),
        "available_count": sum(_valid_cached_bars(_bar_path(cache_dir, symbol), config.start_date, config.end_date) for symbol in requested),
        "failed": failed,
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if "SH000300" in failed or "SH510300" in failed:
        raise RuntimeError(f"基准数据下载失败: {failed}")
    return manifest


def _load_cached_bars(cache_dir: str | Path, symbols: Iterable[str]) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    frames: list[pd.DataFrame] = []
    for symbol in sorted(set(symbols)):
        path = _bar_path(cache_dir, symbol)
        if not path.exists():
            continue
        frame = pd.read_csv(
            path,
            usecols=["date", "symbol", "name", "open", "close", "amount"],
            parse_dates=["date"],
        )
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"没有找到公开行情缓存: {cache_dir}")
    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["symbol"] = result["symbol"].astype(str)
    return result.sort_values(["symbol", "date"]).reset_index(drop=True)


def _build_features(bars: pd.DataFrame, volatility_lookback: int) -> pd.DataFrame:
    frame = bars.copy().sort_values(["symbol", "date"])
    grouped = frame.groupby("symbol", sort=False, group_keys=False)
    frame["return_1d"] = grouped["close"].pct_change(fill_method=None)
    frame["mom_12_1"] = grouped["close"].transform(
        lambda value: value.shift(21) / value.shift(252) - 1.0
    )
    frame["mom_6_1"] = grouped["close"].transform(
        lambda value: value.shift(21) / value.shift(126) - 1.0
    )
    frame["ma_200"] = grouped["close"].transform(
        lambda value: value.rolling(200, min_periods=200).mean()
    )
    frame["trend"] = frame["close"] / frame["ma_200"] - 1.0
    frame["volatility"] = grouped["return_1d"].transform(
        lambda value: value.rolling(
            volatility_lookback, min_periods=max(20, volatility_lookback - 10)
        ).std(ddof=0)
        * math.sqrt(252.0)
    )
    frame["avg_amount_20"] = grouped["amount"].transform(
        lambda value: value.rolling(20, min_periods=15).mean()
    )
    frame["history_days"] = grouped.cumcount() + 1
    return frame


def _fee_rates(when: pd.Timestamp, config: PublicStrategyConfig) -> tuple[float, float]:
    stamp = 0.0005 if when >= pd.Timestamp("2023-08-28") else 0.0010
    transfer = 0.00001 if when >= pd.Timestamp("2022-04-29") else 0.00002
    common = config.commission_rate + config.slippage_bps / 10_000.0 + transfer
    return common, common + stamp


def _calculate_metrics(curve: pd.DataFrame, column: str = "nav") -> dict[str, float | int]:
    data = curve.loc[:, ["date", column]].dropna().copy()
    if len(data) < 2:
        return {
            "start": str(data["date"].min().date()) if len(data) else "",
            "end": str(data["date"].max().date()) if len(data) else "",
            "trading_days": len(data),
            "total_return": float("nan"),
            "cagr": float("nan"),
            "annual_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
        }
    values = data[column].astype(float)
    returns = values.pct_change(fill_method=None).dropna()
    years = max((data["date"].iloc[-1] - data["date"].iloc[0]).days / 365.2425, 1 / 252)
    total_return = values.iloc[-1] / values.iloc[0] - 1.0
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0
    volatility = returns.std(ddof=0) * math.sqrt(252.0)
    sharpe = returns.mean() / returns.std(ddof=0) * math.sqrt(252.0) if returns.std(ddof=0) > 0 else float("nan")
    drawdown = values / values.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    return {
        "start": str(data["date"].iloc[0].date()),
        "end": str(data["date"].iloc[-1].date()),
        "trading_days": len(data),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_volatility": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar": float(cagr / abs(max_drawdown)) if max_drawdown < 0 else float("nan"),
    }


def _select_targets(
    exact: pd.DataFrame,
    membership: pd.DataFrame,
    when: pd.Timestamp,
    held_symbols: set[str],
    config: PublicStrategyConfig,
    benchmark_row: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame, str]:
    active_frame = members_at(membership, when)
    active = set(active_frame["symbol"].astype(str))
    candidates = exact.loc[exact["symbol"].isin(active)].copy()
    candidate_names = active_frame.set_index("symbol")["name"].astype(str)
    candidates["name"] = candidates["symbol"].map(candidate_names).fillna(candidates["name"])
    candidates = candidates.loc[
        (candidates["history_days"] >= config.min_history_days)
        & (candidates["avg_amount_20"] >= config.min_avg_amount)
        & (candidates["close"] >= config.min_price)
    ]
    required = ["mom_12_1", "mom_6_1", "trend", "volatility", "avg_amount_20"]
    candidates = candidates.dropna(subset=required)
    if config.stock_trend_filter:
        candidates = candidates.loc[candidates["trend"] > 0]
    if len(candidates) < config.top_n:
        return {}, pd.DataFrame(), "INSUFFICIENT_CANDIDATES"

    candidates["low_volatility"] = -candidates["volatility"]
    candidates["liquidity"] = np.log1p(candidates["avg_amount_20"])
    weights = config.factor_weights
    weights = weights / weights.sum()
    score = pd.Series(0.0, index=candidates.index)
    for factor, weight in weights.items():
        column = f"z_{factor}"
        candidates[column] = _winsorized_zscore(candidates[factor], config.winsor_quantile)
        score += candidates[column] * float(weight)
    candidates["score"] = score
    ranked = candidates.sort_values(["score", "symbol"], ascending=[False, True]).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["was_held"] = ranked["symbol"].isin(held_symbols)

    kept: list[int] = []
    if config.selection_buffer and held_symbols:
        kept = list(
            ranked.loc[ranked["was_held"] & ranked["rank"].le(config.exit_rank)]
            .head(config.top_n)
            .index
        )
    kept_set = set(kept)
    fill = [index for index in ranked.index if index not in kept_set][
        : max(0, config.top_n - len(kept))
    ]
    selected = ranked.loc[kept + fill].sort_values(["rank", "symbol"]).copy()
    selected["selection_reason"] = np.where(
        selected.index.isin(kept_set), "HOLD_BUFFER", "NEW_ENTRY"
    )

    risk_on = bool(pd.notna(benchmark_row.get("ma")) and benchmark_row["close"] >= benchmark_row["ma"])
    regime = "RISK_ON" if risk_on else "RISK_OFF"
    exposure = config.risk_on_exposure if risk_on else config.risk_off_exposure
    inverse_vol = 1.0 / selected.set_index("symbol")["volatility"].clip(lower=1e-8)
    target = _group_capped_allocation(
        inverse_vol,
        total=exposure,
        stock_cap=config.max_stock_weight,
    )
    selected["target_weight"] = selected["symbol"].map(target).fillna(0.0)
    selected.insert(0, "signal_date", when)
    selected["regime"] = regime
    selected["target_exposure"] = exposure
    return target.to_dict(), selected.reset_index(drop=True), regime


def run_public_backtest(
    membership_path: str | Path,
    cache_dir: str | Path,
    config: PublicStrategyConfig | None = None,
) -> PublicBacktestResult:
    config = config or PublicStrategyConfig()
    config.validate()
    membership = load_membership(membership_path)
    relevant = membership.loc[
        membership["opt-in"].le(pd.Timestamp(config.end_date))
        & (membership["opt-out"].isna() | membership["opt-out"].gt(pd.Timestamp(config.start_date)))
    ]
    symbols = sorted(relevant["symbol"].unique())
    bars = _load_cached_bars(cache_dir, symbols)
    features = _build_features(bars, config.volatility_lookback)

    cache_dir = Path(cache_dir)
    index_bars = _load_cached_bars(cache_dir, ["SH000300"]).sort_values("date")
    benchmark_bars = _load_cached_bars(cache_dir, ["SH510300"]).sort_values("date")
    index_bars["ma"] = index_bars["close"].rolling(
        config.benchmark_ma_days, min_periods=config.benchmark_ma_days
    ).mean()
    index_by_date = index_bars.set_index("date")

    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    calendar = index_bars.loc[index_bars["date"].between(start, end), "date"].drop_duplicates()
    if calendar.empty:
        raise ValueError("回测区间没有沪深300交易日")
    calendar = pd.DatetimeIndex(calendar.sort_values())
    signal_dates = set(
        pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).max().tolist()
    )

    daily = features.loc[features["date"].between(calendar.min(), calendar.max())]
    rows_by_date = {date: group.set_index("symbol") for date, group in daily.groupby("date", sort=False)}
    exact_by_date = {date: group for date, group in features.groupby("date", sort=False) if date in signal_dates}
    benchmark = benchmark_bars.set_index("date").reindex(calendar)
    benchmark["close"] = benchmark["close"].ffill()
    benchmark["benchmark_nav"] = benchmark["close"] / benchmark["close"].iloc[0]

    positions: dict[str, float] = {}
    cash = 1.0
    pending: tuple[pd.Timestamp, dict[str, float], str] | None = None
    curve_rows: list[dict[str, object]] = []
    selection_frames: list[pd.DataFrame] = []
    rebalance_rows: list[dict[str, object]] = []
    missing_execution_symbols = 0
    stale_marks = 0
    prior_close: dict[str, float] = {}
    cash_daily = (1.0 + config.annual_cash_rate) ** (1.0 / 252.0) - 1.0

    for date in calendar:
        today = rows_by_date.get(date)
        if today is None:
            continue

        # Mark yesterday's positions to today's open; suspension/missing bars carry last value.
        for symbol in list(positions):
            if symbol in today.index and symbol in prior_close:
                open_price = float(today.at[symbol, "open"])
                if open_price > 0 and prior_close[symbol] > 0:
                    positions[symbol] *= open_price / prior_close[symbol]
            elif symbol not in today.index:
                stale_marks += 1

        cash *= 1.0 + cash_daily
        if pending is not None:
            signal_date, requested_target, regime = pending
            nav_open = cash + sum(positions.values())
            executable = {
                symbol: weight
                for symbol, weight in requested_target.items()
                if symbol in today.index and float(today.at[symbol, "open"]) > 0
            }
            missing_execution_symbols += len(requested_target) - len(executable)
            target_values = {symbol: nav_open * weight for symbol, weight in executable.items()}
            # A suspended existing position cannot be sold at a fictitious carried
            # price. Keep it unchanged; a suspended new target stays as cash.
            locked_positions = {
                symbol: value for symbol, value in positions.items() if symbol not in today.index
            }
            target_values.update(locked_positions)
            all_symbols = set(positions) | set(target_values)
            buys = sum(max(0.0, target_values.get(symbol, 0.0) - positions.get(symbol, 0.0)) for symbol in all_symbols)
            sells = sum(max(0.0, positions.get(symbol, 0.0) - target_values.get(symbol, 0.0)) for symbol in all_symbols)
            buy_rate, sell_rate = _fee_rates(date, config)
            cost = buys * buy_rate + sells * sell_rate
            positions = {symbol: value for symbol, value in target_values.items() if value > 1e-12}
            cash = nav_open - sum(positions.values()) - cost
            if cash < -1e-9:
                scale = max(0.0, (nav_open - cost) / max(sum(positions.values()), 1e-12))
                positions = {symbol: value * scale for symbol, value in positions.items()}
                cash = nav_open - sum(positions.values()) - cost
            rebalance_rows.append(
                {
                    "signal_date": signal_date,
                    "execution_date": date,
                    "regime": regime,
                    "buy_turnover": buys / nav_open if nav_open else 0.0,
                    "sell_turnover": sells / nav_open if nav_open else 0.0,
                    "one_way_turnover": (buys + sells) / (2.0 * nav_open) if nav_open else 0.0,
                    "cost_rate": cost / nav_open if nav_open else 0.0,
                    "holding_count": len(positions),
                    "target_exposure": sum(target_values.values()) / nav_open if nav_open else 0.0,
                }
            )
            pending = None

        # Mark positions from open to close.
        for symbol in list(positions):
            if symbol in today.index:
                open_price = float(today.at[symbol, "open"])
                close_price = float(today.at[symbol, "close"])
                if open_price > 0 and close_price > 0:
                    positions[symbol] *= close_price / open_price

        nav_close = cash + sum(positions.values())
        curve_rows.append(
            {
                "date": date,
                "nav": nav_close,
                "cash": cash,
                "holdings_value": sum(positions.values()),
                "holding_count": len(positions),
                "benchmark_nav": float(benchmark.at[date, "benchmark_nav"]),
            }
        )

        if date in signal_dates:
            exact = exact_by_date.get(date, pd.DataFrame())
            if not exact.empty and date in index_by_date.index:
                target, selection, regime = _select_targets(
                    exact,
                    membership,
                    date,
                    {symbol for symbol, value in positions.items() if value / max(nav_close, 1e-12) > 1e-5},
                    config,
                    index_by_date.loc[date],
                )
                if target:
                    pending = (date, target, regime)
                    selection_frames.append(selection)

        for symbol in today.index:
            close_price = float(today.at[symbol, "close"])
            if close_price > 0:
                prior_close[str(symbol)] = close_price

    curve = pd.DataFrame(curve_rows)
    if curve.empty:
        raise RuntimeError("公开数据回测没有生成净值")
    curve["nav"] /= float(curve["nav"].iloc[0])
    curve["drawdown"] = curve["nav"] / curve["nav"].cummax() - 1.0
    curve["benchmark_drawdown"] = curve["benchmark_nav"] / curve["benchmark_nav"].cummax() - 1.0
    selections = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    rebalances = pd.DataFrame(rebalance_rows)
    available = set(bars["symbol"].unique())
    monthly_coverage: list[float] = []
    monthly_member_counts: list[int] = []
    for signal_date in sorted(signal_dates):
        active = set(members_at(membership, signal_date)["symbol"].astype(str))
        quoted = set(exact_by_date.get(signal_date, pd.DataFrame()).get("symbol", pd.Series(dtype=str)).astype(str))
        monthly_member_counts.append(len(active))
        monthly_coverage.append(len(active.intersection(quoted)) / max(len(active), 1))
    manifest_path = Path(cache_dir) / "manifest.json"
    cache_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    quality = {
        "membership_intervals": len(membership),
        "relevant_unique_members": len(symbols),
        "members_with_bars": len(available.intersection(symbols)),
        "member_bar_coverage": len(available.intersection(symbols)) / max(len(symbols), 1),
        "month_end_quote_coverage_min": min(monthly_coverage) if monthly_coverage else 0.0,
        "month_end_quote_coverage_median": float(np.median(monthly_coverage)) if monthly_coverage else 0.0,
        "month_end_members_min": min(monthly_member_counts) if monthly_member_counts else 0,
        "month_end_members_max": max(monthly_member_counts) if monthly_member_counts else 0,
        "bar_rows": len(bars),
        "first_bar_date": str(bars["date"].min().date()),
        "last_bar_date": str(bars["date"].max().date()),
        "missing_execution_symbols": missing_execution_symbols,
        "stale_position_marks": stale_marks,
        "adjustment": f"{cache_manifest.get('source', 'public cache')} hfq total-return price",
        "known_limitations": [
            "历史成分来自第三方对中证官方公告的规范化整理，并非中证公司原始机器接口",
            "公开接口没有可靠的历史 ST、历史行业和历史市值快照，因此本测试关闭对应过滤/中性化",
            "权重级回测未模拟 100 股整手、最低 5 元佣金和涨跌停排队；执行日停牌时保留旧持仓、新目标留现金且不重试",
            "沪深300ETF后复权净值作为可投资基准，含基金费率和跟踪误差",
        ],
    }
    return PublicBacktestResult(config, curve, selections, rebalances, quality)


def public_strategy_variants(base: PublicStrategyConfig | None = None) -> list[PublicStrategyConfig]:
    base = base or PublicStrategyConfig()
    return [
        base,
        replace(
            base,
            name="momentum_focus_15",
            top_n=15,
            exit_rank=25,
            max_stock_weight=0.09,
            risk_off_exposure=0.0,
            weight_mom_12_1=0.50,
            weight_mom_6_1=0.30,
            weight_trend=0.15,
            weight_low_volatility=0.05,
            weight_liquidity=0.0,
        ),
        replace(
            base,
            name="balanced_momentum_20",
            risk_off_exposure=0.0,
            weight_mom_12_1=0.45,
            weight_mom_6_1=0.25,
            weight_trend=0.15,
            weight_low_volatility=0.15,
            weight_liquidity=0.0,
        ),
        replace(
            base,
            name="low_volatility_30",
            top_n=30,
            exit_rank=45,
            max_stock_weight=0.05,
            risk_off_exposure=0.20,
            weight_mom_12_1=0.25,
            weight_mom_6_1=0.10,
            weight_trend=0.10,
            weight_low_volatility=0.50,
            weight_liquidity=0.05,
        ),
        replace(
            base,
            name="concentrated_momentum_10",
            top_n=10,
            exit_rank=18,
            max_stock_weight=0.12,
            risk_on_exposure=0.98,
            risk_off_exposure=0.0,
            weight_mom_12_1=0.55,
            weight_mom_6_1=0.30,
            weight_trend=0.15,
            weight_low_volatility=0.0,
            weight_liquidity=0.0,
        ),
        replace(
            base,
            name="medium_momentum_15",
            top_n=15,
            exit_rank=25,
            max_stock_weight=0.09,
            risk_on_exposure=0.98,
            risk_off_exposure=0.15,
            weight_mom_12_1=0.30,
            weight_mom_6_1=0.45,
            weight_trend=0.15,
            weight_low_volatility=0.10,
            weight_liquidity=0.0,
        ),
    ]


def _slice_metrics(curve: pd.DataFrame, start: str, end: str, column: str) -> dict[str, float | int]:
    subset = curve.loc[curve["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    return _calculate_metrics(subset, column)


def write_public_research(
    membership_path: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    base_config: PublicStrategyConfig | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = base_config or PublicStrategyConfig()
    metric_rows: list[dict[str, object]] = []
    results: dict[str, PublicBacktestResult] = {}
    periods = {
        "development_2013_2017": ("2013-01-01", "2017-12-31"),
        "validation_2018_2021": ("2018-01-01", "2021-12-31"),
        "oos_2022_2025": ("2022-01-01", "2025-12-31"),
        "full_2013_2025": (base_config.start_date, base_config.end_date),
    }
    for variant in public_strategy_variants(base_config):
        LOGGER.info("运行公开数据策略: %s", variant.name)
        result = run_public_backtest(membership_path, cache_dir, variant)
        results[variant.name] = result
        variant_dir = output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(variant_dir / "equity_curve.csv", index=False)
        result.selections.to_csv(variant_dir / "selections.csv", index=False)
        result.rebalances.to_csv(variant_dir / "rebalances.csv", index=False)
        (variant_dir / "config.json").write_text(
            json.dumps(asdict(variant), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (variant_dir / "data_quality.json").write_text(
            json.dumps(result.data_quality, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for period, (start, end) in periods.items():
            strategy_metrics = _slice_metrics(result.equity_curve, start, end, "nav")
            benchmark_metrics = _slice_metrics(result.equity_curve, start, end, "benchmark_nav")
            turnover = result.rebalances.loc[
                pd.to_datetime(result.rebalances["execution_date"]).between(start, end),
                "one_way_turnover",
            ].sum()
            years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.2425, 1.0)
            metric_rows.append(
                {
                    "strategy": variant.name,
                    "period": period,
                    **strategy_metrics,
                    "benchmark_cagr": benchmark_metrics["cagr"],
                    "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
                    "annual_one_way_turnover": float(turnover / years),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "period_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 6.5))
    for name, result in results.items():
        axis.plot(result.equity_curve["date"], result.equity_curve["nav"], label=name, linewidth=1.35)
    first_result = next(iter(results.values()))
    axis.plot(
        first_result.equity_curve["date"],
        first_result.equity_curve["benchmark_nav"],
        label="CSI300 ETF total-return benchmark",
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    axis.set_title("Public-data strategy comparison (2013–2025)")
    axis.set_ylabel("NAV")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    chart_path = output_dir / "strategy_comparison.png"
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)

    def percentage(value: object) -> str:
        return "—" if pd.isna(value) else f"{float(value):.2%}"

    report_lines = [
        "# A股公开数据分段回测报告",
        "",
        "本报告使用历史时点沪深300成分区间、公开日线后复权价格、月末信号和下一交易日开盘成交。所有结果均已计入双边佣金、历史卖出印花税、过户费和单边滑点。",
        "",
        "## 分期结果",
        "",
        "| 策略 | 区间 | 年化 | 最大回撤 | 夏普 | 基准年化 | 年化单边换手 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        report_lines.append(
            f"| {row.strategy} | {row.period} | {percentage(row.cagr)} | "
            f"{percentage(row.max_drawdown)} | {row.sharpe:.2f} | "
            f"{percentage(row.benchmark_cagr)} | {percentage(row.annual_one_way_turnover)} |"
        )
    oos = metrics.loc[metrics["period"].eq("oos_2022_2025")].sort_values("cagr", ascending=False)
    best_oos = oos.iloc[0]
    report_lines.extend(
        [
            "",
            "## 结果判读",
            "",
            f"- 冻结样本外期年化最高的是 `{best_oos['strategy']}`：{percentage(best_oos['cagr'])}，最大回撤 {percentage(best_oos['max_drawdown'])}。这只是预定义方案之间的比较，不是收益承诺。",
            f"- 样本外是否达到 15%：{'是' if float(best_oos['cagr']) >= 0.15 else '否'}。未达到时不通过扩大杠杆或回看样本外调参来硬凑目标。",
            "- 开发期、验证期与样本外期应一起看；只在单一区间突出的方案应视为不稳定。",
            "",
            "## 公开数据口径限制",
            "",
        ]
    )
    for limitation in first_result.data_quality["known_limitations"]:
        report_lines.append(f"- {limitation}")
    report_lines.append(
        "- 月末成分当日行情覆盖最低为 "
        f"{first_result.data_quality['month_end_quote_coverage_min']:.2%}、中位数为 "
        f"{first_result.data_quality['month_end_quote_coverage_median']:.2%}；最低点发生在集中停牌时期，"
        "无当日收盘价的证券不会进入当月新选择。"
    )
    report_lines.extend(
        [
            "",
            "严格的交易可实现性结论仍需使用项目的 Tushare 通道复核；公开通道更适合判断因子方向、区间稳定性和成本敏感性。",
        ]
    )
    report_path = output_dir / "PUBLIC_BACKTEST_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "membership_path": str(Path(membership_path).resolve()),
        "cache_dir": str(Path(cache_dir).resolve()),
        "periods": periods,
        "selection_protocol": (
            "六组参数均在运行前预定义；开发期仅用于比较，不根据 2022-2025 样本外结果反向改参。"
        ),
        "variants": [asdict(value) for value in public_strategy_variants(base_config)],
        "limitations": next(iter(results.values())).data_quality["known_limitations"],
    }
    manifest_path = output_dir / "research_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "report": report_path,
        "metrics": metrics_path,
        "chart": chart_path,
        "manifest": manifest_path,
        "output_dir": output_dir,
    }


def write_public_robustness(
    membership_path: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    base_config: PublicStrategyConfig | None = None,
) -> dict[str, Path]:
    """Cost stress and factor ablation for the predeclared momentum candidate."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = base_config or PublicStrategyConfig()
    candidate = next(
        variant for variant in public_strategy_variants(base_config) if variant.name == "momentum_focus_15"
    )
    periods = {
        "oos_2022_2025": ("2022-01-01", "2025-12-31"),
        "full_2013_2025": (candidate.start_date, candidate.end_date),
    }

    cost_rows: list[dict[str, object]] = []
    for slippage in [5.0, 10.0, 20.0]:
        stressed = replace(candidate, name=f"momentum_focus_15_slippage_{slippage:g}", slippage_bps=slippage)
        LOGGER.info("公开数据成本压力: %.0f bps", slippage)
        result = run_public_backtest(membership_path, cache_dir, stressed)
        for period, (start, end) in periods.items():
            cost_rows.append(
                {
                    "slippage_bps": slippage,
                    "period": period,
                    **_slice_metrics(result.equity_curve, start, end, "nav"),
                }
            )
    cost_frame = pd.DataFrame(cost_rows)
    cost_path = output_dir / "cost_stress.csv"
    cost_frame.to_csv(cost_path, index=False)

    factor_fields = {
        "mom_12_1": "weight_mom_12_1",
        "mom_6_1": "weight_mom_6_1",
        "trend": "weight_trend",
        "low_volatility": "weight_low_volatility",
    }
    ablation_configs = [("full", candidate)] + [
        (
            f"without_{factor}",
            replace(candidate, name=f"momentum_focus_15_without_{factor}", **{field: 0.0}),
        )
        for factor, field in factor_fields.items()
    ]
    ablation_rows: list[dict[str, object]] = []
    for case, ablation_config in ablation_configs:
        LOGGER.info("公开数据因子消融: %s", case)
        result = run_public_backtest(membership_path, cache_dir, ablation_config)
        for period, (start, end) in periods.items():
            ablation_rows.append(
                {
                    "case": case,
                    "period": period,
                    **_slice_metrics(result.equity_curve, start, end, "nav"),
                }
            )
    ablation_frame = pd.DataFrame(ablation_rows)
    ablation_path = output_dir / "factor_ablation.csv"
    ablation_frame.to_csv(ablation_path, index=False)

    lines = [
        "# 公开数据稳健性检查",
        "",
        "候选参数在查看冻结样本外结果前已固定为 `momentum_focus_15`。本页只提高成本和删除因子，不根据结果重新优化权重。",
        "",
        "## 滑点压力",
        "",
        "| 单边滑点 | 区间 | 年化 | 最大回撤 | 夏普 |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in cost_frame.itertuples(index=False):
        lines.append(
            f"| {row.slippage_bps:.0f} bps | {row.period} | {row.cagr:.2%} | "
            f"{row.max_drawdown:.2%} | {row.sharpe:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 因子消融",
            "",
            "| 方案 | 区间 | 年化 | 最大回撤 | 夏普 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in ablation_frame.itertuples(index=False):
        lines.append(
            f"| {row.case} | {row.period} | {row.cagr:.2%} | "
            f"{row.max_drawdown:.2%} | {row.sharpe:.2f} |"
        )
    report_path = output_dir / "PUBLIC_ROBUSTNESS_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": report_path, "cost": cost_path, "ablation": ablation_path}
