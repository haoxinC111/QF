from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import queue
import random
import threading
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import requests

from .alpha import (
    ALPHA_MODEL_VERSION,
    DEFAULT_ALPHA_PROFILE,
    QUALITY_MOMENTUM_V1_5_WEIGHTS,
    alpha_profile_governance,
    build_price_alpha_features,
)
from .execution import (
    DEFAULT_EXECUTION_MODEL,
    EXECUTION_MODEL_V1_6,
    SUPPORTED_EXECUTION_MODELS,
    execution_model_governance,
    market_impact_bps,
)
from .factors import _winsorized_zscore
from .portfolio import (
    DEFAULT_PORTFOLIO_MODEL,
    PORTFOLIO_MODEL_V1_6,
    SUPPORTED_PORTFOLIO_MODELS,
    allocate_portfolio,
    portfolio_model_governance,
)
from .provenance import (
    build_file_inventory,
    build_reproducibility_manifest,
    file_fingerprint,
    inventory_sha256,
    record_experiment,
    verify_file_inventory,
    write_artifact_manifest,
    write_json_atomic,
)


LOGGER = logging.getLogger(__name__)
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
PUBLIC_CACHE_SCHEMA_VERSION = 1
PUBLIC_REGIME_SYMBOL = "SH000300"
PUBLIC_PERFORMANCE_BENCHMARK_SYMBOL = "SH510300"
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


try:
    from akshare.stock.cons import hk_js_decode
except ImportError:  # pragma: no cover - optional dependency
    hk_js_decode = None  # type: ignore[assignment]

try:
    from py_mini_racer import MiniRacer
except ImportError:  # pragma: no cover - optional dependency
    try:
        from py_mini_racer import py_mini_racer as _py_mini_racer

        MiniRacer = _py_mini_racer.MiniRacer  # type: ignore[misc]
    except ImportError:
        MiniRacer = None  # type: ignore[misc,assignment]

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
class _SinaPayload:
    symbol: str
    start_date: str
    end_date: str
    encoded_history: str
    factor_payload: dict[str, object] | None


@dataclass(frozen=True)
class _SinaDecodeTask:
    encoded_history: str
    future: concurrent.futures.Future[object]


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
    weight_fip_momentum: float = 0.0
    weight_trend: float = 0.15
    weight_low_volatility: float = 0.20
    weight_low_downside_volatility: float = 0.0
    weight_drawdown_quality: float = 0.0
    weight_liquidity: float = 0.10
    portfolio_model: str = DEFAULT_PORTFOLIO_MODEL
    covariance_lookback_days: int = 120
    minimum_covariance_observations: int = 60
    covariance_shrinkage: float = 0.50
    minimum_variance_blend: float = 0.50
    turnover_smoothing: float = 0.50
    covariance_ridge: float = 1e-6
    commission_rate: float = 0.00025
    slippage_bps: float = 5.0
    market_impact_model: str = DEFAULT_EXECUTION_MODEL
    market_impact_coefficient: float = 0.50
    market_impact_volatility_floor: float = 0.10
    max_market_impact_bps: float = 50.0
    initial_capital: float = 1_000_000.0
    annual_cash_rate: float = 0.015
    minimum_month_end_quote_coverage: float = 0.95
    regime_symbol: str = PUBLIC_REGIME_SYMBOL
    performance_benchmark_symbol: str = PUBLIC_PERFORMANCE_BENCHMARK_SYMBOL

    def validate(self) -> None:
        if pd.Timestamp(self.start_date) >= pd.Timestamp(self.end_date):
            raise ValueError("公开回测 start_date 必须早于 end_date")
        if self.top_n < 2 or self.exit_rank < self.top_n:
            raise ValueError("top_n 至少为 2，exit_rank 不得小于 top_n")
        if self.top_n * self.max_stock_weight + 1e-12 < self.risk_on_exposure:
            raise ValueError("top_n × max_stock_weight 无法容纳 risk_on_exposure")
        if not 0 <= self.risk_off_exposure <= self.risk_on_exposure <= 1:
            raise ValueError("风险仓位必须满足 0 <= risk_off <= risk_on <= 1")
        weights = self.factor_weights
        if (~np.isfinite(weights)).any() or (weights < 0).any():
            raise ValueError("公开策略因子权重必须是非负有限数")
        if weights.sum() <= 0:
            raise ValueError("至少需要一个正的因子权重")
        if self.volatility_lookback < 20:
            raise ValueError("volatility_lookback 至少为 20")
        if self.portfolio_model not in SUPPORTED_PORTFOLIO_MODELS:
            raise ValueError("公开组合模型不受支持: " + self.portfolio_model)
        if self.market_impact_model not in SUPPORTED_EXECUTION_MODELS:
            raise ValueError("公开成交模型不受支持: " + self.market_impact_model)
        if not (
            20
            <= self.minimum_covariance_observations
            <= self.covariance_lookback_days
        ):
            raise ValueError("公开协方差最低观测数必须在 20 与回看天数之间")
        for name, value in {
            "covariance_shrinkage": self.covariance_shrinkage,
            "minimum_variance_blend": self.minimum_variance_blend,
            "turnover_smoothing": self.turnover_smoothing,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} 必须在 [0, 1] 内")
        if not np.isfinite(self.covariance_ridge) or self.covariance_ridge <= 0:
            raise ValueError("covariance_ridge 必须是正有限数")
        execution_values = [
            self.commission_rate,
            self.slippage_bps,
            self.market_impact_coefficient,
            self.market_impact_volatility_floor,
            self.max_market_impact_bps,
        ]
        if any(not np.isfinite(value) or value < 0 for value in execution_values):
            raise ValueError("公开费用和冲击参数必须是非负有限数")
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital 必须是正有限数")
        if not 0 < self.minimum_month_end_quote_coverage <= 1:
            raise ValueError("minimum_month_end_quote_coverage 必须在 (0, 1] 内")
        if not self.regime_symbol or not self.performance_benchmark_symbol:
            raise ValueError("择时指数与业绩基准代码不能为空")
        if self.regime_symbol == self.performance_benchmark_symbol:
            raise ValueError("公开通道的择时指数与业绩基准必须分离")
        if (
            self.regime_symbol != PUBLIC_REGIME_SYMBOL
            or self.performance_benchmark_symbol
            != PUBLIC_PERFORMANCE_BENCHMARK_SYMBOL
        ):
            raise ValueError("公开下载器目前只支持沪深300价格指数与510300ETF基准")

    @property
    def factor_weights(self) -> pd.Series:
        return pd.Series(
            {
                "mom_12_1": self.weight_mom_12_1,
                "mom_6_1": self.weight_mom_6_1,
                "fip_momentum": self.weight_fip_momentum,
                "trend": self.weight_trend,
                "low_volatility": self.weight_low_volatility,
                "low_downside_vol": self.weight_low_downside_volatility,
                "drawdown_quality": self.weight_drawdown_quality,
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


@dataclass(frozen=True)
class _PublicRebalance:
    positions: dict[str, float]
    cash: float
    buys: float
    sells: float
    cost: float
    market_impact_cost: float
    locked_value: float
    executable_scale: float


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


class _SinaDecoder:
    """Own exactly one MiniRacer runtime on one dedicated decoder thread."""

    _STOP = object()

    def __init__(
        self,
        *,
        queue_size: int = 16,
        operation_timeout_seconds: float = 120.0,
        runtime_factory: Callable[[], object] | None = None,
        decode_script: str | None = None,
    ) -> None:
        if queue_size < 1 or operation_timeout_seconds <= 0:
            raise ValueError("解码队列容量和超时必须为正数")
        if runtime_factory is None:
            if MiniRacer is None:
                raise RuntimeError(
                    "新浪公开源需要可选依赖，请先运行 pip install 'akshare>=1.18,<2'"
                )
            runtime_factory = MiniRacer
        if decode_script is None:
            if hk_js_decode is None:
                raise RuntimeError(
                    "新浪公开源需要 akshare.stock.cons.hk_js_decode"
                )
            decode_script = hk_js_decode
        self._runtime_factory = runtime_factory
        self._decode_script = decode_script
        self._operation_timeout_seconds = operation_timeout_seconds
        self._queue: queue.Queue[_SinaDecodeTask | object] = queue.Queue(
            maxsize=queue_size
        )
        self._ready: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._thread = threading.Thread(
            target=self._run,
            name="sina-mini-racer-decoder",
            daemon=True,
        )
        self._state_lock = threading.Lock()
        self._enqueue_lock = threading.Lock()
        self._started = False
        self._closed = False
        self._fatal_error: BaseException | None = None

    def __enter__(self) -> _SinaDecoder:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        self.close(suppress_error=exc_type is not None)
        return False

    def start(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("新浪解码器已经关闭")
            if not self._started:
                self._started = True
                self._thread.start()
        try:
            self._ready.result(timeout=self._operation_timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            raise RuntimeError("新浪解码器启动超时") from exc

    def decode(self, encoded_history: str) -> object:
        if not encoded_history:
            raise ValueError("新浪压缩行情不能为空")
        self.start()
        future: concurrent.futures.Future[object] = concurrent.futures.Future()
        task = _SinaDecodeTask(encoded_history=encoded_history, future=future)
        with self._enqueue_lock:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("新浪解码器已经关闭")
                fatal_error = self._fatal_error
                if fatal_error is not None:
                    raise RuntimeError("新浪解码线程已经异常退出") from fatal_error
            try:
                # Serialize this enqueue with close() marking the decoder closed so
                # the stop sentinel can never overtake an accepted task.
                self._queue.put(task, timeout=self._operation_timeout_seconds)
            except queue.Full as exc:
                raise RuntimeError("新浪解码队列写入超时") from exc
        try:
            return future.result(timeout=self._operation_timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            raise RuntimeError("新浪 JavaScript 解码超时") from exc

    def close(self, *, suppress_error: bool = False) -> None:
        with self._enqueue_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                started = self._started
            if not started:
                return
            try:
                self._queue.put(self._STOP, timeout=self._operation_timeout_seconds)
            except queue.Full:
                if not suppress_error:
                    raise RuntimeError("新浪解码器关闭时队列阻塞")
                return
        self._thread.join(timeout=self._operation_timeout_seconds)
        if self._thread.is_alive():
            if not suppress_error:
                raise RuntimeError("新浪解码线程未能在超时内退出")
            return
        if self._fatal_error is not None and not suppress_error:
            raise RuntimeError("新浪解码线程异常退出") from self._fatal_error

    def _run(self) -> None:
        try:
            with ExitStack() as stack:
                candidate = self._runtime_factory()
                if callable(getattr(candidate, "__enter__", None)) and callable(
                    getattr(candidate, "__exit__", None)
                ):
                    runtime = stack.enter_context(candidate)  # type: ignore[arg-type]
                else:
                    runtime = candidate
                    close = getattr(candidate, "close", None)
                    if callable(close):
                        stack.callback(close)
                runtime.eval(self._decode_script)  # type: ignore[attr-defined]
                self._ready.set_result(None)
                while True:
                    item = self._queue.get()
                    if item is self._STOP:
                        break
                    task = item
                    try:
                        decoded = runtime.call(  # type: ignore[attr-defined]
                            "d", task.encoded_history
                        )
                    except Exception as exc:  # isolate one malformed payload
                        task.future.set_exception(exc)
                    else:
                        task.future.set_result(decoded)
        except BaseException as exc:
            with self._state_lock:
                self._fatal_error = exc
            if not self._ready.done():
                self._ready.set_exception(exc)
            self._fail_pending(exc)
        finally:
            if not self._ready.done():
                error = RuntimeError("新浪解码线程在初始化前退出")
                self._ready.set_exception(error)
                with self._state_lock:
                    self._fatal_error = error

    def _fail_pending(self, error: BaseException) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not self._STOP and not item.future.done():
                item.future.set_exception(error)


def _request_sina_payload(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    retries: int,
    timeout: float,
    pause: float,
) -> _SinaPayload:
    """Fetch raw Sina history/factor payloads without touching the V8 runtime."""
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
            factor_payload: dict[str, object] | None = None
            if symbol != PUBLIC_REGIME_SYMBOL:
                factor_response = requests.get(factor_url, headers=headers, timeout=timeout)
                factor_response.raise_for_status()
                factor_text = factor_response.text.split("=", maxsplit=1)[1].split("\n", maxsplit=1)[0]
                factor_payload = json.loads(factor_text.replace("'", '"'))
                if not isinstance(factor_payload, dict):
                    raise RuntimeError(f"{symbol} 新浪后复权因子结构异常")
            return _SinaPayload(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                encoded_history=encoded.replace('"', ""),
                factor_payload=factor_payload,
            )
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


def _build_sina_bars(payload: _SinaPayload, decoded: object) -> pd.DataFrame:
    """Transform decoded Sina rows on the requesting worker thread."""
    symbol = payload.symbol
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
    frame = frame.loc[
        :, ["date", "open", "close", "high", "low", "volume", "amount"]
    ]
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")

    # The CSI price index has no corporate-action factor. Stocks and ETFs must
    # have a factor so splits/dividends are included in total-return ratios.
    if symbol != PUBLIC_REGIME_SYMBOL:
        factor_payload = payload.factor_payload or {}
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
        factor["hfq_factor"] = pd.to_numeric(
            factor["hfq_factor"], errors="coerce"
        )
        factor = factor.sort_values("date").drop_duplicates("date", keep="last")
        frame = pd.merge_asof(frame, factor, on="date", direction="backward")
        if frame["hfq_factor"].isna().any():
            raise RuntimeError(f"{symbol} 早期行情缺少后复权因子覆盖")
        for column in ["open", "close", "high", "low"]:
            frame[column] = frame[column] * frame["hfq_factor"]
        frame = frame.drop(columns="hfq_factor")

    frame = frame.loc[
        frame["date"].between(
            pd.Timestamp(payload.start_date), pd.Timestamp(payload.end_date)
        )
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


def _request_sina_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    retries: int,
    timeout: float,
    pause: float,
    decoder: _SinaDecoder | None = None,
) -> pd.DataFrame:
    payload = _request_sina_payload(
        symbol,
        start_date,
        end_date,
        retries=retries,
        timeout=timeout,
        pause=pause,
    )
    if decoder is not None:
        return _build_sina_bars(payload, decoder.decode(payload.encoded_history))
    with _SinaDecoder(queue_size=1) as local_decoder:
        return _build_sina_bars(
            payload, local_decoder.decode(payload.encoded_history)
        )


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
    requested = symbols + [PUBLIC_REGIME_SYMBOL, PUBLIC_PERFORMANCE_BENCHMARK_SYMBOL]
    recorded_paths: set[str] | None = None
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("公开缓存 manifest.json 不是有效 JSON") from exc
        if (
            existing_manifest.get("schema_version") == PUBLIC_CACHE_SCHEMA_VERSION
            and existing_manifest.get("files")
            and existing_manifest.get("membership_file")
        ):
            # Verify every previously sealed byte before a resumable download. Files
            # left by an interrupted run are allowed here, but are forced to download
            # again below rather than being trusted and silently added to the seal.
            verified = verify_public_cache(
                membership_path,
                cache_dir,
                _allow_unsealed_files=True,
            )
            recorded_paths = {
                str(item["path"])
                for item in verified["files"]  # type: ignore[union-attr]
                if isinstance(item, dict) and "path" in item
            }
    pending = [
        symbol
        for symbol in requested
        if force
        or (
            recorded_paths is not None
            and _bar_path(cache_dir, symbol)
            .resolve()
            .relative_to(cache_dir)
            .as_posix()
            not in recorded_paths
        )
        or not _valid_cached_bars(
            _bar_path(cache_dir, symbol), config.start_date, config.end_date
        )
    ]
    LOGGER.info(
        "公开数据计划: %d 个历史成分，%d 个待下载，%d 个已缓存",
        len(symbols),
        len(pending),
        len(requested) - len(pending),
    )

    failed: dict[str, str] = {}
    completed = 0
    decoder: _SinaDecoder | None = None

    def worker(symbol: str) -> tuple[str, int]:
        if config.source == "sina":
            if decoder is None:
                raise RuntimeError("新浪单线程解码器未启动")
            frame = _request_sina_bars(
                symbol,
                config.start_date,
                config.end_date,
                retries=config.retries,
                timeout=config.timeout_seconds,
                pause=config.request_pause_seconds,
                decoder=decoder,
            )
        else:
            frame = _request_bars(
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

    with ExitStack() as stack:
        if config.source == "sina" and pending:
            decoder = stack.enter_context(
                _SinaDecoder(
                    queue_size=max(4, config.workers * 2),
                    operation_timeout_seconds=max(
                        120.0, config.timeout_seconds * 2.0
                    ),
                )
            )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.workers
        ) as executor:
            futures = {executor.submit(worker, symbol): symbol for symbol in pending}
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                try:
                    _, rows = future.result()
                    completed += 1
                    if completed % 25 == 0 or completed == len(pending):
                        LOGGER.info(
                            "行情下载进度: %d/%d（最近 %s, %d 行）",
                            completed,
                            len(pending),
                            symbol,
                            rows,
                        )
                except Exception as exc:  # preserve per-symbol failure and continue
                    failed[symbol] = str(exc)
                    LOGGER.warning("%s", failed[symbol])

    available_symbols = [
        symbol
        for symbol in requested
        if _valid_cached_bars(
            _bar_path(cache_dir, symbol), config.start_date, config.end_date
        )
    ]
    invalid_existing = [
        symbol
        for symbol in requested
        if _bar_path(cache_dir, symbol).is_file() and symbol not in available_symbols
    ]
    if invalid_existing:
        raise RuntimeError(
            "公开缓存存在未通过结构/日期校验的行情文件，拒绝写入新指纹: "
            + ", ".join(invalid_existing[:10])
        )
    all_cache_symbols = sorted(
        set(membership["symbol"].astype(str))
        | {PUBLIC_REGIME_SYMBOL, PUBLIC_PERFORMANCE_BENCHMARK_SYMBOL}
    )
    sealed_paths = [
        _bar_path(cache_dir, symbol)
        for symbol in all_cache_symbols
        if _bar_path(cache_dir, symbol).is_file()
    ]
    files = build_file_inventory(
        cache_dir, sealed_paths
    )
    membership_file = file_fingerprint(
        membership_path, logical_path="membership.csv"
    )
    manifest = {
        "schema_version": PUBLIC_CACHE_SCHEMA_VERSION,
        "source": f"{config.source} public daily history via HTTPS",
        "source_url": (
            "https://finance.sina.com.cn/realstock/company/"
            if config.source == "sina"
            else EASTMONEY_KLINE_URL
        ),
        "membership_source": str(Path(membership_path).resolve()),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "download_config": asdict(config),
        "decoder_architecture": (
            "single_dedicated_thread_bounded_queue"
            if config.source == "sina"
            else None
        ),
        "historical_constituent_count": len(symbols),
        "requested_count": len(requested),
        "available_count": len(available_symbols),
        "sealed_file_count": len(files),
        "failed": failed,
        "regime_symbol": PUBLIC_REGIME_SYMBOL,
        "performance_benchmark_symbol": PUBLIC_PERFORMANCE_BENCHMARK_SYMBOL,
        "membership_file": membership_file,
        "files": files,
        "data_fingerprint_sha256": inventory_sha256([membership_file, *files]),
    }
    write_json_atomic(manifest, manifest_path)
    if (
        PUBLIC_REGIME_SYMBOL in failed
        or PUBLIC_PERFORMANCE_BENCHMARK_SYMBOL in failed
    ):
        raise RuntimeError(f"基准数据下载失败: {failed}")
    return manifest


def seal_public_cache(
    membership_path: str | Path,
    cache_dir: str | Path,
) -> dict[str, object]:
    """Create v1.4 fingerprints for a legacy public cache without redownloading."""
    membership = load_membership(membership_path)
    cache_dir = Path(cache_dir).resolve()
    expected = sorted(
        set(membership["symbol"].astype(str))
        | {PUBLIC_REGIME_SYMBOL, PUBLIC_PERFORMANCE_BENCHMARK_SYMBOL}
    )
    paths = [_bar_path(cache_dir, symbol) for symbol in expected]
    available = [path for path in paths if path.is_file()]
    if not available:
        raise FileNotFoundError(f"没有找到可封存的公开行情缓存: {cache_dir}")
    missing_benchmarks = [
        symbol
        for symbol in [PUBLIC_REGIME_SYMBOL, PUBLIC_PERFORMANCE_BENCHMARK_SYMBOL]
        if not _bar_path(cache_dir, symbol).is_file()
    ]
    if missing_benchmarks:
        raise FileNotFoundError(
            "公开缓存缺少择时/业绩基准: " + ", ".join(missing_benchmarks)
        )

    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest: dict[str, object] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError("公开缓存 manifest.json 不是有效 JSON") from exc
    else:
        manifest = {}
    if (
        manifest.get("schema_version") == PUBLIC_CACHE_SCHEMA_VERSION
        and manifest.get("files")
        and manifest.get("membership_file")
    ):
        raise ValueError(
            "公开缓存已经封存；为防止把篡改后的文件重新合法化，只允许执行校验"
        )
    files = build_file_inventory(cache_dir, available)
    membership_file = file_fingerprint(
        membership_path, logical_path="membership.csv"
    )
    manifest.update(
        {
            "schema_version": PUBLIC_CACHE_SCHEMA_VERSION,
            "sealed_at_utc": datetime.now(UTC).isoformat(),
            "membership_source": str(Path(membership_path).resolve()),
            "requested_count": len(expected),
            "available_count": len(available),
            "regime_symbol": PUBLIC_REGIME_SYMBOL,
            "performance_benchmark_symbol": PUBLIC_PERFORMANCE_BENCHMARK_SYMBOL,
            "membership_file": membership_file,
            "files": files,
            "data_fingerprint_sha256": inventory_sha256(
                [membership_file, *files]
            ),
        }
    )
    write_json_atomic(manifest, manifest_path)
    return manifest


def verify_public_cache(
    membership_path: str | Path,
    cache_dir: str | Path,
    *,
    seal_legacy: bool = False,
    _allow_unsealed_files: bool = False,
) -> dict[str, object]:
    cache_dir = Path(cache_dir).resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        if seal_legacy:
            return seal_public_cache(membership_path, cache_dir)
        raise FileNotFoundError(f"公开缓存缺少 manifest.json: {cache_dir}")
    try:
        manifest: dict[str, object] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError("公开缓存 manifest.json 不是有效 JSON") from exc
    if (
        manifest.get("schema_version") != PUBLIC_CACHE_SCHEMA_VERSION
        or not manifest.get("files")
        or not manifest.get("membership_file")
    ):
        if seal_legacy:
            return seal_public_cache(membership_path, cache_dir)
        raise ValueError("公开缓存尚未生成 v1.4 文件指纹")

    verification = verify_file_inventory(cache_dir, manifest["files"])  # type: ignore[arg-type]
    membership = load_membership(membership_path)
    expected_symbols = sorted(
        set(membership["symbol"].astype(str))
        | {PUBLIC_REGIME_SYMBOL, PUBLIC_PERFORMANCE_BENCHMARK_SYMBOL}
    )
    current_paths = {
        _bar_path(cache_dir, symbol).resolve().relative_to(cache_dir).as_posix()
        for symbol in expected_symbols
        if _bar_path(cache_dir, symbol).is_file()
    }
    recorded_paths = {
        str(item["path"])
        for item in manifest["files"]  # type: ignore[union-attr]
        if isinstance(item, dict) and "path" in item
    }
    unsealed_paths = sorted(current_paths.difference(recorded_paths))
    missing_paths = sorted(recorded_paths.difference(current_paths))
    if (unsealed_paths or missing_paths) and not _allow_unsealed_files:
        raise ValueError("公开缓存清单与当前会读取的行情文件集合不一致")
    current_membership = file_fingerprint(
        membership_path, logical_path="membership.csv"
    )
    recorded_membership = manifest["membership_file"]
    if not isinstance(recorded_membership, dict):
        raise ValueError("公开缓存的历史成分文件指纹结构异常")
    if (
        current_membership["size_bytes"] != recorded_membership.get("size_bytes")
        or current_membership["sha256"] != recorded_membership.get("sha256")
    ):
        raise ValueError("当前历史成分文件与公开缓存封存版本不一致")
    combined = inventory_sha256([current_membership, *manifest["files"]])  # type: ignore[list-item]
    if combined != manifest.get("data_fingerprint_sha256"):
        raise ValueError("公开缓存总数据指纹与 manifest.json 不一致")
    return {
        **manifest,
        "verification": {
            **verification,
            "unsealed_paths": unsealed_paths,
            "missing_paths": missing_paths,
        },
    }


def _load_cached_bars(cache_dir: str | Path, symbols: Iterable[str]) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    frames: list[pd.DataFrame] = []
    for symbol in sorted(set(symbols)):
        path = _bar_path(cache_dir, symbol)
        if not path.exists():
            continue
        frame = pd.read_csv(
            path,
            usecols=[
                "date",
                "symbol",
                "name",
                "open",
                "close",
                "volume",
                "amount",
            ],
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
    frame = build_price_alpha_features(
        bars,
        price_column="close",
        volatility_lookback=volatility_lookback,
    )
    grouped = frame.groupby("symbol", sort=False, group_keys=False)
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


def _rebalance_public_positions(
    current_positions: dict[str, float],
    requested_target: dict[str, float],
    tradable_symbols: set[str],
    nav_open: float,
    when: pd.Timestamp,
    config: PublicStrategyConfig,
    signal_liquidity: dict[str, float] | None = None,
    signal_volatility: dict[str, float] | None = None,
) -> _PublicRebalance:
    """Apply a weight-level rebalance without resizing suspended holdings.

    Existing positions without an execution-day quote are locked at their carried
    value. If fees make the requested book unaffordable, only executable targets are
    scaled. Turnover and fees are then recomputed from the actually executable book.
    """
    if not np.isfinite(nav_open) or nav_open <= 0:
        raise ValueError("公开回测调仓时净值必须为正数")
    if any(not np.isfinite(value) or value < 0 for value in current_positions.values()):
        raise ValueError("公开回测当前持仓市值必须是非负有限数")
    if any(not np.isfinite(weight) or weight < 0 for weight in requested_target.values()):
        raise ValueError("公开回测目标权重必须是非负有限数")

    locked = {
        symbol: float(value)
        for symbol, value in current_positions.items()
        if symbol not in tradable_symbols and value > 1e-12
    }
    requested_values = {
        symbol: nav_open * float(weight)
        for symbol, weight in requested_target.items()
        if symbol in tradable_symbols and weight > 0
    }
    buy_rate, sell_rate = _fee_rates(when, config)
    liquidity = signal_liquidity or {}
    volatility = signal_volatility or {}

    def evaluate(scale: float) -> _PublicRebalance:
        executable_targets = {
            symbol: value * scale
            for symbol, value in requested_values.items()
            if value * scale > 1e-12
        }
        positions = {**executable_targets, **locked}
        all_symbols = set(current_positions) | set(positions)
        changes = {
            symbol: positions.get(symbol, 0.0)
            - current_positions.get(symbol, 0.0)
            for symbol in all_symbols
        }
        buys = sum(max(0.0, change) for change in changes.values())
        sells = sum(max(0.0, -change) for change in changes.values())
        market_impact_cost = 0.0
        for symbol, change in changes.items():
            traded_value = abs(float(change))
            if traded_value <= 1e-15:
                continue
            average_amount = float(liquidity.get(symbol, np.nan))
            participation = (
                traded_value * config.initial_capital / average_amount
                if np.isfinite(average_amount) and average_amount > 0
                else 0.0
            )
            impact = market_impact_bps(
                model=config.market_impact_model,
                annualized_volatility=float(volatility.get(symbol, np.nan)),
                participation_rate=participation,
                coefficient=config.market_impact_coefficient,
                annualized_volatility_floor=(
                    config.market_impact_volatility_floor
                ),
                maximum_bps=config.max_market_impact_bps,
            )
            market_impact_cost += traded_value * impact / 10_000.0
        cost = buys * buy_rate + sells * sell_rate + market_impact_cost
        cash = nav_open - sum(positions.values()) - cost
        return _PublicRebalance(
            positions=positions,
            cash=float(cash),
            buys=float(buys),
            sells=float(sells),
            cost=float(cost),
            market_impact_cost=float(market_impact_cost),
            locked_value=float(sum(locked.values())),
            executable_scale=float(scale),
        )

    outcome = evaluate(1.0)
    if outcome.cash >= -1e-12:
        return _PublicRebalance(
            **{**asdict(outcome), "cash": max(0.0, outcome.cash)}
        )

    # Cash is monotonic in this scale because fees are far below the traded value.
    # Bisection avoids scaling locked holdings and also keeps fee calculations exact.
    low = 0.0
    high = 1.0
    feasible = evaluate(low)
    if feasible.cash < -1e-9:
        raise RuntimeError("停牌锁定持仓后仍无法形成非负现金组合")
    for _ in range(80):
        midpoint = (low + high) / 2.0
        candidate = evaluate(midpoint)
        if candidate.cash >= 0:
            low = midpoint
            feasible = candidate
        else:
            high = midpoint
    return _PublicRebalance(
        **{**asdict(feasible), "cash": max(0.0, feasible.cash)}
    )


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
    trailing_returns: pd.DataFrame | None = None,
    current_weights: dict[str, float] | None = None,
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
    required = [
        "mom_12_1",
        "mom_6_1",
        "fip_momentum",
        "trend",
        "volatility",
        "downside_volatility",
        "drawdown_quality",
        "avg_amount_20",
    ]
    candidates = candidates.dropna(subset=required)
    if config.stock_trend_filter:
        candidates = candidates.loc[candidates["trend"] > 0]
    if len(candidates) < config.top_n:
        return {}, pd.DataFrame(), "INSUFFICIENT_CANDIDATES"

    candidates["low_volatility"] = -candidates["volatility"]
    candidates["low_downside_vol"] = -candidates["downside_volatility"]
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
    selected = selected.set_index("symbol", drop=False)
    inverse_vol = 1.0 / selected["volatility"].clip(lower=1e-8)
    allocation = allocate_portfolio(
        model=config.portfolio_model,
        inverse_risk=inverse_vol,
        total=exposure,
        stock_cap=config.max_stock_weight,
        groups=pd.Series("ALL", index=selected.index, dtype="object"),
        group_cap=1.0,
        returns=trailing_returns,
        current_weights=current_weights,
        covariance_lookback_days=config.covariance_lookback_days,
        minimum_covariance_observations=(
            config.minimum_covariance_observations
        ),
        covariance_shrinkage=config.covariance_shrinkage,
        minimum_variance_blend=config.minimum_variance_blend,
        turnover_smoothing=config.turnover_smoothing,
        covariance_ridge=config.covariance_ridge,
    )
    target = allocation.weights
    selected["raw_target_weight"] = allocation.raw_weights
    selected["target_weight"] = selected["symbol"].map(target).fillna(0.0)
    selected["current_weight"] = selected["symbol"].map(
        current_weights or {}
    ).fillna(0.0)
    selected["target_weight_change"] = (
        selected["target_weight"] - selected["current_weight"]
    )
    selected["portfolio_model"] = config.portfolio_model
    selected["portfolio_status"] = allocation.status
    selected["covariance_observations"] = allocation.covariance_observations
    selected.insert(0, "signal_date", when)
    selected["regime"] = regime
    selected["target_exposure"] = exposure
    return target.to_dict(), selected.reset_index(drop=True), regime


def _audit_public_data_quality(
    *,
    membership: pd.DataFrame,
    relevant_symbols: Iterable[str],
    bars: pd.DataFrame,
    signal_dates: Iterable[pd.Timestamp],
    exact_by_date: dict[pd.Timestamp, pd.DataFrame],
    config: PublicStrategyConfig,
    cache_manifest: dict[str, object],
    missing_execution_events: int,
    missing_execution_symbols: Iterable[str],
    stale_marks: int,
) -> dict[str, object]:
    """Build an auditable coverage report; file hashes alone cannot prove completeness."""
    relevant = set(map(str, relevant_symbols))
    available = set(bars["symbol"].astype(str).unique())
    members_without_any_bars = sorted(relevant.difference(available))
    monthly_records: list[dict[str, object]] = []
    for signal_date in sorted(map(pd.Timestamp, signal_dates)):
        active = set(members_at(membership, signal_date)["symbol"].astype(str))
        exact = exact_by_date.get(signal_date, pd.DataFrame())
        quoted = set(exact.get("symbol", pd.Series(dtype=str)).astype(str))
        quoted_active = active.intersection(quoted)
        missing = sorted(active.difference(quoted))
        monthly_records.append(
            {
                "signal_date": str(signal_date.date()),
                "active_members": len(active),
                "quoted_active_members": len(quoted_active),
                "missing_member_count": len(missing),
                "coverage": len(quoted_active) / max(len(active), 1),
                "missing_symbols": missing,
            }
        )

    coverage_values = [float(record["coverage"]) for record in monthly_records]
    member_counts = [int(record["active_members"]) for record in monthly_records]
    below_threshold = [
        record
        for record in monthly_records
        if float(record["coverage"])
        < config.minimum_month_end_quote_coverage
    ]
    warnings: list[str] = []
    if members_without_any_bars:
        warnings.append(
            "历史成分中存在完全无行情证券: "
            + ", ".join(members_without_any_bars)
        )
    if below_threshold:
        warnings.append(
            f"{len(below_threshold)} 个信号月的成分行情覆盖低于 "
            f"{config.minimum_month_end_quote_coverage:.2%}"
        )

    return {
        "data_quality_status": "warning" if warnings else "pass",
        "data_quality_warnings": warnings,
        "membership_intervals": len(membership),
        "relevant_unique_members": len(relevant),
        "members_with_bars": len(available.intersection(relevant)),
        "members_without_any_bars": members_without_any_bars,
        "member_bar_coverage": len(available.intersection(relevant))
        / max(len(relevant), 1),
        "minimum_month_end_quote_coverage": (
            config.minimum_month_end_quote_coverage
        ),
        "month_end_quote_coverage_min": min(coverage_values)
        if coverage_values
        else 0.0,
        "month_end_quote_coverage_median": float(np.median(coverage_values))
        if coverage_values
        else 0.0,
        "month_end_members_min": min(member_counts) if member_counts else 0,
        "month_end_members_max": max(member_counts) if member_counts else 0,
        "month_end_quote_coverage": monthly_records,
        "month_end_quote_coverage_below_threshold": below_threshold,
        "bar_rows": len(bars),
        "first_bar_date": str(bars["date"].min().date()),
        "last_bar_date": str(bars["date"].max().date()),
        # Backward-compatible count plus explicit event/list fields.
        "missing_execution_symbols": missing_execution_events,
        "missing_execution_event_count": missing_execution_events,
        "missing_execution_symbol_list": sorted(
            set(map(str, missing_execution_symbols))
        ),
        "stale_position_marks": stale_marks,
        "adjustment": (
            f"{cache_manifest.get('source', 'public cache')} "
            "hfq total-return price"
        ),
        "regime_symbol": config.regime_symbol,
        "performance_benchmark_symbol": config.performance_benchmark_symbol,
        "known_limitations": [
            "历史成分来自第三方对中证官方公告的规范化整理，并非中证公司原始机器接口",
            "公开接口没有可靠的历史 ST、历史行业和历史市值快照，因此本测试关闭对应过滤/中性化",
            "权重级回测未模拟 100 股整手、最低 5 元佣金、涨跌停排队和容量部分成交；执行日停牌时保留旧持仓、新目标留现金且不重试",
            (
                f"平方根冲击按 {config.initial_capital:,.0f} 元初始资金、"
                "信号日 ADV20 与波动率估算；"
                "它是公开数据近似成本，不是历史逐笔成交回放"
                if config.market_impact_model == EXECUTION_MODEL_V1_6
                else "固定滑点成本不随订单参与率变化"
            ),
            "沪深300ETF后复权净值作为可投资基准，含基金费率和跟踪误差",
        ],
    }


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
    index_bars = _load_cached_bars(cache_dir, [config.regime_symbol]).sort_values("date")
    benchmark_bars = _load_cached_bars(
        cache_dir, [config.performance_benchmark_symbol]
    ).sort_values("date")
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
    return_history = (
        features.pivot(index="date", columns="symbol", values="return_1d")
        .sort_index()
    )
    benchmark = benchmark_bars.set_index("date").reindex(calendar)
    benchmark["close"] = benchmark["close"].ffill()
    benchmark["benchmark_nav"] = benchmark["close"] / benchmark["close"].iloc[0]

    positions: dict[str, float] = {}
    cash = 1.0
    pending: tuple[
        pd.Timestamp,
        dict[str, float],
        str,
        dict[str, float],
        dict[str, float],
    ] | None = None
    curve_rows: list[dict[str, object]] = []
    selection_frames: list[pd.DataFrame] = []
    rebalance_rows: list[dict[str, object]] = []
    missing_execution_events = 0
    missing_execution_symbol_set: set[str] = set()
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
            (
                signal_date,
                requested_target,
                regime,
                signal_liquidity,
                signal_volatility,
            ) = pending
            nav_open = cash + sum(positions.values())
            tradable_symbols = {
                str(symbol)
                for symbol in today.index
                if float(today.at[symbol, "open"]) > 0
                and float(today.at[symbol, "volume"]) > 0
                and float(today.at[symbol, "amount"]) > 0
            }
            missing_for_execution = set(requested_target) - tradable_symbols
            missing_execution_events += len(missing_for_execution)
            missing_execution_symbol_set.update(missing_for_execution)
            outcome = _rebalance_public_positions(
                positions,
                requested_target,
                tradable_symbols,
                nav_open,
                date,
                config,
                signal_liquidity=signal_liquidity,
                signal_volatility=signal_volatility,
            )
            positions = outcome.positions
            cash = outcome.cash
            rebalance_rows.append(
                {
                    "signal_date": signal_date,
                    "execution_date": date,
                    "regime": regime,
                    "buy_turnover": outcome.buys / nav_open if nav_open else 0.0,
                    "sell_turnover": outcome.sells / nav_open if nav_open else 0.0,
                    "one_way_turnover": (
                        (outcome.buys + outcome.sells) / (2.0 * nav_open)
                        if nav_open
                        else 0.0
                    ),
                    "cost_rate": outcome.cost / nav_open if nav_open else 0.0,
                    "market_impact_cost_rate": (
                        outcome.market_impact_cost / nav_open
                        if nav_open
                        else 0.0
                    ),
                    "holding_count": len(positions),
                    "requested_target_exposure": float(sum(requested_target.values())),
                    "actual_target_exposure": (
                        sum(positions.values()) / nav_open if nav_open else 0.0
                    ),
                    "target_exposure": (
                        sum(positions.values()) / nav_open if nav_open else 0.0
                    ),
                    "locked_exposure": outcome.locked_value / nav_open if nav_open else 0.0,
                    "executable_scale": outcome.executable_scale,
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
                current_weights = {
                    symbol: value / max(nav_close, 1e-12)
                    for symbol, value in positions.items()
                    if value > 1e-12
                }
                target, selection, regime = _select_targets(
                    exact,
                    membership,
                    date,
                    {symbol for symbol, value in positions.items() if value / max(nav_close, 1e-12) > 1e-5},
                    config,
                    index_by_date.loc[date],
                    trailing_returns=return_history.loc[
                        return_history.index <= date
                    ].tail(config.covariance_lookback_days),
                    current_weights=current_weights,
                )
                if target:
                    pending = (
                        date,
                        target,
                        regime,
                        dict(
                            zip(
                                exact["symbol"],
                                exact["avg_amount_20"],
                                strict=False,
                            )
                        ),
                        dict(
                            zip(
                                exact["symbol"],
                                exact["volatility"],
                                strict=False,
                            )
                        ),
                    )
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
    manifest_path = Path(cache_dir) / "manifest.json"
    cache_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    quality = _audit_public_data_quality(
        membership=membership,
        relevant_symbols=symbols,
        bars=bars,
        signal_dates=signal_dates,
        exact_by_date=exact_by_date,
        config=config,
        cache_manifest=cache_manifest,
        missing_execution_events=missing_execution_events,
        missing_execution_symbols=missing_execution_symbol_set,
        stale_marks=stale_marks,
    )
    for warning in quality["data_quality_warnings"]:
        LOGGER.warning("公开数据质量: %s", warning)
    return PublicBacktestResult(config, curve, selections, rebalances, quality)


def public_strategy_variants(base: PublicStrategyConfig | None = None) -> list[PublicStrategyConfig]:
    base = base or PublicStrategyConfig()
    legacy_base = replace(
        base,
        weight_fip_momentum=0.0,
        weight_low_downside_volatility=0.0,
        weight_drawdown_quality=0.0,
    )
    return [
        legacy_base,
        replace(
            legacy_base,
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
            legacy_base,
            name="balanced_momentum_20",
            risk_off_exposure=0.0,
            weight_mom_12_1=0.45,
            weight_mom_6_1=0.25,
            weight_trend=0.15,
            weight_low_volatility=0.15,
            weight_liquidity=0.0,
        ),
        replace(
            legacy_base,
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
            legacy_base,
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
            legacy_base,
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
        replace(
            legacy_base,
            name=ALPHA_MODEL_VERSION,
            top_n=15,
            exit_rank=25,
            max_stock_weight=0.09,
            risk_off_exposure=0.0,
            weight_mom_12_1=QUALITY_MOMENTUM_V1_5_WEIGHTS["mom_12_1"],
            weight_mom_6_1=QUALITY_MOMENTUM_V1_5_WEIGHTS["mom_6_1"],
            weight_fip_momentum=QUALITY_MOMENTUM_V1_5_WEIGHTS["fip_momentum"],
            weight_trend=QUALITY_MOMENTUM_V1_5_WEIGHTS["trend"],
            weight_low_volatility=QUALITY_MOMENTUM_V1_5_WEIGHTS["low_vol"],
            weight_low_downside_volatility=QUALITY_MOMENTUM_V1_5_WEIGHTS[
                "low_downside_vol"
            ],
            weight_drawdown_quality=QUALITY_MOMENTUM_V1_5_WEIGHTS[
                "drawdown_quality"
            ],
            weight_liquidity=QUALITY_MOMENTUM_V1_5_WEIGHTS["liquidity"],
        ),
    ]


def public_implementation_variants(
    base: PublicStrategyConfig | None = None,
) -> list[PublicStrategyConfig]:
    """Freeze Alpha while independently toggling v1.6 implementation layers."""
    base = base or PublicStrategyConfig()
    return [
        replace(
            base,
            name="baseline_v1_5_1_public",
            portfolio_model=DEFAULT_PORTFOLIO_MODEL,
            market_impact_model=DEFAULT_EXECUTION_MODEL,
        ),
        replace(
            base,
            name="portfolio_only_v1_6_public",
            portfolio_model=PORTFOLIO_MODEL_V1_6,
            market_impact_model=DEFAULT_EXECUTION_MODEL,
        ),
        replace(
            base,
            name="execution_only_v1_6_public",
            portfolio_model=DEFAULT_PORTFOLIO_MODEL,
            market_impact_model=EXECUTION_MODEL_V1_6,
        ),
        replace(
            base,
            name="combined_v1_6_public",
            portfolio_model=PORTFOLIO_MODEL_V1_6,
            market_impact_model=EXECUTION_MODEL_V1_6,
        ),
    ]


def _public_strategy_governance(name: str) -> dict[str, object]:
    if name == ALPHA_MODEL_VERSION:
        return alpha_profile_governance(ALPHA_MODEL_VERSION)
    return {
        "lifecycle_status": "historical_research_baseline",
        "promotion_decision": "not_applicable",
        "default_eligible": False,
        "reason": "公开通道研究方案，不等同于严格通道生产默认配置",
    }


def _slice_metrics(curve: pd.DataFrame, start: str, end: str, column: str) -> dict[str, float | int]:
    subset = curve.loc[curve["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    return _calculate_metrics(subset, column)


def _annualized_period_turnover(
    rebalances: pd.DataFrame,
    start: str,
    end: str,
) -> float:
    years = max(
        (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.2425,
        1.0,
    )
    required = {"execution_date", "one_way_turnover"}
    if rebalances.empty or not required.issubset(rebalances.columns):
        return 0.0
    within_period = pd.to_datetime(rebalances["execution_date"]).between(
        start, end
    )
    return float(rebalances.loc[within_period, "one_way_turnover"].sum() / years)


def _annualized_period_cost(
    rebalances: pd.DataFrame,
    start: str,
    end: str,
    column: str,
) -> float:
    years = max(
        (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.2425,
        1.0,
    )
    if rebalances.empty or column not in rebalances.columns:
        return 0.0
    within = pd.to_datetime(rebalances["execution_date"]).between(start, end)
    return float(rebalances.loc[within, column].sum() / years)


def write_public_implementation_research(
    membership_path: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    base_config: PublicStrategyConfig | None = None,
) -> dict[str, Path]:
    """Run a weight-level public-data approximation of the v1.6 four arms."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_config = base_config or PublicStrategyConfig()
    cache_manifest = verify_public_cache(
        membership_path,
        cache_dir,
        seal_legacy=True,
    )
    periods = {
        "development_2013_2017": ("2013-01-01", "2017-12-31"),
        "validation_2018_2021": ("2018-01-01", "2021-12-31"),
        "historical_holdout_seen_2022_2025": ("2022-01-01", "2025-12-31"),
        "full_requested_period": (
            base_config.start_date,
            base_config.end_date,
        ),
    }
    rows: list[dict[str, object]] = []
    variant_artifacts: list[Path] = []
    results: dict[str, PublicBacktestResult] = {}
    variants = public_implementation_variants(base_config)
    for variant in variants:
        LOGGER.info("运行公开数据 v1.6 实现归因: %s", variant.name)
        result = run_public_backtest(
            membership_path,
            cache_dir,
            variant,
        )
        results[variant.name] = result
        variant_dir = output / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "equity_curve.csv": result.equity_curve,
            "selections.csv": result.selections,
            "rebalances.csv": result.rebalances,
        }
        for filename, frame in files.items():
            path = variant_dir / filename
            frame.to_csv(path, index=False)
            variant_artifacts.append(path)
        config_path = write_json_atomic(
            asdict(variant),
            variant_dir / "config.json",
        )
        quality_path = write_json_atomic(
            result.data_quality,
            variant_dir / "data_quality.json",
        )
        variant_artifacts.extend([config_path, quality_path])

        portfolio_governance = portfolio_model_governance(
            variant.portfolio_model
        )
        execution_governance = execution_model_governance(
            variant.market_impact_model
        )
        for period, (start, end) in periods.items():
            metrics = _slice_metrics(result.equity_curve, start, end, "nav")
            rows.append(
                {
                    "variant": variant.name,
                    "portfolio_model": variant.portfolio_model,
                    "portfolio_status": portfolio_governance[
                        "lifecycle_status"
                    ],
                    "execution_model": variant.market_impact_model,
                    "execution_status": execution_governance[
                        "lifecycle_status"
                    ],
                    "period": period,
                    **metrics,
                    "annual_one_way_turnover": _annualized_period_turnover(
                        result.rebalances,
                        start,
                        end,
                    ),
                    "annual_market_impact_cost_rate": _annualized_period_cost(
                        result.rebalances,
                        start,
                        end,
                        "market_impact_cost_rate",
                    ),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics_path = output / "implementation_comparison.csv"
    metrics.to_csv(metrics_path, index=False)

    report_lines = [
        "# v1.6 公开数据组合与成交四臂对照",
        "",
        "本报告固定同一 Alpha，只切换组合和成交模型。公开通道是权重级近似："
        "不模拟整手、最低佣金、涨跌停排队、容量部分成交和失败重试。",
        "平方根冲击按配置中的初始资金、信号日 ADV20 与波动率估算，不能替代严格成交引擎。",
        "",
        "| 方案 | 区间 | 组合模型 | 成交模型 | 年化 | 最大回撤 | 夏普 | 年化单边换手 | 年化冲击成本率 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        report_lines.append(
            f"| {row.variant} | {row.period} | {row.portfolio_model} | "
            f"{row.execution_model} | {row.cagr:.2%} | "
            f"{row.max_drawdown:.2%} | {row.sharpe:.2f} | "
            f"{row.annual_one_way_turnover:.2%} | "
            f"{row.annual_market_impact_cost_rate:.2%} |"
        )
    report_lines.extend(
        [
            "",
            "2013–2025 均为已查看历史区间。任何差异都只是绑定当前数据指纹的历史验证，"
            "不能称为未触碰样本外，也不能证明未来收益。",
        ]
    )
    report_path = output / "PUBLIC_V1.6_IMPLEMENTATION_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "protocol": "fixed_alpha_four_arm_public_weight_level_approximation",
        "automatic_parameter_fitting": False,
        "untouched_holdout_certified": False,
        "periods": periods,
        "variants": [asdict(variant) for variant in variants],
        "portfolio_governance": {
            model: portfolio_model_governance(model)
            for model in [DEFAULT_PORTFOLIO_MODEL, PORTFOLIO_MODEL_V1_6]
        },
        "execution_governance": {
            model: execution_model_governance(model)
            for model in [DEFAULT_EXECUTION_MODEL, EXECUTION_MODEL_V1_6]
        },
        "data_fingerprint_sha256": cache_manifest.get(
            "data_fingerprint_sha256"
        ),
        "limitations": next(iter(results.values())).data_quality[
            "known_limitations"
        ],
    }
    manifest_path = write_json_atomic(
        manifest,
        output / "research_manifest.json",
    )
    reproducibility = build_reproducibility_manifest(
        {
            "base_config": asdict(base_config),
            "variants": [asdict(variant) for variant in variants],
            "periods": periods,
            "protocol": manifest["protocol"],
        },
        data_manifest_path=Path(cache_dir) / "manifest.json",
        extra_input_files=[membership_path],
    )
    reproducibility_path = write_json_atomic(
        reproducibility,
        output / "reproducibility.json",
    )
    artifacts = [
        metrics_path,
        report_path,
        manifest_path,
        reproducibility_path,
        *variant_artifacts,
    ]
    registry_path = record_experiment(
        output / "experiment_registry.jsonl",
        reproducibility,
        experiment_type="public_v1_6_implementation_comparison",
        protocol={
            "evaluation": manifest["protocol"],
            "automatic_parameter_fitting": False,
            "untouched_holdout_certified": False,
            "strict_execution_equivalent": False,
        },
        artifacts=artifacts,
    )
    artifact_manifest_path = write_artifact_manifest(
        output,
        [*artifacts, registry_path],
    )
    return {
        "metrics": metrics_path,
        "report": report_path,
        "manifest": manifest_path,
        "reproducibility": reproducibility_path,
        "registry": registry_path,
        "artifacts": artifact_manifest_path,
    }


def write_public_research(
    membership_path: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    base_config: PublicStrategyConfig | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = base_config or PublicStrategyConfig()
    cache_manifest = verify_public_cache(
        membership_path, cache_dir, seal_legacy=True
    )
    metric_rows: list[dict[str, object]] = []
    results: dict[str, PublicBacktestResult] = {}
    variant_artifacts: list[Path] = []
    periods = {
        "development_2013_2017": ("2013-01-01", "2017-12-31"),
        "validation_2018_2021": ("2018-01-01", "2021-12-31"),
        "historical_holdout_seen_2022_2025": ("2022-01-01", "2025-12-31"),
        "full_requested_period": (base_config.start_date, base_config.end_date),
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
        variant_artifacts.extend(
            variant_dir / name
            for name in [
                "equity_curve.csv",
                "selections.csv",
                "rebalances.csv",
                "config.json",
                "data_quality.json",
            ]
        )
        for period, (start, end) in periods.items():
            governance = _public_strategy_governance(variant.name)
            strategy_metrics = _slice_metrics(result.equity_curve, start, end, "nav")
            benchmark_metrics = _slice_metrics(result.equity_curve, start, end, "benchmark_nav")
            metric_rows.append(
                {
                    "strategy": variant.name,
                    "strategy_status": governance["lifecycle_status"],
                    "promotion_decision": governance["promotion_decision"],
                    "period": period,
                    **strategy_metrics,
                    "benchmark_cagr": benchmark_metrics["cagr"],
                    "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
                    "annual_one_way_turnover": _annualized_period_turnover(
                        result.rebalances, start, end
                    ),
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
    axis.set_title(
        f"Public-data strategy comparison ({base_config.start_date}–{base_config.end_date})"
    )
    axis.set_ylabel("NAV")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    chart_path = output_dir / "strategy_comparison.png"
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)

    def percentage(value: object) -> str:
        return "—" if pd.isna(value) else f"{float(value):.2%}"

    def percentage_points(value: object) -> str:
        return "—" if pd.isna(value) else f"{float(value) * 100:+.2f} pp"

    report_lines = [
        "# A股公开数据分段回测报告",
        "",
        "本报告使用历史时点沪深300成分区间、公开日线后复权价格、月末信号和下一交易日开盘成交。所有结果均已计入双边佣金、历史卖出印花税、过户费和单边滑点。",
        "",
        f"严格通道生产默认 Alpha 为 `{DEFAULT_ALPHA_PROFILE}`。"
        f"`{ALPHA_MODEL_VERSION}` 的生命周期状态为 `experimental`，"
        "本轮默认晋级决定为 `rejected`；公开回放继续保留它仅用于审计和前瞻观察。",
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
    metric_index = metrics.set_index(["strategy", "period"])
    report_lines.extend(
        [
            "",
            "## v1.5 Alpha 同约束研究对照",
            "",
            "`quality_momentum_v1_5` 与 `momentum_focus_15` 使用相同持股数、排名缓冲、"
            "单股上限和风险仓位；下表只比较选股 Alpha 变化。"
            "`momentum_focus_15` 是公开研究基准，不是严格通道的 `legacy_v1_4` 默认策略。",
            "",
            "| 区间 | momentum_focus_15 年化 | quality_momentum_v1_5 年化 | 年化改善 | 最大回撤改善（正值更好） | 夏普差 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for period in periods:
        legacy = metric_index.loc[("momentum_focus_15", period)]
        enhanced = metric_index.loc[(ALPHA_MODEL_VERSION, period)]
        report_lines.append(
            f"| {period} | {percentage(legacy['cagr'])} | "
            f"{percentage(enhanced['cagr'])} | "
            f"{percentage_points(enhanced['cagr'] - legacy['cagr'])} | "
            f"{percentage_points(enhanced['max_drawdown'] - legacy['max_drawdown'])} | "
            f"{enhanced['sharpe'] - legacy['sharpe']:.2f} |"
        )
    holdout = metrics.loc[
        metrics["period"].eq("historical_holdout_seen_2022_2025")
    ].sort_values("cagr", ascending=False)
    best_holdout = holdout.iloc[0]
    report_lines.extend(
        [
            "",
            "## 结果判读",
            "",
            f"- 已查看历史保留期年化最高的是 `{best_holdout['strategy']}`：{percentage(best_holdout['cagr'])}，最大回撤 {percentage(best_holdout['max_drawdown'])}。这只是预定义方案之间的比较，不是收益承诺。",
            f"- 该历史保留期是否达到 15%：{'是' if float(best_holdout['cagr']) >= 0.15 else '否'}。2022–2025 的结果已经被查看，今后不能再把它当作未触碰样本外反复调参。",
            "- 开发期、验证期与已查看保留期应一起看；下一段真正的前瞻验证应使用 2026 年以后未参与调参的数据或模拟盘。",
            f"- `{ALPHA_MODEL_VERSION}` 未通过本轮默认晋级；不要把本表解释为 v1.4 默认策略已被替换。",
            "",
            "## 公开数据口径限制",
            "",
        ]
    )
    for limitation in first_result.data_quality["known_limitations"]:
        report_lines.append(f"- {limitation}")
    missing_members = first_result.data_quality["members_without_any_bars"]
    missing_text = ", ".join(missing_members) if missing_members else "无"
    below_threshold = first_result.data_quality[
        "month_end_quote_coverage_below_threshold"
    ]
    below_dates = ", ".join(
        str(record["signal_date"]) for record in below_threshold
    ) or "无"
    report_lines.extend(
        [
            f"- 数据质量状态：`{first_result.data_quality['data_quality_status']}`；"
            f"完全无行情的历史成员：{missing_text}。",
            "- 月末行情覆盖告警阈值为 "
            f"{first_result.data_quality['minimum_month_end_quote_coverage']:.2%}；"
            f"低于阈值的信号月：{below_dates}。逐月分母、缺失数和证券列表见各策略的 `data_quality.json`。",
        ]
    )
    report_lines.append(
        "- 月末成分当日行情覆盖最低为 "
        f"{first_result.data_quality['month_end_quote_coverage_min']:.2%}、中位数为 "
        f"{first_result.data_quality['month_end_quote_coverage_median']:.2%}；"
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
            "六组旧参数在 v1.3 运行前预定义；quality_momentum_v1_5 的公式与权重"
            "在首次生成 v1.5 结果前冻结。但 2013-2025 市场结果已被旧研究查看，"
            "所有区间都只属于历史回放，不得宣称为未触碰样本外。"
        ),
        "alpha_protocol": {
            "model": ALPHA_MODEL_VERSION,
            "weights": dict(QUALITY_MOMENTUM_V1_5_WEIGHTS),
            "governance": alpha_profile_governance(ALPHA_MODEL_VERSION),
            "strict_default_alpha_profile": DEFAULT_ALPHA_PROFILE,
            "parameter_search": "none",
            "historical_market_periods_seen_before_design": True,
        },
        "period_status": {
            "development_2013_2017": "seen_development",
            "validation_2018_2021": "seen_validation",
            "historical_holdout_seen_2022_2025": "seen_historical_holdout",
            "full_requested_period": "seen_aggregate",
            "prospective_2026_onward": "researcher_managed_not_automatically_certified",
        },
        "variants": [asdict(value) for value in public_strategy_variants(base_config)],
        "variant_governance": {
            value.name: _public_strategy_governance(value.name)
            for value in public_strategy_variants(base_config)
        },
        "data_quality": {
            key: first_result.data_quality[key]
            for key in [
                "data_quality_status",
                "data_quality_warnings",
                "member_bar_coverage",
                "members_without_any_bars",
                "minimum_month_end_quote_coverage",
                "month_end_quote_coverage_min",
                "month_end_quote_coverage_below_threshold",
            ]
        },
        "limitations": next(iter(results.values())).data_quality["known_limitations"],
        "data_fingerprint_sha256": cache_manifest.get(
            "data_fingerprint_sha256"
        ),
        "cache_verification": cache_manifest.get("verification", {"verified": True}),
    }
    reproducibility = build_reproducibility_manifest(
        {
            "base_config": asdict(base_config),
            "variants": [asdict(value) for value in public_strategy_variants(base_config)],
            "periods": periods,
        },
        data_manifest_path=Path(cache_dir) / "manifest.json",
        extra_input_files=[membership_path],
    )
    reproducibility_path = write_json_atomic(
        reproducibility, output_dir / "reproducibility.json"
    )
    manifest["reproducibility_file"] = reproducibility_path.name
    manifest["run_fingerprint_sha256"] = reproducibility[
        "run_fingerprint_sha256"
    ]
    manifest_path = output_dir / "research_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    registry_path = record_experiment(
        output_dir / "experiment_registry.jsonl",
        reproducibility,
        experiment_type="public_research",
        protocol={
            "parameters": "six_legacy_v1_3_plus_frozen_quality_momentum_v1_5",
            "historical_holdout_2022_2025": "seen",
            "prospective_holdout_start": "2026-01-01",
            "candidate_promotion_decision": "rejected",
            "untouched_holdout_certified": False,
        },
        artifacts=[
            report_path,
            metrics_path,
            chart_path,
            manifest_path,
            reproducibility_path,
            *variant_artifacts,
        ],
    )
    artifact_manifest_path = write_artifact_manifest(
        output_dir,
        [
            report_path,
            metrics_path,
            chart_path,
            manifest_path,
            reproducibility_path,
            *variant_artifacts,
            registry_path,
        ],
    )
    return {
        "report": report_path,
        "metrics": metrics_path,
        "chart": chart_path,
        "manifest": manifest_path,
        "reproducibility": reproducibility_path,
        "registry": registry_path,
        "artifacts": artifact_manifest_path,
        "output_dir": output_dir,
    }


def write_public_robustness(
    membership_path: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    base_config: PublicStrategyConfig | None = None,
) -> dict[str, Path]:
    """Cost stress and factor ablation for the frozen v1.5 alpha candidate."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = base_config or PublicStrategyConfig()
    cache_manifest = verify_public_cache(
        membership_path, cache_dir, seal_legacy=True
    )
    candidate = next(
        variant
        for variant in public_strategy_variants(base_config)
        if variant.name == ALPHA_MODEL_VERSION
    )
    periods = {
        "historical_holdout_seen_2022_2025": ("2022-01-01", "2025-12-31"),
        "full_requested_period": (candidate.start_date, candidate.end_date),
    }

    cost_rows: list[dict[str, object]] = []
    for slippage in [5.0, 10.0, 20.0]:
        stressed = replace(
            candidate,
            name=f"{ALPHA_MODEL_VERSION}_slippage_{slippage:g}",
            slippage_bps=slippage,
        )
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
        "fip_momentum": "weight_fip_momentum",
        "trend": "weight_trend",
        "low_downside_vol": "weight_low_downside_volatility",
        "drawdown_quality": "weight_drawdown_quality",
    }
    ablation_configs = [("full", candidate)] + [
        (
            f"without_{factor}",
            replace(
                candidate,
                name=f"{ALPHA_MODEL_VERSION}_without_{factor}",
                **{field: 0.0},
            ),
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
        f"`{ALPHA_MODEL_VERSION}` 的公式和权重在生成 v1.5 新结果前固定。"
        "但 2022–2025 市场结果已在旧策略研究中被查看，因此本页只属于历史回放，"
        "不能宣称为未触碰样本外。",
        "",
        f"治理状态：`experimental`；默认晋级决定：`rejected`。"
        f"严格通道继续以 `{DEFAULT_ALPHA_PROFILE}` 为生产默认 Alpha。",
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
            "删除一个因子后，剩余正权重会在综合分数中重新归一化，且因子彼此相关；"
            "因此消融只表示当前组合下的边际证据，不能解释为单因子的独立因果贡献。",
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
    reproducibility = build_reproducibility_manifest(
        {
            "candidate": asdict(candidate),
            "candidate_governance": alpha_profile_governance(ALPHA_MODEL_VERSION),
            "strict_default_alpha_profile": DEFAULT_ALPHA_PROFILE,
            "periods": periods,
            "cost_slippage_bps": [5.0, 10.0, 20.0],
            "factor_ablation": list(factor_fields),
            "data_fingerprint_sha256": cache_manifest.get(
                "data_fingerprint_sha256"
            ),
        },
        data_manifest_path=Path(cache_dir) / "manifest.json",
        extra_input_files=[membership_path],
    )
    reproducibility_path = write_json_atomic(
        reproducibility, output_dir / "reproducibility.json"
    )
    registry_path = record_experiment(
        output_dir / "experiment_registry.jsonl",
        reproducibility,
        experiment_type="public_robustness",
        protocol={
            "candidate": ALPHA_MODEL_VERSION,
            "candidate_promotion_decision": "rejected",
            "historical_holdout_2022_2025": "seen",
            "optimization": "frozen_v1_5_weights_cost_stress_and_ablation_only",
            "untouched_holdout_certified": False,
        },
        artifacts=[report_path, cost_path, ablation_path, reproducibility_path],
    )
    artifact_manifest_path = write_artifact_manifest(
        output_dir,
        [
            report_path,
            cost_path,
            ablation_path,
            reproducibility_path,
            registry_path,
        ],
    )
    return {
        "report": report_path,
        "cost": cost_path,
        "ablation": ablation_path,
        "reproducibility": reproducibility_path,
        "registry": registry_path,
        "artifacts": artifact_manifest_path,
    }
