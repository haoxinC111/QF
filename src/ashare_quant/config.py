from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


def _date_text(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value)


@dataclass(frozen=True)
class DataConfig:
    provider: str = "tushare"
    cache_dir: str = "data/cache"
    token_env: str = "TUSHARE_TOKEN"
    universe_index: str = "399300.SZ"
    benchmark_index: str = "399300.SZ"
    calls_per_minute: int = 180
    retries: int = 5
    refresh: bool = False
    warmup_calendar_days: int = 500


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    initial_cash: float = 1_000_000.0
    annual_risk_free_rate: float = 0.0
    output_dir: str = "results/latest"


@dataclass(frozen=True)
class StrategyConfig:
    top_n: int = 20
    min_history_days: int = 252
    min_avg_amount_million: float = 100.0
    min_price: float = 3.0
    winsor_quantile: float = 0.05
    stock_trend_filter: bool = True
    momentum_12_1_weight: float = 0.35
    momentum_6_1_weight: float = 0.20
    trend_weight: float = 0.15
    low_volatility_weight: float = 0.20
    liquidity_weight: float = 0.10
    volatility_lookback: int = 60
    benchmark_ma_days: int = 200
    risk_on_exposure: float = 0.95
    risk_off_exposure: float = 0.30
    max_stock_weight: float = 0.08
    risk_weight_power: float = 1.0

    @property
    def factor_weights(self) -> dict[str, float]:
        return {
            "mom_12_1": self.momentum_12_1_weight,
            "mom_6_1": self.momentum_6_1_weight,
            "trend": self.trend_weight,
            "low_vol": self.low_volatility_weight,
            "liquidity": self.liquidity_weight,
        }


@dataclass(frozen=True)
class ExecutionConfig:
    lot_size: int = 100
    commission_rate: float = 0.00025
    minimum_commission: float = 5.0
    stamp_duty_sell: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_bps: float = 5.0
    cash_buffer: float = 0.02
    max_participation_of_20d_amount: float = 0.05
    rebalance_retry_days: int = 3


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("配置文件顶层必须是映射结构")
        config = cls(
            data=DataConfig(**dict(raw.get("data", {}))),
            backtest=BacktestConfig(
                **{
                    **dict(raw.get("backtest", {})),
                    **{
                        key: _date_text(value)
                        for key, value in dict(raw.get("backtest", {})).items()
                        if key in {"start_date", "end_date"}
                    },
                }
            ),
            strategy=StrategyConfig(**dict(raw.get("strategy", {}))),
            execution=ExecutionConfig(**dict(raw.get("execution", {}))),
        )
        config.validate()
        return config

    def validate(self) -> None:
        start = datetime.strptime(self.backtest.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.backtest.end_date, "%Y-%m-%d")
        if start >= end:
            raise ValueError("start_date 必须早于 end_date")
        if self.backtest.initial_cash <= 0:
            raise ValueError("initial_cash 必须大于 0")
        if self.strategy.top_n < 2:
            raise ValueError("top_n 至少为 2")
        factor_sum = sum(self.strategy.factor_weights.values())
        if factor_sum <= 0:
            raise ValueError("因子权重之和必须大于 0")
        if not 0 <= self.strategy.winsor_quantile < 0.5:
            raise ValueError("winsor_quantile 必须在 [0, 0.5) 内")
        for name, exposure in {
            "risk_on_exposure": self.strategy.risk_on_exposure,
            "risk_off_exposure": self.strategy.risk_off_exposure,
        }.items():
            if not 0 <= exposure <= 1:
                raise ValueError(f"{name} 必须在 [0, 1] 内")
        if self.strategy.risk_off_exposure > self.strategy.risk_on_exposure:
            raise ValueError("risk_off_exposure 不应高于 risk_on_exposure")
        if self.strategy.top_n * self.strategy.max_stock_weight + 1e-12 < self.strategy.risk_on_exposure:
            raise ValueError("top_n × max_stock_weight 小于 risk_on_exposure，无法完成组合分配")
        if self.execution.lot_size <= 0:
            raise ValueError("lot_size 必须大于 0")
        if not 0 <= self.execution.cash_buffer < 1:
            raise ValueError("cash_buffer 必须在 [0, 1) 内")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_paths(self, base_dir: str | Path) -> "AppConfig":
        """Return a copy with relative cache/output paths anchored at base_dir."""
        base = Path(base_dir).resolve()
        data_values = asdict(self.data)
        backtest_values = asdict(self.backtest)
        if not Path(self.data.cache_dir).is_absolute():
            data_values["cache_dir"] = str(base / self.data.cache_dir)
        if not Path(self.backtest.output_dir).is_absolute():
            backtest_values["output_dir"] = str(base / self.backtest.output_dir)
        return AppConfig(
            data=DataConfig(**data_values),
            backtest=BacktestConfig(**backtest_values),
            strategy=self.strategy,
            execution=self.execution,
        )
