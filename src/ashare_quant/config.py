from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import yaml

from .alpha import QUALITY_MOMENTUM_V1_5_WEIGHTS


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
    regime_index: str = "000300.SH"
    benchmark_index: str = "H00300.CSI"
    benchmark_is_total_return: bool = True
    calls_per_minute: int = 180
    retries: int = 5
    refresh: bool = False
    warmup_calendar_days: int = 500
    strict_validation: bool = True
    industry_standard: str = "SW2021"
    industry_level: str = "L1"


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    initial_cash: float = 1_000_000.0
    annual_risk_free_rate: float = 0.0
    annual_cash_rate: float = 0.0
    maximum_stale_trading_days: int = 20
    stale_price_policy: str = "warn"
    delist_value_policy: str = "zero"
    output_dir: str = "results/latest"


@dataclass(frozen=True)
class StrategyConfig:
    top_n: int = 20
    min_history_days: int = 252
    min_avg_amount_million: float = 100.0
    min_price: float = 3.0
    winsor_quantile: float = 0.05
    stock_trend_filter: bool = True
    momentum_12_1_weight: float = QUALITY_MOMENTUM_V1_5_WEIGHTS["mom_12_1"]
    momentum_6_1_weight: float = QUALITY_MOMENTUM_V1_5_WEIGHTS["mom_6_1"]
    fip_momentum_weight: float = QUALITY_MOMENTUM_V1_5_WEIGHTS["fip_momentum"]
    trend_weight: float = QUALITY_MOMENTUM_V1_5_WEIGHTS["trend"]
    low_volatility_weight: float = QUALITY_MOMENTUM_V1_5_WEIGHTS["low_vol"]
    low_downside_volatility_weight: float = QUALITY_MOMENTUM_V1_5_WEIGHTS[
        "low_downside_vol"
    ]
    drawdown_quality_weight: float = QUALITY_MOMENTUM_V1_5_WEIGHTS[
        "drawdown_quality"
    ]
    liquidity_weight: float = QUALITY_MOMENTUM_V1_5_WEIGHTS["liquidity"]
    volatility_lookback: int = 60
    benchmark_ma_days: int = 200
    risk_on_exposure: float = 0.95
    risk_off_exposure: float = 0.30
    max_stock_weight: float = 0.08
    risk_weight_power: float = 1.0
    selection_buffer_enabled: bool = True
    exit_rank: int = 35
    require_industry: bool = True
    industry_neutralization_enabled: bool = True
    max_industry_weight: float = 0.25
    require_size_data: bool = True
    size_neutralization_enabled: bool = True
    size_neutralization_strength: float = 1.0

    @property
    def factor_weights(self) -> dict[str, float]:
        return {
            "mom_12_1": self.momentum_12_1_weight,
            "mom_6_1": self.momentum_6_1_weight,
            "fip_momentum": self.fip_momentum_weight,
            "trend": self.trend_weight,
            "low_vol": self.low_volatility_weight,
            "low_downside_vol": self.low_downside_volatility_weight,
            "drawdown_quality": self.drawdown_quality_weight,
            "liquidity": self.liquidity_weight,
        }


@dataclass(frozen=True)
class FeeScheduleConfig:
    start_date: str
    end_date: str | None
    stamp_duty_sell: float
    transfer_fee_rate: float

    def contains(self, value: date | datetime | str) -> bool:
        current = pd_timestamp(value)
        start = pd_timestamp(self.start_date)
        end = pd_timestamp(self.end_date) if self.end_date else datetime.max
        return start <= current <= end


def pd_timestamp(value: date | datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return datetime.strptime(str(value)[:10], "%Y-%m-%d")


def _default_fee_schedule() -> tuple[FeeScheduleConfig, ...]:
    return (
        FeeScheduleConfig("1900-01-01", "2022-04-28", 0.0010, 0.00002),
        FeeScheduleConfig("2022-04-29", "2023-08-27", 0.0010, 0.00001),
        FeeScheduleConfig("2023-08-28", None, 0.0005, 0.00001),
    )


@dataclass(frozen=True)
class ExecutionConfig:
    lot_size: int = 100
    commission_rate: float = 0.00025
    minimum_commission: float = 5.0
    fee_schedule: tuple[FeeScheduleConfig, ...] = field(default_factory=_default_fee_schedule)
    slippage_bps: float = 5.0
    cash_buffer: float = 0.02
    max_participation_of_20d_amount: float = 0.05
    rebalance_retry_days: int = 3
    sizing_price: str = "signal_close"
    reject_st_on_execution: bool = True

    def fee_on(self, value: date | datetime | str) -> FeeScheduleConfig:
        matches = [tier for tier in self.fee_schedule if tier.contains(value)]
        if len(matches) != 1:
            raise ValueError(f"{str(value)[:10]} 未匹配到唯一费用区间")
        return matches[0]


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
        execution_values = dict(raw.get("execution", {}))
        fee_values = execution_values.pop("fee_schedule", None)
        fee_schedule = (
            tuple(
                FeeScheduleConfig(
                    **{
                        **dict(item),
                        "start_date": _date_text(dict(item)["start_date"]),
                        "end_date": (
                            None
                            if dict(item).get("end_date") in {None, ""}
                            else _date_text(dict(item)["end_date"])
                        ),
                    }
                )
                for item in fee_values
            )
            if fee_values is not None
            else _default_fee_schedule()
        )
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
            execution=ExecutionConfig(**execution_values, fee_schedule=fee_schedule),
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
        if self.backtest.annual_cash_rate <= -1:
            raise ValueError("annual_cash_rate 必须大于 -100%")
        if self.backtest.maximum_stale_trading_days < 1:
            raise ValueError("maximum_stale_trading_days 必须大于 0")
        if self.backtest.stale_price_policy not in {"warn", "error"}:
            raise ValueError("stale_price_policy 仅支持 warn 或 error")
        if self.backtest.delist_value_policy not in {"zero", "last_close"}:
            raise ValueError("delist_value_policy 仅支持 zero 或 last_close")
        if self.strategy.top_n < 2:
            raise ValueError("top_n 至少为 2")
        if self.strategy.selection_buffer_enabled and self.strategy.exit_rank < self.strategy.top_n:
            raise ValueError("启用排名缓冲时 exit_rank 不能小于 top_n")
        factor_sum = sum(self.strategy.factor_weights.values())
        if factor_sum <= 0:
            raise ValueError("因子权重之和必须大于 0")
        negative_factors = [
            name
            for name, value in self.strategy.factor_weights.items()
            if not 0 <= value < float("inf")
        ]
        if negative_factors:
            raise ValueError("因子权重必须是非负有限数: " + ", ".join(negative_factors))
        if self.strategy.volatility_lookback < 20:
            raise ValueError("volatility_lookback 至少为 20")
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
        if not 0 < self.strategy.max_industry_weight <= 1:
            raise ValueError("max_industry_weight 必须在 (0, 1] 内")
        if not 0 <= self.strategy.size_neutralization_strength <= 1:
            raise ValueError("size_neutralization_strength 必须在 [0, 1] 内")
        if self.data.industry_standard not in {"SW2014", "SW2021"}:
            raise ValueError("industry_standard 仅支持 SW2014 或 SW2021")
        if self.data.industry_level not in {"L1", "L2", "L3"}:
            raise ValueError("industry_level 仅支持 L1、L2 或 L3")
        for name, value in {
            "universe_index": self.data.universe_index,
            "regime_index": self.data.regime_index,
            "benchmark_index": self.data.benchmark_index,
        }.items():
            if not str(value).strip():
                raise ValueError(f"{name} 不能为空")
        if (
            self.data.benchmark_is_total_return
            and self.data.regime_index == self.data.benchmark_index
        ):
            raise ValueError("全收益业绩基准不能同时作为价格择时指数")
        if self.execution.lot_size <= 0:
            raise ValueError("lot_size 必须大于 0")
        if min(
            self.execution.commission_rate,
            self.execution.minimum_commission,
            self.execution.slippage_bps,
            self.execution.max_participation_of_20d_amount,
        ) < 0:
            raise ValueError("费用、滑点和成交占比不能为负")
        if not 0 <= self.execution.cash_buffer < 1:
            raise ValueError("cash_buffer 必须在 [0, 1) 内")
        if self.execution.sizing_price != "signal_close":
            raise ValueError("日线引擎目前仅支持 sizing_price=signal_close")
        if not self.execution.fee_schedule:
            raise ValueError("fee_schedule 不能为空")
        previous_end: datetime | None = None
        for index, tier in enumerate(self.execution.fee_schedule):
            start_date = pd_timestamp(tier.start_date)
            end_date = pd_timestamp(tier.end_date) if tier.end_date else datetime.max
            if start_date > end_date:
                raise ValueError("费用区间开始日期不能晚于结束日期")
            if previous_end is not None and start_date != previous_end + timedelta(days=1):
                raise ValueError("fee_schedule 必须连续且不得重叠")
            if min(tier.stamp_duty_sell, tier.transfer_fee_rate) < 0:
                raise ValueError("费用率不能为负")
            previous_end = end_date if end_date != datetime.max else None
            if end_date == datetime.max and index != len(self.execution.fee_schedule) - 1:
                raise ValueError("开放式费用区间必须放在最后")
        self.execution.fee_on(self.backtest.start_date)
        self.execution.fee_on(self.backtest.end_date)

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
