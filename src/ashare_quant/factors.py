from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .data import MarketDataBundle


@dataclass
class SignalPlan:
    signal_date: pd.Timestamp
    weights: dict[str, float]
    selection: pd.DataFrame
    regime: str
    target_exposure: float
    liquidity: dict[str, float]


def _winsorized_zscore(series: pd.Series, quantile: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() < 2:
        return pd.Series(0.0, index=series.index)
    lower = values.quantile(quantile)
    upper = values.quantile(1.0 - quantile)
    clipped = values.clip(lower, upper)
    deviation = clipped.std(ddof=0)
    if not np.isfinite(deviation) or deviation < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (clipped - clipped.mean()) / deviation


def _capped_allocation(raw: pd.Series, total: float, cap: float) -> pd.Series:
    if raw.empty or total <= 0:
        return pd.Series(0.0, index=raw.index)
    total = min(float(total), len(raw) * cap)
    positive = raw.clip(lower=0.0).astype(float)
    if positive.sum() <= 0:
        positive[:] = 1.0

    result = pd.Series(0.0, index=raw.index)
    remaining = list(raw.index)
    remaining_total = total
    while remaining and remaining_total > 1e-12:
        base = positive.loc[remaining]
        if base.sum() <= 0:
            base = pd.Series(1.0, index=remaining)
        proposal = base / base.sum() * remaining_total
        capped = proposal[proposal > cap + 1e-12]
        if capped.empty:
            result.loc[remaining] = proposal
            break
        for item in capped.index:
            result.loc[item] = cap
            remaining_total -= cap
            remaining.remove(item)
    return result


class MultiFactorStrategy:
    """Point-in-time cross-sectional momentum/low-volatility strategy."""

    def __init__(self, bundle: MarketDataBundle, config: StrategyConfig) -> None:
        self.bundle = bundle.prepare()
        self.config = config
        self.features = self._build_features(self.bundle.bars)
        self.membership_dates = pd.DatetimeIndex(
            sorted(self.bundle.membership["date"].drop_duplicates())
        )
        benchmark = self.bundle.benchmark[["date", "close"]].copy().sort_values("date")
        benchmark["ma"] = benchmark["close"].rolling(
            config.benchmark_ma_days, min_periods=config.benchmark_ma_days
        ).mean()
        self.benchmark = benchmark.set_index("date")

    def _build_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy().sort_values(["symbol", "date"])
        frame["total_close"] = frame["close"] * frame["adj_factor"]
        frame["total_open"] = frame["open"] * frame["adj_factor"]
        grouped = frame.groupby("symbol", sort=False, group_keys=False)
        frame["return_1d"] = grouped["total_close"].pct_change(fill_method=None)
        frame["mom_12_1"] = grouped["total_close"].transform(
            lambda value: value.shift(21) / value.shift(252) - 1.0
        )
        frame["mom_6_1"] = grouped["total_close"].transform(
            lambda value: value.shift(21) / value.shift(126) - 1.0
        )
        frame["ma_200"] = grouped["total_close"].transform(
            lambda value: value.rolling(200, min_periods=200).mean()
        )
        frame["trend"] = frame["total_close"] / frame["ma_200"] - 1.0
        lookback = self.config.volatility_lookback
        frame["volatility"] = grouped["return_1d"].transform(
            lambda value: value.rolling(lookback, min_periods=max(20, lookback - 10)).std(ddof=0)
            * np.sqrt(252.0)
        )
        frame["avg_amount_20"] = grouped["amount"].transform(
            lambda value: value.rolling(20, min_periods=15).mean()
        )
        frame["avg_volume_20"] = grouped["volume"].transform(
            lambda value: value.rolling(20, min_periods=15).mean()
        )
        frame["history_days"] = grouped.cumcount() + 1
        return frame

    def _members_at(self, signal_date: pd.Timestamp) -> set[str]:
        eligible = self.membership_dates[self.membership_dates <= signal_date]
        if eligible.empty:
            return set()
        snapshot = eligible[-1]
        return set(
            self.bundle.membership.loc[
                self.bundle.membership["date"].eq(snapshot), "symbol"
            ].astype(str)
        )

    def _regime_at(self, signal_date: pd.Timestamp) -> tuple[str, float]:
        history = self.benchmark.loc[self.benchmark.index <= signal_date]
        if history.empty:
            return "RISK_OFF", self.config.risk_off_exposure
        latest = history.iloc[-1]
        if pd.notna(latest["ma"]) and latest["close"] >= latest["ma"]:
            return "RISK_ON", self.config.risk_on_exposure
        return "RISK_OFF", self.config.risk_off_exposure

    def generate(self, signal_date: pd.Timestamp | str) -> SignalPlan:
        signal_date = pd.Timestamp(signal_date).normalize()
        exact = self.features.loc[self.features["date"].eq(signal_date)].copy()
        liquidity = dict(zip(exact["symbol"], exact["avg_amount_20"], strict=False))
        members = self._members_at(signal_date)
        candidates = exact.loc[exact["symbol"].isin(members)].copy()
        candidates = candidates.loc[
            (~candidates["is_st"])
            & (candidates["history_days"] >= self.config.min_history_days)
            & (candidates["avg_amount_20"] >= self.config.min_avg_amount_million * 1_000_000.0)
            & (candidates["close"] >= self.config.min_price)
        ]
        if self.config.stock_trend_filter:
            candidates = candidates.loc[candidates["trend"] > 0.0]
        required = ["mom_12_1", "mom_6_1", "trend", "volatility", "avg_amount_20"]
        candidates = candidates.dropna(subset=required)

        regime, requested_exposure = self._regime_at(signal_date)
        if candidates.empty:
            empty = pd.DataFrame(
                columns=[
                    "signal_date",
                    "symbol",
                    "rank",
                    "score",
                    "target_weight",
                    "regime",
                ]
            )
            return SignalPlan(signal_date, {}, empty, regime, 0.0, liquidity)

        factor_inputs = {
            "mom_12_1": candidates["mom_12_1"],
            "mom_6_1": candidates["mom_6_1"],
            "trend": candidates["trend"],
            "low_vol": -candidates["volatility"],
            "liquidity": np.log1p(candidates["avg_amount_20"]),
        }
        weights = self.config.factor_weights
        weight_sum = sum(weights.values())
        candidates["score"] = 0.0
        for factor, values in factor_inputs.items():
            z_column = f"z_{factor}"
            candidates[z_column] = _winsorized_zscore(values, self.config.winsor_quantile)
            candidates["score"] += candidates[z_column] * weights[factor] / weight_sum

        candidates = candidates.sort_values(["score", "symbol"], ascending=[False, True])
        selected = candidates.head(self.config.top_n).copy()
        selected["rank"] = np.arange(1, len(selected) + 1)
        feasible_exposure = min(
            requested_exposure, len(selected) * self.config.max_stock_weight
        )
        inverse_risk = (1.0 / selected["volatility"].clip(lower=1e-6)).pow(
            self.config.risk_weight_power
        )
        selected["target_weight"] = _capped_allocation(
            inverse_risk, feasible_exposure, self.config.max_stock_weight
        )
        selected["signal_date"] = signal_date
        selected["regime"] = regime
        selected["target_exposure"] = feasible_exposure
        target_weights = dict(
            zip(selected["symbol"], selected["target_weight"], strict=False)
        )
        output_columns = [
            "signal_date",
            "symbol",
            "name",
            "rank",
            "score",
            "target_weight",
            "regime",
            "target_exposure",
            "mom_12_1",
            "mom_6_1",
            "trend",
            "volatility",
            "avg_amount_20",
        ]
        return SignalPlan(
            signal_date=signal_date,
            weights=target_weights,
            selection=selected[output_columns].reset_index(drop=True),
            regime=regime,
            target_exposure=float(feasible_exposure),
            liquidity=liquidity,
        )

    def trailing_amount(self, symbol: str, on_or_before: pd.Timestamp) -> float:
        rows = self.features.loc[
            self.features["symbol"].eq(symbol) & self.features["date"].le(on_or_before),
            ["date", "avg_amount_20"],
        ]
        if rows.empty:
            return np.nan
        return float(rows.sort_values("date").iloc[-1]["avg_amount_20"])
