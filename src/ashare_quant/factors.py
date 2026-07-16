from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .alpha import build_price_alpha_features
from .config import PortfolioConfig, StrategyConfig
from .data import MarketDataBundle
from .portfolio import (
    allocate_portfolio,
    group_capped_allocation as _group_capped_allocation,
)


@dataclass
class SignalPlan:
    signal_date: pd.Timestamp
    weights: dict[str, float]
    selection: pd.DataFrame
    regime: str
    target_exposure: float
    liquidity: dict[str, float]
    reference_prices: dict[str, float]
    volatility: dict[str, float]
    portfolio_model: str
    portfolio_status: str


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
    return ((clipped - clipped.mean()) / deviation).fillna(0.0)


def _capped_allocation(raw: pd.Series, total: float, cap: float) -> pd.Series:
    """Backward-compatible single-stock capped allocator."""
    return _group_capped_allocation(raw, total, cap)


class MultiFactorStrategy:
    """Point-in-time multi-factor strategy with industry and size controls."""

    def __init__(
        self,
        bundle: MarketDataBundle,
        config: StrategyConfig,
        portfolio_config: PortfolioConfig | None = None,
    ) -> None:
        self.bundle = bundle.prepare()
        self.config = config
        self.portfolio_config = portfolio_config or PortfolioConfig()
        self.features = self._build_features(self.bundle.bars)
        self.membership_dates = pd.DatetimeIndex(
            sorted(self.bundle.membership["date"].drop_duplicates())
        )
        self.industry_history = {
            str(symbol): group.sort_values(["in_date", "out_date"], na_position="last")
            for symbol, group in self.bundle.industry_membership.groupby("symbol", sort=False)
        }
        regime = self.bundle.regime[["date", "close"]].copy().sort_values("date")
        regime["ma"] = regime["close"].rolling(
            config.benchmark_ma_days, min_periods=config.benchmark_ma_days
        ).mean()
        self.regime_index = regime.set_index("date")

    def _build_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy().sort_values(["symbol", "date"])
        frame["total_close"] = frame["close"] * frame["adj_factor"]
        frame["total_open"] = frame["open"] * frame["adj_factor"]
        frame["log_total_mv"] = np.log(
            pd.to_numeric(frame["total_mv"], errors="coerce").clip(lower=1e-12)
        )
        frame = build_price_alpha_features(
            frame,
            price_column="total_close",
            volatility_lookback=self.config.volatility_lookback,
        )
        grouped = frame.groupby("symbol", sort=False, group_keys=False)
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

    def _industry_at(self, symbol: str, signal_date: pd.Timestamp) -> tuple[str, str]:
        history = self.industry_history.get(str(symbol))
        if history is None or history.empty:
            return "UNKNOWN", "未知行业"
        active = history.loc[
            history["in_date"].le(signal_date)
            & (history["out_date"].isna() | history["out_date"].ge(signal_date))
        ]
        if active.empty:
            return "UNKNOWN", "未知行业"
        latest = active.sort_values(["in_date", "industry_code"]).iloc[-1]
        return str(latest["industry_code"]), str(latest["industry_name"])

    def _regime_at(self, signal_date: pd.Timestamp) -> tuple[str, float]:
        history = self.regime_index.loc[self.regime_index.index <= signal_date]
        if history.empty:
            return "RISK_OFF", self.config.risk_off_exposure
        latest = history.iloc[-1]
        if pd.notna(latest["ma"]) and latest["close"] >= latest["ma"]:
            return "RISK_ON", self.config.risk_on_exposure
        return "RISK_OFF", self.config.risk_off_exposure

    def _neutralize_score(self, candidates: pd.DataFrame) -> pd.Series:
        score = candidates["score_pre_neutral"].to_numpy(dtype=float)
        use_industry = self.config.industry_neutralization_enabled
        use_size = self.config.size_neutralization_enabled
        if not use_industry and not use_size:
            return pd.Series(score, index=candidates.index)

        parts: list[np.ndarray] = [np.ones((len(candidates), 1), dtype=float)]
        industry_width = 0
        if use_industry:
            dummies = pd.get_dummies(
                candidates["industry_code"].astype(str), drop_first=True, dtype=float
            )
            if not dummies.empty:
                values = dummies.to_numpy(dtype=float)
                parts.append(values)
                industry_width = values.shape[1]
        if use_size:
            parts.append(candidates[["z_size"]].to_numpy(dtype=float))

        design = np.column_stack(parts)
        coefficients, *_ = np.linalg.lstsq(design, score, rcond=None)
        adjustment = design[:, 0] * coefficients[0]
        cursor = 1
        if industry_width:
            adjustment += design[:, cursor : cursor + industry_width] @ coefficients[
                cursor : cursor + industry_width
            ]
            cursor += industry_width
        if use_size:
            adjustment += (
                self.config.size_neutralization_strength
                * design[:, cursor]
                * coefficients[cursor]
            )
        residual = score - adjustment
        return _winsorized_zscore(pd.Series(residual, index=candidates.index), 0.0)

    def _select_with_buffer(
        self, ranked: pd.DataFrame, current_holdings: Iterable[str]
    ) -> pd.DataFrame:
        held = set(map(str, current_holdings))
        ranked = ranked.copy()
        ranked["was_held"] = ranked["symbol"].astype(str).isin(held)
        kept_indices: list[int] = []
        if self.config.selection_buffer_enabled and held:
            kept_indices = list(
                ranked.loc[
                    ranked["was_held"] & ranked["rank"].le(self.config.exit_rank)
                ]
                .head(self.config.top_n)
                .index
            )
        fill_indices = [
            index
            for index in ranked.index
            if index not in set(kept_indices)
        ][: max(0, self.config.top_n - len(kept_indices))]
        selected = ranked.loc[kept_indices + fill_indices].copy()
        selected = selected.sort_values(["rank", "symbol"])
        selected["selection_reason"] = np.where(
            selected.index.isin(kept_indices), "HOLD_BUFFER", "NEW_ENTRY"
        )
        return selected

    def _apply_research_score_overlay(
        self,
        candidates: pd.DataFrame,
        signal_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Extension point for explicitly constructed research-only strategies.

        The production strategy deliberately returns the input unchanged.  V2
        shadow strategies override this hook instead of adding PIT fields to
        ``StrategyConfig`` and accidentally making them selectable as a
        production default.
        """
        del signal_date
        return candidates

    def _research_selection_columns(self) -> list[str]:
        """Additional auditable columns emitted by research-only subclasses."""
        return []

    def _empty_plan(
        self,
        *,
        signal_date: pd.Timestamp,
        regime: str,
        liquidity: dict[str, float],
        volatility: dict[str, float],
        status: str = "empty_universe",
    ) -> SignalPlan:
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
        return SignalPlan(
            signal_date=signal_date,
            weights={},
            selection=empty,
            regime=regime,
            target_exposure=0.0,
            liquidity=liquidity,
            reference_prices={},
            volatility=volatility,
            portfolio_model=self.portfolio_config.construction_model,
            portfolio_status=status,
        )

    def generate(
        self,
        signal_date: pd.Timestamp | str,
        current_holdings: Iterable[str] | None = None,
        current_weights: dict[str, float] | None = None,
    ) -> SignalPlan:
        signal_date = pd.Timestamp(signal_date).normalize()
        exact = self.features.loc[self.features["date"].eq(signal_date)].copy()
        liquidity = dict(zip(exact["symbol"], exact["avg_amount_20"], strict=False))
        volatility = dict(zip(exact["symbol"], exact["volatility"], strict=False))
        members = self._members_at(signal_date)
        candidates = exact.loc[exact["symbol"].isin(members)].copy()
        candidates = candidates.loc[
            (~candidates["is_st"])
            & (candidates["history_days"] >= self.config.min_history_days)
            & (
                candidates["avg_amount_20"]
                >= self.config.min_avg_amount_million * 1_000_000.0
            )
            & (candidates["close"] >= self.config.min_price)
        ]
        if self.config.stock_trend_filter:
            candidates = candidates.loc[candidates["trend"] > 0.0]
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
        if self.config.require_size_data:
            required.append("log_total_mv")
        candidates = candidates.dropna(subset=required)

        industries = [
            self._industry_at(symbol, signal_date) for symbol in candidates["symbol"]
        ]
        candidates["industry_code"] = [value[0] for value in industries]
        candidates["industry_name"] = [value[1] for value in industries]
        if self.config.require_industry:
            candidates = candidates.loc[candidates["industry_code"].ne("UNKNOWN")]

        regime, requested_exposure = self._regime_at(signal_date)
        if candidates.empty:
            return self._empty_plan(
                signal_date=signal_date,
                regime=regime,
                liquidity=liquidity,
                volatility=volatility,
            )

        factor_inputs = {
            "mom_12_1": candidates["mom_12_1"],
            "mom_6_1": candidates["mom_6_1"],
            "fip_momentum": candidates["fip_momentum"],
            "trend": candidates["trend"],
            "low_vol": -candidates["volatility"],
            "low_downside_vol": -candidates["downside_volatility"],
            "drawdown_quality": candidates["drawdown_quality"],
            "liquidity": np.log1p(candidates["avg_amount_20"]),
        }
        weights = self.config.factor_weights
        weight_sum = sum(weights.values())
        candidates["score_pre_neutral"] = 0.0
        for factor, values in factor_inputs.items():
            z_column = f"z_{factor}"
            candidates[z_column] = _winsorized_zscore(
                values, self.config.winsor_quantile
            )
            candidates["score_pre_neutral"] += (
                candidates[z_column] * weights[factor] / weight_sum
            )
        candidates = self._apply_research_score_overlay(
            candidates, signal_date
        )
        if candidates.empty:
            return self._empty_plan(
                signal_date=signal_date,
                regime=regime,
                liquidity=liquidity,
                volatility=volatility,
                status="empty_research_overlay",
            )
        candidates["z_size"] = _winsorized_zscore(
            candidates["log_total_mv"], self.config.winsor_quantile
        )
        candidates["score"] = self._neutralize_score(candidates)

        candidates = candidates.sort_values(
            ["score", "symbol"], ascending=[False, True]
        )
        candidates["rank"] = np.arange(1, len(candidates) + 1)
        selected = self._select_with_buffer(candidates, current_holdings or set())
        selected = selected.set_index("symbol", drop=False)
        industry_capacity = (
            selected["industry_code"].nunique() * self.config.max_industry_weight
        )
        feasible_exposure = min(
            requested_exposure,
            len(selected) * self.config.max_stock_weight,
            industry_capacity,
        )
        inverse_risk = (1.0 / selected["volatility"].clip(lower=1e-6)).pow(
            self.config.risk_weight_power
        )
        trailing_returns = self._trailing_returns(
            signal_date,
            selected["symbol"],
            self.portfolio_config.covariance_lookback_days,
        )
        allocation = allocate_portfolio(
            model=self.portfolio_config.construction_model,
            inverse_risk=inverse_risk,
            total=feasible_exposure,
            stock_cap=self.config.max_stock_weight,
            groups=selected["industry_code"],
            group_cap=self.config.max_industry_weight,
            returns=trailing_returns,
            current_weights=current_weights,
            covariance_lookback_days=self.portfolio_config.covariance_lookback_days,
            minimum_covariance_observations=(
                self.portfolio_config.minimum_covariance_observations
            ),
            covariance_shrinkage=self.portfolio_config.covariance_shrinkage,
            minimum_variance_blend=self.portfolio_config.minimum_variance_blend,
            turnover_smoothing=self.portfolio_config.turnover_smoothing,
            covariance_ridge=self.portfolio_config.covariance_ridge,
        )
        selected["raw_target_weight"] = allocation.raw_weights
        selected["target_weight"] = allocation.weights
        selected["current_weight"] = selected["symbol"].map(
            current_weights or {}
        ).fillna(0.0)
        selected["target_weight_change"] = (
            selected["target_weight"] - selected["current_weight"]
        )
        selected["portfolio_model"] = self.portfolio_config.construction_model
        selected["portfolio_status"] = allocation.status
        selected["covariance_observations"] = allocation.covariance_observations
        selected["signal_date"] = signal_date
        selected["signal_close"] = selected["close"]
        selected["regime"] = regime
        selected["target_exposure"] = float(selected["target_weight"].sum())
        target_weights = dict(
            zip(selected["symbol"], selected["target_weight"], strict=False)
        )
        output_columns = [
            "signal_date",
            "symbol",
            "name",
            "industry_code",
            "industry_name",
            "rank",
            "selection_reason",
            "was_held",
            "score_pre_neutral",
            "score",
            "portfolio_model",
            "portfolio_status",
            "covariance_observations",
            "raw_target_weight",
            "current_weight",
            "target_weight_change",
            "target_weight",
            "regime",
            "target_exposure",
            "signal_close",
            "total_mv",
            "circ_mv",
            "z_size",
            "mom_12_1",
            "mom_6_1",
            "information_discreteness",
            "fip_momentum",
            "trend",
            "volatility",
            "downside_volatility",
            "drawdown_quality",
            "avg_amount_20",
            "z_mom_12_1",
            "z_mom_6_1",
            "z_fip_momentum",
            "z_trend",
            "z_low_vol",
            "z_low_downside_vol",
            "z_drawdown_quality",
            "z_liquidity",
        ]
        for column in self._research_selection_columns():
            if column in selected and column not in output_columns:
                output_columns.append(column)
        return SignalPlan(
            signal_date=signal_date,
            weights=target_weights,
            selection=selected[output_columns].reset_index(drop=True),
            regime=regime,
            target_exposure=float(selected["target_weight"].sum()),
            liquidity=liquidity,
            reference_prices=dict(
                zip(selected["symbol"], selected["signal_close"], strict=False)
            ),
            volatility=volatility,
            portfolio_model=self.portfolio_config.construction_model,
            portfolio_status=allocation.status,
        )

    def _trailing_returns(
        self,
        signal_date: pd.Timestamp,
        symbols: Iterable[str],
        lookback_days: int,
    ) -> pd.DataFrame:
        requested = list(map(str, symbols))
        rows = self.features.loc[
            self.features["date"].le(signal_date)
            & self.features["symbol"].isin(requested),
            ["date", "symbol", "return_1d"],
        ]
        if rows.empty:
            return pd.DataFrame(columns=requested, dtype=float)
        return (
            rows.pivot(index="date", columns="symbol", values="return_1d")
            .reindex(columns=requested)
            .sort_index()
            .tail(lookback_days)
        )

    def trailing_amount(self, symbol: str, on_or_before: pd.Timestamp) -> float:
        rows = self.features.loc[
            self.features["symbol"].eq(symbol)
            & self.features["date"].le(on_or_before),
            ["date", "avg_amount_20"],
        ]
        if rows.empty:
            return np.nan
        return float(rows.sort_values("date").iloc[-1]["avg_amount_20"])

    def trailing_volatility(self, symbol: str, on_or_before: pd.Timestamp) -> float:
        rows = self.features.loc[
            self.features["symbol"].eq(symbol)
            & self.features["date"].le(on_or_before),
            ["date", "volatility"],
        ]
        if rows.empty:
            return np.nan
        return float(rows.sort_values("date").iloc[-1]["volatility"])
