from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    reference_prices: dict[str, float]


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


def _group_capped_allocation(
    raw: pd.Series,
    total: float,
    stock_cap: float,
    groups: pd.Series | None = None,
    group_cap: float = 1.0,
) -> pd.Series:
    """Proportionally allocate while respecting stock and optional group caps."""
    if raw.empty or total <= 0:
        return pd.Series(0.0, index=raw.index, dtype=float)
    if stock_cap <= 0 or group_cap <= 0:
        return pd.Series(0.0, index=raw.index, dtype=float)

    preferences = pd.to_numeric(raw, errors="coerce").fillna(0.0).clip(lower=0.0)
    if preferences.sum() <= 0:
        preferences[:] = 1.0
    if groups is None:
        group_labels = pd.Series("ALL", index=raw.index, dtype="object")
        effective_group_cap = 1.0
    else:
        group_labels = groups.reindex(raw.index).fillna("UNKNOWN").astype(str)
        effective_group_cap = float(group_cap)

    target = min(
        float(total),
        len(raw) * float(stock_cap),
        group_labels.nunique() * effective_group_cap,
    )
    result = pd.Series(0.0, index=raw.index, dtype=float)

    for _ in range(len(raw) + group_labels.nunique() + 5):
        remaining_total = target - float(result.sum())
        if remaining_total <= 1e-12:
            break
        stock_room = float(stock_cap) - result
        group_used = result.groupby(group_labels).sum()
        group_room = effective_group_cap - group_labels.map(group_used).fillna(0.0)
        eligible = (stock_room > 1e-12) & (group_room > 1e-12)
        if not eligible.any():
            break

        base = preferences.loc[eligible]
        if base.sum() <= 0:
            base = pd.Series(1.0, index=base.index)
        proposal = base / base.sum() * remaining_total
        alpha = 1.0
        positive = proposal > 1e-15
        if positive.any():
            alpha = min(
                alpha,
                float((stock_room.loc[proposal.index][positive] / proposal[positive]).min()),
            )
        proposal_groups = proposal.groupby(group_labels.loc[proposal.index]).sum()
        used_by_group = result.groupby(group_labels).sum()
        for group, amount in proposal_groups.items():
            if amount > 1e-15:
                room = effective_group_cap - float(used_by_group.get(group, 0.0))
                alpha = min(alpha, room / float(amount))
        alpha = max(0.0, min(1.0, alpha))
        if alpha <= 1e-15:
            break
        result.loc[proposal.index] += proposal * alpha
        if alpha >= 1.0 - 1e-12:
            break
    return result


def _capped_allocation(raw: pd.Series, total: float, cap: float) -> pd.Series:
    """Backward-compatible single-stock capped allocator."""
    return _group_capped_allocation(raw, total, cap)


class MultiFactorStrategy:
    """Point-in-time multi-factor strategy with industry and size controls."""

    def __init__(self, bundle: MarketDataBundle, config: StrategyConfig) -> None:
        self.bundle = bundle.prepare()
        self.config = config
        self.features = self._build_features(self.bundle.bars)
        self.membership_dates = pd.DatetimeIndex(
            sorted(self.bundle.membership["date"].drop_duplicates())
        )
        self.industry_history = {
            str(symbol): group.sort_values(["in_date", "out_date"], na_position="last")
            for symbol, group in self.bundle.industry_membership.groupby("symbol", sort=False)
        }
        benchmark = self.bundle.benchmark[["date", "close"]].copy().sort_values("date")
        benchmark["ma"] = benchmark["close"].rolling(
            config.benchmark_ma_days, min_periods=config.benchmark_ma_days
        ).mean()
        self.benchmark = benchmark.set_index("date")

    def _build_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy().sort_values(["symbol", "date"])
        frame["total_close"] = frame["close"] * frame["adj_factor"]
        frame["total_open"] = frame["open"] * frame["adj_factor"]
        frame["log_total_mv"] = np.log(
            pd.to_numeric(frame["total_mv"], errors="coerce").clip(lower=1e-12)
        )
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
            lambda value: value.rolling(
                lookback, min_periods=max(20, lookback - 10)
            ).std(ddof=0)
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
        history = self.benchmark.loc[self.benchmark.index <= signal_date]
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

    def generate(
        self,
        signal_date: pd.Timestamp | str,
        current_holdings: Iterable[str] | None = None,
    ) -> SignalPlan:
        signal_date = pd.Timestamp(signal_date).normalize()
        exact = self.features.loc[self.features["date"].eq(signal_date)].copy()
        liquidity = dict(zip(exact["symbol"], exact["avg_amount_20"], strict=False))
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
        required = ["mom_12_1", "mom_6_1", "trend", "volatility", "avg_amount_20"]
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
            return SignalPlan(signal_date, {}, empty, regime, 0.0, liquidity, {})

        factor_inputs = {
            "mom_12_1": candidates["mom_12_1"],
            "mom_6_1": candidates["mom_6_1"],
            "trend": candidates["trend"],
            "low_vol": -candidates["volatility"],
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
        candidates["z_size"] = _winsorized_zscore(
            candidates["log_total_mv"], self.config.winsor_quantile
        )
        candidates["score"] = self._neutralize_score(candidates)

        candidates = candidates.sort_values(
            ["score", "symbol"], ascending=[False, True]
        )
        candidates["rank"] = np.arange(1, len(candidates) + 1)
        selected = self._select_with_buffer(candidates, current_holdings or set())
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
        selected["target_weight"] = _group_capped_allocation(
            inverse_risk,
            feasible_exposure,
            self.config.max_stock_weight,
            selected["industry_code"],
            self.config.max_industry_weight,
        )
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
            "target_weight",
            "regime",
            "target_exposure",
            "signal_close",
            "total_mv",
            "circ_mv",
            "z_size",
            "mom_12_1",
            "mom_6_1",
            "trend",
            "volatility",
            "avg_amount_20",
            "z_mom_12_1",
            "z_mom_6_1",
            "z_trend",
            "z_low_vol",
            "z_liquidity",
        ]
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
