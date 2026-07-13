from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd


ALPHA_MODEL_VERSION = "quality_momentum_v1_5"
MOMENTUM_SKIP_DAYS = 21
MOMENTUM_6M_DAYS = 126
MOMENTUM_12M_DAYS = 252
FORMATION_RETURN_DAYS = MOMENTUM_12M_DAYS - MOMENTUM_SKIP_DAYS
DRAWDOWN_LOOKBACK_DAYS = 126
STOCK_TREND_DAYS = 200

# Frozen before any v1.5 result was generated. Liquidity remains an eligibility
# constraint instead of receiving an alpha weight.
QUALITY_MOMENTUM_V1_5_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "mom_12_1": 0.25,
    "mom_6_1": 0.15,
    "fip_momentum": 0.25,
    "trend": 0.10,
    "low_vol": 0.00,
    "low_downside_vol": 0.15,
    "drawdown_quality": 0.10,
    "liquidity": 0.00,
})

LEGACY_V1_4_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "mom_12_1": 0.35,
    "mom_6_1": 0.20,
    "fip_momentum": 0.00,
    "trend": 0.15,
    "low_vol": 0.20,
    "low_downside_vol": 0.00,
    "drawdown_quality": 0.00,
    "liquidity": 0.10,
})


def identify_alpha_profile(weights: Mapping[str, float]) -> str:
    normalized = {str(name): float(value) for name, value in weights.items()}
    for profile, expected in {
        "legacy_v1_4": LEGACY_V1_4_WEIGHTS,
        ALPHA_MODEL_VERSION: QUALITY_MOMENTUM_V1_5_WEIGHTS,
    }.items():
        if normalized.keys() == expected.keys() and all(
            math.isclose(normalized[name], expected[name], abs_tol=1e-12)
            for name in expected
        ):
            return profile
    return "custom"


def _rolling_fraction(
    returns: pd.Series,
    *,
    positive: bool,
) -> pd.Series:
    indicator = (returns > 0 if positive else returns < 0).astype(float)
    indicator = indicator.where(returns.notna())
    return (
        indicator.shift(MOMENTUM_SKIP_DAYS)
        .rolling(
            FORMATION_RETURN_DAYS,
            min_periods=FORMATION_RETURN_DAYS,
        )
        .mean()
    )


def build_price_alpha_features(
    bars: pd.DataFrame,
    *,
    price_column: str,
    volatility_lookback: int,
) -> pd.DataFrame:
    """Build point-in-time price features shared by strict and public research.

    ``information_discreteness`` follows Da, Gurun, and Warachka (2014):
    ``sign(PRET) * (%negative - %positive)`` over the 12-to-1-month formation
    window. ``fip_momentum`` interacts that measure with formation-period
    momentum so gradual winners rank above jump-driven winners, while gradual
    losers remain unattractive to a long-only strategy.
    """
    required = {"symbol", "date", price_column}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError("价格 Alpha 缺少字段: " + ", ".join(missing))
    if volatility_lookback < 20:
        raise ValueError("volatility_lookback 至少为 20")

    frame = bars.copy().sort_values(["symbol", "date"])
    price = pd.to_numeric(frame[price_column], errors="coerce")
    frame[price_column] = price
    grouped = frame.groupby("symbol", sort=False, group_keys=False)

    frame["return_1d"] = grouped[price_column].pct_change(fill_method=None)
    frame["mom_12_1"] = grouped[price_column].transform(
        lambda value: value.shift(MOMENTUM_SKIP_DAYS)
        / value.shift(MOMENTUM_12M_DAYS)
        - 1.0
    )
    frame["mom_6_1"] = grouped[price_column].transform(
        lambda value: value.shift(MOMENTUM_SKIP_DAYS)
        / value.shift(MOMENTUM_6M_DAYS)
        - 1.0
    )
    frame["ma_200"] = grouped[price_column].transform(
        lambda value: value.rolling(
            STOCK_TREND_DAYS,
            min_periods=STOCK_TREND_DAYS,
        ).mean()
    )
    frame["trend"] = frame[price_column] / frame["ma_200"] - 1.0

    minimum_volatility_observations = max(20, volatility_lookback - 10)
    frame["volatility"] = grouped["return_1d"].transform(
        lambda value: value.rolling(
            volatility_lookback,
            min_periods=minimum_volatility_observations,
        ).std(ddof=0)
        * math.sqrt(252.0)
    )
    frame["downside_volatility"] = grouped["return_1d"].transform(
        lambda value: value.clip(upper=0.0)
        .pow(2.0)
        .rolling(
            volatility_lookback,
            min_periods=minimum_volatility_observations,
        )
        .mean()
        .pow(0.5)
        * math.sqrt(252.0)
    )

    positive_fraction = grouped["return_1d"].transform(
        lambda value: _rolling_fraction(value, positive=True)
    )
    negative_fraction = grouped["return_1d"].transform(
        lambda value: _rolling_fraction(value, positive=False)
    )
    frame["information_discreteness"] = np.sign(frame["mom_12_1"]) * (
        negative_fraction - positive_fraction
    )
    frame["fip_momentum"] = frame["mom_12_1"] * (
        1.0 - frame["information_discreteness"]
    )

    rolling_peak = grouped[price_column].transform(
        lambda value: value.rolling(
            DRAWDOWN_LOOKBACK_DAYS,
            min_periods=DRAWDOWN_LOOKBACK_DAYS,
        ).max()
    )
    frame["drawdown_quality"] = frame[price_column] / rolling_peak - 1.0
    return frame
