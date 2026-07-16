from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import AppConfig
from .data import MarketDataBundle
from .pit_data import PointInTimeDataBundle
from .provenance import (
    build_reproducibility_manifest,
    record_experiment,
    sha256_file,
    write_artifact_manifest,
    write_json_atomic,
)


PIT_FACTOR_RESEARCH_VERSION = "pit_factor_research_v2_alpha2"
PIT_COMPOSITE_NAME = "fundamental_value_composite_v2_alpha2"


@dataclass(frozen=True)
class PITFactorDefinition:
    name: str
    family: str
    input_columns: tuple[str, ...]
    direction: str
    formula: str
    description: str
    transform: Callable[[pd.DataFrame], pd.Series]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = (
        frame[column]
        if column in frame
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    return pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _positive_inverse(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _numeric(frame, column)
    return (1.0 / values).where(values > 0)


PIT_FACTOR_DEFINITIONS: tuple[PITFactorDefinition, ...] = (
    PITFactorDefinition(
        name="roe_quality",
        family="profitability",
        input_columns=("roe_pct",),
        direction="higher_is_better",
        formula="roe_pct",
        description="最新可见报告期的供应商口径 ROE。",
        transform=lambda frame: _numeric(frame, "roe_pct"),
    ),
    PITFactorDefinition(
        name="roa_quality",
        family="profitability",
        input_columns=("roa_pct",),
        direction="higher_is_better",
        formula="roa_pct",
        description="最新可见报告期的供应商口径 ROA。",
        transform=lambda frame: _numeric(frame, "roa_pct"),
    ),
    PITFactorDefinition(
        name="gross_margin_quality",
        family="profitability",
        input_columns=("gross_margin_pct",),
        direction="higher_is_better",
        formula="gross_margin_pct",
        description="最新可见报告期毛利率；不把累计值伪装成单季度值。",
        transform=lambda frame: _numeric(frame, "gross_margin_pct"),
    ),
    PITFactorDefinition(
        name="cashflow_quality",
        family="quality",
        input_columns=("operating_cash_to_revenue_pct",),
        direction="higher_is_better",
        formula="operating_cash_to_revenue_pct",
        description="经营现金流相对营业收入的供应商口径指标。",
        transform=lambda frame: _numeric(
            frame, "operating_cash_to_revenue_pct"
        ),
    ),
    PITFactorDefinition(
        name="low_leverage_quality",
        family="quality",
        input_columns=("debt_to_assets_pct",),
        direction="lower_is_better",
        formula="-debt_to_assets_pct",
        description="资产负债率取负；金融行业口径差异必须在暴露诊断中解释。",
        transform=lambda frame: -_numeric(frame, "debt_to_assets_pct"),
    ),
    PITFactorDefinition(
        name="revenue_growth",
        family="growth",
        input_columns=("revenue_yoy_pct",),
        direction="higher_is_better",
        formula="revenue_yoy_pct",
        description="最新可见报告期营业收入同比增速。",
        transform=lambda frame: _numeric(frame, "revenue_yoy_pct"),
    ),
    PITFactorDefinition(
        name="earnings_growth",
        family="growth",
        input_columns=("net_income_yoy_pct",),
        direction="higher_is_better",
        formula="net_income_yoy_pct",
        description="最新可见报告期净利润同比增速。",
        transform=lambda frame: _numeric(frame, "net_income_yoy_pct"),
    ),
    PITFactorDefinition(
        name="earnings_yield",
        family="valuation",
        input_columns=("pe_ttm",),
        direction="higher_is_better",
        formula="1 / pe_ttm, pe_ttm > 0",
        description="正盈利公司的滚动市盈率倒数；亏损公司记为缺失而非极端便宜。",
        transform=lambda frame: _positive_inverse(frame, "pe_ttm"),
    ),
    PITFactorDefinition(
        name="book_to_price",
        family="valuation",
        input_columns=("pb",),
        direction="higher_is_better",
        formula="1 / pb, pb > 0",
        description="正净资产公司的市净率倒数。",
        transform=lambda frame: _positive_inverse(frame, "pb"),
    ),
    PITFactorDefinition(
        name="sales_yield",
        family="valuation",
        input_columns=("ps_ttm",),
        direction="higher_is_better",
        formula="1 / ps_ttm, ps_ttm > 0",
        description="滚动市销率倒数。",
        transform=lambda frame: _positive_inverse(frame, "ps_ttm"),
    ),
    PITFactorDefinition(
        name="dividend_yield",
        family="valuation",
        input_columns=("dividend_yield_ttm_pct",),
        direction="higher_is_better",
        formula="dividend_yield_ttm_pct",
        description="供应商口径滚动股息率。",
        transform=lambda frame: _numeric(frame, "dividend_yield_ttm_pct"),
    ),
)

_PIT_FACTOR_BY_NAME = {
    definition.name: definition for definition in PIT_FACTOR_DEFINITIONS
}
_VALUATION_INPUT_COLUMNS = {
    "turnover_rate_pct",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dividend_yield_ttm_pct",
    "total_market_value_10k_cny",
    "float_market_value_10k_cny",
}


def resolve_pit_factor_names(
    factor_names: Iterable[str] | None = None,
) -> list[str]:
    requested = list(
        dict.fromkeys(str(value).strip() for value in (factor_names or ()))
    )
    if not requested or requested == ["all"]:
        return [definition.name for definition in PIT_FACTOR_DEFINITIONS]
    if "all" in requested:
        raise ValueError("因子列表中的 all 不能和具体因子同时使用")
    unknown = sorted(set(requested).difference(_PIT_FACTOR_BY_NAME))
    if unknown:
        raise ValueError("未知 PIT 因子: " + ", ".join(unknown))
    return requested


def pit_factor_registry_frame(
    factor_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    names = resolve_pit_factor_names(factor_names)
    records = []
    for name in names:
        definition = _PIT_FACTOR_BY_NAME[name]
        records.append(
            {
                "factor": definition.name,
                "family": definition.family,
                "source_kind": (
                    "valuation"
                    if set(definition.input_columns).issubset(
                        _VALUATION_INPUT_COLUMNS
                    )
                    else "fundamental"
                ),
                "input_columns": ",".join(definition.input_columns),
                "direction": definition.direction,
                "formula": definition.formula,
                "description": definition.description,
                "lifecycle_status": "research_only",
                "production_weight": 0.0,
            }
        )
    records.append(
        {
            "factor": PIT_COMPOSITE_NAME,
            "family": "composite",
            "source_kind": "derived",
            "input_columns": ",".join(names),
            "direction": "higher_is_better",
            "formula": "equal_weight_mean_of_available_cross_sectional_zscores",
            "description": "固定等权研究组合；不自动寻优，也不进入生产策略。",
            "lifecycle_status": "research_only",
            "production_weight": 0.0,
        }
    )
    return pd.DataFrame(records)


def compute_pit_factor_values(
    snapshot: pd.DataFrame,
    factor_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    names = resolve_pit_factor_names(factor_names)
    values = pd.DataFrame(index=snapshot.index)
    if "symbol" in snapshot:
        values["symbol"] = snapshot["symbol"].astype(str)
    for name in names:
        transformed = _PIT_FACTOR_BY_NAME[name].transform(snapshot)
        values[name] = pd.to_numeric(transformed, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return values


def _cross_sectional_zscore(
    values: pd.Series,
    winsor_quantile: float,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    valid = numeric.dropna()
    output = pd.Series(np.nan, index=values.index, dtype=float)
    if len(valid) < 3:
        return output
    lower = valid.quantile(winsor_quantile)
    upper = valid.quantile(1.0 - winsor_quantile)
    clipped = valid.clip(lower, upper)
    deviation = float(clipped.std(ddof=0))
    if not math.isfinite(deviation) or deviation < 1e-12:
        return output
    output.loc[clipped.index] = (clipped - float(clipped.mean())) / deviation
    return output


def _spearman_correlation(first: pd.Series, second: pd.Series) -> float:
    """Calculate Spearman correlation without adding SciPy as a dependency."""
    sample = pd.DataFrame(
        {
            "first": pd.to_numeric(first, errors="coerce"),
            "second": pd.to_numeric(second, errors="coerce"),
        }
    ).dropna()
    if (
        len(sample) < 2
        or sample["first"].nunique() < 2
        or sample["second"].nunique() < 2
    ):
        return np.nan
    return float(
        sample["first"].rank(method="average").corr(
            sample["second"].rank(method="average")
        )
    )


def _month_end_signal_dates(
    calendar: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    maximum_horizon: int,
) -> list[pd.Timestamp]:
    eligible = calendar[(calendar >= start_date) & (calendar <= end_date)]
    if eligible.empty:
        return []
    month_ends = (
        pd.Series(eligible, index=eligible)
        .groupby(eligible.to_period("M"))
        .max()
        .tolist()
    )
    locations = {date: index for index, date in enumerate(calendar)}
    return [
        pd.Timestamp(date)
        for date in month_ends
        if locations[pd.Timestamp(date)] + maximum_horizon < len(calendar)
    ]


def _members_at(bundle: MarketDataBundle, date: pd.Timestamp) -> list[str]:
    dates = pd.DatetimeIndex(
        bundle.membership.loc[
            bundle.membership["date"].le(date), "date"
        ].drop_duplicates()
    )
    if dates.empty:
        return []
    snapshot_date = dates.max()
    return sorted(
        bundle.membership.loc[
            bundle.membership["date"].eq(snapshot_date), "symbol"
        ]
        .astype(str)
        .unique()
        .tolist()
    )


def _industry_lookup(
    bundle: MarketDataBundle,
) -> dict[str, pd.DataFrame]:
    return {
        str(symbol): group.sort_values(
            ["in_date", "out_date", "industry_code"], na_position="last"
        )
        for symbol, group in bundle.industry_membership.groupby(
            "symbol", sort=False
        )
    }


def _industry_at(
    history: Mapping[str, pd.DataFrame],
    symbol: str,
    date: pd.Timestamp,
) -> tuple[str, str]:
    records = history.get(str(symbol))
    if records is None or records.empty:
        return "UNKNOWN", "未知行业"
    active = records.loc[
        records["in_date"].le(date)
        & (records["out_date"].isna() | records["out_date"].ge(date))
    ]
    if active.empty:
        return "UNKNOWN", "未知行业"
    latest = active.iloc[-1]
    return str(latest["industry_code"]), str(latest["industry_name"])


def _total_return_prices(
    bundle: MarketDataBundle,
    maximum_stale_days: int,
) -> pd.DataFrame:
    bars = bundle.bars[["date", "symbol", "close", "adj_factor"]].copy()
    bars["total_close"] = (
        pd.to_numeric(bars["close"], errors="coerce")
        * pd.to_numeric(bars["adj_factor"], errors="coerce")
    )
    prices = bars.pivot(index="date", columns="symbol", values="total_close")
    prices = prices.reindex(bundle.calendar).sort_index()
    return (
        prices
        if maximum_stale_days == 0
        else prices.ffill(limit=maximum_stale_days)
    )


def build_pit_factor_panel(
    market: MarketDataBundle,
    pit: PointInTimeDataBundle,
    *,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    factor_names: Iterable[str] | None = None,
    horizons: Sequence[int] = (21, 63),
    minimum_factors_per_symbol: int = 4,
    winsor_quantile: float = 0.05,
    maximum_fundamental_age_days: int | None = 550,
    maximum_valuation_age_days: int | None = 10,
    maximum_stale_price_days: int = 20,
) -> pd.DataFrame:
    """Build a monthly research panel without feeding PIT data into production."""
    names = resolve_pit_factor_names(factor_names)
    requested_horizons = sorted({int(value) for value in horizons})
    if not requested_horizons or min(requested_horizons) < 1:
        raise ValueError("forward horizons 必须是正整数")
    if not 1 <= minimum_factors_per_symbol <= len(names):
        raise ValueError("minimum_factors_per_symbol 超出所选因子数量")
    if not 0 <= winsor_quantile < 0.5:
        raise ValueError("winsor_quantile 必须在 [0, 0.5) 内")
    if maximum_stale_price_days < 0:
        raise ValueError("maximum_stale_price_days 不能为负")

    calendar = pd.DatetimeIndex(pd.to_datetime(market.calendar)).normalize()
    calendar = calendar.unique().sort_values()
    all_month_ends = _month_end_signal_dates(
        calendar,
        pd.Timestamp(start_date).normalize(),
        pd.Timestamp(end_date).normalize(),
        0,
    )
    signal_dates = _month_end_signal_dates(
        calendar,
        pd.Timestamp(start_date).normalize(),
        pd.Timestamp(end_date).normalize(),
        max(requested_horizons),
    )
    if not signal_dates:
        raise ValueError("研究区间不足以形成带前瞻收益标签的月末信号")

    prices = _total_return_prices(market, maximum_stale_price_days)
    calendar_locations = {date: index for index, date in enumerate(calendar)}
    next_signal_dates = {
        date: all_month_ends[index + 1]
        for index, date in enumerate(all_month_ends[:-1])
    }
    industries = _industry_lookup(market)
    records: list[pd.DataFrame] = []

    for signal_date in signal_dates:
        members = _members_at(market, signal_date)
        if not members:
            continue
        snapshot = pit.snapshot(
            signal_date,
            symbols=members,
            maximum_fundamental_age_days=maximum_fundamental_age_days,
            maximum_valuation_age_days=maximum_valuation_age_days,
        )
        frame = pd.DataFrame({"symbol": members})
        if not snapshot.empty:
            frame = frame.merge(snapshot, on="symbol", how="left")
        factor_values = compute_pit_factor_values(frame, names)
        for name in names:
            frame[name] = factor_values[name]
            frame[f"z_{name}"] = _cross_sectional_zscore(
                frame[name], winsor_quantile
            )

        visible = pit.visible_fundamentals(
            signal_date,
            symbols=members,
            maximum_age_days=maximum_fundamental_age_days,
        )
        if not visible.empty:
            latest_visible = (
                visible.sort_values(
                    [
                        "symbol",
                        "metric",
                        "period_end",
                        "available_date",
                        "revision_sequence",
                        "source_row_sha256",
                    ]
                )
                .groupby(["symbol", "metric"], as_index=False, sort=False)
                .tail(1)
            )
        else:
            latest_visible = visible
        for name in names:
            definition = _PIT_FACTOR_BY_NAME[name]
            if set(definition.input_columns).issubset(
                _VALUATION_INPUT_COLUMNS
            ):
                continue
            source_metric = definition.input_columns[0]
            lineage = latest_visible.loc[
                latest_visible["metric"].eq(source_metric),
                [
                    "symbol",
                    "period_end",
                    "available_date",
                    "revision_sequence",
                    "source_row_sha256",
                ],
            ].rename(
                columns={
                    "period_end": f"{name}_period_end",
                    "available_date": f"{name}_available_date",
                    "revision_sequence": f"{name}_revision_sequence",
                    "source_row_sha256": f"{name}_source_row_sha256",
                }
            )
            frame = frame.merge(lineage, on="symbol", how="left")

        z_columns = [f"z_{name}" for name in names]
        frame["available_factor_count"] = frame[z_columns].notna().sum(axis=1)
        composite = frame[z_columns].mean(axis=1, skipna=True).where(
            frame["available_factor_count"].ge(minimum_factors_per_symbol)
        )
        frame[f"z_{PIT_COMPOSITE_NAME}"] = _cross_sectional_zscore(
            composite, winsor_quantile
        )
        frame[PIT_COMPOSITE_NAME] = composite

        industry_values = [
            _industry_at(industries, symbol, signal_date)
            for symbol in frame["symbol"]
        ]
        frame["industry_code"] = [value[0] for value in industry_values]
        frame["industry_name"] = [value[1] for value in industry_values]
        market_value_source = (
            frame["total_market_value_10k_cny"]
            if "total_market_value_10k_cny" in frame
            else pd.Series(np.nan, index=frame.index, dtype=float)
        )
        market_values = pd.to_numeric(market_value_source, errors="coerce")
        frame["log_total_market_value"] = np.log(
            market_values.where(market_values > 0)
        )

        start_prices = prices.reindex(index=[signal_date], columns=members).iloc[0]
        location = calendar_locations[signal_date]
        for horizon in requested_horizons:
            outcome_date = calendar[location + horizon]
            end_prices = prices.reindex(index=[outcome_date], columns=members).iloc[
                0
            ]
            forward = end_prices / start_prices - 1.0
            frame[f"forward_return_{horizon}d"] = frame["symbol"].map(
                forward.to_dict()
            )
            frame[f"outcome_date_{horizon}d"] = outcome_date

        next_signal_date = next_signal_dates.get(signal_date)
        if next_signal_date is not None:
            next_prices = prices.reindex(
                index=[next_signal_date], columns=members
            ).iloc[0]
            next_returns = next_prices / start_prices - 1.0
            frame["forward_return_next_signal"] = frame["symbol"].map(
                next_returns.to_dict()
            )
            frame["next_signal_date"] = next_signal_date
        else:
            frame["forward_return_next_signal"] = np.nan
            frame["next_signal_date"] = pd.NaT

        frame["signal_date"] = signal_date
        frame["universe_size"] = len(members)
        records.append(frame)

    if not records:
        raise ValueError("研究区间没有可用的 PIT 因子截面")
    panel = pd.concat(records, ignore_index=True)
    leading = [
        "signal_date",
        "symbol",
        "industry_code",
        "industry_name",
        "universe_size",
        "available_factor_count",
        "log_total_market_value",
    ]
    remainder = [column for column in panel.columns if column not in leading]
    return panel[leading + remainder].sort_values(
        ["signal_date", "symbol"]
    ).reset_index(drop=True)


def _research_factor_names(factor_names: Iterable[str]) -> list[str]:
    return [*resolve_pit_factor_names(factor_names), PIT_COMPOSITE_NAME]


def calculate_factor_coverage(
    panel: pd.DataFrame,
    factor_names: Iterable[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for signal_date, group in panel.groupby("signal_date", sort=True):
        universe = int(group["universe_size"].max())
        for factor in _research_factor_names(factor_names):
            raw = pd.to_numeric(group.get(factor), errors="coerce")
            score = pd.to_numeric(group.get(f"z_{factor}"), errors="coerce")
            raw_valid = int(raw.notna().sum())
            valid = int(score.notna().sum())
            records.append(
                {
                    "signal_date": signal_date,
                    "factor": factor,
                    "universe_size": universe,
                    "raw_valid_observations": raw_valid,
                    "raw_coverage": (
                        raw_valid / universe if universe else np.nan
                    ),
                    "valid_observations": valid,
                    "coverage": valid / universe if universe else np.nan,
                }
            )
    return pd.DataFrame(records)


def calculate_factor_ic(
    panel: pd.DataFrame,
    factor_names: Iterable[str],
    horizons: Sequence[int],
    *,
    minimum_observations: int = 20,
) -> pd.DataFrame:
    if minimum_observations < 3:
        raise ValueError("minimum_observations 至少为 3")
    records: list[dict[str, Any]] = []
    factors = _research_factor_names(factor_names)
    for signal_date, group in panel.groupby("signal_date", sort=True):
        for factor in factors:
            score_column = f"z_{factor}"
            for horizon in horizons:
                return_column = f"forward_return_{int(horizon)}d"
                sample = group[[score_column, return_column]].dropna()
                ic = (
                    _spearman_correlation(
                        sample[score_column], sample[return_column]
                    )
                    if len(sample) >= minimum_observations
                    else np.nan
                )
                records.append(
                    {
                        "signal_date": signal_date,
                        "factor": factor,
                        "horizon_days": int(horizon),
                        "observations": len(sample),
                        "spearman_ic": ic,
                    }
                )
    return pd.DataFrame(records)


def summarize_factor_ic(ic: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (factor, horizon), group in ic.groupby(
        ["factor", "horizon_days"], sort=True
    ):
        values = pd.to_numeric(group["spearman_ic"], errors="coerce").dropna()
        count = len(values)
        mean = float(values.mean()) if count else np.nan
        deviation = float(values.std(ddof=1)) if count > 1 else np.nan
        records.append(
            {
                "factor": factor,
                "horizon_days": int(horizon),
                "valid_periods": count,
                "mean_ic": mean,
                "median_ic": float(values.median()) if count else np.nan,
                "ic_std": deviation,
                "ic_ir": (
                    mean / deviation
                    if count > 1 and math.isfinite(deviation) and deviation > 0
                    else np.nan
                ),
                "ic_t_stat": (
                    mean / (deviation / math.sqrt(count))
                    if count > 1 and math.isfinite(deviation) and deviation > 0
                    else np.nan
                ),
                "positive_ic_ratio": (
                    float(values.gt(0).mean()) if count else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def _quantile_labels(score: pd.Series, quantiles: int) -> pd.Series:
    output = pd.Series(pd.NA, index=score.index, dtype="Int64")
    valid = pd.to_numeric(score, errors="coerce").dropna()
    if len(valid) < quantiles * 2 or valid.nunique() < 2:
        return output
    ranks = valid.rank(method="first")
    labels = pd.qcut(ranks, quantiles, labels=False) + 1
    output.loc[valid.index] = labels.astype("Int64")
    return output


def calculate_quantile_returns(
    panel: pd.DataFrame,
    factor_names: Iterable[str],
    horizons: Sequence[int],
    *,
    quantiles: int = 5,
) -> pd.DataFrame:
    if quantiles < 2:
        raise ValueError("quantiles 至少为 2")
    records: list[dict[str, Any]] = []
    for signal_date, group in panel.groupby("signal_date", sort=True):
        for factor in _research_factor_names(factor_names):
            assignments = _quantile_labels(group[f"z_{factor}"], quantiles)
            for horizon in horizons:
                returns = pd.to_numeric(
                    group[f"forward_return_{int(horizon)}d"], errors="coerce"
                )
                for quantile in range(1, quantiles + 1):
                    values = returns.loc[assignments.eq(quantile)].dropna()
                    if values.empty:
                        continue
                    records.append(
                        {
                            "signal_date": signal_date,
                            "factor": factor,
                            "horizon_days": int(horizon),
                            "quantile": quantile,
                            "observations": len(values),
                            "mean_forward_return": float(values.mean()),
                            "median_forward_return": float(values.median()),
                        }
                    )
    return pd.DataFrame(records)


def summarize_quantile_returns(
    quantile_returns: pd.DataFrame,
    *,
    quantiles: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if quantile_returns.empty:
        return pd.DataFrame(
            columns=[
                "factor",
                "horizon_days",
                "valid_spread_periods",
                "bottom_quantile_mean_return",
                "top_quantile_mean_return",
                "top_minus_bottom_mean_return",
                "positive_spread_ratio",
            ]
        )
    for (factor, horizon), group in quantile_returns.groupby(
        ["factor", "horizon_days"], sort=True
    ):
        pivot = group.pivot(
            index="signal_date",
            columns="quantile",
            values="mean_forward_return",
        )
        if 1 not in pivot or quantiles not in pivot:
            spreads = pd.Series(dtype=float)
        else:
            spreads = (pivot[quantiles] - pivot[1]).dropna()
        records.append(
            {
                "factor": factor,
                "horizon_days": int(horizon),
                "valid_spread_periods": len(spreads),
                "bottom_quantile_mean_return": (
                    float(pivot[1].mean()) if 1 in pivot else np.nan
                ),
                "top_quantile_mean_return": (
                    float(pivot[quantiles].mean())
                    if quantiles in pivot
                    else np.nan
                ),
                "top_minus_bottom_mean_return": (
                    float(spreads.mean()) if len(spreads) else np.nan
                ),
                "positive_spread_ratio": (
                    float(spreads.gt(0).mean()) if len(spreads) else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def _industry_r_squared(score: pd.Series, industries: pd.Series) -> float:
    sample = pd.DataFrame({"score": score, "industry": industries}).dropna()
    sample = sample.loc[sample["industry"].astype(str).ne("UNKNOWN")]
    if len(sample) < 3 or sample["industry"].nunique() < 2:
        return np.nan
    values = sample["score"].to_numpy(dtype=float)
    total = float(np.square(values - values.mean()).sum())
    if total < 1e-12:
        return np.nan
    fitted = sample.groupby("industry")["score"].transform("mean")
    residual = float(np.square(values - fitted.to_numpy(dtype=float)).sum())
    return max(0.0, min(1.0, 1.0 - residual / total))


def calculate_factor_exposures(
    panel: pd.DataFrame,
    factor_names: Iterable[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for signal_date, group in panel.groupby("signal_date", sort=True):
        for factor in _research_factor_names(factor_names):
            score = pd.to_numeric(group[f"z_{factor}"], errors="coerce")
            size = pd.to_numeric(
                group["log_total_market_value"], errors="coerce"
            )
            valid_size = pd.DataFrame({"score": score, "size": size}).dropna()
            industry_means = pd.DataFrame(
                {"score": score, "industry": group["industry_code"]}
            ).dropna()
            industry_means = industry_means.loc[
                industry_means["industry"].astype(str).ne("UNKNOWN")
            ]
            grouped_means = industry_means.groupby("industry")["score"].mean()
            records.append(
                {
                    "signal_date": signal_date,
                    "factor": factor,
                    "observations": int(score.notna().sum()),
                    "industries": int(industry_means["industry"].nunique()),
                    "size_spearman": (
                        _spearman_correlation(
                            valid_size["score"], valid_size["size"]
                        )
                        if len(valid_size) >= 3
                        else np.nan
                    ),
                    "industry_r_squared": _industry_r_squared(
                        score, group["industry_code"]
                    ),
                    "maximum_absolute_industry_mean_z": (
                        float(grouped_means.abs().max())
                        if not grouped_means.empty
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def summarize_factor_exposures(exposures: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for factor, group in exposures.groupby("factor", sort=True):
        size = pd.to_numeric(group["size_spearman"], errors="coerce")
        industry = pd.to_numeric(
            group["industry_r_squared"], errors="coerce"
        )
        records.append(
            {
                "factor": factor,
                "periods": len(group),
                "median_size_spearman": float(size.median()),
                "median_absolute_size_spearman": float(size.abs().median()),
                "maximum_absolute_size_spearman": float(size.abs().max()),
                "median_industry_r_squared": float(industry.median()),
                "maximum_industry_r_squared": float(industry.max()),
            }
        )
    return pd.DataFrame(records)


def _score_evidence(
    panel: pd.DataFrame,
    score: pd.Series,
    horizons: Sequence[int],
    *,
    quantiles: int,
    minimum_observations: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    working = panel[["signal_date", *[f"forward_return_{int(value)}d" for value in horizons]]].copy()
    working["score"] = score
    for horizon in horizons:
        ics: list[float] = []
        spreads: list[float] = []
        return_column = f"forward_return_{int(horizon)}d"
        for _, group in working.groupby("signal_date", sort=True):
            sample = group[["score", return_column]].dropna()
            if len(sample) >= minimum_observations:
                ics.append(
                    _spearman_correlation(
                        sample["score"], sample[return_column]
                    )
                )
            labels = _quantile_labels(sample["score"], quantiles)
            bottom = sample.loc[labels.eq(1), return_column]
            top = sample.loc[labels.eq(quantiles), return_column]
            if not bottom.empty and not top.empty:
                spreads.append(float(top.mean() - bottom.mean()))
        ic_values = pd.Series(ics, dtype=float).dropna()
        spread_values = pd.Series(spreads, dtype=float).dropna()
        records.append(
            {
                "horizon_days": int(horizon),
                "valid_ic_periods": len(ic_values),
                "mean_ic": float(ic_values.mean()) if len(ic_values) else np.nan,
                "positive_ic_ratio": (
                    float(ic_values.gt(0).mean()) if len(ic_values) else np.nan
                ),
                "valid_spread_periods": len(spread_values),
                "top_minus_bottom_mean_return": (
                    float(spread_values.mean())
                    if len(spread_values)
                    else np.nan
                ),
            }
        )
    return records


def run_pit_factor_ablation(
    panel: pd.DataFrame,
    factor_names: Iterable[str],
    horizons: Sequence[int],
    *,
    quantiles: int = 5,
    minimum_observations: int = 20,
) -> pd.DataFrame:
    names = resolve_pit_factor_names(factor_names)
    cases: list[tuple[str, str, list[str]]] = [("full", "", names)]
    cases.extend(
        (f"without_{removed}", removed, [name for name in names if name != removed])
        for removed in names
    )
    records: list[dict[str, Any]] = []
    for variant, removed, included in cases:
        minimum_available = max(2, math.ceil(len(included) / 2))
        values = panel[[f"z_{name}" for name in included]]
        score = values.mean(axis=1, skipna=True).where(
            values.notna().sum(axis=1).ge(minimum_available)
        )
        for evidence in _score_evidence(
            panel,
            score,
            horizons,
            quantiles=quantiles,
            minimum_observations=minimum_observations,
        ):
            records.append(
                {
                    "variant": variant,
                    "removed_factor": removed,
                    "included_factor_count": len(included),
                    "minimum_available_factor_count": minimum_available,
                    **evidence,
                }
            )
    return pd.DataFrame(records)


def run_pit_factor_rolling_validation(
    panel: pd.DataFrame,
    horizons: Sequence[int],
    *,
    train_years: int = 5,
    test_years: int = 1,
    quantiles: int = 5,
    minimum_observations: int = 20,
) -> pd.DataFrame:
    if train_years < 1 or test_years < 1:
        raise ValueError("train_years 和 test_years 必须至少为 1")
    dates = pd.DatetimeIndex(panel["signal_date"].drop_duplicates()).sort_values()
    overall_start = dates.min()
    overall_end = dates.max()
    test_start = overall_start + pd.DateOffset(years=train_years)
    if test_start > overall_end:
        raise ValueError("研究面板不足以形成滚动验证窗口")
    records: list[dict[str, Any]] = []
    window = 1
    while test_start <= overall_end:
        test_end = min(
            test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1),
            overall_end,
        )
        subset = panel.loc[
            panel["signal_date"].between(test_start, test_end)
        ].copy()
        score = subset[f"z_{PIT_COMPOSITE_NAME}"]
        for evidence in _score_evidence(
            subset,
            score,
            horizons,
            quantiles=quantiles,
            minimum_observations=minimum_observations,
        ):
            records.append(
                {
                    "window": window,
                    "train_start": overall_start,
                    "train_end": test_start - pd.Timedelta(days=1),
                    "test_start": test_start,
                    "test_end": test_end,
                    "signal_dates": int(subset["signal_date"].nunique()),
                    "parameters_fitted": False,
                    **evidence,
                }
            )
        test_start = test_end + pd.Timedelta(days=1)
        window += 1
    return pd.DataFrame(records)


def run_pit_factor_cost_stress(
    panel: pd.DataFrame,
    factor_names: Iterable[str],
    *,
    horizon: int,
    quantiles: int = 5,
    cost_bps: Sequence[float] = (5.0, 10.0, 20.0),
) -> pd.DataFrame:
    costs = sorted({float(value) for value in cost_bps})
    if not costs or min(costs) < 0:
        raise ValueError("cost_bps 必须是非负数")
    use_next_signal = "forward_return_next_signal" in panel
    return_column = (
        "forward_return_next_signal"
        if use_next_signal
        else f"forward_return_{int(horizon)}d"
    )
    return_label = (
        "next_signal_date" if use_next_signal else f"fixed_{int(horizon)}d"
    )
    records: list[dict[str, Any]] = []
    for factor in _research_factor_names(factor_names):
        previous: dict[str, float] = {}
        for signal_date, group in panel.groupby("signal_date", sort=True):
            ranked = group[["symbol", f"z_{factor}", return_column]].dropna(
                subset=[f"z_{factor}"]
            )
            labels = _quantile_labels(ranked[f"z_{factor}"], quantiles)
            selected = ranked.loc[labels.eq(quantiles)].copy()
            weights = {
                str(symbol): 1.0 / len(selected)
                for symbol in selected["symbol"]
            } if not selected.empty else {}
            union = set(previous).union(weights)
            two_sided_turnover = sum(
                abs(weights.get(symbol, 0.0) - previous.get(symbol, 0.0))
                for symbol in union
            )
            realized = pd.to_numeric(
                selected[return_column], errors="coerce"
            ).dropna()
            outcome_coverage = (
                len(realized) / len(selected) if len(selected) else 0.0
            )
            gross_return = (
                float(realized.mean()) if len(realized) else np.nan
            )
            for bps in costs:
                estimated_cost = two_sided_turnover * bps / 10_000.0
                records.append(
                    {
                        "signal_date": signal_date,
                        "factor": factor,
                        "horizon_days": int(horizon),
                        "return_label": return_label,
                        "cost_bps_per_traded_side": bps,
                        "holdings": len(selected),
                        "realized_outcomes": len(realized),
                        "outcome_coverage": outcome_coverage,
                        "two_sided_weight_turnover": two_sided_turnover,
                        "gross_return": gross_return,
                        "estimated_cost": estimated_cost,
                        "net_return": (
                            gross_return - estimated_cost
                            if math.isfinite(gross_return)
                            else np.nan
                        ),
                    }
                )
            previous = weights
    return pd.DataFrame(records)


def _compound_metrics(values: pd.Series) -> tuple[float, float, float]:
    returns = pd.to_numeric(values, errors="coerce").dropna()
    if returns.empty or returns.le(-1).any():
        return np.nan, np.nan, np.nan
    curve = (1.0 + returns).cumprod()
    cumulative = float(curve.iloc[-1] - 1.0)
    annualized = float(curve.iloc[-1] ** (12.0 / len(curve)) - 1.0)
    drawdown = curve / curve.cummax() - 1.0
    return cumulative, annualized, float(drawdown.min())


def summarize_cost_stress(cost_stress: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (factor, horizon, return_label, cost), group in cost_stress.groupby(
        [
            "factor",
            "horizon_days",
            "return_label",
            "cost_bps_per_traded_side",
        ],
        sort=True,
    ):
        gross_cumulative, gross_annualized, _ = _compound_metrics(
            group.sort_values("signal_date")["gross_return"]
        )
        net_cumulative, net_annualized, net_drawdown = _compound_metrics(
            group.sort_values("signal_date")["net_return"]
        )
        records.append(
            {
                "factor": factor,
                "horizon_days": int(horizon),
                "return_label": return_label,
                "cost_bps_per_traded_side": float(cost),
                "periods": len(group),
                "average_outcome_coverage": float(
                    group["outcome_coverage"].mean()
                ),
                "minimum_outcome_coverage": float(
                    group["outcome_coverage"].min()
                ),
                "average_two_sided_weight_turnover": float(
                    group["two_sided_weight_turnover"].mean()
                ),
                "gross_cumulative_return": gross_cumulative,
                "gross_annualized_return": gross_annualized,
                "net_cumulative_return": net_cumulative,
                "net_annualized_return": net_annualized,
                "net_maximum_drawdown": net_drawdown,
            }
        )
    return pd.DataFrame(records)


def build_pit_factor_governance(
    *,
    provider: str,
    primary_horizon: int,
    maximum_cost_bps: float,
    coverage: pd.DataFrame,
    ic_summary: pd.DataFrame,
    quantile_summary: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    rolling: pd.DataFrame,
    cost_summary: pd.DataFrame,
) -> dict[str, Any]:
    composite_coverage = coverage.loc[
        coverage["factor"].eq(PIT_COMPOSITE_NAME), "coverage"
    ]
    ic_row = ic_summary.loc[
        ic_summary["factor"].eq(PIT_COMPOSITE_NAME)
        & ic_summary["horizon_days"].eq(primary_horizon)
    ]
    quantile_row = quantile_summary.loc[
        quantile_summary["factor"].eq(PIT_COMPOSITE_NAME)
        & quantile_summary["horizon_days"].eq(primary_horizon)
    ]
    exposure_row = exposure_summary.loc[
        exposure_summary["factor"].eq(PIT_COMPOSITE_NAME)
    ]
    rolling_rows = rolling.loc[rolling["horizon_days"].eq(primary_horizon)]
    cost_row = cost_summary.loc[
        cost_summary["factor"].eq(PIT_COMPOSITE_NAME)
        & cost_summary["horizon_days"].eq(primary_horizon)
        & cost_summary["cost_bps_per_traded_side"].eq(maximum_cost_bps)
    ]

    def first(frame: pd.DataFrame, column: str) -> float:
        if frame.empty:
            return np.nan
        return float(frame.iloc[0][column])

    evidence = {
        "signal_dates": int(len(composite_coverage)),
        "median_coverage": (
            float(composite_coverage.median())
            if not composite_coverage.empty
            else np.nan
        ),
        "mean_ic": first(ic_row, "mean_ic"),
        "positive_ic_ratio": first(ic_row, "positive_ic_ratio"),
        "top_minus_bottom_mean_return": first(
            quantile_row, "top_minus_bottom_mean_return"
        ),
        "positive_rolling_window_ratio": (
            float(rolling_rows["mean_ic"].gt(0).mean())
            if not rolling_rows.empty
            else np.nan
        ),
        "median_absolute_size_spearman": first(
            exposure_row, "median_absolute_size_spearman"
        ),
        "median_industry_r_squared": first(
            exposure_row, "median_industry_r_squared"
        ),
        "maximum_cost_net_annualized_return": first(
            cost_row, "net_annualized_return"
        ),
        "minimum_cost_outcome_coverage": first(
            cost_row, "minimum_outcome_coverage"
        ),
    }
    thresholds = {
        "signal_dates": 24,
        "median_coverage": 0.60,
        "mean_ic": 0.02,
        "positive_ic_ratio": 0.55,
        "top_minus_bottom_mean_return": 0.0,
        "positive_rolling_window_ratio": 0.60,
        "median_absolute_size_spearman_max": 0.30,
        "median_industry_r_squared_max": 0.30,
        "maximum_cost_net_annualized_return": 0.0,
        "minimum_cost_outcome_coverage": 0.90,
    }
    checks = [
        ("signal_dates", evidence["signal_dates"] >= thresholds["signal_dates"]),
        (
            "median_coverage",
            evidence["median_coverage"] >= thresholds["median_coverage"],
        ),
        ("mean_ic", evidence["mean_ic"] >= thresholds["mean_ic"]),
        (
            "positive_ic_ratio",
            evidence["positive_ic_ratio"] >= thresholds["positive_ic_ratio"],
        ),
        (
            "top_minus_bottom_mean_return",
            evidence["top_minus_bottom_mean_return"]
            > thresholds["top_minus_bottom_mean_return"],
        ),
        (
            "positive_rolling_window_ratio",
            evidence["positive_rolling_window_ratio"]
            >= thresholds["positive_rolling_window_ratio"],
        ),
        (
            "median_absolute_size_spearman_max",
            evidence["median_absolute_size_spearman"]
            <= thresholds["median_absolute_size_spearman_max"],
        ),
        (
            "median_industry_r_squared_max",
            evidence["median_industry_r_squared"]
            <= thresholds["median_industry_r_squared_max"],
        ),
        (
            "maximum_cost_net_annualized_return",
            evidence["maximum_cost_net_annualized_return"]
            > thresholds["maximum_cost_net_annualized_return"],
        ),
        (
            "minimum_cost_outcome_coverage",
            evidence["minimum_cost_outcome_coverage"]
            >= thresholds["minimum_cost_outcome_coverage"],
        ),
    ]
    serialized_checks = [
        {"name": name, "passed": bool(passed)} for name, passed in checks
    ]
    all_passed = all(item["passed"] for item in serialized_checks)
    if provider != "tushare":
        decision = "nonproduction_data_only"
    elif all_passed:
        decision = "eligible_for_manual_candidate_review"
    else:
        decision = "insufficient_consistent_evidence"
    return {
        "schema_version": 1,
        "research_version": PIT_FACTOR_RESEARCH_VERSION,
        "candidate": PIT_COMPOSITE_NAME,
        "lifecycle_status": "research_only",
        "promotion_decision": decision,
        "provider": provider,
        "primary_horizon_days": primary_horizon,
        "maximum_cost_bps_per_traded_side": maximum_cost_bps,
        "evidence": evidence,
        "thresholds": thresholds,
        "checks": serialized_checks,
        "all_quantitative_checks_passed": all_passed,
        "automatic_parameter_fitting": False,
        "automatic_production_promotion": False,
        "production_default_changed": False,
        "untouched_holdout_certified": False,
        "note": (
            "阈值只决定是否允许进入人工候选复核；历史数据即使全部达标，"
            "也不能自动进入 MultiFactorStrategy 或宣称达到收益目标。"
        ),
    }


def _atomic_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix == ".gz":
        frame.to_csv(
            temporary,
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return path


def write_pit_factor_research(
    market: MarketDataBundle,
    pit: PointInTimeDataBundle,
    config: AppConfig,
    output_dir: str | Path,
    *,
    factor_names: Iterable[str] | None = None,
    horizons: Sequence[int] = (21, 63),
    quantiles: int = 5,
    minimum_factors_per_symbol: int = 4,
    minimum_ic_observations: int = 20,
    cost_bps: Sequence[float] = (5.0, 10.0, 20.0),
    train_years: int = 5,
    test_years: int = 1,
) -> dict[str, Path]:
    names = resolve_pit_factor_names(factor_names)
    requested_horizons = sorted({int(value) for value in horizons})
    costs = sorted({float(value) for value in cost_bps})
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"PIT 研究输出目录非空，拒绝覆盖或混入旧产物: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    base_manifest_path = Path(config.data.cache_dir).resolve() / "manifest.json"
    pit_manifest_path = (
        Path(config.point_in_time.cache_dir).resolve() / "manifest.json"
    )
    if not base_manifest_path.is_file() or not pit_manifest_path.is_file():
        raise FileNotFoundError("PIT 研究必须绑定基础行情与 PIT 两份 manifest")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    pit_manifest = json.loads(pit_manifest_path.read_text(encoding="utf-8"))
    if pit.manifest:
        expected = pit.manifest.get("data_fingerprint_sha256")
        actual = pit_manifest.get("data_fingerprint_sha256")
        if expected != actual:
            raise ValueError("内存 PIT 数据与磁盘 manifest 指纹不一致")
    if (
        pit_manifest.get("base_data_fingerprint_sha256")
        != base_manifest.get("data_fingerprint_sha256")
    ):
        raise ValueError("PIT manifest 未绑定当前基础行情数据指纹")

    panel = build_pit_factor_panel(
        market,
        pit,
        start_date=config.backtest.start_date,
        end_date=config.backtest.end_date,
        factor_names=names,
        horizons=requested_horizons,
        minimum_factors_per_symbol=minimum_factors_per_symbol,
        winsor_quantile=config.strategy.winsor_quantile,
        maximum_fundamental_age_days=(
            config.point_in_time.maximum_fundamental_age_days
        ),
        maximum_valuation_age_days=(
            config.point_in_time.maximum_valuation_age_days
        ),
        maximum_stale_price_days=config.backtest.maximum_stale_trading_days,
    )
    registry = pit_factor_registry_frame(names)
    coverage = calculate_factor_coverage(panel, names)
    ic = calculate_factor_ic(
        panel,
        names,
        requested_horizons,
        minimum_observations=minimum_ic_observations,
    )
    ic_summary = summarize_factor_ic(ic)
    quantile_returns = calculate_quantile_returns(
        panel, names, requested_horizons, quantiles=quantiles
    )
    quantile_summary = summarize_quantile_returns(
        quantile_returns, quantiles=quantiles
    )
    exposures = calculate_factor_exposures(panel, names)
    exposure_summary = summarize_factor_exposures(exposures)
    ablation = run_pit_factor_ablation(
        panel,
        names,
        requested_horizons,
        quantiles=quantiles,
        minimum_observations=minimum_ic_observations,
    )
    rolling = run_pit_factor_rolling_validation(
        panel,
        requested_horizons,
        train_years=train_years,
        test_years=test_years,
        quantiles=quantiles,
        minimum_observations=minimum_ic_observations,
    )
    primary_horizon = min(requested_horizons)
    cost_stress = run_pit_factor_cost_stress(
        panel,
        names,
        horizon=primary_horizon,
        quantiles=quantiles,
        cost_bps=costs,
    )
    cost_summary = summarize_cost_stress(cost_stress)
    governance = build_pit_factor_governance(
        provider=str(pit_manifest.get("provider", "unknown")),
        primary_horizon=primary_horizon,
        maximum_cost_bps=max(costs),
        coverage=coverage,
        ic_summary=ic_summary,
        quantile_summary=quantile_summary,
        exposure_summary=exposure_summary,
        rolling=rolling,
        cost_summary=cost_summary,
    )

    frames = {
        "registry": (registry, "pit_factor_registry.csv"),
        "panel": (panel, "pit_factor_panel.csv.gz"),
        "coverage": (coverage, "pit_factor_coverage.csv"),
        "ic": (ic, "pit_factor_ic.csv"),
        "ic_summary": (ic_summary, "pit_factor_ic_summary.csv"),
        "quantiles": (quantile_returns, "pit_factor_quantile_returns.csv"),
        "quantile_summary": (
            quantile_summary,
            "pit_factor_quantile_summary.csv",
        ),
        "exposures": (exposures, "pit_factor_exposures.csv"),
        "exposure_summary": (
            exposure_summary,
            "pit_factor_exposure_summary.csv",
        ),
        "ablation": (ablation, "pit_factor_ablation.csv"),
        "rolling": (rolling, "pit_factor_rolling.csv"),
        "cost_stress": (cost_stress, "pit_factor_cost_stress.csv"),
        "cost_summary": (cost_summary, "pit_factor_cost_summary.csv"),
    }
    written: dict[str, Path] = {}
    for name, (frame, filename) in frames.items():
        written[name] = _atomic_dataframe(frame, output / filename)

    governance_path = write_json_atomic(
        governance, output / "pit_factor_governance.json"
    )
    written["governance"] = governance_path
    research_parameters = {
        "factor_names": names,
        "composite": PIT_COMPOSITE_NAME,
        "horizons": requested_horizons,
        "quantiles": quantiles,
        "minimum_factors_per_symbol": minimum_factors_per_symbol,
        "minimum_ic_observations": minimum_ic_observations,
        "cost_bps": costs,
        "train_years": train_years,
        "test_years": test_years,
        "automatic_parameter_fitting": False,
    }
    reproducibility = build_reproducibility_manifest(
        {
            "app_config": config.to_dict(),
            "pit_factor_research": research_parameters,
        },
        data_manifest_path=base_manifest_path,
        extra_input_files=[pit_manifest_path],
    )
    reproducibility["pit_data"] = {
        "manifest_path": str(pit_manifest_path),
        "manifest_sha256": sha256_file(pit_manifest_path),
        "schema_version": pit_manifest.get("schema_version"),
        "data_fingerprint_sha256": pit_manifest.get(
            "data_fingerprint_sha256"
        ),
        "base_data_fingerprint_sha256": pit_manifest.get(
            "base_data_fingerprint_sha256"
        ),
    }
    reproducibility_path = write_json_atomic(
        reproducibility, output / "reproducibility.json"
    )
    written["reproducibility"] = reproducibility_path
    manifest = {
        "schema_version": 1,
        "research_version": PIT_FACTOR_RESEARCH_VERSION,
        "parameters": research_parameters,
        "production_strategy_changed": False,
        "promotion_decision": governance["promotion_decision"],
        "base_data_fingerprint_sha256": base_manifest.get(
            "data_fingerprint_sha256"
        ),
        "pit_data_fingerprint_sha256": pit_manifest.get(
            "data_fingerprint_sha256"
        ),
        "run_fingerprint_sha256": reproducibility["run_fingerprint_sha256"],
        "files": {name: path.name for name, path in written.items()},
        "limitations": [
            "历史区间不是未查看保留期。",
            "研究组合固定等权，不执行参数寻优。",
            "成本压力是月度等权目标的权重级估算，不替代严格成交回测。",
            "任何结果都不会自动修改 MultiFactorStrategy。",
        ],
    }
    manifest_path = write_json_atomic(
        manifest, output / "pit_factor_research_manifest.json"
    )
    written["manifest"] = manifest_path
    registry_path = record_experiment(
        output / "experiment_registry.jsonl",
        reproducibility,
        experiment_type="pit_factor_research_v2_alpha2",
        protocol={
            **research_parameters,
            "pit_data_fingerprint_sha256": pit_manifest.get(
                "data_fingerprint_sha256"
            ),
            "automatic_production_promotion": False,
        },
        artifacts=written.values(),
    )
    written["experiment_registry"] = registry_path
    artifact_manifest = write_artifact_manifest(output, written.values())
    written["artifacts"] = artifact_manifest
    return written
