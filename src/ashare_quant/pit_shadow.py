from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .backtest import BacktestResult, Backtester
from .config import AppConfig
from .data import MarketDataBundle
from .factors import MultiFactorStrategy, _winsorized_zscore
from .pit_data import PointInTimeDataBundle, require_pit_research_eligible
from .pit_acceptance import require_pit_acceptance
from .pit_research import (
    PIT_COMPOSITE_NAME,
    PIT_FACTOR_RESEARCH_VERSION,
    compute_pit_composite_scores,
    resolve_pit_factor_names,
)
from .provenance import (
    build_reproducibility_manifest,
    record_experiment,
    sha256_file,
    verify_artifact_manifest,
    write_artifact_manifest,
    write_json_atomic,
)
from .report import calculate_metrics


PIT_SHADOW_RESEARCH_VERSION = "pit_shadow_attribution_v2_alpha3"
PIT_SHADOW_CANDIDATE = "price_fundamental_hybrid_v2_alpha3"
PIT_SHADOW_BLEND_WEIGHT = 0.25
PIT_SHADOW_MINIMUM_FACTORS = 4

PRODUCTION_BASELINE_ARM = "production_baseline"
COVERAGE_MATCHED_PRICE_ARM = "coverage_matched_price"
PIT_ONLY_ARM = "pit_only_shadow"
HYBRID_ARM = "hybrid_fixed_25pct_pit"
PIT_SHADOW_ARMS = (
    PRODUCTION_BASELINE_ARM,
    COVERAGE_MATCHED_PRICE_ARM,
    PIT_ONLY_ARM,
    HYBRID_ARM,
)
_RESEARCH_ARMS = {
    COVERAGE_MATCHED_PRICE_ARM,
    PIT_ONLY_ARM,
    HYBRID_ARM,
}


def _signal_dates(
    calendar: pd.DatetimeIndex,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> list[pd.Timestamp]:
    dates = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize()
    dates = dates.unique().sort_values()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    eligible = dates[(dates >= start) & (dates <= end)]
    if eligible.empty:
        return []
    month_ends = (
        pd.Series(eligible, index=eligible)
        .groupby(eligible.to_period("M"))
        .max()
        .tolist()
    )
    output: list[pd.Timestamp] = []
    for value in month_ends:
        signal_date = pd.Timestamp(value)
        next_location = dates.searchsorted(signal_date, side="right")
        if next_location < len(dates) and dates[next_location] <= end:
            output.append(signal_date)
    return output


def _members_at(market: MarketDataBundle, signal_date: pd.Timestamp) -> list[str]:
    eligible = market.membership.loc[market.membership["date"].le(signal_date), "date"]
    if eligible.empty:
        return []
    snapshot_date = eligible.max()
    return sorted(
        market.membership.loc[market.membership["date"].eq(snapshot_date), "symbol"]
        .astype(str)
        .unique()
        .tolist()
    )


def build_pit_shadow_score_panel(
    market: MarketDataBundle,
    pit: PointInTimeDataBundle,
    config: AppConfig,
    *,
    factor_names: Iterable[str] | None = None,
    minimum_factors_per_symbol: int = PIT_SHADOW_MINIMUM_FACTORS,
) -> pd.DataFrame:
    """Build score-only monthly PIT snapshots for strict shadow execution.

    No forward-return field is accepted or produced by this function.  Outcome
    labels remain confined to the Alpha2 research bundle.
    """
    names = resolve_pit_factor_names(factor_names)
    records: list[pd.DataFrame] = []
    for signal_date in _signal_dates(
        market.calendar,
        config.backtest.start_date,
        config.backtest.end_date,
    ):
        members = _members_at(market, signal_date)
        if not members:
            continue
        snapshot = pit.snapshot(
            signal_date,
            symbols=members,
            maximum_fundamental_age_days=(
                config.point_in_time.maximum_fundamental_age_days
            ),
            maximum_valuation_age_days=(
                config.point_in_time.maximum_valuation_age_days
            ),
        )
        frame = pd.DataFrame({"symbol": members})
        if not snapshot.empty:
            frame = frame.merge(snapshot, on="symbol", how="left")
        scores = compute_pit_composite_scores(
            frame,
            names,
            minimum_factors_per_symbol=minimum_factors_per_symbol,
            winsor_quantile=config.strategy.winsor_quantile,
        )

        visible = pit.visible_fundamentals(
            signal_date,
            symbols=members,
            maximum_age_days=config.point_in_time.maximum_fundamental_age_days,
        )
        if visible.empty:
            lineage = pd.DataFrame(
                columns=[
                    "symbol",
                    "latest_fundamental_period_end",
                    "latest_fundamental_available_date",
                    "visible_fundamental_metric_count",
                ]
            )
        else:
            latest = (
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
            lineage = latest.groupby("symbol", as_index=False).agg(
                latest_fundamental_period_end=("period_end", "max"),
                latest_fundamental_available_date=(
                    "available_date",
                    "max",
                ),
                visible_fundamental_metric_count=("metric", "nunique"),
            )

        scores = scores.merge(lineage, on="symbol", how="left")
        for column in ["valuation_date", "valuation_available_date"]:
            if column in frame:
                scores[column] = scores["symbol"].map(frame.set_index("symbol")[column])
            else:
                scores[column] = pd.NaT
        scores["signal_date"] = signal_date
        scores["as_of_date"] = signal_date
        scores["universe_size"] = len(members)
        records.append(scores)

    if not records:
        raise ValueError("回测区间没有可执行的 PIT 影子信号日")
    panel = pd.concat(records, ignore_index=True)
    leading = [
        "signal_date",
        "as_of_date",
        "symbol",
        "universe_size",
        "available_factor_count",
        PIT_COMPOSITE_NAME,
        f"z_{PIT_COMPOSITE_NAME}",
        "latest_fundamental_period_end",
        "latest_fundamental_available_date",
        "visible_fundamental_metric_count",
        "valuation_date",
        "valuation_available_date",
    ]
    remainder = [column for column in panel.columns if column not in leading]
    return (
        panel[leading + remainder]
        .sort_values(["signal_date", "symbol"])
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class PITShadowScorePanel:
    frame: pd.DataFrame
    minimum_factors_per_symbol: int = PIT_SHADOW_MINIMUM_FACTORS

    def __post_init__(self) -> None:
        forbidden = sorted(
            column
            for column in self.frame.columns
            if column.startswith("forward_return")
            or column.startswith("outcome_date")
            or column == "next_signal_date"
        )
        if forbidden:
            raise ValueError(
                "PIT 影子执行分数禁止携带未来收益标签: " + ", ".join(forbidden)
            )
        required = {
            "signal_date",
            "as_of_date",
            "symbol",
            "universe_size",
            "available_factor_count",
            f"z_{PIT_COMPOSITE_NAME}",
        }
        missing = sorted(required.difference(self.frame.columns))
        if missing:
            raise ValueError("PIT 影子分数缺少字段: " + ", ".join(missing))
        if self.minimum_factors_per_symbol < 1:
            raise ValueError("minimum_factors_per_symbol 必须大于 0")
        prepared = self.frame.copy()
        prepared["signal_date"] = pd.to_datetime(
            prepared["signal_date"], errors="coerce"
        ).dt.normalize()
        prepared["symbol"] = prepared["symbol"].astype(str)
        prepared["available_factor_count"] = pd.to_numeric(
            prepared["available_factor_count"], errors="coerce"
        )
        prepared[f"z_{PIT_COMPOSITE_NAME}"] = pd.to_numeric(
            prepared[f"z_{PIT_COMPOSITE_NAME}"], errors="coerce"
        )
        prepared["universe_size"] = pd.to_numeric(
            prepared["universe_size"], errors="coerce"
        )
        if prepared["signal_date"].isna().any():
            raise ValueError("PIT 影子分数包含无效信号日")
        if prepared.duplicated(["signal_date", "symbol"]).any():
            raise ValueError("PIT 影子分数包含重复信号日证券")
        if not prepared["symbol"].str.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)").all():
            raise ValueError("PIT 影子分数包含无效证券代码")
        counts = prepared["available_factor_count"]
        if (
            counts.isna().any()
            or counts.lt(0).any()
            or counts.gt(len(resolve_pit_factor_names())).any()
            or counts.ne(np.floor(counts)).any()
        ):
            raise ValueError("available_factor_count 必须是有效的非负整数")
        scores = prepared[f"z_{PIT_COMPOSITE_NAME}"]
        if (~scores.isna() & ~np.isfinite(scores)).any():
            raise ValueError("PIT 影子分数包含非有限数值")
        universe = prepared["universe_size"]
        if (
            universe.isna().any()
            or universe.lt(1).any()
            or universe.ne(np.floor(universe)).any()
        ):
            raise ValueError("universe_size 必须是正整数")
        for signal_date, group in prepared.groupby("signal_date", sort=False):
            sizes = group["universe_size"].unique()
            if len(sizes) != 1 or int(sizes[0]) != len(group):
                raise ValueError(
                    f"{signal_date.date()} 的 universe_size 与分数行数不一致"
                )
        for column in [
            "as_of_date",
            "latest_fundamental_period_end",
            "latest_fundamental_available_date",
            "valuation_date",
            "valuation_available_date",
        ]:
            if column not in prepared:
                continue
            original = prepared[column]
            values = pd.to_datetime(original, errors="coerce")
            if (original.notna() & values.isna()).any():
                raise ValueError(f"{column} 包含无效日期")
            visible = values.notna()
            if values.loc[visible].gt(prepared.loc[visible, "signal_date"]).any():
                raise ValueError(f"{column} 晚于信号日，拒绝未来信息")
            prepared[column] = values.dt.normalize()
        if not prepared["as_of_date"].eq(prepared["signal_date"]).all():
            raise ValueError("as_of_date 必须与 signal_date 完全一致")
        object.__setattr__(
            self,
            "frame",
            prepared.sort_values(["signal_date", "symbol"]).reset_index(drop=True),
        )

    def at(self, signal_date: pd.Timestamp | str) -> pd.DataFrame:
        date = pd.Timestamp(signal_date).normalize()
        return self.frame.loc[self.frame["signal_date"].eq(date)].copy()

    def coverage(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        score_column = f"z_{PIT_COMPOSITE_NAME}"
        for signal_date, group in self.frame.groupby("signal_date", sort=True):
            eligible = group[score_column].notna() & group["available_factor_count"].ge(
                self.minimum_factors_per_symbol
            )
            universe = (
                int(group["universe_size"].max())
                if "universe_size" in group
                else len(group)
            )
            records.append(
                {
                    "signal_date": signal_date,
                    "universe_size": universe,
                    "score_rows": len(group),
                    "eligible_symbols": int(eligible.sum()),
                    "eligible_coverage": (
                        float(eligible.sum() / universe) if universe else np.nan
                    ),
                }
            )
        return pd.DataFrame(records)


class PITShadowStrategy(MultiFactorStrategy):
    """Research-only score overlay; it is not selectable from AppConfig."""

    def __init__(
        self,
        market: MarketDataBundle,
        config: AppConfig,
        score_panel: PITShadowScorePanel,
        *,
        arm: str,
    ) -> None:
        if arm not in _RESEARCH_ARMS:
            raise ValueError(f"未知 PIT 影子臂: {arm}")
        self.score_panel = score_panel
        self.shadow_arm = arm
        self.blend_weight = PIT_SHADOW_BLEND_WEIGHT
        super().__init__(market, config.strategy, config.portfolio)

    def _apply_research_score_overlay(
        self,
        candidates: pd.DataFrame,
        signal_date: pd.Timestamp,
    ) -> pd.DataFrame:
        snapshot = self.score_panel.at(signal_date)
        score_column = f"z_{PIT_COMPOSITE_NAME}"
        scores = snapshot.set_index("symbol") if not snapshot.empty else snapshot
        output = candidates.copy()
        output["pit_composite_score_input"] = output["symbol"].map(
            scores[score_column] if not snapshot.empty else {}
        )
        output["pit_available_factor_count"] = output["symbol"].map(
            scores["available_factor_count"] if not snapshot.empty else {}
        )
        output["pit_coverage_eligible"] = output[
            "pit_composite_score_input"
        ].notna() & output["pit_available_factor_count"].ge(
            self.score_panel.minimum_factors_per_symbol
        )
        output = output.loc[output["pit_coverage_eligible"]].copy()
        if output.empty:
            return output

        output["price_score_pre_neutral"] = output["score_pre_neutral"]
        output["z_price_shadow"] = _winsorized_zscore(
            output["price_score_pre_neutral"], self.config.winsor_quantile
        )
        output["z_pit_shadow"] = _winsorized_zscore(
            output["pit_composite_score_input"], self.config.winsor_quantile
        )
        if self.shadow_arm == COVERAGE_MATCHED_PRICE_ARM:
            output["score_pre_neutral"] = output["z_price_shadow"]
            effective_weight = 0.0
        elif self.shadow_arm == PIT_ONLY_ARM:
            output["score_pre_neutral"] = output["z_pit_shadow"]
            effective_weight = 1.0
        else:
            output["score_pre_neutral"] = (1.0 - self.blend_weight) * output[
                "z_price_shadow"
            ] + self.blend_weight * output["z_pit_shadow"]
            effective_weight = self.blend_weight
        output["shadow_arm"] = self.shadow_arm
        output["shadow_pit_weight"] = effective_weight
        output["pit_score_signal_date"] = signal_date
        return output

    def _research_selection_columns(self) -> list[str]:
        return [
            "shadow_arm",
            "shadow_pit_weight",
            "pit_score_signal_date",
            "pit_coverage_eligible",
            "pit_available_factor_count",
            "pit_composite_score_input",
            "price_score_pre_neutral",
            "z_price_shadow",
            "z_pit_shadow",
        ]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"缺少 Alpha2 研究产物: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Alpha2 研究产物不是有效 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Alpha2 研究产物必须是 JSON 对象: {path}")
    return payload


def validate_alpha2_research_bundle(
    research_dir: str | Path,
    *,
    base_data_fingerprint: str,
    pit_data_fingerprint: str,
) -> dict[str, Any]:
    root = Path(research_dir).resolve()
    artifact_path = root / "artifact_manifest.json"
    verification = verify_artifact_manifest(artifact_path, strict=True)
    manifest_path = root / "pit_factor_research_manifest.json"
    governance_path = root / "pit_factor_governance.json"
    manifest = _read_json(manifest_path)
    governance = _read_json(governance_path)
    expected_factors = resolve_pit_factor_names()
    parameters = manifest.get("parameters", {})
    checks = {
        "research_version": manifest.get("research_version")
        == PIT_FACTOR_RESEARCH_VERSION,
        "candidate": parameters.get("composite") == PIT_COMPOSITE_NAME,
        "fixed_factor_registry": parameters.get("factor_names") == expected_factors,
        "fixed_minimum_factors": parameters.get("minimum_factors_per_symbol")
        == PIT_SHADOW_MINIMUM_FACTORS,
        "production_unchanged": manifest.get("production_strategy_changed") is False,
        "base_fingerprint": manifest.get("base_data_fingerprint_sha256")
        == base_data_fingerprint,
        "pit_fingerprint": manifest.get("pit_data_fingerprint_sha256")
        == pit_data_fingerprint,
        "governance_candidate": governance.get("candidate") == PIT_COMPOSITE_NAME,
        "automatic_promotion_disabled": governance.get("automatic_production_promotion")
        is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Alpha2 研究包不满足固定候选契约: " + ", ".join(failed))
    return {
        "root": root,
        "manifest_path": manifest_path,
        "governance_path": governance_path,
        "artifact_path": artifact_path,
        "manifest": manifest,
        "governance": governance,
        "artifact_verification": verification,
        "checks": checks,
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


def _run_arm(
    market: MarketDataBundle,
    config: AppConfig,
    score_panel: PITShadowScorePanel,
    arm: str,
) -> tuple[BacktestResult, dict[str, Any]]:
    strategy = (
        None
        if arm == PRODUCTION_BASELINE_ARM
        else PITShadowStrategy(market, config, score_panel, arm=arm)
    )
    result = Backtester(market, config, strategy=strategy).run()
    metrics = calculate_metrics(result, config)
    metrics.update(
        {
            "shadow_arm": arm,
            "effective_alpha_identity": (
                metrics["alpha_profile"]
                if arm == PRODUCTION_BASELINE_ARM
                else f"{PIT_SHADOW_CANDIDATE}:{arm}"
            ),
            "effective_alpha_lifecycle_status": (
                metrics["alpha_profile_status"]
                if arm == PRODUCTION_BASELINE_ARM
                else "research_only"
            ),
            "effective_alpha_promotion_decision": (
                metrics["alpha_promotion_decision"]
                if arm == PRODUCTION_BASELINE_ARM
                else "not_eligible_for_production"
            ),
            "pit_shadow_weight": (
                0.0
                if arm in {PRODUCTION_BASELINE_ARM, COVERAGE_MATCHED_PRICE_ARM}
                else 1.0
                if arm == PIT_ONLY_ARM
                else PIT_SHADOW_BLEND_WEIGHT
            ),
            "production_default_changed": False,
            "warnings": list(result.warnings),
        }
    )
    return result, metrics


def _comparison_frame(metrics_by_arm: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    fields = [
        "cagr",
        "sharpe",
        "max_drawdown",
        "excess_cagr",
        "information_ratio",
        "annual_turnover",
        "total_fees",
        "estimated_fixed_slippage_cost",
        "estimated_market_impact_cost",
        "filled_trade_count",
        "rejected_or_cancelled_order_count",
    ]
    production = metrics_by_arm[PRODUCTION_BASELINE_ARM]
    coverage = metrics_by_arm[COVERAGE_MATCHED_PRICE_ARM]
    records: list[dict[str, Any]] = []
    for arm in PIT_SHADOW_ARMS:
        metrics = metrics_by_arm[arm]
        record: dict[str, Any] = {
            "arm": arm,
            "candidate": (
                "" if arm == PRODUCTION_BASELINE_ARM else PIT_SHADOW_CANDIDATE
            ),
            "lifecycle_status": metrics["effective_alpha_lifecycle_status"],
            "pit_shadow_weight": metrics["pit_shadow_weight"],
        }
        for field in fields:
            value = metrics.get(field)
            record[field] = value
            if isinstance(value, (int, float)) and value is not None:
                production_value = production.get(field)
                coverage_value = coverage.get(field)
                record[f"{field}_delta_vs_production"] = (
                    float(value) - float(production_value)
                    if isinstance(production_value, (int, float))
                    and production_value is not None
                    else np.nan
                )
                record[f"{field}_delta_vs_coverage_matched"] = (
                    float(value) - float(coverage_value)
                    if isinstance(coverage_value, (int, float))
                    and coverage_value is not None
                    else np.nan
                )
        records.append(record)
    return pd.DataFrame(records)


def _annual_metrics(
    results: Mapping[str, BacktestResult],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for arm, result in results.items():
        curve = result.equity_curve.copy().sort_values("date")
        curve["date"] = pd.to_datetime(curve["date"])
        curve["return"] = pd.to_numeric(curve["nav"], errors="coerce").pct_change(
            fill_method=None
        )
        curve["daily_notional"] = (
            pd.to_numeric(curve["cumulative_notional"], errors="coerce")
            .diff()
            .fillna(curve["cumulative_notional"])
        )
        curve["daily_fees"] = (
            pd.to_numeric(curve["cumulative_fees"], errors="coerce")
            .diff()
            .fillna(curve["cumulative_fees"])
        )
        for year, group in curve.groupby(curve["date"].dt.year, sort=True):
            returns = group["return"].dropna()
            if returns.empty:
                continue
            growth = (1.0 + returns).cumprod()
            drawdown = growth / growth.cummax().clip(lower=1.0) - 1.0
            deviation = float(returns.std(ddof=0))
            average_nav = float(group["nav"].mean())
            records.append(
                {
                    "arm": arm,
                    "year": int(year),
                    "trading_days": len(returns),
                    "return": float(growth.iloc[-1] - 1.0),
                    "sharpe": (
                        float(returns.mean() / deviation * math.sqrt(252.0))
                        if deviation > 0
                        else np.nan
                    ),
                    "maximum_drawdown": float(drawdown.min()),
                    "one_sided_turnover": (
                        0.5 * float(group["daily_notional"].sum()) / average_nav
                        if average_nav > 0
                        else np.nan
                    ),
                    "fees": float(group["daily_fees"].sum()),
                }
            )
    return pd.DataFrame(records)


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first.union(second)
    return len(first.intersection(second)) / len(union) if union else np.nan


def _selection_attribution(
    results: Mapping[str, BacktestResult],
) -> pd.DataFrame:
    by_arm: dict[str, dict[pd.Timestamp, pd.DataFrame]] = {}
    for arm, result in results.items():
        selections = result.selections.copy()
        if selections.empty:
            by_arm[arm] = {}
            continue
        selections["signal_date"] = pd.to_datetime(
            selections["signal_date"]
        ).dt.normalize()
        by_arm[arm] = {
            date: group.copy()
            for date, group in selections.groupby("signal_date", sort=True)
        }
    dates = sorted({date for groups in by_arm.values() for date in groups})
    records: list[dict[str, Any]] = []
    for signal_date in dates:
        production = by_arm[PRODUCTION_BASELINE_ARM].get(signal_date, pd.DataFrame())
        coverage = by_arm[COVERAGE_MATCHED_PRICE_ARM].get(signal_date, pd.DataFrame())
        production_symbols = set(production.get("symbol", pd.Series(dtype=str)))
        coverage_symbols = set(coverage.get("symbol", pd.Series(dtype=str)))
        for arm in PIT_SHADOW_ARMS:
            group = by_arm[arm].get(signal_date, pd.DataFrame())
            symbols = set(group.get("symbol", pd.Series(dtype=str)))
            pit_scores = pd.to_numeric(
                group.get(
                    "pit_composite_score_input",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            )
            records.append(
                {
                    "signal_date": signal_date,
                    "arm": arm,
                    "selected_count": len(symbols),
                    "overlap_with_production": len(
                        symbols.intersection(production_symbols)
                    ),
                    "jaccard_with_production": _jaccard(symbols, production_symbols),
                    "overlap_with_coverage_matched": len(
                        symbols.intersection(coverage_symbols)
                    ),
                    "jaccard_with_coverage_matched": _jaccard(
                        symbols, coverage_symbols
                    ),
                    "average_selected_pit_score": (
                        float(pit_scores.mean()) if pit_scores.notna().any() else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def _normalize_cost_bps(cost_bps: Sequence[float]) -> list[float]:
    costs: list[float] = []
    for raw in cost_bps:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("cost_bps 必须全部是有限非负数") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("cost_bps 必须全部是有限非负数")
        costs.append(value)
    if not costs:
        raise ValueError("cost_bps 不能为空")
    return sorted(set(costs))


def _cost_stress(
    market: MarketDataBundle,
    config: AppConfig,
    score_panel: PITShadowScorePanel,
    base_hybrid_metrics: Mapping[str, Any],
    cost_bps: Sequence[float],
) -> pd.DataFrame:
    costs = _normalize_cost_bps(cost_bps)
    records: list[dict[str, Any]] = []
    for bps in costs:
        if math.isclose(bps, config.execution.slippage_bps, abs_tol=1e-12):
            metrics = dict(base_hybrid_metrics)
        else:
            stressed = replace(
                config,
                execution=replace(config.execution, slippage_bps=bps),
            )
            _, metrics = _run_arm(market, stressed, score_panel, HYBRID_ARM)
        records.append(
            {
                "fixed_slippage_bps": bps,
                "market_impact_model": config.execution.market_impact_model,
                "cagr": metrics.get("cagr"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "annual_turnover": metrics.get("annual_turnover"),
                "total_fees": metrics.get("total_fees"),
                "estimated_fixed_slippage_cost": metrics.get(
                    "estimated_fixed_slippage_cost"
                ),
                "estimated_market_impact_cost": metrics.get(
                    "estimated_market_impact_cost"
                ),
            }
        )
    return pd.DataFrame(records)


def build_pit_shadow_governance(
    *,
    provider: str,
    alpha2_governance: Mapping[str, Any],
    coverage: pd.DataFrame,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    cost_stress: pd.DataFrame,
) -> dict[str, Any]:
    def number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    def row(arm: str) -> pd.Series:
        matches = comparison.loc[comparison["arm"].eq(arm)]
        return matches.iloc[0] if not matches.empty else pd.Series(dtype=float)

    hybrid = row(HYBRID_ARM)
    coverage_control = row(COVERAGE_MATCHED_PRICE_ARM)
    annual_pivot = annual.pivot(index="year", columns="arm", values="return")
    if HYBRID_ARM in annual_pivot and COVERAGE_MATCHED_PRICE_ARM in annual_pivot:
        annual_delta = (
            annual_pivot[HYBRID_ARM] - annual_pivot[COVERAGE_MATCHED_PRICE_ARM]
        ).dropna()
    else:
        annual_delta = pd.Series(dtype=float)
    maximum_cost_row = cost_stress.sort_values("fixed_slippage_bps").tail(1)

    evidence = {
        "alpha2_promotion_decision": alpha2_governance.get("promotion_decision"),
        "signal_dates": int(coverage["signal_date"].nunique()),
        "median_score_coverage": float(coverage["eligible_coverage"].median()),
        "minimum_score_coverage": float(coverage["eligible_coverage"].min()),
        "hybrid_cagr_delta_vs_coverage_matched": number(hybrid.get("cagr"))
        - number(coverage_control.get("cagr")),
        "hybrid_sharpe_delta_vs_coverage_matched": number(hybrid.get("sharpe"))
        - number(coverage_control.get("sharpe")),
        "hybrid_drawdown_delta_vs_coverage_matched": number(hybrid.get("max_drawdown"))
        - number(coverage_control.get("max_drawdown")),
        "hybrid_turnover_ratio_vs_coverage_matched": (
            number(hybrid.get("annual_turnover"))
            / number(coverage_control.get("annual_turnover"))
            if number(coverage_control.get("annual_turnover")) > 0
            else np.nan
        ),
        "positive_annual_increment_ratio": (
            float(annual_delta.gt(0).mean()) if len(annual_delta) else np.nan
        ),
        "maximum_cost_bps": float(maximum_cost_row.iloc[0]["fixed_slippage_bps"]),
        "maximum_cost_hybrid_cagr": float(maximum_cost_row.iloc[0]["cagr"]),
    }
    thresholds = {
        "signal_dates": 24,
        "median_score_coverage": 0.60,
        "minimum_score_coverage": 0.40,
        "hybrid_cagr_delta_vs_coverage_matched": 0.0,
        "hybrid_sharpe_delta_vs_coverage_matched": 0.0,
        "hybrid_drawdown_delta_vs_coverage_matched_min": -0.03,
        "hybrid_turnover_ratio_vs_coverage_matched_max": 1.50,
        "positive_annual_increment_ratio": 0.60,
        "maximum_cost_hybrid_cagr": 0.0,
    }

    def finite_compare(value: Any, threshold: float, operator: str) -> bool:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(number):
            return False
        if operator == "ge":
            return number >= threshold
        if operator == "gt":
            return number > threshold
        if operator == "le":
            return number <= threshold
        raise ValueError(operator)

    checks = [
        (
            "alpha2_manual_review_gate",
            evidence["alpha2_promotion_decision"]
            == "eligible_for_manual_candidate_review",
        ),
        (
            "signal_dates",
            evidence["signal_dates"] >= thresholds["signal_dates"],
        ),
        (
            "median_score_coverage",
            finite_compare(
                evidence["median_score_coverage"],
                thresholds["median_score_coverage"],
                "ge",
            ),
        ),
        (
            "minimum_score_coverage",
            finite_compare(
                evidence["minimum_score_coverage"],
                thresholds["minimum_score_coverage"],
                "ge",
            ),
        ),
        (
            "hybrid_cagr_increment",
            finite_compare(
                evidence["hybrid_cagr_delta_vs_coverage_matched"],
                thresholds["hybrid_cagr_delta_vs_coverage_matched"],
                "gt",
            ),
        ),
        (
            "hybrid_sharpe_increment",
            finite_compare(
                evidence["hybrid_sharpe_delta_vs_coverage_matched"],
                thresholds["hybrid_sharpe_delta_vs_coverage_matched"],
                "gt",
            ),
        ),
        (
            "hybrid_drawdown_not_materially_worse",
            finite_compare(
                evidence["hybrid_drawdown_delta_vs_coverage_matched"],
                thresholds["hybrid_drawdown_delta_vs_coverage_matched_min"],
                "ge",
            ),
        ),
        (
            "hybrid_turnover_bounded",
            finite_compare(
                evidence["hybrid_turnover_ratio_vs_coverage_matched"],
                thresholds["hybrid_turnover_ratio_vs_coverage_matched_max"],
                "le",
            ),
        ),
        (
            "positive_annual_increment_ratio",
            finite_compare(
                evidence["positive_annual_increment_ratio"],
                thresholds["positive_annual_increment_ratio"],
                "ge",
            ),
        ),
        (
            "maximum_cost_hybrid_cagr",
            finite_compare(
                evidence["maximum_cost_hybrid_cagr"],
                thresholds["maximum_cost_hybrid_cagr"],
                "gt",
            ),
        ),
    ]
    serialized_checks = [
        {"name": name, "passed": bool(passed)} for name, passed in checks
    ]
    quantitative_passed = all(
        item["passed"]
        for item in serialized_checks
        if item["name"] != "alpha2_manual_review_gate"
    )
    all_passed = all(item["passed"] for item in serialized_checks)
    if provider != "tushare":
        decision = "nonproduction_data_only"
    elif evidence["alpha2_promotion_decision"] != (
        "eligible_for_manual_candidate_review"
    ):
        decision = "blocked_by_alpha2_evidence"
    elif all_passed:
        decision = "eligible_for_forward_paper_tracking"
    else:
        decision = "insufficient_strict_ledger_evidence"
    return {
        "schema_version": 1,
        "research_version": PIT_SHADOW_RESEARCH_VERSION,
        "candidate": PIT_SHADOW_CANDIDATE,
        "lifecycle_status": "research_only",
        "production_weight": 0.0,
        "promotion_decision": decision,
        "provider": provider,
        "evidence": evidence,
        "thresholds": thresholds,
        "checks": serialized_checks,
        "all_quantitative_checks_passed": quantitative_passed,
        "all_checks_passed": all_passed,
        "automatic_parameter_fitting": False,
        "automatic_production_promotion": False,
        "production_default_changed": False,
        "untouched_holdout_certified": False,
        "note": (
            "Alpha3 只允许把固定候选送入前瞻模拟观察；历史严格账本"
            "即使全部通过，也不能自动晋级生产或承诺 15% 年化收益。"
        ),
    }


def write_pit_shadow_research(
    market: MarketDataBundle,
    pit: PointInTimeDataBundle,
    config: AppConfig,
    output_dir: str | Path,
    *,
    alpha2_research_dir: str | Path,
    cost_bps: Sequence[float] = (5.0, 10.0, 20.0),
    acceptance_report: str | Path | None = None,
) -> dict[str, Path]:
    if not config.point_in_time.enabled:
        raise ValueError("PIT 影子研究要求 point_in_time.enabled=true")
    normalized_cost_bps = _normalize_cost_bps(cost_bps)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"PIT 影子研究输出目录非空，拒绝覆盖或混入旧产物: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    base_manifest_path = Path(config.data.cache_dir).resolve() / "manifest.json"
    pit_manifest_path = Path(config.point_in_time.cache_dir).resolve() / "manifest.json"
    if not base_manifest_path.is_file() or not pit_manifest_path.is_file():
        raise FileNotFoundError("PIT 影子研究必须绑定基础行情与 PIT manifest")
    base_manifest = _read_json(base_manifest_path)
    pit_manifest = _read_json(pit_manifest_path)
    require_pit_research_eligible(pit_manifest)
    acceptance = require_pit_acceptance(pit_manifest, acceptance_report)
    base_fingerprint = str(base_manifest.get("data_fingerprint_sha256", ""))
    pit_fingerprint = str(pit_manifest.get("data_fingerprint_sha256", ""))
    if not base_fingerprint or not pit_fingerprint:
        raise ValueError("基础行情与 PIT manifest 必须包含非空数据指纹")
    expected_provider = config.data.provider
    if base_manifest.get("provider") != expected_provider:
        raise ValueError("基础行情 manifest 的 provider 与当前配置不一致")
    if pit_manifest.get("provider") != expected_provider:
        raise ValueError("PIT manifest 的 provider 与当前配置不一致")
    if pit_manifest.get("base_data_fingerprint_sha256") != base_fingerprint:
        raise ValueError("PIT manifest 未绑定当前基础行情数据指纹")
    if pit.manifest.get("data_fingerprint_sha256") != pit_fingerprint:
        raise ValueError("内存 PIT 数据与磁盘 manifest 指纹不一致")
    if pit.manifest.get("provider") != expected_provider:
        raise ValueError("内存 PIT 数据的 provider 与当前配置不一致")
    if pit.manifest.get("base_data_fingerprint_sha256") != base_fingerprint:
        raise ValueError("内存 PIT 数据未绑定当前基础行情数据指纹")

    alpha2 = validate_alpha2_research_bundle(
        alpha2_research_dir,
        base_data_fingerprint=base_fingerprint,
        pit_data_fingerprint=pit_fingerprint,
    )
    score_frame = build_pit_shadow_score_panel(market, pit, config)
    score_panel = PITShadowScorePanel(score_frame)
    coverage = score_panel.coverage()

    results: dict[str, BacktestResult] = {}
    metrics_by_arm: dict[str, dict[str, Any]] = {}
    for arm in PIT_SHADOW_ARMS:
        result, metrics = _run_arm(market, config, score_panel, arm)
        results[arm] = result
        metrics_by_arm[arm] = metrics
    comparison = _comparison_frame(metrics_by_arm)
    annual = _annual_metrics(results)
    selection_attribution = _selection_attribution(results)
    cost_stress = _cost_stress(
        market,
        config,
        score_panel,
        metrics_by_arm[HYBRID_ARM],
        normalized_cost_bps,
    )
    governance = build_pit_shadow_governance(
        provider=str(pit_manifest.get("provider", "unknown")),
        alpha2_governance=alpha2["governance"],
        coverage=coverage,
        comparison=comparison,
        annual=annual,
        cost_stress=cost_stress,
    )

    written: dict[str, Path] = {}
    summary_frames = {
        "score_panel": (score_frame, "pit_shadow_score_panel.csv.gz"),
        "coverage": (coverage, "pit_shadow_coverage.csv"),
        "comparison": (comparison, "pit_shadow_comparison.csv"),
        "annual": (annual, "pit_shadow_annual_metrics.csv"),
        "selection_attribution": (
            selection_attribution,
            "pit_shadow_selection_attribution.csv",
        ),
        "cost_stress": (cost_stress, "pit_shadow_cost_stress.csv"),
    }
    for name, (frame, filename) in summary_frames.items():
        written[name] = _atomic_dataframe(frame, output / filename)

    for arm in PIT_SHADOW_ARMS:
        result = results[arm]
        arm_dir = output / "arms" / arm
        arm_frames = {
            "equity": (result.equity_curve, "equity_curve.csv.gz"),
            "trades": (result.trades, "trades.csv.gz"),
            "orders": (result.orders, "orders.csv.gz"),
            "selections": (result.selections, "selections.csv.gz"),
            "corporate_events": (
                result.corporate_events,
                "corporate_events.csv.gz",
            ),
        }
        for kind, (frame, filename) in arm_frames.items():
            written[f"{arm}_{kind}"] = _atomic_dataframe(frame, arm_dir / filename)
        written[f"{arm}_metrics"] = write_json_atomic(
            metrics_by_arm[arm], arm_dir / "metrics.json"
        )
        written[f"{arm}_final_positions"] = write_json_atomic(
            result.final_positions, arm_dir / "final_positions.json"
        )

    governance_path = write_json_atomic(
        governance, output / "pit_shadow_governance.json"
    )
    written["governance"] = governance_path
    parameters = {
        "candidate": PIT_SHADOW_CANDIDATE,
        "factor_names": resolve_pit_factor_names(),
        "minimum_factors_per_symbol": PIT_SHADOW_MINIMUM_FACTORS,
        "hybrid_pit_weight": PIT_SHADOW_BLEND_WEIGHT,
        "arms": list(PIT_SHADOW_ARMS),
        "cost_bps": normalized_cost_bps,
        "automatic_parameter_fitting": False,
        "pit_acceptance_fingerprint_sha256": (
            acceptance.get("acceptance_fingerprint_sha256")
            if acceptance is not None
            else None
        ),
    }
    extra_inputs = [
        pit_manifest_path,
        alpha2["manifest_path"],
        alpha2["governance_path"],
        alpha2["artifact_path"],
    ]
    if acceptance is not None and acceptance_report is not None:
        extra_inputs.append(Path(acceptance_report).resolve())
    reproducibility = build_reproducibility_manifest(
        {
            "app_config": config.to_dict(),
            "pit_shadow_research": parameters,
        },
        data_manifest_path=base_manifest_path,
        extra_input_files=extra_inputs,
    )
    reproducibility["pit_data"] = {
        "manifest_path": str(pit_manifest_path),
        "manifest_sha256": sha256_file(pit_manifest_path),
        "data_fingerprint_sha256": pit_fingerprint,
        "base_data_fingerprint_sha256": base_fingerprint,
    }
    reproducibility["pit_acceptance"] = (
        {
            "report_path": str(Path(acceptance_report).resolve()),
            "acceptance_fingerprint_sha256": acceptance.get(
                "acceptance_fingerprint_sha256"
            ),
        }
        if acceptance is not None and acceptance_report is not None
        else None
    )
    reproducibility["alpha2_research"] = {
        "directory": str(alpha2["root"]),
        "manifest_sha256": sha256_file(alpha2["manifest_path"]),
        "artifact_manifest_sha256": sha256_file(alpha2["artifact_path"]),
        "promotion_decision": alpha2["governance"].get("promotion_decision"),
    }
    reproducibility_path = write_json_atomic(
        reproducibility, output / "reproducibility.json"
    )
    written["reproducibility"] = reproducibility_path
    manifest = {
        "schema_version": 1,
        "research_version": PIT_SHADOW_RESEARCH_VERSION,
        "parameters": parameters,
        "production_strategy_changed": False,
        "production_default": "legacy_v1_4",
        "candidate_lifecycle_status": "research_only",
        "candidate_production_weight": 0.0,
        "promotion_decision": governance["promotion_decision"],
        "base_data_fingerprint_sha256": base_fingerprint,
        "pit_data_fingerprint_sha256": pit_fingerprint,
        "alpha2_run_fingerprint_sha256": alpha2["manifest"].get(
            "run_fingerprint_sha256"
        ),
        "run_fingerprint_sha256": reproducibility["run_fingerprint_sha256"],
        "files": {
            name: path.relative_to(output).as_posix() for name, path in written.items()
        },
        "limitations": [
            "历史区间已被查看，不是未触碰样本外。",
            "固定 25% PIT 权重不执行自动寻优。",
            "结果只允许进入前瞻模拟观察，不能自动晋级生产。",
            "真实收益结论必须绑定 Tushare PIT 与基础行情双指纹。",
        ],
    }
    manifest_path = write_json_atomic(
        manifest, output / "pit_shadow_research_manifest.json"
    )
    written["manifest"] = manifest_path
    registry_path = record_experiment(
        output / "experiment_registry.jsonl",
        reproducibility,
        experiment_type=PIT_SHADOW_RESEARCH_VERSION,
        protocol={
            **parameters,
            "alpha2_promotion_decision": alpha2["governance"].get("promotion_decision"),
            "automatic_production_promotion": False,
            "untouched_holdout_certified": False,
        },
        artifacts=written.values(),
    )
    written["experiment_registry"] = registry_path
    artifact_manifest = write_artifact_manifest(output, written.values())
    written["artifacts"] = artifact_manifest
    return written
