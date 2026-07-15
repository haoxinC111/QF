from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


DEFAULT_PORTFOLIO_MODEL = "inverse_vol_v1_4"
PORTFOLIO_MODEL_V1_6 = "shrinkage_min_variance_v1_6"
SUPPORTED_PORTFOLIO_MODELS = {
    DEFAULT_PORTFOLIO_MODEL,
    PORTFOLIO_MODEL_V1_6,
}


def portfolio_model_governance(model: str) -> dict[str, object]:
    if model == DEFAULT_PORTFOLIO_MODEL:
        return {
            "lifecycle_status": "promoted",
            "promotion_decision": "promoted",
            "default_eligible": True,
            "reason": "v1.5.1 生产基线：逆波动率并施加单股与行业上限",
        }
    if model == PORTFOLIO_MODEL_V1_6:
        return {
            "lifecycle_status": "experimental",
            "promotion_decision": "pending_validation",
            "default_eligible": False,
            "reason": "v1.6 风险感知与换手平滑模型，等待绑定数据指纹的历史复核",
        }
    return {
        "lifecycle_status": "unknown",
        "promotion_decision": "rejected",
        "default_eligible": False,
        "reason": "未知组合模型",
    }


def group_capped_allocation(
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
                float(
                    (
                        stock_room.loc[proposal.index][positive] / proposal[positive]
                    ).min()
                ),
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


@dataclass(frozen=True)
class PortfolioAllocation:
    weights: pd.Series
    raw_weights: pd.Series
    covariance_observations: int
    status: str


def _shrunk_covariance(
    returns: pd.DataFrame,
    *,
    minimum_observations: int,
    shrinkage: float,
    ridge: float,
) -> tuple[np.ndarray | None, int]:
    clean = returns.replace([np.inf, -np.inf], np.nan)
    observations = int(clean.notna().sum().min()) if not clean.empty else 0
    if observations < minimum_observations:
        return None, observations

    sample = clean.cov(min_periods=minimum_observations).to_numpy(dtype=float)
    diagonal = np.diag(sample).copy()
    valid_variances = diagonal[np.isfinite(diagonal) & (diagonal > 0)]
    if len(valid_variances) != len(diagonal) or len(valid_variances) == 0:
        return None, observations

    sample = np.where(np.isfinite(sample), sample, 0.0)
    sample = (sample + sample.T) / 2.0
    shrunk = (1.0 - shrinkage) * sample + shrinkage * np.diag(diagonal)

    # Pairwise-complete covariance can be slightly indefinite. Projecting its
    # eigenvalues to a small positive floor makes the inverse deterministic and
    # prevents numerical noise from becoming an extreme position.
    scale = max(float(np.median(valid_variances)), 1e-12)
    floor = max(scale * ridge, 1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    positive = np.maximum(eigenvalues, floor)
    covariance = (eigenvectors * positive) @ eigenvectors.T
    covariance = (covariance + covariance.T) / 2.0
    return covariance, observations


def allocate_portfolio(
    *,
    model: str,
    inverse_risk: pd.Series,
    total: float,
    stock_cap: float,
    groups: pd.Series,
    group_cap: float,
    returns: pd.DataFrame | None = None,
    current_weights: Mapping[str, float] | None = None,
    covariance_lookback_days: int = 120,
    minimum_covariance_observations: int = 60,
    covariance_shrinkage: float = 0.50,
    minimum_variance_blend: float = 0.50,
    turnover_smoothing: float = 0.50,
    covariance_ridge: float = 1e-6,
) -> PortfolioAllocation:
    """Build a feasible target without changing the selected security set.

    The v1.6 model blends the legacy inverse-volatility anchor with a long-only
    approximation to the shrinkage-covariance minimum-variance portfolio. It then
    blends that risk target with signal-close holdings before reapplying hard
    single-name and industry caps. Insufficient covariance history falls back to
    the legacy target explicitly instead of using an unstable estimate.
    """
    if model not in SUPPORTED_PORTFOLIO_MODELS:
        raise ValueError("未知组合模型: " + str(model))

    inverse = pd.to_numeric(inverse_risk, errors="coerce").fillna(0.0).clip(lower=0.0)
    legacy = group_capped_allocation(
        inverse,
        total,
        stock_cap,
        groups,
        group_cap,
    )
    if model == DEFAULT_PORTFOLIO_MODEL:
        return PortfolioAllocation(legacy, legacy.copy(), 0, "legacy")

    if returns is None:
        return PortfolioAllocation(
            legacy,
            legacy.copy(),
            0,
            "fallback_missing_returns",
        )
    history = returns.reindex(columns=inverse.index).tail(covariance_lookback_days)
    covariance, observations = _shrunk_covariance(
        history,
        minimum_observations=minimum_covariance_observations,
        shrinkage=covariance_shrinkage,
        ridge=covariance_ridge,
    )
    if covariance is None:
        return PortfolioAllocation(
            legacy,
            legacy.copy(),
            observations,
            "fallback_insufficient_covariance",
        )

    try:
        minimum_variance = np.linalg.pinv(covariance, hermitian=True) @ np.ones(
            len(inverse)
        )
    except np.linalg.LinAlgError:
        return PortfolioAllocation(
            legacy,
            legacy.copy(),
            observations,
            "fallback_covariance_solver",
        )
    minimum_variance = np.clip(minimum_variance, 0.0, None)
    if not np.isfinite(minimum_variance).all() or minimum_variance.sum() <= 1e-15:
        return PortfolioAllocation(
            legacy,
            legacy.copy(),
            observations,
            "fallback_covariance_solver",
        )

    inverse_preference = inverse / max(float(inverse.sum()), 1e-15)
    minimum_variance_preference = pd.Series(
        minimum_variance / minimum_variance.sum(),
        index=inverse.index,
        dtype=float,
    )
    risk_preference = (
        1.0 - minimum_variance_blend
    ) * inverse_preference + minimum_variance_blend * minimum_variance_preference
    risk_target = group_capped_allocation(
        risk_preference,
        total,
        stock_cap,
        groups,
        group_cap,
    )

    current = pd.Series(dict(current_weights or {}), dtype=float).reindex(
        inverse.index, fill_value=0.0
    )
    current = pd.to_numeric(current, errors="coerce").fillna(0.0).clip(lower=0.0)
    if turnover_smoothing > 0 and float(current.sum()) > 1e-12:
        preference = (
            1.0 - turnover_smoothing
        ) * risk_target + turnover_smoothing * current
        final = group_capped_allocation(
            preference,
            total,
            stock_cap,
            groups,
            group_cap,
        )
    else:
        final = risk_target.copy()
    return PortfolioAllocation(final, risk_target, observations, "applied")


__all__ = [
    "DEFAULT_PORTFOLIO_MODEL",
    "PORTFOLIO_MODEL_V1_6",
    "SUPPORTED_PORTFOLIO_MODELS",
    "PortfolioAllocation",
    "allocate_portfolio",
    "group_capped_allocation",
    "portfolio_model_governance",
]
