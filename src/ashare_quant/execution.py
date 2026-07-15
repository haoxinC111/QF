from __future__ import annotations

import math


DEFAULT_EXECUTION_MODEL = "fixed_bps"
EXECUTION_MODEL_V1_6 = "square_root_v1_6"
SUPPORTED_EXECUTION_MODELS = {
    DEFAULT_EXECUTION_MODEL,
    EXECUTION_MODEL_V1_6,
}


def execution_model_governance(model: str) -> dict[str, object]:
    if model == DEFAULT_EXECUTION_MODEL:
        return {
            "lifecycle_status": "promoted",
            "promotion_decision": "promoted",
            "default_eligible": True,
            "reason": "稳定生产基线：固定单边滑点与滞后 ADV20 容量限制",
        }
    if model == EXECUTION_MODEL_V1_6:
        return {
            "lifecycle_status": "experimental",
            "promotion_decision": "pending_validation",
            "default_eligible": False,
            "reason": "v1.6 平方根冲击模型，等待绑定数据指纹和仿真成交校准",
        }
    return {
        "lifecycle_status": "unknown",
        "promotion_decision": "rejected",
        "default_eligible": False,
        "reason": "未知成交模型",
    }


def market_impact_bps(
    *,
    model: str,
    annualized_volatility: float,
    participation_rate: float,
    coefficient: float,
    annualized_volatility_floor: float,
    maximum_bps: float,
) -> float:
    """Return one-way impact in bps using only signal-known inputs."""
    if model not in SUPPORTED_EXECUTION_MODELS:
        raise ValueError("未知成交模型: " + str(model))
    if model == DEFAULT_EXECUTION_MODEL:
        return 0.0
    volatility = float(annualized_volatility)
    if not math.isfinite(volatility):
        volatility = float(annualized_volatility_floor)
    volatility = max(volatility, float(annualized_volatility_floor), 0.0)
    participation = max(float(participation_rate), 0.0)
    daily_volatility = volatility / math.sqrt(252.0)
    impact = coefficient * daily_volatility * math.sqrt(participation) * 10_000.0
    return max(0.0, min(float(maximum_bps), float(impact)))


__all__ = [
    "DEFAULT_EXECUTION_MODEL",
    "EXECUTION_MODEL_V1_6",
    "SUPPORTED_EXECUTION_MODELS",
    "execution_model_governance",
    "market_impact_bps",
]
