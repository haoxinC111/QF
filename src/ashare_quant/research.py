from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .alpha import (
    ALPHA_MODEL_VERSION,
    DEFAULT_ALPHA_PROFILE,
    LEGACY_V1_4_WEIGHTS,
    QUALITY_MOMENTUM_V1_5_WEIGHTS,
    alpha_profile_governance,
)
from .backtest import Backtester
from .config import AppConfig
from .data import MarketDataBundle
from .report import calculate_metrics
from .provenance import (
    build_reproducibility_manifest,
    record_experiment,
    write_artifact_manifest,
    write_json_atomic,
)


FACTOR_FIELDS = {
    "mom_12_1": "momentum_12_1_weight",
    "mom_6_1": "momentum_6_1_weight",
    "fip_momentum": "fip_momentum_weight",
    "trend": "trend_weight",
    "low_vol": "low_volatility_weight",
    "low_downside_vol": "low_downside_volatility_weight",
    "drawdown_quality": "drawdown_quality_weight",
    "liquidity": "liquidity_weight",
}


def _run_metrics(bundle: MarketDataBundle, config: AppConfig) -> dict[str, object]:
    return calculate_metrics(Backtester(bundle, config).run(), config)


def run_factor_ablation(
    bundle: MarketDataBundle,
    config: AppConfig,
    factors: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Run the full model and one leave-one-factor-out backtest per factor."""
    requested = list(
        factors
        or [
            factor
            for factor, field in FACTOR_FIELDS.items()
            if getattr(config.strategy, field) > 0
        ]
    )
    unknown = sorted(set(requested).difference(FACTOR_FIELDS))
    if unknown:
        raise ValueError("未知因子: " + ", ".join(unknown))

    cases: list[tuple[str, AppConfig]] = [("full", config)]
    for factor in requested:
        strategy = replace(config.strategy, **{FACTOR_FIELDS[factor]: 0.0})
        cases.append((f"without_{factor}", replace(config, strategy=strategy)))

    records: list[dict[str, object]] = []
    for variant, case_config in cases:
        metrics = _run_metrics(bundle, case_config)
        records.append(
            {
                "variant": variant,
                "removed_factor": "" if variant == "full" else variant.removeprefix("without_"),
                **metrics,
            }
        )
    return pd.DataFrame(records)


def run_alpha_comparison(
    bundle: MarketDataBundle,
    config: AppConfig,
) -> pd.DataFrame:
    """Compare frozen v1.4 and v1.5 alpha weights under identical controls."""
    profiles = {
        DEFAULT_ALPHA_PROFILE: LEGACY_V1_4_WEIGHTS,
        ALPHA_MODEL_VERSION: QUALITY_MOMENTUM_V1_5_WEIGHTS,
    }
    records: list[dict[str, object]] = []
    for profile, weights in profiles.items():
        strategy = replace(
            config.strategy,
            **{
                FACTOR_FIELDS[factor]: float(weight)
                for factor, weight in weights.items()
            },
        )
        case_config = replace(config, strategy=strategy)
        records.append(
            {
                "alpha_profile": profile,
                "factor_weights": json.dumps(
                    dict(weights),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                **_run_metrics(bundle, case_config),
            }
        )
    return pd.DataFrame(records)


def run_cost_stress(
    bundle: MarketDataBundle,
    config: AppConfig,
    slippage_bps: Sequence[float] = (5.0, 10.0, 20.0),
    commission_multipliers: Sequence[float] = (1.0, 2.0),
) -> pd.DataFrame:
    """Stress broker commission and slippage; statutory taxes stay unchanged."""
    slippages = sorted({float(config.execution.slippage_bps), *map(float, slippage_bps)})
    multipliers = sorted({1.0, *map(float, commission_multipliers)})
    if min(slippages, default=0.0) < 0 or min(multipliers, default=1.0) <= 0:
        raise ValueError("滑点不能为负，佣金倍数必须大于 0")

    records: list[dict[str, object]] = []
    for slippage in slippages:
        for multiplier in multipliers:
            execution = replace(
                config.execution,
                slippage_bps=slippage,
                commission_rate=config.execution.commission_rate * multiplier,
                minimum_commission=config.execution.minimum_commission * multiplier,
            )
            case_config = replace(config, execution=execution)
            metrics = _run_metrics(bundle, case_config)
            records.append(
                {
                    "scenario": f"slippage_{slippage:g}bps_commission_{multiplier:g}x",
                    "slippage_bps": slippage,
                    "commission_multiplier": multiplier,
                    "commission_rate": execution.commission_rate,
                    "minimum_commission": execution.minimum_commission,
                    **metrics,
                }
            )
    return pd.DataFrame(records)


def run_rolling_oos(
    bundle: MarketDataBundle,
    config: AppConfig,
    train_years: int = 5,
    test_years: int = 1,
) -> pd.DataFrame:
    """Evaluate fixed parameters in expanding, non-overlapping test windows.

    The training interval is reported as a research/freeze period. This project does
    not fit parameters automatically, so no test-window observation is used to alter
    the strategy configuration.
    """
    if train_years < 1 or test_years < 1:
        raise ValueError("train_years 和 test_years 必须至少为 1")
    overall_start = pd.Timestamp(config.backtest.start_date)
    overall_end = pd.Timestamp(config.backtest.end_date)
    test_start = overall_start + pd.DateOffset(years=train_years)
    if test_start >= overall_end:
        raise ValueError("回测区间不足以形成滚动样本外窗口")

    records: list[dict[str, object]] = []
    window = 1
    while test_start < overall_end:
        test_end = min(
            test_start + pd.DateOffset(years=test_years) - pd.Timedelta(1, unit="D"),
            overall_end,
        )
        case_config = replace(
            config,
            backtest=replace(
                config.backtest,
                start_date=test_start.strftime("%Y-%m-%d"),
                end_date=test_end.strftime("%Y-%m-%d"),
            ),
        )
        metrics = _run_metrics(bundle, case_config)
        records.append(
            {
                "window": window,
                "train_start": overall_start.strftime("%Y-%m-%d"),
                "train_end": (test_start - pd.Timedelta(1, unit="D")).strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                **metrics,
            }
        )
        test_start = test_end + pd.Timedelta(1, unit="D")
        window += 1
    return pd.DataFrame(records)


def write_research_suite(
    bundle: MarketDataBundle,
    config: AppConfig,
    output_dir: str | Path,
    modes: Iterable[str] = ("alpha", "ablation", "cost", "rolling"),
    slippage_bps: Sequence[float] = (5.0, 10.0, 20.0),
    commission_multipliers: Sequence[float] = (1.0, 2.0),
    train_years: int = 5,
    test_years: int = 1,
) -> dict[str, Path]:
    requested = list(dict.fromkeys(str(mode).strip().lower() for mode in modes))
    supported = {"alpha", "ablation", "cost", "rolling"}
    unknown = sorted(set(requested).difference(supported))
    if unknown:
        raise ValueError("未知研究模式: " + ", ".join(unknown))
    if not requested:
        raise ValueError("至少选择一个研究模式")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if "alpha" in requested:
        path = output / "alpha_comparison.csv"
        run_alpha_comparison(bundle, config).to_csv(path, index=False)
        written["alpha"] = path
    if "ablation" in requested:
        path = output / "factor_ablation.csv"
        run_factor_ablation(bundle, config).to_csv(path, index=False)
        written["ablation"] = path
    if "cost" in requested:
        path = output / "cost_stress.csv"
        run_cost_stress(
            bundle,
            config,
            slippage_bps=slippage_bps,
            commission_multipliers=commission_multipliers,
        ).to_csv(path, index=False)
        written["cost"] = path
    if "rolling" in requested:
        path = output / "rolling_oos.csv"
        run_rolling_oos(
            bundle,
            config,
            train_years=train_years,
            test_years=test_years,
        ).to_csv(path, index=False)
        written["rolling"] = path

    manifest = {
        "modes": requested,
        "slippage_bps": list(map(float, slippage_bps)),
        "commission_multipliers": list(map(float, commission_multipliers)),
        "train_years": train_years,
        "test_years": test_years,
        "note": "滚动样本外使用冻结参数；训练窗只表示参数研究与冻结区间，不执行自动寻优。",
        "alpha_profiles": {
            DEFAULT_ALPHA_PROFILE: dict(LEGACY_V1_4_WEIGHTS),
            ALPHA_MODEL_VERSION: dict(QUALITY_MOMENTUM_V1_5_WEIGHTS),
        },
        "alpha_profile_governance": {
            DEFAULT_ALPHA_PROFILE: alpha_profile_governance(
                DEFAULT_ALPHA_PROFILE
            ),
            ALPHA_MODEL_VERSION: alpha_profile_governance(ALPHA_MODEL_VERSION),
        },
        "default_alpha_profile": DEFAULT_ALPHA_PROFILE,
        "ablation_interpretation": (
            "删除因子后剩余权重会重新归一化；结果只表示当前组合下的边际证据，"
            "不能解释为单因子的独立因果贡献。"
        ),
        "evaluation_protocol": "fixed_parameter_rolling_evaluation",
        "automatic_parameter_fitting": False,
        "untouched_holdout_certified": False,
        "files": {name: path.name for name, path in written.items()},
    }
    reproducibility = build_reproducibility_manifest(
        {
            "app_config": config.to_dict(),
            "research": {
                "modes": requested,
                "slippage_bps": list(map(float, slippage_bps)),
                "commission_multipliers": list(
                    map(float, commission_multipliers)
                ),
                "train_years": train_years,
                "test_years": test_years,
            },
        },
        data_manifest_path=Path(config.data.cache_dir) / "manifest.json",
    )
    reproducibility_path = write_json_atomic(
        reproducibility, output / "reproducibility.json"
    )
    manifest["reproducibility_file"] = reproducibility_path.name
    manifest["run_fingerprint_sha256"] = reproducibility[
        "run_fingerprint_sha256"
    ]
    manifest_path = output / "research_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written["manifest"] = manifest_path
    written["reproducibility"] = reproducibility_path
    registry_path = record_experiment(
        output / "experiment_registry.jsonl",
        reproducibility,
        experiment_type="strict_research_suite",
        protocol={
            "evaluation": "fixed_parameter_rolling_evaluation",
            "automatic_parameter_fitting": False,
            "untouched_holdout_certified": False,
        },
        artifacts=[*written.values()],
    )
    written["registry"] = registry_path
    artifact_manifest_path = write_artifact_manifest(output, written.values())
    written["artifacts"] = artifact_manifest_path
    return written
