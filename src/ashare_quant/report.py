from __future__ import annotations

import base64
import json
import os
import platform
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .backtest import BacktestResult
from .config import AppConfig
from .provenance import (
    build_reproducibility_manifest,
    record_experiment,
    write_artifact_manifest,
    write_json_atomic,
)


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _with_matched_benchmark(curve: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    frame = curve.copy().sort_values("date").reset_index(drop=True)
    if "benchmark_matched_nav" in frame.columns:
        return frame
    benchmark_return = frame["benchmark_nav"].pct_change(fill_method=None).fillna(0.0)
    exposure = frame["gross_exposure"].shift(1).fillna(0.0).clip(0.0, 1.0)
    cash_return = (1.0 + config.backtest.annual_cash_rate) ** (1.0 / 252.0) - 1.0
    matched_return = exposure * benchmark_return + (1.0 - exposure) * cash_return
    matched_return.iloc[0] = 0.0
    frame["benchmark_matched_nav"] = (
        float(frame["nav"].iloc[0]) * (1.0 + matched_return).cumprod()
    )
    return frame


def _series_statistics(values: pd.Series, years: float, daily_rf: float) -> dict[str, float]:
    returns = values.pct_change(fill_method=None).dropna()
    initial = float(values.iloc[0])
    final = float(values.iloc[-1])
    cagr = (final / initial) ** (1.0 / years) - 1.0 if initial > 0 else np.nan
    volatility = returns.std(ddof=0) * np.sqrt(252.0)
    sharpe = (
        (returns - daily_rf).mean() / returns.std(ddof=0) * np.sqrt(252.0)
        if returns.std(ddof=0) > 0
        else np.nan
    )
    drawdown = values / values.cummax() - 1.0
    return {
        "total_return": final / initial - 1.0,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def industry_exposure_table(selections: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "signal_date",
        "industry_code",
        "industry_name",
        "holding_count",
        "target_weight",
        "stock_book_weight",
    ]
    if selections.empty or "industry_code" not in selections.columns:
        return pd.DataFrame(columns=columns)
    frame = selections.copy()
    frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0)
    grouped = (
        frame.groupby(
            ["signal_date", "industry_code", "industry_name"],
            dropna=False,
            as_index=False,
        )
        .agg(
            holding_count=("symbol", "nunique"),
            target_weight=("target_weight", "sum"),
        )
        .sort_values(["signal_date", "target_weight"], ascending=[True, False])
    )
    total = grouped.groupby("signal_date")["target_weight"].transform("sum")
    grouped["stock_book_weight"] = np.where(
        total > 0, grouped["target_weight"] / total, 0.0
    )
    return grouped[columns].reset_index(drop=True)


def style_exposure_table(selections: pd.DataFrame) -> pd.DataFrame:
    style_columns = [
        "z_size",
        "z_mom_12_1",
        "z_mom_6_1",
        "z_trend",
        "z_low_vol",
        "z_liquidity",
        "volatility",
    ]
    columns = [
        "signal_date",
        "selected_count",
        "target_exposure",
        "buffered_count",
        "buffered_selection_ratio",
        *style_columns,
    ]
    if selections.empty:
        return pd.DataFrame(columns=columns)
    available = [column for column in style_columns if column in selections.columns]
    records: list[dict[str, Any]] = []
    for signal_date, group in selections.groupby("signal_date", sort=True):
        weights = pd.to_numeric(group["target_weight"], errors="coerce").fillna(0.0)
        exposure = float(weights.sum())
        buffered = (
            group.get("selection_reason", pd.Series("", index=group.index))
            .astype(str)
            .eq("HOLD_BUFFER")
        )
        record: dict[str, Any] = {
            "signal_date": signal_date,
            "selected_count": int(group["symbol"].nunique()),
            "target_exposure": exposure,
            "buffered_count": int(buffered.sum()),
            "buffered_selection_ratio": float(buffered.mean()),
        }
        for column in available:
            values = pd.to_numeric(group[column], errors="coerce")
            valid = values.notna() & weights.gt(0)
            denominator = float(weights.loc[valid].sum())
            record[column] = (
                float((values.loc[valid] * weights.loc[valid]).sum() / denominator)
                if denominator > 0
                else np.nan
            )
        records.append(record)
    return pd.DataFrame(records).reindex(columns=columns)


def calculate_metrics(result: BacktestResult, config: AppConfig) -> dict[str, Any]:
    curve = _with_matched_benchmark(result.equity_curve, config)
    if curve.empty:
        raise ValueError("净值序列为空")
    returns = curve["nav"].pct_change(fill_method=None).dropna()
    benchmark_returns = curve["benchmark_nav"].pct_change(fill_method=None).dropna()
    matched_returns = curve["benchmark_matched_nav"].pct_change(fill_method=None).dropna()
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1
    ).dropna()
    matched_aligned = pd.concat(
        [returns.rename("strategy"), matched_returns.rename("benchmark")], axis=1
    ).dropna()
    periods = max(1, len(returns))
    years = periods / 252.0
    initial = float(curve["nav"].iloc[0])
    final = float(curve["nav"].iloc[-1])
    cagr = (final / initial) ** (1.0 / years) - 1.0 if initial > 0 else np.nan
    annual_vol = returns.std(ddof=0) * np.sqrt(252.0)
    daily_rf = (1.0 + config.backtest.annual_risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_daily = returns - daily_rf
    sharpe = (
        excess_daily.mean() / returns.std(ddof=0) * np.sqrt(252.0)
        if returns.std(ddof=0) > 0
        else np.nan
    )
    downside = returns[returns < 0].std(ddof=0)
    sortino = (
        excess_daily.mean() / downside * np.sqrt(252.0)
        if np.isfinite(downside) and downside > 0
        else np.nan
    )
    drawdown = curve["nav"] / curve["nav"].cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
    benchmark_stats = _series_statistics(curve["benchmark_nav"], years, daily_rf)
    matched_stats = _series_statistics(curve["benchmark_matched_nav"], years, daily_rf)

    active = matched_aligned["strategy"] - matched_aligned["benchmark"]
    tracking_error = active.std(ddof=0) * np.sqrt(252.0)
    information_ratio = (
        active.mean() / active.std(ddof=0) * np.sqrt(252.0)
        if active.std(ddof=0) > 0
        else np.nan
    )
    benchmark_variance = aligned["benchmark"].var(ddof=0)
    beta = (
        aligned[["strategy", "benchmark"]].cov(ddof=0).loc["strategy", "benchmark"]
        / benchmark_variance
        if benchmark_variance > 0
        else np.nan
    )
    alpha = (
        (aligned["strategy"].mean() - beta * aligned["benchmark"].mean()) * 252.0
        if np.isfinite(beta)
        else np.nan
    )
    rejected = 0
    if not result.orders.empty and "status" in result.orders:
        rejected = int(result.orders["status"].isin(["REJECTED", "CANCELLED"]).sum())
    total_notional = float(curve["cumulative_notional"].iloc[-1])
    annual_turnover = 0.5 * total_notional / max(curve["nav"].mean(), 1e-12) / years
    industry_exposure = industry_exposure_table(result.selections)
    style_exposure = style_exposure_table(result.selections)
    max_industry_weight = (
        float(industry_exposure["target_weight"].max())
        if not industry_exposure.empty
        else np.nan
    )
    buffered_selection_ratio = (
        float(style_exposure["buffered_count"].sum())
        / max(1.0, float(style_exposure["selected_count"].sum()))
        if not style_exposure.empty
        else np.nan
    )
    average_size_exposure = (
        float(style_exposure["z_size"].mean())
        if not style_exposure.empty and style_exposure["z_size"].notna().any()
        else np.nan
    )

    return {
        "start_date": str(pd.Timestamp(curve["date"].iloc[0]).date()),
        "end_date": str(pd.Timestamp(curve["date"].iloc[-1]).date()),
        "trading_days": int(len(curve)),
        "initial_nav": initial,
        "final_nav": final,
        "total_return": _finite(final / initial - 1.0),
        "cagr": _finite(cagr),
        "annual_volatility": _finite(annual_vol),
        "sharpe": _finite(sharpe),
        "sortino": _finite(sortino),
        "max_drawdown": _finite(max_drawdown),
        "calmar": _finite(calmar),
        "benchmark_total_return": _finite(benchmark_stats["total_return"]),
        "benchmark_cagr": _finite(benchmark_stats["cagr"]),
        "benchmark_annual_volatility": _finite(benchmark_stats["volatility"]),
        "benchmark_sharpe": _finite(benchmark_stats["sharpe"]),
        "benchmark_max_drawdown": _finite(benchmark_stats["max_drawdown"]),
        "matched_benchmark_total_return": _finite(matched_stats["total_return"]),
        "matched_benchmark_cagr": _finite(matched_stats["cagr"]),
        "matched_benchmark_annual_volatility": _finite(matched_stats["volatility"]),
        "matched_benchmark_sharpe": _finite(matched_stats["sharpe"]),
        "matched_benchmark_max_drawdown": _finite(matched_stats["max_drawdown"]),
        "excess_cagr": _finite(cagr - matched_stats["cagr"]),
        "tracking_error": _finite(tracking_error),
        "information_ratio": _finite(information_ratio),
        "beta": _finite(beta),
        "annual_alpha": _finite(alpha),
        "positive_day_ratio": _finite((returns > 0).mean()),
        "worst_day": _finite(returns.min()),
        "best_day": _finite(returns.max()),
        "average_exposure": _finite(curve["gross_exposure"].mean()),
        "maximum_exposure": _finite(curve["gross_exposure"].max()),
        "annual_turnover": _finite(annual_turnover),
        "maximum_target_industry_weight": _finite(max_industry_weight),
        "buffered_selection_ratio": _finite(buffered_selection_ratio),
        "average_size_exposure_z": _finite(average_size_exposure),
        "total_fees": float(curve["cumulative_fees"].iloc[-1]),
        "total_dividends_paid": float(curve["cumulative_dividends"].iloc[-1]),
        "total_cash_interest": float(curve["cumulative_cash_interest"].iloc[-1]),
        "total_delist_writeoff": float(curve["cumulative_delist_writeoff"].iloc[-1]),
        "maximum_stale_positions": int(curve["stale_positions"].max()),
        "filled_trade_count": int(len(result.trades)),
        "rejected_or_cancelled_order_count": rejected,
        "ending_position_count": int(len(result.final_positions)),
    }


def _format_metric(name: str, value: Any) -> str:
    percent_metrics = {
        "total_return",
        "cagr",
        "annual_volatility",
        "max_drawdown",
        "benchmark_total_return",
        "benchmark_cagr",
        "benchmark_annual_volatility",
        "benchmark_max_drawdown",
        "matched_benchmark_total_return",
        "matched_benchmark_cagr",
        "matched_benchmark_annual_volatility",
        "matched_benchmark_max_drawdown",
        "excess_cagr",
        "tracking_error",
        "annual_alpha",
        "positive_day_ratio",
        "worst_day",
        "best_day",
        "average_exposure",
        "maximum_exposure",
        "annual_turnover",
        "maximum_target_industry_weight",
        "buffered_selection_ratio",
    }
    if value is None:
        return "N/A"
    if name in percent_metrics:
        return f"{float(value):.2%}"
    if name in {
        "initial_nav",
        "final_nav",
        "total_fees",
        "total_dividends_paid",
        "total_cash_interest",
        "total_delist_writeoff",
    }:
        return f"{float(value):,.2f}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _performance_image(curve: pd.DataFrame) -> str:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ashare-quant-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = curve.copy().sort_values("date")
    frame["strategy"] = frame["nav"] / frame["nav"].iloc[0]
    frame["benchmark"] = frame["benchmark_nav"] / frame["benchmark_nav"].iloc[0]
    frame["matched"] = (
        frame["benchmark_matched_nav"] / frame["benchmark_matched_nav"].iloc[0]
    )
    frame["drawdown"] = frame["nav"] / frame["nav"].cummax() - 1.0
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    axes[0].plot(frame["date"], frame["strategy"], label="Strategy", linewidth=1.6)
    axes[0].plot(frame["date"], frame["benchmark"], label="Total-return benchmark", linewidth=1.2)
    axes[0].plot(frame["date"], frame["matched"], label="Exposure-matched", linewidth=1.2)
    axes[0].set_ylabel("Growth of 1.0")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].fill_between(frame["date"], frame["drawdown"], 0, color="#c44e52", alpha=0.65)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _monthly_table(curve: pd.DataFrame) -> str:
    values = curve.set_index("date")["nav"].resample("ME").last().pct_change(fill_method=None)
    if values.dropna().empty:
        return "<p>月度数据不足。</p>"
    table = pd.DataFrame({"return": values.dropna()})
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot(index="year", columns="month", values="return")
    pivot = pivot.rename(columns={number: f"{number}月" for number in pivot.columns})
    return pivot.map(lambda value: "" if pd.isna(value) else f"{value:.2%}").to_html(
        classes="monthly", border=0
    )


def _html_report(
    result: BacktestResult,
    config: AppConfig,
    metrics: dict[str, Any],
    image_base64: str,
) -> str:
    labels = {
        "total_return": "累计收益",
        "cagr": "年化收益",
        "annual_volatility": "年化波动",
        "sharpe": "夏普比率",
        "max_drawdown": "最大回撤",
        "benchmark_cagr": "全收益基准年化",
        "benchmark_max_drawdown": "全收益基准回撤",
        "matched_benchmark_cagr": "风险匹配基准年化",
        "matched_benchmark_max_drawdown": "风险匹配基准回撤",
        "excess_cagr": "风险匹配年化超额",
        "information_ratio": "信息比率",
        "annual_turnover": "年化单边换手",
        "maximum_target_industry_weight": "最大目标行业权重",
        "buffered_selection_ratio": "缓冲保留入选占比",
        "average_size_exposure_z": "平均市值风格 Z",
        "total_fees": "累计费用",
        "total_dividends_paid": "累计到账分红",
        "total_delist_writeoff": "累计退市核销",
        "filled_trade_count": "成交笔数",
        "rejected_or_cancelled_order_count": "拒单/撤单",
    }
    metric_rows = "".join(
        f"<tr><th>{labels[key]}</th><td>{_format_metric(key, metrics.get(key))}</td></tr>"
        for key in labels
    )
    warnings = [
        "该结果是研究回测，不构成投资建议，也不保证未来收益。",
        "日线模式在信号日冻结股数，次日开盘按滑点成交，无法还原盘口排队。",
        "现金分红与送股由公司行动账本处理；复杂税务和特殊重组仍需人工复核。",
        "行业与市值中性化作用于选股分数；实际组合暴露请同时检查暴露 CSV。",
    ] + result.warnings
    warning_html = "".join(f"<li>{item}</li>" for item in warnings)
    config_text = yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>A股多因子策略回测报告</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;margin:0;background:#f5f7fa;color:#1f2937}}
main{{max-width:1100px;margin:24px auto;padding:0 18px}} h1{{margin-bottom:4px}} .sub{{color:#64748b;margin-top:0}}
.grid{{display:grid;grid-template-columns:minmax(280px,.8fr) minmax(480px,1.8fr);gap:18px}}
.card{{background:white;border-radius:12px;padding:18px;box-shadow:0 2px 12px #0f172a12;margin-bottom:18px;overflow:auto}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:8px 10px;border-bottom:1px solid #e5e7eb;text-align:right}} th:first-child,td:first-child{{text-align:left}}
.monthly th,.monthly td{{font-size:12px;white-space:nowrap}} img{{width:100%;height:auto}} li{{margin:6px 0}}
pre{{white-space:pre-wrap;background:#0f172a;color:#e2e8f0;padding:14px;border-radius:8px;font-size:12px}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>A股多因子策略回测报告</h1><p class="sub">{metrics['start_date']} 至 {metrics['end_date']} · 历史成分 · 信号日定股数 · 次日开盘执行</p>
<div class="grid"><section class="card"><h2>核心指标</h2><table>{metric_rows}</table></section>
<section class="card"><h2>净值与回撤</h2><img alt="performance" src="data:image/png;base64,{image_base64}"></section></div>
<section class="card"><h2>月度收益</h2>{_monthly_table(result.equity_curve)}</section>
<section class="card"><h2>重要说明</h2><ul>{warning_html}</ul></section>
<section class="card"><h2>运行配置</h2><pre>{config_text}</pre></section>
</main></body></html>"""


def write_report(
    result: BacktestResult,
    config: AppConfig,
    *,
    experiment_type: str = "strict_backtest",
    run_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(config.backtest.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.equity_curve = _with_matched_benchmark(result.equity_curve, config)
    metrics = calculate_metrics(result, config)

    result.equity_curve.to_csv(output / "equity_curve.csv", index=False)
    result.trades.to_csv(output / "trades.csv", index=False)
    result.orders.to_csv(output / "orders.csv", index=False)
    result.selections.to_csv(output / "selections.csv", index=False)
    industry_exposure_table(result.selections).to_csv(
        output / "industry_exposure.csv", index=False
    )
    style_exposure_table(result.selections).to_csv(
        output / "style_exposure.csv", index=False
    )
    result.corporate_events.to_csv(output / "corporate_events.csv", index=False)
    (output / "final_positions.json").write_text(
        json.dumps(result.final_positions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    (output / "runtime.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reproducibility = build_reproducibility_manifest(
        {
            "app_config": config.to_dict(),
            "run_context": dict(run_context or {}),
        },
        data_manifest_path=Path(config.data.cache_dir) / "manifest.json",
    )
    write_json_atomic(reproducibility, output / "reproducibility.json")
    image = _performance_image(result.equity_curve)
    (output / "performance.png").write_bytes(base64.b64decode(image))
    (output / "report.html").write_text(
        _html_report(result, config, metrics, image), encoding="utf-8"
    )
    artifact_paths = [
        output / name
        for name in [
            "equity_curve.csv",
            "trades.csv",
            "orders.csv",
            "selections.csv",
            "industry_exposure.csv",
            "style_exposure.csv",
            "corporate_events.csv",
            "final_positions.json",
            "metrics.json",
            "resolved_config.yaml",
            "runtime.json",
            "reproducibility.json",
            "performance.png",
            "report.html",
        ]
    ]
    registry_path = record_experiment(
        output / "experiment_registry.jsonl",
        reproducibility,
        experiment_type=experiment_type,
        protocol={
            "parameters": "fixed_for_this_run",
            "untouched_holdout_certified": False,
            "run_context": dict(run_context or {}),
            "note": "代码记录运行身份，但是否在看过区间结果后调参必须由研究者披露。",
        },
        artifacts=artifact_paths,
    )
    write_artifact_manifest(output, [*artifact_paths, registry_path])
    return metrics


def console_summary(metrics: dict[str, Any]) -> str:
    keys = [
        ("final_nav", "期末净值"),
        ("cagr", "年化收益"),
        ("max_drawdown", "最大回撤"),
        ("sharpe", "夏普比率"),
        ("matched_benchmark_cagr", "风险匹配基准年化"),
        ("excess_cagr", "风险匹配年化超额"),
        ("annual_turnover", "年化单边换手"),
        ("total_fees", "累计费用"),
    ]
    return "\n".join(f"{label}: {_format_metric(key, metrics.get(key))}" for key, label in keys)
