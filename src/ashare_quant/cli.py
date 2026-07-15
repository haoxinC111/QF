from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .backtest import Backtester
from .config import AppConfig, BacktestConfig
from .data import MarketDataBundle, TushareDownloader, make_demo_bundle
from .execution import SUPPORTED_EXECUTION_MODELS
from .portfolio import SUPPORTED_PORTFOLIO_MODELS
from .pit_data import (
    PointInTimeDataBundle,
    TusharePointInTimeDownloader,
    verify_pit_cache,
)
from .report import console_summary, write_report
from .research import write_research_suite
from .provenance import (
    ARTIFACT_MANIFEST_FILENAME,
    sha256_file,
    verify_artifact_manifest,
)
from .public_research import (
    PublicDownloadConfig,
    PublicStrategyConfig,
    download_public_history,
    verify_public_cache,
    write_public_implementation_research,
    write_public_research,
    write_public_robustness,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ashare-quant", description="A股动态股票池多因子研究回测"
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in [
        ("download", "下载并缓存 Tushare 数据"),
        ("backtest", "用已有缓存运行回测"),
        ("all", "下载数据后运行回测"),
        ("validate-data", "校验已有缓存的数据结构"),
    ]:
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--config", default="config.yaml", help="YAML 配置文件")

    pit_download = subparsers.add_parser(
        "pit-download", help="下载并封存 V2 财报/估值时点数据侧车"
    )
    pit_download.add_argument(
        "--config", default="config.yaml", help="YAML 配置文件"
    )

    pit_verify = subparsers.add_parser(
        "pit-verify", help="校验 PIT 缓存、基础行情身份与证券覆盖率"
    )
    pit_verify.add_argument(
        "--config", default="config.yaml", help="YAML 配置文件"
    )

    pit_snapshot = subparsers.add_parser(
        "pit-snapshot", help="导出指定日期真正可见的财报/估值快照"
    )
    pit_snapshot.add_argument(
        "--config", default="config.yaml", help="YAML 配置文件"
    )
    pit_snapshot.add_argument("--date", required=True, help="快照日期 YYYY-MM-DD")
    pit_snapshot.add_argument(
        "--symbols", help="可选，逗号分隔的证券代码（例如 600000.SH,000001.SZ）"
    )
    pit_snapshot.add_argument("--output", help="可选 CSV 输出路径")
    pit_snapshot.add_argument(
        "--force", action="store_true", help="允许覆盖既有快照及其元数据"
    )

    research = subparsers.add_parser(
        "research", help="运行 Alpha、组合/成交归因、成本压力和滚动评估"
    )
    research.add_argument("--config", default="config.yaml", help="YAML 配置文件")
    research.add_argument(
        "--modes",
        default="alpha,ablation,cost,rolling,implementation",
        help="逗号分隔：alpha,ablation,cost,rolling,implementation",
    )
    research.add_argument("--output", help="研究结果目录，默认在回测输出目录下")
    research.add_argument("--slippage-bps", default="5,10,20")
    research.add_argument("--commission-multipliers", default="1,2")
    research.add_argument("--train-years", type=int, default=5)
    research.add_argument("--test-years", type=int, default=1)

    demo = subparsers.add_parser("demo", help="无需 Token 的离线端到端演示")
    demo.add_argument("--output", default="results/demo", help="演示报告输出目录")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument(
        "--portfolio-model",
        choices=sorted(SUPPORTED_PORTFOLIO_MODELS),
        help="覆盖组合模型；省略时使用生产基线",
    )
    demo.add_argument(
        "--execution-model",
        choices=sorted(SUPPORTED_EXECUTION_MODELS),
        help="覆盖成交模型；省略时使用生产基线",
    )

    public_download = subparsers.add_parser(
        "public-download", help="从公开 HTTPS 接口下载历史沪深300成分日线"
    )
    public_download.add_argument("--membership", required=True, help="csi300.csv 历史成分文件")
    public_download.add_argument("--cache", default="data/public_eastmoney")
    public_download.add_argument("--start", default="2012-01-01")
    public_download.add_argument("--end", default="2025-12-31")
    public_download.add_argument("--workers", type=int, default=6)
    public_download.add_argument("--source", choices=["sina", "eastmoney"], default="sina")
    public_download.add_argument("--force", action="store_true")

    public_research = subparsers.add_parser(
        "public-research", help="在公开历史成分与行情上运行分期研究"
    )
    public_research.add_argument("--membership", required=True, help="csi300.csv 历史成分文件")
    public_research.add_argument("--cache", default="data/public_eastmoney")
    public_research.add_argument("--output", default="results/public_research")
    public_research.add_argument("--start", default="2013-01-01")
    public_research.add_argument("--end", default="2025-12-31")

    public_robustness = subparsers.add_parser(
        "public-robustness", help="对公开数据 v1.5 Alpha 运行成本压力与因子消融"
    )
    public_robustness.add_argument("--membership", required=True, help="csi300.csv 历史成分文件")
    public_robustness.add_argument("--cache", default="data/public_eastmoney")
    public_robustness.add_argument("--output", default="results/public_research/robustness")
    public_robustness.add_argument("--start", default="2013-01-01")
    public_robustness.add_argument("--end", default="2025-12-31")

    public_implementation = subparsers.add_parser(
        "public-implementation",
        help="在公开缓存上运行 v1.6 组合/成交权重级四臂对照",
    )
    public_implementation.add_argument(
        "--membership",
        required=True,
        help="csi300.csv 历史成分文件",
    )
    public_implementation.add_argument(
        "--cache",
        default="data/public_eastmoney",
    )
    public_implementation.add_argument(
        "--output",
        default="results/public_implementation_v1_6",
    )
    public_implementation.add_argument("--start", default="2013-01-01")
    public_implementation.add_argument("--end", default="2025-12-31")
    public_implementation.add_argument(
        "--initial-capital",
        type=float,
        default=1_000_000.0,
        help="平方根冲击参与率换算的初始资金，默认 100 万元",
    )

    public_verify = subparsers.add_parser(
        "public-verify", help="验证公开行情缓存和历史成分文件的 SHA256 指纹"
    )
    public_verify.add_argument("--membership", required=True, help="csi300.csv 历史成分文件")
    public_verify.add_argument("--cache", default="data/public_eastmoney")
    public_verify.add_argument(
        "--seal-legacy",
        action="store_true",
        help="为 v1.3 旧缓存首次生成 v1.4 指纹，不重新下载行情",
    )

    result_verify = subparsers.add_parser(
        "result-verify", help="验证回测或研究输出的 SHA256 指纹"
    )
    result_verify.add_argument("--output", required=True, help="结果输出目录")
    result_verify.add_argument(
        "--strict",
        action="store_true",
        help="同时拒绝 artifact_manifest.json 未登记的额外文件",
    )
    return parser


def _load_config(path_text: str) -> AppConfig:
    path = Path(path_text).resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {path}")
    return AppConfig.from_yaml(path).resolve_paths(path.parent)


def _run_backtest(
    bundle: MarketDataBundle,
    config: AppConfig,
    *,
    experiment_type: str = "strict_backtest",
    run_context: dict[str, object] | None = None,
) -> None:
    result = Backtester(bundle, config).run()
    metrics = write_report(
        result,
        config,
        experiment_type=experiment_type,
        run_context=run_context,
    )
    print(console_summary(metrics))
    print(f"完整报告: {Path(config.backtest.output_dir) / 'report.html'}")


def _pit_bundle(config: AppConfig) -> PointInTimeDataBundle:
    if not config.point_in_time.enabled:
        raise ValueError(
            "PIT 数据未启用；请在 point_in_time.enabled 设置为 true"
        )
    return PointInTimeDataBundle.from_cache(
        config.point_in_time.cache_dir,
        strict=True,
        expected_config=config,
        base_manifest_path=Path(config.data.cache_dir) / "manifest.json",
    )


def _write_snapshot(
    snapshot: pd.DataFrame,
    output_text: str,
    *,
    force: bool,
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    # Kept local to the CLI because this is a derived inspection artifact, not
    # part of the immutable PIT cache itself.
    output = Path(output_text).resolve()
    metadata_path = output.with_name(output.name + ".manifest.json")
    if not force and (output.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"快照输出已存在，拒绝覆盖；如确认请加 --force: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    snapshot.to_csv(temporary, index=False)
    os.replace(temporary, output)
    payload = {
        **metadata,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_file": output.name,
        "snapshot_rows": len(snapshot),
        "snapshot_sha256": sha256_file(output),
    }
    metadata_temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    metadata_temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(metadata_temporary, metadata_path)
    return output, metadata_path


def _dispatch(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.command == "demo":
        config = AppConfig()
        backtest_values = asdict(config.backtest)
        backtest_values.update(
            {
                "start_date": "2018-01-01",
                "end_date": "2025-12-31",
                "output_dir": str(Path(args.output).resolve()),
            }
        )
        config = AppConfig(
            data=replace(config.data, provider="synthetic_demo"),
            backtest=BacktestConfig(**backtest_values),
            strategy=config.strategy,
            portfolio=replace(
                config.portfolio,
                construction_model=(
                    args.portfolio_model
                    or config.portfolio.construction_model
                ),
            ),
            execution=replace(
                config.execution,
                market_impact_model=(
                    args.execution_model
                    or config.execution.market_impact_model
                ),
            ),
        )
        _run_backtest(
            make_demo_bundle(seed=args.seed),
            config,
            experiment_type="synthetic_demo",
            run_context={
                "seed": args.seed,
                "investable_data": False,
                "portfolio_model_override": args.portfolio_model,
                "execution_model_override": args.execution_model,
            },
        )
        return 0

    if args.command == "public-download":
        manifest = download_public_history(
            args.membership,
            args.cache,
            PublicDownloadConfig(
                start_date=args.start,
                end_date=args.end,
                source=args.source,
                workers=args.workers,
            ),
            force=args.force,
        )
        print(
            f"公开数据完成: {manifest['available_count']}/{manifest['requested_count']} 个代码，"
            f"失败 {len(manifest['failed'])} 个"
        )
        return 0

    if args.command == "public-research":
        written = write_public_research(
            args.membership,
            args.cache,
            args.output,
            PublicStrategyConfig(
                start_date=args.start,
                end_date=args.end,
            ),
        )
        print("公开数据研究完成:")
        for name, path in written.items():
            print(f"  {name}: {path}")
        return 0

    if args.command == "public-robustness":
        written = write_public_robustness(
            args.membership,
            args.cache,
            args.output,
            PublicStrategyConfig(start_date=args.start, end_date=args.end),
        )
        print("公开数据稳健性检查完成:")
        for name, path in written.items():
            print(f"  {name}: {path}")
        return 0

    if args.command == "public-implementation":
        written = write_public_implementation_research(
            args.membership,
            args.cache,
            args.output,
            PublicStrategyConfig(
                start_date=args.start,
                end_date=args.end,
                initial_capital=args.initial_capital,
            ),
        )
        print("公开数据 v1.6 组合/成交四臂对照完成:")
        for name, path in written.items():
            print(f"  {name}: {path}")
        return 0

    if args.command == "public-verify":
        manifest = verify_public_cache(
            args.membership,
            args.cache,
            seal_legacy=args.seal_legacy,
        )
        verification = manifest.get("verification", {"verified": True})
        print(
            f"公开缓存校验通过: {manifest.get('available_count', 0)} 个文件，"
            f"指纹 {manifest.get('data_fingerprint_sha256', '')}，"
            f"状态 {verification.get('verified', True)}"
        )
        return 0

    if args.command == "result-verify":
        manifest_path = Path(args.output).resolve() / ARTIFACT_MANIFEST_FILENAME
        verification = verify_artifact_manifest(manifest_path, strict=args.strict)
        print(
            f"结果校验通过: {verification['file_count']} 个文件，"
            f"集合指纹 {verification['artifact_set_sha256']}，"
            f"未封存 {len(verification['unsealed_paths'])} 个"
        )
        return 0

    config = _load_config(args.config)
    if args.command == "pit-download":
        bundle = TusharePointInTimeDownloader(config).download()
        quality = bundle.manifest["data_quality"]
        print(
            "PIT 数据完成: "
            f"财报 {quality['fundamental_rows']:,} 行/"
            f"{quality['fundamental_symbols']} 只，"
            f"估值 {quality['valuation_rows']:,} 行/"
            f"{quality['valuation_symbols']} 只，"
            f"指纹 {bundle.manifest['data_fingerprint_sha256']}"
        )
        return 0
    if args.command == "pit-verify":
        if not config.point_in_time.enabled:
            raise ValueError(
                "PIT 数据未启用；请在 point_in_time.enabled 设置为 true"
            )
        manifest = verify_pit_cache(
            config.point_in_time.cache_dir,
            expected_config=config,
            base_manifest_path=Path(config.data.cache_dir) / "manifest.json",
        )
        quality = manifest["data_quality"]
        print(
            "PIT 缓存校验通过: "
            f"财报覆盖 {quality['fundamental_symbol_coverage']:.2%}，"
            f"估值覆盖 {quality['valuation_symbol_coverage']:.2%}，"
            f"指纹 {manifest['data_fingerprint_sha256']}"
        )
        return 0
    if args.command == "pit-snapshot":
        bundle = _pit_bundle(config)
        symbols = (
            [value.strip() for value in args.symbols.split(",") if value.strip()]
            if args.symbols
            else None
        )
        snapshot = bundle.snapshot(
            args.date,
            symbols=symbols,
            maximum_fundamental_age_days=(
                config.point_in_time.maximum_fundamental_age_days
            ),
            maximum_valuation_age_days=(
                config.point_in_time.maximum_valuation_age_days
            ),
        )
        if args.output:
            output, metadata_path = _write_snapshot(
                snapshot,
                args.output,
                force=args.force,
                metadata={
                    "schema_version": 1,
                    "as_of_date": str(args.date),
                    "requested_symbols": symbols,
                    "pit_data_fingerprint_sha256": bundle.manifest.get(
                        "data_fingerprint_sha256"
                    ),
                    "base_data_fingerprint_sha256": bundle.manifest.get(
                        "base_data_fingerprint_sha256"
                    ),
                },
            )
            print(
                f"PIT 快照完成: {len(snapshot)} 行，{output}，"
                f"SHA256 {sha256_file(output)}，元数据 {metadata_path}"
            )
        else:
            print(snapshot.head(20).to_string(index=False))
            print(f"PIT 快照: {len(snapshot)} 行（终端最多显示 20 行）")
        return 0
    if args.command == "download":
        bundle = TushareDownloader(config).download()
        print(
            f"数据完成: {bundle.bars['symbol'].nunique()} 只股票，"
            f"{len(bundle.bars):,} 行日线"
        )
        return 0
    if args.command == "validate-data":
        bundle = MarketDataBundle.from_cache(
            config.data.cache_dir,
            strict=config.data.strict_validation,
            expected_config=config,
        )
        summary = (
            f"校验通过: {bundle.bars['symbol'].nunique()} 只股票，"
            f"{len(bundle.bars):,} 行日线，{len(bundle.membership):,} 条成分记录，"
            f"{len(bundle.industry_membership):,} 条行业区间，"
            f"{len(bundle.corporate_actions):,} 条公司行动"
        )
        if config.point_in_time.enabled:
            pit_bundle = _pit_bundle(config)
            quality = pit_bundle.manifest["data_quality"]
            summary += (
                "；PIT 财报/估值覆盖 "
                f"{quality['fundamental_symbol_coverage']:.2%}/"
                f"{quality['valuation_symbol_coverage']:.2%}"
            )
        print(summary)
        return 0
    if args.command == "research":
        bundle = MarketDataBundle.from_cache(
            config.data.cache_dir,
            strict=config.data.strict_validation,
            expected_config=config,
        )
        output = (
            Path(args.output).resolve()
            if args.output
            else Path(config.backtest.output_dir) / "research"
        )
        modes = [value.strip() for value in args.modes.split(",") if value.strip()]
        slippages = [
            float(value.strip())
            for value in args.slippage_bps.split(",")
            if value.strip()
        ]
        commission_multipliers = [
            float(value.strip())
            for value in args.commission_multipliers.split(",")
            if value.strip()
        ]
        written = write_research_suite(
            bundle,
            config,
            output,
            modes=modes,
            slippage_bps=slippages,
            commission_multipliers=commission_multipliers,
            train_years=args.train_years,
            test_years=args.test_years,
        )
        print("研究完成:")
        for name, path in written.items():
            print(f"  {name}: {path}")
        return 0
    if args.command == "all":
        bundle = TushareDownloader(config).download()
    else:
        bundle = MarketDataBundle.from_cache(
            config.data.cache_dir,
            strict=config.data.strict_validation,
            expected_config=config,
        )
    _run_backtest(bundle, config)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
