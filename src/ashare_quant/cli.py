from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .backtest import Backtester
from .config import AppConfig, BacktestConfig
from .data import MarketDataBundle, TushareDownloader, make_demo_bundle
from .report import console_summary, write_report
from .research import write_research_suite
from .provenance import ARTIFACT_MANIFEST_FILENAME, verify_artifact_manifest
from .public_research import (
    PublicDownloadConfig,
    PublicStrategyConfig,
    download_public_history,
    verify_public_cache,
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

    research = subparsers.add_parser(
        "research", help="运行 Alpha 对照、因子消融、成本压力和滚动评估"
    )
    research.add_argument("--config", default="config.yaml", help="YAML 配置文件")
    research.add_argument(
        "--modes",
        default="alpha,ablation,cost,rolling",
        help="逗号分隔：alpha,ablation,cost,rolling",
    )
    research.add_argument("--output", help="研究结果目录，默认在回测输出目录下")
    research.add_argument("--slippage-bps", default="5,10,20")
    research.add_argument("--commission-multipliers", default="1,2")
    research.add_argument("--train-years", type=int, default=5)
    research.add_argument("--test-years", type=int, default=1)

    demo = subparsers.add_parser("demo", help="无需 Token 的离线端到端演示")
    demo.add_argument("--output", default="results/demo", help="演示报告输出目录")
    demo.add_argument("--seed", type=int, default=7)

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
            execution=config.execution,
        )
        _run_backtest(
            make_demo_bundle(seed=args.seed),
            config,
            experiment_type="synthetic_demo",
            run_context={"seed": args.seed, "investable_data": False},
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
            PublicStrategyConfig(start_date=args.start, end_date=args.end),
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
        print(
            f"校验通过: {bundle.bars['symbol'].nunique()} 只股票，"
            f"{len(bundle.bars):,} 行日线，{len(bundle.membership):,} 条成分记录，"
            f"{len(bundle.industry_membership):,} 条行业区间，"
            f"{len(bundle.corporate_actions):,} 条公司行动"
        )
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
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
