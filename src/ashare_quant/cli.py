from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .backtest import Backtester
from .config import AppConfig, BacktestConfig
from .data import MarketDataBundle, TushareDownloader, make_demo_bundle
from .report import console_summary, write_report


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

    demo = subparsers.add_parser("demo", help="无需 Token 的离线端到端演示")
    demo.add_argument("--output", default="results/demo", help="演示报告输出目录")
    demo.add_argument("--seed", type=int, default=7)
    return parser


def _load_config(path_text: str) -> AppConfig:
    path = Path(path_text).resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {path}")
    return AppConfig.from_yaml(path).resolve_paths(path.parent)


def _run_backtest(bundle: MarketDataBundle, config: AppConfig) -> None:
    result = Backtester(bundle, config).run()
    metrics = write_report(result, config)
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
            data=config.data,
            backtest=BacktestConfig(**backtest_values),
            strategy=config.strategy,
            execution=config.execution,
        )
        _run_backtest(make_demo_bundle(seed=args.seed), config)
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
        bundle = MarketDataBundle.from_cache(config.data.cache_dir)
        print(
            f"校验通过: {bundle.bars['symbol'].nunique()} 只股票，"
            f"{len(bundle.bars):,} 行日线，{len(bundle.membership):,} 条成分记录"
        )
        return 0
    if args.command == "all":
        bundle = TushareDownloader(config).download()
    else:
        bundle = MarketDataBundle.from_cache(config.data.cache_dir)
    _run_backtest(bundle, config)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
