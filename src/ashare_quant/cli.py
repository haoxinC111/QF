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
    require_pit_research_eligible,
    verify_pit_cache,
)
from .pit_lake import build_pit_cache_from_archive, verify_archive_pit_cache
from .pit_acceptance import run_pit_acceptance
from .pit_research import write_pit_factor_research
from .pit_shadow import write_pit_shadow_research
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
from .archive import (
    ArchiveConfig,
    TushareCompatibleHttpProvider,
    TaskStateDB,
    default_inventory,
    run_permission_probe,
    run_phase_a_sample,
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

    pit_research = subparsers.add_parser(
        "pit-research",
        help="运行 V2 Alpha2 基本面/估值因子覆盖、IC、分组与滚动研究",
    )
    pit_research.add_argument(
        "--config", default="config.yaml", help="YAML 配置文件"
    )
    pit_research.add_argument(
        "--output",
        default="results/pit_factor_research_v2_alpha2",
        help="必须为空的研究结果目录",
    )
    pit_research.add_argument(
        "--factors",
        default="all",
        help="逗号分隔的 PIT 因子名，默认 all",
    )
    pit_research.add_argument(
        "--horizons", default="21,63", help="逗号分隔的前瞻交易日数"
    )
    pit_research.add_argument("--quantiles", type=int, default=5)
    pit_research.add_argument(
        "--minimum-factors-per-symbol", type=int, default=4
    )
    pit_research.add_argument(
        "--minimum-ic-observations", type=int, default=20
    )
    pit_research.add_argument(
        "--cost-bps", default="5,10,20", help="逗号分隔的单边成本压力"
    )
    pit_research.add_argument("--train-years", type=int, default=5)
    pit_research.add_argument("--test-years", type=int, default=1)
    pit_research.add_argument(
        "--acceptance-report",
        default="results/pit_acceptance_v2_alpha5/acceptance_report.json",
        help="Alpha5 严格验收回执；新归档 PIT 缓存必须提供",
    )

    pit_shadow = subparsers.add_parser(
        "pit-shadow",
        help="运行 V2 Alpha3 PIT 候选的严格账本四臂影子归因",
    )
    pit_shadow.add_argument(
        "--config", default="config.yaml", help="YAML 配置文件"
    )
    pit_shadow.add_argument(
        "--alpha2-research",
        default="results/pit_factor_research_v2_alpha2",
        help="已严格封存且绑定相同数据指纹的 Alpha2 研究目录",
    )
    pit_shadow.add_argument(
        "--output",
        default="results/pit_shadow_v2_alpha3",
        help="必须为空的 Alpha3 影子归因目录",
    )
    pit_shadow.add_argument(
        "--cost-bps",
        default="5,10,20",
        help="逗号分隔的固定滑点压力；PIT 混合权重固定不寻优",
    )
    pit_shadow.add_argument(
        "--acceptance-report",
        default="results/pit_acceptance_v2_alpha5/acceptance_report.json",
        help="与当前 PIT 指纹绑定的 Alpha5 严格验收回执",
    )

    pit_lake_build = subparsers.add_parser(
        "pit-lake-build",
        help="离线将已验收 Bronze 归档转换为严格 PIT 研究缓存",
    )
    pit_lake_build.add_argument(
        "--config", default="config.yaml", help="YAML 配置文件"
    )
    pit_lake_build.add_argument(
        "--archive-root", required=True, help="data_lake 根目录或 fixtures 根目录"
    )
    pit_lake_build.add_argument("--catalog", help="状态库路径；默认根目录 catalog/archive.duckdb")
    pit_lake_build.add_argument("--schema-registry", help="Schema 注册目录")
    pit_lake_build.add_argument("--reports-root", help="批次报告根目录")
    pit_lake_build.add_argument("--output", help="输出 PIT 缓存；默认读取配置")
    pit_lake_build.add_argument(
        "--fixture-mode",
        action="store_true",
        help="显式允许不完整样例，仅用于工程验证且禁止 Alpha 研究",
    )
    pit_lake_build.add_argument("--buckets", type=int, default=32)
    pit_lake_build.add_argument(
        "--force", action="store_true", help="原子替换既有受支持 PIT 缓存"
    )

    pit_lake_verify = subparsers.add_parser(
        "pit-lake-verify",
        help="校验归档 PIT 缓存；可选重放所有 Bronze/Schema/批次 SHA256",
    )
    pit_lake_verify.add_argument(
        "--config", default="config.yaml", help="YAML 配置文件"
    )
    pit_lake_verify.add_argument("--output", help="PIT 缓存；默认读取配置")
    pit_lake_verify.add_argument(
        "--fixture-mode", action="store_true", help="校验工程样例缓存"
    )
    pit_lake_verify.add_argument(
        "--archive-root", help="提供后会重放源 Bronze 与证据 SHA256"
    )
    pit_lake_verify.add_argument("--catalog", help="源状态库路径")
    pit_lake_verify.add_argument("--schema-registry", help="源 Schema 注册目录")
    pit_lake_verify.add_argument("--reports-root", help="源批次报告根目录")

    pit_acceptance = subparsers.add_parser(
        "pit-acceptance",
        help="一键构建或复用 PIT、重放源证据并封存 Alpha5 验收回执",
    )
    pit_acceptance.add_argument(
        "--config", default="config.yaml", help="YAML 配置文件"
    )
    pit_acceptance.add_argument(
        "--archive-root", required=True, help="data_lake 或 fixtures 根目录"
    )
    pit_acceptance.add_argument("--catalog", help="归档状态库路径")
    pit_acceptance.add_argument("--schema-registry", help="Schema 注册目录")
    pit_acceptance.add_argument("--reports-root", help="批次报告根目录")
    pit_acceptance.add_argument("--cache", help="PIT 缓存；默认读取配置")
    pit_acceptance.add_argument(
        "--output",
        default="results/pit_acceptance_v2_alpha5",
        help="必须不存在或为空的封存验收目录",
    )
    pit_acceptance.add_argument(
        "--fixture-mode",
        action="store_true",
        help="只做工程验收，回执固定为 engineering_only",
    )
    pit_acceptance.add_argument("--buckets", type=int, default=32)
    pit_acceptance.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="原子重建既有受支持 PIT 缓存；默认优先复用",
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

    archive_probe = subparsers.add_parser(
        "archive-probe", help="对归档 endpoint 做权限探针并生成 permission_report.json"
    )
    archive_probe.add_argument(
        "--config", default="config.archive.yaml", help="归档 YAML 配置文件"
    )
    archive_probe.add_argument(
        "--priorities", default="P0,P1", help="逗号分隔的优先级过滤"
    )

    archive_sample = subparsers.add_parser(
        "archive-sample", help="执行 Phase A 小样本归档（5日/50股/4季度）"
    )
    archive_sample.add_argument(
        "--config", default="config.archive.yaml", help="归档 YAML 配置文件"
    )
    return parser


def _load_archive_config(path_text: str) -> ArchiveConfig:
    path = Path(path_text).resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到归档配置文件: {path}")
    return ArchiveConfig.from_yaml(path)


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


def _pit_bundle(
    config: AppConfig, *, require_research_eligible: bool = False
) -> PointInTimeDataBundle:
    if not config.point_in_time.enabled:
        raise ValueError(
            "PIT 数据未启用；请在 point_in_time.enabled 设置为 true"
        )
    manifest_path = Path(config.point_in_time.cache_dir) / "manifest.json"
    if require_research_eligible:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"缺少 PIT manifest: {manifest_path}")
        require_pit_research_eligible(
            json.loads(manifest_path.read_text(encoding="utf-8"))
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

    if args.command == "archive-probe":
        archive_config = _load_archive_config(args.config)
        archive_config.validate_for_run(batch_id="A_probe")
        provider = TushareCompatibleHttpProvider(
            url_env=archive_config.provider.base_url_env,
            token_env=archive_config.provider.token_env,
            forbid_token_env=archive_config.provider.forbid_token_env,
            source_provider=archive_config.provider.name,
            allowed_hosts=archive_config.provider.allowed_hosts,
            connect_timeout=archive_config.provider.request.get(
                "connect_timeout_seconds", 10.0
            ),
            read_timeout=archive_config.provider.request.get(
                "read_timeout_seconds", 120.0
            ),
            accept_encoding=archive_config.provider.request.get(
                "accept_encoding", "gzip"
            ),
            follow_cross_host_redirects=archive_config.provider.request.get(
                "follow_cross_host_redirects", False
            ),
            api_key_env=archive_config.provider.api_key_env,
            api_key_header=archive_config.provider.api_key_header,
        )
        inventory = default_inventory()
        priorities = [p.strip() for p in args.priorities.split(",") if p.strip()]
        report = run_permission_probe(archive_config, provider, inventory, priorities)
        # Merge probe results into the endpoint inventory artifact.
        from .archive.probe import probe_results_for_inventory

        inventory_path = archive_config.catalog_dir / "endpoint_inventory.yaml"
        inventory.to_yaml(inventory_path, probe_results=probe_results_for_inventory(report))
        summary = report["summary"]
        print(
            f"权限探针完成: {summary.get('success', 0)} 成功 / "
            f"{summary.get('confirmed_empty', 0)} 空 / "
            f"{summary.get('denied', 0)} 拒绝 / "
            f"{summary.get('invalid_params', 0)} 参数错误 / "
            f"{summary.get('not_found', 0)} 未找到 / "
            f"{summary.get('incompatible', 0)} 不兼容 / "
            f"{summary.get('transient_error', 0)} 瞬时错误，"
            f"报告 {archive_config.reports_dir / 'permission_report.json'}，"
            f"清单 {inventory_path}"
        )
        return 0

    if args.command == "archive-sample":
        archive_config = _load_archive_config(args.config)
        archive_config.validate_for_run(batch_id="A_probe")
        provider = TushareCompatibleHttpProvider(
            url_env=archive_config.provider.base_url_env,
            token_env=archive_config.provider.token_env,
            forbid_token_env=archive_config.provider.forbid_token_env,
            source_provider=archive_config.provider.name,
            allowed_hosts=archive_config.provider.allowed_hosts,
            connect_timeout=archive_config.provider.request.get(
                "connect_timeout_seconds", 10.0
            ),
            read_timeout=archive_config.provider.request.get(
                "read_timeout_seconds", 120.0
            ),
            accept_encoding=archive_config.provider.request.get(
                "accept_encoding", "gzip"
            ),
            follow_cross_host_redirects=archive_config.provider.request.get(
                "follow_cross_host_redirects", False
            ),
            api_key_env=archive_config.provider.api_key_env,
            api_key_header=archive_config.provider.api_key_header,
        )
        inventory = default_inventory()
        db = TaskStateDB(archive_config.catalog_dir / "archive.duckdb")
        summary = run_phase_a_sample(archive_config, provider, inventory, db)
        print(
            f"Phase A 样本完成: {summary['tasks_completed']} 成功 / "
            f"{summary['tasks_failed']} 失败 / {summary['tasks_empty']} 空，"
            f"总行数 {summary['rows_total']}，耗时 {summary['elapsed_seconds']}s"
        )
        return 0

    config = _load_config(args.config)
    if args.command == "pit-lake-build":
        manifest = build_pit_cache_from_archive(
            config,
            args.archive_root,
            catalog_path=args.catalog,
            schema_registry=args.schema_registry,
            reports_root=args.reports_root,
            output_dir=args.output,
            fixture_mode=args.fixture_mode,
            bucket_count=args.buckets,
            overwrite=args.force,
        )
        quality = manifest["data_quality"]
        print(
            "归档 PIT 构建完成: "
            f"模式 {manifest['archive_bridge']['mode']}，"
            f"财报 {quality['fundamental_rows']:,} 行，"
            f"估值 {quality['valuation_rows']:,} 行，"
            f"研究资格 {manifest['research_eligible']}，"
            f"指纹 {manifest['data_fingerprint_sha256']}"
        )
        if manifest["archive_bridge"].get("acceptance_required"):
            print(
                "下一步必须运行 pit-acceptance 生成与该指纹绑定的严格验收回执，"
                "再运行 pit-research / pit-shadow。"
            )
        return 0
    if args.command == "pit-lake-verify":
        verification = verify_archive_pit_cache(
            config,
            args.output or config.point_in_time.cache_dir,
            fixture_mode=args.fixture_mode,
            archive_root=args.archive_root,
            catalog_path=args.catalog,
            schema_registry=args.schema_registry,
            reports_root=args.reports_root,
        )
        print(
            "归档 PIT 校验通过: "
            f"模式 {verification['mode']}，"
            f"任务 {verification['selected_task_count']}，"
            f"源文件复核 {verification['source_files_verified']}，"
            f"研究资格 {verification['research_eligible']}，"
            f"指纹 {verification['data_fingerprint_sha256']}"
        )
        return 0
    if args.command == "pit-acceptance":
        report = run_pit_acceptance(
            config,
            args.archive_root,
            catalog_path=args.catalog,
            schema_registry=args.schema_registry,
            reports_root=args.reports_root,
            cache_dir=args.cache,
            output_dir=args.output,
            fixture_mode=args.fixture_mode,
            bucket_count=args.buckets,
            rebuild_cache=args.rebuild_cache,
        )
        print(
            "Alpha5 PIT 验收完成: "
            f"decision={report['decision']}，"
            f"mode={report['mode']}，"
            f"research_eligible={report['research_eligible']}，"
            f"指纹={report['acceptance_fingerprint_sha256']}，"
            f"报告={Path(args.output).resolve()}"
        )
        return 2 if report["decision"] == "blocked" else 0
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
    if args.command == "pit-research":
        market = MarketDataBundle.from_cache(
            config.data.cache_dir,
            strict=config.data.strict_validation,
            expected_config=config,
        )
        pit_bundle = _pit_bundle(config, require_research_eligible=True)
        factors = [
            value.strip() for value in args.factors.split(",") if value.strip()
        ]
        horizons = [
            int(value.strip())
            for value in args.horizons.split(",")
            if value.strip()
        ]
        costs = [
            float(value.strip())
            for value in args.cost_bps.split(",")
            if value.strip()
        ]
        written = write_pit_factor_research(
            market,
            pit_bundle,
            config,
            args.output,
            factor_names=factors,
            horizons=horizons,
            quantiles=args.quantiles,
            minimum_factors_per_symbol=args.minimum_factors_per_symbol,
            minimum_ic_observations=args.minimum_ic_observations,
            cost_bps=costs,
            train_years=args.train_years,
            test_years=args.test_years,
            acceptance_report=args.acceptance_report,
        )
        print("PIT Alpha2 因子研究完成（生产策略未改变）:")
        for name, path in written.items():
            print(f"  {name}: {path}")
        return 0
    if args.command == "pit-shadow":
        market = MarketDataBundle.from_cache(
            config.data.cache_dir,
            strict=config.data.strict_validation,
            expected_config=config,
        )
        pit_bundle = _pit_bundle(config, require_research_eligible=True)
        costs = [
            float(value.strip())
            for value in args.cost_bps.split(",")
            if value.strip()
        ]
        written = write_pit_shadow_research(
            market,
            pit_bundle,
            config,
            args.output,
            alpha2_research_dir=args.alpha2_research,
            cost_bps=costs,
            acceptance_report=args.acceptance_report,
        )
        print("PIT Alpha3 严格账本影子归因完成（生产策略未改变）:")
        for name, path in written.items():
            print(f"  {name}: {path}")
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
