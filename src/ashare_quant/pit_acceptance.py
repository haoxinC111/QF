"""Alpha5 sealed acceptance gate for archive-backed PIT research data.

The Alpha4 bridge proves that selected source bytes can be replayed.  Alpha5
adds the promotion boundary around that cache: it orchestrates build/reuse,
source replay, point-in-time temporal validation, time-sliced coverage checks,
and a sealed receipt that downstream research must present.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .config import AppConfig
from .pit_data import PointInTimeDataBundle
from .pit_lake import build_pit_cache_from_archive, verify_archive_pit_cache
from .provenance import (
    ARTIFACT_MANIFEST_FILENAME,
    build_reproducibility_manifest,
    payload_sha256,
    verify_artifact_manifest,
    write_artifact_manifest,
    write_json_atomic,
)


PIT_ACCEPTANCE_SCHEMA_VERSION = 1
PIT_ACCEPTANCE_VERSION = "v2_alpha5"
ACCEPTANCE_REPORT_FILENAME = "acceptance_report.json"
ACCEPTANCE_MARKDOWN_FILENAME = "acceptance_report.md"
REPRODUCIBILITY_FILENAME = "reproducibility.json"
RESEARCH_DECISION_PASS = "pass"
RESEARCH_DECISION_ENGINEERING_ONLY = "engineering_only"
RESEARCH_DECISION_BLOCKED = "blocked"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 文件无效: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return payload


def _stage(
    stage_id: str,
    status: str,
    *,
    summary: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"pass", "fail", "skipped"}:
        raise ValueError(f"未知验收阶段状态: {status}")
    return {
        "id": stage_id,
        "status": status,
        "summary": summary,
        "details": dict(details or {}),
    }


def _error_details(exc: Exception) -> dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def _coverage_summary(values: Sequence[float]) -> dict[str, float]:
    series = pd.Series(values, dtype="float64")
    if series.empty:
        raise ValueError("时点覆盖率序列为空")
    return {
        "minimum": float(series.min()),
        "p05": float(series.quantile(0.05)),
        "median": float(series.median()),
        "latest": float(series.iloc[-1]),
    }


def _audit_dates(
    calendar: Iterable[pd.Timestamp | str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(list(calendar))).normalize()
    dates = dates.unique().sort_values()
    dates = dates[(dates >= start) & (dates <= end)]
    if dates.empty:
        raise ValueError("PIT 交易日历没有覆盖回测区间")
    monthly = (
        pd.DataFrame({"date": dates})
        .assign(month=lambda frame: frame["date"].dt.to_period("M"))
        .groupby("month", sort=True)["date"]
        .max()
    )
    result = pd.DatetimeIndex(monthly.to_list()).unique().sort_values()
    if dates[-1] not in result:
        result = result.append(pd.DatetimeIndex([dates[-1]])).unique().sort_values()
    return result


def _active_members_by_date(
    config: AppConfig,
    manifest_symbols: set[str],
    audit_dates: pd.DatetimeIndex,
    *,
    fixture_mode: bool,
) -> tuple[dict[pd.Timestamp, set[str]], dict[str, Any]]:
    if fixture_mode:
        return (
            {pd.Timestamp(date): set(manifest_symbols) for date in audit_dates},
            {
                "mode": "fixture_manifest_symbols",
                "snapshot_count": 1,
                "maximum_snapshot_age_days": 0,
            },
        )

    membership_path = Path(config.data.cache_dir).resolve() / "membership.csv.gz"
    if not membership_path.is_file():
        raise FileNotFoundError(f"缺少基础行情历史成分: {membership_path}")
    membership = pd.read_csv(membership_path)
    required = {"date", "symbol"}
    missing = required.difference(membership.columns)
    if missing:
        raise ValueError(f"历史成分缺少字段: {sorted(missing)}")
    membership = membership[["date", "symbol"]].copy()
    membership["date"] = pd.to_datetime(
        membership["date"], errors="coerce"
    ).dt.normalize()
    membership["symbol"] = membership["symbol"].astype(str)
    if membership.isna().any().any():
        raise ValueError("历史成分包含无效日期或证券")
    membership = membership.loc[
        membership["symbol"].isin(manifest_symbols)
    ].drop_duplicates()
    if membership.empty:
        raise ValueError("历史成分与 PIT 证券全集没有交集")
    snapshots = pd.DatetimeIndex(
        sorted(membership["date"].unique())
    )
    if len(snapshots) > 1:
        maximum_gap = int(
            snapshots.to_series(index=range(len(snapshots))).diff().dt.days.max()
        )
        if maximum_gap > 62:
            raise ValueError(f"历史成分快照存在超过 62 天断档: {maximum_gap} 天")
    else:
        maximum_gap = 0
    groups = {
        pd.Timestamp(date): set(group["symbol"].astype(str))
        for date, group in membership.groupby("date", sort=True)
    }
    active: dict[pd.Timestamp, set[str]] = {}
    maximum_age = 0
    for audit_date in audit_dates:
        location = int(snapshots.searchsorted(audit_date, side="right")) - 1
        if location < 0:
            raise ValueError(f"回测时点之前没有历史成分快照: {audit_date.date()}")
        snapshot_date = pd.Timestamp(snapshots[location])
        age = int((pd.Timestamp(audit_date) - snapshot_date).days)
        maximum_age = max(maximum_age, age)
        if age > 62:
            raise ValueError(
                f"历史成分快照相对验收时点过旧: {audit_date.date()} <- "
                f"{snapshot_date.date()} ({age} 天)"
            )
        symbols = groups[snapshot_date]
        if not symbols:
            raise ValueError(f"历史成分快照为空: {snapshot_date.date()}")
        active[pd.Timestamp(audit_date)] = symbols
    return active, {
        "mode": "historical_membership",
        "snapshot_count": len(snapshots),
        "maximum_snapshot_gap_days": maximum_gap,
        "maximum_snapshot_age_days": maximum_age,
    }


def _available_dates_by_symbol(
    frame: pd.DataFrame,
) -> dict[str, pd.DatetimeIndex]:
    result: dict[str, pd.DatetimeIndex] = {}
    for symbol, group in frame.groupby("symbol", sort=False):
        dates = pd.DatetimeIndex(
            pd.to_datetime(group["available_date"])
        ).normalize()
        result[str(symbol)] = dates.unique().sort_values()
    return result


def _is_recently_visible(
    dates: pd.DatetimeIndex | None,
    when: pd.Timestamp,
    maximum_age_days: int,
) -> bool:
    if dates is None or dates.empty:
        return False
    location = int(dates.searchsorted(when, side="right")) - 1
    if location < 0:
        return False
    return int((when - pd.Timestamp(dates[location])).days) <= maximum_age_days


def audit_pit_time_slices(
    config: AppConfig,
    bundle: PointInTimeDataBundle,
    *,
    fixture_mode: bool,
) -> dict[str, Any]:
    """Audit monthly as-of coverage against the historical active universe."""
    start = pd.Timestamp(config.backtest.start_date).normalize()
    end = pd.Timestamp(config.backtest.end_date).normalize()
    audit_dates = _audit_dates(bundle.calendar, start, end)
    manifest_symbols = set(map(str, bundle.manifest.get("symbols", [])))
    if not manifest_symbols:
        raise ValueError("PIT manifest 证券全集为空")
    active_by_date, membership = _active_members_by_date(
        config,
        manifest_symbols,
        audit_dates,
        fixture_mode=fixture_mode,
    )
    fundamental_dates = _available_dates_by_symbol(bundle.fundamentals)
    valuation_dates = _available_dates_by_symbol(bundle.valuations)
    threshold = float(config.point_in_time.minimum_symbol_coverage)
    rows: list[dict[str, Any]] = []
    for audit_date in audit_dates:
        when = pd.Timestamp(audit_date)
        active = active_by_date[when]
        fundamental_covered = {
            symbol
            for symbol in active
            if _is_recently_visible(
                fundamental_dates.get(symbol),
                when,
                config.point_in_time.maximum_fundamental_age_days,
            )
        }
        valuation_covered = {
            symbol
            for symbol in active
            if _is_recently_visible(
                valuation_dates.get(symbol),
                when,
                config.point_in_time.maximum_valuation_age_days,
            )
        }
        denominator = max(len(active), 1)
        rows.append(
            {
                "date": str(when.date()),
                "active_symbols": len(active),
                "fundamental_symbols": len(fundamental_covered),
                "valuation_symbols": len(valuation_covered),
                "fundamental_coverage": len(fundamental_covered) / denominator,
                "valuation_coverage": len(valuation_covered) / denominator,
                "missing_fundamental_sample": sorted(
                    active - fundamental_covered
                )[:20],
                "missing_valuation_sample": sorted(active - valuation_covered)[:20],
            }
        )
    fundamental_summary = _coverage_summary(
        [float(row["fundamental_coverage"]) for row in rows]
    )
    valuation_summary = _coverage_summary(
        [float(row["valuation_coverage"]) for row in rows]
    )
    below = [
        row["date"]
        for row in rows
        if row["fundamental_coverage"] < threshold
        or row["valuation_coverage"] < threshold
    ]
    manifest_end = pd.Timestamp(bundle.manifest["requested_end"])
    if bundle.valuations["date"].max() > manifest_end:
        raise ValueError("估值数据超过 manifest 请求结束日")
    if bundle.fundamentals["announcement_date"].max() > manifest_end:
        raise ValueError("财报公告数据超过 manifest 请求结束日")
    return {
        "passed": not below,
        "minimum_required_coverage": threshold,
        "audit_frequency": "last_trading_day_of_month",
        "audit_date_count": len(rows),
        "membership": membership,
        "fundamental": fundamental_summary,
        "valuation": valuation_summary,
        "dates_below_threshold": below,
        "time_slices": rows,
        "temporal_contract": {
            "strict_bundle_validation": True,
            "announcement_lag_verified": True,
            "valuation_lag_verified": True,
            "revision_sequences_verified": True,
            "duplicate_valuation_keys_rejected": True,
        },
    }


def _pit_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    bridge = manifest.get("archive_bridge", {})
    if not isinstance(bridge, Mapping):
        bridge = {}
    return {
        "data_fingerprint_sha256": manifest.get("data_fingerprint_sha256"),
        "base_data_fingerprint_sha256": manifest.get(
            "base_data_fingerprint_sha256"
        ),
        "selected_task_set_sha256": bridge.get("selected_task_set_sha256"),
        "archive_bridge_schema_version": bridge.get("schema_version"),
        "acceptance_required": bool(bridge.get("acceptance_required")),
        "batch_snapshots": dict(bridge.get("batch_snapshots", {})),
    }


def _acceptance_identity(
    *,
    mode: str,
    decision: str,
    pit_identity: Mapping[str, Any],
    stages: Sequence[Mapping[str, Any]],
    temporal_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    temporal_summary = None
    if temporal_audit is not None:
        temporal_summary = {
            "passed": temporal_audit.get("passed"),
            "minimum_required_coverage": temporal_audit.get(
                "minimum_required_coverage"
            ),
            "audit_date_count": temporal_audit.get("audit_date_count"),
            "fundamental": temporal_audit.get("fundamental"),
            "valuation": temporal_audit.get("valuation"),
            "dates_below_threshold": temporal_audit.get(
                "dates_below_threshold"
            ),
        }
    return {
        "schema_version": PIT_ACCEPTANCE_SCHEMA_VERSION,
        "acceptance_version": PIT_ACCEPTANCE_VERSION,
        "mode": mode,
        "decision": decision,
        "pit_identity": dict(pit_identity),
        "stage_status": [
            {"id": item.get("id"), "status": item.get("status")}
            for item in stages
        ],
        "temporal_summary": temporal_summary,
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# V2 Alpha5 PIT 验收报告",
        "",
        f"- decision: `{report['decision']}`",
        f"- mode: `{report['mode']}`",
        f"- research eligible: `{report['research_eligible']}`",
        f"- acceptance fingerprint: `{report['acceptance_fingerprint_sha256']}`",
        "",
        "## 阶段结果",
        "",
        "| 阶段 | 状态 | 摘要 |",
        "|---|---|---|",
    ]
    for stage in report["stages"]:
        summary = str(stage["summary"]).replace("|", "\\|")
        lines.append(f"| {stage['id']} | {stage['status']} | {summary} |")
    temporal = report.get("temporal_audit")
    if isinstance(temporal, Mapping):
        lines += [
            "",
            "## 时点覆盖",
            "",
            f"- 验收时点: {temporal['audit_date_count']}",
            f"- 最低要求: {float(temporal['minimum_required_coverage']):.2%}",
            f"- 财报最低覆盖: {float(temporal['fundamental']['minimum']):.2%}",
            f"- 估值最低覆盖: {float(temporal['valuation']['minimum']):.2%}",
            f"- 未通过时点: {len(temporal['dates_below_threshold'])}",
        ]
    lines += [
        "",
        "## 结论",
        "",
        str(report["conclusion"]),
        "",
    ]
    return "\n".join(lines)


def _publish_report(staging: Path, output: Path) -> None:
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"验收输出目录必须为空: {output}")
        output.rmdir()
    os.replace(staging, output)


def run_pit_acceptance(
    config: AppConfig,
    archive_root: str | Path,
    *,
    catalog_path: str | Path | None = None,
    schema_registry: str | Path | None = None,
    reports_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    output_dir: str | Path = "results/pit_acceptance_v2_alpha5",
    fixture_mode: bool = False,
    bucket_count: int = 32,
    rebuild_cache: bool = False,
) -> dict[str, Any]:
    """Build/reuse, verify, audit, and seal an Alpha5 acceptance receipt."""
    config.validate()
    archive = Path(archive_root).resolve()
    cache = (
        Path(cache_dir).resolve()
        if cache_dir
        else Path(config.point_in_time.cache_dir).resolve()
    )
    output = Path(output_dir).resolve()
    if output == cache or output in cache.parents or cache in output.parents:
        raise ValueError("验收报告目录与 PIT 缓存目录不能互相嵌套")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"验收输出目录必须不存在或为空: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent)
    ).resolve()
    mode = "fixture" if fixture_mode else "strict"
    stages: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {}
    verification: dict[str, Any] | None = None
    temporal_audit: dict[str, Any] | None = None

    try:
        try:
            manifest_path = cache / "manifest.json"
            if rebuild_cache or not manifest_path.is_file():
                manifest = build_pit_cache_from_archive(
                    config,
                    archive,
                    catalog_path=catalog_path,
                    schema_registry=schema_registry,
                    reports_root=reports_root,
                    output_dir=cache,
                    fixture_mode=fixture_mode,
                    bucket_count=bucket_count,
                    overwrite=rebuild_cache,
                )
                action = "rebuilt" if rebuild_cache else "built"
            else:
                manifest = _read_json(manifest_path)
                action = "reused"
            stages.append(
                _stage(
                    "cache_build",
                    "pass",
                    summary=f"PIT cache {action}",
                    details={
                        "action": action,
                        "data_fingerprint_sha256": manifest.get(
                            "data_fingerprint_sha256"
                        ),
                    },
                )
            )
        except Exception as exc:
            stages.append(
                _stage(
                    "cache_build",
                    "fail",
                    summary="PIT 缓存构建或复用失败",
                    details=_error_details(exc),
                )
            )

        if stages[-1]["status"] == "pass":
            try:
                verification = verify_archive_pit_cache(
                    config,
                    cache,
                    fixture_mode=fixture_mode,
                    archive_root=archive,
                    catalog_path=catalog_path,
                    schema_registry=schema_registry,
                    reports_root=reports_root,
                )
                if (
                    verification["source_files_verified"]
                    != verification["selected_task_count"]
                ):
                    raise ValueError("源文件复核数量与选中任务数不一致")
                stages.append(
                    _stage(
                        "source_replay",
                        "pass",
                        summary="Bronze/Schema/批次证据完整重放通过",
                        details=verification,
                    )
                )
            except Exception as exc:
                stages.append(
                    _stage(
                        "source_replay",
                        "fail",
                        summary="源证据重放失败",
                        details=_error_details(exc),
                    )
                )
        else:
            stages.append(
                _stage(
                    "source_replay",
                    "skipped",
                    summary="缓存阶段失败，未执行源证据重放",
                )
            )

        if stages[-1]["status"] == "pass":
            try:
                base_manifest = (
                    None
                    if fixture_mode
                    else Path(config.data.cache_dir).resolve() / "manifest.json"
                )
                bundle = PointInTimeDataBundle.from_cache(
                    cache,
                    strict=True,
                    expected_config=config,
                    base_manifest_path=base_manifest,
                )
                temporal_audit = audit_pit_time_slices(
                    config, bundle, fixture_mode=fixture_mode
                )
                if not temporal_audit["passed"]:
                    raise ValueError(
                        "逐月时点覆盖不足: "
                        + ", ".join(temporal_audit["dates_below_threshold"][:10])
                    )
                stages.append(
                    _stage(
                        "temporal_coverage",
                        "pass",
                        summary="PIT 时点规则与逐月活跃证券覆盖通过",
                        details={
                            "audit_date_count": temporal_audit[
                                "audit_date_count"
                            ],
                            "fundamental": temporal_audit["fundamental"],
                            "valuation": temporal_audit["valuation"],
                        },
                    )
                )
            except Exception as exc:
                stages.append(
                    _stage(
                        "temporal_coverage",
                        "fail",
                        summary="PIT 时点或逐月覆盖验收失败",
                        details=_error_details(exc),
                    )
                )
        else:
            stages.append(
                _stage(
                    "temporal_coverage",
                    "skipped",
                    summary="源证据阶段失败，未执行时点覆盖验收",
                )
            )

        required_passed = all(item["status"] == "pass" for item in stages)
        research_eligible = bool(manifest.get("research_eligible"))
        if required_passed and fixture_mode and not research_eligible:
            decision = RESEARCH_DECISION_ENGINEERING_ONLY
            conclusion = "工程验收通过；fixture 明确禁止生成 Alpha 证据。"
        elif required_passed and not fixture_mode and research_eligible:
            decision = RESEARCH_DECISION_PASS
            conclusion = "严格验收通过；该回执允许绑定同一 PIT 指纹的研究运行。"
        else:
            decision = RESEARCH_DECISION_BLOCKED
            conclusion = "验收存在阻塞项；不得运行或晋级真实 PIT Alpha 研究。"
        pit_identity = _pit_identity(manifest)
        identity = _acceptance_identity(
            mode=mode,
            decision=decision,
            pit_identity=pit_identity,
            stages=stages,
            temporal_audit=temporal_audit,
        )
        report = {
            "schema_version": PIT_ACCEPTANCE_SCHEMA_VERSION,
            "acceptance_version": PIT_ACCEPTANCE_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "mode": mode,
            "decision": decision,
            "research_eligible": research_eligible,
            "pit_identity": pit_identity,
            "stages": stages,
            "source_verification": verification,
            "temporal_audit": temporal_audit,
            "acceptance_identity": identity,
            "acceptance_fingerprint_sha256": payload_sha256(identity),
            "conclusion": conclusion,
        }
        report_path = write_json_atomic(
            report, staging / ACCEPTANCE_REPORT_FILENAME
        )
        markdown_path = staging / ACCEPTANCE_MARKDOWN_FILENAME
        markdown_path.write_text(_markdown_report(report), encoding="utf-8")
        reproducibility = build_reproducibility_manifest(
            asdict(config),
            data_manifest_path=(cache / "manifest.json")
            if (cache / "manifest.json").is_file()
            else None,
        )
        reproducibility_path = write_json_atomic(
            reproducibility, staging / REPRODUCIBILITY_FILENAME
        )
        write_artifact_manifest(
            staging, [report_path, markdown_path, reproducibility_path]
        )
        verify_artifact_manifest(
            staging / ARTIFACT_MANIFEST_FILENAME, strict=True
        )
        _publish_report(staging, output)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_pit_acceptance_receipt(
    report_path: str | Path,
    pit_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a sealed pass receipt and bind it to one exact PIT cache."""
    target = Path(report_path).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"缺少 Alpha5 验收回执: {target}")
    verify_artifact_manifest(
        target.parent / ARTIFACT_MANIFEST_FILENAME, strict=True
    )
    report = _read_json(target)
    identity = report.get("acceptance_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Alpha5 验收回执缺少确定性身份")
    if (
        report.get("schema_version") != PIT_ACCEPTANCE_SCHEMA_VERSION
        or report.get("acceptance_version") != PIT_ACCEPTANCE_VERSION
        or report.get("decision") != RESEARCH_DECISION_PASS
        or report.get("mode") != "strict"
        or not report.get("research_eligible")
        or payload_sha256(identity)
        != report.get("acceptance_fingerprint_sha256")
        or identity.get("decision") != report.get("decision")
        or identity.get("mode") != report.get("mode")
        or identity.get("pit_identity") != report.get("pit_identity")
    ):
        raise ValueError("Alpha5 验收回执身份、模式或决策无效")
    expected = _pit_identity(pit_manifest)
    if report.get("pit_identity") != expected:
        raise ValueError("Alpha5 验收回执与当前 PIT 数据指纹不一致")
    return report


def require_pit_acceptance(
    pit_manifest: Mapping[str, Any],
    report_path: str | Path | None,
) -> dict[str, Any] | None:
    bridge = pit_manifest.get("archive_bridge", {})
    if not isinstance(bridge, Mapping):
        return None
    version = int(bridge.get("schema_version", 0))
    required = bridge.get("mode") == "strict" and version >= 1
    if not required:
        return None
    if version >= 2 and not bridge.get("acceptance_required"):
        raise ValueError("严格归档 PIT 的 Alpha5 验收门禁标志无效")
    if report_path is None:
        raise ValueError("该严格归档 PIT 缓存要求 Alpha5 验收回执")
    return verify_pit_acceptance_receipt(report_path, pit_manifest)


__all__ = [
    "ACCEPTANCE_REPORT_FILENAME",
    "PIT_ACCEPTANCE_SCHEMA_VERSION",
    "PIT_ACCEPTANCE_VERSION",
    "audit_pit_time_slices",
    "require_pit_acceptance",
    "run_pit_acceptance",
    "verify_pit_acceptance_receipt",
]
