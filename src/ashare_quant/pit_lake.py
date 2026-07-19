"""Offline Bronze data-lake to strict Point-in-Time cache bridge.

The archive downloader and the PIT research layer intentionally have separate
contracts.  This module is the audited boundary between them: it reads the
archive catalog in read-only mode, binds successful partitions to batch
decisions and checksums, validates registered schemas, and emits the existing
PIT v1 cache format without making any network request.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import AppConfig
from .pit_data import (
    CANONICAL_STATEMENT_REPORT_TYPE,
    FUNDAMENTAL_COLUMNS,
    PIT_CACHE_SCHEMA_VERSION,
    VALUATION_COLUMNS,
    PointInTimeDataBundle,
    _atomic_csv,
    _atomic_json,
    _clear_known_cache,
    _verify_base_market_cache,
    normalize_fundamental_source,
    normalize_valuations,
)
from .provenance import (
    build_file_inventory,
    inventory_sha256,
    payload_sha256,
    sha256_file,
)


ARCHIVE_PIT_BRIDGE_VERSION = 2
SUPPORTED_ARCHIVE_PIT_BRIDGE_VERSIONS = {1, ARCHIVE_PIT_BRIDGE_VERSION}
ARCHIVE_LINEAGE_FILENAME = "archive_lineage.json.gz"

STRICT_BATCHES = ("B0_reference", "B1_market", "B3_financial")
STRICT_ENDPOINT_BATCH = {
    "trade_cal": "B0_reference",
    "daily_basic": "B1_market",
    "income_vip": "B3_financial",
    "balancesheet_vip": "B3_financial",
    "cashflow_vip": "B3_financial",
    "fina_indicator_vip": "B3_financial",
}
FUNDAMENTAL_ALIASES = {
    "income": ("income_vip", "income"),
    "balancesheet": ("balancesheet_vip", "balancesheet"),
    "cashflow": ("cashflow_vip", "cashflow"),
    "fina_indicator": ("fina_indicator_vip", "fina_indicator"),
}
TERMINAL_STATUSES = {"success", "confirmed_empty"}

_CATALOG_COLUMNS = {
    "task_id",
    "api_name",
    "params_json",
    "status",
    "row_count",
    "bronze_path",
    "schema_fingerprint",
    "raw_sha256",
}

_STRING_FUNDAMENTAL_COLUMNS = {
    "symbol",
    "source",
    "metric",
    "unit",
    "report_type",
    "company_type",
    "source_update_flag",
    "source_row_sha256",
}
_DATE_FUNDAMENTAL_COLUMNS = {
    "period_end",
    "announcement_date",
    "available_date",
}
_DATE_VALUATION_COLUMNS = {"date", "available_date"}

_FUNDAMENTAL_ARROW_SCHEMA = pa.schema(
    [
        pa.field(
            column,
            (
                pa.string()
                if column in _STRING_FUNDAMENTAL_COLUMNS
                else pa.timestamp("ns")
                if column in _DATE_FUNDAMENTAL_COLUMNS
                else pa.int64()
                if column == "revision_sequence"
                else pa.float64()
            ),
        )
        for column in FUNDAMENTAL_COLUMNS
    ]
)
_VALUATION_ARROW_SCHEMA = pa.schema(
    [
        pa.field(
            column,
            pa.string()
            if column == "symbol"
            else pa.timestamp("ns")
            if column in _DATE_VALUATION_COLUMNS
            else pa.float64(),
        )
        for column in VALUATION_COLUMNS
    ]
)


@dataclass(frozen=True)
class ArchiveTaskRecord:
    task_id: str
    api_name: str
    params: dict[str, Any]
    status: str
    row_count: int
    bronze_path: str
    schema_fingerprint: str
    raw_sha256: str


@dataclass(frozen=True)
class BatchEvidence:
    batch: str
    snapshot_id: str
    decision_path: Path
    decision_sha256: str
    manifest_path: Path
    manifest_sha256: str
    checksum_path: Path
    checksum_sha256: str
    tasks: tuple[dict[str, Any], ...]
    checksums: Mapping[str, str]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 文件无效: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return payload


def _read_archive_tasks(path: Path) -> dict[str, ArchiveTaskRecord]:
    catalog = path.resolve()
    if not catalog.is_file():
        raise FileNotFoundError(f"找不到归档状态库: {catalog}")
    connection = sqlite3.connect(f"{catalog.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(archive_tasks)")
        }
        missing = _CATALOG_COLUMNS.difference(columns)
        if missing:
            raise ValueError(f"归档状态库缺少字段: {sorted(missing)}")
        rows = connection.execute(
            "SELECT task_id, api_name, params_json, status, row_count, "
            "bronze_path, schema_fingerprint, raw_sha256 FROM archive_tasks"
        ).fetchall()
    finally:
        connection.close()
    result: dict[str, ArchiveTaskRecord] = {}
    for row in rows:
        try:
            params = json.loads(row["params_json"] or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"归档任务 params_json 无效: {row['task_id']}"
            ) from exc
        if not isinstance(params, dict):
            raise ValueError(f"归档任务参数不是对象: {row['task_id']}")
        record = ArchiveTaskRecord(
            task_id=str(row["task_id"]),
            api_name=str(row["api_name"]),
            params=params,
            status=str(row["status"]),
            row_count=int(row["row_count"]),
            bronze_path=str(row["bronze_path"] or ""),
            schema_fingerprint=str(row["schema_fingerprint"] or ""),
            raw_sha256=str(row["raw_sha256"] or ""),
        )
        result[record.task_id] = record
    return result


def remap_archive_path(recorded_path: str, archive_root: str | Path) -> tuple[Path, str]:
    """Map a catalog path from another machine onto a supplied data-lake root."""
    if not recorded_path.strip():
        raise ValueError("归档任务没有 Bronze 路径")
    root = Path(archive_root).resolve()
    recorded = Path(recorded_path)
    parts = list(recorded.parts)
    if "data_lake" in parts:
        suffix = parts[parts.index("data_lake") + 1 :]
    elif not recorded.is_absolute():
        suffix = parts[1:] if parts and parts[0] == "data_lake" else parts
    else:
        raise ValueError(f"绝对路径不含 data_lake 锚点: {recorded_path}")
    if not suffix or any(part in {"", ".", ".."} for part in suffix):
        raise ValueError(f"归档路径包含不安全分量: {recorded_path}")
    candidate = root.joinpath(*suffix).resolve()
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"归档路径越过指定根目录: {recorded_path}") from exc
    return candidate, relative


def _parse_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"checksums.sha256 第 {number} 行无效: {path}")
        digest, relative = parts[0].lower(), parts[1].strip().lstrip("*")
        previous = checksums.get(relative)
        if previous is not None and previous != digest:
            raise ValueError(f"同一路径存在冲突 SHA256: {relative}")
        checksums[relative] = digest
    return checksums


def _load_batch_evidence(reports_root: Path, batch: str) -> BatchEvidence:
    root = reports_root.resolve() / "batches" / batch
    decision_path = root / "batch_decision.json"
    manifest_path = root / "batch_manifest.jsonl"
    checksum_path = root / "checksums.sha256"
    for path in (decision_path, manifest_path, checksum_path):
        if not path.is_file():
            raise FileNotFoundError(f"严格归档缺少批次证据: {path}")
    decision = _read_json(decision_path)
    gates = decision.get("gates", {})
    if (
        decision.get("batch") != batch
        or decision.get("decision") != "pass"
        or not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        raise ValueError(f"批次 {batch} 未通过全部准入门")
    snapshot_id = str(decision.get("snapshot_id", ""))
    if not snapshot_id:
        raise ValueError(f"批次 {batch} 缺少 snapshot_id")
    candidates: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"批次 manifest 存在无效 JSON: {manifest_path}") from exc
        if isinstance(payload, dict) and payload.get("snapshot_id") == snapshot_id:
            candidates.append(payload)
    if not candidates:
        raise ValueError(f"批次 {batch} manifest 未找到决策对应快照 {snapshot_id}")
    manifest = candidates[-1]
    manifest_tasks = manifest.get("tasks", [])
    if not isinstance(manifest_tasks, list) or not manifest_tasks:
        raise ValueError(f"批次 {batch} manifest 任务清单为空")
    by_status: dict[str, int] = {}
    for task in manifest_tasks:
        if not isinstance(task, dict):
            raise ValueError(f"批次 {batch} manifest 任务记录无效")
        status = str(task.get("status", ""))
        by_status[status] = by_status.get(status, 0) + 1
    if by_status != manifest.get("by_status"):
        raise ValueError(f"批次 {batch} manifest 状态汇总不一致")
    if any(status not in TERMINAL_STATUSES for status in by_status):
        raise ValueError(f"批次 {batch} 仍含非终态任务: {by_status}")
    return BatchEvidence(
        batch=batch,
        snapshot_id=snapshot_id,
        decision_path=decision_path,
        decision_sha256=sha256_file(decision_path),
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        checksum_path=checksum_path,
        checksum_sha256=sha256_file(checksum_path),
        tasks=tuple(manifest_tasks),
        checksums=_parse_checksums(checksum_path),
    )


def _select_strict_tasks(
    catalog: Mapping[str, ArchiveTaskRecord],
    batches: Mapping[str, BatchEvidence],
) -> tuple[list[ArchiveTaskRecord], dict[str, str]]:
    selected: list[ArchiveTaskRecord] = []
    source_by_api = {
        "income_vip": "income",
        "balancesheet_vip": "balancesheet",
        "cashflow_vip": "cashflow",
        "fina_indicator_vip": "fina_indicator",
    }
    for api_name, batch in STRICT_ENDPOINT_BATCH.items():
        evidence = batches[batch]
        rows = [
            task
            for task in evidence.tasks
            if str(task.get("api_name")) == api_name
        ]
        if not rows:
            raise ValueError(f"严格批次 {batch} 未包含必需端点 {api_name}")
        success_count = 0
        for row in rows:
            task_id = str(row.get("task_id", ""))
            record = catalog.get(task_id)
            if record is None:
                raise ValueError(f"批次任务不在状态库中: {api_name}/{task_id}")
            manifest_status = str(row.get("status", ""))
            if record.status != manifest_status:
                raise ValueError(
                    f"批次证据与当前状态库不一致: {task_id} "
                    f"{manifest_status}!={record.status}"
                )
            if record.api_name != api_name:
                raise ValueError(f"任务端点身份不一致: {task_id}")
            if record.status == "confirmed_empty":
                continue
            if record.status != "success":
                raise ValueError(f"必需端点存在阻塞任务: {api_name}/{record.status}")
            if evidence.snapshot_id not in Path(record.bronze_path).name:
                raise ValueError(f"任务 Bronze 文件未绑定批次快照: {task_id}")
            selected.append(record)
            success_count += 1
        if success_count == 0:
            raise ValueError(f"必需端点没有成功分区: {api_name}")
    return selected, source_by_api


def _select_fixture_tasks(
    catalog: Mapping[str, ArchiveTaskRecord], archive_root: Path
) -> tuple[list[ArchiveTaskRecord], dict[str, str]]:
    selected: list[ArchiveTaskRecord] = []
    source_by_api: dict[str, str] = {}
    for api_name in ("trade_cal", "daily_basic"):
        rows = []
        for task in catalog.values():
            if task.api_name != api_name or task.status != "success":
                continue
            try:
                path, _ = remap_archive_path(task.bronze_path, archive_root)
            except ValueError:
                continue
            if path.is_file():
                rows.append(task)
        if not rows:
            raise ValueError(f"样例包没有可用端点: {api_name}")
        selected.extend(rows)
    for source, aliases in FUNDAMENTAL_ALIASES.items():
        chosen: list[ArchiveTaskRecord] = []
        chosen_api = ""
        for api_name in aliases:
            rows = []
            for task in catalog.values():
                if task.api_name != api_name or task.status != "success":
                    continue
                try:
                    path, _ = remap_archive_path(task.bronze_path, archive_root)
                except ValueError:
                    continue
                if path.is_file():
                    rows.append(task)
            if rows:
                chosen, chosen_api = rows, api_name
                break
        if not chosen:
            raise ValueError(f"样例包没有可用财务端点: {source}")
        selected.extend(chosen)
        source_by_api[chosen_api] = source
    unique = {task.task_id: task for task in selected}
    return list(unique.values()), source_by_api


class _BucketStore:
    def __init__(
        self,
        root: Path,
        name: str,
        schema: pa.Schema,
        bucket_count: int,
    ) -> None:
        self.root = root / name
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.bucket_count = bucket_count
        self.writers: dict[int, pq.ParquetWriter] = {}

    def _bucket(self, symbol: str) -> int:
        digest = hashlib.sha256(symbol.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.bucket_count

    def append(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        buckets = frame["symbol"].astype(str).map(self._bucket)
        for bucket, group in frame.groupby(buckets, sort=True):
            normalized = group[list(self.schema.names)].copy()
            for field in self.schema:
                if pa.types.is_string(field.type):
                    normalized[field.name] = (
                        normalized[field.name].astype("string").fillna("")
                    )
                elif pa.types.is_timestamp(field.type):
                    normalized[field.name] = pd.to_datetime(
                        normalized[field.name], errors="coerce"
                    )
                elif pa.types.is_integer(field.type):
                    normalized[field.name] = pd.to_numeric(
                        normalized[field.name], errors="raise"
                    ).astype("int64")
                else:
                    normalized[field.name] = pd.to_numeric(
                        normalized[field.name], errors="coerce"
                    ).astype("float64")
            table = pa.Table.from_pandas(
                normalized,
                schema=self.schema,
                preserve_index=False,
                safe=True,
            )
            number = int(bucket)
            writer = self.writers.get(number)
            if writer is None:
                path = self.root / f"bucket-{number:04d}.parquet"
                writer = pq.ParquetWriter(path, self.schema, compression="zstd")
                self.writers[number] = writer
            writer.write_table(table)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()

    def paths(self) -> list[Path]:
        return sorted(self.root.glob("bucket-*.parquet"))


def _registered_schema(
    task: ArchiveTaskRecord,
    frame: pd.DataFrame,
    schema_root: Path,
    evidence: dict[tuple[str, str], dict[str, Any]],
) -> None:
    fingerprint = task.schema_fingerprint
    if not fingerprint or len(fingerprint) != 64:
        raise ValueError(f"任务缺少有效 schema_fingerprint: {task.task_id}")
    path = schema_root / task.api_name / f"{fingerprint}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Schema 注册项不存在: {path}")
    payload = _read_json(path)
    expected_columns = payload.get("columns")
    if (
        payload.get("endpoint") != task.api_name
        or payload.get("fingerprint") != fingerprint
        or expected_columns != list(frame.columns)
    ):
        raise ValueError(f"Bronze 列结构与注册 Schema 不一致: {task.task_id}")
    key = (task.api_name, fingerprint)
    evidence[key] = {
        "api_name": task.api_name,
        "fingerprint": fingerprint,
        "registry_relative_path": f"{task.api_name}/{fingerprint}.json",
        "registry_sha256": sha256_file(path),
        "columns": list(expected_columns),
    }


def _load_partition(
    task: ArchiveTaskRecord,
    archive_root: Path,
    schema_root: Path,
    expected_checksum: str | None,
    schema_evidence: dict[tuple[str, str], dict[str, Any]],
) -> tuple[pd.DataFrame, str, str]:
    path, relative = remap_archive_path(task.bronze_path, archive_root)
    if not path.is_file():
        raise FileNotFoundError(f"Bronze 分区不存在: {path}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_checksum is not None and digest != expected_checksum:
        raise ValueError(f"Bronze SHA256 不一致: {relative}")
    frame = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
    if len(frame) != task.row_count:
        raise ValueError(
            f"Bronze 行数与状态库不一致: {task.task_id} "
            f"{len(frame)}!={task.row_count}"
        )
    _registered_schema(task, frame, schema_root, schema_evidence)
    return frame, relative, digest


def _normalized_report_type(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .fillna("")
    )


def _resequence_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
    frame = frame[FUNDAMENTAL_COLUMNS].drop_duplicates().copy()
    row_key = [
        "symbol",
        "source",
        "period_end",
        "report_type",
        "company_type",
        "source_row_sha256",
    ]
    group_key = row_key[:-1]
    revisions = (
        frame[
            [
                *row_key,
                "announcement_date",
                "available_date",
                "source_update_flag",
            ]
        ]
        .drop_duplicates(row_key)
        .sort_values(
            [
                *group_key,
                "announcement_date",
                "available_date",
                "source_update_flag",
                "source_row_sha256",
            ]
        )
    )
    revisions["revision_sequence"] = (
        revisions.groupby(group_key, dropna=False).cumcount() + 1
    )
    frame = frame.drop(columns="revision_sequence").merge(
        revisions[[*row_key, "revision_sequence"]],
        on=row_key,
        how="left",
        validate="many_to_one",
    )
    return frame[FUNDAMENTAL_COLUMNS].sort_values(
        [
            "symbol",
            "metric",
            "period_end",
            "available_date",
            "revision_sequence",
        ]
    )


def _deduplicate_valuations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=VALUATION_COLUMNS)
    frame = frame[VALUATION_COLUMNS].drop_duplicates().copy()
    duplicated = frame.duplicated(["symbol", "date"], keep=False)
    if duplicated.any():
        sample = (
            frame.loc[duplicated, ["symbol", "date"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(f"估值分区存在冲突证券交易日: {sample}")
    return frame.sort_values(["symbol", "date"])


def _atomic_gzip_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
            zipped.write(
                json.dumps(
                    dict(payload),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
    os.replace(temporary, path)


def _lineage_task_payload(
    task: ArchiveTaskRecord,
    *,
    relative_path: str,
    bronze_sha256: str,
    canonical_source: str | None,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "api_name": task.api_name,
        "canonical_source": canonical_source,
        "status": task.status,
        "row_count": task.row_count,
        "bronze_relative_path": relative_path,
        "bronze_sha256": bronze_sha256,
        "schema_fingerprint": task.schema_fingerprint,
        "raw_sha256": task.raw_sha256,
    }


def _audit_from_partitions(
    symbols: Sequence[str],
    fundamental_stats: Mapping[str, Any],
    valuation_stats: Mapping[str, Any],
) -> dict[str, Any]:
    expected = set(symbols)
    fundamental_symbols = set(fundamental_stats["symbols"])
    valuation_symbols = set(valuation_stats["symbols"])
    denominator = max(len(expected), 1)
    available_dates = [
        value
        for value in (
            fundamental_stats.get("last_available_date"),
            valuation_stats.get("last_available_date"),
        )
        if value is not None
    ]
    return {
        "fundamental_rows": int(fundamental_stats["rows"]),
        "valuation_rows": int(valuation_stats["rows"]),
        "fundamental_symbols": len(fundamental_symbols),
        "valuation_symbols": len(valuation_symbols),
        "expected_symbols": len(expected),
        "fundamental_symbol_coverage": len(expected & fundamental_symbols)
        / denominator,
        "valuation_symbol_coverage": len(expected & valuation_symbols)
        / denominator,
        "missing_fundamental_symbols": sorted(expected - fundamental_symbols),
        "missing_valuation_symbols": sorted(expected - valuation_symbols),
        "metrics": sorted(fundamental_stats["metrics"]),
        "first_period_end": (
            str(fundamental_stats["first_period_end"].date())
            if fundamental_stats.get("first_period_end") is not None
            else None
        ),
        "last_available_date": (
            str(max(available_dates).date()) if available_dates else None
        ),
    }


def _compact_buckets(
    store: _BucketStore,
    output_dir: Path,
    symbols: Sequence[str],
    *,
    fundamentals: bool,
) -> dict[str, Any]:
    expected = set(symbols)
    seen: set[str] = set()
    row_count = 0
    metrics: set[str] = set()
    first_period_end: pd.Timestamp | None = None
    last_available_date: pd.Timestamp | None = None
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in store.paths():
        frame = pd.read_parquet(path, engine="pyarrow")
        frame = frame.loc[frame["symbol"].astype(str).isin(expected)].copy()
        frame = (
            _resequence_fundamentals(frame)
            if fundamentals
            else _deduplicate_valuations(frame)
        )
        for symbol, group in frame.groupby("symbol", sort=True):
            text = str(symbol)
            filename = text.replace(".", "_") + ".csv.gz"
            _atomic_csv(group, output_dir / filename)
            seen.add(text)
            row_count += len(group)
            current_last = pd.to_datetime(group["available_date"]).max()
            if last_available_date is None or current_last > last_available_date:
                last_available_date = current_last
            if fundamentals:
                metrics.update(group["metric"].astype(str))
                current_first = pd.to_datetime(group["period_end"]).min()
                if first_period_end is None or current_first < first_period_end:
                    first_period_end = current_first
    columns = FUNDAMENTAL_COLUMNS if fundamentals else VALUATION_COLUMNS
    for symbol in symbols:
        if symbol in seen:
            continue
        filename = symbol.replace(".", "_") + ".csv.gz"
        _atomic_csv(pd.DataFrame(columns=columns), output_dir / filename)
    return {
        "rows": row_count,
        "symbols": seen,
        "metrics": metrics,
        "first_period_end": first_period_end,
        "last_available_date": last_available_date,
    }


def _publish_cache(staging: Path, output: Path, overwrite: bool) -> None:
    if output.exists() and not any(output.iterdir()):
        output.rmdir()
    if not output.exists():
        os.replace(staging, output)
        return
    if not overwrite:
        raise FileExistsError(f"PIT 缓存已存在，拒绝覆盖: {output}")
    backup = output.with_name(output.name + ".pre-alpha4")
    if backup.exists():
        raise FileExistsError(f"发现未清理的 PIT 交换备份: {backup}")
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except Exception:
        os.replace(backup, output)
        raise
    _clear_known_cache(backup)
    backup.rmdir()


def build_pit_cache_from_archive(
    config: AppConfig,
    archive_root: str | Path,
    *,
    catalog_path: str | Path | None = None,
    schema_registry: str | Path | None = None,
    reports_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    fixture_mode: bool = False,
    bucket_count: int = 32,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a sealed PIT cache from local Bronze partitions only."""
    config.validate()
    if not config.point_in_time.enabled:
        raise ValueError("归档转 PIT 要求 point_in_time.enabled=true")
    if bucket_count < 1 or bucket_count > 128:
        raise ValueError("bucket_count 必须在 1..128")
    root = Path(archive_root).resolve()
    catalog = Path(catalog_path).resolve() if catalog_path else root / "catalog" / "archive.duckdb"
    schema_root = (
        Path(schema_registry).resolve()
        if schema_registry
        else root / "catalog" / "schema_registry"
    )
    reports = Path(reports_root).resolve() if reports_root else root / "reports"
    output = (
        Path(output_dir).resolve()
        if output_dir
        else Path(config.point_in_time.cache_dir).resolve()
    )
    if not root.is_dir():
        raise FileNotFoundError(f"归档根目录不存在: {root}")
    if not schema_root.is_dir():
        raise FileNotFoundError(f"Schema 注册目录不存在: {schema_root}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"PIT 输出路径不是目录: {output}")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"PIT 缓存已存在，拒绝覆盖: {output}")
        # Validate the complete sealed file set before it can enter the atomic
        # backup/swap path.  A partial or unrelated directory is never deleted.
        PointInTimeDataBundle.from_cache(output, strict=True)

    task_catalog = _read_archive_tasks(catalog)
    batches: dict[str, BatchEvidence] = {}
    if fixture_mode:
        selected, source_by_api = _select_fixture_tasks(task_catalog, root)
        expected_symbols: set[str] | None = None
        base_manifest_path: Path | None = None
    else:
        batches = {
            batch: _load_batch_evidence(reports, batch)
            for batch in STRICT_BATCHES
        }
        selected, source_by_api = _select_strict_tasks(task_catalog, batches)
        base_manifest_path = (
            Path(config.data.cache_dir).resolve() / "manifest.json"
        )
        expected_symbols = _verify_base_market_cache(base_manifest_path, config)

    selected.sort(
        key=lambda task: (
            task.api_name,
            json.dumps(task.params, sort_keys=True, ensure_ascii=False),
            task.task_id,
        )
    )
    selected_by_api: dict[str, list[ArchiveTaskRecord]] = {}
    for task in selected:
        selected_by_api.setdefault(task.api_name, []).append(task)

    start = pd.Timestamp(config.backtest.start_date) - pd.DateOffset(
        years=config.point_in_time.history_years
    )
    end = pd.Timestamp(config.backtest.end_date)
    calendar_end = end + pd.Timedelta(31, unit="D")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=output.name + ".alpha4-", dir=output.parent)
    ).resolve()
    bucket_root = staging / "_buckets"
    valuation_store = _BucketStore(
        bucket_root, "valuations", _VALUATION_ARROW_SCHEMA, bucket_count
    )
    fundamental_store = _BucketStore(
        bucket_root, "fundamentals", _FUNDAMENTAL_ARROW_SCHEMA, bucket_count
    )
    schema_evidence: dict[tuple[str, str], dict[str, Any]] = {}
    lineage_tasks: dict[str, dict[str, Any]] = {}

    def load(task: ArchiveTaskRecord) -> pd.DataFrame:
        expected_checksum: str | None = None
        if not fixture_mode:
            batch = STRICT_ENDPOINT_BATCH[task.api_name]
            _, relative = remap_archive_path(task.bronze_path, root)
            expected_checksum = batches[batch].checksums.get(relative)
            if expected_checksum is None:
                raise ValueError(f"批次 checksums 未封存 Bronze: {relative}")
        frame, relative, digest = _load_partition(
            task,
            root,
            schema_root,
            expected_checksum,
            schema_evidence,
        )
        lineage_tasks[task.task_id] = _lineage_task_payload(
            task,
            relative_path=relative,
            bronze_sha256=digest,
            canonical_source=source_by_api.get(task.api_name),
        )
        return frame

    try:
        calendar_frames: list[pd.DataFrame] = []
        for task in selected_by_api.get("trade_cal", []):
            raw = load(task)
            if "cal_date" not in raw or "is_open" not in raw:
                raise ValueError("trade_cal 缺少 cal_date/is_open")
            open_rows = raw.loc[pd.to_numeric(raw["is_open"], errors="coerce").eq(1)]
            calendar_frames.append(open_rows[["cal_date"]])
        if not calendar_frames:
            raise ValueError("没有可用交易日历")
        calendar = pd.DatetimeIndex(
            pd.to_datetime(
                pd.concat(calendar_frames, ignore_index=True)["cal_date"]
                .astype("string")
                .str.replace(r"\.0$", "", regex=True),
                errors="coerce",
                format="mixed",
            )
        ).dropna().normalize().unique().sort_values()
        calendar = calendar[(calendar >= start) & (calendar <= calendar_end)]
        if (
            calendar.empty
            or calendar.min() > start + pd.Timedelta(7, unit="D")
            or calendar.max() < end - pd.Timedelta(7, unit="D")
        ):
            raise ValueError(
                f"交易日历未覆盖请求区间: {start.date()}..{end.date()}"
            )

        observed_valuation_symbols: set[str] = set()
        for task in selected_by_api.get("daily_basic", []):
            raw = load(task)
            if "trade_date" not in raw:
                trade_date = task.params.get("trade_date")
                if not trade_date:
                    raise ValueError(f"daily_basic 缺 trade_date 且任务参数未提供: {task.task_id}")
                raw = raw.copy()
                raw["trade_date"] = str(trade_date)
            if "ts_code" not in raw:
                raise ValueError(f"daily_basic 缺 ts_code: {task.task_id}")
            dates = pd.to_datetime(
                raw["trade_date"].astype("string").str.replace(r"\.0$", "", regex=True),
                errors="coerce",
                format="mixed",
            )
            mask = dates.between(start, end)
            if expected_symbols is not None:
                mask &= raw["ts_code"].astype(str).isin(expected_symbols)
            raw = raw.loc[mask].copy()
            if raw.empty:
                continue
            normalized = normalize_valuations(
                raw,
                calendar,
                config.point_in_time.valuation_lag_trading_days,
            )
            observed_valuation_symbols.update(normalized["symbol"].astype(str))
            valuation_store.append(normalized)
        valuation_store.close()

        observed_fundamental_symbols: set[str] = set()
        for api_name, canonical_source in sorted(source_by_api.items()):
            for task in selected_by_api.get(api_name, []):
                raw = load(task)
                if "ts_code" not in raw:
                    raise ValueError(f"{api_name} 缺 ts_code: {task.task_id}")
                if expected_symbols is not None:
                    raw = raw.loc[raw["ts_code"].astype(str).isin(expected_symbols)].copy()
                if canonical_source != "fina_indicator":
                    if "report_type" not in raw:
                        raise ValueError(f"{api_name} 缺 report_type")
                    raw = raw.loc[
                        _normalized_report_type(raw["report_type"]).eq(
                            CANONICAL_STATEMENT_REPORT_TYPE
                        )
                    ].copy()
                    raw["report_type"] = CANONICAL_STATEMENT_REPORT_TYPE
                if raw.empty:
                    continue
                normalized = normalize_fundamental_source(
                    canonical_source,
                    raw,
                    calendar,
                    config.point_in_time.fundamental_lag_trading_days,
                )
                normalized = normalized.loc[
                    normalized["announcement_date"].between(start, end)
                ]
                if normalized.empty:
                    continue
                observed_fundamental_symbols.update(
                    normalized["symbol"].astype(str)
                )
                fundamental_store.append(normalized)
        valuation_store.close()
        fundamental_store.close()

        if expected_symbols is None:
            expected_symbols = (
                observed_fundamental_symbols & observed_valuation_symbols
            )
        symbols = sorted(expected_symbols)
        if not symbols:
            raise ValueError("归档桥接没有共同的财报/估值证券")

        fundamental_stats = _compact_buckets(
            fundamental_store,
            staging / "fundamentals",
            symbols,
            fundamentals=True,
        )
        valuation_stats = _compact_buckets(
            valuation_store,
            staging / "valuations",
            symbols,
            fundamentals=False,
        )
        _atomic_csv(pd.DataFrame({"date": calendar}), staging / "calendar.csv.gz")
        shutil.rmtree(bucket_root)

        audit = _audit_from_partitions(
            symbols, fundamental_stats, valuation_stats
        )
        if (
            not fixture_mode
            and (
                audit["fundamental_symbol_coverage"]
                < config.point_in_time.minimum_symbol_coverage
                or audit["valuation_symbol_coverage"]
                < config.point_in_time.minimum_symbol_coverage
            )
        ):
            raise ValueError(
                "严格归档转 PIT 覆盖不足: "
                f"财报={audit['fundamental_symbol_coverage']:.2%}，"
                f"估值={audit['valuation_symbol_coverage']:.2%}"
            )

        selected_task_rows = sorted(
            lineage_tasks.values(), key=lambda item: item["task_id"]
        )
        task_set_sha256 = payload_sha256(selected_task_rows)
        batch_rows = [
            {
                "batch": evidence.batch,
                "snapshot_id": evidence.snapshot_id,
                "decision_relative_path": (
                    Path("batches") / evidence.batch / "batch_decision.json"
                ).as_posix(),
                "decision_sha256": evidence.decision_sha256,
                "manifest_relative_path": (
                    Path("batches") / evidence.batch / "batch_manifest.jsonl"
                ).as_posix(),
                "manifest_sha256": evidence.manifest_sha256,
                "checksums_relative_path": (
                    Path("batches") / evidence.batch / "checksums.sha256"
                ).as_posix(),
                "checksums_sha256": evidence.checksum_sha256,
            }
            for evidence in batches.values()
        ]
        lineage = {
            "schema_version": ARCHIVE_PIT_BRIDGE_VERSION,
            "mode": "fixture" if fixture_mode else "strict",
            "research_eligible": not fixture_mode,
            "catalog_sha256": sha256_file(catalog),
            "selected_task_set_sha256": task_set_sha256,
            "selected_tasks": selected_task_rows,
            "registered_schemas": sorted(
                schema_evidence.values(),
                key=lambda item: (item["api_name"], item["fingerprint"]),
            ),
            "batch_evidence": sorted(batch_rows, key=lambda item: item["batch"]),
            "requested_start": str(start.date()),
            "requested_end": str(end.date()),
            "symbols": symbols,
        }
        lineage_path = staging / ARCHIVE_LINEAGE_FILENAME
        _atomic_gzip_json(lineage, lineage_path)

        base_manifest_sha256 = None
        base_fingerprint = None
        if base_manifest_path is not None:
            base_payload = _read_json(base_manifest_path)
            base_manifest_sha256 = sha256_file(base_manifest_path)
            base_fingerprint = base_payload.get("data_fingerprint_sha256")
        data_paths = [
            staging / "calendar.csv.gz",
            lineage_path,
            *sorted((staging / "fundamentals").glob("*.csv.gz")),
            *sorted((staging / "valuations").glob("*.csv.gz")),
        ]
        files = build_file_inventory(staging, data_paths)
        manifest = {
            "schema_version": PIT_CACHE_SCHEMA_VERSION,
            "provider": config.point_in_time.provider,
            "statement_report_type": CANONICAL_STATEMENT_REPORT_TYPE,
            "requested_start": str(start.date()),
            "requested_end": str(end.date()),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "fundamental_lag_trading_days": config.point_in_time.fundamental_lag_trading_days,
            "valuation_lag_trading_days": config.point_in_time.valuation_lag_trading_days,
            "base_manifest_sha256": base_manifest_sha256,
            "base_data_fingerprint_sha256": base_fingerprint,
            "symbols": symbols,
            "data_quality": audit,
            "research_eligible": not fixture_mode,
            "archive_bridge": {
                "schema_version": ARCHIVE_PIT_BRIDGE_VERSION,
                "mode": lineage["mode"],
                "acceptance_required": not fixture_mode,
                "lineage_file": ARCHIVE_LINEAGE_FILENAME,
                "selected_task_set_sha256": task_set_sha256,
                "catalog_sha256": lineage["catalog_sha256"],
                "batch_snapshots": {
                    item["batch"]: item["snapshot_id"] for item in batch_rows
                },
            },
            "files": files,
            "data_fingerprint_sha256": inventory_sha256(files),
        }
        _atomic_json(manifest, staging / "manifest.json")
        PointInTimeDataBundle.from_cache(
            staging,
            strict=True,
            expected_config=config,
            base_manifest_path=base_manifest_path,
        )
        _publish_cache(staging, output, overwrite)
        return _read_json(output / "manifest.json")
    except Exception:
        valuation_store.close()
        fundamental_store.close()
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _read_lineage(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / ARCHIVE_LINEAGE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"归档 PIT 缺少血缘文件: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"归档 PIT 血缘文件无效: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("归档 PIT 血缘顶层必须是对象")
    return payload


def verify_archive_pit_cache(
    config: AppConfig,
    cache_dir: str | Path,
    *,
    fixture_mode: bool = False,
    archive_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
    schema_registry: str | Path | None = None,
    reports_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the sealed PIT cache and optionally replay all source evidence."""
    root = Path(cache_dir).resolve()
    base_manifest = (
        None
        if fixture_mode
        else Path(config.data.cache_dir).resolve() / "manifest.json"
    )
    bundle = PointInTimeDataBundle.from_cache(
        root,
        strict=True,
        expected_config=config,
        base_manifest_path=base_manifest,
    )
    lineage = _read_lineage(root)
    manifest = bundle.manifest
    tasks = lineage.get("selected_tasks", [])
    task_set_sha256 = payload_sha256(tasks)
    if (
        lineage.get("schema_version")
        not in SUPPORTED_ARCHIVE_PIT_BRIDGE_VERSIONS
        or task_set_sha256 != lineage.get("selected_task_set_sha256")
        or task_set_sha256
        != manifest.get("archive_bridge", {}).get("selected_task_set_sha256")
        or bool(lineage.get("research_eligible"))
        != bool(manifest.get("research_eligible"))
        or lineage.get("mode") != ("fixture" if fixture_mode else "strict")
    ):
        raise ValueError("归档 PIT 血缘身份或研究资格不一致")

    source_files_verified = 0
    if archive_root is not None:
        source_root = Path(archive_root).resolve()
        schema_root = (
            Path(schema_registry).resolve()
            if schema_registry
            else source_root / "catalog" / "schema_registry"
        )
        reports = (
            Path(reports_root).resolve()
            if reports_root
            else source_root / "reports"
        )
        catalog = (
            Path(catalog_path).resolve()
            if catalog_path
            else source_root / "catalog" / "archive.duckdb"
        )
        catalog_rows = _read_archive_tasks(catalog)
        for item in tasks:
            task_id = str(item.get("task_id", ""))
            record = catalog_rows.get(task_id)
            if record is None or record.status != "success":
                raise ValueError(f"源状态库不再包含成功任务: {task_id}")
            path, relative = remap_archive_path(
                "data_lake/" + str(item["bronze_relative_path"]), source_root
            )
            if sha256_file(path) != item.get("bronze_sha256"):
                raise ValueError(f"源 Bronze 已变化: {path}")
            _, catalog_relative = remap_archive_path(
                record.bronze_path, source_root
            )
            if (
                record.api_name != item.get("api_name")
                or record.row_count != int(item.get("row_count", -1))
                or record.schema_fingerprint != item.get("schema_fingerprint")
                or record.raw_sha256 != item.get("raw_sha256")
                or catalog_relative != relative
            ):
                raise ValueError(f"源任务身份已变化: {task_id}")
            source_files_verified += 1
        for item in lineage.get("registered_schemas", []):
            path = schema_root / str(item["registry_relative_path"])
            if sha256_file(path) != item.get("registry_sha256"):
                raise ValueError(f"源 Schema 已变化: {path}")
        for item in lineage.get("batch_evidence", []):
            for path_key, hash_key in (
                ("decision_relative_path", "decision_sha256"),
                ("manifest_relative_path", "manifest_sha256"),
                ("checksums_relative_path", "checksums_sha256"),
            ):
                path = reports / str(item[path_key])
                if sha256_file(path) != item.get(hash_key):
                    raise ValueError(f"源批次证据已变化: {path}")
    return {
        "verified": True,
        "mode": lineage["mode"],
        "research_eligible": bool(lineage["research_eligible"]),
        "selected_task_count": len(tasks),
        "selected_task_set_sha256": task_set_sha256,
        "source_files_verified": source_files_verified,
        "data_fingerprint_sha256": manifest["data_fingerprint_sha256"],
    }


__all__ = [
    "ARCHIVE_LINEAGE_FILENAME",
    "ARCHIVE_PIT_BRIDGE_VERSION",
    "SUPPORTED_ARCHIVE_PIT_BRIDGE_VERSIONS",
    "build_pit_cache_from_archive",
    "remap_archive_path",
    "verify_archive_pit_cache",
]
