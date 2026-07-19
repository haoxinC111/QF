"""Archive task execution: resume, split-on-truncation, and empty handling."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ArchiveConfig
from .provider import ArchiveProvider, ArchiveResponse
from .schema import SchemaRegistry, schema_fingerprint
from .state import DownloadTask, TaskStateDB, TaskStatus
from .storage import store_response
from .throttle import RateLimitedClient, RetryPolicy, TokenBucket

logger = logging.getLogger(__name__)


@dataclass
class EndpointSpec:
    api_name: str
    dataset: str
    priority: str
    primary_key: list[str]
    primary_split: str | None
    fallback_split: str | None
    all_fields: bool
    fields: str
    params_template: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_empty: int = 0
    rows_total: int = 0
    errors: list[str] = field(default_factory=list)


def task_id(
    provider_name: str,
    api_name: str,
    params: dict[str, Any],
    fields: str,
    snapshot_id: str,
) -> str:
    payload = json.dumps(
        {
            "provider": provider_name,
            "api_name": api_name,
            "params": params,
            "fields": fields,
            "snapshot_id": snapshot_id,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ArchivePipeline:
    """Execute endpoint tasks with resumption and truncation detection."""

    def __init__(
        self,
        config: ArchiveConfig,
        provider: ArchiveProvider,
        db: TaskStateDB,
        *,
        snapshot_id: str | None = None,
        symbol_universe: list[str] | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.db = db
        self.snapshot_id = snapshot_id or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        self.symbol_universe = list(symbol_universe or [])
        self.schema_registry = SchemaRegistry(config.catalog_dir / "schema_registry")
        self.bucket = TokenBucket(config.rate_limit.calls_per_minute)
        self.client = RateLimitedClient(
            self.bucket,
            RetryPolicy(
                max_attempts=config.rate_limit.retry_attempts,
                retry_statuses=tuple(config.rate_limit.retry_statuses),
            ),
        )
        self.observed_caps: dict[str, int] = {}

    def run_tasks(
        self,
        specs: list[EndpointSpec],
        *,
        skip_existing: bool = True,
    ) -> PipelineResult:
        result = PipelineResult()
        for spec in specs:
            try:
                task = self._get_or_create_task(spec)
                if skip_existing and task.status in (TaskStatus.SUCCESS, TaskStatus.CONFIRMED_EMPTY):
                    logger.info("跳过已完成任务: %s", task.task_id)
                    result.tasks_completed += 1
                    continue
                self._execute_task(task, spec, result)
            except Exception as exc:  # noqa: BLE001
                msg = f"{spec.api_name} 任务失败: {exc}"
                logger.exception(msg)
                result.errors.append(msg)
                result.tasks_failed += 1
        return result

    def _get_or_create_task(self, spec: EndpointSpec) -> DownloadTask:
        params = dict(spec.params_template)
        tid = task_id(
            self.config.provider.name,
            spec.api_name,
            params,
            spec.fields,
            self.snapshot_id,
        )
        existing = self.db.get(tid)
        if existing:
            return existing
        task = DownloadTask(
            task_id=tid,
            api_name=spec.api_name,
            params=params,
            fields=spec.fields,
            dataset=spec.dataset,
            priority=spec.priority,
            primary_key=spec.primary_key,
            primary_split=spec.primary_split,
            fallback_split=spec.fallback_split,
        )
        self.db.upsert(task)
        return task

    def _execute_task(
        self,
        task: DownloadTask,
        spec: EndpointSpec,
        result: PipelineResult,
    ) -> None:
        task.status = TaskStatus.RUNNING
        task.attempts += 1
        self.db.upsert(task)

        response = self.client.call(
            self.provider.request,
            spec.api_name,
            task.params,
            fields=spec.fields,
        )

        if response.status == "denied":
            task.status = TaskStatus.DENIED
            task.last_error = response.message
            self.db.upsert(task)
            result.tasks_failed += 1
            result.errors.append(f"{spec.api_name} 权限不足: {response.message}")
            return

        if response.status in ("invalid_params", "not_found", "incompatible"):
            task.status = TaskStatus.INVALID_PARAMS
            task.last_error = response.message
            self.db.upsert(task)
            result.tasks_failed += 1
            result.errors.append(f"{spec.api_name} 参数/接口错误({response.status}): {response.message}")
            return

        if response.status == "transient_error":
            # Retry budget already exhausted inside RateLimitedClient; mark the
            # task retryable so the next run resumes it instead of storing an
            # empty payload as if it were real data.
            task.status = TaskStatus.RETRYABLE_ERROR
            task.last_error = response.message
            self.db.upsert(task)
            result.tasks_failed += 1
            result.errors.append(f"{spec.api_name} 瞬时错误: {response.message}")
            return

        if response.status == "empty":
            # Distinguish true empty from permission-denied masquerading as empty.
            if self._should_confirm_empty(response, spec):
                task.status = TaskStatus.CONFIRMED_EMPTY
            else:
                task.status = TaskStatus.SUSPECT_TRUNCATED
            task.row_count = 0
            task.fetched_at_utc = response.fetched_at_utc
            task.elapsed_seconds = response.elapsed_seconds
            task.raw_sha256 = response.raw_payload_sha256
            self.db.upsert(task)
            result.tasks_empty += 1
            return

        # Truncation detection.
        cap = self._observed_row_cap(spec.api_name)
        if self._is_suspect_truncated(response, spec, cap):
            task.status = TaskStatus.SUSPECT_TRUNCATED
            task.last_error = (
                f"疑似截断: rows={response.row_count} >= cap={cap}"
            )
            self.db.upsert(task)
            result.errors.append(task.last_error)
            result.tasks_failed += 1
            # Recursively split and retry.
            sub_specs = self._split_spec(spec, response)
            sub_result = self.run_tasks(sub_specs, skip_existing=True)
            result.tasks_completed += sub_result.tasks_completed
            result.tasks_failed += sub_result.tasks_failed
            result.tasks_empty += sub_result.tasks_empty
            result.rows_total += sub_result.rows_total
            result.errors.extend(sub_result.errors)
            return

        # Schema drift guard: refuse to persist when the column layout matches
        # no registered fingerprint. Historical schema variants (verified by a
        # human) are registered as additional fingerprints; only never-seen
        # layouts are quarantined.
        fingerprint = schema_fingerprint(response.columns)
        known_fingerprints = self.schema_registry.fingerprints(spec.api_name)
        if known_fingerprints and fingerprint not in known_fingerprints:
            registered_hint = sorted(known_fingerprints)[-1]
            task.status = TaskStatus.QUARANTINED
            task.last_error = (
                f"schema 漂移阻断: 已登记 {len(known_fingerprints)} 个指纹均不匹配 "
                f"(最近 {registered_hint[:12]} != 本次 {fingerprint[:12]})"
            )
            task.metadata["schema_drift"] = {
                "registered": registered_hint,
                "observed": fingerprint,
                "columns": list(response.columns),
            }
            self.db.upsert(task)
            result.tasks_failed += 1
            result.errors.append(f"{spec.api_name} {task.last_error}")
            return

        # Persist raw + bronze.
        partition_key = self._partition_key(spec, task.params)
        stored = store_response(
            raw_dir=self.config.raw_dir / self.snapshot_id / spec.api_name,
            bronze_dir=self.config.bronze_dir / spec.api_name,
            api_name=spec.api_name,
            params=task.params,
            columns=response.columns,
            items=response.items,
            raw_payload=response.raw_payload,
            snapshot_id=self.snapshot_id,
            partition_key=partition_key,
            compression=self.config.parquet_compression,
            immutable=self.config.immutable_raw,
        )

        self.schema_registry.save(
            spec.api_name,
            fingerprint,
            {
                "endpoint": spec.api_name,
                "fingerprint": fingerprint,
                "columns": response.columns,
                "snapshot_id": self.snapshot_id,
                "row_count": response.row_count,
            },
        )

        task.status = TaskStatus.SUCCESS
        task.row_count = stored.row_count
        task.raw_path = str(stored.raw_path)
        task.bronze_path = str(stored.bronze_path)
        task.schema_fingerprint = fingerprint
        task.raw_sha256 = stored.raw_sha256
        task.fetched_at_utc = response.fetched_at_utc
        task.elapsed_seconds = response.elapsed_seconds
        self.db.upsert(task)

        result.tasks_completed += 1
        result.rows_total += stored.row_count

    def _observed_row_cap(self, api_name: str) -> int | None:
        if api_name in self.observed_caps:
            return self.observed_caps[api_name]
        return None

    def _is_suspect_truncated(
        self,
        response: ArchiveResponse,
        spec: EndpointSpec,
        cap: int | None,
    ) -> bool:
        if cap is None:
            return False
        if response.row_count < cap:
            return False
        # Additional heuristic: if response exactly equals a known cap, suspect.
        if response.row_count == cap:
            return True
        # If the last primary key looks like a round boundary, also suspect.
        if spec.primary_key and response.items:
            last = response.items[-1]
            idx = response.columns.index(spec.primary_key[0])
            boundary = last[idx]
            if boundary in (cap, str(cap)):
                return True
        return False

    def _should_confirm_empty(self, response: ArchiveResponse, spec: EndpointSpec) -> bool:
        # An empty response is only confirmed empty if the gateway explicitly
        # returns code 0 and no error message.  Any error message suggests
        # permission or parameter problems rather than true absence of data.
        return "权限" not in response.message and "error" not in response.message.lower()

    def _partition_key(self, spec: EndpointSpec, params: dict[str, Any]) -> str:
        parts = []
        for key in (spec.primary_split, spec.fallback_split):
            if key and key in params:
                parts.append(f"{key}={params[key]}")
        # Always append date/period dimension keys (not only as fallback):
        # symbol-split endpoints like index_daily are bisected into date-range
        # siblings sharing the same ts_code, and without the range in the name
        # they would collide on a single raw/bronze file.  Deduplicate when the
        # split key itself is one of these (e.g. trade_date single-day slices).
        for key in (
            "trade_date",
            "start_date",
            "end_date",
            "ann_date",
            "f_ann_date",
            "period",
            "float_date",
            "month",
            "surv_date",
            "report_date",
        ):
            if key in params:
                part = f"{key}={params[key]}"
                if part not in parts:
                    parts.append(part)
        if not parts:
            # When the endpoint has no split key, include all request params so that
            # distinct filters (e.g. stock_basic list_status) do not collide.
            for key in sorted(params):
                parts.append(f"{key}={params[key]}")
        if not parts:
            parts.append("all")
        # Guard against over-long partition names (e.g. comma-joined symbol
        # chunks): hash long values to keep file names portable.
        safe_parts = []
        for part in parts:
            if len(part) > 48:
                key, _, value = part.partition("=")
                digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]  # noqa: S324
                safe_parts.append(f"{key}=chunk_{digest}")
            else:
                safe_parts.append(part)
        return "_".join(safe_parts)

    def _split_spec(
        self,
        spec: EndpointSpec,
        response: ArchiveResponse,
    ) -> list[EndpointSpec]:
        """Generate sub-tasks when a response hits the row cap."""
        sub_specs: list[EndpointSpec] = []
        params = dict(spec.params_template)

        # Try date bisection first if a date range is present.
        start_date = params.get("start_date") or params.get("start")
        end_date = params.get("end_date") or params.get("end")
        if start_date and end_date and spec.primary_split:
            mid = self._bisect_date(start_date, end_date)
            if mid and mid != end_date:
                left = dict(params)
                right = dict(params)
                # Update date range and keep the split key only if it is the
                # actual date parameter used by the endpoint.
                left["start_date"] = start_date
                left["end_date"] = mid
                right["start_date"] = mid
                right["end_date"] = end_date
                for d in (left, right):
                    # Preserve the split key only if the original request used it.
                    if spec.primary_split in params and spec.primary_split not in ("start_date", "end_date", "start", "end"):
                        d[spec.primary_split] = f"{d['start_date']}_{d['end_date']}"
                    elif spec.primary_split in d and spec.primary_split not in ("start_date", "end_date", "start", "end"):
                        del d[spec.primary_split]
                sub_specs.append(self._spec_with_params(spec, left))
                sub_specs.append(self._spec_with_params(spec, right))
                return sub_specs

        # Fallback: split along the symbol dimension.  The split key may be
        # absent (whole-market request), a comma-joined string, or a list.
        if spec.fallback_split:
            symbols_value = params.get(spec.fallback_split)
            if isinstance(symbols_value, str) and "," in symbols_value:
                symbols = [s for s in symbols_value.split(",") if s]
            elif isinstance(symbols_value, (list, tuple)):
                symbols = list(symbols_value)
            elif symbols_value:
                # Single explicit symbol: cannot split further.
                symbols = []
            else:
                # No symbol filter yet: fall back to the full universe.
                symbols = list(self.symbol_universe)
            if len(symbols) > 1:
                mid_idx = len(symbols) // 2
                left = dict(params)
                right = dict(params)
                left[spec.fallback_split] = ",".join(symbols[:mid_idx])
                right[spec.fallback_split] = ",".join(symbols[mid_idx:])
                sub_specs.append(self._spec_with_params(spec, left))
                sub_specs.append(self._spec_with_params(spec, right))
                return sub_specs

        logger.error("无法进一步拆分任务: %s params=%s", spec.api_name, params)
        return []

    def _bisect_date(self, start: str, end: str) -> str | None:
        try:
            from datetime import datetime, timedelta

            fmt = "%Y%m%d"
            s = datetime.strptime(start, fmt)
            e = datetime.strptime(end, fmt)
            if e <= s:
                return None
            mid = s + (e - s) // 2
            return (mid + timedelta(days=1)).strftime(fmt)
        except ValueError:
            return None

    def _spec_with_params(self, spec: EndpointSpec, params: dict[str, Any]) -> EndpointSpec:
        return EndpointSpec(
            api_name=spec.api_name,
            dataset=spec.dataset,
            priority=spec.priority,
            primary_key=spec.primary_key,
            primary_split=spec.primary_split,
            fallback_split=spec.fallback_split,
            all_fields=spec.all_fields,
            fields=spec.fields,
            params_template=params,
        )

    def probe_row_cap(self, api_name: str, params: dict[str, Any], fields: str) -> int:
        """Run a small probe and record observed row count as cap estimate."""
        response = self.client.call(
            self.provider.request,
            api_name,
            params,
            fields=fields,
        )
        cap = response.row_count if response.is_success else 0
        self.observed_caps[api_name] = cap
        return cap


def build_manifest(
    config: ArchiveConfig,
    db: TaskStateDB,
    snapshot_id: str,
    *,
    git_commit: str | None = None,
    tasks: list[DownloadTask] | None = None,
) -> dict[str, Any]:
    """Build a manifest of tasks (all tasks unless ``tasks`` is provided)."""
    if tasks is None:
        tasks = db.list_tasks()
    by_status: dict[str, int] = {}
    for t in tasks:
        by_status[t.status.value] = by_status.get(t.status.value, 0) + 1
    return {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "provider": config.provider.name,
        "git_commit": git_commit,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tasks": len(tasks),
        "by_status": by_status,
        "tasks": [
            {
                "task_id": t.task_id,
                "api_name": t.api_name,
                "dataset": t.dataset,
                "priority": t.priority,
                "status": t.status.value,
                "row_count": t.row_count,
                "raw_sha256": t.raw_sha256,
                "bronze_path": t.bronze_path,
                "schema_fingerprint": t.schema_fingerprint,
                "fetched_at_utc": t.fetched_at_utc,
            }
            for t in tasks
        ],
    }


def write_manifest(
    config: ArchiveConfig,
    manifest: dict[str, Any],
) -> Path:
    path = config.catalog_dir / "download_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False) + "\n")
    return path


def write_checksums(config: ArchiveConfig, snapshot_id: str) -> Path:
    """Write SHA256 checksums for raw and bronze files in the snapshot."""
    path = config.catalog_dir / "checksums.sha256"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for root in (config.raw_dir, config.bronze_dir):
        snapshot_root = root / snapshot_id
        if not snapshot_root.exists():
            continue
        for file_path in sorted(snapshot_root.rglob("*")):
            if file_path.is_file():
                h = hashlib.sha256(file_path.read_bytes()).hexdigest()
                rel = file_path.relative_to(config.archive_root)
                lines.append(f"{h}  {rel}\n")
    # Atomic append-like write (overwrite is OK for checksum file).
    path.write_text("".join(lines), encoding="utf-8")
    return path
