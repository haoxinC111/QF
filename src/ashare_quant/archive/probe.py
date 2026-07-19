"""Permission probes and endpoint health reports."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .config import ArchiveConfig
from .provider import ArchiveProvider
from .registry import EndpointInventory, InventoryEndpoint
from .schema import schema_fingerprint
from .throttle import RateLimitedClient, RetryPolicy, TokenBucket

logger = logging.getLogger(__name__)

# Probe statuses follow DATA_ACQUISITION_V2_DESIGN.md §10:
# success / denied / not_found / incompatible / transient_error / confirmed_empty
PROBE_STATUSES = (
    "success",
    "denied",
    "not_found",
    "incompatible",
    "transient_error",
    "confirmed_empty",
    "invalid_params",
)


@dataclass
class ProbeResult:
    api_name: str
    status: str
    message: str
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    sample_min_date: str | None = None
    sample_max_date: str | None = None
    elapsed_seconds: float = 0.0
    supports_fields_param: bool = False
    supports_date_param: bool = False
    supports_symbol_param: bool = False
    observed_row_cap: int | None = None
    request_params: dict[str, Any] = field(default_factory=dict)
    request_path: str = ""
    schema_sha256: str = ""
    http_status: int | None = None


def _date_column_index(columns: list[str]) -> int | None:
    for candidate in (
        "trade_date", "cal_date", "ann_date", "end_date", "f_ann_date",
        "report_date", "surv_date", "nav_date", "float_date", "date",
        "month", "quarter",
    ):
        if candidate in columns:
            return columns.index(candidate)
    return None


def _extract_date_range(items: list[list[Any]], date_idx: int | None) -> tuple[str | None, str | None]:
    if date_idx is None or not items:
        return None, None
    values = [row[date_idx] for row in items if row[date_idx] is not None]
    if not values:
        return None, None
    return str(min(values)), str(max(values))


def _default_probe_params(endpoint: InventoryEndpoint) -> dict[str, Any]:
    """Fallback probe params when the endpoint has no explicit probe_params."""
    params = dict(endpoint.params or {})
    if endpoint.primary_split in ("trade_date", "end_date", "ann_date"):
        params[endpoint.primary_split] = "20250102"
    if endpoint.fallback_split == "ts_code" and "ts_code" not in params:
        params["ts_code"] = "000001.SZ"
    return params


def probe_endpoint(
    provider: ArchiveProvider,
    endpoint: InventoryEndpoint,
    client: RateLimitedClient,
    *,
    probe_params: dict[str, Any] | None = None,
) -> ProbeResult:
    """Probe a single endpoint with a tiny request."""
    params = dict(probe_params or endpoint.probe_params or _default_probe_params(endpoint))

    try:
        response = client.call(
            provider.request,
            endpoint.api_name,
            params,
            fields=endpoint.fields,
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            api_name=endpoint.api_name,
            status="transient_error",
            message=f"探针异常: {exc}",
            request_params=params,
        )

    status_map = {
        "success": "success",
        "empty": "confirmed_empty",
        "denied": "denied",
        "invalid_params": "invalid_params",
        "not_found": "not_found",
        "incompatible": "incompatible",
        "transient_error": "transient_error",
    }
    probe_status = status_map.get(response.status, "transient_error")

    date_idx = _date_column_index(response.columns)
    min_date, max_date = _extract_date_range(response.items, date_idx)

    supports_fields = False
    supports_date = any(
        k in params for k in ("trade_date", "start_date", "end_date", "period", "ann_date")
    )
    supports_symbol = "ts_code" in params

    schema_sha = schema_fingerprint(response.columns) if response.columns else ""

    return ProbeResult(
        api_name=endpoint.api_name,
        status=probe_status,
        message=response.message,
        columns=response.columns,
        row_count=response.row_count,
        sample_min_date=min_date,
        sample_max_date=max_date,
        elapsed_seconds=response.elapsed_seconds,
        supports_fields_param=supports_fields,
        supports_date_param=supports_date,
        supports_symbol_param=supports_symbol,
        observed_row_cap=response.row_count if response.is_success else 0,
        request_params=params,
        request_path="/".join(provider.health().get("base_url", "").split("/")[-1:]) or "/",
        schema_sha256=schema_sha,
        http_status=response.http_status,
    )


def run_permission_probe(
    config: ArchiveConfig,
    provider: ArchiveProvider,
    inventory: EndpointInventory,
    priorities: list[str] | None = None,
) -> dict[str, Any]:
    """Probe all enabled endpoints and write permission_report.json."""
    bucket = TokenBucket(config.rate_limit.calls_per_minute)
    client = RateLimitedClient(
        bucket,
        RetryPolicy(
            max_attempts=config.rate_limit.retry_attempts,
            retry_statuses=tuple(config.rate_limit.retry_statuses),
        ),
    )

    endpoints = inventory.list_by_priority(priorities)
    results: list[ProbeResult] = []
    for endpoint in endpoints:
        logger.info("权限探针: %s", endpoint.api_name)
        result = probe_endpoint(provider, endpoint, client)
        results.append(result)
        logger.info(
            "  -> %s rows=%d %.2fs %s",
            result.status, result.row_count, result.elapsed_seconds, result.message[:60],
        )
        # Small jitter between probes.
        time.sleep(0.2)

    summary_counts: dict[str, int] = {}
    for r in results:
        summary_counts[r.status] = summary_counts.get(r.status, 0) + 1

    report = {
        "schema_version": 2,
        "provider": config.provider.name,
        "probed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "priorities": priorities,
        "summary": {
            "total": len(results),
            **{s: summary_counts.get(s, 0) for s in PROBE_STATUSES},
        },
        "endpoints": [
            {
                "api_name": r.api_name,
                "status": r.status,
                "message": r.message,
                "request_path": r.request_path,
                "request_params": r.request_params,
                "columns": r.columns,
                "row_count": r.row_count,
                "sample_min_date": r.sample_min_date,
                "sample_max_date": r.sample_max_date,
                "elapsed_seconds": round(r.elapsed_seconds, 3),
                "supports_fields_param": r.supports_fields_param,
                "supports_date_param": r.supports_date_param,
                "supports_symbol_param": r.supports_symbol_param,
                "observed_row_cap": r.observed_row_cap,
                "schema_sha256": r.schema_sha256,
                "http_status": r.http_status,
            }
            for r in results
        ],
    }

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    path = config.reports_dir / "permission_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("权限报告已写入: %s", path)
    return report


def probe_results_for_inventory(report: dict[str, Any]) -> dict[str, Any]:
    """Condense a permission report into per-endpoint inventory annotations."""
    condensed: dict[str, Any] = {}
    for ep in report.get("endpoints", []):
        condensed[ep["api_name"]] = {
            "status": ep["status"],
            "message": ep["message"][:200],
            "row_count": ep["row_count"],
            "sample_min_date": ep["sample_min_date"],
            "sample_max_date": ep["sample_max_date"],
            "observed_row_cap": ep["observed_row_cap"],
            "schema_sha256": ep["schema_sha256"],
            "http_status": ep["http_status"],
        }
    return condensed
