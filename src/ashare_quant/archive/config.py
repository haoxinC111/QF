"""Archive configuration loader and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProviderConfig:
    kind: str
    name: str
    base_url_env: str
    token_env: str
    forbid_token_env: list[str] = field(default_factory=list)
    authorization_confirmed: bool = False
    local_archival_allowed: bool = False
    allowed_hosts: list[str] = field(default_factory=list)
    api_key_env: str | None = None
    api_key_header: str = "X-API-Key"
    request: dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitConfig:
    calls_per_minute: float = 75.0
    initial_workers: int = 1
    maximum_workers: int = 2
    promotion_after_successes: int = 1000
    retry_attempts: int = 5
    retry_statuses: list[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])


@dataclass
class ArchiveConfig:
    schema_version: int
    provider: ProviderConfig
    rate_limit: RateLimitConfig
    archive_root: Path
    raw_format: str
    table_format: str
    parquet_compression: str
    immutable_raw: bool
    save_all_fields: bool
    write_request_manifest: bool
    write_schema_fingerprint: bool
    checksum: str
    scope: dict[str, Any]
    completeness: dict[str, Any]
    point_in_time: dict[str, Any]
    batches: list[dict[str, Any]]

    @property
    def raw_dir(self) -> Path:
        return self.archive_root / "raw" / self.provider.name

    @property
    def bronze_dir(self) -> Path:
        return self.archive_root / "bronze" / self.provider.name

    @property
    def catalog_dir(self) -> Path:
        return self.archive_root / "catalog"

    @property
    def reports_dir(self) -> Path:
        return self.archive_root / "reports"

    def ensure_dirs(self) -> None:
        for directory in (self.raw_dir, self.bronze_dir, self.catalog_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: Path) -> "ArchiveConfig":
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if data.get("schema_version") != 1:
            raise ValueError(f"不支持的 archive schema 版本: {data.get('schema_version')}")

        provider_data = data.get("provider", {})
        provider = ProviderConfig(
            kind=provider_data.get("kind", "tushare_compatible_http"),
            name=provider_data.get("name", "research_proxy_unverified"),
            base_url_env=provider_data.get("base_url_env", "QF_ARCHIVE_API_URL"),
            token_env=provider_data.get("token_env", "QF_ARCHIVE_API_TOKEN"),
            forbid_token_env=provider_data.get("forbid_token_env", ["TUSHARE_TOKEN"]),
            authorization_confirmed=bool(provider_data.get("authorization_confirmed", False)),
            local_archival_allowed=bool(provider_data.get("local_archival_allowed", False)),
            allowed_hosts=provider_data.get("allowed_hosts", []),
            api_key_env=provider_data.get("api_key_env"),
            api_key_header=provider_data.get("api_key_header", "X-API-Key"),
            request=provider_data.get("request", {}),
        )

        rate_data = data.get("rate_limit", {})
        rate_limit = RateLimitConfig(
            calls_per_minute=float(rate_data.get("calls_per_minute", 75.0)),
            initial_workers=int(rate_data.get("initial_workers", 1)),
            maximum_workers=int(rate_data.get("maximum_workers", 2)),
            promotion_after_successes=int(rate_data.get("promotion_after_successes", 1000)),
            retry_attempts=int(rate_data.get("retry_attempts", 5)),
            retry_statuses=[int(s) for s in rate_data.get("retry_statuses", [429, 500, 502, 503, 504])],
        )

        archive_data = data.get("archive", {})
        archive_root = Path(archive_data.get("root", "data_lake"))
        if not archive_root.is_absolute():
            archive_root = path.parent / archive_root

        return cls(
            schema_version=int(data["schema_version"]),
            provider=provider,
            rate_limit=rate_limit,
            archive_root=archive_root.resolve(),
            raw_format=archive_data.get("raw_format", "json.zst"),
            table_format=archive_data.get("table_format", "parquet"),
            parquet_compression=archive_data.get("parquet_compression", "zstd"),
            immutable_raw=bool(archive_data.get("immutable_raw", True)),
            save_all_fields=bool(archive_data.get("save_all_fields", True)),
            write_request_manifest=bool(archive_data.get("write_request_manifest", True)),
            write_schema_fingerprint=bool(archive_data.get("write_schema_fingerprint", True)),
            checksum=archive_data.get("checksum", "sha256"),
            scope=dict(data.get("scope", {})),
            completeness=dict(data.get("completeness", {})),
            point_in_time=dict(data.get("point_in_time", {})),
            batches=list(data.get("batches", [])),
        )

    def validate_for_run(self, *, batch_id: str | None = None) -> None:
        """Validate configuration and raise before any network call."""
        url = os.environ.get(self.provider.base_url_env)
        if not url:
            raise ValueError(
                f"归档网关 URL 未设置: {self.provider.base_url_env}"
            )
        token = os.environ.get(self.provider.token_env)
        if not token:
            raise ValueError(
                f"归档网关 Token 未设置: {self.provider.token_env}"
            )
        if self.provider.token_env in self.provider.forbid_token_env:
            raise ValueError(
                f"Token 环境变量 {self.provider.token_env} 在 forbid_token_env 中，"
                f"不能复用官方 Token 变量"
            )

        # Authorization gates only block full batches, not permission probes.
        if batch_id and batch_id != "A_probe":
            if not self.provider.authorization_confirmed:
                raise ValueError(
                    "provider.authorization_confirmed=false，禁止非探针批次下载"
                )
            if not self.provider.local_archival_allowed:
                raise ValueError(
                    "provider.local_archival_allowed=false，禁止本地长期归档"
                )

        if self.rate_limit.calls_per_minute <= 0:
            raise ValueError("rate_limit.calls_per_minute 必须大于 0")
        if self.rate_limit.initial_workers < 1:
            raise ValueError("rate_limit.initial_workers 必须至少为 1")

        if self.checksum != "sha256":
            raise ValueError(f"不支持的 checksum 算法: {self.checksum}")

    def redacted_copy(self) -> dict[str, Any]:
        """Return a config dict safe for logging (no Token)."""
        return {
            "schema_version": self.schema_version,
            "provider": {
                "kind": self.provider.kind,
                "name": self.provider.name,
                "base_url_env": self.provider.base_url_env,
                "token_env": self.provider.token_env,
                "authorization_confirmed": self.provider.authorization_confirmed,
                "local_archival_allowed": self.provider.local_archival_allowed,
                "allowed_hosts": self.provider.allowed_hosts,
            },
            "rate_limit": {
                "calls_per_minute": self.rate_limit.calls_per_minute,
                "initial_workers": self.rate_limit.initial_workers,
                "maximum_workers": self.rate_limit.maximum_workers,
            },
            "archive_root": str(self.archive_root),
        }
