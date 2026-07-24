"""Tests for the data archive V2 module."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd

# Allow direct unittest execution from repo root.
sys_path_inserted = False
if "src" not in os.environ.get("PYTHONPATH", ""):
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    sys_path_inserted = True

from ashare_quant.archive import (
    ArchiveConfig,
    ArchivePipeline,
    DownloadTask,
    MockArchiveProvider,
    TaskStateDB,
    TaskStatus,
    TushareCompatibleHttpProvider,
)
from ashare_quant.archive.config import ProviderConfig, RateLimitConfig
from ashare_quant.archive.pipeline import EndpointSpec, task_id
from ashare_quant.archive.probe import _default_probe_params, probe_endpoint
from ashare_quant.archive.provider import _redact_payload
from ashare_quant.archive.registry import InventoryEndpoint
from ashare_quant.archive.schema import schema_fingerprint
from ashare_quant.archive.storage import (
    load_bronze_parquet,
    load_raw_json_zst,
    save_bronze_parquet,
    save_raw_json_zst,
    sha256_file,
    store_response,
)
from ashare_quant.archive.throttle import RateLimitedClient, RetryPolicy, TokenBucket


class TestTokenRedaction(unittest.TestCase):
    def test_redacts_token_field(self) -> None:
        payload = json.dumps({"api_name": "daily", "token": "secret123", "params": {}})
        redacted = _redact_payload(payload)
        self.assertIn("<redacted>", redacted)
        self.assertNotIn("secret123", redacted)

    def test_redacts_api_key_header(self) -> None:
        payload = json.dumps({"X-API-Key": "dummy-api-key"})
        redacted = _redact_payload(payload)
        self.assertIn("<redacted>", redacted)
        self.assertNotIn("dummy-api-key", redacted)


class TestTushareCompatibleHttpProvider(unittest.TestCase):
    def test_requires_url_env(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                TushareCompatibleHttpProvider()
        self.assertIn("QF_ARCHIVE_API_URL", str(ctx.exception))

    def test_requires_https(self) -> None:
        env = {"QF_ARCHIVE_API_URL": "http://example.com", "QF_ARCHIVE_API_TOKEN": "t"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                TushareCompatibleHttpProvider()
        self.assertIn("HTTPS", str(ctx.exception))

    def test_forbids_official_token_env(self) -> None:
        env = {
            "QF_ARCHIVE_API_URL": "https://example.com",
            "QF_ARCHIVE_API_TOKEN": "ok",
            "TUSHARE_TOKEN": "official",
        }
        with patch.dict(os.environ, env, clear=True):
            provider = TushareCompatibleHttpProvider()
            self.assertEqual(provider._token, "ok")

    def test_respects_allowed_hosts(self) -> None:
        env = {"QF_ARCHIVE_API_URL": "https://fastapic.example.com", "QF_ARCHIVE_API_TOKEN": "t"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                TushareCompatibleHttpProvider(allowed_hosts=["other.example.com"])
        self.assertIn("不在允许列表", str(ctx.exception))


class TestArchiveConfigValidation(unittest.TestCase):
    def _make_config(self, **overrides: object) -> ArchiveConfig:
        provider = ProviderConfig(
            kind="tushare_compatible_http",
            name="test",
            base_url_env="QF_ARCHIVE_API_URL",
            token_env="QF_ARCHIVE_API_TOKEN",
            forbid_token_env=["TUSHARE_TOKEN"],
            authorization_confirmed=False,
            local_archival_allowed=False,
        )
        rate = RateLimitConfig()
        defaults = {
            "schema_version": 1,
            "provider": provider,
            "rate_limit": rate,
            "archive_root": Path("/tmp/test_lake"),
            "raw_format": "json.zst",
            "table_format": "parquet",
            "parquet_compression": "zstd",
            "immutable_raw": True,
            "save_all_fields": True,
            "write_request_manifest": True,
            "write_schema_fingerprint": True,
            "checksum": "sha256",
            "scope": {},
            "completeness": {},
            "point_in_time": {},
            "batches": [],
        }
        defaults.update(overrides)
        return ArchiveConfig(**defaults)

    def test_missing_url_token_raises(self) -> None:
        config = self._make_config()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                config.validate_for_run()
            self.assertIn("QF_ARCHIVE_API_URL", str(ctx.exception))

    def test_batch_b_blocked_without_authorization(self) -> None:
        config = self._make_config()
        env = {"QF_ARCHIVE_API_URL": "https://example.com", "QF_ARCHIVE_API_TOKEN": "t"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                config.validate_for_run(batch_id="B_p0_core")
            self.assertIn("authorization_confirmed", str(ctx.exception))

    def test_probe_allowed_without_authorization(self) -> None:
        config = self._make_config()
        env = {"QF_ARCHIVE_API_URL": "https://example.com", "QF_ARCHIVE_API_TOKEN": "t"}
        with patch.dict(os.environ, env, clear=True):
            config.validate_for_run(batch_id="A_probe")

    def test_redacted_copy_hides_token(self) -> None:
        config = self._make_config()
        redacted = config.redacted_copy()
        text = json.dumps(redacted)
        self.assertNotIn("secret", text.lower())
        # token_env is the name of the env var, not its value; that is acceptable.
        self.assertIn("token_env", text.lower())


class TestRawAndBronzeStorage(unittest.TestCase):
    def test_round_trip_json_zst(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json.zst"
            payload = b'{"api_name":"daily","items":[["000001.SZ"]]}'
            sha256, size = save_raw_json_zst(path, payload)
            # Digest semantics: SHA256 of the compressed bytes on disk, so the
            # catalog value can be re-verified with sha256_file at any time.
            self.assertEqual(sha256, sha256_file(path))
            self.assertGreater(size, 0)
            self.assertEqual(load_raw_json_zst(path), payload)

    def test_immutable_raw_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json.zst"
            save_raw_json_zst(path, b"a")
            with self.assertRaises(FileExistsError):
                save_raw_json_zst(path, b"b")

    def test_immutable_raw_allows_identical_rewrite(self) -> None:
        # Batch resume depends on this: re-downloading identical content to an
        # existing path must be accepted (idempotent), not raise.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json.zst"
            sha_first, _ = save_raw_json_zst(path, b'{"a":1}')
            sha_second, _ = save_raw_json_zst(path, b'{"a":1}')
            self.assertEqual(sha_first, sha_second)
            self.assertEqual(sha_first, sha256_file(path))

    def test_bronze_parquet_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.parquet"
            df = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]})
            save_bronze_parquet(path, df)
            loaded = load_bronze_parquet(path)
            pd.testing.assert_frame_equal(loaded, df)

    def test_store_response_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            bronze_dir = Path(tmp) / "bronze"
            stored = store_response(
                raw_dir=raw_dir,
                bronze_dir=bronze_dir,
                api_name="daily",
                params={"trade_date": "20250102"},
                columns=["ts_code", "close"],
                items=[["000001.SZ", 10.0]],
                raw_payload=b'{"items":[["000001.SZ",10.0]]}',
                snapshot_id="snap1",
                partition_key="trade_date=20250102",
            )
            self.assertEqual(stored.row_count, 1)
            self.assertTrue(stored.raw_path.exists())
            self.assertTrue(stored.bronze_path.exists())
            self.assertEqual(len(stored.raw_sha256), 64)


class TestTaskStateDB(unittest.TestCase):
    def test_persist_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = TaskStateDB(Path(tmp) / "tasks.db")
            task = EndpointSpec(
                api_name="daily",
                dataset="market_daily",
                priority="P0",
                primary_key=["ts_code", "trade_date"],
                primary_split="trade_date",
                fallback_split="ts_code",
                all_fields=True,
                fields="",
                params_template={"trade_date": "20250102"},
            )
            from ashare_quant.archive.pipeline import task_id

            tid = task_id("test", task.api_name, task.params_template, task.fields, "snap")
            db.upsert(
                DownloadTask(
                    task_id=tid,
                    api_name=task.api_name,
                    params=task.params_template,
                    fields=task.fields,
                    dataset=task.dataset,
                    priority=task.priority,
                    primary_key=task.primary_key,
                    primary_split=task.primary_split,
                    fallback_split=task.fallback_split,
                    status=TaskStatus.SUCCESS,
                    row_count=123,
                )
            )
            loaded = db.get(tid)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.status, TaskStatus.SUCCESS)
            self.assertEqual(loaded.row_count, 123)


class TestPipelineTruncationAndResume(unittest.TestCase):
    def _make_config(self, tmp: str) -> ArchiveConfig:
        provider = ProviderConfig(
            kind="mock",
            name="mock",
            base_url_env="QF_ARCHIVE_API_URL",
            token_env="QF_ARCHIVE_API_TOKEN",
        )
        return ArchiveConfig(
            schema_version=1,
            provider=provider,
            rate_limit=RateLimitConfig(calls_per_minute=1000),
            archive_root=Path(tmp),
            raw_format="json.zst",
            table_format="parquet",
            parquet_compression="zstd",
            immutable_raw=True,
            save_all_fields=True,
            write_request_manifest=True,
            write_schema_fingerprint=True,
            checksum="sha256",
            scope={},
            completeness={},
            point_in_time={},
            batches=[],
        )

    def test_skips_completed_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(tmp)
            provider = MockArchiveProvider()
            provider.register(
                "trade_cal",
                {"exchange": "SSE", "is_open": "1"},
                ["cal_date", "is_open"],
                [["20250102", "1"]],
            )
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            pipeline = ArchivePipeline(config, provider, db, snapshot_id="snap1")
            spec = EndpointSpec(
                api_name="trade_cal",
                dataset="calendar",
                priority="P0",
                primary_key=["cal_date"],
                primary_split=None,
                fallback_split=None,
                all_fields=True,
                fields="",
                params_template={"exchange": "SSE", "is_open": "1"},
            )
            result1 = pipeline.run_tasks([spec])
            self.assertEqual(result1.tasks_completed, 1)
            result2 = pipeline.run_tasks([spec])
            self.assertEqual(result2.tasks_completed, 1)  # Counted as completed.

    def test_truncation_split_by_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(tmp)
            # Provider returns exactly 3 rows for any date range, simulating a cap.
            provider = MockArchiveProvider()
            provider.register(
                "daily",
                {"start_date": "20250101", "end_date": "20250110"},
                ["ts_code", "trade_date", "close"],
                [["000001.SZ", "20250110", "10"]] * 3,
            )
            # 不重叠二分: 左 [0101,0105], 右 [0106,0110]。
            provider.register(
                "daily",
                {"start_date": "20250101", "end_date": "20250105"},
                ["ts_code", "trade_date", "close"],
                [["000001.SZ", "20250105", "10"]],
            )
            provider.register(
                "daily",
                {"start_date": "20250106", "end_date": "20250110"},
                ["ts_code", "trade_date", "close"],
                [["000001.SZ", "20250110", "10"]],
            )
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            pipeline = ArchivePipeline(config, provider, db, snapshot_id="snap1")
            pipeline.observed_caps["daily"] = 3
            spec = EndpointSpec(
                api_name="daily",
                dataset="market_daily",
                priority="P0",
                primary_key=["ts_code", "trade_date"],
                primary_split="trade_date",
                fallback_split="ts_code",
                all_fields=True,
                fields="",
                params_template={"start_date": "20250101", "end_date": "20250110"},
            )
            result = pipeline.run_tasks([spec])
            # Parent fails; two sub-tasks succeed.
            self.assertEqual(result.tasks_completed, 2)


# ---------------------------------------------------------------------------
# Fake HTTP layer for provider tests (no real network).
# ---------------------------------------------------------------------------

_HTTP_ENV = {
    "QF_ARCHIVE_API_URL": "https://proxy.example.com/pro",
    "QF_ARCHIVE_API_TOKEN": "qf-secret-token",
}


def _ok_payload(rows: int = 1) -> dict[str, Any]:
    return {
        "code": 0,
        "data": {
            "fields": ["ts_code", "close"],
            "items": [["000001.SZ", 10.0 + i] for i in range(rows)],
        },
    }


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(
        self,
        status_code: int = 200,
        payload: dict[str, Any] | None = None,
        raw: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        if raw is not None:
            self.content = raw
        elif payload is not None:
            self.content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        else:
            self.content = b""
        self.text = self.content.decode("utf-8", errors="replace")
        self.encoding = "utf-8"

    @property
    def is_redirect(self) -> bool:
        return self.status_code in (301, 302, 303, 307, 308)

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))


class FakeSession:
    """Queue-based fake session; captures every outgoing request."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.headers: dict[str, str] = {}
        self.requests: list[dict[str, Any]] = []

    def _next(self) -> FakeResponse:
        if not self._responses:
            raise AssertionError("FakeSession 响应队列已空")
        return self._responses.pop(0)

    def post(self, url: str, data: bytes | None = None, **kwargs: Any) -> FakeResponse:
        self.requests.append({"method": "POST", "url": url, "data": data})
        return self._next()

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"method": "GET", "url": url, "data": None})
        return self._next()


def _http_provider(session: FakeSession, extra_env: dict[str, str] | None = None) -> TushareCompatibleHttpProvider:
    env = dict(_HTTP_ENV)
    if extra_env:
        env.update(extra_env)
    with patch.dict(os.environ, env, clear=True):
        return TushareCompatibleHttpProvider(session=session)


def _fast_client(max_attempts: int = 5) -> RateLimitedClient:
    return RateLimitedClient(
        TokenBucket(100_000),
        RetryPolicy(
            max_attempts=max_attempts,
            retry_statuses=(429, 500, 502, 503, 504),
            base_delay_seconds=0.001,
            max_delay_seconds=0.005,
        ),
    )


class TestProviderHttpLayer(unittest.TestCase):
    """Items 2/3/4/5/6/7/8/9 of the Phase A.1 test matrix."""

    def test_official_token_never_sent_to_gateway(self) -> None:
        # Item 2: even when TUSHARE_TOKEN is present, only QF token goes out.
        fake = FakeSession([FakeResponse(200, payload=_ok_payload())])
        provider = _http_provider(fake, {"TUSHARE_TOKEN": "official-secret"})
        resp = provider.request("daily", {"trade_date": "20250102"})
        self.assertTrue(resp.is_success)
        sent = fake.requests[0]["data"].decode("utf-8")
        self.assertNotIn("official-secret", sent)
        self.assertEqual(json.loads(sent)["token"], "qf-secret-token")

    def test_error_messages_do_not_leak_token(self) -> None:
        # Item 1 (extension): error text and logs must not carry the token.
        fake = FakeSession([FakeResponse(500, raw=b"upstream boom")])
        provider = _http_provider(fake)
        with self.assertLogs("ashare_quant.archive.provider", level="WARNING") as logs:
            resp = provider.request("daily", {"trade_date": "20250102"})
        self.assertEqual(resp.status, "transient_error")
        self.assertNotIn("qf-secret-token", resp.message)
        self.assertNotIn("qf-secret-token", "\n".join(logs.output))

    def test_cross_host_redirect_rejected(self) -> None:
        # Item 3: a 302 to an unregistered host must not be followed.
        fake = FakeSession(
            [FakeResponse(302, headers={"Location": "https://evil.example.com/pro"})]
        )
        provider = _http_provider(fake)
        resp = provider.request("daily", {"trade_date": "20250102"})
        self.assertEqual(resp.status, "transient_error")
        self.assertIn("禁止跨域跳转", resp.message)
        # Only the original POST was made; the redirect target was never hit.
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(fake.requests[0]["method"], "POST")

    def test_401_403_classified_denied(self) -> None:
        # Item 4.
        fake = FakeSession([FakeResponse(401), FakeResponse(403)])
        provider = _http_provider(fake)
        for expected_http in (401, 403):
            resp = provider.request("daily", {"trade_date": "20250102"})
            self.assertEqual(resp.status, "denied")
            self.assertEqual(resp.http_status, expected_http)

    def test_429_retried_until_success(self) -> None:
        # Item 5a: 429 must be retried, not classified as invalid params.
        fake = FakeSession(
            [FakeResponse(429), FakeResponse(429), FakeResponse(200, payload=_ok_payload())]
        )
        provider = _http_provider(fake)
        resp = _fast_client().call(provider.request, "daily", {"trade_date": "20250102"})
        self.assertTrue(resp.is_success)
        self.assertEqual(len(fake.requests), 3)

    def test_5xx_sequence_retried(self) -> None:
        # Item 6: 500/502/503/504 are all transient and retried.
        fake = FakeSession(
            [
                FakeResponse(500),
                FakeResponse(502),
                FakeResponse(503),
                FakeResponse(504),
                FakeResponse(200, payload=_ok_payload()),
            ]
        )
        provider = _http_provider(fake)
        resp = _fast_client().call(provider.request, "daily", {"trade_date": "20250102"})
        self.assertTrue(resp.is_success)
        self.assertEqual(len(fake.requests), 5)

    def test_retry_exhaustion_returns_transient_error(self) -> None:
        # Item 6 (boundary): after the retry budget, surface transient_error.
        fake = FakeSession([FakeResponse(500), FakeResponse(500), FakeResponse(500)])
        provider = _http_provider(fake)
        resp = _fast_client(max_attempts=3).call(
            provider.request, "daily", {"trade_date": "20250102"}
        )
        self.assertEqual(resp.status, "transient_error")
        self.assertEqual(resp.http_status, 500)
        self.assertEqual(len(fake.requests), 3)

    def test_http_200_html_body_is_transient(self) -> None:
        # Item 7: HTTP 200 with an HTML error page is not valid data.
        fake = FakeSession([FakeResponse(200, raw=b"<html><body>Bad Gateway</body></html>")])
        provider = _http_provider(fake)
        resp = provider.request("daily", {"trade_date": "20250102"})
        self.assertEqual(resp.status, "transient_error")
        self.assertIn("不是合法 JSON", resp.message)

    def test_http_200_malformed_structure(self) -> None:
        # Item 8: valid JSON but broken structure / error code.
        fake = FakeSession(
            [
                FakeResponse(200, payload={"code": 0, "data": {"fields": [], "items": "oops"}}),
                FakeResponse(200, payload={"code": 40001, "msg": "参数错误"}),
            ]
        )
        provider = _http_provider(fake)
        resp = provider.request("daily", {"trade_date": "20250102"})
        self.assertEqual(resp.status, "transient_error")
        self.assertIn("items", resp.message)
        resp = provider.request("daily", {"trade_date": "20250102"})
        self.assertEqual(resp.status, "invalid_params")

    def test_data_wrapped_payload_compatible(self) -> None:
        # Item 9: both wrapped {"code":0,"data":{...}} and flat formats parse.
        fake = FakeSession(
            [
                FakeResponse(200, payload=_ok_payload(rows=2)),
                FakeResponse(
                    200,
                    payload={"fields": ["ts_code"], "items": [["000002.SZ"]]},
                ),
            ]
        )
        provider = _http_provider(fake)
        resp = provider.request("daily", {"trade_date": "20250102"})
        self.assertTrue(resp.is_success)
        self.assertEqual(resp.columns, ["ts_code", "close"])
        self.assertEqual(resp.row_count, 2)
        resp = provider.request("daily", {"trade_date": "20250103"})
        self.assertTrue(resp.is_success)
        self.assertEqual(resp.columns, ["ts_code"])
        self.assertEqual(resp.items, [["000002.SZ"]])


class TestGlobalRateLimit(unittest.TestCase):
    """Item 5b: one shared token bucket paces all workers globally."""

    def test_shared_bucket_paces_clients(self) -> None:
        bucket = TokenBucket(60)  # 1 req/s, initial burst of 1 token.
        policy = RetryPolicy(max_attempts=1)
        client_a = RateLimitedClient(bucket, policy)
        client_b = RateLimitedClient(bucket, policy)
        self.assertIs(client_a.bucket, client_b.bucket)

        fake = FakeSession(
            [FakeResponse(200, payload=_ok_payload()), FakeResponse(200, payload=_ok_payload())]
        )
        provider = _http_provider(fake)
        start = time.monotonic()
        client_a.call(provider.request, "daily", {"trade_date": "20250102"})
        # After the first call the shared bucket is empty for everyone.
        self.assertGreater(bucket.estimate_wait(), 0.3)
        client_b.call(provider.request, "daily", {"trade_date": "20250103"})
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 0.8)


class TestPipelineHelpers(unittest.TestCase):
    def _make_config(self, tmp: str, provider_name: str = "mock", **rate: Any) -> ArchiveConfig:
        provider = ProviderConfig(
            kind="mock",
            name=provider_name,
            base_url_env="QF_ARCHIVE_API_URL",
            token_env="QF_ARCHIVE_API_TOKEN",
        )
        return ArchiveConfig(
            schema_version=1,
            provider=provider,
            rate_limit=RateLimitConfig(calls_per_minute=10000, **rate),
            archive_root=Path(tmp),
            raw_format="json.zst",
            table_format="parquet",
            parquet_compression="zstd",
            immutable_raw=True,
            save_all_fields=True,
            write_request_manifest=True,
            write_schema_fingerprint=True,
            checksum="sha256",
            scope={},
            completeness={},
            point_in_time={},
            batches=[],
        )

    def _spec(self, api_name: str, params: dict[str, Any], **kw: Any) -> EndpointSpec:
        return EndpointSpec(
            api_name=api_name,
            dataset=kw.get("dataset", "market_daily"),
            priority="P0",
            primary_key=kw.get("primary_key", ["ts_code", "trade_date"]),
            primary_split=kw.get("primary_split", "trade_date"),
            fallback_split=kw.get("fallback_split", "ts_code"),
            all_fields=True,
            fields="",
            params_template=params,
        )

    def _pipeline(self, tmp: str) -> ArchivePipeline:
        config = self._make_config(tmp)
        db = TaskStateDB(config.catalog_dir / "tasks.db")
        return ArchivePipeline(config, MockArchiveProvider(), db, snapshot_id="snap1")

    def test_partition_key_disambiguates_symbol_date_range_siblings(self) -> None:
        """index_daily 同 ts_code 按年度切段:partition_key 必须含日期段,否则撞名."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            spec = self._spec("index_daily", {}, primary_split="ts_code")
            key_2020 = pipeline._partition_key(
                spec, {"ts_code": "000001.SH", "start_date": "20200101", "end_date": "20201231"}
            )
            key_2021 = pipeline._partition_key(
                spec, {"ts_code": "000001.SH", "start_date": "20210101", "end_date": "20211231"}
            )
            self.assertNotEqual(key_2020, key_2021)
            self.assertIn("ts_code=000001.SH", key_2020)
            self.assertIn("start_date=20200101", key_2020)
            self.assertIn("end_date=20201231", key_2020)

    def test_partition_key_single_day_not_duplicated(self) -> None:
        """split 键本身是 trade_date 时不得重复追加."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            spec = self._spec("daily", {"trade_date": "20250102"})
            key = pipeline._partition_key(spec, {"trade_date": "20250102"})
            self.assertEqual(key, "trade_date=20250102")

    def test_partition_key_symbol_only_unchanged(self) -> None:
        """无日期键的 symbol 全历史任务保持原命名."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            spec = self._spec("dividend", {}, primary_split="ts_code", fallback_split=None)
            key = pipeline._partition_key(spec, {"ts_code": "600036.SH"})
            self.assertEqual(key, "ts_code=600036.SH")

    def test_partition_key_symbol_period_pair_unique(self) -> None:
        """财务 VIP 接口 ts_code x period 组合:period 是维度键,不得撞名."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            spec = self._spec(
                "income_vip", {}, primary_split="period", primary_key=["ts_code", "end_date"]
            )
            key_a = pipeline._partition_key(spec, {"ts_code": "600036.SH", "period": "20240331"})
            key_b = pipeline._partition_key(spec, {"ts_code": "600036.SH", "period": "20240630"})
            self.assertNotEqual(key_a, key_b)
            self.assertIn("period=20240331", key_a)
            self.assertIn("ts_code=600036.SH", key_a)


class TestSymbolUniverseFallbackSplit(TestPipelineHelpers):
    """Item 11: single-day request hitting the cap splits by ts_code."""

    def test_single_day_cap_splits_by_symbol_universe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(tmp)
            cols = ["ts_code", "trade_date", "close"]
            provider = MockArchiveProvider()
            provider.register("daily", {"trade_date": "20250102"}, cols, [["X", "20250102", 1]] * 3)
            provider.register(
                "daily",
                {"trade_date": "20250102", "ts_code": "A.SZ,B.SZ"},
                cols,
                [["A.SZ", "20250102", 1], ["B.SZ", "20250102", 2]],
            )
            provider.register(
                "daily",
                {"trade_date": "20250102", "ts_code": "C.SZ,D.SZ"},
                cols,
                [["C.SZ", "20250102", 3], ["D.SZ", "20250102", 4]],
            )
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            pipeline = ArchivePipeline(
                config,
                provider,
                db,
                snapshot_id="snap1",
                symbol_universe=["A.SZ", "B.SZ", "C.SZ", "D.SZ"],
            )
            pipeline.observed_caps["daily"] = 3
            spec = self._spec("daily", {"trade_date": "20250102"})
            result = pipeline.run_tasks([spec])
            self.assertEqual(result.tasks_completed, 2)
            self.assertEqual(result.rows_total, 4)
            # Both symbol chunks persisted as separate partitions.
            bronze_files = list((config.bronze_dir / "daily").glob("*.parquet"))
            self.assertEqual(len(bronze_files), 2)


class TestStockBasicPartitions(TestPipelineHelpers):
    """Item 12: stock_basic L/D/P write distinct partitions."""

    def test_list_status_partitions_do_not_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(tmp)
            cols = ["ts_code", "name"]
            provider = MockArchiveProvider()
            for status, code in (("L", "000001.SZ"), ("D", "000002.SZ"), ("P", "000003.SZ")):
                provider.register(
                    "stock_basic", {"list_status": status}, cols, [[code, f"name-{status}"]]
                )
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            pipeline = ArchivePipeline(config, provider, db, snapshot_id="snap1")
            specs = [
                self._spec(
                    "stock_basic",
                    {"list_status": s},
                    primary_key=["ts_code"],
                    primary_split=None,
                    fallback_split=None,
                )
                for s in ("L", "D", "P")
            ]
            result = pipeline.run_tasks(specs)
            self.assertEqual(result.tasks_completed, 3)
            raw_files = sorted(p.name for p in (config.raw_dir / "snap1" / "stock_basic").glob("*.json.zst"))
            self.assertEqual(len(raw_files), 3)
            for status in ("L", "D", "P"):
                self.assertTrue(any(f"list_status={status}" in name for name in raw_files))


class TestFinancialTsCodeGuard(TestPipelineHelpers):
    """Item 13: financial endpoints missing ts_code get one auto-filled for probes."""

    def test_probe_auto_fills_ts_code(self) -> None:
        endpoint = InventoryEndpoint(
            api_name="income_vip",
            priority="P0",
            dataset="financial_income",
            primary_key=["ts_code", "end_date", "report_type"],
            primary_split="period",
            fallback_split="ts_code",
            params={"period": "20241231"},
        )
        params = _default_probe_params(endpoint)
        self.assertIn("ts_code", params)

        provider = MockArchiveProvider()
        seen: dict[str, Any] = {}
        original = provider.request

        def recording(api_name: str, params: dict[str, Any] | None = None, *, fields: str = ""):
            seen.update(params or {})
            return original(api_name, params, fields=fields)

        provider.request = recording  # type: ignore[assignment]
        client = RateLimitedClient(TokenBucket(10000), RetryPolicy(max_attempts=1))
        probe_endpoint(provider, endpoint, client)
        self.assertEqual(seen.get("ts_code"), "000001.SZ")
        self.assertEqual(seen.get("period"), "20241231")


class TestEmptyVsTransient(TestPipelineHelpers):
    """Item 14: confirmed_empty and transient_error stay distinct."""

    def test_true_empty_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(tmp)
            provider = MockArchiveProvider()  # nothing registered -> empty
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            pipeline = ArchivePipeline(config, provider, db, snapshot_id="snap1")
            spec = self._spec("daily", {"trade_date": "20250102"})
            result = pipeline.run_tasks([spec])
            self.assertEqual(result.tasks_empty, 1)
            task = db.list_tasks()[0]
            self.assertEqual(task.status, TaskStatus.CONFIRMED_EMPTY)

    def test_server_error_marked_retryable_not_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(tmp, provider_name="http_test", retry_attempts=1)
            fake = FakeSession([FakeResponse(500, raw=b"boom")])
            provider = _http_provider(fake)
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            pipeline = ArchivePipeline(config, provider, db, snapshot_id="snap1")
            spec = self._spec("daily", {"trade_date": "20250102"})
            result = pipeline.run_tasks([spec])
            self.assertEqual(result.tasks_failed, 1)
            self.assertEqual(result.tasks_empty, 0)
            task = db.list_tasks()[0]
            self.assertEqual(task.status, TaskStatus.RETRYABLE_ERROR)
            # Nothing was persisted for a transient failure.
            self.assertFalse((config.raw_dir / "snap1").exists())


class TestAtomicPublish(unittest.TestCase):
    """Item 16: failed writes leave no half-written tmp files."""

    def test_bronze_failure_leaves_no_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.parquet"
            df = pd.DataFrame({"a": [1]})
            with patch.object(pd.DataFrame, "to_parquet", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_bronze_parquet(path, df)
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])
            self.assertFalse(path.exists())

    def test_raw_failure_leaves_no_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json.zst"
            with patch.object(Path, "replace", side_effect=OSError("io error")):
                with self.assertRaises(OSError):
                    save_raw_json_zst(path, b'{"a":1}')
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])
            self.assertFalse(path.exists())


class TestIntegrityChecks(TestPipelineHelpers):
    """Items 17/18: row consistency and corruption detection."""

    def test_raw_bronze_row_count_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            items = [[f"00000{i}.SZ", 10.0 + i] for i in range(5)]
            payload = json.dumps(
                {"fields": ["ts_code", "close"], "items": items}, ensure_ascii=False
            ).encode("utf-8")
            stored = store_response(
                raw_dir=Path(tmp) / "raw",
                bronze_dir=Path(tmp) / "bronze",
                api_name="daily",
                params={"trade_date": "20250102"},
                columns=["ts_code", "close"],
                items=items,
                raw_payload=payload,
                snapshot_id="snap1",
                partition_key="trade_date=20250102",
            )
            raw_json = json.loads(load_raw_json_zst(stored.raw_path).decode("utf-8"))
            bronze_df = load_bronze_parquet(stored.bronze_path)
            self.assertEqual(len(raw_json["items"]), len(bronze_df))
            self.assertEqual(stored.row_count, len(bronze_df))

    def test_sha256_corruption_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json.zst"
            sha256, _ = save_raw_json_zst(path, b'{"items":[["a"]]}')
            # Corrupt the zstd frame header (magic number) on disk.
            blob = bytearray(path.read_bytes())
            blob[1] ^= 0xFF
            path.write_bytes(bytes(blob))
            self.assertNotEqual(sha256_file(path), sha256)
            with self.assertRaises(Exception):
                load_raw_json_zst(path)

    def test_payload_bitflip_breaks_json_and_sha(self) -> None:
        # A silent bitflip inside the payload is caught by checksum mismatch
        # even when the zstd frame itself still decodes.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json.zst"
            sha256, _ = save_raw_json_zst(path, b'{"items":[["a"]]}')
            blob = bytearray(path.read_bytes())
            blob[-2] ^= 0xFF
            path.write_bytes(bytes(blob))
            self.assertNotEqual(sha256_file(path), sha256)
            with self.assertRaises(Exception):
                json.loads(load_raw_json_zst(path).decode("utf-8"))


class TestSchemaDriftBlocked(TestPipelineHelpers):
    """Item 19: a changed column layout quarantines the task."""

    def test_schema_drift_quarantines_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(tmp)
            provider = MockArchiveProvider()
            provider.register(
                "daily", {"trade_date": "20250101"}, ["ts_code", "close"], [["000001.SZ", 10.0]]
            )
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            pipeline = ArchivePipeline(config, provider, db, snapshot_id="snap1")
            result1 = pipeline.run_tasks([self._spec("daily", {"trade_date": "20250101"})])
            self.assertEqual(result1.tasks_completed, 1)

            # Same endpoint, new column appears -> drift must block persistence.
            provider.register(
                "daily",
                {"trade_date": "20250102"},
                ["ts_code", "close", "extra_col"],
                [["000001.SZ", 10.0, "x"]],
            )
            result2 = pipeline.run_tasks([self._spec("daily", {"trade_date": "20250102"})])
            self.assertEqual(result2.tasks_completed, 0)
            self.assertEqual(result2.tasks_failed, 1)
            task = [t for t in db.list_tasks() if t.params.get("trade_date") == "20250102"][0]
            self.assertEqual(task.status, TaskStatus.QUARANTINED)
            self.assertIn("schema 漂移", task.last_error)
            raw_files = list((config.raw_dir / "snap1" / "daily").glob("*.json.zst"))
            self.assertEqual(len(raw_files), 1)  # only the first task persisted

    def test_registered_variant_passes_drift_guard(self) -> None:
        # Historical schema variants (e.g. daily_basic's early short-column
        # era) become legal once registered; only never-seen layouts block.
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(tmp)
            provider = MockArchiveProvider()
            provider.register(
                "daily_basic", {"trade_date": "20250101"},
                ["ts_code", "close", "pe"], [["000001.SZ", 10.0, 8.0]],
            )
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            pipeline = ArchivePipeline(config, provider, db, snapshot_id="snap1")
            result1 = pipeline.run_tasks([self._spec("daily_basic", {"trade_date": "20250101"})])
            self.assertEqual(result1.tasks_completed, 1)

            # Register the early short-column variant manually (human-verified).
            early_cols = ["ts_code", "pe"]
            pipeline.schema_registry.save(
                "daily_basic",
                schema_fingerprint(early_cols),
                {
                    "endpoint": "daily_basic",
                    "fingerprint": schema_fingerprint(early_cols),
                    "columns": early_cols,
                    "snapshot_id": "manual_registration",
                    "row_count": 0,
                },
            )
            provider.register(
                "daily_basic", {"trade_date": "20140108"},
                early_cols, [["000001.SZ", 8.0]],
            )
            result2 = pipeline.run_tasks([self._spec("daily_basic", {"trade_date": "20140108"})])
            self.assertEqual(result2.tasks_completed, 1)
            task = [t for t in db.list_tasks() if t.params.get("trade_date") == "20140108"][0]
            self.assertEqual(task.status, TaskStatus.SUCCESS)

            # A THIRD, never-registered layout is still quarantined.
            provider.register(
                "daily_basic", {"trade_date": "20140109"},
                ["ts_code", "pe", "mystery"], [["000001.SZ", 8.0, 1.0]],
            )
            result3 = pipeline.run_tasks([self._spec("daily_basic", {"trade_date": "20140109"})])
            self.assertEqual(result3.tasks_failed, 1)
            task3 = [t for t in db.list_tasks() if t.params.get("trade_date") == "20140109"][0]
            self.assertEqual(task3.status, TaskStatus.QUARANTINED)


class TestIdempotentRerun(TestPipelineHelpers):
    """Item 20: rerunning the same task changes nothing."""

    def test_rerun_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(tmp)
            provider = MockArchiveProvider()
            provider.register(
                "daily", {"trade_date": "20250102"}, ["ts_code", "close"], [["000001.SZ", 10.0]]
            )
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            pipeline = ArchivePipeline(config, provider, db, snapshot_id="snap1")
            spec = self._spec("daily", {"trade_date": "20250102"})
            result1 = pipeline.run_tasks([spec])
            self.assertEqual(result1.tasks_completed, 1)

            raw_files = list((config.raw_dir / "snap1" / "daily").glob("*.json.zst"))
            sha_before = sha256_file(raw_files[0])

            result2 = pipeline.run_tasks([spec])
            self.assertEqual(result2.tasks_completed, 1)
            self.assertEqual(sha256_file(raw_files[0]), sha_before)
            self.assertEqual(
                len(list((config.raw_dir / "snap1" / "daily").glob("*.json.zst"))), 1
            )
            self.assertEqual(len(list((config.bronze_dir / "daily").glob("*.parquet"))), 1)
            task = db.list_tasks()[0]
            self.assertEqual(task.attempts, 1)  # second run skipped execution


class TestCrossProviderIsolation(TestPipelineHelpers):
    """Item 21: same primary key from two providers never silently overwrites."""

    def test_providers_write_isolated_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_a = self._make_config(tmp, provider_name="proxy_a")
            config_b = self._make_config(tmp, provider_name="proxy_b")
            cols = ["ts_code", "close"]
            provider_a = MockArchiveProvider(source_provider="proxy_a")
            provider_b = MockArchiveProvider(source_provider="proxy_b")
            provider_a.register("daily", {"trade_date": "20250102"}, cols, [["000001.SZ", 10.0]])
            provider_b.register("daily", {"trade_date": "20250102"}, cols, [["000001.SZ", 99.0]])
            db = TaskStateDB(config_a.catalog_dir / "tasks.db")  # shared state db
            spec = self._spec("daily", {"trade_date": "20250102"})

            pipe_a = ArchivePipeline(config_a, provider_a, db, snapshot_id="snap1")
            pipe_b = ArchivePipeline(config_b, provider_b, db, snapshot_id="snap1")
            self.assertEqual(pipe_a.run_tasks([spec]).tasks_completed, 1)
            self.assertEqual(pipe_b.run_tasks([spec]).tasks_completed, 1)

            raw_a = list((config_a.raw_dir / "snap1" / "daily").glob("*.json.zst"))
            raw_b = list((config_b.raw_dir / "snap1" / "daily").glob("*.json.zst"))
            self.assertEqual(len(raw_a), 1)
            self.assertEqual(len(raw_b), 1)
            self.assertNotEqual(raw_a[0], raw_b[0])
            # Both payloads survive with their own values.
            items_a = json.loads(load_raw_json_zst(raw_a[0]).decode("utf-8"))["items"]
            items_b = json.loads(load_raw_json_zst(raw_b[0]).decode("utf-8"))["items"]
            self.assertEqual(items_a[0][1], 10.0)
            self.assertEqual(items_b[0][1], 99.0)
            # Task ids differ per provider in the shared state db.
            tasks = db.list_tasks(api_name="daily")
            self.assertEqual(len(tasks), 2)


class TestBatchExpansion(unittest.TestCase):
    """Batch runner period iterators and endpoint expansion."""

    def test_iter_months_covers_window(self) -> None:
        from ashare_quant.archive.batch import iter_months

        months = iter_months("20250115", "20250310")
        self.assertEqual(months[0], ("20250115", "20250131"))
        self.assertEqual(months[1], ("20250201", "20250228"))
        self.assertEqual(months[2], ("20250301", "20250310"))
        # No gaps and no overlaps.
        self.assertEqual(len(months), 3)

    def test_iter_quarters(self) -> None:
        from ashare_quant.archive.batch import iter_quarters

        periods = iter_quarters("20240101", "20241231")
        self.assertEqual(periods, ["20240331", "20240630", "20240930", "20241231"])

    def test_expand_trade_date_and_symbol(self) -> None:
        from ashare_quant.archive.batch import BatchContext, expand_endpoint

        ctx = BatchContext(
            universe=["000001.SZ", "600000.SH"],
            trade_dates=["20250102", "20250103"],
            latest_trade_date="20250103",
            latest_report_period="20241231",
        )
        daily_ep = InventoryEndpoint(
            api_name="daily",
            priority="P0",
            dataset="market_daily",
            primary_key=["ts_code", "trade_date"],
            primary_split="trade_date",
            fallback_split="ts_code",
            split_unit="trade_date",
            earliest_date="19901219",
        )
        specs = expand_endpoint(daily_ep, ctx)
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].params_template["trade_date"], "20250102")

        fin_ep = InventoryEndpoint(
            api_name="income_vip",
            priority="P0",
            dataset="financial_pit",
            primary_key=["ts_code", "end_date", "report_type", "update_flag"],
            primary_split="period",
            fallback_split="ts_code",
            split_unit="symbol",
            earliest_date="19950331",
        )
        specs = expand_endpoint(fin_ep, ctx)
        self.assertEqual(len(specs), 2)
        for spec in specs:
            # Full window attached so cap hits can bisect by date range.
            self.assertIn("start_date", spec.params_template)
            self.assertIn("end_date", spec.params_template)
            self.assertIn("ts_code", spec.params_template)

    def test_expand_index_weight_uses_index_code(self) -> None:
        from ashare_quant.archive.batch import BatchContext, expand_endpoint

        ctx = BatchContext(
            universe=[],
            trade_dates=[],
            latest_trade_date="20251231",
            latest_report_period="20250930",
            index_codes=["000300.SH"],
        )
        ep = InventoryEndpoint(
            api_name="index_weight",
            priority="P0",
            dataset="index_membership",
            primary_key=["index_code", "trade_date", "con_code"],
            primary_split="trade_date",
            fallback_split=None,
            split_unit="index_year",
            earliest_date="20240101",
        )
        specs = expand_endpoint(ep, ctx)
        self.assertEqual(len(specs), 2)  # 2024 + 2025
        self.assertEqual(specs[0].params_template["index_code"], "000300.SH")
        self.assertNotIn("ts_code", specs[0].params_template)

    def test_expand_index_basic_segments_cover_market_and_csi_categories(self) -> None:
        """index_basic 分 market + CSI category 段,规避 8000 行截断."""
        from ashare_quant.archive.batch import (
            CSI_INDEX_CATEGORIES,
            INDEX_BASIC_MARKETS,
            BatchContext,
            expand_endpoint,
        )

        ctx = BatchContext(
            universe=[], trade_dates=[], latest_trade_date="20251231",
            latest_report_period="20250930",
        )
        ep = InventoryEndpoint(
            api_name="index_basic",
            priority="P0",
            dataset="index_metadata",
            primary_key=["ts_code"],
            primary_split=None,
            fallback_split=None,
            split_unit="index_basic_segments",
        )
        specs = expand_endpoint(ep, ctx)
        # CSI 只走 category 细分段,不出现在单 market 段里(单段全字段会被截断)。
        self.assertEqual(len(specs), len(INDEX_BASIC_MARKETS) - 1 + len(CSI_INDEX_CATEGORIES))
        seen = [tuple(sorted(s.params_template.items())) for s in specs]
        self.assertEqual(len(seen), len(set(seen)), "分段不得重复")
        markets = {s.params_template["market"] for s in specs if "category" not in s.params_template}
        self.assertEqual(markets, set(INDEX_BASIC_MARKETS) - {"CSI"})
        csi_cats = {s.params_template["category"] for s in specs if "category" in s.params_template}
        self.assertEqual(csi_cats, set(CSI_INDEX_CATEGORIES))
        for s in specs:
            if "category" in s.params_template:
                self.assertEqual(s.params_template["market"], "CSI")

    def test_expand_index_year_main_uses_main_universe(self) -> None:
        """主力宇宙:仅用 index_codes_main,忽略全量 index_codes."""
        from ashare_quant.archive.batch import BatchContext, expand_endpoint

        ctx = BatchContext(
            universe=[],
            trade_dates=[],
            latest_trade_date="20251231",
            latest_report_period="20250930",
            index_codes=["SHOULD_NOT_APPEAR"],
            index_codes_main=["000300.SH", "000905.SH"],
        )
        ep = InventoryEndpoint(
            api_name="index_weight",
            priority="P0",
            dataset="index_membership",
            primary_key=["index_code", "trade_date", "con_code"],
            primary_split="trade_date",
            fallback_split=None,
            split_unit="index_year_main",
            earliest_date="20250101",
        )
        specs = expand_endpoint(ep, ctx)
        self.assertEqual(len(specs), 2)  # 2 只主力指数 x 1 年段
        codes = {s.params_template["index_code"] for s in specs}
        self.assertEqual(codes, {"000300.SH", "000905.SH"})


class TestB2RepairRegressions(unittest.TestCase):
    """B2_repair 回归: 显式 fields、行数上限、不重叠二分、BISECTED 终态。

    背景(2026-07-20): 网关对 index_weight 单次响应有 7000 行硬上限且超限
    时静默保留尾部月份;个别全年查询还会返回仅 con_code 一列的畸形 schema。
    以下测试防止显式 fields 或上限配置回退。
    """

    def _make_config(self, tmp: str) -> ArchiveConfig:
        provider = ProviderConfig(
            kind="mock",
            name="mock",
            base_url_env="QF_ARCHIVE_API_URL",
            token_env="QF_ARCHIVE_API_TOKEN",
        )
        return ArchiveConfig(
            schema_version=1,
            provider=provider,
            rate_limit=RateLimitConfig(calls_per_minute=1000),
            archive_root=Path(tmp),
            raw_format="json.zst",
            table_format="parquet",
            parquet_compression="zstd",
            immutable_raw=True,
            save_all_fields=True,
            write_request_manifest=True,
            write_schema_fingerprint=True,
            checksum="sha256",
            scope={},
            completeness={},
            point_in_time={},
            batches=[],
        )

    def test_index_weight_registry_fields_empty_and_cap(self) -> None:
        """index_weight 必须 fields=""(全粒度)且登记 row_cap=7000。

        显式 fields 会让网关只返回月末权重(丢月中调样快照);
        未登记 row_cap 则 7000 行硬上限的静默截断无法被检测。
        """
        from ashare_quant.archive.registry import default_inventory

        ep = default_inventory().endpoints["index_weight"]
        self.assertEqual(ep.fields, "")
        self.assertEqual(ep.row_cap, 7000)
        spec = ep.to_spec()
        self.assertEqual(spec.fields, "")
        self.assertEqual(spec.row_cap, 7000)

    def test_spec_row_cap_triggers_bisect_without_observed_cap(self) -> None:
        """规格级 row_cap 在没有任何运行时探测时也必须触发截断拆分。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(tmp)
            provider = MockArchiveProvider()
            provider.register(
                "index_weight",
                {"index_code": "X", "start_date": "20250101", "end_date": "20251231"},
                ["index_code", "con_code", "trade_date", "weight"],
                [["X", "000001.SZ", "20251231", 1.0]] * 3,
            )
            provider.register(
                "index_weight",
                {"index_code": "X", "start_date": "20250101", "end_date": "20250702"},
                ["index_code", "con_code", "trade_date", "weight"],
                [["X", "000001.SZ", "20250630", 1.0]],
            )
            provider.register(
                "index_weight",
                {"index_code": "X", "start_date": "20250703", "end_date": "20251231"},
                ["index_code", "con_code", "trade_date", "weight"],
                [["X", "000001.SZ", "20251231", 1.0]],
            )
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            pipeline = ArchivePipeline(config, provider, db, snapshot_id="snap_cap")
            # 注意: 不设置 pipeline.observed_caps,只靠规格级 row_cap。
            spec = EndpointSpec(
                api_name="index_weight",
                dataset="index_membership",
                priority="P0",
                primary_key=["index_code", "trade_date", "con_code"],
                primary_split="trade_date",
                fallback_split="index_code",
                all_fields=True,
                fields="index_code,con_code,trade_date,weight",
                params_template={"index_code": "X", "start_date": "20250101", "end_date": "20251231"},
                row_cap=3,
            )
            result = pipeline.run_tasks([spec])
            self.assertEqual(result.tasks_completed, 2)
            from ashare_quant.archive.pipeline import task_id

            parent_id = task_id("mock", "index_weight", dict(spec.params_template), spec.fields, "snap_cap")
            parent = db.get(parent_id)
            self.assertEqual(parent.status, TaskStatus.BISECTED)
            self.assertEqual(len(parent.metadata.get("bisected_into", [])), 2)
            # 不重叠且不丢天: 左 [0101,0702], 右 [0703,1231]。
            left = db.get(parent.metadata["bisected_into"][0])
            right = db.get(parent.metadata["bisected_into"][1])
            self.assertEqual(
                (left.params["start_date"], left.params["end_date"]),
                ("20250101", "20250702"),
            )
            self.assertEqual(
                (right.params["start_date"], right.params["end_date"]),
                ("20250703", "20251231"),
            )
            self.assertEqual(left.status, TaskStatus.SUCCESS)
            self.assertEqual(right.status, TaskStatus.SUCCESS)

    def test_unsplittable_parent_stays_suspect_truncated(self) -> None:
        """无法再拆分时父任务保持 suspect_truncated,进入 failure_queue。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(tmp)
            provider = MockArchiveProvider()
            provider.register(
                "index_weight",
                {"index_code": "X", "trade_date": "20250110"},
                ["index_code", "con_code", "trade_date", "weight"],
                [["X", "000001.SZ", "20250110", 1.0]] * 3,
            )
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            pipeline = ArchivePipeline(config, provider, db, snapshot_id="snap_unsplit")
            spec = EndpointSpec(
                api_name="index_weight",
                dataset="index_membership",
                priority="P0",
                primary_key=["index_code", "trade_date", "con_code"],
                primary_split="trade_date",
                fallback_split="index_code",
                all_fields=True,
                fields="index_code,con_code,trade_date,weight",
                params_template={"index_code": "X", "trade_date": "20250110"},
                row_cap=3,
            )
            pipeline.run_tasks([spec])
            from ashare_quant.archive.pipeline import task_id

            parent = db.get(task_id("mock", "index_weight", dict(spec.params_template), spec.fields, "snap_unsplit"))
            self.assertEqual(parent.status, TaskStatus.SUSPECT_TRUNCATED)


class TestB3RowCapRegressions(unittest.TestCase):
    """B3_financial 9 端点 row_cap 登记回归(2026-07-21 预检实测)。

    实测硬上限: balancesheet_vip=7000, income_vip=9000, fina_indicator_vip=12000,
    fina_mainbz_vip=10000(单季度恰满), disclosure_date=6000(单年恰满)。
    cashflow_vip/forecast_vip/express_vip/fina_audit 上限不可探(多码被拒或
    未触顶),登记保守默认 7000。B3 任务按 symbol 拆全历史,单股最大历史
    ~1,570 行(fina_mainbz_vip),上限本不可达;登记只为防御+重启后仍生效。
    """

    EXPECTED_CAPS = {
        "income_vip": 9000,
        "balancesheet_vip": 7000,
        "cashflow_vip": 7000,
        "fina_indicator_vip": 12000,
        "forecast_vip": 7000,
        "express_vip": 7000,
        "fina_audit": 7000,
        "fina_mainbz_vip": 10000,
        "disclosure_date": 6000,
    }

    def test_b3_endpoints_row_cap_registered(self) -> None:
        from ashare_quant.archive.registry import default_inventory

        endpoints = default_inventory().endpoints
        for api, cap in self.EXPECTED_CAPS.items():
            ep = endpoints[api]
            self.assertEqual(ep.batch, "B3_financial", api)
            self.assertEqual(ep.row_cap, cap, api)
            self.assertEqual(ep.to_spec().row_cap, cap, api)

    def test_b3_endpoints_fields_empty(self) -> None:
        """canary 实测(2026-07-21): 9 端点 fields="" 与显式 fields 主键集合
        与列完全一致;统一 fields="" 避免 index_weight 式粒度退化。"""
        from ashare_quant.archive.registry import default_inventory

        endpoints = default_inventory().endpoints
        for api in self.EXPECTED_CAPS:
            self.assertEqual(endpoints[api].fields, "", api)


class TestB3PrimaryKeyRegressions(unittest.TestCase):
    """B3 财务版本主键回归(2026-07-21 用户指令)。

    报表类主键必须纳入全部版本字段,确保同一报告期的修订/更正版本共存,
    Bronze 绝不按简化主键去重。列存在性依据 preflight/b3_columns.json 实测。
    """

    EXPECTED_PKS = {
        "income_vip": ["ts_code", "end_date", "ann_date", "f_ann_date", "report_type", "comp_type", "update_flag"],
        "balancesheet_vip": ["ts_code", "end_date", "ann_date", "f_ann_date", "report_type", "comp_type", "update_flag"],
        "cashflow_vip": ["ts_code", "end_date", "ann_date", "f_ann_date", "report_type", "comp_type", "update_flag"],
        "fina_indicator_vip": ["ts_code", "end_date", "ann_date", "update_flag"],
        "forecast_vip": ["ts_code", "end_date", "ann_date", "type", "update_flag"],
        "express_vip": ["ts_code", "end_date", "ann_date", "update_flag"],
        "fina_audit": ["ts_code", "end_date", "ann_date"],
        "fina_mainbz_vip": ["ts_code", "end_date", "bz_item", "bz_code"],
        "disclosure_date": ["ts_code", "end_date"],
    }

    def test_b3_primary_keys_include_revision_fields(self) -> None:
        from ashare_quant.archive.registry import default_inventory

        endpoints = default_inventory().endpoints
        for api, pk in self.EXPECTED_PKS.items():
            self.assertEqual(endpoints[api].primary_key, pk, api)

    def test_revision_rows_coexist_in_bronze(self) -> None:
        """同一 (ts_code,end_date) 的多版公告必须全部落盘,一行不丢。"""
        with tempfile.TemporaryDirectory() as tmp:
            from ashare_quant.archive.storage import load_bronze_parquet, store_response

            columns = ["ts_code", "end_date", "ann_date", "f_ann_date", "report_type", "comp_type", "update_flag", "revenue"]
            items = [
                ["000001.SZ", "20241231", "20250315", "20250315", "1", "2", "0", 100.0],   # 首版
                ["000001.SZ", "20241231", "20250420", "20250315", "1", "2", "1", 105.0],   # 修订版(update_flag 变化)
                ["000001.SZ", "20241231", "20250420", "20250315", "4", "2", "1", 105.0],   # 更正公告(report_type 变化)
            ]
            stored = store_response(
                Path(tmp) / "raw", Path(tmp) / "bronze", "income_vip", {},
                columns, items, b"{}", snapshot_id="snap_rev", partition_key="k",
            )
            df = load_bronze_parquet(stored.bronze_path)
            self.assertEqual(len(df), 3)
            self.assertEqual(sorted(df["revenue"]), [100.0, 105.0, 105.0])
            self.assertEqual(df["update_flag"].tolist(), ["0", "1", "1"])

    def test_aborted_prestart_status_exists(self) -> None:
        self.assertEqual(TaskStatus.ABORTED_PRESTART.value, "aborted_prestart")


class TestProviderRetryAfter(unittest.TestCase):
    """429 响应必须透出 Retry-After 头供调用方退避(2026-07-21)。"""

    def test_retry_after_header_parsed(self) -> None:
        fake = FakeSession([FakeResponse(429, raw=b"slow down", headers={"Retry-After": "7"})])
        provider = _http_provider(fake)
        resp = provider.request("daily", {"trade_date": "20250102"})
        self.assertEqual(resp.status, "transient_error")
        self.assertEqual(resp.retry_after_seconds, 7.0)

    def test_retry_after_absent_defaults_none(self) -> None:
        fake = FakeSession([FakeResponse(429, raw=b"slow down")])
        provider = _http_provider(fake)
        resp = provider.request("daily", {"trade_date": "20250102"})
        self.assertIsNone(resp.retry_after_seconds)

    def test_retry_after_non_numeric_ignored(self) -> None:
        fake = FakeSession([FakeResponse(429, raw=b"slow down", headers={"Retry-After": "soon"})])
        provider = _http_provider(fake)
        resp = provider.request("daily", {"trade_date": "20250102"})
        self.assertIsNone(resp.retry_after_seconds)


class _DripResponse:
    """持续慢滴字节的服务端响应:每个分块间隔都小于 read timeout,
    但累计耗时超过总期限——复现 2026-07-21 B3 假死场景。"""

    def __init__(self, payload: bytes, chunk_seconds: float, chunk_size: int = 4096) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.is_redirect = False
        self._payload = payload
        self._chunk_seconds = chunk_seconds
        self._chunk_size = chunk_size
        self.closed = False

    def iter_content(self, chunk_size: int = 65536):
        for i in range(0, len(self._payload), self._chunk_size):
            time.sleep(self._chunk_seconds)
            yield self._payload[i : i + self._chunk_size]

    def close(self) -> None:
        self.closed = True


class TestTotalResponseDeadline(unittest.TestCase):
    """180s 硬性总响应期限回归(2026-07-21 用户指令)。

    服务端持续滴字节但不触发 read timeout 时,累计到期必须主动中断,
    标记 transient_error(pipeline 进而标 retryable),不完整字节不发布。
    """

    def test_slow_drip_aborted_by_total_deadline(self) -> None:
        payload = json.dumps({"code": 0, "msg": "", "data": {"fields": ["a"], "items": [[1]] * 100}}).encode()
        # 每 0.05s 滴 4KB: 单次间隔远小于 read timeout,20+ 块累计 > 0.3s 期限。
        drip = _DripResponse(payload * 50, chunk_seconds=0.05, chunk_size=4096)
        fake = FakeSession([drip])
        with patch.dict(os.environ, dict(_HTTP_ENV), clear=True):
            provider = TushareCompatibleHttpProvider(session=fake, total_deadline_seconds=0.3)
        start = time.monotonic()
        resp = provider.request("balancesheet_vip", {"ts_code": "688168.SH"})
        elapsed = time.monotonic() - start
        self.assertEqual(resp.status, "transient_error")
        self.assertIn("总响应期限超时", resp.message)
        self.assertEqual(resp.raw_payload, b"")  # 不完整字节不进入响应
        self.assertLess(elapsed, 5.0)  # 远早于完整滴完(~100s+)
        self.assertTrue(drip.closed)  # 连接被主动关闭

    def test_normal_response_within_deadline_unaffected(self) -> None:
        fake = FakeSession([FakeResponse(200, payload=_ok_payload())])
        with patch.dict(os.environ, dict(_HTTP_ENV), clear=True):
            provider = TushareCompatibleHttpProvider(session=fake, total_deadline_seconds=30.0)
        resp = provider.request("daily", {"trade_date": "20250102"})
        self.assertTrue(resp.is_success)

    def test_drip_task_marked_retryable_and_nothing_published(self) -> None:
        """管线级: 滴字节任务经总期限中断后标 retryable_error,Raw/Bronze 零落盘。"""
        payload = json.dumps({"code": 0, "msg": "", "data": {"fields": ["a"], "items": [[1]] * 100}}).encode()
        drip = _DripResponse(payload * 50, chunk_seconds=0.05, chunk_size=4096)
        fake = FakeSession([drip] * 3)  # 重试 3 次都是滴字节
        with patch.dict(os.environ, dict(_HTTP_ENV), clear=True):
            provider = TushareCompatibleHttpProvider(session=fake, total_deadline_seconds=0.3)
        with tempfile.TemporaryDirectory() as tmp:
            config = TestB2RepairRegressions()._make_config(tmp)
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            pipeline = ArchivePipeline(config, provider, db, snapshot_id="snap_drip")
            spec = EndpointSpec(
                api_name="balancesheet_vip",
                dataset="financial_pit",
                priority="P0",
                primary_key=["ts_code", "end_date"],
                primary_split="period",
                fallback_split="ts_code",
                all_fields=True,
                fields="",
                params_template={"ts_code": "688168.SH"},
            )
            result = pipeline.run_tasks([spec])
            self.assertEqual(result.tasks_failed, 1)
            from ashare_quant.archive.pipeline import task_id

            task = db.get(task_id(config.provider.name, "balancesheet_vip", dict(spec.params_template), "", "snap_drip"))
            self.assertEqual(task.status, TaskStatus.RETRYABLE_ERROR)
            self.assertIn("总响应期限超时", task.last_error)
            # 原子性: 不完整响应不产生任何 Raw/Bronze 文件。
            self.assertEqual(list(Path(tmp).rglob("*.json.zst")), [])
            self.assertEqual(list(Path(tmp).rglob("*.parquet")), [])
            self.assertEqual(list(Path(tmp).rglob("*.tmp")), [])


class TestResumeSpecsLoading(unittest.TestCase):
    """断点续跑 fail-closed 回归(2026-07-22 用户指令, context 漂移事件)。

    创建 snapshot 后交易日推进一天,恢复时 task ID、任务数、参数和
    manifest SHA 必须完全不变;context 缺失/SHA 不符/参数被篡改时
    必须 fail-closed,禁止按最新交易日重建。
    """

    def _setup(self, tmp: str, *, with_context_sha: bool = True):
        from ashare_quant.archive.registry import default_inventory
        from ashare_quant.archive.state import DownloadTask, TaskStateDB, TaskStatus

        config = TestPipelineHelpers._make_config(self, tmp)
        db = TaskStateDB(config.catalog_dir / "tasks.db")
        inventory = default_inventory()
        ep = inventory.endpoints["balancesheet_vip"]
        snapshot = "snapA"
        codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
        rows = []
        for code in codes:
            params = {"ts_code": code, "start_date": "19950101", "end_date": "20260720"}
            tid = task_id("mock", "balancesheet_vip", params, "", snapshot)
            db.upsert(DownloadTask(
                task_id=tid, api_name="balancesheet_vip", params=params, fields="",
                dataset=ep.dataset, priority=ep.priority, primary_key=list(ep.primary_key),
                primary_split=ep.primary_split, fallback_split=ep.fallback_split,
                status=TaskStatus.SUCCESS, row_count=100,
            ))
            rows.append((tid, params))
        reports_dir = config.reports_dir / "batches" / "B9_test"
        reports_dir.mkdir(parents=True, exist_ok=True)
        context_sha = "a" * 64
        manifest = {
            "schema_version": 1, "snapshot_id": snapshot, "provider": "mock",
            "total_tasks": len(rows), "by_status": {"success": len(rows)},
            "tasks": [{"task_id": tid, "api_name": "balancesheet_vip"} for tid, _ in rows],
        }
        if with_context_sha:
            manifest["context_sha256"] = context_sha
        manifest_path = reports_dir / "batch_manifest.jsonl"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        (reports_dir / f"context_{context_sha[:12]}.json").write_text(
            json.dumps({"context_sha256": context_sha, "latest_trade_date": "20260720"}),
            encoding="utf-8")
        return config, db, inventory, snapshot, rows, manifest_path, reports_dir

    def test_resume_invariant_to_trade_date_advance(self) -> None:
        """交易日推进一天(存在漂移 context)时,回放结果必须完全不变。"""
        from ashare_quant.archive.batch import load_resume_specs

        with tempfile.TemporaryDirectory() as tmp:
            config, db, inventory, snapshot, rows, manifest_path, reports_dir = self._setup(tmp)
            # 模拟"交易日推进一天": 目录里多出一个新日期的漂移 context。
            drifted_sha = "b" * 64
            (reports_dir / f"context_{drifted_sha[:12]}.json").write_text(
                json.dumps({"context_sha256": drifted_sha, "latest_trade_date": "20260721"}),
                encoding="utf-8")
            sha_before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            specs = load_resume_specs(config, db, inventory, "B9_test", snapshot)
            self.assertEqual(len(specs), 3)  # 任务数不变
            for (tid, params), spec in zip(rows, specs, strict=True):
                self.assertEqual(spec.params_template, params)  # 参数不变(end_date=20260720)
                self.assertEqual(spec.params_template["end_date"], "20260720")
                self.assertEqual(task_id("mock", spec.api_name, spec.params_template,
                                         spec.fields, snapshot), tid)  # task ID 不变
            sha_after = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            self.assertEqual(sha_before, sha_after)  # manifest SHA 不变

    def test_resume_fail_closed_on_missing_manifest(self) -> None:
        from ashare_quant.archive.batch import load_resume_specs

        with tempfile.TemporaryDirectory() as tmp:
            config, db, inventory, snapshot, rows, manifest_path, _ = self._setup(tmp)
            manifest_path.unlink()
            with self.assertRaises(FileNotFoundError):
                load_resume_specs(config, db, inventory, "B9_test", snapshot)

    def test_resume_fail_closed_on_context_sha_mismatch(self) -> None:
        from ashare_quant.archive.batch import load_resume_specs

        with tempfile.TemporaryDirectory() as tmp:
            config, db, inventory, snapshot, rows, _, reports_dir = self._setup(tmp)
            # 封存 context 文件内容与 manifest 记录的 SHA 不符。
            (reports_dir / f"context_{'a' * 12}.json").write_text(
                json.dumps({"context_sha256": "c" * 64}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_resume_specs(config, db, inventory, "B9_test", snapshot)

    def test_resume_fail_closed_on_ambiguous_legacy_context(self) -> None:
        """老 manifest 无 context_sha256 且存在多个 context 文件时 fail-closed。"""
        from ashare_quant.archive.batch import load_resume_specs

        with tempfile.TemporaryDirectory() as tmp:
            config, db, inventory, snapshot, rows, _, reports_dir = self._setup(
                tmp, with_context_sha=False)
            drifted_sha = "b" * 64
            (reports_dir / f"context_{drifted_sha[:12]}.json").write_text(
                json.dumps({"context_sha256": drifted_sha}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_resume_specs(config, db, inventory, "B9_test", snapshot)

    def test_resume_fail_closed_on_tampered_params(self) -> None:
        """db 行参数与 task_id 不一致(疑似篡改)时 fail-closed。"""
        import sqlite3

        from ashare_quant.archive.batch import load_resume_specs

        with tempfile.TemporaryDirectory() as tmp:
            config, db, inventory, snapshot, rows, _, _ = self._setup(tmp)
            tid, params = rows[0]
            # upsert 设计上拒绝改 params(身份不可变),用原生 SQL 模拟篡改。
            tampered = dict(params, end_date="20260721")
            with sqlite3.connect(db.path) as conn:
                conn.execute(
                    "UPDATE archive_tasks SET params_json=? WHERE task_id=?",
                    (json.dumps(tampered, sort_keys=True, ensure_ascii=False), tid),
                )
            with self.assertRaises(ValueError):
                load_resume_specs(config, db, inventory, "B9_test", snapshot)

    def test_orphaned_context_drift_status_exists(self) -> None:
        self.assertEqual(TaskStatus.ORPHANED_CONTEXT_DRIFT.value, "orphaned_context_drift")
        self.assertEqual(TaskStatus.SUPERSEDED_INVALID_PARTITION.value,
                         "superseded_invalid_partition")

    def test_superseded_truncated_cap_status_exists(self) -> None:
        """2026-07-23 B4 repair: 恰满真实 cap 被截断的任务由窗口二分任务集替代。"""
        self.assertEqual(TaskStatus.SUPERSEDED_TRUNCATED_CAP.value,
                         "superseded_truncated_cap")

    def test_superseded_legacy_collision_status_exists(self) -> None:
        """2026-07-23 B2 清理: 撞名事件僵尸 running 由新一代覆盖区间任务承载。"""
        self.assertEqual(TaskStatus.SUPERSEDED_LEGACY_COLLISION.value,
                         "superseded_legacy_collision")


class TestFrozenSpecs(unittest.TestCase):
    """冻结任务清单回归(2026-07-22 用户指令): 物化封存后,无论交易日如何
    推进,启动/恢复回放的 task ID、任务数、参数和 manifest SHA 必须不变。"""

    def _frozen(self, tmp: str):
        from ashare_quant.archive.batch import save_frozen_specs
        from ashare_quant.archive.registry import default_inventory

        config = TestPipelineHelpers._make_config(self, tmp)
        inventory = default_inventory()
        ep = inventory.endpoints["pledge_stat"]
        specs = []
        for code in ("000001.SZ", "000002.SZ"):
            spec = ep.to_spec()
            spec.params_template = {"ts_code": code}
            specs.append(spec)
        ctx_record = {"context_sha256": "d" * 64, "latest_trade_date": "20260721"}
        path = Path(tmp) / "frozen_specs_snapF.json"
        payload = save_frozen_specs(
            path, batch_id="B9_test", snapshot_id="snapF",
            provider_name="mock", context_record=ctx_record, specs=specs,
        )
        return config, inventory, path, payload

    def test_frozen_roundtrip_invariant(self) -> None:
        from ashare_quant.archive.batch import load_frozen_specs

        with tempfile.TemporaryDirectory() as tmp:
            config, inventory, path, payload = self._frozen(tmp)
            sha_before = hashlib.sha256(path.read_bytes()).hexdigest()
            snapshot, specs = load_frozen_specs(path, config, inventory)
            self.assertEqual(snapshot, "snapF")
            self.assertEqual(len(specs), 2)  # 任务数不变
            self.assertEqual([s.params_template["ts_code"] for s in specs],
                             ["000001.SZ", "000002.SZ"])  # 参数不变
            ids = sorted(task_id("mock", s.api_name, s.params_template, s.fields, "snapF")
                         for s in specs)
            manifest_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
            self.assertEqual(manifest_sha, payload["manifest_sha256"])  # task ID/manifest SHA 不变
            self.assertEqual(sha_before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_frozen_fail_closed_on_tamper(self) -> None:
        from ashare_quant.archive.batch import load_frozen_specs

        with tempfile.TemporaryDirectory() as tmp:
            config, inventory, path, payload = self._frozen(tmp)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["specs"][0]["params"]["ts_code"] = "600000.SH"  # 篡改参数
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_frozen_specs(path, config, inventory)

    def test_resume_falls_back_to_frozen_when_manifest_missing(self) -> None:
        from ashare_quant.archive.batch import load_resume_specs
        from ashare_quant.archive.state import TaskStateDB

        with tempfile.TemporaryDirectory() as tmp:
            config, inventory, frozen_path, _ = self._frozen(tmp)
            reports_dir = config.reports_dir / "batches" / "B9_test"
            reports_dir.mkdir(parents=True, exist_ok=True)
            target = reports_dir / "frozen_specs_snapF.json"
            target.write_text(frozen_path.read_text(encoding="utf-8"), encoding="utf-8")
            db = TaskStateDB(config.catalog_dir / "tasks.db")  # 空库: 批次尚未运行
            specs = load_resume_specs(config, db, inventory, "B9_test", "snapF")
            self.assertEqual(len(specs), 2)

    def test_resume_fail_closed_when_neither_manifest_nor_frozen(self) -> None:
        from ashare_quant.archive.batch import load_resume_specs
        from ashare_quant.archive.state import TaskStateDB

        with tempfile.TemporaryDirectory() as tmp:
            config = TestPipelineHelpers._make_config(self, tmp)
            from ashare_quant.archive.registry import default_inventory
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            with self.assertRaises(FileNotFoundError):
                load_resume_specs(config, db, default_inventory(), "B9_test", "snapX")


class TestB4Regressions(unittest.TestCase):
    """B4_events 8 端点注册回归(2026-07-22 预检实测)。

    实测要点(preflight/b4_canary.json + b4_followup.json):
    - repurchase row_cap=2000(宽区间/年度/202402 月恰满;月拆+自动二分);
    - share_float 多码 chunk 静默空、top10_holders 多码 chunk 静默截断
      (000001.SZ 单查 298 行 vs chunk 内 58 行),split_unit 必须 symbol;
    - 其余端点单码全历史最大 141~1,108 行,上限不可达,登记保守 7000。
    """

    EXPECTED_PKS = {
        "repurchase": ["ts_code", "ann_date", "end_date"],
        "share_float": ["ts_code", "float_date"],
        "pledge_stat": ["ts_code", "end_date"],
        "pledge_detail": ["ts_code", "start_date", "pledgor"],
        "stk_holdernumber": ["ts_code", "end_date", "ann_date"],
        "stk_holdertrade": ["ts_code", "ann_date", "holder_name", "in_de"],
        "top10_holders": ["ts_code", "end_date", "holder_name"],
        "top10_floatholders": ["ts_code", "end_date", "holder_name"],
    }
    EXPECTED_CAPS = {
        "repurchase": 2000,
        # 6000 实测(2026-07-23 B4): 持有人粒度,528 只全历史恰好 6000 行截断
        "share_float": 6000,
        "pledge_stat": 7000,
        "pledge_detail": 7000,
        "stk_holdernumber": 7000,
        "stk_holdertrade": 7000,
        "top10_holders": 7000,
        "top10_floatholders": 7000,
    }
    EXPECTED_SPLIT_UNITS = {
        "repurchase": "month",
        "share_float": "symbol",
        "pledge_stat": "symbol",
        "pledge_detail": "symbol",
        "stk_holdernumber": "symbol",
        "stk_holdertrade": "symbol",
        "top10_holders": "symbol",
        "top10_floatholders": "symbol",
    }

    def test_b4_primary_keys(self) -> None:
        from ashare_quant.archive.registry import default_inventory

        endpoints = default_inventory().endpoints
        for api, pk in self.EXPECTED_PKS.items():
            self.assertEqual(endpoints[api].primary_key, pk, api)

    def test_b4_row_caps_registered(self) -> None:
        from ashare_quant.archive.registry import default_inventory

        endpoints = default_inventory().endpoints
        for api, cap in self.EXPECTED_CAPS.items():
            self.assertEqual(endpoints[api].row_cap, cap, api)

    def test_b4_split_units_symbol_or_month(self) -> None:
        """share_float/top10_holders 严禁回退到 symbol_chunk(静默空/截断)。"""
        from ashare_quant.archive.registry import default_inventory

        endpoints = default_inventory().endpoints
        for api, unit in self.EXPECTED_SPLIT_UNITS.items():
            self.assertEqual(endpoints[api].split_unit, unit, api)

    def test_b4_fields_empty(self) -> None:
        """B4 全部端点 fields=""(top10_floatholders 显式 fields 返回空,实测)。"""
        from ashare_quant.archive.registry import default_inventory

        endpoints = default_inventory().endpoints
        for api in self.EXPECTED_PKS:
            self.assertEqual(endpoints[api].fields, "", api)


class _DriftFallbackProvider(MockArchiveProvider):
    """fields="" 返回缺字段畸形响应;显式 fields 行为可编排(成功/仍缺字段/未知 schema)。"""

    SUSPEND_FULL = ["ts_code", "trade_date", "suspend_timing", "suspend_type"]
    SUSPEND_DRIFT = ["ts_code", "suspend_type", "suspend_timing"]

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode  # "success" | "still_drift" | "unknown_schema"
        self.calls: list[tuple[str, str]] = []

    def request(self, api_name, params=None, *, fields=""):
        self.calls.append((api_name, fields))
        if fields:
            cols = {
                "success": self.SUSPEND_FULL,
                "still_drift": self.SUSPEND_DRIFT,
                "unknown_schema": ["ts_code", "some_new_col"],
            }[self.mode]
        else:
            cols = self.SUSPEND_DRIFT
        items = [["000001.SZ", "20260708", "09:30", "S"]] if cols != ["ts_code", "some_new_col"] else [["x", 1]]
        key = f"{api_name}:{json.dumps(dict(params or {}), sort_keys=True, ensure_ascii=False)}"
        self._responses[key] = items
        self._columns[api_name] = cols
        return super().request(api_name, params, fields=fields)


class TestExplicitFieldsFallback(TestPipelineHelpers):
    """E 类缺字段显式 fields 回退(2026-07-23 用户批准,图片方案第 4 条)。"""

    REGISTERED_FP = "b10eeb0e0474db9a146be4ea05bdd994bf0224ffe30e7c1b0d0644217fb1ad3d"

    def _setup(self, tmp: str, mode: str):
        config = self._make_config(tmp)
        db = TaskStateDB(config.catalog_dir / "tasks.db")
        provider = _DriftFallbackProvider(mode)
        pipeline = ArchivePipeline(config, provider, db, snapshot_id="snap1")
        pipeline.schema_registry.save("suspend_d", self.REGISTERED_FP, {
            "endpoint": "suspend_d", "fingerprint": self.REGISTERED_FP,
            "columns": _DriftFallbackProvider.SUSPEND_FULL,
            "snapshot_id": "snap1", "row_count": 0,
        })
        spec = EndpointSpec(
            api_name="suspend_d", dataset="trading_status", priority="P0",
            primary_key=["ts_code", "trade_date"], primary_split="trade_date",
            fallback_split=None, all_fields=True, fields="",
            params_template={"trade_date": "20260708"},
        )
        return config, db, provider, pipeline, spec

    def test_fallback_success(self) -> None:
        """缺字段畸形 → 显式 fields 回退成功 → success,审计完整。"""
        with tempfile.TemporaryDirectory() as tmp:
            config, db, provider, pipeline, spec = self._setup(tmp, "success")
            result = pipeline.run_tasks([spec])
            self.assertEqual(result.tasks_completed, 1)
            tid = task_id("mock", "suspend_d", {"trade_date": "20260708"}, "", "snap1")
            task = db.get(tid)
            self.assertEqual(task.status, TaskStatus.SUCCESS)
            audit = task.metadata["explicit_fields_fallback"]
            self.assertTrue(audit["accepted"])
            self.assertEqual(audit["explicit_fields"],
                             "ts_code,trade_date,suspend_timing,suspend_type")
            self.assertEqual(len(audit["drift_response_sha256"]), 64)
            self.assertEqual(len(audit["retry_response_sha256"]), 64)
            # 审计 JSONL + 畸形原始响应存档
            audit_log = config.catalog_dir / "migrations" / "explicit_fields_fallback_audit.jsonl"
            self.assertTrue(audit_log.exists())
            self.assertTrue((config.catalog_dir / "quarantine_evidence" / f"{tid}_drift.json").exists())

    def test_fallback_still_drift_stays_quarantined(self) -> None:
        """回退响应仍缺字段 → 不接受,保持 quarantined。"""
        with tempfile.TemporaryDirectory() as tmp:
            _, db, _, pipeline, spec = self._setup(tmp, "still_drift")
            result = pipeline.run_tasks([spec])
            self.assertEqual(result.tasks_completed, 0)
            tid = task_id("mock", "suspend_d", {"trade_date": "20260708"}, "", "snap1")
            task = db.get(tid)
            self.assertEqual(task.status, TaskStatus.QUARANTINED)
            self.assertFalse(task.metadata["explicit_fields_fallback"]["accepted"])

    def test_unknown_schema_no_fallback(self) -> None:
        """回退响应出现未知/新增列 → 禁止回退,保持 quarantined。"""
        with tempfile.TemporaryDirectory() as tmp:
            _, db, _, pipeline, spec = self._setup(tmp, "unknown_schema")
            pipeline.run_tasks([spec])
            tid = task_id("mock", "suspend_d", {"trade_date": "20260708"}, "", "snap1")
            self.assertEqual(db.get(tid).status, TaskStatus.QUARANTINED)

    def test_at_most_one_retry(self) -> None:
        """每个任务最多一次显式 fields 重试。"""
        with tempfile.TemporaryDirectory() as tmp:
            _, _, provider, pipeline, spec = self._setup(tmp, "still_drift")
            pipeline.run_tasks([spec])
            explicit_calls = [c for c in provider.calls if c[1]]
            self.assertEqual(len(explicit_calls), 1)

    def test_task_identity_stable(self) -> None:
        """回退前后 task_id 不变(fields 不入身份)。"""
        with tempfile.TemporaryDirectory() as tmp:
            _, db, _, pipeline, spec = self._setup(tmp, "success")
            tid_before = task_id("mock", "suspend_d", dict(spec.params_template), spec.fields, "snap1")
            pipeline.run_tasks([spec])
            task = db.get(tid_before)
            self.assertIsNotNone(task)
            self.assertEqual(task.task_id, tid_before)
            self.assertEqual(task.fields, "")

    def test_original_response_not_overwritten(self) -> None:
        """畸形原始响应存档于 quarantine_evidence 且内容与回退响应不同。"""
        with tempfile.TemporaryDirectory() as tmp:
            config, db, _, pipeline, spec = self._setup(tmp, "success")
            pipeline.run_tasks([spec])
            tid = task_id("mock", "suspend_d", {"trade_date": "20260708"}, "", "snap1")
            drift = (config.catalog_dir / "quarantine_evidence" / f"{tid}_drift.json").read_bytes()
            task = db.get(tid)
            audit = task.metadata["explicit_fields_fallback"]
            self.assertEqual(hashlib.sha256(drift).hexdigest(), audit["drift_response_sha256"])
            self.assertNotEqual(audit["drift_response_sha256"], audit["retry_response_sha256"])
            # Raw 落盘文件(压缩)解压后的载荷 SHA = 回退响应 SHA
            import zstandard
            payload = zstandard.ZstdDecompressor().stream_reader(
                open(task.raw_path, "rb")).read()
            self.assertEqual(hashlib.sha256(payload).hexdigest(),
                             audit["retry_response_sha256"])

    def test_non_whitelisted_endpoint_no_fallback(self) -> None:
        """非白名单端点即使缺字段也不回退(无显式 fields 调用)。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(tmp)
            db = TaskStateDB(config.catalog_dir / "tasks.db")
            provider = _DriftFallbackProvider("success")
            pipeline = ArchivePipeline(config, provider, db, snapshot_id="snap1")
            full = ["ts_code", "trade_date", "close"]
            fp = schema_fingerprint(full)
            pipeline.schema_registry.save("daily", fp, {
                "endpoint": "daily", "fingerprint": fp, "columns": full,
                "snapshot_id": "snap1", "row_count": 0})
            spec = EndpointSpec(
                api_name="daily", dataset="market_daily", priority="P0",
                primary_key=["ts_code", "trade_date"], primary_split="trade_date",
                fallback_split=None, all_fields=True, fields="",
                params_template={"trade_date": "20260708"})
            pipeline.run_tasks([spec])
            tid = task_id("mock", "daily", {"trade_date": "20260708"}, "", "snap1")
            self.assertEqual(db.get(tid).status, TaskStatus.QUARANTINED)
            self.assertEqual([c for c in provider.calls if c[1]], [])


if __name__ == "__main__":
    unittest.main()