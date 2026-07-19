"""Tests for the data archive V2 module."""

from __future__ import annotations

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
from ashare_quant.archive.pipeline import EndpointSpec
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
            provider.register(
                "daily",
                {"start_date": "20250101", "end_date": "20250106"},
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


if __name__ == "__main__":
    unittest.main()
