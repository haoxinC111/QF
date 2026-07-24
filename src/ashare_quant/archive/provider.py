"""Archive provider protocol and HTTP-compatible adapter."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

REDACTED = "<redacted>"


@dataclass(frozen=True)
class ArchiveResponse:
    """Normalized response from any archive provider."""

    api_name: str
    params: dict[str, Any]
    fields: str
    columns: list[str]
    items: list[list[Any]]
    status: str  # success, empty, denied, not_found, invalid_params, transient_error
    message: str
    fetched_at_utc: str
    elapsed_seconds: float
    raw_payload: bytes
    raw_payload_sha256: str
    http_status: int | None = None
    source_provider: str = ""
    source_endpoint: str = ""
    source_request_id: str = ""
    # HTTP 429 时网关 Retry-After 头(秒),供调用方退避;无该头为 None。
    retry_after_seconds: float | None = None

    @property
    def row_count(self) -> int:
        return len(self.items)

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    @property
    def is_empty(self) -> bool:
        return self.status == "empty"


class ArchiveProvider(ABC):
    """Replaceable data-source adapter for the archive pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @abstractmethod
    def request(
        self,
        api_name: str,
        params: dict[str, Any] | None = None,
        *,
        fields: str = "",
    ) -> ArchiveResponse:
        """Execute one endpoint request and return a normalized response."""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return provider health / permission metadata without sending Token."""


def _redact_payload(text: str | bytes) -> str:
    """Replace token-like values with <redacted> for safe logging."""
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError:
            return "<binary>"
    # Redact standard Tushare token field.
    text = re.sub(r'("token"\s*:\s*")[^"]*"', r'\1' + REDACTED + '"', text)
    # Redact X-API-Key style header values in logged strings.
    text = re.sub(r'("X-API-Key"\s*:\s*")[^"]*"', r'\1' + REDACTED + '"', text)
    return text


def _request_id(api_name: str, params: dict[str, Any], fields: str) -> str:
    payload = json.dumps(
        {"api_name": api_name, "params": params, "fields": fields},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class TotalDeadlineExceeded(Exception):
    """单请求累计耗时超过硬性总响应期限(服务端挂起/慢滴字节)。"""


class TushareCompatibleHttpProvider(ArchiveProvider):
    """Direct HTTP adapter for Tushare-compatible gateways.

    Reads the gateway URL from ``url_env`` and the access Token from
    ``token_env``.  The official ``TUSHARE_TOKEN`` is explicitly forbidden
    from being sent to third-party hosts.
    """

    def __init__(
        self,
        *,
        url_env: str = "QF_ARCHIVE_API_URL",
        token_env: str = "QF_ARCHIVE_API_TOKEN",
        forbid_token_env: list[str] | None = None,
        source_provider: str = "tushare_compatible_proxy",
        allowed_hosts: list[str] | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
        total_deadline_seconds: float = 180.0,
        accept_encoding: str = "gzip",
        follow_cross_host_redirects: bool = False,
        api_key_env: str | None = None,
        api_key_header: str = "X-API-Key",
        session: Any | None = None,
    ) -> None:
        self.url_env = url_env
        self.token_env = token_env
        self.forbid_token_env = set(forbid_token_env or ["TUSHARE_TOKEN"])
        self.source_provider = source_provider
        self.allowed_hosts = set(allowed_hosts or [])
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        # 单请求硬性总响应期限(2026-07-21 用户指令): 服务端持续滴字节会让
        # read timeout 永不触发;此处按请求起点累计计时,到期主动中断。
        # 仅覆盖连接+响应接收;调用方的 Retry-After/退避等待不计入。
        self.total_deadline_seconds = total_deadline_seconds
        self.accept_encoding = accept_encoding
        self.follow_cross_host_redirects = follow_cross_host_redirects
        self.api_key_env = api_key_env
        self.api_key_header = api_key_header

        self._base_url = self._load_base_url()
        self._token = self._load_token()
        self._session = session if session is not None else requests.Session()
        self._session.headers.update(
            {
                "Accept-Encoding": accept_encoding,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        self._load_api_key()

    @property
    def name(self) -> str:
        return self.source_provider

    def _load_base_url(self) -> str:
        url = os.environ.get(self.url_env)
        if not url:
            raise ValueError(
                f"环境变量 {self.url_env} 未设置，无法构造归档 HTTP 适配器"
            )
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise ValueError(f"归档网关必须使用 HTTPS: {self.url_env}")
        if self.allowed_hosts and parsed.netloc not in self.allowed_hosts:
            raise ValueError(
                f"主机 {parsed.netloc} 不在允许列表中；"
                f"请在配置 allowed_hosts 中显式登记"
            )
        return url.rstrip("/")

    def _load_token(self) -> str:
        # Safety: make sure the official Tushare token is never reused here.
        for env_name in self.forbid_token_env:
            if env_name == self.token_env:
                continue
            if os.environ.get(env_name) and self.url_env != env_name:
                logger.warning(
                    "检测到 %s 已设置，但本适配器只使用 %s，"
                    "不会把官方 Token 发往第三方网关",
                    env_name,
                    self.token_env,
                )
        token = os.environ.get(self.token_env)
        if not token:
            raise ValueError(
                f"环境变量 {self.token_env} 未设置，无法访问兼容网关"
            )
        return token.strip()

    def _load_api_key(self) -> None:
        """Optionally add an API key header required by some gateways."""
        if not self.api_key_env:
            return
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            self._session.headers[self.api_key_header] = api_key.strip()
            logger.debug("Added %s header from %s", self.api_key_header, self.api_key_env)

    def _allowed_redirect(self, response: requests.Response) -> bool:
        if self.follow_cross_host_redirects:
            return True
        if not response.is_redirect:
            return True
        location = response.headers.get("Location", "")
        parsed = urlparse(location)
        return parsed.netloc in self.allowed_hosts or parsed.netloc == ""

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.source_provider,
            "url_env": self.url_env,
            "token_env": self.token_env,
            "base_url": self._base_url,
            "token_present": bool(self._token),
            "token_length": len(self._token),
            "api_key_header": self.api_key_header if self.api_key_env else None,
            "api_key_env": self.api_key_env,
            "api_key_present": bool(self._session.headers.get(self.api_key_header))
            if self.api_key_env
            else False,
            "allowed_hosts": sorted(self.allowed_hosts),
            "connect_timeout": self.connect_timeout,
            "read_timeout": self.read_timeout,
        }

    def request(
        self,
        api_name: str,
        params: dict[str, Any] | None = None,
        *,
        fields: str = "",
    ) -> ArchiveResponse:
        params = dict(params or {})
        payload = {
            "api_name": api_name,
            "token": self._token,
            "params": params,
            "fields": fields,
        }
        payload_json = json.dumps(payload, ensure_ascii=False)
        request_id = _request_id(api_name, params, fields)
        start = time.perf_counter()
        deadline_ts = time.monotonic() + self.total_deadline_seconds
        fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        logger.debug(
            "Provider request %s %s params=%s fields=%s",
            self.source_provider,
            api_name,
            _redact_payload(json.dumps(params, ensure_ascii=False)),
            fields,
        )

        try:
            response = self._session.post(
                self._base_url,
                data=payload_json.encode("utf-8"),
                timeout=(self.connect_timeout, self.read_timeout),
                allow_redirects=False,
                stream=True,
            )
            elapsed = time.perf_counter() - start
        except requests.Timeout as exc:
            elapsed = time.perf_counter() - start
            return self._error_response(
                api_name,
                params,
                fields,
                "transient_error",
                f"请求超时: {exc}",
                fetched_at,
                elapsed,
                b"",
                http_status=None,
                request_id=request_id,
            )
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - start
            return self._error_response(
                api_name,
                params,
                fields,
                "transient_error",
                f"网络错误: {exc}",
                fetched_at,
                elapsed,
                b"",
                http_status=None,
                request_id=request_id,
            )

        if not self._allowed_redirect(response):
            elapsed = time.perf_counter() - start
            return self._error_response(
                api_name,
                params,
                fields,
                "transient_error",
                f"禁止跨域跳转至 {response.headers.get('Location')}",
                fetched_at,
                elapsed,
                response.content,
                http_status=response.status_code,
                request_id=request_id,
            )

        if response.is_redirect:
            # Follow the redirect ourselves so we can enforce allowed_hosts.
            location = response.headers["Location"]
            try:
                response = self._session.get(
                    location,
                    timeout=(self.connect_timeout, self.read_timeout),
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException as exc:
                elapsed = time.perf_counter() - start
                return self._error_response(
                    api_name,
                    params,
                    fields,
                    "transient_error",
                    f"跳转请求失败: {exc}",
                    fetched_at,
                    elapsed,
                    b"",
                    http_status=None,
                    request_id=request_id,
                )

        try:
            raw = self._read_body_with_deadline(response, deadline_ts)
        except TotalDeadlineExceeded:
            elapsed = time.perf_counter() - start
            return self._error_response(
                api_name,
                params,
                fields,
                "transient_error",
                f"总响应期限超时({self.total_deadline_seconds:g}s): 服务端挂起或慢滴字节,已主动中断",
                fetched_at,
                elapsed,
                b"",
                http_status=response.status_code,
                request_id=request_id,
            )
        elapsed = time.perf_counter() - start
        raw_sha256 = hashlib.sha256(raw).hexdigest()

        if response.status_code == 401 or response.status_code == 403:
            return self._error_response(
                api_name,
                params,
                fields,
                "denied",
                f"权限不足: HTTP {response.status_code}",
                fetched_at,
                elapsed,
                raw,
                http_status=response.status_code,
                request_id=request_id,
            )

        if response.status_code == 429:
            retry_after: float | None = None
            raw_ra = response.headers.get("Retry-After")
            if raw_ra:
                try:
                    retry_after = max(0.0, float(raw_ra))
                except ValueError:
                    retry_after = None  # HTTP-date 形式暂不解析,退回指数退避
            return self._error_response(
                api_name,
                params,
                fields,
                "transient_error",
                "请求频率受限: HTTP 429",
                fetched_at,
                elapsed,
                raw,
                http_status=response.status_code,
                request_id=request_id,
                retry_after_seconds=retry_after,
            )

        if response.status_code >= 500:
            return self._error_response(
                api_name,
                params,
                fields,
                "transient_error",
                f"服务端错误: HTTP {response.status_code}",
                fetched_at,
                elapsed,
                raw,
                http_status=response.status_code,
                request_id=request_id,
            )

        if response.status_code != 200:
            return self._error_response(
                api_name,
                params,
                fields,
                "invalid_params",
                f"HTTP {response.status_code}",
                fetched_at,
                elapsed,
                raw,
                http_status=response.status_code,
                request_id=request_id,
            )

        try:
            # 用已按总期限完整接收的 raw 解析(stream=True 下 response.json()
            # 会重复消费流);不完整响应不会到达这里(期限中断在上面已返回)。
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return self._error_response(
                api_name,
                params,
                fields,
                "transient_error",
                f"响应不是合法 JSON: {exc}",
                fetched_at,
                elapsed,
                raw,
                http_status=response.status_code,
                request_id=request_id,
            )

        code = data.get("code")
        msg = data.get("msg", "")
        # Tushare-compatible errors are usually signaled by a non-zero code.
        if code not in (None, 0, "0"):
            msg_text = str(msg)
            msg_lower = msg_text.lower()
            if "权限" in msg_text or "permission" in msg_lower or "denied" in msg_lower:
                status = "denied"
            elif "未知" in msg_text or "not found" in msg_lower or "不存在" in msg_text:
                status = "not_found"
            elif "不支持" in msg_text or "incompatible" in msg_lower:
                status = "incompatible"
            else:
                status = "invalid_params"
            return self._error_response(
                api_name,
                params,
                fields,
                status,
                f"接口返回错误: {msg} (code={code})",
                fetched_at,
                elapsed,
                raw,
                http_status=response.status_code,
                request_id=request_id,
            )

        # Some compatible gateways wrap the Tushare payload in a "data" object.
        payload_data = data.get("data") if isinstance(data.get("data"), dict) else data
        columns = list(payload_data.get("fields", []))
        items = payload_data.get("items", [])
        if not isinstance(items, list):
            return self._error_response(
                api_name,
                params,
                fields,
                "transient_error",
                "响应中 items 不是列表",
                fetched_at,
                elapsed,
                raw,
                http_status=response.status_code,
                request_id=request_id,
            )

        status = "success" if items else "empty"
        message = f"OK {len(items)} rows" if items else "空结果"

        return ArchiveResponse(
            api_name=api_name,
            params=params,
            fields=fields,
            columns=columns,
            items=items,
            status=status,
            message=message,
            fetched_at_utc=fetched_at,
            elapsed_seconds=elapsed,
            raw_payload=raw,
            raw_payload_sha256=raw_sha256,
            http_status=response.status_code,
            source_provider=self.source_provider,
            source_endpoint=self._base_url,
            source_request_id=request_id,
        )

    def _read_body_with_deadline(self, response: Any, deadline_ts: float) -> bytes:
        """分块读取响应体,累计超过总期限即关闭连接并抛 TotalDeadlineExceeded。

        read timeout 是"两次字节到达间隔"上限,服务端持续慢滴字节时永不
        触发;这里按请求起点的 monotonic 期限累计计时,每个分块到达后检查。
        临时/不完整字节只在内存中,绝不发布到 Raw/Bronze。
        """
        iter_content = getattr(response, "iter_content", None)
        if iter_content is None:
            # 测试替身等无流式接口的响应: 一次性读出后检查期限。
            body = response.content
            if time.monotonic() > deadline_ts:
                raise TotalDeadlineExceeded
            return body
        chunks: list[bytes] = []
        try:
            for chunk in iter_content(chunk_size=65536):
                chunks.append(chunk)
                if time.monotonic() > deadline_ts:
                    raise TotalDeadlineExceeded
        except TotalDeadlineExceeded:
            close = getattr(response, "close", None)
            if callable(close):
                close()
            raise
        if time.monotonic() > deadline_ts:
            close = getattr(response, "close", None)
            if callable(close):
                close()
            raise TotalDeadlineExceeded
        return b"".join(chunks)

    def _error_response(
        self,
        api_name: str,
        params: dict[str, Any],
        fields: str,
        status: str,
        message: str,
        fetched_at_utc: str,
        elapsed_seconds: float,
        raw_payload: bytes,
        http_status: int | None,
        request_id: str,
        retry_after_seconds: float | None = None,
    ) -> ArchiveResponse:
        logger.warning(
            "Provider error %s %s: %s (http=%s, request_id=%s)",
            self.source_provider,
            api_name,
            message,
            http_status,
            request_id,
        )
        return ArchiveResponse(
            api_name=api_name,
            params=params,
            fields=fields,
            columns=[],
            items=[],
            status=status,
            message=message,
            fetched_at_utc=fetched_at_utc,
            elapsed_seconds=elapsed_seconds,
            raw_payload=raw_payload,
            raw_payload_sha256=hashlib.sha256(raw_payload).hexdigest(),
            http_status=http_status,
            source_provider=self.source_provider,
            source_endpoint=self._base_url,
            source_request_id=request_id,
            retry_after_seconds=retry_after_seconds,
        )


class MockArchiveProvider(ArchiveProvider):
    """In-memory provider for unit tests and offline demonstrations."""

    def __init__(
        self,
        responses: dict[str, list[list[Any]]] | None = None,
        columns: dict[str, list[str]] | None = None,
        source_provider: str = "mock",
    ) -> None:
        self._responses = responses or {}
        self._columns = columns or {}
        self.source_provider = source_provider

    @property
    def name(self) -> str:
        return self.source_provider

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.source_provider,
            "registered_endpoints": sorted(self._responses.keys()),
        }

    def request(
        self,
        api_name: str,
        params: dict[str, Any] | None = None,
        *,
        fields: str = "",
    ) -> ArchiveResponse:
        params = dict(params or {})
        request_id = _request_id(api_name, params, fields)
        fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        key = f"{api_name}:{json.dumps(params, sort_keys=True, ensure_ascii=False)}"
        items = self._responses.get(key, [])
        cols = self._columns.get(api_name, [])
        raw = json.dumps(
            {"api_name": api_name, "fields": cols, "items": items},
            ensure_ascii=False,
        ).encode("utf-8")
        status = "success" if items else "empty"
        return ArchiveResponse(
            api_name=api_name,
            params=params,
            fields=fields,
            columns=list(cols),
            items=items,
            status=status,
            message="mock",
            fetched_at_utc=fetched_at,
            elapsed_seconds=0.0,
            raw_payload=raw,
            raw_payload_sha256=hashlib.sha256(raw).hexdigest(),
            http_status=200,
            source_provider=self.source_provider,
            source_endpoint="mock://localhost",
            source_request_id=request_id,
        )

    def register(
        self,
        api_name: str,
        params: dict[str, Any],
        columns: list[str],
        items: list[list[Any]],
    ) -> None:
        key = f"{api_name}:{json.dumps(params, sort_keys=True, ensure_ascii=False)}"
        self._responses[key] = items
        self._columns[api_name] = columns
