"""Global rate limiting and retry helpers."""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TokenBucket:
    """Thread-safe token bucket with smooth refill."""

    def __init__(self, calls_per_minute: float) -> None:
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be positive")
        self._rate = calls_per_minute / 60.0
        self._tokens = self._rate
        self._max_tokens = max(1.0, self._rate * 2)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: float = 1.0, timeout: float | None = None) -> bool:
        deadline = time.monotonic() + (timeout or float("inf"))
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_update
                self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
                self._last_update = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                wait_needed = (tokens - self._tokens) / self._rate
            wait_needed = max(0.001, wait_needed * (1 + random.uniform(0, 0.05)))
            if time.monotonic() + wait_needed > deadline:
                return False
            time.sleep(wait_needed)

    def estimate_wait(self, tokens: float = 1.0) -> float:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self._max_tokens, self._tokens + (now - self._last_update) * self._rate)
            self._last_update = now
            if self._tokens >= tokens:
                return 0.0
            return (tokens - self._tokens) / self._rate


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0

    def sleep_for_attempt(self, attempt: int) -> float:
        delay = self.base_delay_seconds * (2 ** (attempt - 1))
        delay = min(delay, self.max_delay_seconds)
        jitter = random.uniform(0, delay * 0.2)
        return delay + jitter


class RateLimitedClient:
    """Wraps a provider call with token-bucket pacing and retry logic."""

    def __init__(
        self,
        bucket: TokenBucket,
        retry_policy: RetryPolicy,
    ) -> None:
        self.bucket = bucket
        self.retry_policy = retry_policy

    def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        last_exception: Exception | None = None
        last_result: Any = None
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            self.bucket.consume(tokens=1.0)
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                status = getattr(exc, "status", None)
                http_status = getattr(exc, "http_status", None)
                retryable = (
                    status in self.retry_policy.retry_statuses
                    or http_status in self.retry_policy.retry_statuses
                )
                if not retryable or attempt >= self.retry_policy.max_attempts:
                    raise
                delay = self.retry_policy.sleep_for_attempt(attempt)
                logger.warning(
                    "Archive request failed (attempt %d/%d), retry in %.2fs: %s",
                    attempt,
                    self.retry_policy.max_attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)
                continue
            # Response-level retry: providers that return error objects
            # (instead of raising) still honour the retry policy.
            last_result = result
            http_status = getattr(result, "http_status", None)
            retryable = http_status in self.retry_policy.retry_statuses
            if not retryable or attempt >= self.retry_policy.max_attempts:
                return result
            delay = self.retry_policy.sleep_for_attempt(attempt)
            logger.warning(
                "Archive request got HTTP %s (attempt %d/%d), retry in %.2fs",
                http_status,
                attempt,
                self.retry_policy.max_attempts,
                delay,
            )
            time.sleep(delay)
        if last_result is not None:
            return last_result
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry loop exited without result")
