"""Task state database for resumable archive downloads."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    CONFIRMED_EMPTY = "confirmed_empty"
    RETRYABLE_ERROR = "retryable_error"
    DENIED = "denied"
    INVALID_PARAMS = "invalid_params"
    SUSPECT_TRUNCATED = "suspect_truncated"
    QUARANTINED = "quarantined"


@dataclass
class DownloadTask:
    task_id: str
    api_name: str
    params: dict[str, Any]
    fields: str
    dataset: str
    priority: str
    primary_key: list[str]
    primary_split: str | None
    fallback_split: str | None
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    last_error: str = ""
    row_count: int = 0
    raw_path: str = ""
    bronze_path: str = ""
    schema_fingerprint: str = ""
    raw_sha256: str = ""
    fetched_at_utc: str = ""
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> tuple[Any, ...]:
        return (
            self.task_id,
            self.api_name,
            json.dumps(self.params, sort_keys=True, ensure_ascii=False),
            self.fields,
            self.dataset,
            self.priority,
            json.dumps(self.primary_key, ensure_ascii=False),
            self.primary_split,
            self.fallback_split,
            self.status.value,
            self.attempts,
            self.last_error,
            self.row_count,
            self.raw_path,
            self.bronze_path,
            self.schema_fingerprint,
            self.raw_sha256,
            self.fetched_at_utc,
            self.elapsed_seconds,
            json.dumps(self.metadata, sort_keys=True, ensure_ascii=False),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DownloadTask":
        return cls(
            task_id=row["task_id"],
            api_name=row["api_name"],
            params=json.loads(row["params_json"]),
            fields=row["fields"],
            dataset=row["dataset"],
            priority=row["priority"],
            primary_key=json.loads(row["primary_key_json"] or "[]"),
            primary_split=row["primary_split"],
            fallback_split=row["fallback_split"],
            status=TaskStatus(row["status"]),
            attempts=row["attempts"],
            last_error=row["last_error"],
            row_count=row["row_count"],
            raw_path=row["raw_path"],
            bronze_path=row["bronze_path"],
            schema_fingerprint=row["schema_fingerprint"],
            raw_sha256=row["raw_sha256"],
            fetched_at_utc=row["fetched_at_utc"],
            elapsed_seconds=row["elapsed_seconds"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )


class TaskStateDB:
    """SQLite-backed persistent task registry."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS archive_tasks (
                    task_id TEXT PRIMARY KEY,
                    api_name TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    fields TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    primary_key_json TEXT,
                    primary_split TEXT,
                    fallback_split TEXT,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    row_count INTEGER NOT NULL DEFAULT 0,
                    raw_path TEXT NOT NULL DEFAULT '',
                    bronze_path TEXT NOT NULL DEFAULT '',
                    schema_fingerprint TEXT NOT NULL DEFAULT '',
                    raw_sha256 TEXT NOT NULL DEFAULT '',
                    fetched_at_utc TEXT NOT NULL DEFAULT '',
                    elapsed_seconds REAL NOT NULL DEFAULT 0.0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at_utc TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_status ON archive_tasks(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api ON archive_tasks(api_name)"
            )

    def upsert(self, task: DownloadTask) -> None:
        updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                INSERT INTO archive_tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status=excluded.status,
                    attempts=excluded.attempts,
                    last_error=excluded.last_error,
                    row_count=excluded.row_count,
                    raw_path=excluded.raw_path,
                    bronze_path=excluded.bronze_path,
                    schema_fingerprint=excluded.schema_fingerprint,
                    raw_sha256=excluded.raw_sha256,
                    fetched_at_utc=excluded.fetched_at_utc,
                    elapsed_seconds=excluded.elapsed_seconds,
                    metadata_json=excluded.metadata_json,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (*task.to_row(), updated_at),
            )

    def get(self, task_id: str) -> DownloadTask | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM archive_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            return DownloadTask.from_row(row) if row else None

    def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        api_name: str | None = None,
        dataset: str | None = None,
    ) -> list[DownloadTask]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM archive_tasks WHERE 1=1"
            params: list[Any] = []
            if status:
                query += " AND status = ?"
                params.append(status.value)
            if api_name:
                query += " AND api_name = ?"
                params.append(api_name)
            if dataset:
                query += " AND dataset = ?"
                params.append(dataset)
            rows = conn.execute(query, params).fetchall()
            return [DownloadTask.from_row(row) for row in rows]

    def count_by_status(self) -> dict[str, int]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM archive_tasks GROUP BY status"
            ).fetchall()
            return {row[0]: row[1] for row in rows}

    def reset_retryable(self) -> int:
        with sqlite3.connect(self.path) as conn:
            cur = conn.execute(
                "UPDATE archive_tasks SET status = ?, attempts = 0, last_error = '' "
                "WHERE status = ?",
                (TaskStatus.PENDING.value, TaskStatus.RETRYABLE_ERROR.value),
            )
            return cur.rowcount

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.count_total(),
            "by_status": self.count_by_status(),
        }

    def count_total(self) -> int:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM archive_tasks").fetchone()
            return row[0] if row else 0
