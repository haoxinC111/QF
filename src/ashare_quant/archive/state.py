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
    # 父任务触发行数上限后被拆分为不重叠子任务,且全部子任务已终态解决;
    # 父任务自身不落盘(数据由子任务承载),属于已解决的终态。
    BISECTED = "bisected"
    # 所属 snapshot 在正式启动前被中止(2026-07-21 用户指令):
    # 非终态行被受控迁移到本状态,任务与已落盘文件保留不删,
    # 该 snapshot 永不续跑,后续批次使用新 snapshot(新 task_id)。
    ABORTED_PRESTART = "aborted_prestart"
    # 任务的参数拆分方式被证明与网关能力不匹配(2026-07-22 B3 repair):
    # 该任务永不可取到数据,由替代拆分方式的新任务集承载数据
    # (supersedes 映射与替代 manifest SHA 记录于 metadata/审计文件),
    # 绝不标 success,属于已解决的终态。首个用例: fina_audit 年段拆分。
    SUPERSEDED_INVALID_PARTITION = "superseded_invalid_partition"
    # context 漂移孤儿任务(2026-07-22 用户指令): 断点续跑错误地用新交易日
    # 重建 context,导致 params(如 end_date)漂移、task_id 重算,产生不属于
    # 原 manifest 的任务。此类任务的数据库记录与 Raw/Bronze 文件保留不删、
    # 不覆盖,但 research_eligible=false,且从 manifest/decision/fixtures/
    # 研究选择器中排除(它们不在原 manifest 的 task_id 集合内,天然排除)。
    ORPHANED_CONTEXT_DRIFT = "orphaned_context_drift"
    # 响应恰满真实行上限被静默截断的任务(2026-07-23 B4 repair 用户指令):
    # 该任务的数据可能不完整,由不重叠日期窗口递归二分的任务集承载数据
    # (supersedes 映射与 repair manifest SHA 记录于 metadata/审计文件),
    # 绝不标 success,属于已解决的终态。首个用例: share_float 真实 cap 6000。
    SUPERSEDED_TRUNCATED_CAP = "superseded_truncated_cap"
    # 2026-07-19 index_weight 撞名事件遗留的僵尸 running 任务(2026-07-23
    # 用户指令): 旧宇宙/旧交易日生成的 2026 年段任务在 universe 重建后从未被
    # 重跑,永远停在 running。其日期范围已被新一代同名指数、同区间(或覆盖
    # 区间)的任务完全承载,经审计迁移到本状态,research_eligible=false,
    # 绝不标 success,属于已解决的终态;原状态与后继映射保留在迁移 JSONL。
    SUPERSEDED_LEGACY_COLLISION = "superseded_legacy_collision"


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
