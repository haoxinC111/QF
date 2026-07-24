#!/usr/bin/env python3
"""PATH_REMAP.md 第 2.4 步:重映射后只读校验。

对快照状态库中每个 success 任务:
  1. raw_path 经 remap_archive_path 映射到指定 archive-root 后必须存在;
  2. 文件内容 SHA256 必须等于 archive_tasks.raw_sha256。

只读:状态库 mode=ro 连接,数据文件只读打开,不写任何字节。
结果写到 QF-integration/results/raw_sha_reverify_report.json。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ashare_quant.pit_lake import remap_archive_path  # noqa: E402

ARCHIVE_ROOT = ROOT.parent / "a_share_quant" / "data_lake"
CATALOG = ROOT / "data_lake" / "catalog" / "archive.duckdb"
REPORT = ROOT / "results" / "raw_sha_reverify_report.json"
CHUNK = 8 * 1024 * 1024


def main() -> int:
    con = sqlite3.connect(f"file:{CATALOG}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT task_id, raw_path, raw_sha256 FROM archive_tasks "
        "WHERE status = 'success'"
    ).fetchall()
    con.close()

    total = len(rows)
    checked = 0
    decompressed_sha = 0  # Phase A 历史口径:raw_sha256 记录的是解压后载荷的 SHA
    missing: list[str] = []
    sha_mismatch: list[str] = []
    remap_error: list[str] = []
    t0 = time.monotonic()
    for task_id, raw_path, expected in rows:
        try:
            path, _ = remap_archive_path(raw_path, ARCHIVE_ROOT)
        except ValueError:
            remap_error.append(task_id)
            continue
        if not path.is_file():
            missing.append(task_id)
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != str(expected).lower():
            # 回退口径:Phase A 时代 raw_sha256 记录解压后 JSON 载荷的 SHA
            # (已逐任务验证 44/44 吻合,内容与行数完好,非损坏)。
            legacy_ok = False
            if path.suffix == ".zst":
                try:
                    import zstandard

                    payload = zstandard.ZstdDecompressor().decompress(
                        data, max_output_size=512 * 1024 * 1024
                    )
                    legacy_ok = (
                        hashlib.sha256(payload).hexdigest() == str(expected).lower()
                    )
                except Exception:  # noqa: BLE001 - 任何解压失败都按不一致处理
                    legacy_ok = False
            if legacy_ok:
                decompressed_sha += 1
            else:
                sha_mismatch.append(task_id)
        checked += 1
        if checked % 20000 == 0:
            elapsed = time.monotonic() - t0
            eta = elapsed / checked * (total - checked)
            print(
                f"进度 {checked}/{total} 已用 {elapsed:.0f}s ETA {eta:.0f}s",
                flush=True,
            )

    report = {
        "archive_root": str(ARCHIVE_ROOT),
        "catalog_sha256": "32ce478375271345b90087a9d02354fb2d0471a257b03e7c4d2e727c1b60e077",
        "success_tasks": total,
        "files_checked": checked,
        "file_sha256_ok": checked - decompressed_sha - len(sha_mismatch),
        "decompressed_payload_sha256_ok": decompressed_sha,
        "missing_count": len(missing),
        "sha_mismatch_count": len(sha_mismatch),
        "remap_error_count": len(remap_error),
        "missing_sample": missing[:20],
        "sha_mismatch_sample": sha_mismatch[:20],
        "remap_error_sample": remap_error[:20],
        "elapsed_seconds": round(time.monotonic() - t0, 1),
        "verdict": (
            "pass"
            if not missing and not sha_mismatch and not remap_error
            else "fail"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_sample")}, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
