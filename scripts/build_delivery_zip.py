#!/usr/bin/env python
"""Build the Phase A.1 + P0 delivery ZIP (v2.0.0a11).

Whitelist-based collection: source, tests, docs, config example, and the
sanitized report set. The archive MUST NOT contain tokens, local config,
or Raw/Bronze data. Every included text file is scanned for the live
QF_ARCHIVE_API_TOKEN value and the string of the official TUSHARE_TOKEN
environment variable before packing.

Usage:
    python scripts/build_delivery_zip.py --version 2.0.0a11
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS = REPO_ROOT / "data_lake" / "reports"
CATALOG = REPO_ROOT / "data_lake" / "catalog"
OUT_DIR = REPO_ROOT / "dist"

# Whitelisted source/doc files (relative to repo root).
SOURCE_GLOBS = [
    "src/**/*.py",
    "tests/**/*.py",
    "docs/*.md",
    "scripts/*.py",
    "scripts/*.sh",
]
SOURCE_FILES = [
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "README_REPRODUCE.md",
    "config.archive.example.yaml",
    ".gitignore",
]
REPORT_FILES = [
    "permission_report.json",
    "empty_task_review.json",
    "market_cross_source_details.csv",
    "financial_cross_source_details.csv",
    "cross_source_summary.json",
    "cross_source_validation.md",
    "phase_a1_decision.json",
    "soak_test_report.json",
    "soak_test_report.md",
    "test_run_summary.txt",
]
CATALOG_FILES = ["endpoint_inventory.yaml"]
BATCH_GLOBS = [
    "batch_manifest.jsonl",
    "checksums.sha256",
    "coverage_report.md",
    "schema_report.json",
    "failure_queue.jsonl",
    "batch_decision.json",
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".toml", ".lock", ".sh", ".jsonl", ".sha256"}


def _secret_values() -> dict[str, str]:
    """Secrets that must never appear in the package (label -> value)."""
    secrets: dict[str, str] = {}
    qf = os.environ.get("QF_ARCHIVE_API_TOKEN", "").strip()
    if qf:
        secrets["QF_ARCHIVE_API_TOKEN"] = qf
    ts = os.environ.get("TUSHARE_TOKEN", "").strip()
    if ts:
        secrets["TUSHARE_TOKEN"] = ts
    return secrets


def collect_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in SOURCE_GLOBS:
        files.update(REPO_ROOT.glob(pattern))
    for rel in SOURCE_FILES:
        path = REPO_ROOT / rel
        if path.exists():
            files.add(path)
    for rel in REPORT_FILES:
        path = REPORTS / rel
        if path.exists():
            files.add(path)
        else:
            print(f"  [warn] 报告缺失: {rel}")
    for rel in CATALOG_FILES:
        path = CATALOG / rel
        if path.exists():
            files.add(path)
        else:
            print(f"  [warn] catalog 缺失: {rel}")
    batches_dir = REPORTS / "batches"
    if batches_dir.exists():
        for batch in sorted(batches_dir.iterdir()):
            if not batch.is_dir():
                continue
            for name in BATCH_GLOBS:
                path = batch / name
                if path.exists():
                    files.add(path)
                else:
                    print(f"  [warn] 批次产物缺失: {batch.name}/{name}")
    return sorted(files)


def scan_for_secrets(files: list[Path]) -> list[str]:
    """Return a list of leak descriptions; empty means clean."""
    secrets = _secret_values()
    leaks: list[str] = []
    forbidden_names = {"config.archive.yaml"}
    for path in files:
        if path.name in forbidden_names:
            leaks.append(f"禁止入包文件: {path.name}")
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, value in secrets.items():
            if value and value in text:
                leaks.append(f"{path.relative_to(REPO_ROOT)} 含 {label} 值")
    return leaks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="2.0.0a11")
    args = parser.parse_args()

    files = collect_files()
    print(f"收集文件 {len(files)} 个")

    leaks = scan_for_secrets(files)
    if leaks:
        print("泄漏扫描未通过:")
        for leak in leaks:
            print(f"  - {leak}")
        return 1
    print("泄漏扫描: 通过")

    OUT_DIR.mkdir(exist_ok=True)
    zip_path = OUT_DIR / f"QF_data_acquisition_phase_a1_p0_{args.version}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(REPO_ROOT))
    print(f"ZIP: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KiB)")

    # Integrity: unzip -t + external SHA256.
    test = subprocess.run(["unzip", "-t", str(zip_path)], capture_output=True, text=True)
    if test.returncode != 0:
        print(test.stdout[-2000:])
        print("unzip -t 失败")
        return 1
    print("unzip -t: 通过")

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    sha_path = zip_path.with_suffix(".zip.sha256")
    sha_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    print(f"SHA256: {digest}")
    print(f"校验文件: {sha_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
