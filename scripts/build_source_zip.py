#!/usr/bin/env python
"""Build a deterministic, secret-scanned full-source release ZIP."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_FILES = {
    ".gitignore",
    "LICENSE",
    "config.archive.example.yaml",
    "config.example.yaml",
    "pyproject.toml",
    "requirements-public.txt",
    "requirements.txt",
    "run.py",
    "uv.lock",
}
SOURCE_PATTERNS = (
    "*.md",
    "docs/**/*.md",
    "scripts/**/*.py",
    "scripts/**/*.sh",
    "src/**/*.py",
    "tests/**/*.py",
)
TEXT_SUFFIXES = {
    ".lock",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_NAMES = {"config.archive.yaml", "config.yaml", ".env"}
FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_source_files() -> list[Path]:
    files = {REPO_ROOT / name for name in ROOT_FILES}
    for pattern in SOURCE_PATTERNS:
        files.update(REPO_ROOT.glob(pattern))
    result = sorted(
        path.resolve()
        for path in files
        if path.is_file()
        and path.name not in FORBIDDEN_NAMES
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    missing = sorted(
        name for name in ROOT_FILES if not (REPO_ROOT / name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"发布必需文件缺失: {missing}")
    return result


def scan_for_secrets(files: list[Path]) -> list[str]:
    values = {
        name: value.strip()
        for name in ("QF_ARCHIVE_API_TOKEN", "TUSHARE_TOKEN")
        if len(value := os.environ.get(name, "").strip()) >= 8
    }
    leaks: list[str] = []
    for path in files:
        if path.name in FORBIDDEN_NAMES:
            leaks.append(f"禁止入包文件: {path.relative_to(REPO_ROOT)}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for label, value in values.items():
            if value in content:
                leaks.append(
                    f"{path.relative_to(REPO_ROOT)} 含当前 {label} 值"
                )
    return leaks


def _write_internal_checksums(staging: Path, files: list[Path]) -> Path:
    lines = [
        f"{_sha256(path)}  ./{path.relative_to(staging).as_posix()}"
        for path in sorted(files)
    ]
    target = staging / "PACKAGE_SHA256SUMS.txt"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _zip_file(
    archive: zipfile.ZipFile,
    source: Path,
    arcname: str,
) -> None:
    info = zipfile.ZipInfo(arcname, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    mode = 0o755 if os.access(source, os.X_OK) else 0o644
    info.external_attr = mode << 16
    archive.writestr(info, source.read_bytes())


def build(version: str, output_dir: Path) -> tuple[Path, Path, int]:
    files = collect_source_files()
    leaks = scan_for_secrets(files)
    if leaks:
        raise ValueError("发布包泄漏扫描失败:\n- " + "\n- ".join(leaks))
    root_name = f"ashare_quant_multifactor_v{version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{root_name}.zip"
    checksum_path = zip_path.with_suffix(".zip.sha256")
    with tempfile.TemporaryDirectory(prefix="qf-source-release-") as directory:
        staging = Path(directory) / root_name
        staging.mkdir()
        staged_files: list[Path] = []
        for source in files:
            target = staging / source.relative_to(REPO_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            staged_files.append(target)
        staged_files.append(_write_internal_checksums(staging, staged_files))
        temporary_zip = zip_path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(
            temporary_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for source in sorted(staged_files):
                relative = source.relative_to(staging).as_posix()
                _zip_file(archive, source, f"{root_name}/{relative}")
        os.replace(temporary_zip, zip_path)
    digest = _sha256(zip_path)
    checksum_path.write_text(
        f"{digest}  {zip_path.name}\n", encoding="utf-8"
    )
    return zip_path, checksum_path, len(files) + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="2.0.0a11")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "dist")
    args = parser.parse_args()
    zip_path, checksum_path, count = build(args.version, args.output.resolve())
    print(f"完整源码包: {zip_path}")
    print(f"文件数量: {count}")
    print(f"ZIP SHA256: {_sha256(zip_path)}")
    print(f"外部校验文件: {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
