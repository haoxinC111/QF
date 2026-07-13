from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping


REPRODUCIBILITY_SCHEMA_VERSION = 2
ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
DEPENDENCY_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "PyYAML",
    "matplotlib",
    "requests",
    "tushare",
    "akshare",
    "akracer",
    "mini-racer",
    "py-mini-racer",
)
RUNTIME_MODULES = ("py_mini_racer",)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(
    path: str | Path,
    *,
    root: str | Path | None = None,
    logical_path: str | None = None,
) -> dict[str, Any]:
    target = Path(path).resolve()
    if logical_path is not None:
        display = logical_path
    elif root is not None:
        display = target.relative_to(Path(root).resolve()).as_posix()
    else:
        display = target.name
    return {
        "path": display,
        "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def build_file_inventory(
    root: str | Path,
    paths: Iterable[str | Path],
) -> list[dict[str, Any]]:
    base = Path(root).resolve()
    unique = sorted({Path(path).resolve() for path in paths})
    return [file_fingerprint(path, root=base) for path in unique]


def inventory_sha256(files: Iterable[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "path": str(item["path"]),
            "size_bytes": int(item["size_bytes"]),
            "sha256": str(item["sha256"]),
        }
        for item in files
    ]
    return payload_sha256(sorted(normalized, key=lambda item: item["path"]))


def verify_file_inventory(
    root: str | Path,
    files: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    base = Path(root).resolve()
    entries = list(files)
    if not entries:
        raise ValueError("数据清单没有文件指纹，无法验证缓存完整性")
    errors: list[str] = []
    for item in entries:
        if not isinstance(item, Mapping):
            errors.append("文件指纹记录不是映射结构")
            continue
        relative = Path(str(item.get("path", "")))
        target = (base / relative).resolve()
        if target != base and base not in target.parents:
            errors.append(f"非法清单路径: {relative}")
            continue
        if not target.is_file():
            errors.append(f"缺少文件: {relative.as_posix()}")
            continue
        expected_size = int(item.get("size_bytes", -1))
        if target.stat().st_size != expected_size:
            errors.append(f"文件大小变化: {relative.as_posix()}")
            continue
        expected_hash = str(item.get("sha256", ""))
        if sha256_file(target) != expected_hash:
            errors.append(f"SHA256 不匹配: {relative.as_posix()}")
    if errors:
        sample = "；".join(errors[:5])
        raise ValueError(f"缓存完整性校验失败（{len(errors)} 项）: {sample}")
    return {
        "verified": True,
        "file_count": len(entries),
        "inventory_sha256": inventory_sha256(entries),
    }


def _project_root(start: str | Path | None = None) -> Path | None:
    current = Path(start).resolve() if start else Path(__file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _git_metadata(root: Path | None) -> dict[str, Any]:
    if root is None or not (root / ".git").exists():
        return {"available": False, "commit": None, "branch": None, "dirty": None}

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current") or None
        dirty = bool(run("status", "--porcelain", "--untracked-files=all"))
        return {"available": True, "commit": commit, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "branch": None, "dirty": None}


def _source_fingerprint(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"file_count": 0, "sha256": None, "files": []}
    candidates: list[Path] = []
    for name in [
        "pyproject.toml",
        "requirements.txt",
        "requirements-public.txt",
        "uv.lock",
        "run.py",
    ]:
        path = root / name
        if path.is_file():
            candidates.append(path)
    source_root = root / "src"
    if source_root.is_dir():
        candidates.extend(source_root.rglob("*.py"))
    files = build_file_inventory(root, candidates)
    return {
        "file_count": len(files),
        "sha256": inventory_sha256(files) if files else None,
        "files": files,
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _runtime_module_provenance() -> dict[str, dict[str, Any]]:
    """Identify importable runtime modules, including shared namespace providers."""
    package_providers = metadata.packages_distributions()
    modules: dict[str, dict[str, Any]] = {}
    for module_name in RUNTIME_MODULES:
        providers = sorted(package_providers.get(module_name, []))
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        modules[module_name] = {
            "available": bool(origin and origin.is_file()),
            "providers": providers,
            "origin": str(origin) if origin else None,
            "sha256": sha256_file(origin) if origin and origin.is_file() else None,
        }
    return modules


def build_reproducibility_manifest(
    config_payload: Mapping[str, Any],
    *,
    data_manifest_path: str | Path | None = None,
    extra_input_files: Iterable[str | Path] = (),
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    source = _source_fingerprint(root)
    git = _git_metadata(root)
    dependencies = _dependency_versions()
    runtime_modules = _runtime_module_provenance()
    config = dict(config_payload)

    data: dict[str, Any] = {
        "manifest_path": None,
        "manifest_sha256": None,
        "schema_version": None,
        "data_fingerprint_sha256": None,
    }
    if data_manifest_path is not None:
        manifest_path = Path(data_manifest_path).resolve()
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            data = {
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "schema_version": payload.get("schema_version"),
                "data_fingerprint_sha256": payload.get("data_fingerprint_sha256"),
            }

    inputs = [
        file_fingerprint(path, logical_path=Path(path).name)
        for path in sorted({Path(value).resolve() for value in extra_input_files})
    ]
    identity = {
        "config_sha256": payload_sha256(config),
        "source_sha256": source["sha256"],
        "git_commit": git["commit"],
        "data_manifest_sha256": data["manifest_sha256"],
        "data_fingerprint_sha256": data["data_fingerprint_sha256"],
        "extra_inputs_sha256": inventory_sha256(inputs) if inputs else None,
        "dependencies": dependencies,
        "runtime_modules": {
            name: {
                "available": details["available"],
                "providers": details["providers"],
                "sha256": details["sha256"],
            }
            for name, details in runtime_modules.items()
        },
        "python": platform.python_version(),
    }
    return {
        "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_fingerprint_sha256": payload_sha256(identity),
        "config": {"sha256": identity["config_sha256"], "resolved": config},
        "source": source,
        "git": git,
        "data": data,
        "extra_inputs": inputs,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "dependencies": dependencies,
            "modules": runtime_modules,
        },
    }


def write_json_atomic(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(target)
    return target


def build_artifact_inventory(
    root: str | Path,
    artifacts: Iterable[str | Path],
) -> list[dict[str, Any]]:
    """Fingerprint result files relative to their output directory."""
    base = Path(root).resolve()
    resolved: set[Path] = set()
    for value in artifacts:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = base / candidate
        candidate = candidate.resolve()
        if candidate == base or base not in candidate.parents:
            raise ValueError(f"结果文件不在输出目录内: {candidate}")
        if not candidate.is_file():
            raise FileNotFoundError(f"结果文件不存在: {candidate}")
        resolved.add(candidate)
    return [file_fingerprint(path, root=base) for path in sorted(resolved)]


def write_artifact_manifest(
    root: str | Path,
    artifacts: Iterable[str | Path],
    *,
    filename: str = ARTIFACT_MANIFEST_FILENAME,
) -> Path:
    """Seal generated result files; the manifest intentionally excludes itself."""
    base = Path(root).resolve()
    target = base / filename
    files = build_artifact_inventory(base, artifacts)
    payload = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": files,
        "artifact_set_sha256": inventory_sha256(files),
    }
    return write_json_atomic(payload, target)


def verify_artifact_manifest(
    manifest_path: str | Path,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Verify every sealed result byte and optionally reject unsealed files."""
    target = Path(manifest_path).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"缺少结果指纹清单: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"结果指纹清单不是有效 JSON: {target}") from exc
    if payload.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"不支持的结果指纹版本: {payload.get('schema_version')!r}"
        )
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("结果指纹清单没有文件记录")
    expected_set_hash = str(payload.get("artifact_set_sha256", ""))
    actual_set_hash = inventory_sha256(files)
    if expected_set_hash != actual_set_hash:
        raise ValueError("结果指纹清单自身的集合 SHA256 不匹配")
    verification = verify_file_inventory(target.parent, files)
    sealed_paths = {str(item["path"]) for item in files}
    actual_paths = {
        path.relative_to(target.parent).as_posix()
        for path in target.parent.rglob("*")
        if path.is_file() and path != target
    }
    unsealed_paths = sorted(actual_paths - sealed_paths)
    if strict and unsealed_paths:
        raise ValueError(
            "结果目录包含未封存文件: " + ", ".join(unsealed_paths[:10])
        )
    return {
        **verification,
        "artifact_set_sha256": actual_set_hash,
        "unsealed_paths": unsealed_paths,
        "strict": strict,
    }


def record_experiment(
    registry_path: str | Path,
    reproducibility: Mapping[str, Any],
    *,
    experiment_type: str,
    protocol: Mapping[str, Any],
    artifacts: Iterable[str | Path],
) -> Path:
    """Append an idempotent, auditable run record to a JSONL registry."""
    target = Path(registry_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    experiment_id = str(reproducibility["run_fingerprint_sha256"])
    artifact_files = build_artifact_inventory(target.parent, artifacts)
    record = {
        "experiment_id": experiment_id,
        "created_at_utc": reproducibility.get("created_at_utc"),
        "experiment_type": experiment_type,
        "protocol": dict(protocol),
        "config_sha256": reproducibility["config"]["sha256"],
        "source_sha256": reproducibility["source"]["sha256"],
        "git_commit": reproducibility["git"]["commit"],
        "git_dirty": reproducibility["git"]["dirty"],
        "data_fingerprint_sha256": reproducibility["data"][
            "data_fingerprint_sha256"
        ],
        "artifacts": artifact_files,
    }
    existing: list[dict[str, Any]] = []
    if target.is_file():
        for number, line in enumerate(
            target.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"实验登记表第 {number} 行不是有效 JSON，拒绝覆盖"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"实验登记表第 {number} 行不是对象结构")
            existing.append(parsed)
    matching_index = next(
        (
            index
            for index, item in enumerate(existing)
            if item.get("experiment_id") == experiment_id
        ),
        None,
    )
    if matching_index is not None:
        # Output directories are overwritten on a rerun. Keep the registry entry for
        # this reproducible identity aligned with the bytes that now exist on disk.
        existing[matching_index] = record
    temporary = target.with_name(target.name + ".tmp")
    lines = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in existing]
    if matching_index is None:
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target
