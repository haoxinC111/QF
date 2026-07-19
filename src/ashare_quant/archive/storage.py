"""Raw json.zst storage and Bronze Parquet conversion."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import zstandard

logger = logging.getLogger(__name__)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_raw_json_zst(
    path: Path,
    payload: bytes,
    *,
    immutable: bool = True,
) -> tuple[str, int]:
    """Compress raw JSON payload with Zstandard and write atomically.

    The returned digest is the SHA256 of the COMPRESSED bytes on disk, so it
    can be re-verified with sha256_file(path) at any time (immutability
    checks, catalog checksums and soak verification all share this semantic).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstandard.ZstdCompressor()
    compressed = cctx.compress(payload)
    compressed_sha256 = sha256_bytes(compressed)
    if immutable and path.exists():
        existing_sha256 = sha256_file(path)
        if existing_sha256 == compressed_sha256:
            logger.debug("Raw file already exists with identical SHA256: %s", path)
            return existing_sha256, path.stat().st_size
        raise FileExistsError(
            f"Raw 文件已存在但 SHA256 不一致，禁止覆盖: {path}"
        )

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_bytes(compressed)
        tmp.replace(path)
    except Exception:
        # Atomic publish: never leave a half-written tmp file behind.
        tmp.unlink(missing_ok=True)
        raise
    return compressed_sha256, len(compressed)


def load_raw_json_zst(path: Path) -> bytes:
    """Decompress a raw json.zst file back to original JSON bytes."""
    compressed = path.read_bytes()
    dctx = zstandard.ZstdDecompressor()
    return dctx.decompress(compressed)


def raw_to_dataframe(
    columns: list[str],
    items: list[list[Any]],
) -> pd.DataFrame:
    """Convert provider items to DataFrame preserving column order."""
    df = pd.DataFrame(items, columns=columns)
    return df


def save_bronze_parquet(
    path: Path,
    df: pd.DataFrame,
    *,
    compression: str = "zstd",
    immutable: bool = True,
) -> tuple[str, int]:
    """Write a Bronze Parquet partition atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        # Parquet rewriting is deterministic only if schema/order are fixed;
        # for safety we refuse to overwrite.
        raise FileExistsError(
            f"Bronze 文件已存在，禁止覆盖: {path}"
        )

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, engine="pyarrow", compression=compression, index=False)
        tmp.replace(path)
    except Exception:
        # Atomic publish: never leave a half-written tmp file behind.
        tmp.unlink(missing_ok=True)
        raise
    return sha256_file(path), path.stat().st_size


def load_bronze_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


@dataclass
class StoredPartition:
    """Lineage record for one saved partition."""

    raw_path: Path
    bronze_path: Path
    raw_sha256: str
    bronze_sha256: str
    raw_bytes: int
    bronze_bytes: int
    row_count: int
    columns: list[str]


def store_response(
    raw_dir: Path,
    bronze_dir: Path,
    api_name: str,
    params: dict[str, Any],
    columns: list[str],
    items: list[list[Any]],
    raw_payload: bytes,
    *,
    snapshot_id: str,
    partition_key: str,
    compression: str = "zstd",
    immutable: bool = True,
) -> StoredPartition:
    """Persist raw JSON and Bronze Parquet for a single response."""
    safe_api = "".join(c if c.isalnum() or c in "-_" else "_" for c in api_name)
    raw_name = f"{safe_api}_{partition_key}_{snapshot_id}.json.zst"
    bronze_name = f"{safe_api}_{partition_key}_{snapshot_id}.parquet"
    raw_path = raw_dir / raw_name
    bronze_path = bronze_dir / bronze_name

    raw_sha256, raw_bytes = save_raw_json_zst(
        raw_path, raw_payload, immutable=immutable
    )

    df = raw_to_dataframe(columns, items)
    bronze_sha256, bronze_bytes = save_bronze_parquet(
        bronze_path, df, compression=compression, immutable=immutable
    )

    return StoredPartition(
        raw_path=raw_path,
        bronze_path=bronze_path,
        raw_sha256=raw_sha256,
        bronze_sha256=bronze_sha256,
        raw_bytes=raw_bytes,
        bronze_bytes=bronze_bytes,
        row_count=len(df),
        columns=list(df.columns),
    )
