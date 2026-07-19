"""Schema fingerprinting for archive responses."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def normalize_dtype(dtype: Any) -> str:
    """Return a canonical string for a pandas/pyarrow dtype."""
    name = str(dtype)
    # Collapse specific nullable integer/float variants for stability.
    if name.startswith("int") or name.startswith("Int"):
        return "integer"
    if name.startswith("float") or name.startswith("Float"):
        return "float"
    if name.startswith("bool") or name.startswith("Bool"):
        return "boolean"
    if name.startswith("datetime"):
        return "datetime"
    return "string"


def schema_fingerprint(
    columns: list[str],
    df: pd.DataFrame | None = None,
    dtypes: dict[str, str] | None = None,
) -> str:
    """Compute a stable SHA256 fingerprint of a table schema.

    Parameters
    ----------
    columns:
        Ordered column names.
    df:
        Optional DataFrame from which to infer normalized dtypes.
    dtypes:
        Optional explicit dtype map.  If both are provided, ``dtypes`` wins.
    """
    if dtypes is None:
        if df is not None:
            dtypes = {col: normalize_dtype(df[col].dtype) for col in columns}
        else:
            dtypes = {col: "unknown" for col in columns}
    schema = {
        "columns": columns,
        "dtypes": {col: dtypes.get(col, "unknown") for col in columns},
    }
    payload = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def schema_diff(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Return differences between two schema fingerprints."""
    left_cols = set(left.get("columns", []))
    right_cols = set(right.get("columns", []))
    return {
        "added_columns": sorted(right_cols - left_cols),
        "removed_columns": sorted(left_cols - right_cols),
        "dtype_changes": {
            col: {"left": left.get("dtypes", {}).get(col), "right": right.get("dtypes", {}).get(col)}
            for col in left_cols & right_cols
            if left.get("dtypes", {}).get(col) != right.get("dtypes", {}).get(col)
        },
    }


class SchemaRegistry:
    """On-disk registry of schema fingerprints per endpoint."""

    def __init__(self, directory: Any) -> None:
        from pathlib import Path

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, endpoint: str, fingerprint: str) -> Any:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in endpoint)
        return self.directory / safe / f"{fingerprint}.json"

    def save(
        self,
        endpoint: str,
        fingerprint: str,
        schema: dict[str, Any],
    ) -> None:
        path = self._path(endpoint, fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(schema, ensure_ascii=False, indent=2)
        path.write_text(payload, encoding="utf-8")

    def load(self, endpoint: str, fingerprint: str) -> dict[str, Any] | None:
        path = self._path(endpoint, fingerprint)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def latest(self, endpoint: str) -> dict[str, Any] | None:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in endpoint)
        directory = self.directory / safe
        if not directory.exists():
            return None
        files = sorted(directory.glob("*.json"))
        if not files:
            return None
        latest = files[-1]
        return json.loads(latest.read_text(encoding="utf-8"))

    def fingerprints(self, endpoint: str) -> set[str]:
        """All registered fingerprints for an endpoint.

        Legitimate historical schema variants (e.g. daily_basic's early
        4-column era) are registered as extra fingerprints after manual
        verification, so the drift guard can accept any known layout while
        still quarantining never-seen ones.
        """
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in endpoint)
        directory = self.directory / safe
        if not directory.exists():
            return set()
        return {path.stem for path in directory.glob("*.json")}
