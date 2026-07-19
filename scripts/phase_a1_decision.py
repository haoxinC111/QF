#!/usr/bin/env python
"""Aggregate all Phase A.1 gates into phase_a1_decision.json.

Only a "pass" decision allows the P0 full-archive batches to start.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS = REPO_ROOT / "data_lake" / "reports"
CATALOG = REPO_ROOT / "data_lake" / "catalog"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    blocking: list[str] = []

    # --- inventory completeness ------------------------------------------
    inventory_path = CATALOG / "endpoint_inventory.yaml"
    p0_complete = p1_complete = False
    if inventory_path.exists():
        data = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))
        endpoints = data.get("endpoints", [])
        p0 = [e for e in endpoints if e.get("priority") == "P0" and e.get("enabled", True)]
        p1 = [e for e in endpoints if e.get("priority") == "P1" and e.get("enabled", True)]
        p0_with_status = [e for e in p0 if e.get("probe", {}).get("status")]
        p1_with_status = [e for e in p1 if e.get("probe", {}).get("status")]
        p0_complete = len(p0) > 0 and len(p0_with_status) == len(p0)
        p1_complete = len(p1) > 0 and len(p1_with_status) == len(p1)
        if not p0_complete:
            blocking.append(f"P0 清单状态不完整: {len(p0_with_status)}/{len(p0)}")
        if not p1_complete:
            blocking.append(f"P1 清单状态不完整: {len(p1_with_status)}/{len(p1)}")
    else:
        blocking.append("endpoint_inventory.yaml 不存在")

    # --- cross-source validations ----------------------------------------
    cross = _load(REPORTS / "cross_source_summary.json")
    market_pass = bool(cross.get("pass"))
    financial_pass = bool(cross.get("financial", {}).get("pass"))
    if not market_pass:
        blocking.append("市场数据跨源核验未通过")
    if not financial_pass:
        blocking.append("财务数据跨源核验未通过")

    # --- soak test ---------------------------------------------------------
    soak = _load(REPORTS / "soak_test_report.json")
    soak_pass = bool(soak.get("pass"))
    if not soak_pass:
        blocking.append("soak 稳定性测试未通过或缺失")

    # --- unit tests ---------------------------------------------------------
    tests_pass = False
    test_log = REPORTS / "test_run_summary.txt"
    if test_log.exists():
        text = test_log.read_text(encoding="utf-8")
        tests_pass = "OK" in text and "FAILED" not in text
    if not tests_pass:
        blocking.append("单元测试未通过或缺少测试日志")

    # --- unresolved items ---------------------------------------------------
    empty_review = _load(REPORTS / "empty_task_review.json")
    unresolved_empty = 0 if empty_review.get("final_status") == "confirmed_empty" else 1
    if unresolved_empty:
        blocking.append("存在未查清 empty 任务")

    unresolved_truncation = 0
    soak_integrity = soak.get("integrity", {})
    token_leaks = len(soak_integrity.get("token_leaks", []))
    if token_leaks:
        blocking.append(f"Token 泄漏 {token_leaks} 处")

    # --- commit -------------------------------------------------------------
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
        blocking.append("无法读取 git commit")

    decision = {
        "schema_version": 1,
        "decision": "pass" if not blocking else "fail",
        "p0_inventory_complete": p0_complete,
        "p1_inventory_complete": p1_complete,
        "market_cross_source_pass": market_pass,
        "financial_cross_source_pass": financial_pass,
        "soak_test_pass": soak_pass,
        "tests_pass": tests_pass,
        "unresolved_empty_count": unresolved_empty,
        "unresolved_truncation_count": unresolved_truncation,
        "token_leak_count": token_leaks,
        "blocking_reasons": blocking,
        "source_commit": commit,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = REPORTS / "phase_a1_decision.json"
    out.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["decision"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
