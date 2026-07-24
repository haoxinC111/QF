#!/usr/bin/env python
"""物化并封存 B4_events context + 冻结任务清单(2026-07-22 用户指令)。

build_context 本地优先(封存 B0/B2 bronze,记录 context SHA),展开 B4 任务,
写入 frozen_specs_<snapshot>.json;启动/恢复一律从冻结清单回放
(load_resume_specs 的 frozen 回退路径),禁止按最新交易日重建。

Usage:
    set -a; . ./.env; set +a && uv run --no-sync python scripts/freeze_b4_specs.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ashare_quant.archive.batch import expand_batch, save_frozen_specs  # noqa: E402
from ashare_quant.archive.config import ArchiveConfig  # noqa: E402
from ashare_quant.archive.provider import TushareCompatibleHttpProvider  # noqa: E402
from ashare_quant.archive.registry import default_inventory  # noqa: E402
from run_p0_batch import build_context  # noqa: E402

BATCH = "B4_events"


def main() -> int:
    config = ArchiveConfig.from_yaml(REPO_ROOT / "config.archive.yaml")
    config.validate_for_run(batch_id=BATCH)
    config.ensure_dirs()
    provider = TushareCompatibleHttpProvider(
        url_env=config.provider.base_url_env,
        token_env=config.provider.token_env,
        forbid_token_env=config.provider.forbid_token_env,
        source_provider=config.provider.name,
        allowed_hosts=config.provider.allowed_hosts,
        api_key_env=config.provider.api_key_env,
        api_key_header=config.provider.api_key_header,
    )
    ctx = build_context(provider)
    inventory = default_inventory()
    specs = expand_batch(inventory, BATCH, ctx)

    snapshot_id = time.strftime(f"p0_{BATCH}_%Y%m%d_%H%M%S", time.gmtime())
    artifact_dir = REPO_ROOT / "data_lake" / "reports" / "batches" / BATCH
    context_record = {
        "batch": BATCH,
        "context_sha256": ctx.context_sha256,
        "universe_size": len(ctx.universe),
        "latest_trade_date": ctx.latest_trade_date,
        "latest_report_period": ctx.latest_report_period,
        "index_codes": len(ctx.index_codes),
        "index_codes_main": len(ctx.index_codes_main),
        "sources": ctx.sources,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    context_path = artifact_dir / f"context_{ctx.context_sha256[:12]}.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    context_path.write_text(json.dumps(context_record, ensure_ascii=False, indent=1), encoding="utf-8")

    frozen_path = artifact_dir / f"frozen_specs_{snapshot_id}.json"
    payload = save_frozen_specs(
        frozen_path,
        batch_id=BATCH,
        snapshot_id=snapshot_id,
        provider_name=config.provider.name,
        context_record=context_record,
        specs=specs,
    )
    by_api: dict[str, int] = {}
    for s in payload["specs"]:
        by_api[s["api_name"]] = by_api.get(s["api_name"], 0) + 1
    print(json.dumps({
        "snapshot_id": snapshot_id,
        "context_sha256": ctx.context_sha256,
        "latest_trade_date": ctx.latest_trade_date,
        "task_count": payload["task_count"],
        "by_api": by_api,
        "manifest_sha256": payload["manifest_sha256"],
        "frozen_path": str(frozen_path),
        "context_path": str(context_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
