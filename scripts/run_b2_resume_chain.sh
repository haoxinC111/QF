#!/usr/bin/env bash
# Detached ("released") P0 chain: resume B2 with its existing snapshot, then
# hand off to the standard driver for B3 -> B4.  Intended to run under
# nohup/setsid, independent of any Claude session.  The API token is read
# only from the environment (QF_ARCHIVE_API_TOKEN); nothing is written here.
set -uo pipefail
cd "$(dirname "$0")/.."

SNAPSHOT_B2="${1:-p0_B2_universe_20260717_022655}"
echo "===== $(date -u +%FT%TZ) 放生链条启动: B2 断点续跑 (snapshot=$SNAPSHOT_B2) ====="
uv run --no-sync python scripts/run_p0_batch.py --batch B2_universe --snapshot "$SNAPSHOT_B2"
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "===== 批次 B2_universe 未通过 (rc=$rc)，停止后续批次 ====="
    exit $rc
fi
echo "===== $(date -u +%FT%TZ) 批次 B2_universe PASS，接续 B3/B4 ====="
exec bash scripts/run_all_p0_batches.sh B3_financial
