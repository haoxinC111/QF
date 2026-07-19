#!/usr/bin/env bash
# Sequential P0 batch driver: B0 -> B4 with per-batch decision gates.
# Resumable: completed tasks are skipped via the shared state db.
# Logs: data_lake/reports/<batch>.log
set -uo pipefail
cd "$(dirname "$0")/.."

BATCHES=(B0_reference B1_market B2_universe B3_financial B4_events)
START_FROM="${1:-B0_reference}"
skip=true
for batch in "${BATCHES[@]}"; do
    if $skip; then
        if [[ "$batch" == "$START_FROM" ]]; then skip=false; else continue; fi
    fi
    echo "===== $(date -u +%FT%TZ) 启动批次 $batch ====="
    uv run --no-sync python scripts/run_p0_batch.py --batch "$batch"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "===== 批次 $batch 未通过 (rc=$rc)，停止后续批次 ====="
        exit $rc
    fi
    echo "===== $(date -u +%FT%TZ) 批次 $batch PASS ====="
done
echo "===== 全部批次完成 ====="
