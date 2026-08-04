#!/usr/bin/env bash
# Recovery pass: retry the strict run's recoverable failures with the
# stage1_recovery preset (relaxed source selection, identical
# placement gates). Appends to the same output dir/ledger; new rows
# are tagged gate_profile=relaxed_v1. Resumable exactly like the
# main run — safe to kill and re-run at any time.

set -u
cd "$(dirname "$0")"

SEED=1337
SHARDS=6
TARGET=20000
OUTDIR="output/stage1_full"
PY="venv/bin/python"
LOG="$OUTDIR/logs/recovery_run.log"

mkdir -p "$OUTDIR/logs"
echo "=== RECOVERY RUN start $(date -u +%FT%TZ) ===" >> "$LOG"

for i in $(seq 0 $((SHARDS - 1))); do
    echo "=== Recovery shard $i/$SHARDS start $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    "$PY" generator/main.py \
        --stage stage1_recovery \
        --retry-failed \
        --num-images "$TARGET" \
        --seed "$SEED" \
        --output-dir "$OUTDIR" \
        --shard "$i" --num-shards "$SHARDS" \
        >> "$LOG" 2>&1
    rc=$?
    echo "=== Recovery shard $i/$SHARDS exit=$rc $(date -u +%FT%TZ) ===" | tee -a "$LOG"
done

echo "=== RECOVERY RUN done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
echo "Ledger count: $(wc -l < "$OUTDIR/completed.jsonl" 2>/dev/null || echo 0)"
