#!/usr/bin/env bash
# Overnight volume pass (user-approved levers, 2026-07-18):
#   Phase A: same-object re-paste on proven images (fast, high yield)
#   Phase B: recovery walk over all recoverable failures with the
#            relaxed_v2 profile (min paste size 1.2%, rec1 stream)
# Both phases: stage1_recovery_v2 preset, rows tagged relaxed_v2,
# placement gates identical to strict. Fully resumable.

set -u
cd "$(dirname "$0")"

SEED=1337
SHARDS=6
TARGET=20000
OUTDIR="output/stage1_full"
PY="venv/bin/python"
LOG="$OUTDIR/logs/overnight_run.log"

mkdir -p "$OUTDIR/logs"
echo "=== OVERNIGHT RUN start $(date -u +%FT%TZ) ===" >> "$LOG"

for i in $(seq 0 $((SHARDS - 1))); do
    echo "=== RePaste shard $i/$SHARDS start $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    "$PY" generator/main.py \
        --stage stage1_recovery_v2 \
        --extend-variants --reuse-sources \
        --num-images "$TARGET" --seed "$SEED" \
        --output-dir "$OUTDIR" \
        --shard "$i" --num-shards "$SHARDS" >> "$LOG" 2>&1
    echo "=== RePaste shard $i/$SHARDS exit=$? $(date -u +%FT%TZ) ===" | tee -a "$LOG"
done

for i in $(seq 0 $((SHARDS - 1))); do
    echo "=== RecoveryV2 shard $i/$SHARDS start $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    "$PY" generator/main.py \
        --stage stage1_recovery_v2 \
        --retry-failed \
        --num-images "$TARGET" --seed "$SEED" \
        --output-dir "$OUTDIR" \
        --shard "$i" --num-shards "$SHARDS" >> "$LOG" 2>&1
    echo "=== RecoveryV2 shard $i/$SHARDS exit=$? $(date -u +%FT%TZ) ===" | tee -a "$LOG"
done

echo "=== OVERNIGHT RUN done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
echo "Ledger count: $(wc -l < "$OUTDIR/completed.jsonl" 2>/dev/null || echo 0)"
