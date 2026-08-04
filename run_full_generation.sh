#!/usr/bin/env bash
# Full-scale Stage 1 generation over all of COCO train2017, sharded.
#
# Shards run sequentially so peak memory stays at ~1/SHARDS of the
# annotation data (WSL2-safe). All shards append to the same output
# directory through the resume ledger (completed.jsonl), so this
# script — and any individual shard — can be killed and re-run at
# any time without losing or duplicating work.

set -u
cd "$(dirname "$0")"

SEED=1337
SHARDS=6
TARGET=20000
OUTDIR="output/stage1_full"
PY="venv/bin/python"
LOG="$OUTDIR/logs/full_run.log"

mkdir -p "$OUTDIR/logs"
echo "=== FULL RUN start $(date -u +%FT%TZ) seed=$SEED target=$TARGET shards=$SHARDS ===" >> "$LOG"

for i in $(seq 0 $((SHARDS - 1))); do
    echo "=== Shard $i/$SHARDS start $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    "$PY" generator/main.py \
        --stage stage1 \
        --num-images "$TARGET" \
        --seed "$SEED" \
        --output-dir "$OUTDIR" \
        --shard "$i" --num-shards "$SHARDS" \
        >> "$LOG" 2>&1
    rc=$?
    echo "=== Shard $i/$SHARDS exit=$rc $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    # A non-zero shard is resumable — continue with the next shard;
    # rerunning this script later picks up whatever was missed.
done

echo "=== FULL RUN done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
echo "Ledger count: $(wc -l < "$OUTDIR/completed.jsonl" 2>/dev/null || echo 0)"
