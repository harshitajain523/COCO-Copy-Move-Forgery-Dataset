#!/usr/bin/env bash
# Variant-extension pass: revisit the images that already produced
# samples (proven placeable) and generate additional variants from
# not-yet-used source objects, under the stage1_recovery source
# gates. Rows tagged gate_profile=relaxed_v1; variant indices
# continue from each image's existing ones (max 8 total). Resumable.

set -u
cd "$(dirname "$0")"

SEED=1337
SHARDS=6
TARGET=20000
OUTDIR="output/stage1_full"
PY="venv/bin/python"
LOG="$OUTDIR/logs/extension_run.log"

mkdir -p "$OUTDIR/logs"
echo "=== EXTENSION RUN start $(date -u +%FT%TZ) ===" >> "$LOG"

for i in $(seq 0 $((SHARDS - 1))); do
    echo "=== Extension shard $i/$SHARDS start $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    "$PY" generator/main.py \
        --stage stage1_recovery \
        --extend-variants \
        --num-images "$TARGET" \
        --seed "$SEED" \
        --output-dir "$OUTDIR" \
        --shard "$i" --num-shards "$SHARDS" \
        >> "$LOG" 2>&1
    rc=$?
    echo "=== Extension shard $i/$SHARDS exit=$rc $(date -u +%FT%TZ) ===" | tee -a "$LOG"
done

echo "=== EXTENSION RUN done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
echo "Ledger count: $(wc -l < "$OUTDIR/completed.jsonl" 2>/dev/null || echo 0)"
