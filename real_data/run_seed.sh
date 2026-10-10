#!/bin/bash
# One training seed of the application (paper Section 6): train the model for
# both policy domains, estimate every outcome month and summary with saved
# bootstrap draws, and keep a compact per-seed file for pooling.
#
#   bash real_data/run_seed.sh SEED
#
# Environment variables (optional):
#   DATA_ROOT  inputs from build_inputs.py   (default real_data/inputs)
#   OUT_ROOT   output directory              (default real_data/runs)
#   DEVICE     training device: cpu, cuda, mps or auto (default cpu)
#   PYTHON     python executable              (default python3)
#   KEEP_WORK  1 keeps trained models and full estimate files
set -euo pipefail
SEED="$1"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO/real_data/inputs}"
OUT_ROOT="${OUT_ROOT:-$REPO/real_data/runs}"
DEVICE="${DEVICE:-cpu}"
PY="${PYTHON:-python3}"

TAG="$(printf 'seed_%04d' "$SEED")"
FINAL="$OUT_ROOT/$TAG.json.gz"
if [ -s "$FINAL" ]; then echo "$TAG already done"; exit 0; fi
WORK="$OUT_ROOT/work/$TAG"
mkdir -p "$WORK/logs"

for FAMILY in Business_Economic_Restrictions Education_Childcare; do
  "$PY" "$REPO/real_data/train.py" \
    --data-dir "$DATA_ROOT/$FAMILY" --output-dir "$WORK/$FAMILY" \
    --epochs 120 --batch-size 128 --seed "$SEED" --device "$DEVICE" \
    > "$WORK/logs/train_$FAMILY.log" 2>&1
done

# Outcome months: September and October 2020 (Business), July and August 2020 (Education).
estimate() {
  "$PY" "$REPO/real_data/estimate.py" \
    --model-path "$WORK/$1/trained_model.pt" \
    --data-dir "$DATA_ROOT/$1" \
    --output-dir "$WORK/$1/estimates" \
    --end-period "$2" --outcome-aggregation "$3" \
    --top-k 8 --n-bootstrap 1000 --k-hops 1 --n-partitions 3 --device cpu \
    --save-draws > "$WORK/logs/estimate_$1_$2_$3.log" 2>&1
}
for AGG in last_week final_month_average; do
  estimate Business_Economic_Restrictions 2020-09 "$AGG"
  estimate Business_Economic_Restrictions 2020-10 "$AGG"
  estimate Education_Childcare 2020-07 "$AGG"
  estimate Education_Childcare 2020-08 "$AGG"
done

"$PY" "$REPO/real_data/compact_seed.py" "$WORK" "$SEED" "$FINAL"
[ "${KEEP_WORK:-0}" = 1 ] || rm -rf "$WORK"
echo "$TAG done -> $FINAL"
