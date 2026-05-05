#!/usr/bin/env bash
# Evaluate one or more phased fine-tune checkpoints on an external TSPLIB
# directory (default: /root/autodl-tmp/tsplib-master). Bias config is
# auto-attached from the ckpt's embedded ``bias_cfg`` field, so no extra
# flags are required even when the trained model used kNN bias.
#
# Outputs:
#   - per-run JSON: <run_dir>/external_test_<ckpt_name>.json
#   - aggregated CSV: $SUMMARY_FILE  (default: results/external_test_summary.csv)
#
# Usage:
#   bash scripts/test_on_external.sh                          # all phased winners + baseline, default tsplib-master
#   bash scripts/test_on_external.sh result/<dir1> result/<dir2>
#   DATA_PATH=/root/autodl-tmp/tsplib-master \
#     SCALE_MIN=0 SCALE_MAX=500 \
#     TARGET_CKPT=checkpoint-phase_3_leader_best.pt \
#     bash scripts/test_on_external.sh
#
# Notes:
#   - SCALE_MAX defaults to 1000 because POMO trained on TSP100 generalizes
#     poorly past ~500-1000 nodes, and very large instances (d18512, brd14051,
#     etc.) will OOM on a 24GB GPU with aug_factor=8.
#   - Baseline checkpoint is always evaluated first as a reference row.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

DATA_PATH="${DATA_PATH:-/root/autodl-tmp/tsplib-master}"
SCALE_MIN="${SCALE_MIN:-0}"
SCALE_MAX="${SCALE_MAX:-1000}"
TARGET_CKPT="${TARGET_CKPT:-checkpoint-phase_3_leader_best.pt}"
SUMMARY_FILE="${SUMMARY_FILE:-./results/external_test_summary.csv}"
SKIP_BASELINE="${SKIP_BASELINE:-false}"

if [[ ! -d "$DATA_PATH" ]]; then
    echo "ERROR: DATA_PATH not found: $DATA_PATH" >&2
    exit 2
fi

# Collect run dirs: explicit args, else auto-discover phased runs.
if [[ $# -gt 0 ]]; then
    RUN_DIRS=("$@")
else
    mapfile -t RUN_DIRS < <(find ./result -maxdepth 1 -type d -name "20*_phased_*" 2>/dev/null | sort)
fi

mkdir -p "$(dirname "$SUMMARY_FILE")"

echo "=================================================="
echo "[external_test] data_path : $DATA_PATH"
echo "[external_test] scale     : [$SCALE_MIN, $SCALE_MAX)"
echo "[external_test] target    : $TARGET_CKPT"
echo "[external_test] runs      : ${#RUN_DIRS[@]}"
echo "[external_test] summary   : $SUMMARY_FILE"
echo "=================================================="

# Reset summary CSV.
echo "run_name,ckpt,solved,total,avg_no_aug_gap,avg_aug_gap" > "$SUMMARY_FILE"

# Helper: append a row from a result JSON to the summary CSV.
_append_row() {
    local out_json="$1" run_name="$2" ckpt_name="$3"
    if [[ ! -f "$out_json" ]]; then
        echo "$run_name,$ckpt_name,NA,NA,NA,NA" >> "$SUMMARY_FILE"
        return
    fi
    python - "$out_json" "$SUMMARY_FILE" "$run_name" "$ckpt_name" <<'PY'
import json, sys
out_json, summary, run_name, ckpt_name = sys.argv[1:]
d = json.load(open(out_json))
def _f(v): return f"{v:.4f}" if isinstance(v, (int, float)) else "NA"
row = (
    f"{run_name},{ckpt_name},"
    f"{d.get('solved_instance_num', 0)},{d.get('total_instance_num', 0)},"
    f"{_f(d.get('avg_no_aug_gap'))},{_f(d.get('avg_aug_gap'))}\n"
)
with open(summary, "a") as f:
    f.write(row)
PY
}

# Helper: run test.py with the standard flags. The ckpt's embedded bias_cfg
# is auto-attached by TSPTester_LIB (no manual flag needed).
_run_eval() {
    local ckpt="$1" run_name="$2" out_json="$3"
    python test.py \
        --data_path "$DATA_PATH" \
        --checkpoint_path "$ckpt" \
        --augmentation_enable true \
        --aug_factor 8 \
        --detailed_log false \
        --scale_min "$SCALE_MIN" --scale_max "$SCALE_MAX" \
        --output_json "$out_json" \
        --run_name "$run_name"
}

# 1) Baseline reference (always first row, unless SKIP_BASELINE=true).
BASELINE_CKPT="./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt"
if [[ "$SKIP_BASELINE" != "true" && -f "$BASELINE_CKPT" ]]; then
    BASELINE_OUT="./results/external_test_baseline.json"
    echo ""
    echo "------------------------------------------"
    echo "[1/?] baseline (checkpoint-3000.pt)"
    echo "------------------------------------------"
    if _run_eval "$BASELINE_CKPT" "baseline_3000ep" "$BASELINE_OUT"; then
        _append_row "$BASELINE_OUT" "baseline_3000ep" "checkpoint-3000.pt"
    else
        echo "(baseline failed; continuing)"
        echo "baseline_3000ep,checkpoint-3000.pt,FAIL,FAIL,FAIL,FAIL" >> "$SUMMARY_FILE"
    fi
fi

# 2) Iterate run dirs.
TOTAL="${#RUN_DIRS[@]}"
i=0
for RAW_DIR in "${RUN_DIRS[@]}"; do
    i=$((i + 1))
    if [[ ! -d "$RAW_DIR" ]]; then
        echo "[skip] not a directory: $RAW_DIR"
        continue
    fi
    RUN_DIR="$(readlink -f "$RAW_DIR")"
    NAME="$(basename "$RUN_DIR")"
    CKPT="$RUN_DIR/$TARGET_CKPT"

    if [[ ! -f "$CKPT" ]]; then
        echo "[skip $i/$TOTAL] $NAME — no $TARGET_CKPT"
        continue
    fi

    OUT_JSON="$RUN_DIR/external_test_$(basename "$TARGET_CKPT" .pt).json"
    echo ""
    echo "------------------------------------------"
    echo "[$i/$TOTAL] $NAME"
    echo "------------------------------------------"
    if _run_eval "$CKPT" "$NAME" "$OUT_JSON"; then
        _append_row "$OUT_JSON" "$NAME" "$TARGET_CKPT"
    else
        echo "(failed; continuing)"
        echo "$NAME,$TARGET_CKPT,FAIL,FAIL,FAIL,FAIL" >> "$SUMMARY_FILE"
    fi
done

echo ""
echo "=================================================="
echo "Summary ($SUMMARY_FILE):"
echo "=================================================="
column -t -s, "$SUMMARY_FILE"
