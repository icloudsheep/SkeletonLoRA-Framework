#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_ENV="skeleton_lora_fe"
PIPELINE_ID="${PIPELINE_ID:-$(date +%Y%m%d_%H%M%S)}"
PIPELINE_DIR="$ROOT/output/seclora_3b_pipeline_$PIPELINE_ID"
RUN_MAP="$PIPELINE_DIR/run_ids.tsv"
PIPELINE_LOG="$PIPELINE_DIR/pipeline.log"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
    echo "[pipeline] activate the expected environment first:" >&2
    echo "  conda activate $EXPECTED_ENV" >&2
    exit 1
fi

mkdir -p "$PIPELINE_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

TRAIN_LABELS=(
    "sel2s-0.125pct"
    "sel2s-1pct"
    "sel2s-10pct"
    "sel2s-25pct"
    "full-sk"
)
TRAIN_CONFIGS=(
    "configs/seclora_3b_sel2s_000125.yaml"
    "configs/seclora_3b_sel2s_001.yaml"
    "configs/seclora_3b_sel2s_010.yaml"
    "configs/seclora_3b_sel2s_025.yaml"
    "configs/seclora_3b_full_sk.yaml"
)
MODERN_LOSS_CONFIG="configs/seclora_loss_3b_modern_001.yaml"

for config in "${TRAIN_CONFIGS[@]}" "$MODERN_LOSS_CONFIG"; do
    if [[ ! -f "$ROOT/$config" ]]; then
        echo "[pipeline] missing config: $config" >&2
        exit 1
    fi
done

if [[ ! -d "$ROOT/models/open_llama_3b_v2" ]]; then
    echo "[pipeline] missing model: models/open_llama_3b_v2" >&2
    exit 1
fi
if [[ ! -f "$ROOT/datasets/mmlu/mmlu_auxiliary_train.jsonl" ]]; then
    echo "[pipeline] missing MMLU training data" >&2
    exit 1
fi
if [[ ! -d "$ROOT/evaluation/mmlu" ]]; then
    echo "[pipeline] missing MMLU evaluation data" >&2
    exit 1
fi
if [[ ! -d "$ROOT/datasets/natural-instructions" ]]; then
    echo "[pipeline] missing Natural Instructions data" >&2
    exit 1
fi

latest_run_id() {
    local run_name="$1"
    find "$ROOT/output" -mindepth 1 -maxdepth 1 -type d \
        -name "${run_name}_*" -printf '%f\n' | sort | tail -n 1
}

train_one() {
    local label="$1"
    local config="$2"
    local run_name="seclora-${label}"
    local run_id

    echo
    echo "[pipeline] TRAIN START label=$label config=$config"
    RUN_NAME="$run_name" bash "$ROOT/run.sh" "$ROOT/$config"
    run_id="$(latest_run_id "$run_name")"
    if [[ -z "$run_id" || \
          ! -f "$ROOT/output/$run_id/checkpoints/final/adapter_model.safetensors" ]]; then
        echo "[pipeline] failed to locate completed run for $label" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\n' "$label" "$config" "$run_id" >> "$RUN_MAP"
    echo "[pipeline] TRAIN DONE label=$label run_id=$run_id"
}

evaluate_one() {
    local label="$1"
    local config="$2"
    local run_id="$3"

    echo
    echo "[pipeline] MMLU START label=$label run_id=$run_id"
    bash "$ROOT/evaluate.sh" \
        "$ROOT/$config" "$run_id" mmlu adapter \
        "$ROOT/configs/evaluation.yaml"
    if [[ ! -f "$ROOT/output/$run_id/metrics/mmlu.csv" ]]; then
        echo "[pipeline] MMLU output missing for $run_id" >&2
        exit 1
    fi
    echo "[pipeline] MMLU DONE label=$label run_id=$run_id"
}

echo "[pipeline] id=$PIPELINE_ID"
echo "[pipeline] output=$PIPELINE_DIR"
echo "[pipeline] phase 1/3: five independent end-to-end training runs"
printf 'label\tconfig\trun_id\n' > "$RUN_MAP"
for index in "${!TRAIN_CONFIGS[@]}"; do
    train_one "${TRAIN_LABELS[$index]}" "${TRAIN_CONFIGS[$index]}"
done

echo
echo "[pipeline] phase 2/3: MMLU adapter evaluation for all five runs"
while IFS=$'\t' read -r label config run_id; do
    [[ "$label" == "label" ]] && continue
    evaluate_one "$label" "$config" "$run_id"
done < "$RUN_MAP"

echo
echo "[pipeline] phase 3/3: 100-round modern loss run"
MODERN_LABEL="modern-loss-1pct"
MODERN_RUN_NAME="seclora-${MODERN_LABEL}"
RUN_NAME="$MODERN_RUN_NAME" \
    bash "$ROOT/run.sh" "$ROOT/$MODERN_LOSS_CONFIG"
MODERN_RUN_ID="$(latest_run_id "$MODERN_RUN_NAME")"
if [[ -z "$MODERN_RUN_ID" || \
      ! -f "$ROOT/output/$MODERN_RUN_ID/metrics/step.csv" ]]; then
    echo "[pipeline] failed to locate completed modern-loss run" >&2
    exit 1
fi
printf '%s\t%s\t%s\n' \
    "$MODERN_LABEL" "$MODERN_LOSS_CONFIG" "$MODERN_RUN_ID" >> "$RUN_MAP"

"$CONDA_PREFIX/bin/python" "$ROOT/plot_seclora_loss.py" \
    --series "SecLoRA SEL-2S 1%=$ROOT/output/$MODERN_RUN_ID" \
    --output "$PIPELINE_DIR/seclora_3b_modern_loss.pdf"

echo
echo "[pipeline] ALL DONE"
echo "[pipeline] run map: $RUN_MAP"
echo "[pipeline] log: $PIPELINE_LOG"
echo "[pipeline] modern loss run: $MODERN_RUN_ID"
echo "[pipeline] loss plot: $PIPELINE_DIR/seclora_3b_modern_loss.pdf"
