#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_ENV="skeleton_lora_fe"

if [[ -n "${SECLORA_PIPELINE_DIR:-}" ]]; then
    if [[ "$SECLORA_PIPELINE_DIR" = /* ]]; then
        PIPELINE_DIR="$SECLORA_PIPELINE_DIR"
    else
        PIPELINE_DIR="$ROOT/$SECLORA_PIPELINE_DIR"
    fi
else
    PIPELINE_DIR="$(
        find "$ROOT/output" -mindepth 1 -maxdepth 1 -type d \
            -name 'seclora_3b_pipeline_*' -print | sort | tail -n 1
    )"
fi

if [[ -z "$PIPELINE_DIR" || ! -d "$PIPELINE_DIR" ]]; then
    echo "[stage2] existing seclora_3b_pipeline directory not found" >&2
    echo "[stage2] set it explicitly, for example:" >&2
    echo "  SECLORA_PIPELINE_DIR=output/seclora_3b_pipeline_<timestamp> ..." >&2
    exit 1
fi

RUN_MAP="$PIPELINE_DIR/run_ids.tsv"
PIPELINE_LOG="$PIPELINE_DIR/pipeline.log"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
    echo "[stage2] activate the expected environment first:" >&2
    echo "  conda activate $EXPECTED_ENV" >&2
    exit 1
fi

exec > >(tee -a "$PIPELINE_LOG") 2>&1

SEL0125_CONFIG="configs/seclora_3b_sel2s_000125.yaml"
SEL1_CONFIG="configs/seclora_3b_sel2s_001.yaml"
SEL10_CONFIG="configs/seclora_3b_sel2s_010.yaml"
SEL25_CONFIG="configs/seclora_3b_sel2s_025.yaml"
MODERN_CONFIG="configs/seclora_loss_3b_modern_001.yaml"

for config in \
    "$SEL0125_CONFIG" "$SEL1_CONFIG" "$SEL10_CONFIG" \
    "$SEL25_CONFIG" "$MODERN_CONFIG"; do
    if [[ ! -f "$ROOT/$config" ]]; then
        echo "[stage2] missing config: $config" >&2
        exit 1
    fi
done

if [[ ! -d "$ROOT/models/open_llama_3b_v2" ]]; then
    echo "[stage2] missing model: models/open_llama_3b_v2" >&2
    exit 1
fi
if [[ ! -d "$ROOT/evaluation/mmlu" ]]; then
    echo "[stage2] missing MMLU evaluation data" >&2
    exit 1
fi
if [[ ! -d "$ROOT/datasets/natural-instructions" ]]; then
    echo "[stage2] missing Natural Instructions data" >&2
    exit 1
fi

latest_completed_run() {
    local run_name="$1"
    local config="$2"
    local run_id

    while IFS= read -r run_id; do
        [[ -z "$run_id" ]] && continue
        if [[ -f "$ROOT/output/$run_id/checkpoints/final/adapter_model.safetensors" && \
              -f "$ROOT/output/$run_id/experiment.yaml" ]] && \
           cmp -s "$ROOT/output/$run_id/experiment.yaml" "$ROOT/$config"; then
            printf '%s\n' "$run_id"
            return 0
        fi
    done < <(
        find "$ROOT/output" -mindepth 1 -maxdepth 1 -type d \
            -name "${run_name}_*" -printf '%f\n' | sort -r
    )
    return 1
}

require_existing_run() {
    local label="$1"
    local run_name="$2"
    local config="$3"
    local override="$4"
    local run_id

    if [[ -n "$override" ]]; then
        run_id="$override"
    else
        run_id="$(latest_completed_run "$run_name" "$config" || true)"
    fi
    if [[ -z "$run_id" || \
          ! -f "$ROOT/output/$run_id/checkpoints/final/adapter_model.safetensors" ]]; then
        echo "[stage2] no completed $label run found" >&2
        echo "[stage2] set the explicit run id and retry, for example:" >&2
        echo "  SEL1_RUN_ID=<run-id> bash run_seclora_3b_stage2.sh" >&2
        exit 1
    fi
    printf '%s\n' "$run_id"
}

train_or_reuse() {
    local label="$1"
    local run_name="$2"
    local config="$3"
    local run_id

    run_id="$(latest_completed_run "$run_name" "$config" || true)"
    if [[ -n "$run_id" ]]; then
        echo "[stage2] TRAIN REUSE label=$label run_id=$run_id" >&2
        printf '%s\n' "$run_id"
        return 0
    fi

    echo "[stage2] TRAIN START label=$label config=$config" >&2
    RUN_NAME="$run_name" bash "$ROOT/run.sh" "$ROOT/$config" >&2
    run_id="$(latest_completed_run "$run_name" "$config" || true)"
    if [[ -z "$run_id" ]]; then
        echo "[stage2] failed to locate completed run for $label" >&2
        exit 1
    fi
    echo "[stage2] TRAIN DONE label=$label run_id=$run_id" >&2
    printf '%s\n' "$run_id"
}

evaluate_or_reuse() {
    local label="$1"
    local config="$2"
    local run_id="$3"
    local result="$ROOT/output/$run_id/metrics/mmlu.csv"

    if [[ -s "$result" ]]; then
        echo "[stage2] MMLU REUSE label=$label run_id=$run_id"
        return 0
    fi
    echo "[stage2] MMLU START label=$label run_id=$run_id"
    bash "$ROOT/evaluate.sh" \
        "$ROOT/$config" "$run_id" mmlu adapter \
        "$ROOT/configs/evaluation.yaml"
    if [[ ! -s "$result" ]]; then
        echo "[stage2] MMLU output missing for $run_id" >&2
        exit 1
    fi
    echo "[stage2] MMLU DONE label=$label run_id=$run_id"
}

record_run() {
    local label="$1"
    local config="$2"
    local run_id="$3"
    if awk -F '\t' -v label="$label" -v run_id="$run_id" \
        '$1 == label && $3 == run_id { found = 1 } END { exit !found }' \
        "$RUN_MAP"; then
        return 0
    fi
    printf '%s\t%s\t%s\n' "$label" "$config" "$run_id" >> "$RUN_MAP"
}

echo "[stage2] continuing pipeline=$PIPELINE_DIR"
if [[ ! -f "$RUN_MAP" ]]; then
    printf 'label\tconfig\trun_id\n' > "$RUN_MAP"
fi

SEL0125_RUN_ID="$(require_existing_run \
    "SEL-2S 0.125%" "seclora-sel2s-0.125pct" "$SEL0125_CONFIG" \
    "${SEL0125_RUN_ID:-}")"
SEL1_RUN_ID="$(require_existing_run \
    "SEL-2S 1%" "seclora-sel2s-1pct" "$SEL1_CONFIG" \
    "${SEL1_RUN_ID:-}")"
record_run "sel2s-0.125pct" "$SEL0125_CONFIG" "$SEL0125_RUN_ID"
record_run "sel2s-1pct" "$SEL1_CONFIG" "$SEL1_RUN_ID"

echo
echo "[stage2] phase 1/3: evaluate completed 0.125% and 1% runs"
evaluate_or_reuse "sel2s-0.125pct" "$SEL0125_CONFIG" "$SEL0125_RUN_ID"
evaluate_or_reuse "sel2s-1pct" "$SEL1_CONFIG" "$SEL1_RUN_ID"

echo
echo "[stage2] phase 2/3: run 1% modern loss and draw rolling loss"
if [[ -n "${MODERN_RUN_ID:-}" ]]; then
    MODERN_RUN_ID="$(require_existing_run \
        "modern-loss-1pct" "seclora-stage2-modern-loss-1pct" \
        "$MODERN_CONFIG" "$MODERN_RUN_ID")"
    echo "[stage2] TRAIN REUSE label=modern-loss-1pct run_id=$MODERN_RUN_ID"
else
    MODERN_RUN_ID="$(train_or_reuse \
        "modern-loss-1pct" "seclora-stage2-modern-loss-1pct" "$MODERN_CONFIG")"
fi
record_run "modern-loss-1pct" "$MODERN_CONFIG" "$MODERN_RUN_ID"
"$CONDA_PREFIX/bin/python" "$ROOT/plot_seclora_loss.py" \
    --series "SecLoRA SEL-2S 1%=$ROOT/output/$MODERN_RUN_ID" \
    --output "$PIPELINE_DIR/seclora_3b_modern_loss.pdf"

echo
echo "[stage2] phase 3/3: train and evaluate 10% and 25%"
SEL10_RUN_ID="$(train_or_reuse \
    "sel2s-10pct" "seclora-stage2-sel2s-10pct" "$SEL10_CONFIG")"
record_run "sel2s-10pct" "$SEL10_CONFIG" "$SEL10_RUN_ID"
evaluate_or_reuse "sel2s-10pct" "$SEL10_CONFIG" "$SEL10_RUN_ID"

SEL25_RUN_ID="$(train_or_reuse \
    "sel2s-25pct" "seclora-stage2-sel2s-25pct" "$SEL25_CONFIG")"
record_run "sel2s-25pct" "$SEL25_CONFIG" "$SEL25_RUN_ID"
evaluate_or_reuse "sel2s-25pct" "$SEL25_CONFIG" "$SEL25_RUN_ID"

echo
echo "[stage2] ALL DONE"
echo "[stage2] run map: $RUN_MAP"
echo "[stage2] log: $PIPELINE_LOG"
echo "[stage2] loss plot: $PIPELINE_DIR/seclora_3b_modern_loss.pdf"
