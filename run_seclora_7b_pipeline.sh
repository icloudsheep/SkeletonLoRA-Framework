#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_ENV="skeleton_lora_fe"
PIPELINE_ID="${PIPELINE_ID:-$(date +%Y%m%d_%H%M%S)}"

if [[ -n "${SECLORA_PIPELINE_DIR:-}" ]]; then
    if [[ "$SECLORA_PIPELINE_DIR" = /* ]]; then
        PIPELINE_DIR="$SECLORA_PIPELINE_DIR"
    else
        PIPELINE_DIR="$ROOT/$SECLORA_PIPELINE_DIR"
    fi
else
    PIPELINE_DIR="$ROOT/output/seclora_7b_pipeline_$PIPELINE_ID"
fi

RUN_MAP="$PIPELINE_DIR/run_ids.tsv"
PIPELINE_LOG="$PIPELINE_DIR/pipeline.log"

SEL0125_CONFIG="configs/seclora_7b_sel2s_000125.yaml"
SEL1_CONFIG="configs/seclora_7b_sel2s_001.yaml"
SEL10_CONFIG="configs/seclora_7b_sel2s_010.yaml"
SEL25_CONFIG="configs/seclora_7b_sel2s_025.yaml"
MODERN_CONFIG="configs/seclora_loss_7b_modern_001.yaml"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
    echo "[7b-pipeline] activate the expected environment first:" >&2
    echo "  conda activate $EXPECTED_ENV" >&2
    exit 1
fi

for config in \
    "$SEL0125_CONFIG" "$SEL1_CONFIG" "$SEL10_CONFIG" \
    "$SEL25_CONFIG" "$MODERN_CONFIG"; do
    if [[ ! -f "$ROOT/$config" ]]; then
        echo "[7b-pipeline] missing config: $config" >&2
        exit 1
    fi
done

if [[ ! -d "$ROOT/models/open_llama_7b_v2" ]]; then
    echo "[7b-pipeline] missing model: models/open_llama_7b_v2" >&2
    exit 1
fi
if [[ ! -f "$ROOT/datasets/mmlu/mmlu_auxiliary_train.jsonl" ]]; then
    echo "[7b-pipeline] missing MMLU training data" >&2
    exit 1
fi
if [[ ! -d "$ROOT/evaluation/mmlu" ]]; then
    echo "[7b-pipeline] missing MMLU evaluation data" >&2
    exit 1
fi
if [[ ! -d "$ROOT/datasets/natural-instructions" ]]; then
    echo "[7b-pipeline] missing Natural Instructions data" >&2
    exit 1
fi

NATIVE_EXTENSION="$(
    find "$ROOT/seclora/native" -maxdepth 1 -type f \
        -name '_seclora_native*.so' -print -quit
)"
if [[ -z "$NATIVE_EXTENSION" ]]; then
    echo "[7b-pipeline] SecLoRA native extension is missing" >&2
    echo "  bash seclora/setup_autodl.sh" >&2
    exit 1
fi
if ! "$CONDA_PREFIX/bin/python" -c 'import matplotlib' >/dev/null 2>&1; then
    echo "[7b-pipeline] matplotlib is required before the long run" >&2
    echo "  python -m pip install 'matplotlib>=3.8,<4'" >&2
    exit 1
fi

mkdir -p "$PIPELINE_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

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

train_or_reuse() {
    local label="$1"
    local run_name="$2"
    local config="$3"
    local run_id

    run_id="$(latest_completed_run "$run_name" "$config" || true)"
    if [[ -n "$run_id" ]]; then
        echo "[7b-pipeline] TRAIN REUSE label=$label run_id=$run_id" >&2
        printf '%s\n' "$run_id"
        return 0
    fi

    echo "[7b-pipeline] TRAIN START label=$label config=$config" >&2
    RUN_NAME="$run_name" bash "$ROOT/run.sh" "$ROOT/$config" >&2
    run_id="$(latest_completed_run "$run_name" "$config" || true)"
    if [[ -z "$run_id" ]]; then
        echo "[7b-pipeline] failed to locate completed run for $label" >&2
        exit 1
    fi
    echo "[7b-pipeline] TRAIN DONE label=$label run_id=$run_id" >&2
    printf '%s\n' "$run_id"
}

evaluate_or_reuse() {
    local label="$1"
    local config="$2"
    local run_id="$3"
    local result="$ROOT/output/$run_id/metrics/mmlu.csv"

    if [[ -s "$result" ]]; then
        echo "[7b-pipeline] MMLU REUSE label=$label run_id=$run_id"
        return 0
    fi
    echo "[7b-pipeline] MMLU START label=$label run_id=$run_id"
    bash "$ROOT/evaluate.sh" \
        "$ROOT/$config" "$run_id" mmlu adapter \
        "$ROOT/configs/evaluation.yaml"
    if [[ ! -s "$result" ]]; then
        echo "[7b-pipeline] MMLU output missing for $run_id" >&2
        exit 1
    fi
    echo "[7b-pipeline] MMLU DONE label=$label run_id=$run_id"
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

run_end_to_end() {
    local label="$1"
    local run_name="$2"
    local config="$3"
    local run_id

    run_id="$(train_or_reuse "$label" "$run_name" "$config")"
    record_run "$label" "$config" "$run_id"
    evaluate_or_reuse "$label" "$config" "$run_id"
}

echo "[7b-pipeline] pipeline=$PIPELINE_DIR"
echo "[7b-pipeline] model=OpenLLaMA-v2 7B threads=25"
if [[ ! -f "$RUN_MAP" ]]; then
    printf 'label\tconfig\trun_id\n' > "$RUN_MAP"
fi

echo
echo "[7b-pipeline] phase 1/3: 0.125% and 1% training plus MMLU"
run_end_to_end \
    "sel2s-0.125pct" "seclora-7b-sel2s-0.125pct" "$SEL0125_CONFIG"
run_end_to_end \
    "sel2s-1pct" "seclora-7b-sel2s-1pct" "$SEL1_CONFIG"

echo
echo "[7b-pipeline] phase 2/3: 1% Modern Loss and rolling-loss plot"
MODERN_RUN_ID="$(train_or_reuse \
    "modern-loss-1pct" "seclora-7b-modern-loss-1pct" "$MODERN_CONFIG")"
record_run "modern-loss-1pct" "$MODERN_CONFIG" "$MODERN_RUN_ID"
"$CONDA_PREFIX/bin/python" "$ROOT/plot_seclora_loss.py" \
    --series "SecLoRA SEL-2S 1%=$ROOT/output/$MODERN_RUN_ID" \
    --output "$PIPELINE_DIR/seclora_7b_modern_loss.pdf"

echo
echo "[7b-pipeline] phase 3/3: 10% and 25% training plus MMLU"
run_end_to_end \
    "sel2s-10pct" "seclora-7b-sel2s-10pct" "$SEL10_CONFIG"
run_end_to_end \
    "sel2s-25pct" "seclora-7b-sel2s-25pct" "$SEL25_CONFIG"

echo
echo "[7b-pipeline] ALL DONE"
echo "[7b-pipeline] run map: $RUN_MAP"
echo "[7b-pipeline] log: $PIPELINE_LOG"
echo "[7b-pipeline] loss plot: $PIPELINE_DIR/seclora_7b_modern_loss.pdf"
