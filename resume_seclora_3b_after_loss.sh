#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_ENV="skeleton_lora_fe"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
    echo "[resume] activate the expected environment first:" >&2
    echo "  conda activate $EXPECTED_ENV" >&2
    exit 1
fi

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
    echo "[resume] pipeline directory not found" >&2
    echo "[resume] set SECLORA_PIPELINE_DIR=output/seclora_3b_pipeline_<timestamp>" >&2
    exit 1
fi

RUN_MAP="$PIPELINE_DIR/run_ids.tsv"
if [[ -z "${MODERN_RUN_ID:-}" && -f "$RUN_MAP" ]]; then
    MODERN_RUN_ID="$(
        awk -F '\t' '$1 == "modern-loss-1pct" { value = $3 } END { print value }' \
            "$RUN_MAP"
    )"
fi
if [[ -z "${MODERN_RUN_ID:-}" ]]; then
    MODERN_RUN_ID="$(
        find "$ROOT/output" -mindepth 1 -maxdepth 1 -type d \
            -name 'seclora-stage2-modern-loss-1pct_*' -printf '%f\n' | \
            sort | tail -n 1
    )"
fi

RUN_DIR="$ROOT/output/${MODERN_RUN_ID:-}"
STEP_CSV="$RUN_DIR/metrics/step.csv"
CLIENT_ROUND_CSV="$RUN_DIR/metrics/client_round.csv"
FINAL_ADAPTER="$RUN_DIR/checkpoints/final/adapter_model.safetensors"

if [[ -z "${MODERN_RUN_ID:-}" || ! -s "$STEP_CSV" || \
      ! -s "$CLIENT_ROUND_CSV" || ! -f "$FINAL_ADAPTER" ]]; then
    echo "[resume] completed Modern Loss run not found: ${MODERN_RUN_ID:-<unset>}" >&2
    exit 1
fi

MAX_ROUND="$(awk -F ',' 'NR > 1 && $1 + 0 > max { max = $1 + 0 } END { print max + 0 }' \
    "$CLIENT_ROUND_CSV")"
if [[ "$MAX_ROUND" -ne 100 ]]; then
    echo "[resume] Modern Loss run is incomplete: max_round=$MAX_ROUND" >&2
    exit 1
fi

if ! "$CONDA_PREFIX/bin/python" -c 'import matplotlib' >/dev/null 2>&1; then
    echo "[resume] matplotlib is missing; install it before resuming:" >&2
    echo "  python -m pip install 'matplotlib>=3.8,<4'" >&2
    exit 1
fi

echo "[resume] pipeline=$PIPELINE_DIR"
echo "[resume] verified completed Modern Loss run=$MODERN_RUN_ID (100 rounds)"
echo "[resume] stage2 will reuse completed 0.125%, 1%, and Modern Loss runs"
echo "[resume] continuing with plot generation, then 10% and 25% training/MMLU"

SECLORA_PIPELINE_DIR="$PIPELINE_DIR" \
MODERN_RUN_ID="$MODERN_RUN_ID" \
    bash "$ROOT/run_seclora_3b_stage2.sh"
