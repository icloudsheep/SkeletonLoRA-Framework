#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$ROOT/configs/default.yaml}"
RUN_ID="${2:-}"
TARGET="${3:-train}"
FOURTH_ARG="${4:-}"
FIFTH_ARG="${5:-}"
OUTPUT_TAG="${6:-}"
EXPECTED_ENV="skeleton_lora_fe"

if [[ "$FOURTH_ARG" == "adapter" || "$FOURTH_ARG" == "base" ]]; then
    MODEL_MODE="$FOURTH_ARG"
    EVALUATION_CONFIG="${FIFTH_ARG:-$ROOT/configs/evaluation.yaml}"
else
    EVALUATION_CONFIG="${FOURTH_ARG:-$ROOT/configs/evaluation.yaml}"
    MODEL_MODE="${FIFTH_ARG:-adapter}"
fi

if [[ "$MODEL_MODE" != "adapter" && "$MODEL_MODE" != "base" ]]; then
    echo "[evaluate.sh] 错误：model-mode 必须是 adapter 或 base，收到 $MODEL_MODE"
    exit 1
fi

if [[ -z "$RUN_ID" ]]; then
    echo "[evaluate.sh] 错误：run-id 不能为空"
    echo "[evaluate.sh] 用法：bash evaluate.sh [config] <run-id> [train|mmlu|gsm8k] [adapter|base] [evaluation-config] [output-tag]"
    echo "[evaluate.sh] 示例：bash evaluate.sh configs/ckks.yaml 2026-07-27_19-31-58 mmlu"
    echo "[evaluate.sh] 原生：bash evaluate.sh configs/ckks.yaml 2026-07-27_19-31-58 mmlu base"
    exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "[evaluate.sh] 错误：当前没有激活 Conda 环境"
    echo "[evaluate.sh] 请先执行：conda activate $EXPECTED_ENV"
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
    echo "[evaluate.sh] 错误：当前环境是 ${CONDA_DEFAULT_ENV:-unknown}"
    echo "[evaluate.sh] 请先执行：conda activate $EXPECTED_ENV"
    exit 1
fi

PYTHON="$CONDA_PREFIX/bin/python"
LIBSTDCPP="$CONDA_PREFIX/lib/libstdc++.so.6"

if [[ ! -x "$PYTHON" ]]; then
    echo "[evaluate.sh] 错误：找不到 Python：$PYTHON"
    exit 1
fi

if [[ ! -e "$LIBSTDCPP" ]]; then
    echo "[evaluate.sh] 错误：找不到 $LIBSTDCPP"
    exit 1
fi

cd "$ROOT"

export HF_HOME="$ROOT/hf-cache"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$LIBSTDCPP${LD_PRELOAD:+:$LD_PRELOAD}"

mkdir -p "$HF_HOME"

echo "[evaluate.sh] conda env=$CONDA_DEFAULT_ENV"
echo "[evaluate.sh] python=$PYTHON"
echo "[evaluate.sh] config=$CONFIG"
echo "[evaluate.sh] run_id=$RUN_ID"
echo "[evaluate.sh] target=$TARGET"
echo "[evaluate.sh] model_mode=$MODEL_MODE"
echo "[evaluate.sh] evaluation_config=$EVALUATION_CONFIG"
echo "[evaluate.sh] output_tag=${OUTPUT_TAG:-none}"

OUTPUT_ARGS=()
if [[ -n "$OUTPUT_TAG" ]]; then
    OUTPUT_ARGS=(--output-tag "$OUTPUT_TAG")
fi

"$PYTHON" "$ROOT/evaluate.py" \
    --config "$CONFIG" \
    --run-id "$RUN_ID" \
    --target "$TARGET" \
    --model-mode "$MODEL_MODE" \
    --evaluation-config "$EVALUATION_CONFIG" \
    "${OUTPUT_ARGS[@]}"
