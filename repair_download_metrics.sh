#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-}"
RUN_ID="${2:-}"
EXPECTED_ENV="skeleton_lora_fe"

if [[ -z "$CONFIG" || -z "$RUN_ID" ]]; then
    echo "[repair-download] 用法：bash repair_download_metrics.sh <config> <run-id> [--exact]"
    exit 1
fi
shift 2

if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" || -z "${CONDA_PREFIX:-}" ]]; then
    echo "[repair-download] 请先执行：conda activate $EXPECTED_ENV"
    exit 1
fi

PYTHON="$CONDA_PREFIX/bin/python"
LIBSTDCPP="$CONDA_PREFIX/lib/libstdc++.so.6"
if [[ ! -x "$PYTHON" || ! -e "$LIBSTDCPP" ]]; then
    echo "[repair-download] Conda 环境缺少 Python 或 libstdc++.so.6"
    exit 1
fi

cd "$ROOT"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$LIBSTDCPP${LD_PRELOAD:+:$LD_PRELOAD}"

"$PYTHON" "$ROOT/repair_download_metrics.py" \
    --config "$CONFIG" \
    --run-id "$RUN_ID" \
    "$@"
