#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${ROOT_DIR}"

"${ROOT_DIR}/seclora/native/build/seclora_session_selftest"

"${PYTHON_BIN}" -m unittest \
    tests.test_seclora \
    tests.test_seclora_native \
    tests.test_training_metrics

"${PYTHON_BIN}" main.py --config configs/seclora_smoke.yaml

echo "SecLoRA unit, native-crypto, and end-to-end smoke tests passed."
