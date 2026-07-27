#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m unittest \
    tests.test_seclora \
    tests.test_seclora_native

"${PYTHON_BIN}" main.py --config configs/seclora_smoke.yaml

echo "SecLoRA unit, native-crypto, and end-to-end smoke tests passed."
