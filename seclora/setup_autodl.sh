#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

missing=0
for command_name in git cmake c++; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing system command: ${command_name}" >&2
        missing=1
    fi
done
if [[ "${missing}" -ne 0 ]]; then
    cat >&2 <<'EOF'
Install native build tools first:
  apt-get update
  apt-get install -y build-essential cmake git libgmp-dev

Without root access, use:
  conda install -y -c conda-forge cmake ninja cxx-compiler gmp
EOF
    exit 1
fi

if ! printf '#include <gmp.h>\n' | c++ -x c++ -E - >/dev/null 2>&1; then
    cat >&2 <<'EOF'
GMP development headers are missing. Install one of:
  apt-get install -y libgmp-dev
  conda install -y -c conda-forge gmp
EOF
    exit 1
fi

"${PYTHON_BIN}" -c \
    'import torch, numpy, yaml, peft; print("framework environment: OK")'
"${PYTHON_BIN}" -m pip install \
    -r "${ROOT_DIR}/seclora/requirements-native.txt"

PYTHON_BIN="${PYTHON_BIN}" bash "${ROOT_DIR}/seclora/native/build.sh"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -c \
    'from seclora.native import _seclora_native; print("SecLoRA native import: OK")'

echo "AutoDL SecLoRA environment is ready."
