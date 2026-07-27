#!/usr/bin/env bash
set -euo pipefail

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCL_DIR="${SECLORA_MCL_DIR:-${NATIVE_DIR}/third_party/mcl}"
MCL_COMMIT="7af8ea79d240d24f24c8fb049c0bcd74464d677b"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ "${1:-}" == "clean" ]]; then
    rm -rf "${NATIVE_DIR}/build"
    rm -f "${NATIVE_DIR}"/_seclora_native*.so
fi

for command_name in git cmake c++; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing native build command: ${command_name}" >&2
        echo "Run: bash seclora/setup_autodl.sh" >&2
        exit 1
    fi
done

if ! "${PYTHON_BIN}" -m pybind11 --cmakedir >/dev/null 2>&1; then
    echo "pybind11 is required. Install it with:" >&2
    echo "  ${PYTHON_BIN} -m pip install pybind11" >&2
    exit 1
fi

if [[ ! -d "${MCL_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${MCL_DIR}")"
    git clone https://github.com/herumi/mcl.git "${MCL_DIR}"
fi
if ! git -C "${MCL_DIR}" cat-file -e "${MCL_COMMIT}^{commit}" 2>/dev/null; then
    git -C "${MCL_DIR}" fetch origin "${MCL_COMMIT}"
fi
git -C "${MCL_DIR}" checkout --detach "${MCL_COMMIT}"

cmake \
    -S "${NATIVE_DIR}" \
    -B "${NATIVE_DIR}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')" \
    -DSECLORA_MCL_DIR="${MCL_DIR}"
cmake --build "${NATIVE_DIR}/build" --parallel "$(nproc 2>/dev/null || echo 4)"

cd "${NATIVE_DIR}/../.."
"${PYTHON_BIN}" -c \
    'from seclora.native import _seclora_native; print("SecLoRA native import: OK")'

echo "Built SecLoRA native module in ${NATIVE_DIR}"
