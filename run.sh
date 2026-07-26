#!/usr/bin/env bash
# 快速启动脚本：
# 1. 检查 Conda 环境
# 2. 强制优先使用 Conda 环境中的 C++ 运行库
# 3. 后台启动 TensorBoard
# 4. 执行训练
# 5. 脚本退出时关闭 TensorBoard

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$ROOT/configs/default.yaml}"
PORT="${TB_PORT:-6006}"
EXPECTED_ENV="skeleton_lora_fe"

# 检查是否已经激活正确的 Conda 环境
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "[run.sh] 错误：当前没有激活 Conda 环境"
    echo "[run.sh] 请先执行：conda activate $EXPECTED_ENV"
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
    echo "[run.sh] 错误：当前环境是 ${CONDA_DEFAULT_ENV:-unknown}"
    echo "[run.sh] 请先执行：conda activate $EXPECTED_ENV"
    exit 1
fi

PYTHON="$CONDA_PREFIX/bin/python"
TENSORBOARD="$CONDA_PREFIX/bin/tensorboard"
LIBSTDCPP="$CONDA_PREFIX/lib/libstdc++.so.6"

# 检查程序是否来自当前 Conda 环境
if [[ ! -x "$PYTHON" ]]; then
    echo "[run.sh] 错误：找不到 Python：$PYTHON"
    exit 1
fi

if [[ ! -x "$TENSORBOARD" ]]; then
    echo "[run.sh] 错误：找不到 TensorBoard：$TENSORBOARD"
    echo "[run.sh] 可执行：conda install tensorboard"
    exit 1
fi

# 检查 Conda 环境中的 libstdc++
if [[ ! -e "$LIBSTDCPP" ]]; then
    echo "[run.sh] 错误：找不到 $LIBSTDCPP"
    echo "[run.sh] 请执行："
    echo "conda install -c conda-forge 'libstdcxx-ng>=13' 'libgcc-ng>=13' -y"
    exit 1
fi

if ! grep -aFq "GLIBCXX_3.4.31" "$LIBSTDCPP"; then
    echo "[run.sh] 错误：Conda 环境中的 libstdc++ 版本仍然过低"
    echo "[run.sh] 请执行："
    echo "conda install -c conda-forge 'libstdcxx-ng>=13' 'libgcc-ng>=13' -y"
    exit 1
fi

cd "$ROOT"

export HF_HOME="$ROOT/hf-cache"

# 仅在当前脚本及其子进程中优先使用 Conda 动态库
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 强制加载 Conda 环境中的新版 libstdc++
export LD_PRELOAD="$LIBSTDCPP${LD_PRELOAD:+:$LD_PRELOAD}"

TB_LOGDIR="$ROOT/output"
mkdir -p "$TB_LOGDIR"
mkdir -p "$HF_HOME"

echo "[run.sh] conda env=$CONDA_DEFAULT_ENV"
echo "[run.sh] python=$PYTHON"
echo "[run.sh] tensorboard=$TENSORBOARD"
echo "[run.sh] libstdc++=$LIBSTDCPP"
echo "[run.sh] config=$CONFIG"

# 启动 TensorBoard
"$TENSORBOARD" \
    --logdir "$TB_LOGDIR" \
    --port "$PORT" \
    >/dev/null 2>&1 &

TB_PID=$!

echo "[run.sh] tensorboard pid=$TB_PID port=$PORT logdir=$TB_LOGDIR"

cleanup() {
    if [[ -n "${TB_PID:-}" ]] && kill -0 "$TB_PID" 2>/dev/null; then
        kill "$TB_PID" 2>/dev/null || true
        wait "$TB_PID" 2>/dev/null || true
        echo "[run.sh] 已杀掉 tensorboard pid=$TB_PID"
    fi
}

trap cleanup EXIT INT TERM

# 启动训练
"$PYTHON" "$ROOT/main.py" --config "$CONFIG"