#!/usr/bin/env bash
# 快速启动脚本: 后台起 tensorboard,再跑训练。
# 脚本退出时通过 trap 杀掉 tensorboard 进程。
# evaluate.py 尚未实现,暂不接入。

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-$ROOT/configs/smoke.yaml}"
PORT="${TB_PORT:-6006}"

export HF_HOME="$ROOT/hf-cache"

TB_LOGDIR="$ROOT/output"

tensorboard --logdir "$TB_LOGDIR" --port "$PORT" >/dev/null 2>&1 &
TB_PID=$!
echo "[run.sh] tensorboard pid=$TB_PID port=$PORT logdir=$TB_LOGDIR"

cleanup() {
    if kill -0 "$TB_PID" 2>/dev/null; then
        kill "$TB_PID" 2>/dev/null || true
        echo "[run.sh] 已杀掉 tensorboard pid=$TB_PID"
    fi
}
trap cleanup EXIT INT TERM

python "$ROOT/main.py" --config "$CONFIG"
