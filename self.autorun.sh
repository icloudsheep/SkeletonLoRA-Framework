#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIGS=(
    ckks-AB-10-Skeleton.yaml
    ckks-AB-10.yaml
    ckks-AB-25-Skeleton.yaml
    ckks-AB-25.yaml
)

for config in "${CONFIGS[@]}"; do
    echo "[autorun] running $config"
    bash "$ROOT/run.sh" "$ROOT/configs/$config"
done

echo "[autorun] all done, shutting down"
shutdown -h now
