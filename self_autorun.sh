#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIGS=(
    loss-AB-1-Skeleton-NI.yaml
    loss-AB-1-NI.yaml
#    ckks-AB-1.yaml
#    ckks-AB-10-Skeleton.yaml
#    ckks-AB-10.yaml
#    ckks-AB-25-Skeleton.yaml
#    ckks-AB-25.yaml
)

for config in "${CONFIGS[@]}"; do
    echo "[autorun] running $config"
    bash "$ROOT/run.sh" "$ROOT/configs/$config"
done

#bash evaluate.sh configs/ckks-AB-1-Skeleton.yaml "3B-AB-01-SKE-MMLU" mmlu
#bash evaluate.sh configs/ckks-AB-1.yaml "3B-AB-01-MMLU" mmlu
#bash evaluate.sh configs/ckks-AB-10-Skeleton.yaml "3B-AB-10-SKE-MMLU" mmlu
#bash evaluate.sh configs/ckks-AB-10.yaml "3B-AB-10-MMLU" mmlu
#bash evaluate.sh configs/ckks-AB-25-Skeleton.yaml "3B-AB-25-SKE-MMLU" mmlu
#bash evaluate.sh configs/ckks-AB-25.yaml "3B-AB-25-MMLU" mmlu

echo "[autorun] all done, shutting down"
#/usr/bin/shutdown -h now
