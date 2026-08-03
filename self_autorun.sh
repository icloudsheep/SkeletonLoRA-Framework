#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIGS=(
    ckks-AB-0.yaml
    ckks-AB-0-Skeleton.yaml
    loss-AB-0-NI.yaml
    loss-AB-0-Skeleton-NI.yaml
    ckks-7b-AB-1.yaml
    ckks-7b-AB-1-Skeleton.yaml
    ckks-7b-AB-0.yaml
    ckks-7b-AB-0-Skeleton.yaml
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
/usr/bin/shutdown -h now
