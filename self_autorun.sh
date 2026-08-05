#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
EVALUATIONS=(
    "configs/ckks-AB-0.yaml" "3B-AB-00-MMLU"
    "configs/ckks-AB-0-Skeleton.yaml" "3B-AB-00-SKE-MMLU"
    "configs/ckks-AB-1.yaml" "3B-AB-01-MMLU"
    "configs/ckks-AB-1-Skeleton.yaml" "3B-AB-01-SKE-MMLU"
    "configs/ckks-7b-AB-0.yaml" "7B-AB-00-MMLU"
    "configs/ckks-7b-AB-0-Skeleton.yaml" "7B-AB-00-SKE-MMLU"
    "configs/ckks-7b-AB-1.yaml" "7B-AB-01-MMLU"
    "configs/ckks-7b-AB-1-Skeleton.yaml" "7B-AB-01-SKE-MMLU"
)

for ((index = 0; index < ${#EVALUATIONS[@]}; index += 2)); do
    config="${EVALUATIONS[index]}"
    run_id="${EVALUATIONS[index + 1]}"
    old_result="$ROOT/output/$run_id/metrics/mmlu.csv"
    new_result="$ROOT/output/$run_id/metrics/mmlu_subject_v1.csv"
    comparison="$ROOT/output/$run_id/metrics/mmlu_subject_v1_comparison.txt"
    if [[ ! -f "$old_result" ]]; then
        echo "[autorun] missing legacy MMLU result: $old_result"
        exit 1
    fi
    echo "[autorun] evaluating MMLU with subject: config=$config run_id=$run_id"
    bash "$ROOT/evaluate.sh" \
        "$ROOT/$config" \
        "$run_id" \
        mmlu \
        adapter \
        "$ROOT/configs/evaluation.yaml" \
        subject_v1
    "$CONDA_PREFIX/bin/python" "$ROOT/evaluation/mmlu_compare.py" \
        "$old_result" \
        "$new_result" \
        | tee "$comparison"
done

echo "[autorun] all done, shutting down"
/usr/bin/shutdown -h now
