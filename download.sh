#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HFD="$ROOT/hfd.sh"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

usage() {
    cat <<'EOF'
Usage: bash download.sh [TARGET ...]

Download models and datasets required by SkeletonLoRA-Framework. With no TARGET,
all resources are downloaded. Available targets:

  llama3bv2  openlm-research/open_llama_3b_v2
  llama7bv2  openlm-research/open_llama_7b_v2
  dolly       databricks/databricks-dolly-15k
  natural-instructions  Muennighoff/natural-instructions train split
  mmlu-train  cais/mmlu auxiliary_train split
  gsm8k-train openai/gsm8k main train split
  mmlu        cais/mmlu test split
  gsm8k       openai/gsm8k main test split
  all         all targets above

Examples:
  bash download.sh
  bash download.sh llama3bv2 dolly
  bash download.sh mmlu-train gsm8k-train
  bash download.sh natural-instructions
  bash download.sh mmlu gsm8k
EOF
}

die() {
    printf '[download.sh] error: %s\n' "$*" >&2
    exit 1
}

[[ -f "$HFD" ]] || die "download helper not found: $HFD"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

targets=("$@")
((${#targets[@]} > 0)) || targets=(all)

needs_pyarrow=false
for target in "${targets[@]}"; do
    case "$target" in
        llama3bv2|llama7bv2|dolly|natural-instructions) ;;
        mmlu-train|gsm8k-train|mmlu|gsm8k|all) needs_pyarrow=true ;;
        *) usage >&2; die "unknown target: $target" ;;
    esac
done

if [[ $needs_pyarrow == true ]]; then
    command -v python >/dev/null 2>&1 || die "python is required to convert Parquet datasets"
    python -c 'import pyarrow.parquet' >/dev/null 2>&1 \
        || die "pyarrow is required; install it in the current Python environment first"
fi

mkdir -p "$ROOT/models" "$ROOT/datasets" "$ROOT/evaluation"

download_repo() {
    local label=$1
    shift
    printf '\n[download.sh] downloading %s\n' "$label"
    bash "$HFD" "$@"
}

convert_parquet() {
    local benchmark=$1
    local source=$2
    local destination=$3

    [[ -s "$source" ]] || die "downloaded Parquet file is missing or empty: $source"
    printf '[download.sh] converting %s to %s\n' "$source" "$destination"

    python "$ROOT/utils/convert_hf_parquet.py" "$benchmark" "$source" "$destination"
}

download_llama3bv2() {
    download_repo "OpenLLaMA 3B v2" \
        openlm-research/open_llama_3b_v2 \
        --local-dir "$ROOT/models/open_llama_3b_v2"
}

download_llama7bv2() {
    download_repo "OpenLLaMA 7B v2" \
        openlm-research/open_llama_7b_v2 \
        --local-dir "$ROOT/models/open_llama_7b_v2"
}

download_dolly() {
    download_repo "Databricks Dolly 15k" \
        databricks/databricks-dolly-15k \
        --dataset \
        --include databricks-dolly-15k.jsonl \
        --local-dir "$ROOT/datasets/databricks-dolly-15k"
}

download_natural_instructions() {
    download_repo "Super-NaturalInstructions training set" \
        Muennighoff/natural-instructions \
        --dataset \
        --include 'train/*.jsonl' \
        --local-dir "$ROOT/datasets/natural-instructions"
}

download_mmlu_train() {
    local directory="$ROOT/datasets/mmlu"
    download_repo "MMLU auxiliary training set" \
        cais/mmlu \
        --dataset \
        --include all/auxiliary_train-00000-of-00001.parquet \
        --local-dir "$directory"
    convert_parquet mmlu_train \
        "$directory/all/auxiliary_train-00000-of-00001.parquet" \
        "$directory/mmlu_auxiliary_train.jsonl"
}

download_gsm8k_train() {
    local directory="$ROOT/datasets/gsm8k"
    download_repo "GSM8K main training set" \
        openai/gsm8k \
        --dataset \
        --include main/train-00000-of-00001.parquet \
        --local-dir "$directory"
    convert_parquet gsm8k \
        "$directory/main/train-00000-of-00001.parquet" \
        "$directory/train.jsonl"
}

download_mmlu() {
    local directory="$ROOT/evaluation/mmlu"
    download_repo "MMLU test set" \
        cais/mmlu \
        --dataset \
        --include all/test-00000-of-00001.parquet \
        --local-dir "$directory"
    convert_parquet mmlu \
        "$directory/all/test-00000-of-00001.parquet" \
        "$directory/mmlu_test.jsonl"
}

download_gsm8k() {
    local directory="$ROOT/evaluation/gsm8k"
    download_repo "GSM8K main test set" \
        openai/gsm8k \
        --dataset \
        --include main/test-00000-of-00001.parquet \
        --local-dir "$directory"
    convert_parquet gsm8k \
        "$directory/main/test-00000-of-00001.parquet" \
        "$directory/test.jsonl"
}

download_all() {
    download_llama3bv2
    download_llama7bv2
    download_dolly
    download_natural_instructions
    download_mmlu_train
    download_gsm8k_train
    download_mmlu
    download_gsm8k
}

for target in "${targets[@]}"; do
    case "$target" in
        llama3bv2) download_llama3bv2 ;;
        llama7bv2) download_llama7bv2 ;;
        dolly) download_dolly ;;
        natural-instructions) download_natural_instructions ;;
        mmlu-train) download_mmlu_train ;;
        gsm8k-train) download_gsm8k_train ;;
        mmlu) download_mmlu ;;
        gsm8k) download_gsm8k ;;
        all) download_all ;;
    esac
done

printf '\n[download.sh] all requested downloads are ready\n'
