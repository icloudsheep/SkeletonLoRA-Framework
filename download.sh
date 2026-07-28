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
  mmlu        cais/mmlu test split
  gsm8k       openai/gsm8k main test split
  all         all targets above

Examples:
  bash download.sh
  bash download.sh llama3bv2 dolly
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
        llama3bv2|llama7bv2|dolly) ;;
        mmlu|gsm8k|all) needs_pyarrow=true ;;
        *) usage >&2; die "unknown target: $target" ;;
    esac
done

if [[ $needs_pyarrow == true ]]; then
    command -v python >/dev/null 2>&1 || die "python is required to convert evaluation datasets"
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

    # Validate records before replacing an existing JSONL, so a failed conversion remains recoverable.
    python - "$benchmark" "$source" "$destination" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

import pyarrow.parquet as parquet

benchmark, source_arg, destination_arg = sys.argv[1:]
source = Path(source_arg)
destination = Path(destination_arg)
destination.parent.mkdir(parents=True, exist_ok=True)

required_columns = {
    "mmlu": {"question", "choices", "answer", "subject"},
    "gsm8k": {"question", "answer"},
}[benchmark]
parquet_file = parquet.ParquetFile(source)
missing = required_columns.difference(parquet_file.schema_arrow.names)
if missing:
    raise ValueError(f"{source} is missing columns: {', '.join(sorted(missing))}")

temporary_path = None
record_count = 0
try:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as output:
        temporary_path = Path(output.name)
        for batch in parquet_file.iter_batches(columns=sorted(required_columns)):
            for record in batch.to_pylist():
                question = record["question"]
                answer = record["answer"]
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"record {record_count + 1} has an invalid question")

                if benchmark == "mmlu":
                    choices = record["choices"]
                    subject = record["subject"]
                    if (
                        not isinstance(choices, list)
                        or len(choices) != 4
                        or not all(isinstance(choice, str) for choice in choices)
                    ):
                        raise ValueError(f"MMLU record {record_count + 1} must have four choices")
                    if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < 4:
                        raise ValueError(f"MMLU record {record_count + 1} has an invalid answer")
                    if not isinstance(subject, str) or not subject.strip():
                        raise ValueError(f"MMLU record {record_count + 1} has an invalid subject")
                    converted = {
                        "question": question,
                        "choices": choices,
                        "answer": answer,
                        "subject": subject,
                    }
                else:
                    if not isinstance(answer, str) or "####" not in answer:
                        raise ValueError(f"GSM8K record {record_count + 1} has an invalid answer")
                    converted = {"question": question, "answer": answer}

                output.write(json.dumps(converted, ensure_ascii=False) + "\n")
                record_count += 1

    if record_count == 0:
        raise ValueError(f"{source} contains no records")
    os.replace(temporary_path, destination)
except BaseException:
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)
    raise

print(f"[download.sh] wrote {record_count} records to {destination}")
PY
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
    download_mmlu
    download_gsm8k
}

for target in "${targets[@]}"; do
    case "$target" in
        llama3bv2) download_llama3bv2 ;;
        llama7bv2) download_llama7bv2 ;;
        dolly) download_dolly ;;
        mmlu) download_mmlu ;;
        gsm8k) download_gsm8k ;;
        all) download_all ;;
    esac
done

printf '\n[download.sh] all requested downloads are ready\n'
