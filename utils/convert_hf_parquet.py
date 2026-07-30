"""Convert downloaded Hugging Face Parquet splits to validated JSONL."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


REQUIRED_COLUMNS = {
    "mmlu": {"question", "choices", "answer", "subject"},
    "mmlu_train": {"question", "choices", "answer", "subject"},
    "gsm8k": {"question", "answer"},
}


def convert_record(benchmark: str, record: dict[str, Any], record_number: int) -> dict:
    """Validate one source record and return its JSONL representation."""
    question = record["question"]
    answer = record["answer"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"record {record_number} has an invalid question")

    if benchmark in {"mmlu", "mmlu_train"}:
        choices = record["choices"]
        subject = record["subject"]
        if (
            not isinstance(choices, list)
            or len(choices) != 4
            or not all(isinstance(choice, str) for choice in choices)
        ):
            raise ValueError(f"MMLU record {record_number} must have four choices")
        if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < 4:
            raise ValueError(f"MMLU record {record_number} has an invalid answer")
        if benchmark == "mmlu_train" and (
            subject is None or (isinstance(subject, str) and not subject.strip())
        ):
            subject = "unknown"
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError(f"MMLU record {record_number} has an invalid subject")
        return {
            "question": question,
            "choices": choices,
            "answer": answer,
            "subject": subject,
        }

    if benchmark == "gsm8k":
        if not isinstance(answer, str) or "####" not in answer:
            raise ValueError(f"GSM8K record {record_number} has an invalid answer")
        return {"question": question, "answer": answer}

    raise ValueError(f"unsupported benchmark: {benchmark}")


def convert_parquet(benchmark: str, source: Path, destination: Path) -> int:
    """Convert a Parquet split atomically and return the record count."""
    import pyarrow.parquet as parquet

    if benchmark not in REQUIRED_COLUMNS:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    required_columns = REQUIRED_COLUMNS[benchmark]
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
                    converted = convert_record(benchmark, record, record_count + 1)
                    output.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    record_count += 1

        if record_count == 0:
            raise ValueError(f"{source} contains no records")
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return record_count


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: convert_hf_parquet.py BENCHMARK SOURCE DESTINATION")
    benchmark, source_arg, destination_arg = sys.argv[1:]
    destination = Path(destination_arg)
    record_count = convert_parquet(benchmark, Path(source_arg), destination)
    print(f"[download.sh] wrote {record_count} records to {destination}")


if __name__ == "__main__":
    main()
