"""MMLU 本地数据加载与零样本四选一准确率评测。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from evaluation.common import (
    BenchmarkResult,
    completion_log_probability,
    load_local_tokenizer,
    require_dataset_path,
)


CHOICE_LABELS = ("A", "B", "C", "D")


def evaluate_mmlu(
    *,
    model: torch.nn.Module,
    model_config: dict,
    benchmark_config: dict,
    device: torch.device,
    run_id: str,
) -> BenchmarkResult:
    path = require_dataset_path(str(benchmark_config.get("path", "")), "mmlu")
    records = load_mmlu_records(path)
    tokenizer = load_local_tokenizer(model_config)
    max_length = int(benchmark_config.get("max_length", 2048))
    if max_length < 2:
        raise ValueError("evaluation.mmlu.max_length 必须至少为 2")

    question_rows = []
    subject_totals: dict[str, list[int]] = {}
    progress = tqdm(records, desc="[evaluate] MMLU", unit="question", dynamic_ncols=True)
    for index, record in enumerate(progress):
        prompt = _format_prompt(record)
        scores = [
            completion_log_probability(
                model,
                tokenizer,
                prompt,
                f" {label}",
                device,
                max_length,
            )
            for label in CHOICE_LABELS
        ]
        prediction = CHOICE_LABELS[max(range(len(scores)), key=scores.__getitem__)]
        correct = prediction == record["answer"]
        totals = subject_totals.setdefault(record["subject"], [0, 0])
        totals[0] += int(correct)
        totals[1] += 1
        question_rows.append(
            {
                "run_id": run_id,
                "benchmark": "mmlu",
                "row_type": "question",
                "subject": record["subject"],
                "question_id": index,
                "prediction": prediction,
                "answer": record["answer"],
                "correct": int(correct),
            }
        )
        correct_count = sum(item[0] for item in subject_totals.values())
        progress.set_postfix(accuracy=f"{correct_count / len(question_rows):.4f}")

    correct_count = sum(row["correct"] for row in question_rows)
    overall = correct_count / len(question_rows)
    summary_rows = [
        {
            "run_id": run_id,
            "benchmark": "mmlu",
            "row_type": "overall",
            "subject": "ALL",
            "correct_count": correct_count,
            "total_count": len(question_rows),
            "accuracy": overall,
        }
    ]
    summary_rows.extend(
        {
            "run_id": run_id,
            "benchmark": "mmlu",
            "row_type": "subject",
            "subject": subject,
            "correct_count": correct,
            "total_count": total,
            "accuracy": correct / total,
        }
        for subject, (correct, total) in sorted(subject_totals.items())
    )
    subject_summary = ", ".join(
        f"{subject}={correct}/{total}"
        for subject, (correct, total) in sorted(subject_totals.items())
    )
    return BenchmarkResult(
        fieldnames=[
            "run_id",
            "benchmark",
            "row_type",
            "subject",
            "question_id",
            "prediction",
            "answer",
            "correct",
            "correct_count",
            "total_count",
            "accuracy",
        ],
        rows=summary_rows + question_rows,
        summary=(
            f"MMLU accuracy={overall:.6f} ({correct_count}/{len(question_rows)}); "
            f"subjects: {subject_summary}"
        ),
    )


def load_mmlu_records(path: Path) -> list[dict]:
    if path.is_file():
        files = [path]
    else:
        test_dir = path / "test"
        search_root = test_dir if test_dir.is_dir() else path
        files = sorted(search_root.glob("*_test.csv"))
        if not files:
            files = sorted(search_root.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"MMLU 路径中没有 *_test.csv 或 JSONL 文件: {path}")

    records = []
    for file_path in files:
        if file_path.suffix.lower() == ".csv":
            records.extend(_load_csv(file_path))
        elif file_path.suffix.lower() == ".jsonl":
            records.extend(_load_jsonl(file_path))
        else:
            raise ValueError(f"MMLU 不支持的文件格式: {file_path}")
    if not records:
        raise ValueError(f"MMLU 数据集为空: {path}")
    return records


def _load_csv(path: Path) -> list[dict]:
    subject = path.stem.removesuffix("_test")
    records = []
    with path.open(newline="", encoding="utf-8") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if len(row) != 6:
                raise ValueError(f"MMLU CSV {path} 第 {line_number} 行必须包含 6 列")
            records.append(_validate_record(row[0], row[1:5], row[5], subject, path, line_number))
    return records


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"MMLU JSONL {path} 第 {line_number} 行无法解析") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"MMLU JSONL {path} 第 {line_number} 行必须是对象")
            records.append(
                _validate_record(
                    raw.get("question"),
                    raw.get("choices"),
                    raw.get("answer"),
                    raw.get("subject", path.stem),
                    path,
                    line_number,
                )
            )
    return records


def _validate_record(question, choices, answer, subject, path: Path, line_number: int) -> dict:
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"MMLU {path} 第 {line_number} 行 question 无效")
    if not isinstance(choices, list) or len(choices) != 4 or not all(
        isinstance(choice, str) for choice in choices
    ):
        raise ValueError(f"MMLU {path} 第 {line_number} 行 choices 必须包含 4 个字符串")
    if isinstance(answer, int) and 0 <= answer < 4:
        answer = CHOICE_LABELS[answer]
    answer = str(answer).strip().upper()
    if answer not in CHOICE_LABELS:
        raise ValueError(f"MMLU {path} 第 {line_number} 行 answer 必须为 A/B/C/D 或 0-3")
    return {
        "question": question.strip(),
        "choices": [choice.strip() for choice in choices],
        "answer": answer,
        "subject": str(subject).strip() or "unknown",
    }


def _format_prompt(record: dict) -> str:
    options = "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(CHOICE_LABELS, record["choices"])
    )
    return (
        "The following is a multiple choice question. Choose the correct answer.\n\n"
        f"Question: {record['question']}\n{options}\nAnswer:"
    )
