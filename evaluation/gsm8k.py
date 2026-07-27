"""GSM8K 本地数据加载与最终数值答案准确率评测。"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from tqdm.auto import tqdm

from evaluation.common import BenchmarkResult, load_local_tokenizer, require_dataset_path


REFERENCE_ANSWER_PATTERN = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
NUMBER_PATTERN = re.compile(r"-?[\d,]+(?:\.\d+)?")


def evaluate_gsm8k(
    *,
    model: torch.nn.Module,
    model_config: dict,
    benchmark_config: dict,
    device: torch.device,
    run_id: str,
) -> BenchmarkResult:
    path = require_dataset_path(str(benchmark_config.get("path", "")), "gsm8k")
    records = load_gsm8k_records(path)
    tokenizer = load_local_tokenizer(model_config)
    batch_size = int(benchmark_config.get("batch_size", 1))
    max_input_length = int(benchmark_config.get("max_input_length", 1536))
    max_new_tokens = int(benchmark_config.get("max_new_tokens", 512))
    if batch_size <= 0 or max_input_length <= 0 or max_new_tokens <= 0:
        raise ValueError("GSM8K batch_size、max_input_length 和 max_new_tokens 必须为正整数")

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    rows = []
    batches = range(0, len(records), batch_size)
    progress = tqdm(batches, total=(len(records) + batch_size - 1) // batch_size,
                    desc="[evaluate] GSM8K", unit="batch", dynamic_ncols=True)
    try:
        for start in progress:
            batch = records[start:start + batch_size]
            prompts = [_format_prompt(record["question"]) for record in batch]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_tokens = generated[:, encoded["input_ids"].shape[1]:]
            responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for offset, (record, response) in enumerate(zip(batch, responses)):
                prediction = extract_numeric_answer(response)
                answer = extract_reference_answer(record["answer"])
                correct = prediction is not None and prediction == answer
                rows.append(
                    {
                        "run_id": run_id,
                        "benchmark": "gsm8k",
                        "question_id": start + offset,
                        "prediction": "" if prediction is None else prediction,
                        "answer": answer,
                        "correct": int(correct),
                        "response": response.strip(),
                    }
                )
            progress.set_postfix(
                accuracy=f"{sum(row['correct'] for row in rows) / len(rows):.4f}"
            )
    finally:
        tokenizer.padding_side = old_padding_side

    correct_count = sum(row["correct"] for row in rows)
    accuracy = correct_count / len(rows)
    return BenchmarkResult(
        fieldnames=[
            "run_id",
            "benchmark",
            "question_id",
            "prediction",
            "answer",
            "correct",
            "response",
        ],
        rows=rows,
        summary=f"GSM8K exact_match={accuracy:.6f} ({correct_count}/{len(rows)})",
    )


def load_gsm8k_records(path: Path) -> list[dict[str, str]]:
    if path.is_dir():
        candidates = [path / "test.jsonl", path / "gsm8k_test.jsonl"]
        path = next((candidate for candidate in candidates if candidate.is_file()), path / "test.jsonl")
    if not path.is_file():
        raise FileNotFoundError(f"GSM8K JSONL 文件不存在: {path}")

    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"GSM8K JSONL 第 {line_number} 行无法解析") from exc
            if not isinstance(record, dict):
                raise ValueError(f"GSM8K JSONL 第 {line_number} 行必须是对象")
            question, answer = record.get("question"), record.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"GSM8K JSONL 第 {line_number} 行 question 无效")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"GSM8K JSONL 第 {line_number} 行 answer 无效")
            extract_reference_answer(answer)
            records.append({"question": question.strip(), "answer": answer.strip()})
    if not records:
        raise ValueError(f"GSM8K 数据集为空: {path}")
    return records


def extract_reference_answer(text: str) -> str:
    match = REFERENCE_ANSWER_PATTERN.search(text)
    if match is None:
        raise ValueError("GSM8K 参考答案缺少 #### 最终数值")
    return _normalize_number(match.group(1))


def extract_numeric_answer(text: str) -> str | None:
    reference_match = REFERENCE_ANSWER_PATTERN.search(text)
    if reference_match is not None:
        return _normalize_number(reference_match.group(1))
    matches = NUMBER_PATTERN.findall(text)
    return _normalize_number(matches[-1]) if matches else None


def _normalize_number(raw: str) -> str:
    try:
        value = Decimal(raw.replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError(f"无法解析数值答案: {raw}") from exc
    normalized = format(value.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def _format_prompt(question: str) -> str:
    return (
        "Solve the following math problem. Show your reasoning and give the final numeric "
        f"answer after ####.\n\nQuestion: {question}\nAnswer:"
    )
