"""MMLU auxiliary train 的本地加载、答案监督编码与 IID 分片。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from datasets.dolly import (
    TokenizedCausalLMDataset,
    _iid_uniform_shards,
    _truncate_prompt,
    _truncate_response,
)


CHOICE_LABELS = ("A", "B", "C", "D")


def build_mmlu_train_shards(config: dict) -> List[Dataset]:
    """读取 MMLU Train JSONL，编码正确选项后按客户端确定性均分。"""
    dataset_cfg = config["dataset"]
    if dataset_cfg["split_method"] != "iid_uniform":
        raise NotImplementedError(
            "mmlu_train 数据集目前只支持 iid_uniform，"
            f"收到 {dataset_cfg['split_method']}"
        )

    path = _resolve_dataset_path(dataset_cfg["path"])
    max_length = int(dataset_cfg.get("max_length", 512))
    if max_length < 2:
        raise ValueError("dataset.max_length 必须至少为 2")

    model_path = Path(config["model"]["path"]).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地 tokenizer 目录不存在: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer 同时缺少 pad_token_id 和 eos_token_id")

    encoded = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"MMLU Train JSONL 第 {line_number} 行无法解析") from exc
            encoded.append(
                _encode_record(
                    record,
                    tokenizer=tokenizer,
                    pad_token_id=pad_token_id,
                    max_length=max_length,
                    line_number=line_number,
                )
            )

    if not encoded:
        raise ValueError(f"MMLU Train 数据集为空: {path}")

    dataset = TokenizedCausalLMDataset(
        input_ids=torch.stack([sample["input_ids"] for sample in encoded]),
        attention_mask=torch.stack([sample["attention_mask"] for sample in encoded]),
        labels=torch.stack([sample["labels"] for sample in encoded]),
    )
    return _iid_uniform_shards(
        dataset,
        num_clients=int(config["federated"]["num_clients"]),
        seed=int(config["seed"]),
    )


def _resolve_dataset_path(raw_path: str) -> Path:
    if not raw_path:
        raise ValueError("dataset.path 不能为空")
    path = Path(raw_path).expanduser()
    if path.is_dir():
        path = path / "mmlu_auxiliary_train.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"MMLU Train JSONL 文件不存在: {path}")
    return path


def _encode_record(
    record: object,
    *,
    tokenizer,
    pad_token_id: int,
    max_length: int,
    line_number: int,
) -> dict[str, torch.Tensor]:
    validated = _validate_record(record, line_number)
    prompt_ids = tokenizer(
        _format_prompt(validated),
        add_special_tokens=True,
    )["input_ids"]
    response_ids = tokenizer(
        f" {validated['answer']}",
        add_special_tokens=False,
    )["input_ids"]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and (not response_ids or response_ids[-1] != eos_token_id):
        response_ids.append(eos_token_id)

    response_budget = min(len(response_ids), max(1, min(max_length - 1, max_length // 2)))
    response_ids = _truncate_response(response_ids, response_budget, eos_token_id)
    prompt_ids = _truncate_prompt(prompt_ids, max_length - len(response_ids))
    if not response_ids:
        raise ValueError(f"MMLU Train JSONL 第 {line_number} 行无法产生监督 token")

    input_ids = prompt_ids + response_ids
    padding_length = max_length - len(input_ids)
    return {
        "input_ids": torch.tensor(
            input_ids + [pad_token_id] * padding_length,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [1] * len(input_ids) + [0] * padding_length,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [-100] * len(prompt_ids) + response_ids + [-100] * padding_length,
            dtype=torch.long,
        ),
    }


def _validate_record(record: object, line_number: int) -> dict:
    if not isinstance(record, dict):
        raise ValueError(f"MMLU Train JSONL 第 {line_number} 行必须是 JSON 对象")
    question = record.get("question")
    choices = record.get("choices")
    answer = record.get("answer")
    subject = record.get("subject", "unknown")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"MMLU Train JSONL 第 {line_number} 行 question 无效")
    if not isinstance(choices, list) or len(choices) != 4 or not all(
        isinstance(choice, str) and choice.strip() for choice in choices
    ):
        raise ValueError(
            f"MMLU Train JSONL 第 {line_number} 行 choices 必须包含 4 个非空字符串"
        )
    if isinstance(answer, int) and not isinstance(answer, bool) and 0 <= answer < 4:
        answer = CHOICE_LABELS[answer]
    elif isinstance(answer, str):
        answer = answer.strip().upper()
    if answer not in CHOICE_LABELS:
        raise ValueError(
            f"MMLU Train JSONL 第 {line_number} 行 answer 必须为 A/B/C/D 或 0-3"
        )
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
        f"Subject: {record['subject']}\n"
        f"Question: {record['question']}\n{options}\nAnswer:"
    )
