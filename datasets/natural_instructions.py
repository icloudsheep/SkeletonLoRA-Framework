"""Super-NaturalInstructions 训练集的本地加载、编码与 IID 分片。"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Iterable, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from datasets.dolly import (
    TokenizedCausalLMDataset,
    _iid_uniform_shards,
    _truncate_prompt,
    _truncate_response,
)


logger = logging.getLogger("skeleton_lora")


def build_natural_instructions_shards(config: dict) -> List[Dataset]:
    """读取训练任务 JSONL，编码目标文本并按客户端确定性均分。"""
    dataset_cfg = config["dataset"]
    if dataset_cfg["split_method"] != "iid_uniform":
        raise NotImplementedError(
            "natural_instructions 数据集目前只支持 iid_uniform，"
            f"收到 {dataset_cfg['split_method']}"
        )

    train_files = _resolve_train_files(dataset_cfg["path"])
    max_length = int(dataset_cfg.get("max_length", 512))
    if max_length < 2:
        raise ValueError("dataset.max_length 必须至少为 2")
    max_samples = dataset_cfg.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("dataset.max_samples 必须为正整数")

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
    for record, source in _iter_records(
        train_files,
        seed=int(config["seed"]),
        max_samples=max_samples,
    ):
        encoded.append(
            _encode_record(
                record,
                tokenizer=tokenizer,
                pad_token_id=pad_token_id,
                max_length=max_length,
                source=source,
            )
        )

    if not encoded:
        raise ValueError(f"Natural Instructions 训练集为空: {dataset_cfg['path']}")
    logger.info(
        "Natural Instructions 数据集已编码: 有效样本=%d max_length=%d",
        len(encoded),
        max_length,
    )
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


def _resolve_train_files(raw_path: str) -> list[Path]:
    if not raw_path:
        raise ValueError("dataset.path 不能为空")
    path = Path(raw_path).expanduser()
    if path.is_file():
        return [path]
    train_dir = path / "train" if path.is_dir() else path
    files = sorted(train_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Natural Instructions 训练文件不存在: {train_dir}")
    return files


def _iter_records(
    files: Iterable[Path],
    *,
    seed: int,
    max_samples: int | None,
):
    ordered_files = list(files)
    random.Random(seed).shuffle(ordered_files)
    per_file_limit = (
        math.ceil(max_samples / len(ordered_files))
        if max_samples is not None
        else None
    )
    emitted = 0
    skipped_empty_targets = 0
    overflow_files: list[Path] = []
    for path in ordered_files:
        emitted_from_file = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path} 第 {line_number} 行无法解析") from exc
                if _has_empty_target(record):
                    skipped_empty_targets += 1
                    continue
                yield record, f"{path}:{line_number}"
                emitted += 1
                emitted_from_file += 1
                if max_samples is not None and emitted >= max_samples:
                    _log_skipped_empty_targets(skipped_empty_targets)
                    return
                if per_file_limit is not None and emitted_from_file >= per_file_limit:
                    overflow_files.append(path)
                    break

    if max_samples is not None and emitted < max_samples:
        for path in overflow_files:
            valid_seen = 0
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path} 第 {line_number} 行无法解析") from exc
                    if _has_empty_target(record):
                        if valid_seen >= per_file_limit:
                            skipped_empty_targets += 1
                        continue
                    valid_seen += 1
                    if valid_seen <= per_file_limit:
                        continue
                    yield record, f"{path}:{line_number}"
                    emitted += 1
                    if emitted >= max_samples:
                        _log_skipped_empty_targets(skipped_empty_targets)
                        return
    _log_skipped_empty_targets(skipped_empty_targets)


def _has_empty_target(record: object) -> bool:
    """识别无法产生 causal-LM 监督信号的空目标。"""
    if not isinstance(record, dict):
        return False
    target = record.get("targets")
    return isinstance(target, str) and not target.strip()


def _log_skipped_empty_targets(count: int) -> None:
    if count:
        logger.warning(
            "Natural Instructions 已跳过空 targets 样本: count=%d",
            count,
        )


def _encode_record(
    record: object,
    *,
    tokenizer,
    pad_token_id: int,
    max_length: int,
    source: str,
) -> dict[str, torch.Tensor]:
    validated = _validate_record(record, source)
    prompt_ids = tokenizer(
        _format_prompt(validated["definition"], validated["inputs"]),
        add_special_tokens=True,
        verbose=False,
    )["input_ids"]
    response_ids = tokenizer(
        validated["targets"],
        add_special_tokens=False,
        verbose=False,
    )["input_ids"]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and (not response_ids or response_ids[-1] != eos_token_id):
        response_ids.append(eos_token_id)

    response_budget = min(len(response_ids), max(1, min(max_length - 1, max_length // 2)))
    response_ids = _truncate_response(response_ids, response_budget, eos_token_id)
    prompt_ids = _truncate_prompt(prompt_ids, max_length - len(response_ids))
    if not response_ids:
        raise ValueError(f"Natural Instructions {source} 无法产生监督 token")

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


def _validate_record(record: object, source: str) -> dict[str, str]:
    if not isinstance(record, dict):
        raise ValueError(f"Natural Instructions {source} 必须是 JSON 对象")
    values: dict[str, str] = {}
    for key in ("definition", "targets"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Natural Instructions {source} 字段 {key} 必须为非空字符串")
        values[key] = value.strip()
    inputs = record.get("inputs")
    if not isinstance(inputs, str):
        raise ValueError(f"Natural Instructions {source} 字段 inputs 必须为字符串")
    values["inputs"] = inputs.strip()
    return values


def _format_prompt(definition: str, inputs: str) -> str:
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{definition.strip()}\n\n"
    )
    if inputs.strip():
        prompt += f"### Input:\n{inputs.strip()}\n\n"
    return prompt + "### Response:\n"
