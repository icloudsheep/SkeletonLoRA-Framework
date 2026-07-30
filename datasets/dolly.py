"""Databricks Dolly 15k 的本地加载、指令微调编码与 IID 分片。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer


class TokenizedCausalLMDataset(Dataset):
    """把已编码的固定长度张量保存在内存中，避免每轮重复分词。"""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def build_dolly_shards(config: dict) -> List[Dataset]:
    """读取 Dolly JSONL，一次性编码后按客户端确定性打乱并均分。"""
    dataset_cfg = config["dataset"]
    if dataset_cfg["split_method"] != "iid_uniform":
        raise NotImplementedError(
            "dolly_15k 数据集目前只支持 iid_uniform，"
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
                raise ValueError(f"Dolly JSONL 第 {line_number} 行无法解析") from exc
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
        raise ValueError(f"Dolly 数据集为空: {path}")

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
        path = path / "databricks-dolly-15k.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Dolly JSONL 文件不存在: {path}")
    return path


def _encode_record(
    record: object,
    *,
    tokenizer,
    pad_token_id: int,
    max_length: int,
    line_number: int,
) -> dict[str, torch.Tensor]:
    if not isinstance(record, dict):
        raise ValueError(f"Dolly JSONL 第 {line_number} 行必须是 JSON 对象")
    for key in ("instruction", "context", "response"):
        if key not in record or not isinstance(record[key], str):
            raise ValueError(f"Dolly JSONL 第 {line_number} 行缺少字符串字段 {key}")
    if not record["instruction"].strip() or not record["response"].strip():
        raise ValueError(f"Dolly JSONL 第 {line_number} 行 instruction/response 不能为空")

    prompt = _format_prompt(record["instruction"], record["context"])
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    response_ids = tokenizer(record["response"], add_special_tokens=False)["input_ids"]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and (not response_ids or response_ids[-1] != eos_token_id):
        response_ids.append(eos_token_id)

    response_budget = min(len(response_ids), max(1, min(max_length - 1, max_length // 2)))
    response_ids = _truncate_response(response_ids, response_budget, eos_token_id)
    prompt_ids = _truncate_prompt(prompt_ids, max_length - len(response_ids))
    input_ids = prompt_ids + response_ids
    supervised_length = len(response_ids)
    if supervised_length == 0:
        raise ValueError(f"Dolly JSONL 第 {line_number} 行无法产生监督 token")

    padding_length = max_length - len(input_ids)
    attention_mask = [1] * len(input_ids) + [0] * padding_length
    labels = [-100] * len(prompt_ids) + response_ids + [-100] * padding_length
    input_ids += [pad_token_id] * padding_length
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _truncate_prompt(token_ids: list[int], budget: int) -> list[int]:
    if len(token_ids) <= budget:
        return token_ids
    if budget <= 0:
        return []
    if budget == 1:
        return token_ids[-1:]
    return token_ids[:1] + token_ids[-(budget - 1):]


def _truncate_response(
    token_ids: list[int],
    budget: int,
    eos_token_id: int | None,
) -> list[int]:
    if len(token_ids) <= budget:
        return token_ids
    if budget <= 0:
        return []
    if eos_token_id is not None and budget >= 2 and token_ids[-1] == eos_token_id:
        return token_ids[: budget - 1] + [eos_token_id]
    return token_ids[:budget]


def _format_prompt(instruction: str, context: str) -> str:
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction.strip()}\n\n"
    )
    if context.strip():
        prompt += f"### Context:\n{context.strip()}\n\n"
    return prompt + "### Response:\n"


def _iid_uniform_shards(
    dataset: Dataset,
    *,
    num_clients: int,
    seed: int,
) -> List[Dataset]:
    if num_clients <= 0:
        raise ValueError("federated.num_clients 必须为正整数")
    if num_clients > len(dataset):
        raise ValueError(
            f"客户端数量 {num_clients} 不能超过数据样本数量 {len(dataset)}"
        )
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(seed),
    )
    return [
        Subset(dataset, shard.tolist())
        for shard in torch.tensor_split(indices, num_clients)
    ]
