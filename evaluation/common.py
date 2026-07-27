"""专业基准共享的数据结构与模型打分工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


@dataclass
class BenchmarkResult:
    fieldnames: list[str]
    rows: list[dict[str, Any]]
    summary: str


def require_dataset_path(raw_path: str, benchmark: str) -> Path:
    if not raw_path:
        raise ValueError(
            f"evaluation.{benchmark}.path 为空，请配置本地 {benchmark.upper()} 数据集路径"
        )
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{benchmark.upper()} 数据集路径不存在: {path}")
    return path


def load_local_tokenizer(model_config: dict):
    model_path = Path(model_config["path"]).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地 tokenizer 目录不存在: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer 同时缺少 pad_token_id 和 eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def completion_log_probability(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
    max_length: int,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not completion_ids:
        raise ValueError("候选答案分词后为空")
    if max_length <= len(completion_ids):
        raise ValueError("MMLU max_length 必须大于候选答案 token 数")
    prompt_ids = prompt_ids[-(max_length - len(completion_ids)):]
    input_ids = torch.tensor([prompt_ids + completion_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    start = len(prompt_ids) - 1
    score = 0.0
    for offset, token_id in enumerate(completion_ids):
        score += float(log_probs[start + offset, token_id].item())
    return score
