"""OpenLLaMA 本地权重加载器。"""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def build_open_llama(model_cfg: dict) -> nn.Module:
    dtype = _torch_dtype(model_cfg.get("dtype", "float32"))
    path = Path(model_cfg["path"]).expanduser()
    return AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=dtype,
        local_files_only=True,
    )


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"不支持的 OpenLLaMA dtype: {name}")
