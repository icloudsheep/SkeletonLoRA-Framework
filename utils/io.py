"""IO 工具: yaml 加载、safetensors 存读、原始字节写入。"""

from pathlib import Path
from typing import Dict

import torch
import yaml
from safetensors.torch import load_file, save_file


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_state_dict_safetensors(state_dict: Dict[str, torch.Tensor], path: str | Path) -> None:
    """把张量字典存成 safetensors 文件;张量会 detach、contiguous 并挪到 CPU。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.detach().contiguous().cpu() for k, v in state_dict.items()}
    save_file(payload, str(path))


def load_state_dict_safetensors(path: str | Path) -> Dict[str, torch.Tensor]:
    return load_file(str(path))


def save_bytes(data: bytes, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
