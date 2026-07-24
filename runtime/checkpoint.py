"""checkpoint 落盘: 保存 LoRA 权重并维护 final 软链。"""

from pathlib import Path
from typing import Dict

import torch

from utils import save_state_dict_safetensors


def save_round_checkpoint(ckpt_root: Path, rnd: int, aggregated: Dict[str, torch.Tensor]) -> Path:
    round_dir = ckpt_root / f"round_{rnd:02d}"
    a_sd = {k: v for k, v in aggregated.items() if "lora_A" in k}
    b_sd = {k: v for k, v in aggregated.items() if "lora_B" in k}
    save_state_dict_safetensors(a_sd, round_dir / "A.safetensors")
    save_state_dict_safetensors(b_sd, round_dir / "B.safetensors")
    save_state_dict_safetensors(aggregated, round_dir / "adapter_model.safetensors")
    return round_dir


def link_final(ckpt_root: Path, num_rounds: int) -> Path:
    link = ckpt_root / "final"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(f"round_{num_rounds:02d}")
    return link
