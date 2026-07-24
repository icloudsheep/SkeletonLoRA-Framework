"""按 model kind 分派的 loss 计算与 batch 迁移工具。"""

import torch


def compute_loss(model: torch.nn.Module, batch, kind: str) -> torch.Tensor:
    if kind == "dummy":
        x, y = batch
        return ((model(x) - y) ** 2).mean()
    if kind == "open_llama":
        if not isinstance(batch, dict):
            raise TypeError("open_llama batch 必须是包含 labels 的 dict")
        outputs = model(**batch)
        if outputs.loss is None:
            raise ValueError("open_llama batch 缺少可计算 causal-LM loss 的 labels")
        return outputs.loss
    raise ValueError(f"未知的 model kind: {kind}")


def move_batch(batch, device: torch.device):
    if isinstance(batch, dict):
        return {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
    if isinstance(batch, (list, tuple)):
        return [b.to(device) if torch.is_tensor(b) else b for b in batch]
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch
