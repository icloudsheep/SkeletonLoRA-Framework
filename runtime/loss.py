"""按 model kind 分派的 loss 计算与 batch 迁移工具。"""

import torch


def compute_loss(model: torch.nn.Module, batch, kind: str) -> torch.Tensor:
    if kind == "dummy":
        x, y = batch
        return ((model(x) - y) ** 2).mean()
    if kind == "open_llama":
        # 占位 [注: 截至 2026-07-20]: 等本地权重与数据集接入后再补 causal-LM 的 loss。
        raise NotImplementedError("open_llama 的 loss 分支尚未实现")
    raise ValueError(f"未知的 model kind: {kind}")


def move_batch(batch, device: torch.device):
    if isinstance(batch, (list, tuple)):
        return [b.to(device) if torch.is_tensor(b) else b for b in batch]
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch
