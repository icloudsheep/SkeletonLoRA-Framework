"""聚合结果的 SVD 后处理。

被 server 在 aggregate 之后调用。对聚合结果里的每个二维张量做 SVD 截断，
只保留前 rank 个奇异分量，保证下发的 A / B 与原始 LoRA rank 兼容。
"""

from typing import Dict

import torch


def svd_truncate(state_dict: Dict[str, torch.Tensor], rank: int) -> Dict[str, torch.Tensor]:
    """对 state_dict 中每个二维张量做 top-rank SVD 截断。

    非二维张量原样透传;返回字典的 key 与原 dtype 保持不变。
    """
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if tensor.dim() != 2:
            out[name] = tensor
            continue
        dtype = tensor.dtype
        # SVD 需要浮点;bf16 / half 先升到 float32,算完再降回原 dtype。
        work = tensor.float()
        u, s, vh = torch.linalg.svd(work, full_matrices=False)
        k = min(rank, s.numel())
        truncated = (u[:, :k] * s[:k]) @ vh[:k, :]
        out[name] = truncated.to(dtype)
    return out
