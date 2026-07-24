"""LoRA 乘积聚合工具。"""

from typing import Dict, List

import torch


def aggregate_lora_products(
    plaintexts: List[Dict[str, torch.Tensor]],
    rank: int,
) -> Dict[str, torch.Tensor]:
    if not plaintexts:
        raise ValueError("plaintexts 不能为空")

    out: Dict[str, torch.Tensor] = {}
    first = plaintexts[0]
    handled = set()

    for a_key, a_tensor in first.items():
        if "lora_A" not in a_key:
            continue
        b_key = a_key.replace("lora_A", "lora_B", 1)
        if b_key not in first:
            raise KeyError(f"缺少与 {a_key} 配对的 {b_key}")

        dtype = a_tensor.dtype
        product = None
        for state in plaintexts:
            a = state[a_key].float()
            b = state[b_key].float()
            client_product = b @ a
            product = client_product if product is None else product + client_product

        assert product is not None
        product = product / len(plaintexts)
        b_new, a_new = factorize_lora_product(product, rank)
        out[a_key] = a_new.to(dtype)
        out[b_key] = b_new.to(first[b_key].dtype)
        handled.add(a_key)
        handled.add(b_key)

    for key in first:
        if key in handled:
            continue
        out[key] = torch.stack([p[key].float() for p in plaintexts]).mean(dim=0).to(first[key].dtype)

    return out


def factorize_lora_product(product: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(product.float(), full_matrices=False)
    k = min(rank, s.numel())
    sqrt_s = torch.sqrt(s[:k])
    b = u[:, :k] * sqrt_s.unsqueeze(0)
    a = sqrt_s.unsqueeze(1) * vh[:k, :]
    return b, a
