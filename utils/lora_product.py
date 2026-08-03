"""LoRA 乘积聚合工具。"""

import math
from typing import Dict, List

import torch


def aggregate_lora_products(
    plaintexts: List[Dict[str, torch.Tensor]],
    rank: int,
) -> Dict[str, torch.Tensor]:
    """等权平均客户端 LoRA 乘积，并截断分解回目标 rank。"""
    if not plaintexts:
        raise ValueError("plaintexts 不能为空")
    if rank <= 0:
        raise ValueError("rank 必须为正整数")

    out: Dict[str, torch.Tensor] = {}
    first = plaintexts[0]
    expected_keys = set(first)
    for client_id, state in enumerate(plaintexts[1:], start=1):
        if set(state) != expected_keys:
            raise ValueError(
                f"client {client_id} 的 state_dict 参数集合与 client 0 不一致"
            )
    handled = set()

    for a_key, a_tensor in first.items():
        if "lora_A" not in a_key:
            continue
        b_key = a_key.replace("lora_A", "lora_B", 1)
        if b_key not in first:
            raise KeyError(f"缺少与 {a_key} 配对的 {b_key}")

        dtype = a_tensor.dtype
        a_factors: list[torch.Tensor] = []
        b_factors: list[torch.Tensor] = []
        for client_id, state in enumerate(plaintexts):
            a_factor = state[a_key]
            b_factor = state[b_key]
            if a_factor.dim() != 2 or b_factor.dim() != 2:
                raise ValueError(
                    f"client {client_id} 的 LoRA A/B 必须都是二维张量"
                )
            if b_factor.size(1) != a_factor.size(0):
                raise ValueError(
                    f"client {client_id} 的 LoRA A/B 形状不匹配: "
                    f"{tuple(b_factor.shape)} @ {tuple(a_factor.shape)}"
                )
            if (
                a_factor.shape != a_tensor.shape
                or b_factor.shape != first[b_key].shape
            ):
                raise ValueError(
                    f"client {client_id} 的 LoRA A/B 形状与 client 0 不一致"
                )
            a_factors.append(a_factor.float())
            b_factors.append(b_factor.float())

        # mean(B_i A_i) 可精确写成 B_cat A_cat。只对至多
        # (num_clients * rank) 阶的小 core 做 SVD，避免构造完整权重矩阵。
        scale = 1.0 / math.sqrt(len(plaintexts))
        b_cat = torch.cat(b_factors, dim=1) * scale
        a_cat = torch.cat(a_factors, dim=0) * scale
        b_new, a_new = factorize_lora_factors(b_cat, a_cat, rank)
        out[a_key] = a_new.to(dtype)
        out[b_key] = b_new.to(first[b_key].dtype)
        handled.add(a_key)
        handled.add(b_key)

    for key in first:
        if key in handled:
            continue
        out[key] = (
            torch.stack([state[key].float() for state in plaintexts])
            .mean(dim=0)
            .to(first[key].dtype)
        )

    return out


def factorize_lora_product(
    product: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对二维稠密乘积执行截断 SVD，并返回平衡缩放的 B/A 因子。"""
    if product.dim() != 2:
        raise ValueError("LoRA product 必须是二维张量")
    if rank <= 0:
        raise ValueError("rank 必须为正整数")
    u, s, vh = torch.linalg.svd(product.float(), full_matrices=False)
    k = min(rank, s.numel())
    sqrt_s = torch.sqrt(s[:k])
    b = u[:, :k] * sqrt_s.unsqueeze(0)
    a = sqrt_s.unsqueeze(1) * vh[:k, :]
    return b, a


def factorize_lora_factors(
    b_factor: torch.Tensor,
    a_factor: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """通过 QR 和小型 core SVD 截断分解 B/A，避免构造稠密乘积。"""
    if b_factor.dim() != 2 or a_factor.dim() != 2:
        raise ValueError("LoRA factors 必须都是二维张量")
    if b_factor.size(1) != a_factor.size(0):
        raise ValueError(
            f"LoRA factors 形状不匹配: {tuple(b_factor.shape)} @ {tuple(a_factor.shape)}"
        )
    if rank <= 0:
        raise ValueError("rank 必须为正整数")

    q_b, r_b = torch.linalg.qr(b_factor.float(), mode="reduced")
    q_a, r_a = torch.linalg.qr(a_factor.float().T, mode="reduced")
    core = r_b @ r_a.T
    u, s, vh = torch.linalg.svd(core, full_matrices=False)
    k = min(rank, s.numel())
    sqrt_s = torch.sqrt(s[:k])
    b = (q_b @ u[:, :k]) * sqrt_s.unsqueeze(0)
    a = sqrt_s.unsqueeze(1) * (vh[:k, :] @ q_a.T)
    return b, a
