"""Low-rank reconstruction without materializing a full m-by-n matrix."""

from __future__ import annotations

from typing import Tuple

import torch


def factorize_low_rank_product(
    left: torch.Tensor,
    right: torch.Tensor,
    target_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return B, A with B@A equal to the best rank-k approximation of left@right."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError(
            f"Incompatible low-rank factors: left={tuple(left.shape)} "
            f"right={tuple(right.shape)}"
        )
    if target_rank <= 0:
        raise ValueError("target_rank must be positive")

    work_left = left.to(dtype=torch.float64)
    work_right = right.to(dtype=torch.float64)
    q_left, r_left = torch.linalg.qr(work_left, mode="reduced")
    q_right, r_right = torch.linalg.qr(work_right.T, mode="reduced")
    core = r_left @ r_right.T
    u, singular, vh = torch.linalg.svd(core, full_matrices=False)
    rank = min(target_rank, singular.numel())
    sqrt_singular = torch.sqrt(torch.clamp_min(singular[:rank], 0.0))
    b = (q_left @ u[:, :rank]) * sqrt_singular.unsqueeze(0)
    a = sqrt_singular.unsqueeze(1) * (vh[:rank, :] @ q_right.T)
    if rank < target_rank:
        b = torch.cat(
            [b, torch.zeros(b.shape[0], target_rank - rank, dtype=b.dtype)],
            dim=1,
        )
        a = torch.cat(
            [a, torch.zeros(target_rank - rank, a.shape[1], dtype=a.dtype)],
            dim=0,
        )
    return b, a


def decode_skeleton(
    c: torch.Tensor,
    m: torch.Tensor,
    s: torch.Tensor,
    *,
    scale: int,
    clients: int,
    target_rank: int,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode integer C/M/S for the unweighted average and return LoRA B/A."""
    c64 = torch.as_tensor(c, dtype=torch.float64)
    m64 = torch.as_tensor(m, dtype=torch.float64)
    s64 = torch.as_tensor(s, dtype=torch.float64)
    if m64.ndim != 2 or m64.shape[0] != m64.shape[1]:
        raise ValueError(f"Skeleton M must be square, got {tuple(m64.shape)}")
    if c64.shape[1] != m64.shape[0] or s64.shape[0] != m64.shape[0]:
        raise ValueError(
            f"Incompatible skeleton shapes C={tuple(c64.shape)} "
            f"M={tuple(m64.shape)} S={tuple(s64.shape)}"
        )

    if m64.shape[0] == 0:
        return (
            torch.zeros((c64.shape[0], target_rank), dtype=output_dtype),
            torch.zeros((target_rank, s64.shape[1]), dtype=output_dtype),
        )

    right = torch.linalg.solve(m64, s64)
    left = c64 / float(clients * scale * scale)
    b, a = factorize_low_rank_product(left, right, target_rank)
    return b.to(output_dtype), a.to(output_dtype)
