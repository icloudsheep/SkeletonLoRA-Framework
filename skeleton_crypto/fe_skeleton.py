"""骨架索引选择与 CUR 重建工具。

本文件复制自 /Users/bilibili/Github/SkeletonLoRA/fe_skeleton.py。
"""

import numpy as np


def select_uniform_rect_indices(n_rows, n_cols, r):
    """分别在矩形矩阵的行维度和列维度上选择 uniform 索引。"""
    if n_rows < r or n_cols < r:
        raise ValueError(
            f"矩阵维度 ({n_rows}, {n_cols}) 小于 skeleton rank={r}"
        )
    rows = np.linspace(0, n_rows - 1, r, dtype=int)
    cols = np.linspace(0, n_cols - 1, r, dtype=int)
    if len(np.unique(rows)) != r or len(np.unique(cols)) != r:
        raise ValueError(f"uniform 索引重复：shape=({n_rows}, {n_cols}), rank={r}")
    return rows, cols


def cur_reconstruct_with_stats(
    C_r,
    R_r,
    I_r,
    J_r,
    condition_threshold=1e12,
):
    """执行矩形 CUR，并返回可解释的数值稳定性统计。"""
    C_r = np.asarray(C_r, dtype=np.float64)
    R_r = np.asarray(R_r, dtype=np.float64)
    I_r = np.asarray(I_r, dtype=int)
    J_r = np.asarray(J_r, dtype=int)
    r = len(J_r)
    stats = {
        "rank": r,
        "numerical_rank": None,
        "max_singular_value": None,
        "min_singular_value": None,
        "condition_number": None,
        "inverse_method": None,
        "ok": False,
        "failure_reason": "",
    }
    if C_r.ndim != 2 or R_r.ndim != 2 or C_r.shape[1] != r or R_r.shape[0] != r:
        stats["failure_reason"] = (
            f"骨架形状不匹配：C={C_r.shape}，R={R_r.shape}，r={r}"
        )
        return None, False, stats
    if len(I_r) != r or C_r.shape[0] <= max(I_r, default=-1):
        stats["failure_reason"] = "行索引与 C 形状不匹配"
        return None, False, stats
    M_r = R_r[:, J_r]
    try:
        singular_values = np.linalg.svd(M_r, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        stats["failure_reason"] = f"交叉块 SVD 失败：{exc}"
        return None, False, stats
    if singular_values.size:
        stats["max_singular_value"] = float(singular_values[0])
        stats["min_singular_value"] = float(singular_values[-1])
        stats["condition_number"] = float(np.linalg.cond(M_r))
        stats["numerical_rank"] = int(np.linalg.matrix_rank(M_r))
    try:
        if not np.isfinite(stats["condition_number"]):
            raise np.linalg.LinAlgError("交叉块条件数非有限")
        if stats["condition_number"] > condition_threshold:
            inverse = np.linalg.pinv(M_r)
            stats["inverse_method"] = "pinv"
        else:
            inverse = np.linalg.inv(M_r)
            stats["inverse_method"] = "inv"
    except np.linalg.LinAlgError:
        try:
            inverse = np.linalg.pinv(M_r)
            stats["inverse_method"] = "pinv"
        except np.linalg.LinAlgError as exc:
            stats["failure_reason"] = f"交叉块求逆失败：{exc}"
            return None, False, stats
    dW_rec = C_r @ inverse @ R_r
    stats["ok"] = bool(np.all(np.isfinite(dW_rec)))
    if not stats["ok"]:
        stats["failure_reason"] = "CUR 输出包含非有限值"
        return None, False, stats
    return dW_rec, True, stats
