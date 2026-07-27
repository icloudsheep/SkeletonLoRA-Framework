"""full、partial 和明文 baseline 的矩形分块规则。

本文件复制自 /Users/bilibili/Github/SkeletonLoRA/fe_modes.py。
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ModePartition:
    """一个 AB 对的行列加密分区。"""

    mode: str
    ratio: float | None
    encrypted_rows: np.ndarray
    plain_rows: np.ndarray
    encrypted_cols: np.ndarray
    plain_cols: np.ndarray

    @property
    def output_encrypted_rows(self):
        return self.encrypted_rows

    @property
    def output_encrypted_cols(self):
        return self.encrypted_cols


def _prefix_count(length, ratio):
    if ratio < 0 or ratio > 100:
        raise ValueError(f"比例必须在 [0, 100]，实际为 {ratio}")
    return min(length, math.ceil(length * ratio / 100))


def build_partition(out_features, in_features, mode, ratio=None):
    """按模式构造 B 行和 A 列的加密索引。"""
    if mode == "plain_baseline":
        row_count = col_count = 0
    elif mode == "full":
        row_count = out_features
        col_count = in_features
    elif mode in {"partial_A", "partial_AB"}:
        if ratio is None:
            raise ValueError(f"模式 {mode} 需要 ratio")
        row_count = out_features if mode == "partial_AB" else 0
        col_count = _prefix_count(in_features, ratio)
        if mode == "partial_AB":
            row_count = _prefix_count(out_features, ratio)
    else:
        raise ValueError(f"未知实验模式：{mode}")
    rows = np.arange(out_features, dtype=int)
    cols = np.arange(in_features, dtype=int)
    return ModePartition(
        mode=mode,
        ratio=ratio,
        encrypted_rows=rows[:row_count],
        plain_rows=rows[row_count:],
        encrypted_cols=cols[:col_count],
        plain_cols=cols[col_count:],
    )
