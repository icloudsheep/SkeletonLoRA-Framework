"""数据集注册表 + 一次性分片缓存。

`build_shards(config)` 在 main 启动时调用一次:读原始数据集,按 config.dataset.split_method
在客户端之间划分,返回一份 per-client 的 Dataset 列表。

`build_dataloader(config, shard)` 每轮由 main 调用,把预先算好的分片包成 DataLoader。
不再重复读盘。
"""

from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.dummy import build_dummy_shards


def build_shards(config: dict) -> List[Dataset]:
    kind = config["dataset"]["kind"]
    if kind == "dummy":
        return build_dummy_shards(config)
    raise ValueError(f"未知的 dataset kind: {kind}")


def build_dataloader(config: dict, shard: Dataset) -> DataLoader:
    return DataLoader(
        shard,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
