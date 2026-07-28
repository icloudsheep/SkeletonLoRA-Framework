"""数据集注册表 + 一次性分片缓存。

`build_shards(config)` 在 main 启动时调用一次:读原始数据集,按 config.dataset.split_method
在客户端之间划分,返回一份 per-client 的 Dataset 列表。

`build_dataloader(config, shard)` 把预先算好的分片包成 DataLoader。
不再重复读盘。
"""

from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.dolly import build_dolly_shards
from datasets.dummy import build_dummy_shards
from datasets.mmlu_train import build_mmlu_train_shards


def build_shards(config: dict) -> List[Dataset]:
    kind = config["dataset"]["kind"]
    if kind == "dummy":
        return build_dummy_shards(config)
    if kind == "dolly_15k":
        return build_dolly_shards(config)
    if kind == "mmlu_train":
        return build_mmlu_train_shards(config)
    raise ValueError(f"未知的 dataset kind: {kind}")


def build_dataloader(
    config: dict,
    shard: Dataset,
    *,
    round_id: int = 0,
    client_id: int = 0,
) -> DataLoader:
    if round_id < 0:
        raise ValueError("round_id 不能为负数")
    if client_id < 0:
        raise ValueError("client_id 不能为负数")
    num_clients = int(config["federated"]["num_clients"])
    if num_clients <= 0:
        raise ValueError("federated.num_clients 必须为正整数")
    if client_id >= num_clients:
        raise ValueError("client_id 必须小于 federated.num_clients")
    shuffle_seed = int(config["seed"]) + round_id * num_clients + client_id
    return DataLoader(
        shard,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(shuffle_seed),
    )
