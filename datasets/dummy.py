"""冒烟用假数据集: 随机 (input, target) 对,配合 MSE loss。

固定 generator 生成一次,再按 iid_uniform 均分给各客户端。
"""

from typing import List

import torch
from torch.utils.data import Dataset, TensorDataset


def build_dummy_shards(config: dict) -> List[Dataset]:
    seed = config["seed"]
    n = config["dataset"]["num_samples"]
    in_dim = config["dataset"]["input_dim"]
    out_dim = config["dataset"]["output_dim"]
    num_clients = config["federated"]["num_clients"]
    split_method = config["dataset"]["split_method"]

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    y = torch.randn(n, out_dim, generator=g)

    if split_method != "iid_uniform":
        raise NotImplementedError(f"dummy 数据集只支持 iid_uniform,收到 {split_method}")

    # 全局打乱后等分给每个客户端。
    perm = torch.randperm(n, generator=g)
    x = x[perm]
    y = y[perm]

    shard_size = n // num_clients
    shards: List[Dataset] = []
    for i in range(num_clients):
        s, e = i * shard_size, (i + 1) * shard_size
        shards.append(TensorDataset(x[s:e], y[s:e]))
    return shards
