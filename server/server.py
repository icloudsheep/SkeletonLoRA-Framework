"""服务端: 固化 `decrypt → aggregate` 两步顺序,函数体由 main 注入。

SVD 截断已挪到 main 中显性调用,不再在 server 内部固定。
"""

from typing import Any, Callable, Dict, List

import torch


class Server:
    def __init__(
        self,
        decrypt_fn: Callable[[Any], Dict[str, torch.Tensor]],
        aggregate_fn: Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]],
    ) -> None:
        self._decrypt = decrypt_fn
        self._aggregate = aggregate_fn

    def decrypt_aggregate(self, ciphertexts: List[Any]) -> Dict[str, torch.Tensor]:
        return self._aggregate([self._decrypt(c) for c in ciphertexts])
