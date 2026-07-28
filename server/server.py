"""服务端: 解密聚合由 main 注入。"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


class Server:
    def __init__(
        self,
        decrypt_fn: Callable[[Any, int, int], Dict[str, torch.Tensor]],
        aggregate_fn: Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]],
        secure_aggregate_fn: Optional[
            Callable[[List[Tuple[int, Any]], int], Dict[str, torch.Tensor]]
        ] = None,
        secure_stream_aggregate_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._decrypt = decrypt_fn
        self._aggregate = aggregate_fn
        self._secure_aggregate = secure_aggregate_fn
        self._secure_stream_aggregate = secure_stream_aggregate_fn

    def decrypt_aggregate(
        self,
        ciphertexts: List[Tuple[int, Any]],
        round_id: int,
    ) -> Dict[str, torch.Tensor]:
        if self._secure_aggregate is not None:
            return self._secure_aggregate(ciphertexts, round_id)
        plaintexts = [
            self._decrypt(ciphertext, client_id, round_id)
            for client_id, ciphertext in ciphertexts
        ]
        return self._aggregate(plaintexts)

    def stream_decrypt_aggregate(
        self,
        plaintexts: List[Tuple[int, Dict[str, torch.Tensor]]],
        round_id: int,
        progress=None,
    ):
        """以流式安全聚合实现处理一轮客户端 LoRA 状态。"""
        if self._secure_stream_aggregate is None:
            raise RuntimeError("未配置流式安全聚合函数")
        return self._secure_stream_aggregate(
            plaintexts,
            round_id,
            progress=progress,
        )
