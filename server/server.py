"""服务端聚合薄壳，不持有客户端解密函数或密钥。"""

from typing import Any, Callable, List, Optional, Tuple


class Server:
    def __init__(
        self,
        aggregate_fn: Callable[[List[Any]], Any],
        secure_aggregate_fn: Optional[
            Callable[[List[Tuple[int, Any]], int], Any]
        ] = None,
        secure_stream_aggregate_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._aggregate = aggregate_fn
        self._secure_aggregate = secure_aggregate_fn
        self._secure_stream_aggregate = secure_stream_aggregate_fn

    def aggregate(
        self,
        payloads: List[Tuple[int, Any]],
        round_id: int,
    ) -> Any:
        if self._secure_aggregate is not None:
            return self._secure_aggregate(payloads, round_id)
        return self._aggregate([payload for _, payload in payloads])

    def aggregate_stream(
        self,
        payloads: List[Tuple[int, Any]],
        round_id: int,
        progress=None,
    ):
        """流式聚合客户端 payload，不执行客户端解密。"""
        if self._secure_stream_aggregate is None:
            raise RuntimeError("未配置流式安全聚合函数")
        return self._secure_stream_aggregate(
            payloads,
            round_id,
            progress=progress,
        )
