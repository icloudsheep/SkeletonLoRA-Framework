"""客户端身份、加密入口和跨轮训练指标状态。"""

from collections import deque
from typing import Any, Callable, Dict

import torch


class Client:
    def __init__(
        self,
        client_id: int,
        encrypt_fn: Callable[[Dict[str, torch.Tensor], int, int], Any],
    ) -> None:
        self.client_id = client_id
        self.adapter_name = f"client_{client_id}"
        self.encrypt = encrypt_fn  # 直接挂为属性,由 main 在构造时传入
        self.loss_history: deque[float] = deque()
