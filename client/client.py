"""客户端: 只持有身份,加密逻辑通过函数指针从 main 注入。"""

from typing import Any, Callable, Dict

import torch


class Client:
    def __init__(self, client_id: int, encrypt_fn: Callable[[Dict[str, torch.Tensor]], Any]) -> None:
        self.client_id = client_id
        self.adapter_name = f"client_{client_id}"
        self.encrypt = encrypt_fn  # 直接挂为属性,由 main 在构造时传入
