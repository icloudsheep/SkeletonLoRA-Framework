"""客户端持有身份，以及协议注入的加密与聚合 payload 解密函数。"""

from typing import Any, Callable, Dict

import torch


class Client:
    def __init__(
        self,
        client_id: int,
        encrypt_fn: Callable[[Dict[str, torch.Tensor], int, int], Any],
        decrypt_fn: Callable[[dict, int], Any],
    ) -> None:
        self.client_id = client_id
        self.adapter_name = f"client_{client_id}"
        self.encrypt = encrypt_fn
        self.decrypt = decrypt_fn
