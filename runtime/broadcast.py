"""广播: 把上一轮聚合结果灌回每个 adapter。"""

from typing import Dict, Iterable

import torch
from peft import set_peft_model_state_dict


def broadcast_to_adapters(model: torch.nn.Module, clients: Iterable, state: Dict[str, torch.Tensor]) -> None:
    for c in clients:
        set_peft_model_state_dict(model, state, adapter_name=c.adapter_name)
