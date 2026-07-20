"""peft 相关的 adapter 构建与参数遍历工具。"""

import re
from typing import List

import torch
from peft import LoraConfig, get_peft_model


def build_peft_model(base: torch.nn.Module, num_clients: int, lora_cfg: dict) -> torch.nn.Module:
    """把 base 用 peft 包起来,并注册 N 个命名 adapter(client_0 .. client_{N-1})。"""
    cfg = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=0.0,
        bias="none",
    )
    # peft 支持任意 nn.Module(源码里也这么用),但类型 stub 只声明了 PreTrainedModel;
    # 这里对 dummy 模型 / 未来接入的自定义模型都可正常工作。
    model = get_peft_model(base, cfg, adapter_name="client_0")  # type: ignore[arg-type]
    for i in range(1, num_clients):
        model.add_adapter(f"client_{i}", cfg)
    return model


def strip_adapter_suffix(param_name: str) -> str:
    """把 `...lora_A.client_3.weight` 里的 `.client_3.` 剥掉,得到跨客户端稳定的层 key。"""
    return re.sub(r"\.client_\d+\.", ".", param_name)


def adapter_trainable_params(model: torch.nn.Module, adapter_name: str) -> List[torch.nn.Parameter]:
    return [
        p for name, p in model.named_parameters()
        if p.requires_grad and f".{adapter_name}." in name
    ]


def adapter_named_params(model: torch.nn.Module, adapter_name: str):
    for name, p in model.named_parameters():
        if p.requires_grad and f".{adapter_name}." in name:
            yield name, p
