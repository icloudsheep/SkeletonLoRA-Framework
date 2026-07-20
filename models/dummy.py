"""冒烟用假模型。

只有一个名为 "linear" 的 nn.Linear,配合 peft target_modules=["linear"]
挂 LoRA。前向接一个张量返回 logits,loss 与 dummy 数据集给的 target 做 MSE。
"""

import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_dummy_model(model_cfg: dict) -> DummyModel:
    return DummyModel(hidden_size=model_cfg["hidden_size"])
