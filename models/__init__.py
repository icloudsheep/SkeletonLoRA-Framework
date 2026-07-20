"""模型注册表。按 config.model.kind 分派到具体的 build_ 函数。"""

import torch.nn as nn

from models.dummy import build_dummy_model


def build_model(model_cfg: dict) -> nn.Module:
    kind = model_cfg["kind"]
    if kind == "dummy":
        return build_dummy_model(model_cfg)
    if kind == "open_llama":
        # 延迟导入,避免在 dummy 冒烟场景下拉起 transformers。
        from models.open_llama import build_open_llama
        return build_open_llama(model_cfg)
    raise ValueError(f"未知的 model kind: {kind}")
