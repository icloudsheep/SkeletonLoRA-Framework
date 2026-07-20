"""真基座模型加载器。等本地权重就位后再实现,当前为占位 [注: 截至 2026-07-20]。"""

import torch.nn as nn


def build_open_llama(model_cfg: dict) -> nn.Module:
    # 待办 [注: 截至 2026-07-20]: 用 transformers.AutoModelForCausalLM.from_pretrained(
    #   model_cfg["path"], torch_dtype=torch.bfloat16, local_files_only=True) 加载。
    raise NotImplementedError(
        "open_llama 加载器暂未实现 —— 等本地权重与数据集接入后再补。"
    )
