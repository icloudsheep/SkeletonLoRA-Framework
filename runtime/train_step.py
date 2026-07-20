"""单个客户端一整轮本地训练 + 记录 grad_norm / step loss 的细节工具。

从 main 提取,让 main 的循环只剩「广播 → 训练 → 加密 → 聚合 → 存盘」五步骨架。
"""

from typing import Dict

import torch
from peft import get_peft_model_state_dict

from datasets import build_dataloader
from runtime.loss import compute_loss, move_batch
from runtime.peft_ops import adapter_named_params, adapter_trainable_params, strip_adapter_suffix


def train_client_one_round(
    *,
    model: torch.nn.Module,
    client,
    shard,
    config: dict,
    device: torch.device,
    rnd: int,
    csv_w,
    tb_w,
) -> Dict[str, torch.Tensor]:
    """让 `client` 在自己的分片上跑完本轮 local_steps 步,返回该 adapter 的明文 state_dict。

    过程中把 loss 与 per-layer grad_norm 同时写入 csv 与 tensorboard。
    """
    local_steps = config["federated"]["local_steps"]
    model_kind = config["model"]["kind"]

    # peft 在运行时把 model 的 forward 换成带 adapter 的版本;pyright 的
    # torch stub 只看到 nn.Module 基类,故此处对 model(x) 的静态推断会误判。
    model.set_adapter(client.adapter_name)  # type: ignore[operator]
    dataloader = build_dataloader(config, shard)
    optimizer = torch.optim.AdamW(
        adapter_trainable_params(model, client.adapter_name),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    data_iter = iter(dataloader)
    for step in range(local_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = move_batch(batch, device)

        optimizer.zero_grad()
        loss = compute_loss(model, batch, kind=model_kind)
        loss.backward()

        global_step = (rnd - 1) * local_steps + step
        for name, p in adapter_named_params(model, client.adapter_name):
            if p.grad is None:
                continue
            layer_key = strip_adapter_suffix(name)
            gn = p.grad.detach().float().norm().item()
            csv_w.grad_norm.write({
                "round": rnd, "client_id": client.client_id, "step": step,
                "layer_name": layer_key, "grad_norm": gn,
            })
            tb_w.client(client.client_id).add_scalar(f"grad_norm/{layer_key}", gn, global_step=global_step)

        optimizer.step()

        loss_val = float(loss.detach().item())
        csv_w.step.write({"round": rnd, "client_id": client.client_id, "step": step, "loss": loss_val})
        tb_w.client(client.client_id).add_scalar("loss", loss_val, global_step=global_step)

    plaintext = get_peft_model_state_dict(model, adapter_name=client.adapter_name)
    return {k: v.detach().cpu() for k, v in plaintext.items()}
