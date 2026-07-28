"""带终端进度日志的客户端本地训练。"""

from __future__ import annotations

import logging
import time
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
    logger: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """完成一个客户端的本地训练，并输出逐步 loss 与耗时。"""
    local_steps = int(config["federated"]["local_steps"])
    if local_steps <= 0:
        raise ValueError("federated.local_steps 必须为正整数")
    model_kind = config["model"]["kind"]

    model.set_adapter(client.adapter_name)  # type: ignore[operator]
    model.train()
    trainable_params = adapter_trainable_params(model, client.adapter_name)
    if not trainable_params:
        raise RuntimeError(
            f"adapter {client.adapter_name} 没有可训练参数，请检查 lora.target_modules"
        )
    dataloader = build_dataloader(
        config,
        shard,
        round_id=rnd,
        client_id=client.client_id,
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    logger.info(
        "round %d client %d: 开始本地训练 local_steps=%d",
        rnd,
        client.client_id,
        local_steps,
    )
    _synchronize(device)
    train_started = time.perf_counter()
    data_iter = iter(dataloader)
    final_loss = float("nan")

    for step in range(local_steps):
        _synchronize(device)
        step_started = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = move_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, batch, kind=model_kind)
        if not torch.isfinite(loss).all().item():
            raise FloatingPointError(
                f"round {rnd} client {client.client_id} step {step + 1} "
                f"出现非有限 loss: {loss.detach().item()}"
            )
        loss.backward()

        global_step = (rnd - 1) * local_steps + step
        for name, parameter in adapter_named_params(model, client.adapter_name):
            if parameter.grad is None:
                continue
            layer_key = strip_adapter_suffix(name)
            grad_norm = parameter.grad.detach().float().norm().item()
            csv_w.grad_norm.write(
                {
                    "round": rnd,
                    "client_id": client.client_id,
                    "step": step,
                    "layer_name": layer_key,
                    "grad_norm": grad_norm,
                }
            )
            tb_w.client(client.client_id).add_scalar(
                f"grad_norm/{layer_key}", grad_norm, global_step=global_step
            )

        optimizer.step()
        final_loss = float(loss.detach().item())
        csv_w.step.write(
            {
                "round": rnd,
                "client_id": client.client_id,
                "step": step,
                "loss": final_loss,
            }
        )
        tb_w.client(client.client_id).add_scalar(
            "loss", final_loss, global_step=global_step
        )
        _synchronize(device)
        step_time = time.perf_counter() - step_started
        logger.info(
            "round %d client %d step %d/%d: loss=%.6f step_time=%.3fs",
            rnd,
            client.client_id,
            step + 1,
            local_steps,
            final_loss,
            step_time,
        )

    _synchronize(device)
    train_time = time.perf_counter() - train_started
    logger.info(
        "round %d client %d: 本地训练完成 train_time=%.3fs final_loss=%.6f",
        rnd,
        client.client_id,
        train_time,
        final_loss,
    )
    plaintext = get_peft_model_state_dict(model, adapter_name=client.adapter_name)
    return {key: value.detach().cpu() for key, value in plaintext.items()}


def _synchronize(device: torch.device) -> None:
    """等待 CUDA 工作完成，使终端显示的耗时包含异步 GPU 计算。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
