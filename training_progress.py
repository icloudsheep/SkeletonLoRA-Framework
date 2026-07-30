"""带终端进度日志的客户端本地训练。"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
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
    """完成一个客户端的本地训练，并记录逐步及轮级训练指标。"""
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
    moving_window = int(config["train"].get("loss_moving_average_window", 20))
    if moving_window <= 0:
        raise ValueError("train.loss_moving_average_window 必须为正整数")
    recent_losses: deque[float] = deque(maxlen=moving_window)
    losses: list[float] = []
    loss_sum_total = 0.0
    supervised_tokens_total = 0
    step_times: list[float] = []

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
        grad_square_sum = 0.0
        for name, parameter in adapter_named_params(model, client.adapter_name):
            if parameter.grad is None:
                continue
            layer_key = strip_adapter_suffix(name)
            grad_norm = parameter.grad.detach().float().norm().item()
            grad_square_sum += grad_norm * grad_norm
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

        global_grad_norm = math.sqrt(grad_square_sum)
        optimizer.step()
        final_loss = float(loss.detach().item())
        recent_losses.append(final_loss)
        moving_avg_loss = sum(recent_losses) / len(recent_losses)
        supervised_tokens = _supervised_token_count(batch, model_kind)
        loss_sum = final_loss * supervised_tokens
        loss_sum_total += loss_sum
        supervised_tokens_total += supervised_tokens
        losses.append(final_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        perplexity = _perplexity(final_loss, model_kind)
        _synchronize(device)
        step_time = time.perf_counter() - step_started
        step_times.append(step_time)
        supervised_tokens_per_second = (
            supervised_tokens / step_time if supervised_tokens else 0.0
        )
        csv_w.step.write(
            {
                "round": rnd,
                "client_id": client.client_id,
                "step": step,
                "loss": final_loss,
                "loss_moving_avg": moving_avg_loss,
                "perplexity": perplexity,
                "supervised_tokens": supervised_tokens,
                "loss_sum": loss_sum,
                "global_grad_norm": global_grad_norm,
                "learning_rate": learning_rate,
                "step_time": step_time,
                "supervised_tokens_per_second": supervised_tokens_per_second,
            }
        )
        client_tb = tb_w.client(client.client_id)
        client_tb.add_scalar("loss/current", final_loss, global_step=global_step)
        client_tb.add_scalar("loss/moving_avg", moving_avg_loss, global_step=global_step)
        client_tb.add_scalar("grad_norm/global", global_grad_norm, global_step=global_step)
        client_tb.add_scalar("learning_rate", learning_rate, global_step=global_step)
        client_tb.add_scalar("performance/step_time", step_time, global_step=global_step)
        client_tb.add_scalar(
            "performance/supervised_tokens_per_second",
            supervised_tokens_per_second,
            global_step=global_step,
        )
        if model_kind == "open_llama":
            client_tb.add_scalar("perplexity", perplexity, global_step=global_step)
            client_tb.add_scalar(
                "supervision/tokens", supervised_tokens, global_step=global_step
            )
        logger.info(
            "round %d client %d step %d/%d: loss=%.6f loss_ma=%.6f "
            "ppl=%s supervised_tokens=%d grad_norm=%.6f lr=%.3e "
            "supervised_tokens_per_second=%.2f step_time=%.3fs",
            rnd,
            client.client_id,
            step + 1,
            local_steps,
            final_loss,
            moving_avg_loss,
            f"{perplexity:.4f}" if model_kind == "open_llama" else "n/a",
            supervised_tokens,
            global_grad_norm,
            learning_rate,
            supervised_tokens_per_second,
            step_time,
        )

    _synchronize(device)
    train_time = time.perf_counter() - train_started
    mean_loss = sum(losses) / len(losses)
    token_weighted_mean_loss = loss_sum_total / supervised_tokens_total
    round_perplexity = _perplexity(token_weighted_mean_loss, model_kind)
    final_moving_avg_loss = sum(recent_losses) / len(recent_losses)
    csv_w.client_round.write(
        {
            "round": rnd,
            "client_id": client.client_id,
            "steps": local_steps,
            "mean_loss": mean_loss,
            "token_weighted_mean_loss": token_weighted_mean_loss,
            "min_loss": min(losses),
            "max_loss": max(losses),
            "final_loss": final_loss,
            "final_moving_avg_loss": final_moving_avg_loss,
            "perplexity": round_perplexity,
            "supervised_tokens": supervised_tokens_total,
            "mean_step_time": sum(step_times) / len(step_times),
            "train_time": train_time,
        }
    )
    client_tb = tb_w.client(client.client_id)
    client_tb.add_scalar("round/mean_loss", mean_loss, global_step=rnd)
    client_tb.add_scalar(
        "round/token_weighted_mean_loss", token_weighted_mean_loss, global_step=rnd
    )
    if model_kind == "open_llama":
        client_tb.add_scalar("round/perplexity", round_perplexity, global_step=rnd)
    logger.info(
        "round %d client %d: 本地训练完成 train_time=%.3fs "
        "mean_loss=%.6f token_weighted_mean_loss=%.6f final_loss=%.6f "
        "final_loss_ma=%.6f supervised_tokens=%d",
        rnd,
        client.client_id,
        train_time,
        mean_loss,
        token_weighted_mean_loss,
        final_loss,
        final_moving_avg_loss,
        supervised_tokens_total,
    )
    plaintext = get_peft_model_state_dict(model, adapter_name=client.adapter_name)
    return {key: value.detach().cpu() for key, value in plaintext.items()}


def _synchronize(device: torch.device) -> None:
    """等待 CUDA 工作完成，使终端显示的耗时包含异步 GPU 计算。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _supervised_token_count(batch, model_kind: str) -> int:
    if model_kind == "open_llama":
        labels = batch.get("labels") if isinstance(batch, dict) else None
        if labels is None:
            raise ValueError("open_llama batch 缺少 labels")
        count = int((labels != -100).sum().item())
        if count <= 0:
            raise ValueError("open_llama batch 没有监督 token")
        return count
    if model_kind == "dummy":
        first = batch[0] if isinstance(batch, (list, tuple)) else batch
        return int(first.shape[0])
    raise ValueError(f"未知的 model kind: {model_kind}")


def _perplexity(loss: float, model_kind: str) -> float:
    if model_kind != "open_llama":
        return float("nan")
    return math.exp(min(loss, 20.0))
