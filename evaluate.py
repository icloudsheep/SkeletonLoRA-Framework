"""训练产物评估入口。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from datasets import build_dataloader, build_shards
from models import build_model
from runtime import build_peft_model, pick_device
from runtime.loss import compute_loss, move_batch
from utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml 配置文件路径")
    parser.add_argument("--run-id", required=True, help="output 下的 RUN_ID")
    args = parser.parse_args()

    root = Path(__file__).parent
    config = load_yaml(args.config)
    run_dir = root / "output" / args.run_id
    adapter_path = run_dir / "checkpoints" / "final" / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"找不到 final adapter: {adapter_path}")

    device = pick_device()
    model = build_peft_model(build_model(config["model"]), 1, config["lora"]).to(device)
    state = load_file(str(adapter_path))
    set_peft_model_state_dict(model, state, adapter_name="client_0")
    model.eval()

    shards = build_shards(config)
    rows = []
    for client_id, shard in enumerate(shards):
        loss = _eval_loss(model, build_dataloader(config, shard), config["model"]["kind"], device)
        rows.append({
            "run_id": args.run_id,
            "client_id": client_id,
            "loss": loss,
            "perplexity": _perplexity(loss, config["model"]["kind"]),
        })

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "eval.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "client_id", "loss", "perplexity"])
        writer.writeheader()
        writer.writerows(rows)


def _eval_loss(model: torch.nn.Module, dataloader, model_kind: str, device: torch.device) -> float:
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            loss = compute_loss(model, move_batch(batch, device), model_kind)
            total_loss += float(loss.detach().item())
            total_batches += 1
    if total_batches == 0:
        raise ValueError("评估数据为空")
    return total_loss / total_batches


def _perplexity(loss: float, model_kind: str) -> float:
    if model_kind != "open_llama":
        return float("nan")
    return math.exp(loss) if loss < 100 else float("inf")


if __name__ == "__main__":
    main()
