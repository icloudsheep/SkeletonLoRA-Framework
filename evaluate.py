"""训练产物评估入口。"""

from __future__ import annotations

import argparse
import csv
import math
import re
import time
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from tqdm.auto import tqdm

from datasets import build_dataloader, build_shards
from evaluation import run_benchmark
from models import build_model
from runtime import build_peft_model, pick_device
from runtime.loss import compute_loss, move_batch
from utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml 配置文件路径")
    parser.add_argument("--run-id", required=True, help="output 下的 RUN_ID")
    parser.add_argument(
        "--target",
        choices=("train", "mmlu", "gsm8k"),
        default="train",
        help="评测目标，train 为原客户端训练分片",
    )
    parser.add_argument(
        "--evaluation-config",
        default=str(Path(__file__).parent / "configs" / "evaluation.yaml"),
        help="专业基准配置文件路径",
    )
    parser.add_argument(
        "--model-mode",
        choices=("adapter", "base"),
        default="adapter",
        help="adapter 加载聚合 LoRA，base 只评测原始底座模型",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="附加到结果文件名的标签，用于保留同一评测的不同口径",
    )
    args = parser.parse_args()
    output_filename = _output_filename(
        args.target, args.model_mode, args.output_tag
    )

    root = Path(__file__).parent
    config = load_yaml(args.config)
    run_dir = root / "output" / args.run_id
    device = pick_device()
    model = _build_evaluation_model(config, run_dir, args.model_mode, device)
    model.eval()
    print(
        f"[evaluate] 模型就绪: mode={args.model_mode} "
        f"model={config['model'].get('name', config['model']['kind'])} device={device}"
    )

    if args.target != "train":
        evaluation_config = load_yaml(args.evaluation_config).get("evaluation", {})
        if args.target not in evaluation_config:
            raise KeyError(f"评测配置缺少 evaluation.{args.target}")
        result = run_benchmark(
            args.target,
            model=model,
            model_config=config["model"],
            benchmark_config=evaluation_config[args.target],
            device=device,
            run_id=args.run_id,
        )
        fieldnames, rows = _add_model_mode(
            result.fieldnames, result.rows, args.model_mode
        )
        output_path = run_dir / "metrics" / output_filename
        _write_rows(output_path, fieldnames, rows)
        print(f"[evaluate] {result.summary}")
        print(f"[evaluate] 全部完成: 结果已保存到 {output_path}")
        return

    _evaluate_training_shards(
        model,
        config,
        device,
        run_dir,
        args.run_id,
        args.model_mode,
        output_filename,
    )


def _build_evaluation_model(
    config: dict,
    run_dir: Path,
    model_mode: str,
    device: torch.device,
) -> torch.nn.Module:
    """按评测模式构建底座模型，或加载聚合后的 LoRA adapter。"""
    if model_mode == "base":
        return build_model(config["model"]).to(device)
    if model_mode != "adapter":
        raise ValueError(f"未知的评测模型模式: {model_mode}")

    adapter_path = run_dir / "checkpoints" / "final" / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"找不到 final adapter: {adapter_path}")
    model = build_peft_model(
        build_model(config["model"]), 1, config["lora"]
    ).to(device)
    state = load_file(str(adapter_path))
    set_peft_model_state_dict(model, state, adapter_name="client_0")
    return model


def _evaluate_training_shards(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    run_dir: Path,
    run_id: str,
    model_mode: str,
    output_filename: str,
) -> None:
    shards = build_shards(config)
    rows = []
    print(f"[evaluate] 开始评估: target=train run_id={run_id} clients={len(shards)} device={device}")
    for client_id, shard in enumerate(shards):
        dataloader = build_dataloader(config, shard)
        print(
            f"[evaluate] client {client_id}: 开始评估 "
            f"samples={len(shard)} batches={len(dataloader)}"
        )
        started = time.perf_counter()
        loss = _eval_loss(
            model,
            dataloader,
            config["model"]["kind"],
            device,
            client_id=client_id,
        )
        elapsed = time.perf_counter() - started
        perplexity = _perplexity(loss, config["model"]["kind"])
        rows.append({
            "run_id": run_id,
            "model_mode": model_mode,
            "client_id": client_id,
            "loss": loss,
            "perplexity": perplexity,
        })
        print(
            f"[evaluate] client {client_id}: 评估完成 "
            f"loss={loss:.6f} perplexity={perplexity:.6f} elapsed={elapsed:.1f}s"
        )

    output_path = run_dir / "metrics" / output_filename
    _write_rows(
        output_path,
        ["run_id", "model_mode", "client_id", "loss", "perplexity"],
        rows,
    )
    print(f"[evaluate] 全部完成: 结果已保存到 {output_path}")


def _add_model_mode(
    fieldnames: list[str],
    rows: list[dict],
    model_mode: str,
) -> tuple[list[str], list[dict]]:
    output_fieldnames = list(fieldnames)
    if "model_mode" not in output_fieldnames:
        output_fieldnames.insert(1, "model_mode")
    output_rows = [{**row, "model_mode": model_mode} for row in rows]
    return output_fieldnames, output_rows


def _output_filename(target: str, model_mode: str, output_tag: str = "") -> str:
    if output_tag and re.fullmatch(r"[A-Za-z0-9_-]+", output_tag) is None:
        raise ValueError("output-tag 只能包含字母、数字、下划线或连字符")
    stem = "eval" if target == "train" else target
    mode_suffix = "_base" if model_mode == "base" else ""
    tag_suffix = f"_{output_tag}" if output_tag else ""
    return f"{stem}{mode_suffix}{tag_suffix}.csv"


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _eval_loss(
    model: torch.nn.Module,
    dataloader,
    model_kind: str,
    device: torch.device,
    *,
    client_id: int,
) -> float:
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        progress = tqdm(
            dataloader,
            desc=f"[evaluate] client {client_id}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in progress:
            loss = compute_loss(model, move_batch(batch, device), model_kind)
            current_loss = float(loss.detach().item())
            total_loss += current_loss
            total_batches += 1
            progress.set_postfix(
                loss=f"{current_loss:.6f}",
                avg_loss=f"{total_loss / total_batches:.6f}",
            )
    if total_batches == 0:
        raise ValueError("评估数据为空")
    return total_loss / total_batches


def _perplexity(loss: float, model_kind: str) -> float:
    if model_kind != "open_llama":
        return float("nan")
    return math.exp(loss) if loss < 100 else float("inf")


if __name__ == "__main__":
    main()
