"""Replay CKKS aggregate downloads from a saved adapter without model training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import time

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--exact",
        action="store_true",
        help=(
            "执行完整 K 客户端聚合重放；"
            "默认只重放决定下发大小的代表性块"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).parent
    config = _load_yaml(args.config)
    run_dir = root / "output" / args.run_id
    checkpoint = run_dir / "checkpoints" / "final" / "adapter_model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到 final adapter: {checkpoint}")
    round_csv = run_dir / "metrics" / "round.csv"
    upload_size = _mean_upload_size(round_csv)
    num_clients = int(config["federated"]["num_clients"])
    num_rounds = int(config["federated"]["num_rounds"])
    rank = int(config["lora"]["rank"])
    from safetensors.torch import load_file

    state = load_file(str(checkpoint))
    from skeleton_crypto import SkeletonLoRACrypto

    crypto = SkeletonLoRACrypto(
        config["encryption"],
        num_clients=num_clients,
        rank=rank,
    )

    started = time.perf_counter()
    if args.exact:
        _, stats = crypto.secure_aggregate_streaming(
            [(client_id, state) for client_id in range(num_clients)],
            round_id=1,
        )
        replay = {
            "download_size": stats["download_size"],
            "layer_count": sum("lora_A" in key for key in state),
            "encrypted_blocks": "",
            "plaintext_blocks": "",
        }
        measurement_mode = "exact_protocol_replay"
    else:
        replay = crypto.replay_download_size(state, round_id=1)
        measurement_mode = "representative_protocol_replay"
    elapsed = time.perf_counter() - started

    download_size = int(replay["download_size"])
    total_traffic = (upload_size + download_size) * num_clients * num_rounds
    row = {
        "run_id": args.run_id,
        "config": str(args.config),
        "measurement_mode": measurement_mode,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "ratio_percent": config["encryption"].get("ratio", ""),
        "skeleton": int(bool(config["encryption"].get("skeleton"))),
        "upload_bytes_per_client": f"{upload_size:.3f}",
        "download_bytes_per_client": download_size,
        "upload_mb_per_client": f"{upload_size / 2 ** 20:.6f}",
        "download_mb_per_client": f"{download_size / 2 ** 20:.6f}",
        "total_traffic_gb": f"{total_traffic / 2 ** 30:.6f}",
        "layer_count": replay["layer_count"],
        "encrypted_blocks": replay["encrypted_blocks"],
        "plaintext_blocks": replay["plaintext_blocks"],
        "elapsed_seconds": f"{elapsed:.6f}",
    }
    output_path = args.output or run_dir / "metrics" / "download_traffic_replay.csv"
    _write_result(output_path, row)
    print(
        f"[repair-download] run_id={args.run_id} mode={measurement_mode} "
        f"upload={row['upload_mb_per_client']} MiB/client "
        f"download={row['download_mb_per_client']} MiB/client "
        f"total={row['total_traffic_gb']} GiB/run elapsed={elapsed:.3f}s"
    )
    print(f"[repair-download] wrote {output_path}")


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须是映射: {path}")
    for key in ("federated", "lora", "encryption"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"配置文件缺少 {key}: {path}")
    return config


def _mean_upload_size(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"找不到原实验 round.csv: {path}")
    values = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if "ciphertext_size" not in (reader.fieldnames or []):
            raise ValueError(f"{path} 缺少 ciphertext_size 列")
        for line_number, row in enumerate(reader, start=2):
            try:
                value = float(row["ciphertext_size"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path} 第 {line_number} 行 ciphertext_size 无效"
                ) from exc
            if value < 0:
                raise ValueError(
                    f"{path} 第 {line_number} 行 ciphertext_size 不能为负数"
                )
            values.append(value)
    if not values:
        raise ValueError(f"{path} 不包含通信记录")
    return statistics.fmean(values)


def _write_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
