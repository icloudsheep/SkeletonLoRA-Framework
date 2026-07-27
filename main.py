"""使用 SkeletonLoRA CKKS 密文聚合的联邦 LoRA 微调入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from client import Client
from datasets import build_shards
from models import build_model
from runtime import (
    broadcast_to_adapters,
    build_peft_model,
    link_final,
    pick_device,
    prepare_run_paths,
    save_round_checkpoint,
    seed_all,
)
from server import Server
from skeleton_crypto import SkeletonLoRACrypto
from training_progress import train_client_one_round
from utils import CsvWriters, TbWriters, build_logger, load_yaml, perf_timer, sizeof


def _default_encryption_config(num_clients: int, rank: int) -> dict:
    return {
        "scheme": "ckks",
        "mode": "full",
        "ratio": None,
        "skeleton": True,
        "skeleton_rank": num_clients * rank,
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "global_scale": 2 ** 40,
        "cur_condition_threshold": 1e12,
    }


def _plaintext_path_disabled(*_args, **_kwargs):
    raise RuntimeError("CKKS 模式不允许回退到明文聚合路径")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml 配置文件路径")
    config = load_yaml(parser.parse_args().config)

    paths = prepare_run_paths(Path(__file__).parent)
    logger = build_logger("skeleton_lora", paths.log_dir / "train.log", config["logging"]["level"])
    logger.info("run_id=%s", paths.run_id)

    seed_all(config["seed"])
    device = pick_device()
    logger.info("使用设备: %s", device)

    num_clients = config["federated"]["num_clients"]
    num_rounds = config["federated"]["num_rounds"]
    rank = config["lora"]["rank"]

    crypto = SkeletonLoRACrypto(
        config.get("encryption", _default_encryption_config(num_clients, rank)),
        num_clients=num_clients,
        rank=rank,
    )
    logger.info(
        "CKKS 已就绪: mode=%s skeleton=%s skeleton_rank=%d degree=%d",
        crypto.config.mode,
        crypto.config.skeleton,
        crypto.config.skeleton_rank,
        crypto.config.poly_modulus_degree,
    )

    model = build_peft_model(build_model(config["model"]), num_clients, config["lora"]).to(device)
    logger.info("模型就绪: kind=%s num_adapters=%d", config["model"]["kind"], num_clients)

    shards = build_shards(config)
    clients = [Client(client_id=i, encrypt_fn=crypto.encrypt) for i in range(num_clients)]
    server = Server(
        decrypt_fn=_plaintext_path_disabled,
        aggregate_fn=_plaintext_path_disabled,
        secure_aggregate_fn=crypto.secure_aggregate,
    )

    csv_w = CsvWriters(metrics_dir=paths.metrics_dir)
    tb_w = TbWriters(tb_dir=paths.tb_dir, num_clients=num_clients)

    # 首轮为 None,之后持有上一轮聚合结果(state_dict),广播时灌回每个 adapter。
    global_state: Optional[dict] = None

    for rnd in range(1, num_rounds + 1):
        logger.info("=== round %d/%d ===", rnd, num_rounds)

        if global_state is not None:
            broadcast_to_adapters(model, clients, global_state)

        ciphertexts, client_rows = [], []
        for c in clients:
            plaintext = train_client_one_round(
                model=model, client=c, shard=shards[c.client_id],
                config=config, device=device, rnd=rnd, csv_w=csv_w, tb_w=tb_w,
                logger=logger,
            )
            with perf_timer() as t_enc:
                ciphertext = c.encrypt(plaintext, c.client_id, rnd)
            p_size, c_size = sizeof(plaintext), sizeof(ciphertext)
            ciphertexts.append((c.client_id, ciphertext))
            client_rows.append({
                "round": rnd, "client_id": c.client_id,
                "encrypt_time": t_enc.value, "plaintext_size": p_size, "ciphertext_size": c_size,
            })
            logger.info("round %d client %d: 加密耗时=%.6fs 明文=%dB 密文=%dB",
                        rnd, c.client_id, t_enc.value, p_size, c_size)
            tb_w.client(c.client_id).add_scalar("encrypt_time", t_enc.value, global_step=rnd)
            tb_w.client(c.client_id).add_scalar("plaintext_size", p_size, global_step=rnd)
            tb_w.client(c.client_id).add_scalar("ciphertext_size", c_size, global_step=rnd)

        # 预初始化让静态检查器知道 with 之外 aggregated 一定有值。
        aggregated: dict = {}
        with perf_timer() as t_agg:
            aggregated = server.decrypt_aggregate(ciphertexts, rnd)
        aggregated = {k: v.detach().cpu() for k, v in aggregated.items()}
        b_size = sizeof(aggregated)
        logger.info("round %d 聚合完成: 耗时=%.6fs 下发大小=%dB", rnd, t_agg.value, b_size)
        tb_w.global_.add_scalar("aggregate_time", t_agg.value, global_step=rnd)
        tb_w.global_.add_scalar("broadcast_size", b_size, global_step=rnd)

        for row in client_rows:
            row["aggregate_time"], row["broadcast_size"] = t_agg.value, b_size
            csv_w.round.write(row)

        round_dir = save_round_checkpoint(paths.ckpt_root, rnd, aggregated)
        logger.info("round %d checkpoint 已保存到 %s", rnd, round_dir)
        global_state = aggregated

    link_final(paths.ckpt_root, num_rounds)
    csv_w.close()
    tb_w.close()
    logger.info("run %s 全部完成", paths.run_id)


if __name__ == "__main__":
    main()
