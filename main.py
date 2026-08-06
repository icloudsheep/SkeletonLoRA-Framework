"""联邦 LoRA 微调入口: 极简编排。

用户仅在下方钩子中填写业务逻辑,其他不动:
  - encrypt_fn(state_dict, client_id, round_id) -> ciphertext (Any)
  - decrypt_fn(ciphertext, client_id, round_id) -> state_dict
  - aggregate_fn(list[state_dict], rank) -> state_dict
  - secure_aggregate_fn(list[(client_id, ciphertext)], round_id) -> state_dict
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from statistics import mean
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
from seclora import build_seclora_backend
from server import Server
from training_progress import train_client_one_round
from utils import CsvWriters, TbWriters, aggregate_lora_products, build_logger, load_yaml, perf_timer, sizeof

# ==================== 用户可填的加解密聚合逻辑 ====================
# 默认为「恒等 + 乘积 FedAvg 等权」,保证明文链路可跑通;换真加密只改下方钩子。

# lambda 是「一屏可读的业务钩子占位」,替换真加密时可换成 def;
# 静态检查器的 E731(lambda 赋名)在此为设计选择,不修。
encrypt_fn = lambda state_dict, client_id, round_id: state_dict  # noqa: E731

decrypt_fn = lambda ciphertext, client_id, round_id: ciphertext  # noqa: E731

aggregate_fn = lambda plaintexts, rank: aggregate_lora_products(plaintexts, rank=rank)  # noqa: E731

secure_aggregate_fn = None

# =================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml 配置文件路径")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)

    paths = prepare_run_paths(Path(__file__).parent)
    shutil.copy2(config_path, paths.out_dir / "experiment.yaml")
    logger = build_logger("skeleton_lora", paths.log_dir / "train.log", config["logging"]["level"])
    logger.info("run_id=%s", paths.run_id)

    seed_all(config["seed"])
    device = pick_device()
    logger.info("使用设备: %s", device)

    num_clients = config["federated"]["num_clients"]
    num_rounds = config["federated"]["num_rounds"]
    rank = config["lora"]["rank"]
    seclora_backend = build_seclora_backend(
        config,
        num_clients=num_clients,
        rank=rank,
        metrics_dir=paths.metrics_dir,
    )
    active_encrypt_fn = seclora_backend.encrypt if seclora_backend else encrypt_fn
    active_secure_aggregate_fn = (
        seclora_backend.secure_aggregate
        if seclora_backend
        else secure_aggregate_fn
    )
    logger.info(
        "聚合后端: %s",
        (
            f"SecLoRA {seclora_backend.config.mode.upper()}"
            if seclora_backend
            else "plaintext product FedAvg"
        ),
    )

    model = build_peft_model(build_model(config["model"]), num_clients, config["lora"]).to(device)
    logger.info("模型就绪: kind=%s num_adapters=%d", config["model"]["kind"], num_clients)

    shards = build_shards(config)
    clients = [
        Client(client_id=i, encrypt_fn=active_encrypt_fn)
        for i in range(num_clients)
    ]
    server = Server(
        decrypt_fn=decrypt_fn,
        aggregate_fn=lambda plaintexts: aggregate_fn(plaintexts, rank),
        secure_aggregate_fn=active_secure_aggregate_fn,
    )

    csv_w = CsvWriters(metrics_dir=paths.metrics_dir)
    tb_w = TbWriters(tb_dir=paths.tb_dir, num_clients=num_clients)

    # 首轮为 None,之后持有上一轮聚合结果(state_dict),广播时灌回每个 adapter。
    global_state: Optional[dict] = None

    for rnd in range(1, num_rounds + 1):
        logger.info("=== round %d/%d ===", rnd, num_rounds)

        if global_state is not None:
            broadcast_to_adapters(model, clients, global_state)

        ciphertexts, client_rows, seclora_client_rows = [], [], []
        for c in clients:
            plaintext = train_client_one_round(
                model=model, client=c, shard=shards[c.client_id],
                config=config, device=device, rnd=rnd, csv_w=csv_w, tb_w=tb_w,
                logger=logger,
            )
            with perf_timer() as t_enc:
                ciphertext = c.encrypt(plaintext, c.client_id, rnd)
            p_size = sizeof(plaintext)
            c_size = (
                seclora_backend.ciphertext_size(ciphertext)
                if seclora_backend
                else sizeof(ciphertext)
            )
            ciphertexts.append((c.client_id, ciphertext))
            client_rows.append({
                "round": rnd, "client_id": c.client_id,
                "encrypt_time": t_enc.value, "plaintext_size": p_size, "ciphertext_size": c_size,
            })
            logger.info("round %d client %d: 加密耗时=%.6fs 明文=%dB 密文=%dB",
                        rnd, c.client_id, t_enc.value, p_size, c_size)
            if seclora_backend:
                metric_row = {
                    "round": rnd,
                    "client_id": c.client_id,
                    **ciphertext.metrics,
                }
                seclora_client_rows.append(metric_row)
                csv_w.seclora_client.write(metric_row)
                logger.info(
                    "round %d client %d SecLoRA: precompute=%.6fs "
                    "online=%.6fs serialize=%.6fs S_P=%dB S_D=%dB "
                    "candidates=(B:%d,A:%d)",
                    rnd,
                    c.client_id,
                    metric_row["precompute_wall_sec"],
                    metric_row["client_online_wall_sec"],
                    metric_row["serialize_wall_sec"],
                    metric_row["sp_upload_bytes"],
                    metric_row["sd_upload_bytes"],
                    metric_row["candidate_b_labels"],
                    metric_row["candidate_a_labels"],
                )
            tb_w.client(c.client_id).add_scalar("encrypt_time", t_enc.value, global_step=rnd)
            tb_w.client(c.client_id).add_scalar("plaintext_size", p_size, global_step=rnd)
            tb_w.client(c.client_id).add_scalar("ciphertext_size", c_size, global_step=rnd)

        # 预初始化让静态检查器知道 with 之外 aggregated 一定有值。
        aggregated: dict = {}
        with perf_timer() as t_agg:
            aggregated = server.decrypt_aggregate(ciphertexts, rnd)
        aggregated = {k: v.detach().cpu() for k, v in aggregated.items()}
        aggregate_metrics = (
            seclora_backend.last_aggregate_metrics if seclora_backend else None
        )
        b_size = (
            aggregate_metrics["download_bytes_per_client"]
            if aggregate_metrics
            else sizeof(aggregated)
        )
        logger.info("round %d 聚合完成: 耗时=%.6fs 下发大小=%dB", rnd, t_agg.value, b_size)
        tb_w.global_.add_scalar("aggregate_time", t_agg.value, global_step=rnd)
        tb_w.global_.add_scalar("broadcast_size", b_size, global_step=rnd)

        for row in client_rows:
            row["aggregate_time"], row["broadcast_size"] = t_agg.value, b_size
            csv_w.round.write(row)

        if seclora_backend:
            assert aggregate_metrics is not None
            client_online_values = [
                row["client_online_wall_sec"] for row in seclora_client_rows
            ]
            client_total_values = [
                row["client_total_crypto_wall_sec"]
                for row in seclora_client_rows
            ]
            encrypted_scalar_values = [
                row["encrypted_scalars"] for row in seclora_client_rows
            ]
            sp_upload_values = [
                row["sp_upload_bytes"] for row in seclora_client_rows
            ]
            sd_upload_values = [
                row["sd_upload_bytes"] for row in seclora_client_rows
            ]
            upload_values = [row["upload_bytes"] for row in seclora_client_rows]
            upload_per_client_mean = mean(upload_values)
            upload_per_client_max = max(upload_values)
            download_per_client = aggregate_metrics["download_bytes_per_client"]
            system_critical = (
                max(client_online_values)
                + aggregate_metrics["server_parallel_critical_wall_sec"]
            )
            network_100mbps = (
                8.0
                * (upload_per_client_mean + download_per_client)
                / 100_000_000.0
            )
            seclora_round_row = {
                "round": rnd,
                "mode": aggregate_metrics["mode"],
                "ratio": aggregate_metrics["ratio"],
                "num_clients": num_clients,
                "client_online_mean_wall_sec": mean(client_online_values),
                "client_online_max_wall_sec": max(client_online_values),
                "client_total_crypto_mean_wall_sec": mean(client_total_values),
                "encrypted_scalars_per_client_mean": mean(
                    encrypted_scalar_values
                ),
                **{
                    key: aggregate_metrics[key]
                    for key in (
                        "sp_wall_sec",
                        "sd_wall_sec",
                        "fe_aggregate_wall_sec",
                        "bsgs_wall_sec",
                        "cur_skeleton_wall_sec",
                        "decrypt_wall_sec",
                        "sd_dfe_mask_wall_sec",
                        "sd_fe_eval_wall_sec",
                        "sd_bsgs_search_wall_sec",
                        "sd_control_wall_sec",
                        "cur_reconstruct_wall_sec",
                        "experiment_verify_wall_sec",
                        "server_common_control_wall_sec",
                        "output_reconstruct_wall_sec",
                        "observed_serial_server_wall_sec",
                        "server_parallel_critical_wall_sec",
                    )
                },
                "system_critical_wall_sec": system_critical,
                "network_100mbps_wall_sec": network_100mbps,
                "e2e_100mbps_wall_sec": system_critical + network_100mbps,
                "sp_upload_bytes_per_client_mean": mean(sp_upload_values),
                "sd_upload_bytes_per_client_mean": mean(sd_upload_values),
                "upload_bytes_per_client_mean": upload_per_client_mean,
                "upload_bytes_per_client_max": upload_per_client_max,
                "upload_bytes_all_clients": sum(upload_values),
                "download_c_bytes_per_client": aggregate_metrics[
                    "download_c_bytes_per_client"
                ],
                "download_m_bytes_per_client": aggregate_metrics[
                    "download_m_bytes_per_client"
                ],
                "download_s_bytes_per_client": aggregate_metrics[
                    "download_s_bytes_per_client"
                ],
                "download_bytes_per_client": download_per_client,
                "download_bytes_all_clients": download_per_client * num_clients,
                "round_traffic_bytes_all_clients": (
                    sum(upload_values) + download_per_client * num_clients
                ),
                "protected_skeleton_cells": aggregate_metrics[
                    "protected_skeleton_cells"
                ],
                "pivot_candidate_cells": aggregate_metrics[
                    "pivot_candidate_cells"
                ],
            }
            csv_w.seclora_round.write(seclora_round_row)
            for layer_row in seclora_backend.last_layer_metrics:
                csv_w.seclora_layer.write(
                    {
                        "round": rnd,
                        "mode": aggregate_metrics["mode"],
                        "ratio": aggregate_metrics["ratio"],
                        **layer_row,
                    }
                )
            logger.info(
                "round %d SecLoRA critical: client=%.6fs FE=%.6fs "
                "BSGS=%.6fs CUR=%.6fs decrypt=%.6fs S_P=%.6fs "
                "S_D=%.6fs output=%.6fs system=%.6fs "
                "upload/client=%.0fB download/client=%dB",
                rnd,
                max(client_online_values),
                aggregate_metrics["fe_aggregate_wall_sec"],
                aggregate_metrics["bsgs_wall_sec"],
                aggregate_metrics["cur_skeleton_wall_sec"],
                aggregate_metrics["decrypt_wall_sec"],
                aggregate_metrics["sp_wall_sec"],
                aggregate_metrics["sd_wall_sec"],
                aggregate_metrics["output_reconstruct_wall_sec"],
                system_critical,
                upload_per_client_mean,
                download_per_client,
            )

        round_dir = save_round_checkpoint(paths.ckpt_root, rnd, aggregated)
        logger.info("round %d checkpoint 已保存到 %s", rnd, round_dir)
        global_state = aggregated

    link_final(paths.ckpt_root, num_rounds)
    csv_w.close()
    tb_w.close()
    if seclora_backend:
        seclora_backend.close()
    logger.info("run %s 全部完成", paths.run_id)


if __name__ == "__main__":
    main()
