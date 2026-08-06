"""CSV + TensorBoard 写入器。

四份 CSV:
  step.csv       -> 每步 loss、困惑度、梯度、学习率与耗时
  client_round.csv -> 每个客户端在一轮本地训练中的 loss 汇总
  round.csv      -> round, client_id, encrypt_time, plaintext_size,
                    ciphertext_size, aggregate_time, broadcast_size
  grad_norm.csv  -> round, client_id, step, layer_name, grad_norm

TensorBoard 按客户端拆子目录,方便在同一块 board 上对比各客户端曲线。
"""

import csv
from pathlib import Path
from typing import Dict, List

from torch.utils.tensorboard import SummaryWriter


STEP_COLS = [
    "round",
    "client_id",
    "step",
    "loss",
    "loss_moving_avg",
    "perplexity",
    "supervised_tokens",
    "loss_sum",
    "global_grad_norm",
    "learning_rate",
    "step_time",
    "supervised_tokens_per_second",
]
CLIENT_ROUND_COLS = [
    "round",
    "client_id",
    "steps",
    "mean_loss",
    "token_weighted_mean_loss",
    "min_loss",
    "max_loss",
    "final_loss",
    "final_moving_avg_loss",
    "perplexity",
    "supervised_tokens",
    "mean_step_time",
    "train_time",
]
ROUND_COLS = [
    "round",
    "client_id",
    "encrypt_time",
    "plaintext_size",
    "ciphertext_size",
    "aggregate_time",
    "broadcast_size",
]
SECLORA_CLIENT_COLS = [
    "round",
    "client_id",
    "mode",
    "ratio",
    "layer_count",
    "quantize_pack_wall_sec",
    "precompute_wall_sec",
    "online_crypto_wall_sec",
    "serialize_wall_sec",
    "client_online_wall_sec",
    "client_total_crypto_wall_sec",
    "protected_b_labels",
    "protected_a_labels",
    "candidate_b_labels",
    "candidate_a_labels",
    "encrypted_scalars",
    "sp_upload_bytes",
    "sd_upload_bytes",
    "upload_bytes",
]
SECLORA_ROUND_COLS = [
    "round",
    "mode",
    "ratio",
    "num_clients",
    "client_online_mean_wall_sec",
    "client_online_max_wall_sec",
    "client_total_crypto_mean_wall_sec",
    "encrypted_scalars_per_client_mean",
    "fe_aggregate_wall_sec",
    "bsgs_wall_sec",
    "cur_skeleton_wall_sec",
    "decrypt_wall_sec",
    "sp_wall_sec",
    "sd_wall_sec",
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
    "system_critical_wall_sec",
    "network_100mbps_wall_sec",
    "e2e_100mbps_wall_sec",
    "sp_upload_bytes_per_client_mean",
    "sd_upload_bytes_per_client_mean",
    "upload_bytes_per_client_mean",
    "upload_bytes_per_client_max",
    "upload_bytes_all_clients",
    "download_c_bytes_per_client",
    "download_m_bytes_per_client",
    "download_s_bytes_per_client",
    "download_bytes_per_client",
    "download_bytes_all_clients",
    "round_traffic_bytes_all_clients",
    "protected_skeleton_cells",
    "pivot_candidate_cells",
]
SECLORA_LAYER_COLS = [
    "round",
    "mode",
    "ratio",
    "layer_id",
    "layer_name",
    "rows",
    "cols",
    "selected_rank",
    "baseline_checks",
    "baseline_relative_error",
    "decrypted_cells",
    "pivot_candidate_cells",
    "download_c_bytes",
    "download_m_bytes",
    "download_s_bytes",
]
GRAD_NORM_COLS = ["round", "client_id", "step", "layer_name", "grad_norm"]


class _CsvFile:
    def __init__(self, path: Path, columns: List[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._fp, fieldnames=columns)
        self._writer.writeheader()
        self._fp.flush()

    def write(self, row: Dict) -> None:
        self._writer.writerow(row)
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


class CsvWriters:
    """持有训练、通信和梯度指标 CSV 的写入器。"""

    def __init__(self, metrics_dir: Path) -> None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self.step = _CsvFile(metrics_dir / "step.csv", STEP_COLS)
        self.client_round = _CsvFile(
            metrics_dir / "client_round.csv", CLIENT_ROUND_COLS
        )
        self.round = _CsvFile(metrics_dir / "round.csv", ROUND_COLS)
        self.seclora_client = _CsvFile(
            metrics_dir / "seclora_client.csv", SECLORA_CLIENT_COLS
        )
        self.seclora_round = _CsvFile(
            metrics_dir / "seclora_round.csv", SECLORA_ROUND_COLS
        )
        self.seclora_layer = _CsvFile(
            metrics_dir / "seclora_layer.csv", SECLORA_LAYER_COLS
        )
        self.grad_norm = _CsvFile(metrics_dir / "grad_norm.csv", GRAD_NORM_COLS)

    def close(self) -> None:
        self.step.close()
        self.client_round.close()
        self.round.close()
        self.seclora_client.close()
        self.seclora_round.close()
        self.seclora_layer.close()
        self.grad_norm.close()


class TbWriters:
    """按客户端拆分的 SummaryWriter,外加一个 server 级全局写入器。"""

    def __init__(self, tb_dir: Path, num_clients: int) -> None:
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.global_ = SummaryWriter(log_dir=str(tb_dir / "server"))
        self.clients: List[SummaryWriter] = [
            SummaryWriter(log_dir=str(tb_dir / f"client_{i}")) for i in range(num_clients)
        ]

    def client(self, client_id: int) -> SummaryWriter:
        return self.clients[client_id]

    def close(self) -> None:
        self.global_.close()
        for w in self.clients:
            w.close()
