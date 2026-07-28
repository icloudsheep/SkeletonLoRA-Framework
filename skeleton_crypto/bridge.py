"""Adapt SkeletonLoRA's CKKS protocol to Framework state dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
import resource
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

from skeleton_crypto.fe_context import create_secret_context, derive_public_context
from skeleton_crypto.fe_modes import build_partition
from skeleton_crypto.fe_outer_hybrid import (
    accumulate_block,
    aggregate,
    build_blocks,
    decrypt_block,
    decrypt_result,
    encrypt_upload,
    finalize_block,
    iter_block_uploads,
)
from skeleton_crypto.fe_skeleton import cur_reconstruct_with_stats, select_uniform_rect_indices
from utils import factorize_lora_product


@dataclass(frozen=True)
class CryptoConfig:
    mode: str
    ratio: float | None
    skeleton: bool
    skeleton_rank: int
    poly_modulus_degree: int
    coeff_mod_bit_sizes: list[int]
    global_scale: float
    cur_condition_threshold: float
    memory_strategy: str
    max_rss_bytes: int | None
    progress_interval_blocks: int


class SkeletonLoRACrypto:
    """Encrypt, aggregate and decrypt LoRA A/B pairs with CKKS."""

    protocol = "skeleton_lora_ckks_v1"

    def __init__(self, config: dict, *, num_clients: int, rank: int) -> None:
        self.config = _parse_config(config)
        self.num_clients = int(num_clients)
        self.rank = int(rank)
        if self.num_clients <= 0 or self.rank <= 0:
            raise ValueError("num_clients 和 rank 必须为正整数")
        self.secret_context = create_secret_context(
            self.config.poly_modulus_degree,
            self.config.coeff_mod_bit_sizes,
            self.config.global_scale,
            galois=False,
        )
        self.public_context = derive_public_context(self.secret_context)

    def encrypt(self, state_dict: dict[str, torch.Tensor], client_id: int, round_id: int) -> dict:
        """Encrypt every paired LoRA A/B tensor in one client state dictionary."""
        if client_id < 0 or client_id >= self.num_clients:
            raise ValueError(f"client_id 越界: {client_id}")
        layers: dict[str, dict[str, Any]] = {}
        handled: set[str] = set()
        for a_key in sorted(key for key in state_dict if "lora_A" in key):
            b_key = a_key.replace("lora_A", "lora_B", 1)
            if b_key not in state_dict:
                raise KeyError(f"缺少与 {a_key} 配对的 {b_key}")
            a_tensor = state_dict[a_key]
            b_tensor = state_dict[b_key]
            self._validate_pair(a_key, a_tensor, b_key, b_tensor)
            out_features, in_features = b_tensor.shape[0], a_tensor.shape[1]
            rows, cols = self._skeleton_indices(out_features, in_features)
            partition = build_partition(
                out_features,
                in_features,
                self.config.mode,
                self.config.ratio,
            )
            blocks = build_blocks(
                out_features,
                in_features,
                partition,
                self.config.skeleton,
                skeleton_rows=rows,
                skeleton_cols=cols,
                max_slots=self.config.poly_modulus_degree // 2,
            )
            layers[a_key] = {
                "b_key": b_key,
                "a_dtype": a_tensor.dtype,
                "b_dtype": b_tensor.dtype,
                "skeleton_rows": rows.tolist(),
                "skeleton_cols": cols.tolist(),
                "upload": encrypt_upload(
                    b_tensor.detach().cpu().numpy(),
                    a_tensor.detach().cpu().numpy(),
                    self.public_context,
                    self.rank,
                    partition,
                    blocks,
                ),
            }
            handled.update((a_key, b_key))
        if not layers:
            raise ValueError("state_dict 中没有 LoRA A/B 参数")
        unsupported = sorted(set(state_dict) - handled)
        if unsupported:
            raise ValueError(f"存在未配对或不支持加密的参数: {unsupported}")
        return {
            "protocol": self.protocol,
            "client_id": client_id,
            "round_id": round_id,
            "layers": layers,
        }

    def secure_aggregate(self, ciphertexts: list[tuple[int, dict]], round_id: int) -> dict[str, torch.Tensor]:
        """Aggregate encrypted LoRA products and return factorized A/B tensors."""
        packages = self._validate_packages(ciphertexts, round_id)
        first_layers = packages[0]["layers"]
        output: dict[str, torch.Tensor] = {}
        for a_key in sorted(first_layers):
            layer_packages = [package["layers"][a_key] for package in packages]
            self._validate_layer_metadata(a_key, layer_packages)
            encrypted_product = aggregate(
                [layer["upload"] for layer in layer_packages],
                self.public_context,
                self.num_clients,
            )
            product, _ = decrypt_result(encrypted_product, self.secret_context)
            if self.config.skeleton:
                product = self._reconstruct_product(a_key, product, layer_packages[0])
            product_tensor = torch.from_numpy(np.asarray(product, dtype=np.float32))
            b_new, a_new = factorize_lora_product(product_tensor, self.rank)
            first = layer_packages[0]
            output[a_key] = a_new.to(dtype=first["a_dtype"])
            output[first["b_key"]] = b_new.to(dtype=first["b_dtype"])
        return output

    def secure_aggregate_streaming(
        self,
        plaintexts: list[tuple[int, dict[str, torch.Tensor]]],
        round_id: int,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """逐层逐块聚合客户端状态，仅解密每个块的聚合结果。"""
        ordered_plaintexts = sorted(plaintexts, key=lambda item: item[0])
        states = self._validate_plaintexts(ordered_plaintexts)
        layer_keys = sorted(key for key in states[0] if "lora_A" in key)
        client_stats = {
            client_id: {"encrypt_time": 0.0, "ciphertext_size": 0}
            for client_id, _ in ordered_plaintexts
        }
        output: dict[str, torch.Tensor] = {}
        observed_peak = _peak_rss_bytes()

        for layer_index, a_key in enumerate(layer_keys, start=1):
            b_key = a_key.replace("lora_A", "lora_B", 1)
            a_tensors = [state[a_key] for state in states]
            b_tensors = [state[b_key] for state in states]
            for a_tensor, b_tensor in zip(a_tensors, b_tensors):
                self._validate_pair(a_key, a_tensor, b_key, b_tensor)
            self._validate_plain_layer_metadata(a_key, a_tensors, b_key, b_tensors)

            out_features = b_tensors[0].shape[0]
            in_features = a_tensors[0].shape[1]
            rows, cols = self._skeleton_indices(out_features, in_features)
            partition = build_partition(
                out_features,
                in_features,
                self.config.mode,
                self.config.ratio,
            )
            blocks = build_blocks(
                out_features,
                in_features,
                partition,
                self.config.skeleton,
                skeleton_rows=rows,
                skeleton_cols=cols,
                max_slots=self.config.poly_modulus_degree // 2,
            )
            if self.config.skeleton:
                layer_buffer_bytes = (
                    out_features * cols.size + rows.size * in_features
                ) * np.dtype(np.float32).itemsize
                self._check_memory_budget(
                    a_key,
                    "分配 CUR 交叉矩阵前",
                    reserve_bytes=layer_buffer_bytes,
                )
                cross_columns = np.zeros((out_features, cols.size), dtype=np.float32)
                cross_rows = np.zeros((rows.size, in_features), dtype=np.float32)
                row_positions = {int(value): index for index, value in enumerate(rows)}
                col_positions = {int(value): index for index, value in enumerate(cols)}
                product = None
            else:
                layer_buffer_bytes = (
                    out_features
                    * in_features
                    * np.dtype(np.float32).itemsize
                )
                self._check_memory_budget(
                    a_key,
                    "分配完整结果矩阵前",
                    reserve_bytes=layer_buffer_bytes,
                )
                product = np.zeros((out_features, in_features), dtype=np.float32)
                cross_columns = cross_rows = None
                row_positions = col_positions = {}

            self._emit(
                progress,
                event="layer_start",
                layer=a_key,
                layer_index=layer_index,
                layer_count=len(layer_keys),
                block_count=len(blocks),
                rss_bytes=_current_rss_bytes(),
                peak_rss_bytes=_peak_rss_bytes(),
            )
            for block_index, block in enumerate(blocks, start=1):
                self._check_memory_budget(a_key, f"开始 block {block_index} 前")
                accumulated = None
                for (client_id, _), a_tensor, b_tensor in zip(
                    ordered_plaintexts, a_tensors, b_tensors
                ):
                    iterator = iter_block_uploads(
                        b_tensor.numpy(),
                        a_tensor.numpy(),
                        self.public_context,
                        self.rank,
                        block,
                    )
                    while True:
                        encrypt_started = time.perf_counter()
                        try:
                            upload = next(iterator)
                        except StopIteration:
                            break
                        client_stats[client_id]["encrypt_time"] += (
                            time.perf_counter() - encrypt_started
                        )
                        client_stats[client_id]["ciphertext_size"] += _upload_size(upload)
                        accumulated = accumulate_block(
                            accumulated,
                            upload,
                            self.public_context,
                        )
                        del upload

                block_result = finalize_block(accumulated, block, self.num_clients)
                del accumulated
                values = decrypt_block(block_result, self.secret_context)
                block_matrix = values.reshape(
                    block.col_indices.size,
                    block.row_indices.size,
                ).T
                if product is not None:
                    product[np.ix_(block.row_indices, block.col_indices)] = block_matrix
                else:
                    if block.cols_selected:
                        selected_col_positions = [
                            col_positions[int(value)] for value in block.col_indices
                        ]
                        cross_columns[
                            np.ix_(block.row_indices, selected_col_positions)
                        ] = block_matrix
                    if block.rows_selected:
                        selected_row_positions = [
                            row_positions[int(value)] for value in block.row_indices
                        ]
                        cross_rows[
                            np.ix_(selected_row_positions, block.col_indices)
                        ] = block_matrix
                del block_result, values, block_matrix

                observed_peak = max(observed_peak, _peak_rss_bytes())
                if self._should_report_block(block_index, len(blocks)):
                    self._emit(
                        progress,
                        event="block_complete",
                        layer=a_key,
                        layer_index=layer_index,
                        layer_count=len(layer_keys),
                        block_index=block_index,
                        block_count=len(blocks),
                        rss_bytes=_current_rss_bytes(),
                        peak_rss_bytes=observed_peak,
                    )

            if product is None:
                reconstruction_reserve = (
                    out_features
                    * in_features
                    * np.dtype(np.float64).itemsize
                )
                self._check_memory_budget(
                    a_key,
                    "CUR 重建前",
                    reserve_bytes=reconstruction_reserve,
                )
                product, ok, reconstruction_stats = cur_reconstruct_with_stats(
                    cross_columns,
                    cross_rows,
                    rows,
                    cols,
                    condition_threshold=self.config.cur_condition_threshold,
                )
                if not ok or product is None:
                    raise RuntimeError(
                        f"{a_key} 的 CUR 重建失败: "
                        f"{reconstruction_stats['failure_reason']}"
                    )
            product_array = np.asarray(product, dtype=np.float32)
            del product
            product_tensor = torch.from_numpy(product_array)
            self._check_memory_budget(a_key, "低秩分解前")
            b_new, a_new = factorize_lora_product(product_tensor, self.rank)
            output[a_key] = a_new.to(dtype=a_tensors[0].dtype)
            output[b_key] = b_new.to(dtype=b_tensors[0].dtype)
            del product_array, product_tensor, blocks, a_tensors, b_tensors
            del b_new, a_new, partition, rows, cols
            if self.config.skeleton:
                del cross_columns, cross_rows
            observed_peak = max(observed_peak, _peak_rss_bytes())
            self._emit(
                progress,
                event="layer_complete",
                layer=a_key,
                layer_index=layer_index,
                layer_count=len(layer_keys),
                rss_bytes=_current_rss_bytes(),
                peak_rss_bytes=observed_peak,
            )

        return output, {
            "strategy": self.config.memory_strategy,
            "clients": client_stats,
            "peak_rss_bytes": observed_peak,
        }

    def _skeleton_indices(self, out_features: int, in_features: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.config.skeleton:
            return np.arange(out_features), np.arange(in_features)
        return select_uniform_rect_indices(
            out_features,
            in_features,
            self.config.skeleton_rank,
        )

    def _reconstruct_product(self, a_key: str, product: np.ndarray, layer: dict) -> np.ndarray:
        rows = np.asarray(layer["skeleton_rows"], dtype=int)
        cols = np.asarray(layer["skeleton_cols"], dtype=int)
        reconstructed, ok, stats = cur_reconstruct_with_stats(
            product[:, cols],
            product[rows, :],
            rows,
            cols,
            condition_threshold=self.config.cur_condition_threshold,
        )
        if not ok or reconstructed is None:
            raise RuntimeError(
                f"{a_key} 的 CUR 重建失败: {stats['failure_reason']}"
            )
        return reconstructed

    def _validate_pair(
        self,
        a_key: str,
        a_tensor: torch.Tensor,
        b_key: str,
        b_tensor: torch.Tensor,
    ) -> None:
        if a_tensor.ndim != 2 or b_tensor.ndim != 2:
            raise ValueError(
                f"LoRA A/B 必须为二维: {a_key}={tuple(a_tensor.shape)}, "
                f"{b_key}={tuple(b_tensor.shape)}"
            )
        if a_tensor.shape[0] != self.rank or b_tensor.shape[1] != self.rank:
            raise ValueError(
                f"LoRA rank 不匹配: {a_key}={tuple(a_tensor.shape)}, "
                f"{b_key}={tuple(b_tensor.shape)}, rank={self.rank}"
            )
        max_slots = self.config.poly_modulus_degree // 2
        out_features = b_tensor.shape[0]
        if out_features > max_slots:
            raise ValueError(
                f"{b_key} 输出维度 {out_features} 超过 CKKS 槽位数 {max_slots}"
            )

    def _validate_packages(self, ciphertexts: list[tuple[int, dict]], round_id: int) -> list[dict]:
        if len(ciphertexts) != self.num_clients:
            raise ValueError(
                f"密文客户端数应为 {self.num_clients}，实际为 {len(ciphertexts)}"
            )
        ordered = sorted(ciphertexts, key=lambda item: item[0])
        client_ids = [client_id for client_id, _ in ordered]
        if client_ids != list(range(self.num_clients)):
            raise ValueError(f"客户端 ID 必须连续且唯一，实际为 {client_ids}")
        packages = []
        for client_id, package in ordered:
            if package.get("protocol") != self.protocol:
                raise ValueError(f"client {client_id} 的密文协议不匹配")
            if package.get("client_id") != client_id or package.get("round_id") != round_id:
                raise ValueError(f"client {client_id} 的密文身份或轮次不匹配")
            packages.append(package)
        layer_keys = set(packages[0]["layers"])
        for package in packages[1:]:
            if set(package["layers"]) != layer_keys:
                raise ValueError("各客户端的 LoRA A/B 集合不一致")
        return packages

    def _validate_plaintexts(
        self,
        plaintexts: list[tuple[int, dict[str, torch.Tensor]]],
    ) -> list[dict[str, torch.Tensor]]:
        if len(plaintexts) != self.num_clients:
            raise ValueError(
                f"明文客户端数应为 {self.num_clients}，实际为 {len(plaintexts)}"
            )
        ordered = sorted(plaintexts, key=lambda item: item[0])
        client_ids = [client_id for client_id, _ in ordered]
        if client_ids != list(range(self.num_clients)):
            raise ValueError(f"客户端 ID 必须连续且唯一，实际为 {client_ids}")
        states = [state for _, state in ordered]
        if not states or not any("lora_A" in key for key in states[0]):
            raise ValueError("state_dict 中没有 LoRA A/B 参数")
        expected_keys = set(states[0])
        for state in states:
            if set(state) != expected_keys:
                raise ValueError("各客户端的 LoRA A/B 集合不一致")
            unsupported = [
                key for key in state
                if "lora_A" not in key and "lora_B" not in key
            ]
            if unsupported:
                raise ValueError(f"存在未配对或不支持加密的参数: {unsupported}")
            for a_key in (key for key in state if "lora_A" in key):
                b_key = a_key.replace("lora_A", "lora_B", 1)
                if b_key not in state:
                    raise KeyError(f"缺少与 {a_key} 配对的 {b_key}")
        return states

    @staticmethod
    def _validate_plain_layer_metadata(a_key, a_tensors, b_key, b_tensors) -> None:
        expected = (a_tensors[0].shape, a_tensors[0].dtype,
                    b_tensors[0].shape, b_tensors[0].dtype)
        for a_tensor, b_tensor in zip(a_tensors[1:], b_tensors[1:]):
            actual = (a_tensor.shape, a_tensor.dtype, b_tensor.shape, b_tensor.dtype)
            if actual != expected:
                raise ValueError(f"各客户端的 {a_key}/{b_key} 元数据不一致")

    def _check_memory_budget(
        self,
        a_key: str,
        stage: str,
        *,
        reserve_bytes: int = 0,
    ) -> None:
        limit = self.config.max_rss_bytes
        if limit is None:
            return
        current = _current_rss_bytes()
        projected = current + reserve_bytes
        if projected > limit:
            raise MemoryError(
                f"{a_key} {stage}的预计 RSS {_format_bytes(projected)} "
                f"超过 encryption.memory.max_rss_gb={_format_bytes(limit)} "
                f"(current={_format_bytes(current)}, reserve={_format_bytes(reserve_bytes)})"
            )

    def _should_report_block(self, block_index: int, block_count: int) -> bool:
        interval = self.config.progress_interval_blocks
        return block_index == 1 or block_index == block_count or block_index % interval == 0

    @staticmethod
    def _emit(progress, **event) -> None:
        if progress is not None:
            progress(event)

    @staticmethod
    def _validate_layer_metadata(a_key: str, layers: list[dict]) -> None:
        first = layers[0]
        expected = (
            first["b_key"],
            first["a_dtype"],
            first["b_dtype"],
            first["skeleton_rows"],
            first["skeleton_cols"],
            first["upload"]["shape"],
        )
        for layer in layers[1:]:
            actual = (
                layer["b_key"],
                layer["a_dtype"],
                layer["b_dtype"],
                layer["skeleton_rows"],
                layer["skeleton_cols"],
                layer["upload"]["shape"],
            )
            if actual != expected:
                raise ValueError(f"各客户端的 {a_key} 元数据不一致")


def _parse_config(raw: dict) -> CryptoConfig:
    if raw.get("scheme") != "ckks":
        raise ValueError("encryption.scheme 必须为 ckks")
    mode = raw.get("mode", "full")
    if mode not in {"full", "partial_A", "partial_AB"}:
        raise ValueError(f"不支持的 encryption.mode: {mode}")
    ratio = raw.get("ratio")
    if mode in {"partial_A", "partial_AB"} and ratio is None:
        raise ValueError(f"encryption.mode={mode} 时必须配置 ratio")
    skeleton_rank = int(raw.get("skeleton_rank", 0))
    skeleton = bool(raw.get("skeleton", True))
    if skeleton and skeleton_rank <= 0:
        raise ValueError("启用 skeleton 时 skeleton_rank 必须为正整数")
    degree = int(raw["poly_modulus_degree"])
    if degree <= 0 or degree % 2:
        raise ValueError("poly_modulus_degree 必须为正偶数")
    memory = raw.get("memory", {})
    if not isinstance(memory, dict):
        raise ValueError("encryption.memory 必须为映射")
    memory_strategy = memory.get("strategy", "layer_block_stream")
    if memory_strategy != "layer_block_stream":
        raise ValueError(
            f"不支持的 encryption.memory.strategy: {memory_strategy}"
        )
    max_rss_gb = memory.get("max_rss_gb")
    if max_rss_gb is not None and float(max_rss_gb) <= 0:
        raise ValueError("encryption.memory.max_rss_gb 必须为正数")
    progress_interval_blocks = int(memory.get("progress_interval_blocks", 25))
    if progress_interval_blocks <= 0:
        raise ValueError("encryption.memory.progress_interval_blocks 必须为正整数")
    return CryptoConfig(
        mode=mode,
        ratio=None if ratio is None else float(ratio),
        skeleton=skeleton,
        skeleton_rank=skeleton_rank,
        poly_modulus_degree=degree,
        coeff_mod_bit_sizes=[int(value) for value in raw["coeff_mod_bit_sizes"]],
        global_scale=float(raw["global_scale"]),
        cur_condition_threshold=float(raw.get("cur_condition_threshold", 1e12)),
        memory_strategy=memory_strategy,
        max_rss_bytes=(
            None if max_rss_gb is None else int(float(max_rss_gb) * 1024 ** 3)
        ),
        progress_interval_blocks=progress_interval_blocks,
    )


def _upload_size(upload: dict[str, Any]) -> int:
    counter = _ByteCounter()
    pickle.Pickler(counter).dump(upload)
    return counter.size


class _ByteCounter:
    def __init__(self) -> None:
        self.size = 0

    def write(self, data: bytes) -> int:
        length = len(data)
        self.size += length
        return length


def _peak_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _current_rss_bytes() -> int:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/statm", encoding="ascii") as statm:
                resident_pages = int(statm.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
    return _peak_rss_bytes()


def _format_bytes(value: int) -> str:
    return f"{value / 1024 ** 3:.2f} GiB"
