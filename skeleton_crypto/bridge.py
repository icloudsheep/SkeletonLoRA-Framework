"""Adapt SkeletonLoRA's CKKS protocol to Framework state dictionaries."""

from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass
import math
import multiprocessing
import os
import pickle
import queue
import resource
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

from skeleton_crypto.fe_context import create_secret_context, derive_public_context
from skeleton_crypto.fe_modes import build_partition
from skeleton_crypto.fe_outer_hybrid import (
    aggregate,
    build_blocks,
    decrypt_result,
    encrypt_upload,
)
from skeleton_crypto.fe_skeleton import cur_reconstruct_with_stats, select_uniform_rect_indices
from skeleton_crypto.process_worker import (
    aggregate_layer_process,
    initialize_process_worker,
)
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
    max_workers: int
    worker_reserve_bytes: int
    max_system_memory_ratio: float


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
        """使用独立进程并行聚合各层，并限制进程总内存。"""
        ordered_plaintexts = sorted(plaintexts, key=lambda item: item[0])
        states = self._validate_plaintexts(ordered_plaintexts)
        layer_keys = sorted(key for key in states[0] if "lora_A" in key)
        for a_key in layer_keys:
            b_key = a_key.replace("lora_A", "lora_B", 1)
            a_tensors = [state[a_key] for state in states]
            b_tensors = [state[b_key] for state in states]
            for a_tensor, b_tensor in zip(a_tensors, b_tensors):
                self._validate_pair(a_key, a_tensor, b_key, b_tensor)
            self._validate_plain_layer_metadata(
                a_key, a_tensors, b_key, b_tensors
            )
        client_stats = {
            client_id: {"encrypt_time": 0.0, "ciphertext_size": 0}
            for client_id, _ in ordered_plaintexts
        }
        output: dict[str, torch.Tensor] = {}
        observed_peak = _aggregate_memory_bytes()
        memory_limit = self._effective_memory_limit()
        admission_limit = int(memory_limit * 0.9)
        process_states = {
            key: [state[key].detach().cpu().numpy() for state in states]
            for key in states[0]
        }
        worker_reserve = self._process_worker_reserve_bytes(process_states)
        cpu_resources = _cpu_resources()
        effective_workers = min(
            len(layer_keys),
            self._effective_worker_count(
                memory_limit,
                worker_reserve_bytes=worker_reserve,
                cpu_limit=cpu_resources[3],
            ),
        )
        worker_config = {
            "mode": self.config.mode,
            "ratio": self.config.ratio,
            "skeleton": self.config.skeleton,
            "skeleton_rank": self.config.skeleton_rank,
            "poly_modulus_degree": self.config.poly_modulus_degree,
            "cur_condition_threshold": self.config.cur_condition_threshold,
            "progress_interval_blocks": self.config.progress_interval_blocks,
        }
        public_context = self.public_context.serialize()
        secret_context = self.secret_context.serialize(save_secret_key=True)
        self._emit(
            progress,
            event="parallel_start",
            backend="process",
            requested_workers=self.config.max_workers,
            effective_workers=effective_workers,
            layer_workers=effective_workers,
            worker_reserve_bytes=worker_reserve,
            admission_limit_bytes=admission_limit,
            memory_limit_bytes=memory_limit,
            rss_bytes=_aggregate_memory_bytes(),
            peak_rss_bytes=observed_peak,
            host_cpu_count=cpu_resources[0],
            affinity_cpu_count=cpu_resources[1],
            cpu_quota_label=cpu_resources[2],
        )
        process_context = multiprocessing.get_context("spawn")
        progress_queue = process_context.Queue()
        layer_pool = ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=process_context,
            initializer=initialize_process_worker,
            initargs=(
                process_states,
                worker_config,
                self.rank,
                self.num_clients,
                public_context,
                secret_context,
                progress_queue,
                memory_limit,
            ),
        )
        layer_tasks = list(enumerate(layer_keys, start=1))
        layer_futures: dict[Future, str] = {}
        next_layer = 0
        minimum_workers = effective_workers
        peak_active_workers = 0
        peak_inflight_layers = 0
        worker_downgrade_count = 0
        memory_wait_count = 0
        active_worker_pids: set[int] = set()

        def drain_progress() -> None:
            nonlocal observed_peak, peak_active_workers
            while True:
                try:
                    event = progress_queue.get_nowait()
                except queue.Empty:
                    return
                observed_peak = max(observed_peak, event["peak_rss_bytes"])
                worker_pid = event.get("worker_pid")
                if event["event"] == "layer_start" and worker_pid is not None:
                    active_worker_pids.add(worker_pid)
                    peak_active_workers = max(
                        peak_active_workers, len(active_worker_pids)
                    )
                elif event["event"] == "layer_complete" and worker_pid is not None:
                    active_worker_pids.discard(worker_pid)
                self._emit(progress, **event)

        def dispatch_layers() -> None:
            nonlocal next_layer, minimum_workers, peak_inflight_layers
            nonlocal worker_downgrade_count
            current = _aggregate_memory_bytes()
            available = max(0, admission_limit - current)
            allowed = max(1, min(effective_workers, available // worker_reserve))
            allowed = int(allowed)
            if allowed < minimum_workers:
                minimum_workers = allowed
                worker_downgrade_count += 1
            while next_layer < len(layer_tasks) and len(layer_futures) < allowed:
                layer_index, a_key = layer_tasks[next_layer]
                future = layer_pool.submit(
                    aggregate_layer_process,
                    (layer_index, len(layer_keys), a_key),
                )
                layer_futures[future] = a_key
                next_layer += 1
            peak_inflight_layers = max(peak_inflight_layers, len(layer_futures))

        try:
            dispatch_layers()
            while layer_futures:
                completed, _ = wait(
                    layer_futures,
                    timeout=0.1,
                    return_when=FIRST_COMPLETED,
                )
                drain_progress()
                for future in completed:
                    layer_futures.pop(future)
                    result = future.result()
                    a_key = result["a_key"]
                    b_key = result["b_key"]
                    output[a_key] = torch.from_numpy(result["a_new"]).to(
                        dtype=states[0][a_key].dtype
                    )
                    output[b_key] = torch.from_numpy(result["b_new"]).to(
                        dtype=states[0][b_key].dtype
                    )
                    for client_id, values in result["client_stats"].items():
                        client_id = int(client_id)
                        client_stats[client_id]["encrypt_time"] += values["encrypt_time"]
                        client_stats[client_id]["ciphertext_size"] += values["ciphertext_size"]
                    observed_peak = max(
                        observed_peak,
                        result["peak_memory_bytes"],
                        _aggregate_memory_bytes(),
                    )
                if completed:
                    dispatch_layers()
                elif _aggregate_memory_bytes() >= admission_limit:
                    memory_wait_count += 1
            for _ in range(10):
                drain_progress()
                if progress_queue.empty():
                    break
                time.sleep(0.01)
        except BaseException:
            for future in layer_futures:
                future.cancel()
            raise
        finally:
            layer_pool.shutdown(wait=True, cancel_futures=True)
            progress_queue.close()
            progress_queue.join_thread()

        observed_peak = max(observed_peak, _aggregate_memory_bytes())
        return output, {
            "strategy": self.config.memory_strategy,
            "clients": client_stats,
            "peak_rss_bytes": observed_peak,
            "parallel": {
                "backend": "process",
                "requested_workers": self.config.max_workers,
                "effective_workers": effective_workers,
                "peak_active_workers": peak_active_workers,
                "peak_inflight_layers": peak_inflight_layers,
                "minimum_workers": minimum_workers,
                "worker_downgrade_count": worker_downgrade_count,
                "memory_wait_count": memory_wait_count,
                "worker_reserve_bytes": worker_reserve,
                "admission_limit_bytes": admission_limit,
                "memory_limit_bytes": memory_limit,
            },
        }

    def _block_reserve_bytes(self, block) -> int:
        elements = block.row_indices.size * block.col_indices.size
        if block.encrypted:
            coefficient_bits = sum(self.config.coeff_mod_bit_sizes)
            # CKKS 密文即使只使用少量槽位，仍保留完整多项式。
            ciphertext_bytes = (
                self.config.poly_modulus_degree * coefficient_bits * 4 // 8
            )
            result_bytes = max(elements * 16, ciphertext_bytes)
        else:
            result_bytes = max(1, elements * 16)
        # 预留每个客户端的序列化结果和一份块级聚合结果。
        return result_bytes * (self.num_clients + 1)

    @staticmethod
    def _reconstruction_reserve_bytes(
        out_features: int, in_features: int, rows: int, cols: int
    ) -> int:
        float64 = np.dtype(np.float64).itemsize
        # CUR 的输入转换、交叉块求逆和完整输出可能同时存在。
        return (
            out_features * cols
            + rows * in_features
            + rows * cols
            + out_features * in_features
        ) * float64

    def _factorization_reserve_bytes(
        self, out_features: int, in_features: int
    ) -> int:
        float32 = np.dtype(np.float32).itemsize
        # SVD 保留输入、奇异向量及新 A/B 的上界，按目标 rank 保守估算。
        return (
            2 * out_features * in_features
            + 2 * (out_features * self.rank + self.rank * in_features)
        ) * float32

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

    def _effective_memory_limit(self) -> int:
        system_limit = int(
            _total_memory_bytes() * self.config.max_system_memory_ratio
        )
        if self.config.max_rss_bytes is None:
            return system_limit
        return min(system_limit, self.config.max_rss_bytes)

    def _effective_worker_count(
        self,
        memory_limit: int,
        *,
        worker_reserve_bytes: int | None = None,
        cpu_limit: int | None = None,
    ) -> int:
        reserve = (
            self.config.worker_reserve_bytes
            if worker_reserve_bytes is None
            else int(worker_reserve_bytes)
        )
        available = memory_limit - _aggregate_memory_bytes()
        if available < reserve:
            raise MemoryError(
                "启动并行 CKKS 所需的单 worker 预留内存不足: "
                f"available={_format_bytes(max(0, available))}, "
                f"worker_reserve={_format_bytes(reserve)}, "
                f"limit={_format_bytes(memory_limit)}"
            )
        memory_workers = max(1, available // reserve)
        limits = [self.config.max_workers, int(memory_workers)]
        if cpu_limit is not None:
            limits.append(cpu_limit)
        return min(limits)

    def _process_worker_reserve_bytes(
        self,
        states: dict[str, list[np.ndarray]],
    ) -> int:
        state_bytes = sum(array.nbytes for arrays in states.values() for array in arrays)
        largest_layer_reserve = 0
        for a_key in (key for key in states if "lora_A" in key):
            b_key = a_key.replace("lora_A", "lora_B", 1)
            out_features = states[b_key][0].shape[0]
            in_features = states[a_key][0].shape[1]
            rows, cols = self._skeleton_indices(out_features, in_features)
            partition = build_partition(
                out_features, in_features, self.config.mode, self.config.ratio
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
            layer_buffer_bytes = (
                out_features * cols.size + rows.size * in_features
                if self.config.skeleton
                else out_features * in_features
            ) * np.dtype(np.float32).itemsize
            reconstruction_reserve = (
                self._reconstruction_reserve_bytes(
                    out_features, in_features, rows.size, cols.size
                )
                if self.config.skeleton
                else 0
            )
            factorization_reserve = self._factorization_reserve_bytes(
                out_features, in_features
            )
            layer_reserve = (
                layer_buffer_bytes
                + max(self._block_reserve_bytes(block) for block in blocks)
                + max(reconstruction_reserve, factorization_reserve)
            )
            largest_layer_reserve = max(largest_layer_reserve, layer_reserve)
        return self.config.worker_reserve_bytes + state_bytes + largest_layer_reserve

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
    max_rss_gb_value = None if max_rss_gb is None else float(max_rss_gb)
    if max_rss_gb_value is not None and (
        not math.isfinite(max_rss_gb_value) or max_rss_gb_value <= 0
    ):
        raise ValueError("encryption.memory.max_rss_gb 必须为有限正数")
    progress_interval_blocks = int(memory.get("progress_interval_blocks", 25))
    if progress_interval_blocks <= 0:
        raise ValueError("encryption.memory.progress_interval_blocks 必须为正整数")
    max_workers = int(memory.get("max_workers", min(32, os.cpu_count() or 1)))
    if max_workers <= 0:
        raise ValueError("encryption.memory.max_workers 必须为正整数")
    worker_reserve_mb = float(memory.get("worker_reserve_mb", 64))
    if not math.isfinite(worker_reserve_mb) or worker_reserve_mb <= 0:
        raise ValueError("encryption.memory.worker_reserve_mb 必须为有限正数")
    max_system_memory_ratio = float(memory.get("max_system_memory_ratio", 0.8))
    if not 0 < max_system_memory_ratio <= 0.8:
        raise ValueError(
            "encryption.memory.max_system_memory_ratio 必须在 (0, 0.8]"
        )
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
            None
            if max_rss_gb_value is None
            else int(max_rss_gb_value * 1024 ** 3)
        ),
        progress_interval_blocks=progress_interval_blocks,
        max_workers=max_workers,
        worker_reserve_bytes=int(worker_reserve_mb * 1024 ** 2),
        max_system_memory_ratio=max_system_memory_ratio,
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


def _aggregate_memory_bytes() -> int:
    """读取容器当前内存使用量；不可用时退回当前进程 RSS。"""
    if sys.platform.startswith("linux"):
        for path in (
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ):
            try:
                with open(path, encoding="ascii") as usage_file:
                    return int(usage_file.read().strip())
            except (OSError, ValueError):
                continue
    return _current_rss_bytes()


def _cpu_resources() -> tuple[int, int, str, int]:
    """返回主机、CPU affinity、cgroup quota 和有效 CPU 数。"""
    host_cpu_count = max(1, os.cpu_count() or 1)
    try:
        affinity_cpu_count = max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        affinity_cpu_count = host_cpu_count

    quota_cores, quota_label = _cgroup_cpu_quota()
    if quota_cores is None:
        return (
            host_cpu_count,
            affinity_cpu_count,
            quota_label,
            affinity_cpu_count,
        )
    quota_limit = max(1, math.ceil(quota_cores))
    return (
        host_cpu_count,
        affinity_cpu_count,
        f"{quota_cores:.2f} cores",
        min(affinity_cpu_count, quota_limit),
    )


def _cgroup_cpu_quota() -> tuple[float | None, str]:
    """读取 cgroup v2 或 v1 的 CPU quota 和检测状态。"""
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="ascii") as quota_file:
            quota_raw, period_raw = quota_file.read().strip().split()
        if quota_raw == "max":
            return None, "unlimited"
        quota = int(quota_raw)
        period = int(period_raw)
        if quota > 0 and period > 0:
            cores = quota / period
            return cores, f"{cores:.2f} cores"
    except (OSError, ValueError):
        pass

    try:
        with open(
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="ascii"
        ) as quota_file:
            quota = int(quota_file.read().strip())
        with open(
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="ascii"
        ) as period_file:
            period = int(period_file.read().strip())
        if quota > 0 and period > 0:
            cores = quota / period
            return cores, f"{cores:.2f} cores"
        if quota < 0:
            return None, "unlimited"
    except (OSError, ValueError):
        pass
    return None, "unavailable"


def _total_memory_bytes() -> int:
    limits = []
    try:
        limits.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        pass
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path, encoding="ascii") as limit_file:
                raw = limit_file.read().strip()
            if raw != "max":
                value = int(raw)
                if value > 0:
                    limits.append(value)
        except (OSError, ValueError):
            continue
    if not limits:
        raise RuntimeError("无法读取系统总内存，不能建立 80% 并行内存上限")
    return min(limits)


def _format_bytes(value: int) -> str:
    return f"{value / 1024 ** 3:.2f} GiB"
