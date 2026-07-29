"""Adapt SkeletonLoRA's CKKS protocol to Framework state dictionaries."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
import pickle
import resource
import sys
import threading
import time
from typing import Any, Callable

import numpy as np
import torch

from skeleton_crypto.fe_context import clone_context, create_secret_context, derive_public_context
from skeleton_crypto.fe_modes import build_partition
from skeleton_crypto.fe_outer_hybrid import (
    accumulate_block,
    accumulate_serialized,
    aggregate,
    build_blocks,
    decrypt_block,
    decrypt_result,
    encrypt_upload,
    finalize_block,
    iter_block_uploads,
    serialize_accumulated,
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
    max_workers: int
    worker_reserve_bytes: int
    max_system_memory_ratio: float


class _WorkerMemoryGate:
    """按 RSS 预算限制并行任务的实际活跃数量。"""

    def __init__(
        self,
        max_workers: int,
        memory_limit_bytes: int,
        worker_reserve_bytes: int,
        admission_limit_bytes: int | None = None,
    ) -> None:
        self.requested_workers = max_workers
        self.current_worker_limit = max_workers
        self.memory_limit_bytes = memory_limit_bytes
        self.admission_limit_bytes = min(
            memory_limit_bytes,
            (
                memory_limit_bytes
                if admission_limit_bytes is None
                else admission_limit_bytes
            ),
        )
        self.worker_reserve_bytes = max(1, int(worker_reserve_bytes))
        self._baseline_rss_bytes = _current_rss_bytes()
        self._condition = threading.Condition()
        self._active_workers = 0
        self._active_slots: set[int] = set()
        self._reserved_bytes = 0
        self.peak_active_workers = 0
        self.memory_wait_count = 0
        self.worker_downgrade_count = 0
        self.min_workers = max_workers
        self._cancelled = False

    @contextmanager
    def task(self, reserve_bytes: int):
        worker_slot = self._acquire(reserve_bytes, uses_worker=True)
        try:
            yield worker_slot
            self._ensure_within_limit()
        finally:
            self._release(reserve_bytes, uses_worker=True, worker_slot=worker_slot)

    @contextmanager
    def reserve(self, reserve_bytes: int):
        self._acquire(reserve_bytes, uses_worker=False)
        try:
            yield
            self._ensure_within_limit()
        finally:
            self._release(reserve_bytes, uses_worker=False, worker_slot=None)

    def checkpoint(self) -> None:
        """刷新 RSS 和 worker 上限，并在触及硬上限时终止计算。"""
        with self._condition:
            self._raise_if_cancelled()
            current = _current_rss_bytes()
            self._ensure_within_limit(current)
            self._refresh_worker_limit(current)

    @property
    def active_workers(self) -> int:
        with self._condition:
            return self._active_workers

    @property
    def reserved_bytes(self) -> int:
        with self._condition:
            return self._reserved_bytes

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()

    def _acquire(
        self,
        reserve_bytes: int,
        *,
        uses_worker: bool,
    ) -> int | None:
        reserve_bytes = max(0, int(reserve_bytes))
        # layer reserve 持有期间会等待子任务；准入时额外检查一个 worker
        # 配额，确保至少一个子任务始终可以继续推进。
        gate_worker_reserve = 0 if uses_worker else self.worker_reserve_bytes
        with self._condition:
            waited_for_memory = False
            while True:
                self._raise_if_cancelled()
                current = _current_rss_bytes()
                self._ensure_within_limit(current)
                self._refresh_worker_limit(current)
                committed = max(
                    current,
                    self._baseline_rss_bytes
                    + self._reserved_bytes
                    + self._active_workers * self.worker_reserve_bytes,
                )
                projected = (
                    committed
                    + reserve_bytes
                    + gate_worker_reserve
                    + int(uses_worker) * self.worker_reserve_bytes
                )
                worker_available = (
                    not uses_worker
                    or (
                        self._active_workers < self.current_worker_limit
                        and any(
                            slot not in self._active_slots
                            for slot in range(self.current_worker_limit)
                        )
                    )
                )
                memory_available = projected <= self.admission_limit_bytes
                progress_worker = (
                    uses_worker
                    and self._active_workers == 0
                    and projected <= self.memory_limit_bytes
                )
                if worker_available and (memory_available or progress_worker):
                    break
                if (
                    not memory_available
                    and self._reserved_bytes == 0
                    and self._active_workers == 0
                ):
                    raise MemoryError(
                        "单个任务的预计 RSS "
                        f"{_format_bytes(projected)} "
                        "超过并行准入上限 "
                        f"{_format_bytes(self.admission_limit_bytes)}"
                    )
                waited_for_memory = waited_for_memory or not memory_available
                self._condition.wait(timeout=0.05)
            if waited_for_memory:
                self.memory_wait_count += 1
            worker_slot = None
            if uses_worker:
                worker_slot = next(
                    slot
                    for slot in range(self.current_worker_limit)
                    if slot not in self._active_slots
                )
                self._active_slots.add(worker_slot)
                self._active_workers += 1
            self._reserved_bytes += reserve_bytes
            self.peak_active_workers = max(
                self.peak_active_workers, self._active_workers
            )
            return worker_slot

    def _release(
        self,
        reserve_bytes: int,
        *,
        uses_worker: bool,
        worker_slot: int | None,
    ) -> None:
        reserve_bytes = max(0, int(reserve_bytes))
        with self._condition:
            if uses_worker:
                if worker_slot is None or worker_slot not in self._active_slots:
                    raise RuntimeError("并行内存门控 worker 槽位失衡")
                self._active_slots.remove(worker_slot)
                self._active_workers -= 1
            self._reserved_bytes -= reserve_bytes
            if (
                self._active_workers < 0
                or self._reserved_bytes < 0
            ):
                raise RuntimeError("并行内存门控计数失衡")
            self._condition.notify_all()

    def _refresh_worker_limit(self, current_rss_bytes: int) -> None:
        committed = max(
            current_rss_bytes,
            self._baseline_rss_bytes
            + self._reserved_bytes
            + self._active_workers * self.worker_reserve_bytes,
        )
        available = max(0, self.admission_limit_bytes - committed)
        additional_workers = int(available // self.worker_reserve_bytes)
        target = max(
            1,
            self._active_workers,
            min(
                self.requested_workers,
                self._active_workers + additional_workers,
            ),
        )
        if target < self.current_worker_limit:
            self.min_workers = min(self.min_workers, target)
            self.worker_downgrade_count += 1
        if target != self.current_worker_limit:
            self.current_worker_limit = target
            self._condition.notify_all()

    def _ensure_within_limit(self, current: int | None = None) -> None:
        current = _current_rss_bytes() if current is None else current
        if current > self.memory_limit_bytes:
            raise MemoryError(
                f"当前 RSS {_format_bytes(current)} 超过内存上限 "
                f"{_format_bytes(self.memory_limit_bytes)}"
            )

    def _raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise RuntimeError("CKKS 并行聚合已取消")


class _ThreadContexts:
    """为每个受门控的 worker 槽位延迟创建独立 CKKS context。"""

    def __init__(self, public_context, secret_context) -> None:
        self._public_context = public_context
        self._secret_context = secret_context
        self._public_contexts = {}
        self._secret_contexts = {}
        self._clone_lock = threading.Lock()

    def public(self, worker_slot: int):
        if worker_slot not in self._public_contexts:
            with self._clone_lock:
                if worker_slot not in self._public_contexts:
                    self._public_contexts[worker_slot] = clone_context(
                        self._public_context
                    )
        return self._public_contexts[worker_slot]

    def secret(self, worker_slot: int):
        if worker_slot not in self._secret_contexts:
            with self._clone_lock:
                if worker_slot not in self._secret_contexts:
                    self._secret_contexts[worker_slot] = clone_context(
                        self._secret_context
                    )
        return self._secret_contexts[worker_slot]


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
        """并行聚合层、块和客户端，并按内存预算限制实际活跃 worker。"""
        ordered_plaintexts = sorted(plaintexts, key=lambda item: item[0])
        states = self._validate_plaintexts(ordered_plaintexts)
        layer_keys = sorted(key for key in states[0] if "lora_A" in key)
        client_stats = {
            client_id: {"encrypt_time": 0.0, "ciphertext_size": 0}
            for client_id, _ in ordered_plaintexts
        }
        output: dict[str, torch.Tensor] = {}
        observed_peak = _peak_rss_bytes()
        memory_limit = self._effective_memory_limit()
        effective_workers = self._effective_worker_count(memory_limit)
        gate = _WorkerMemoryGate(
            effective_workers,
            memory_limit,
            self.config.worker_reserve_bytes,
            admission_limit_bytes=int(memory_limit * 0.9),
        )
        contexts = _ThreadContexts(self.public_context, self.secret_context)
        layer_workers = min(effective_workers, len(layer_keys))
        block_window = max(1, effective_workers // layer_workers)
        stop_event = threading.Event()
        self._emit(
            progress,
            event="parallel_start",
            requested_workers=self.config.max_workers,
            effective_workers=effective_workers,
            layer_workers=layer_workers,
            memory_limit_bytes=memory_limit,
            rss_bytes=_current_rss_bytes(),
            peak_rss_bytes=observed_peak,
        )

        layer_pool = ThreadPoolExecutor(
            max_workers=layer_workers,
            thread_name_prefix="ckks-layer",
        )
        block_pool = ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix="ckks-block",
        )
        client_pool = ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix="ckks-client",
        )
        layer_futures: dict[Future, str] = {}
        try:
            layer_futures = {
                layer_pool.submit(
                    self._aggregate_streaming_layer,
                    layer_index,
                    len(layer_keys),
                    a_key,
                    states,
                    ordered_plaintexts,
                    block_pool,
                    client_pool,
                    gate,
                    contexts,
                    block_window,
                    stop_event,
                    progress,
                ): a_key
                for layer_index, a_key in enumerate(layer_keys, start=1)
            }
            for future in as_completed(layer_futures):
                a_key, b_key, a_new, b_new, layer_stats = future.result()
                output[a_key] = a_new
                output[b_key] = b_new
                for client_id, values in layer_stats.items():
                    client_stats[client_id]["encrypt_time"] += values["encrypt_time"]
                    client_stats[client_id]["ciphertext_size"] += values["ciphertext_size"]
                observed_peak = max(observed_peak, _peak_rss_bytes())
        except BaseException:
            stop_event.set()
            gate.cancel()
            for future in layer_futures:
                future.cancel()
            raise
        finally:
            stop_event.set()
            gate.cancel()
            layer_pool.shutdown(wait=True, cancel_futures=True)
            block_pool.shutdown(wait=True, cancel_futures=True)
            client_pool.shutdown(wait=True, cancel_futures=True)

        return output, {
            "strategy": self.config.memory_strategy,
            "clients": client_stats,
            "peak_rss_bytes": observed_peak,
            "parallel": {
                "requested_workers": self.config.max_workers,
                "effective_workers": effective_workers,
                "peak_active_workers": gate.peak_active_workers,
                "minimum_workers": gate.min_workers,
                "worker_downgrade_count": gate.worker_downgrade_count,
                "memory_wait_count": gate.memory_wait_count,
                "memory_limit_bytes": memory_limit,
            },
        }

    def _aggregate_streaming_layer(
        self,
        layer_index,
        layer_count,
        a_key,
        states,
        ordered_plaintexts,
        block_pool,
        client_pool,
        gate,
        contexts,
        block_window,
        stop_event,
        progress,
    ):
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
        max_block_reserve = max(self._block_reserve_bytes(block) for block in blocks)
        if self.config.skeleton:
            layer_buffer_bytes = (
                out_features * cols.size + rows.size * in_features
            ) * np.dtype(np.float32).itemsize
        else:
            layer_buffer_bytes = (
                out_features * in_features * np.dtype(np.float32).itemsize
            )
        block_window_reserve = max_block_reserve * block_window
        if self.config.skeleton:
            reconstruction_reserve = self._reconstruction_reserve_bytes(
                out_features, in_features, rows.size, cols.size
            )
        else:
            reconstruction_reserve = 0
        factorization_reserve = self._factorization_reserve_bytes(
            out_features, in_features
        )
        layer_reserve = layer_buffer_bytes + block_window_reserve + max(
            reconstruction_reserve, factorization_reserve
        )
        with gate.reserve(layer_reserve):
            if self.config.skeleton:
                cross_columns = np.zeros((out_features, cols.size), dtype=np.float32)
                cross_rows = np.zeros((rows.size, in_features), dtype=np.float32)
                product = None
            else:
                product = np.zeros((out_features, in_features), dtype=np.float32)
                cross_columns = cross_rows = None
            return self._process_streaming_layer(
                layer_index,
                layer_count,
                a_key,
                b_key,
                a_tensors,
                b_tensors,
                ordered_plaintexts,
                rows,
                cols,
                blocks,
                product,
                cross_columns,
                cross_rows,
                block_pool,
                client_pool,
                gate,
                contexts,
                block_window,
                stop_event,
                progress,
            )

    def _process_streaming_layer(
        self,
        layer_index,
        layer_count,
        a_key,
        b_key,
        a_tensors,
        b_tensors,
        ordered_plaintexts,
        rows,
        cols,
        blocks,
        product,
        cross_columns,
        cross_rows,
        block_pool,
        client_pool,
        gate,
        contexts,
        block_window,
        stop_event,
        progress,
    ):
        row_positions = {int(value): index for index, value in enumerate(rows)}
        col_positions = {int(value): index for index, value in enumerate(cols)}
        self._emit(
            progress,
            event="layer_start",
            layer=a_key,
            layer_index=layer_index,
            layer_count=layer_count,
            block_count=len(blocks),
            rss_bytes=_current_rss_bytes(),
            peak_rss_bytes=_peak_rss_bytes(),
        )
        layer_stats = {
            client_id: {"encrypt_time": 0.0, "ciphertext_size": 0}
            for client_id, _ in ordered_plaintexts
        }
        pending: dict[Future, int] = {}
        next_block = 0
        try:
            while next_block < len(blocks) or pending:
                if stop_event.is_set():
                    raise RuntimeError("CKKS 并行聚合已取消")
                while next_block < len(blocks) and len(pending) < block_window:
                    block = blocks[next_block]
                    future = block_pool.submit(
                        self._aggregate_streaming_block,
                        block,
                        ordered_plaintexts,
                        a_tensors,
                        b_tensors,
                        client_pool,
                        gate,
                        contexts,
                        stop_event,
                    )
                    pending[future] = next_block + 1
                    next_block += 1
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    block_index = pending.pop(future)
                    block, block_matrix, block_stats = future.result()
                    if product is not None:
                        product[np.ix_(block.row_indices, block.col_indices)] = block_matrix
                    else:
                        if block.cols_selected:
                            selected_cols = [
                                col_positions[int(value)] for value in block.col_indices
                            ]
                            cross_columns[
                                np.ix_(block.row_indices, selected_cols)
                            ] = block_matrix
                        if block.rows_selected:
                            selected_rows = [
                                row_positions[int(value)] for value in block.row_indices
                            ]
                            cross_rows[
                                np.ix_(selected_rows, block.col_indices)
                            ] = block_matrix
                    for client_id, values in block_stats.items():
                        layer_stats[client_id]["encrypt_time"] += values["encrypt_time"]
                        layer_stats[client_id]["ciphertext_size"] += values["ciphertext_size"]
                    if self._should_report_block(block_index, len(blocks)):
                        self._emit(
                            progress,
                            event="block_complete",
                            layer=a_key,
                            layer_index=layer_index,
                            layer_count=layer_count,
                            block_index=block_index,
                            block_count=len(blocks),
                            rss_bytes=_current_rss_bytes(),
                            peak_rss_bytes=_peak_rss_bytes(),
                        )
        except BaseException:
            stop_event.set()
            for future in pending:
                future.cancel()
            raise

        if product is None:
            with gate.task(0):
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
        product_tensor = torch.from_numpy(product_array)
        with gate.task(0):
            b_new, a_new = factorize_lora_product(product_tensor, self.rank)
        a_new = a_new.to(dtype=a_tensors[0].dtype)
        b_new = b_new.to(dtype=b_tensors[0].dtype)
        self._emit(
            progress,
            event="layer_complete",
            layer=a_key,
            layer_index=layer_index,
            layer_count=layer_count,
            rss_bytes=_current_rss_bytes(),
            peak_rss_bytes=_peak_rss_bytes(),
        )
        return a_key, b_key, a_new, b_new, layer_stats

    def _aggregate_streaming_block(
        self,
        block,
        ordered_plaintexts,
        a_tensors,
        b_tensors,
        client_pool,
        gate,
        contexts,
        stop_event,
    ):
        if stop_event.is_set():
            raise RuntimeError("CKKS 并行聚合已取消")
        # layer 准入预算覆盖当前并发窗口中所有 block 的完整生命周期。
        client_futures = []
        for (client_id, _), a_tensor, b_tensor in zip(
            ordered_plaintexts, a_tensors, b_tensors
        ):
            client_futures.append(
                (
                    client_id,
                    client_pool.submit(
                        self._aggregate_client_block,
                        block,
                        a_tensor,
                        b_tensor,
                        gate,
                        contexts,
                        stop_event,
                    ),
                )
            )
        block_stats = {}
        try:
            for client_id, future in client_futures:
                serialized, values = future.result()
                block_stats[client_id] = {**values, "serialized": serialized}
        except BaseException:
            for _, future in client_futures:
                future.cancel()
            raise
        with gate.task(0) as worker_slot:
            public_context = contexts.public(worker_slot)
            accumulated = None
            for client_id, _ in client_futures:
                values = block_stats[client_id]
                accumulated = accumulate_serialized(
                    accumulated,
                    values.pop("serialized"),
                    public_context,
                )
                block_stats[client_id] = values
            block_result = finalize_block(accumulated, block, self.num_clients)
            values = decrypt_block(block_result, contexts.secret(worker_slot))
            block_matrix = values.reshape(
                block.col_indices.size, block.row_indices.size
            ).T.copy()
        return block, block_matrix, block_stats

    def _aggregate_client_block(
        self, block, a_tensor, b_tensor, gate, contexts, stop_event
    ):
        if stop_event.is_set():
            raise RuntimeError("CKKS 并行聚合已取消")
        # 外层 layer 任务已覆盖 block/client 结果的完整生命周期预算。
        with gate.task(0) as worker_slot:
            public_context = contexts.public(worker_slot)
            iterator = iter_block_uploads(
                b_tensor.numpy(),
                a_tensor.numpy(),
                public_context,
                self.rank,
                block,
            )
            accumulated = None
            encrypt_time = 0.0
            ciphertext_size = 0
            while True:
                encrypt_started = time.perf_counter()
                try:
                    upload = next(iterator)
                except StopIteration:
                    break
                gate.checkpoint()
                encrypt_time += time.perf_counter() - encrypt_started
                ciphertext_size += _upload_size(upload)
                accumulated = accumulate_block(accumulated, upload, public_context)
            return serialize_accumulated(accumulated), {
                "encrypt_time": encrypt_time,
                "ciphertext_size": ciphertext_size,
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

    def _effective_memory_limit(self) -> int:
        system_limit = int(
            _total_memory_bytes() * self.config.max_system_memory_ratio
        )
        if self.config.max_rss_bytes is None:
            return system_limit
        return min(system_limit, self.config.max_rss_bytes)

    def _effective_worker_count(self, memory_limit: int) -> int:
        available = memory_limit - _current_rss_bytes()
        if available < self.config.worker_reserve_bytes:
            raise MemoryError(
                "启动并行 CKKS 所需的单 worker 预留内存不足: "
                f"available={_format_bytes(max(0, available))}, "
                f"worker_reserve={_format_bytes(self.config.worker_reserve_bytes)}, "
                f"limit={_format_bytes(memory_limit)}"
            )
        memory_workers = max(1, available // self.config.worker_reserve_bytes)
        return min(self.config.max_workers, int(memory_workers))

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
