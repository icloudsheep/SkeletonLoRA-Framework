"""在独立进程中执行单层 CKKS 聚合。"""

from __future__ import annotations

import os
import pickle
import resource
import sys
import time
from typing import Any

import numpy as np
import tenseal as ts
import torch

from skeleton_crypto.fe_modes import build_partition
from skeleton_crypto.fe_outer_hybrid import (
    accumulate_block,
    build_blocks,
    decrypt_block,
    finalize_block,
)
from skeleton_crypto.fe_skeleton import (
    cur_reconstruct_with_stats,
    select_uniform_rect_indices,
)
from utils import factorize_lora_product


_RUNTIME: dict[str, Any] = {}


def initialize_process_worker(
    states: dict[str, list[np.ndarray]],
    config: dict[str, Any],
    rank: int,
    num_clients: int,
    public_context: bytes,
    secret_context: bytes,
    progress_queue,
    memory_limit_bytes: int,
) -> None:
    """初始化进程私有的模型状态和 CKKS context。"""
    torch.set_num_threads(1)
    _RUNTIME.update(
        states=states,
        config=config,
        rank=rank,
        num_clients=num_clients,
        public_context=ts.context_from(public_context),
        secret_context=ts.context_from(secret_context),
        progress_queue=progress_queue,
        memory_limit_bytes=memory_limit_bytes,
        peak_memory_bytes=_aggregate_memory_bytes(),
    )


def aggregate_layer_process(task: tuple[int, int, str]) -> dict[str, Any]:
    """完成一层的分块加密、聚合、解密和低秩分解。"""
    layer_wall_started = time.perf_counter()
    layer_cpu_started = time.process_time()
    layer_index, layer_count, a_key = task
    b_key = a_key.replace("lora_A", "lora_B", 1)
    states = _RUNTIME["states"]
    a_arrays = states[a_key]
    b_arrays = states[b_key]
    config = _RUNTIME["config"]
    out_features = b_arrays[0].shape[0]
    in_features = a_arrays[0].shape[1]

    _emit(
        event="layer_prepare_start",
        layer=a_key,
        layer_index=layer_index,
        layer_count=layer_count,
    )
    prepare_started = time.perf_counter()
    if config["skeleton"]:
        rows, cols = select_uniform_rect_indices(
            out_features, in_features, config["skeleton_rank"]
        )
    else:
        rows = np.arange(out_features)
        cols = np.arange(in_features)
    partition = build_partition(
        out_features, in_features, config["mode"], config["ratio"]
    )
    blocks = build_blocks(
        out_features,
        in_features,
        partition,
        config["skeleton"],
        skeleton_rows=rows,
        skeleton_cols=cols,
        max_slots=config["poly_modulus_degree"] // 2,
    )
    _emit(
        event="layer_prepare_complete",
        layer=a_key,
        layer_index=layer_index,
        layer_count=layer_count,
        block_count=len(blocks),
        prepare_time=time.perf_counter() - prepare_started,
    )
    _emit(
        event="layer_start",
        layer=a_key,
        layer_index=layer_index,
        layer_count=layer_count,
        block_count=len(blocks),
        layer_elapsed=0.0,
        worker_cpu_seconds=0.0,
        worker_cpu_utilization=0.0,
    )

    if config["skeleton"]:
        product = None
        cross_columns = np.zeros((out_features, cols.size), dtype=np.float32)
        cross_rows = np.zeros((rows.size, in_features), dtype=np.float32)
        row_positions = {int(value): index for index, value in enumerate(rows)}
        col_positions = {int(value): index for index, value in enumerate(cols)}
    else:
        product = np.zeros((out_features, in_features), dtype=np.float32)
        cross_columns = cross_rows = None
        row_positions = col_positions = {}

    client_stats = {
        client_id: {"encrypt_time": 0.0, "ciphertext_size": 0}
        for client_id in range(_RUNTIME["num_clients"])
    }
    for block_index, block in enumerate(blocks, start=1):
        _ensure_memory_limit()
        accumulated = None
        for client_id, (a_array, b_array) in enumerate(zip(a_arrays, b_arrays)):
            iterator = _iter_uploads(b_array, a_array, block)
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
                    accumulated, upload, _RUNTIME["public_context"]
                )
        block_result = finalize_block(accumulated, block, _RUNTIME["num_clients"])
        values = decrypt_block(block_result, _RUNTIME["secret_context"])
        block_matrix = values.reshape(
            block.col_indices.size, block.row_indices.size
        ).T
        if product is not None:
            product[np.ix_(block.row_indices, block.col_indices)] = block_matrix
        else:
            if block.cols_selected:
                selected_cols = [col_positions[int(value)] for value in block.col_indices]
                cross_columns[np.ix_(block.row_indices, selected_cols)] = block_matrix
            if block.rows_selected:
                selected_rows = [row_positions[int(value)] for value in block.row_indices]
                cross_rows[np.ix_(selected_rows, block.col_indices)] = block_matrix
        if _should_report(block_index, len(blocks), config["progress_interval_blocks"]):
            layer_elapsed = time.perf_counter() - layer_wall_started
            worker_cpu_seconds = time.process_time() - layer_cpu_started
            _emit(
                event="block_complete",
                layer=a_key,
                layer_index=layer_index,
                layer_count=layer_count,
                block_index=block_index,
                block_count=len(blocks),
                layer_elapsed=layer_elapsed,
                worker_cpu_seconds=worker_cpu_seconds,
                worker_cpu_utilization=(
                    100.0 * worker_cpu_seconds / max(layer_elapsed, 1e-9)
                ),
            )

    if product is None:
        product, ok, reconstruction_stats = cur_reconstruct_with_stats(
            cross_columns,
            cross_rows,
            rows,
            cols,
            condition_threshold=config["cur_condition_threshold"],
        )
        if not ok or product is None:
            raise RuntimeError(
                f"{a_key} 的 CUR 重建失败: "
                f"{reconstruction_stats['failure_reason']}"
            )

    product_tensor = torch.from_numpy(np.asarray(product, dtype=np.float32))
    b_new, a_new = factorize_lora_product(product_tensor, _RUNTIME["rank"])
    layer_elapsed = time.perf_counter() - layer_wall_started
    worker_cpu_seconds = time.process_time() - layer_cpu_started
    _emit(
        event="layer_complete",
        layer=a_key,
        layer_index=layer_index,
        layer_count=layer_count,
        layer_elapsed=layer_elapsed,
        worker_cpu_seconds=worker_cpu_seconds,
        worker_cpu_utilization=(
            100.0 * worker_cpu_seconds / max(layer_elapsed, 1e-9)
        ),
    )
    return {
        "a_key": a_key,
        "b_key": b_key,
        "a_new": a_new.numpy(),
        "b_new": b_new.numpy(),
        "client_stats": client_stats,
        "peak_memory_bytes": _RUNTIME["peak_memory_bytes"],
    }


def _iter_uploads(b_array, a_array, block):
    from skeleton_crypto.fe_outer_hybrid import iter_block_uploads

    return iter_block_uploads(
        b_array,
        a_array,
        _RUNTIME["public_context"],
        _RUNTIME["rank"],
        block,
    )


def _emit(**event: Any) -> None:
    current = _aggregate_memory_bytes()
    _RUNTIME["peak_memory_bytes"] = max(_RUNTIME["peak_memory_bytes"], current)
    _RUNTIME["progress_queue"].put(
        {
            **event,
            "worker_pid": os.getpid(),
            "rss_bytes": current,
            "peak_rss_bytes": _RUNTIME["peak_memory_bytes"],
        }
    )


def _ensure_memory_limit() -> None:
    current = _aggregate_memory_bytes()
    _RUNTIME["peak_memory_bytes"] = max(_RUNTIME["peak_memory_bytes"], current)
    if current > _RUNTIME["memory_limit_bytes"]:
        raise MemoryError(
            f"CKKS 进程总内存 {current / 1024 ** 3:.2f} GiB 超过硬上限 "
            f"{_RUNTIME['memory_limit_bytes'] / 1024 ** 3:.2f} GiB"
        )


def _aggregate_memory_bytes() -> int:
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
        try:
            with open("/proc/self/statm", encoding="ascii") as statm:
                resident_pages = int(statm.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


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


def _should_report(block_index: int, block_count: int, interval: int) -> bool:
    return block_index == 1 or block_index == block_count or block_index % interval == 0
