"""在独立进程中执行单层客户端上传与公钥聚合协议。"""

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
    finalize_block,
)
from skeleton_crypto.fe_skeleton import select_uniform_rect_indices


_RUNTIME: dict[str, Any] = {}


def initialize_process_worker(
    states: dict[str, list[np.ndarray]],
    config: dict[str, Any],
    rank: int,
    num_clients: int,
    public_context: bytes,
    progress_queue,
    memory_limit_bytes: int,
) -> None:
    """初始化只含公钥 CKKS context 的协议仿真进程。"""
    torch.set_num_threads(1)
    _RUNTIME.update(
        states=states,
        config=config,
        rank=rank,
        num_clients=num_clients,
        public_context=ts.context_from(public_context),
        progress_queue=progress_queue,
        memory_limit_bytes=memory_limit_bytes,
        peak_memory_bytes=_aggregate_memory_bytes(),
    )


def aggregate_layer_process(task: tuple[int, int, str]) -> dict[str, Any]:
    """依次生成客户端上传并执行公钥聚合，返回序列化下发 payload。"""
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

    client_stats = {
        client_id: {"encrypt_time": 0.0, "ciphertext_size": 0}
        for client_id in range(_RUNTIME["num_clients"])
    }
    aggregate_blocks = []
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
        aggregate_blocks.append(block_result)
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

    layer_payload = {
        "protocol": config["protocol"],
        "round_id": config["round_id"],
        "a_key": a_key,
        "b_key": b_key,
        "a_dtype": str(a_arrays[0].dtype),
        "b_dtype": str(b_arrays[0].dtype),
        "skeleton_rows": rows.tolist(),
        "skeleton_cols": cols.tolist(),
        "aggregate_result": {
            "shape": (out_features, in_features),
            "blocks": aggregate_blocks,
        },
    }
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
        "layer_payload": layer_payload,
        "download_size": _upload_size(layer_payload),
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
