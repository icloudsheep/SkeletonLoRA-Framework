"""矩形外积 CKKS 的 full/partial 混合聚合协议。

本文件复制自 /Users/bilibili/Github/SkeletonLoRA/fe_outer_hybrid.py，仅调整包内导入方式。
"""

from dataclasses import dataclass

import numpy as np
import tenseal as ts

from skeleton_crypto.fe_modes import ModePartition


@dataclass
class Block:
    """一个结果矩形块及其坐标。"""

    row_indices: np.ndarray
    col_indices: np.ndarray
    rows_selected: bool
    cols_selected: bool
    b_encrypted: bool
    a_encrypted: bool

    @property
    def encrypted(self):
        return self.b_encrypted or self.a_encrypted


def _split(base, encrypted, selected, split_encryption):
    base = np.asarray(base, dtype=int)
    encrypted = set(int(item) for item in encrypted)
    selected = set(int(item) for item in selected)
    groups = []
    for selected_flag, selected_values in ((True, selected), (False, set())):
        output_indices = [int(item) for item in base if int(item) in selected_values]
        if not selected_flag:
            output_indices = [int(item) for item in base if int(item) not in selected]
        if not output_indices:
            continue
        if split_encryption:
            for encrypted_flag in (True, False):
                indices = [
                    item for item in output_indices if (item in encrypted) == encrypted_flag
                ]
                if indices:
                    groups.append((np.array(indices, dtype=int), selected_flag, encrypted_flag))
        else:
            encrypted_flag = bool(encrypted.intersection(output_indices))
            groups.append(
                (np.array(output_indices, dtype=int), selected_flag, encrypted_flag)
            )
    return groups


def _split_by_slots(indices, row_count, max_slots):
    if max_slots is None:
        return [np.asarray(indices, dtype=int)]
    if row_count <= 0 or row_count > max_slots:
        raise ValueError(
            f"结果块行数 {row_count} 无法放入 {max_slots} 个 CKKS 槽位"
        )
    group_size = max(1, max_slots // row_count)
    indices = np.asarray(indices, dtype=int)
    return [indices[start:start + group_size] for start in range(0, len(indices), group_size)]


def build_blocks(out_features, in_features, partition: ModePartition, skeleton,
                 skeleton_rows=None, skeleton_cols=None, max_slots=None):
    """构造不重叠的明文/密文结果块。"""
    all_rows = np.arange(out_features, dtype=int)
    all_cols = np.arange(in_features, dtype=int)
    selected_rows = all_rows if not skeleton else np.asarray(skeleton_rows, dtype=int)
    selected_cols = all_cols if not skeleton else np.asarray(skeleton_cols, dtype=int)
    row_groups = _split(
        all_rows,
        partition.encrypted_rows,
        selected_rows,
        split_encryption=partition.mode == "partial_AB",
    )
    col_groups = _split(
        all_cols,
        partition.encrypted_cols,
        selected_cols,
        split_encryption=partition.mode in {"partial_A", "partial_AB"},
    )
    blocks = []
    for rows, rows_selected, rows_encrypted in row_groups:
        for col_group, cols_selected, cols_encrypted in col_groups:
            for cols in _split_by_slots(col_group, rows.size, max_slots):
                if skeleton and not rows_selected and not cols_selected:
                    continue
                blocks.append(
                    Block(
                        row_indices=rows,
                        col_indices=cols,
                        rows_selected=rows_selected,
                        cols_selected=cols_selected,
                        b_encrypted=rows_encrypted,
                        a_encrypted=cols_encrypted,
                    )
                )
    if not blocks:
        raise ValueError("没有可聚合的结果块")
    if not skeleton:
        covered = sum(block.row_indices.size * block.col_indices.size for block in blocks)
        if covered != out_features * in_features:
            raise ValueError("结果块未覆盖完整矩阵")
    return blocks


def _encrypt_or_plain(values, encrypted, public_ctx):
    values = np.asarray(values, dtype=np.float64)
    if encrypted:
        return ("ct", ts.ckks_vector(public_ctx, values.tolist()).serialize())
    return ("plain", values)


def _block_term(B, A, block, component, public_ctx):
    rows = block.row_indices
    cols = block.col_indices
    vector = np.tile(B[rows, component], cols.size)
    scalar = np.repeat(np.asarray(A[component, cols], dtype=np.float64), rows.size)
    return (
        _encrypt_or_plain(vector, block.b_encrypted, public_ctx),
        _encrypt_or_plain(scalar, block.a_encrypted, public_ctx),
    )


def _block_terms(B, A, block, rank, public_ctx):
    return [
        _block_term(B, A, block, component, public_ctx)
        for component in range(rank)
    ]


def iter_block_uploads(B, A, public_ctx, rank, block):
    """逐项生成一个客户端的单块上传，限制 rank 展开数据的驻留数量。"""
    B = np.asarray(B)
    A = np.asarray(A)
    if B.ndim != 2 or A.ndim != 2 or B.shape[1] != rank or A.shape[0] != rank:
        raise ValueError(f"A/B 形状不符合 rank={rank}：A={A.shape}，B={B.shape}")
    if not block.encrypted:
        product = B[block.row_indices, :] @ A[:, block.col_indices]
        yield {
            "kind": "plain_product",
            "payload": np.asarray(product.T.reshape(-1), dtype=np.float64),
        }
        return
    for component in range(rank):
        yield {
            "kind": "term",
            "operands": _block_term(B, A, block, component, public_ctx),
        }


def encrypt_upload(B, A, public_ctx, rank, partition, blocks):
    """生成一个客户端的矩形混合上传包。"""
    B = np.asarray(B, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    if B.shape[1] != rank or A.shape[0] != rank or B.shape[0] == 0 or A.shape[1] == 0:
        raise ValueError(f"A/B 形状不符合 rank={rank}：A={A.shape}，B={B.shape}")
    return {
        "shape": (B.shape[0], A.shape[1]),
        "mode": partition.mode,
        "ratio": partition.ratio,
        "blocks": [
            {
                "row_indices": block.row_indices.tolist(),
                "col_indices": block.col_indices.tolist(),
                "terms": _block_terms(B, A, block, rank, public_ctx),
            }
            for block in blocks
        ],
    }


def _load_operand(operand, public_ctx):
    kind, payload = operand
    if kind == "ct":
        return ts.ckks_vector_from(public_ctx, payload)
    return np.asarray(payload, dtype=np.float64)


def _multiply(left, right):
    left_ct = isinstance(left, ts.CKKSVector)
    right_ct = isinstance(right, ts.CKKSVector)
    if left_ct:
        return left * right
    if right_ct:
        return right * left
    return left * right


def _serialize_result(value):
    if isinstance(value, ts.CKKSVector):
        return {"kind": "ct", "payload": value.serialize()}
    return {"kind": "plain", "payload": np.asarray(value, dtype=np.float64)}


def serialize_accumulated(value):
    """序列化单个 worker 的局部聚合值，隔离不同 worker 的 CKKS context。"""
    if value is None:
        raise ValueError("局部聚合状态不能为空")
    return _serialize_result(value)


def accumulate_serialized(accumulated, serialized, public_ctx):
    """将独立 worker 产生的序列化局部结果合并到当前 CKKS context。"""
    kind = serialized["kind"]
    if kind == "ct":
        current = ts.ckks_vector_from(public_ctx, serialized["payload"])
    elif kind == "plain":
        current = np.asarray(serialized["payload"], dtype=np.float64)
    else:
        raise ValueError(f"未知局部聚合类型: {kind}")
    if accumulated is None:
        return current
    if isinstance(accumulated, ts.CKKSVector):
        accumulated += current
    else:
        np.add(accumulated, current, out=accumulated)
    return accumulated


def accumulate_block(accumulated, upload, public_ctx):
    """把一个客户端的块上传累加到未解密的服务端块状态。"""
    if upload["kind"] == "plain_product":
        current = np.asarray(upload["payload"], dtype=np.float64)
    elif upload["kind"] == "term":
        left_raw, right_raw = upload["operands"]
        current = _multiply(
            _load_operand(left_raw, public_ctx),
            _load_operand(right_raw, public_ctx),
        )
    elif upload["kind"] == "terms":
        current = None
        for left_raw, right_raw in upload["terms"]:
            term = _multiply(
                _load_operand(left_raw, public_ctx),
                _load_operand(right_raw, public_ctx),
            )
            current = term if current is None else current + term
    else:
        raise ValueError(f"未知块上传类型: {upload['kind']}")
    if current is None:
        raise ValueError("块上传不包含可聚合数据")
    if accumulated is None:
        return current
    if isinstance(accumulated, ts.CKKSVector):
        accumulated += current
    else:
        np.add(accumulated, current, out=accumulated)
    return accumulated


def finalize_block(accumulated, block, n_clients):
    """完成一个块的客户端均值计算并序列化结果。"""
    if accumulated is None:
        raise ValueError("块聚合状态不能为空")
    if n_clients <= 0:
        raise ValueError("n_clients 必须为正整数")
    scale = 1.0 / n_clients
    if isinstance(accumulated, ts.CKKSVector):
        accumulated *= scale
    else:
        np.multiply(accumulated, scale, out=accumulated)
    return {
        "row_indices": block.row_indices.tolist(),
        "col_indices": block.col_indices.tolist(),
        **_serialize_result(accumulated),
    }


def decrypt_block(block_result, secret_ctx):
    """解密一个聚合块，返回按列连续排列的浮点数组。"""
    if block_result["kind"] == "ct":
        values = ts.ckks_vector_from(secret_ctx, block_result["payload"]).decrypt()
    elif block_result["kind"] == "plain":
        values = block_result["payload"]
    else:
        raise ValueError(f"未知聚合块类型: {block_result['kind']}")
    return np.asarray(values, dtype=np.float64)


def aggregate(uploads, public_ctx, n_clients):
    """服务端聚合每个结果块，不解密任何密文。"""
    if not uploads:
        raise ValueError("uploads 不能为空")
    results = []
    for block_index, block_upload in enumerate(uploads[0]["blocks"]):
        accumulated = None
        for upload in uploads:
            current = None
            for left_raw, right_raw in upload["blocks"][block_index]["terms"]:
                term = _multiply(
                    _load_operand(left_raw, public_ctx),
                    _load_operand(right_raw, public_ctx),
                )
                current = term if current is None else current + term
            accumulated = current if accumulated is None else accumulated + current
        accumulated = accumulated * (1.0 / n_clients)
        results.append(
            {
                "row_indices": block_upload["row_indices"],
                "col_indices": block_upload["col_indices"],
                **_serialize_result(accumulated),
            }
        )
    return {"shape": uploads[0]["shape"], "blocks": results}


def decrypt_result(aggregate_result, secret_ctx):
    """客户端合并明文块和解密后的密文块。"""
    shape = tuple(aggregate_result["shape"])
    matrix = np.zeros(shape, dtype=np.float64)
    ciphertext_count = 0
    plaintext_count = 0
    for block in aggregate_result["blocks"]:
        rows = np.asarray(block["row_indices"], dtype=int)
        cols = np.asarray(block["col_indices"], dtype=int)
        if block["kind"] == "ct":
            flat = np.asarray(ts.ckks_vector_from(secret_ctx, block["payload"]).decrypt())
            ciphertext_count += rows.size * cols.size
            for offset, col in enumerate(cols):
                matrix[rows, col] = flat[offset * rows.size:(offset + 1) * rows.size]
        else:
            values = np.asarray(block["payload"], dtype=np.float64)
            plaintext_count += rows.size * cols.size
            for offset, col in enumerate(cols):
                matrix[rows, col] = values[offset * rows.size:(offset + 1) * rows.size]
    return matrix, {
        "ciphertext_elements": ciphertext_count,
        "plaintext_elements": plaintext_count,
        "total_elements": int(np.prod(shape)),
    }
