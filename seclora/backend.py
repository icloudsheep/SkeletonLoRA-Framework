"""Stateful bridge between the federated loop and native SecLoRA sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import time

import torch

from seclora.config import SecLoRAConfig
from seclora.low_rank import decode_skeleton
from seclora.native import create_native_session
from seclora.state import (
    Manifest,
    canonicalize_lora_state,
    native_layer_payload,
    validate_manifest,
)


@dataclass(frozen=True)
class EncryptedClientUpdate:
    client_id: int
    round_id: int
    payload: Any
    serialized_size_bytes: int
    metrics: Dict[str, Any]


class SecLoRABackend:
    def __init__(
        self,
        *,
        config: SecLoRAConfig,
        num_clients: int,
        rank: int,
        metrics_dir: Path,
        native_session: Any = None,
    ) -> None:
        self.config = config
        self.num_clients = num_clients
        self.rank = rank
        self.metrics_dir = metrics_dir
        self._manifest: Optional[Manifest] = None
        self._last_aggregate_metrics: Dict[str, Any] = {}
        self._last_layer_metrics: list[Dict[str, Any]] = []
        self._session = native_session or create_native_session(
            num_clients=num_clients,
            rank=rank,
            ratio=config.ratio,
            sfp=config.sfp,
            xmax=config.xmax,
            threads=config.threads,
            mode=config.mode,
        )

    def encrypt(
        self,
        state_dict: Dict[str, torch.Tensor],
        client_id: int,
        round_id: int,
    ) -> EncryptedClientUpdate:
        prepare_started = time.perf_counter()
        manifest, factors = canonicalize_lora_state(state_dict, self.rank)
        if self._manifest is None:
            self._manifest = manifest
        else:
            validate_manifest(self._manifest, manifest)
        layer_payload = native_layer_payload(factors)
        python_prepare_wall_sec = time.perf_counter() - prepare_started

        native_update = self._session.encrypt_client(
            client_id,
            round_id,
            layer_payload,
        )
        size = int(native_update.serialized_size_bytes)
        metrics = {
            "mode": self.config.mode,
            "ratio": self.config.ratio,
            "layer_count": len(manifest),
            "quantize_pack_wall_sec": (
                python_prepare_wall_sec
                + float(native_update.binding_input_copy_wall_sec)
                + float(native_update.quantize_pack_wall_sec)
            ),
            "precompute_wall_sec": float(native_update.precompute_wall_sec),
            "online_crypto_wall_sec": float(
                native_update.online_crypto_wall_sec
            ),
            "serialize_wall_sec": float(native_update.serialize_wall_sec),
            "sp_upload_bytes": int(native_update.sp_plain_bytes),
            "sd_upload_bytes": int(native_update.sd_cipher_bytes),
            "upload_bytes": size,
            "protected_b_labels": int(native_update.protected_b_labels),
            "protected_a_labels": int(native_update.protected_a_labels),
            "candidate_b_labels": int(native_update.candidate_b_labels),
            "candidate_a_labels": int(native_update.candidate_a_labels),
        }
        metrics["encrypted_scalars"] = self.rank * (
            metrics["protected_b_labels"]
            + metrics["protected_a_labels"]
            + metrics["candidate_b_labels"]
            + metrics["candidate_a_labels"]
        )
        metrics["client_online_wall_sec"] = (
            metrics["quantize_pack_wall_sec"]
            + metrics["online_crypto_wall_sec"]
            + metrics["serialize_wall_sec"]
        )
        metrics["client_total_crypto_wall_sec"] = (
            metrics["client_online_wall_sec"] + metrics["precompute_wall_sec"]
        )
        return EncryptedClientUpdate(
            client_id=client_id,
            round_id=round_id,
            payload=native_update,
            serialized_size_bytes=size,
            metrics=metrics,
        )

    def secure_aggregate(
        self,
        ciphertexts: Iterable[Tuple[int, EncryptedClientUpdate]],
        round_id: int,
    ) -> Dict[str, torch.Tensor]:
        manifest = self._require_manifest()
        ordered = sorted(ciphertexts, key=lambda item: item[0])
        client_ids = [client_id for client_id, _ in ordered]
        if client_ids != list(range(self.num_clients)):
            raise ValueError(
                f"Expected clients 0..{self.num_clients - 1}, got {client_ids}"
            )
        for client_id, update in ordered:
            if update.client_id != client_id or update.round_id != round_id:
                raise ValueError("Ciphertext metadata does not match aggregation round")

        native_layers = self._session.aggregate_round(
            round_id,
            [update.payload for _, update in ordered],
        )
        by_id = {int(layer.layer_id): layer for layer in native_layers}
        if set(by_id) != set(range(len(manifest))):
            raise ValueError(
                f"Native aggregate returned layer ids {sorted(by_id)}, "
                f"expected 0..{len(manifest) - 1}"
            )

        output: Dict[str, torch.Tensor] = {}
        layer_metrics: list[Dict[str, Any]] = []
        output_started = time.perf_counter()
        for spec in manifest:
            layer = by_id[spec.layer_id]
            b, a = decode_skeleton(
                layer.c,
                layer.m,
                layer.s,
                scale=self.config.scale,
                clients=self.num_clients,
                target_rank=self.rank,
                output_dtype=spec.dtype,
            )
            if tuple(a.shape) != (spec.rank, spec.cols):
                raise ValueError(f"Decoded A shape mismatch for {spec.a_key}")
            if tuple(b.shape) != (spec.rows, spec.rank):
                raise ValueError(f"Decoded B shape mismatch for {spec.b_key}")
            output[spec.a_key] = a.contiguous()
            output[spec.b_key] = b.contiguous()
            layer_metrics.append(
                {
                    "layer_id": spec.layer_id,
                    "layer_name": spec.name,
                    "rows": spec.rows,
                    "cols": spec.cols,
                    "selected_rank": int(layer.selected_rank),
                    "baseline_checks": int(layer.baseline_checks),
                    "baseline_relative_error": float(
                        layer.baseline_relative_error
                    ),
                    "decrypted_cells": int(layer.decrypted_cells),
                    "pivot_candidate_cells": int(
                        getattr(layer, "pivot_candidate_cells", 0)
                    ),
                    "download_c_bytes": int(layer.download_c_bytes),
                    "download_m_bytes": int(layer.download_m_bytes),
                    "download_s_bytes": int(layer.download_s_bytes),
                }
            )
        output_reconstruct_wall_sec = time.perf_counter() - output_started
        native_metrics = self._session.last_round_metrics
        sp_wall_sec = float(native_metrics.sp_wall_sec)
        sd_wall_sec = float(native_metrics.sd_wall_sec)
        dfe_mask_wall_sec = float(native_metrics.sd_dfe_mask_wall_sec)
        pairing_fe_wall_sec = float(native_metrics.sd_fe_eval_wall_sec)
        fe_aggregate_wall_sec = dfe_mask_wall_sec + pairing_fe_wall_sec
        bsgs_wall_sec = float(native_metrics.sd_bsgs_search_wall_sec)
        cur_skeleton_wall_sec = float(native_metrics.cur_skeleton_wall_sec)
        common_control_wall_sec = float(
            native_metrics.server_common_control_wall_sec
        )
        decrypt_wall_sec = (
            common_control_wall_sec
            + max(sp_wall_sec, sd_wall_sec)
            + cur_skeleton_wall_sec
        )
        server_critical_wall_sec = (
            decrypt_wall_sec + output_reconstruct_wall_sec
        )
        self._last_aggregate_metrics = {
            "mode": self.config.mode,
            "ratio": self.config.ratio,
            "sp_wall_sec": sp_wall_sec,
            "sd_wall_sec": sd_wall_sec,
            "sd_dfe_mask_wall_sec": dfe_mask_wall_sec,
            "sd_fe_eval_wall_sec": pairing_fe_wall_sec,
            "sd_bsgs_search_wall_sec": bsgs_wall_sec,
            "sd_control_wall_sec": float(native_metrics.sd_control_wall_sec),
            "fe_aggregate_wall_sec": fe_aggregate_wall_sec,
            "bsgs_wall_sec": bsgs_wall_sec,
            "cur_skeleton_wall_sec": cur_skeleton_wall_sec,
            "decrypt_wall_sec": decrypt_wall_sec,
            "cur_reconstruct_wall_sec": float(
                native_metrics.cur_reconstruct_wall_sec
            ),
            "experiment_verify_wall_sec": float(
                native_metrics.experiment_verify_wall_sec
            ),
            "server_common_control_wall_sec": common_control_wall_sec,
            "output_reconstruct_wall_sec": output_reconstruct_wall_sec,
            "observed_serial_server_wall_sec": float(
                native_metrics.observed_serial_server_wall_sec
            ) + output_reconstruct_wall_sec,
            "server_parallel_critical_wall_sec": server_critical_wall_sec,
            "protected_skeleton_cells": int(
                native_metrics.protected_skeleton_cells
            ),
            "pivot_candidate_cells": int(native_metrics.pivot_candidate_cells),
            "download_c_bytes_per_client": int(
                native_metrics.download_c_bytes_per_client
            ),
            "download_m_bytes_per_client": int(
                native_metrics.download_m_bytes_per_client
            ),
            "download_s_bytes_per_client": int(
                native_metrics.download_s_bytes_per_client
            ),
            "download_bytes_per_client": int(
                native_metrics.download_bytes_per_client
            ),
        }
        self._last_layer_metrics = layer_metrics
        return output

    @staticmethod
    def ciphertext_size(update: EncryptedClientUpdate) -> int:
        return update.serialized_size_bytes

    @property
    def last_aggregate_metrics(self) -> Dict[str, Any]:
        return dict(self._last_aggregate_metrics)

    @property
    def last_layer_metrics(self) -> list[Dict[str, Any]]:
        return [dict(row) for row in self._last_layer_metrics]

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if close is not None:
            close()

    def _require_manifest(self) -> Manifest:
        if self._manifest is None:
            raise RuntimeError("No client update has been encrypted")
        return self._manifest


def build_seclora_backend(
    root_config: dict,
    *,
    num_clients: int,
    rank: int,
    metrics_dir: Path,
) -> Optional[SecLoRABackend]:
    config = SecLoRAConfig.from_root(root_config)
    if config is None:
        return None
    return SecLoRABackend(
        config=config,
        num_clients=num_clients,
        rank=rank,
        metrics_dir=metrics_dir,
    )
