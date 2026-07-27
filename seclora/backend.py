"""Stateful bridge between the federated loop and the native SEL-2S session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

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
        self._session = native_session or create_native_session(
            num_clients=num_clients,
            rank=rank,
            ratio=config.ratio,
            sfp=config.sfp,
            xmax=config.xmax,
            threads=config.threads,
        )

    def encrypt(
        self,
        state_dict: Dict[str, torch.Tensor],
        client_id: int,
        round_id: int,
    ) -> EncryptedClientUpdate:
        manifest, factors = canonicalize_lora_state(state_dict, self.rank)
        if self._manifest is None:
            self._manifest = manifest
        else:
            validate_manifest(self._manifest, manifest)

        native_update = self._session.encrypt_client(
            client_id,
            round_id,
            native_layer_payload(factors),
        )
        size = int(native_update.serialized_size_bytes)
        return EncryptedClientUpdate(
            client_id=client_id,
            round_id=round_id,
            payload=native_update,
            serialized_size_bytes=size,
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
        return output

    @staticmethod
    def ciphertext_size(update: EncryptedClientUpdate) -> int:
        return update.serialized_size_bytes

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
