"""Canonical LoRA A/B pairing at the Python/native boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch


@dataclass(frozen=True)
class LayerSpec:
    layer_id: int
    a_key: str
    b_key: str
    rank: int
    rows: int
    cols: int
    dtype: torch.dtype

    @property
    def name(self) -> str:
        return self.a_key.replace("lora_A", "lora")


@dataclass(frozen=True)
class LayerFactors:
    spec: LayerSpec
    a: torch.Tensor
    b: torch.Tensor


Manifest = Tuple[LayerSpec, ...]


def canonicalize_lora_state(
    state_dict: Dict[str, torch.Tensor],
    expected_rank: int,
) -> Tuple[Manifest, Tuple[LayerFactors, ...]]:
    a_keys = sorted(key for key in state_dict if "lora_A" in key)
    if not a_keys:
        raise ValueError("LoRA state_dict contains no lora_A tensors")

    paired_keys = set()
    specs = []
    factors = []
    for layer_id, a_key in enumerate(a_keys):
        b_key = a_key.replace("lora_A", "lora_B", 1)
        if b_key not in state_dict:
            raise KeyError(f"Missing B tensor paired with {a_key}: {b_key}")

        a = state_dict[a_key]
        b = state_dict[b_key]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"LoRA tensors must be matrices: {a_key}, {b_key}")
        if a.shape[0] != expected_rank or b.shape[1] != expected_rank:
            raise ValueError(
                f"Rank mismatch for {a_key}: A={tuple(a.shape)} "
                f"B={tuple(b.shape)} expected_rank={expected_rank}"
            )
        if b.shape[1] != a.shape[0]:
            raise ValueError(
                f"Incompatible LoRA pair {a_key}: A={tuple(a.shape)} "
                f"B={tuple(b.shape)}"
            )

        spec = LayerSpec(
            layer_id=layer_id,
            a_key=a_key,
            b_key=b_key,
            rank=expected_rank,
            rows=int(b.shape[0]),
            cols=int(a.shape[1]),
            dtype=a.dtype,
        )
        specs.append(spec)
        factors.append(
            LayerFactors(
                spec=spec,
                a=_native_tensor(a),
                b=_native_tensor(b),
            )
        )
        paired_keys.update((a_key, b_key))

    unpaired = set(state_dict) - paired_keys
    if unpaired:
        names = ", ".join(sorted(unpaired))
        raise ValueError(f"SecLoRA received unsupported non-LoRA tensors: {names}")
    return tuple(specs), tuple(factors)


def validate_manifest(expected: Manifest, received: Manifest) -> None:
    if expected == received:
        return
    expected_rows = tuple(_manifest_row(spec) for spec in expected)
    received_rows = tuple(_manifest_row(spec) for spec in received)
    raise ValueError(
        "LoRA layer manifest changed across clients or rounds: "
        f"expected={expected_rows}, received={received_rows}"
    )


def native_layer_payload(factors: Iterable[LayerFactors]) -> list[dict]:
    return [
        {
            "layer_id": item.spec.layer_id,
            "name": item.spec.name,
            "a": item.a.numpy(),
            "b": item.b.numpy(),
        }
        for item in factors
    ]


def _native_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _manifest_row(spec: LayerSpec) -> tuple:
    return spec.a_key, spec.b_key, spec.rows, spec.cols, spec.rank
