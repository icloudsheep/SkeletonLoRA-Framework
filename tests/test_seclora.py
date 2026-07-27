"""Regression tests for the Python/native SecLoRA boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from seclora.backend import SecLoRABackend
from seclora.config import SecLoRAConfig
from seclora.low_rank import factorize_low_rank_product
from seclora.state import canonicalize_lora_state


class _FakeNativeUpdate:
    def __init__(self, client_id: int, round_id: int, layers: list[dict]) -> None:
        self.client_id = client_id
        self.round_id = round_id
        self.layers = layers
        self.serialized_size_bytes = sum(
            layer["a"].nbytes + layer["b"].nbytes for layer in layers
        )


class _FakeNativeLayer:
    def __init__(self, layer_id: int, product: torch.Tensor) -> None:
        rows = product.shape[0]
        self.layer_id = layer_id
        self.c = torch.eye(rows, dtype=torch.int64)
        self.m = torch.eye(rows, dtype=torch.int64)
        self.s = product.to(torch.int64)


class _FakeNativeSession:
    def __init__(self, scale: int, xmax: float) -> None:
        self.scale = scale
        self.xmax = xmax
        self.closed = False

    def encrypt_client(
        self,
        client_id: int,
        round_id: int,
        layers: list[dict],
    ) -> _FakeNativeUpdate:
        quantized = []
        for layer in layers:
            quantized.append(
                {
                    **layer,
                    "a": self._quantize(layer["a"]),
                    "b": self._quantize(layer["b"]),
                }
            )
        return _FakeNativeUpdate(client_id, round_id, quantized)

    def aggregate_round(
        self,
        round_id: int,
        updates: list[_FakeNativeUpdate],
    ) -> list[_FakeNativeLayer]:
        if any(update.round_id != round_id for update in updates):
            raise ValueError("round mismatch")
        output = []
        for layer_index in range(len(updates[0].layers)):
            product = None
            for update in updates:
                layer = update.layers[layer_index]
                current = torch.as_tensor(layer["b"]) @ torch.as_tensor(layer["a"])
                product = current if product is None else product + current
            output.append(
                _FakeNativeLayer(
                    int(updates[0].layers[layer_index]["layer_id"]),
                    product,
                )
            )
        return output

    def close(self) -> None:
        self.closed = True

    def _quantize(self, value) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        return torch.round(
            torch.clamp(tensor, -self.xmax, self.xmax) * self.scale
        ).to(torch.int64)


class SecLoRAStateTest(unittest.TestCase):
    def test_manifest_is_sorted_and_pairs_a_b(self) -> None:
        state = {
            "layer.1.lora_B.weight": torch.ones(3, 2),
            "layer.0.lora_A.weight": torch.ones(2, 4),
            "layer.1.lora_A.weight": torch.ones(2, 4),
            "layer.0.lora_B.weight": torch.ones(3, 2),
        }
        manifest, factors = canonicalize_lora_state(state, expected_rank=2)

        self.assertEqual(
            [spec.a_key for spec in manifest],
            ["layer.0.lora_A.weight", "layer.1.lora_A.weight"],
        )
        self.assertEqual([item.spec.layer_id for item in factors], [0, 1])
        self.assertTrue(all(item.a.dtype == torch.float32 for item in factors))
        self.assertTrue(all(item.a.is_contiguous() for item in factors))

    def test_manifest_rejects_unpaired_tensors(self) -> None:
        with self.assertRaises(KeyError):
            canonicalize_lora_state(
                {"layer.lora_A.weight": torch.ones(2, 3)},
                expected_rank=2,
            )


class SecLoRALowRankTest(unittest.TestCase):
    def test_factorization_does_not_materialize_or_change_product(self) -> None:
        torch.manual_seed(3)
        left = torch.randn(9, 3)
        right = torch.randn(3, 7)
        b, a = factorize_low_rank_product(left, right, target_rank=3)

        self.assertEqual(tuple(b.shape), (9, 3))
        self.assertEqual(tuple(a.shape), (3, 7))
        self.assertTrue(
            torch.allclose(b @ a, left.double() @ right.double(), atol=1e-10)
        )


class SecLoRABackendTest(unittest.TestCase):
    def test_backend_round_trip_uses_joint_native_aggregate(self) -> None:
        config = SecLoRAConfig(
            mode="sel-2s",
            ratio=0.25,
            sfp=12,
            xmax=1.0,
            threads=4,
        )
        session = _FakeNativeSession(config.scale, config.xmax)
        with tempfile.TemporaryDirectory() as tmp:
            backend = SecLoRABackend(
                config=config,
                num_clients=2,
                rank=2,
                metrics_dir=Path(tmp),
                native_session=session,
            )
            states = [
                {
                    "layer.lora_A.weight": torch.tensor(
                        [[0.25, 0.0, 0.5], [0.0, 0.25, 0.0]]
                    ),
                    "layer.lora_B.weight": torch.tensor(
                        [[0.5, 0.0], [0.0, 0.5], [0.25, 0.25]]
                    ),
                },
                {
                    "layer.lora_A.weight": torch.tensor(
                        [[0.0, 0.5, 0.0], [0.25, 0.0, 0.25]]
                    ),
                    "layer.lora_B.weight": torch.tensor(
                        [[0.5, 0.0], [0.0, 0.5], [0.25, 0.25]]
                    ),
                },
            ]
            updates = [
                backend.encrypt(state, client_id, round_id=1)
                for client_id, state in enumerate(states)
            ]
            result = backend.secure_aggregate(
                list(enumerate(updates)),
                round_id=1,
            )

            expected = sum(
                state["layer.lora_B.weight"] @ state["layer.lora_A.weight"]
                for state in states
            ) / len(states)
            actual = (
                result["layer.lora_B.weight"]
                @ result["layer.lora_A.weight"]
            )
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
            self.assertGreater(backend.ciphertext_size(updates[0]), 0)
            backend.close()
            self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
