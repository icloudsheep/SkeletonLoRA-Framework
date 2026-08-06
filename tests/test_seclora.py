"""Regression tests for the Python/native SecLoRA boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        self.sp_plain_bytes = self.serialized_size_bytes
        self.sd_cipher_bytes = 0
        self.protected_b_labels = 0
        self.protected_a_labels = 0
        self.candidate_b_labels = 0
        self.candidate_a_labels = 0
        self.binding_input_copy_wall_sec = 0.001
        self.quantize_pack_wall_sec = 0.001
        self.precompute_wall_sec = 0.002
        self.online_crypto_wall_sec = 0.003
        self.serialize_wall_sec = 0.004


class _FakeNativeLayer:
    def __init__(self, layer_id: int, product: torch.Tensor) -> None:
        rows = product.shape[0]
        self.layer_id = layer_id
        self.c = torch.eye(rows, dtype=torch.int64)
        self.m = torch.eye(rows, dtype=torch.int64)
        self.s = product.to(torch.int64)
        self.selected_rank = rows
        self.baseline_checks = 1
        self.baseline_relative_error = 0.0
        self.decrypted_cells = int(product.numel())
        self.pivot_candidate_cells = 0
        self.download_c_bytes = 0
        self.download_m_bytes = int(rows * rows * 8)
        self.download_s_bytes = int(product.numel() * 8)


class _FakeNativeSession:
    def __init__(self, scale: int, xmax: float) -> None:
        self.scale = scale
        self.xmax = xmax
        self.closed = False
        self.last_round_metrics = SimpleNamespace(
            sp_wall_sec=0.01,
            sd_wall_sec=0.02,
            sd_dfe_mask_wall_sec=0.002,
            sd_fe_eval_wall_sec=0.008,
            sd_bsgs_search_wall_sec=0.009,
            sd_control_wall_sec=0.001,
            cur_skeleton_wall_sec=0.006,
            cur_reconstruct_wall_sec=0.003,
            experiment_verify_wall_sec=0.004,
            server_common_control_wall_sec=0.005,
            observed_serial_server_wall_sec=0.04,
            protected_skeleton_cells=1,
            pivot_candidate_cells=0,
            download_c_bytes_per_client=0,
            download_m_bytes_per_client=8,
            download_s_bytes_per_client=8,
            download_bytes_per_client=16,
        )

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


class SecLoRAConfigTest(unittest.TestCase):
    def test_full_sk_requires_full_ratio(self) -> None:
        SecLoRAConfig(
            mode="full-sk", ratio=1.0, sfp=22, xmax=0.03125, threads=25
        ).validate()
        with self.assertRaises(ValueError):
            SecLoRAConfig(
                mode="full-sk",
                ratio=0.25,
                sfp=22,
                xmax=0.03125,
                threads=25,
            ).validate()


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
            metrics = backend.last_aggregate_metrics
            self.assertAlmostEqual(metrics["fe_aggregate_wall_sec"], 0.01)
            self.assertAlmostEqual(metrics["bsgs_wall_sec"], 0.009)
            self.assertAlmostEqual(metrics["cur_skeleton_wall_sec"], 0.006)
            self.assertAlmostEqual(metrics["decrypt_wall_sec"], 0.031)
            self.assertAlmostEqual(
                metrics["server_parallel_critical_wall_sec"],
                metrics["decrypt_wall_sec"]
                + metrics["output_reconstruct_wall_sec"],
            )
            backend.close()
            self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
