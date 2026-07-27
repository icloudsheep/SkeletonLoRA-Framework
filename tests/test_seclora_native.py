"""Small real-cryptography regression for the optional native extension."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from seclora.native import _seclora_native
except ImportError:
    _seclora_native = None


@unittest.skipIf(_seclora_native is None, "SecLoRA native extension is not built")
class SecLoRANativeTest(unittest.TestCase):
    def test_selective_skeleton_reconstructs_quantized_sum(self) -> None:
        clients = 2
        rank = 2
        rows = 6
        cols = 5
        scale = 1 << 8
        session = _seclora_native.SelectiveTwoServerSession(
            num_clients=clients,
            rank=rank,
            ratio=0.25,
            sfp=8,
            xmax=1.0,
            threads=4,
        )

        updates = []
        expected = np.zeros((rows, cols), dtype=np.int64)
        for client_id in range(clients):
            a_quant = np.fromfunction(
                lambda k, col: (client_id + 1) * (k + 1) + col,
                (rank, cols),
                dtype=int,
            ).astype(np.int64)
            b_quant = np.fromfunction(
                lambda row, k: (row + 1) + (client_id + 1) * (k + 1),
                (rows, rank),
                dtype=int,
            ).astype(np.int64)
            a_quant[:, -1] = 0
            b_quant[-1, :] = 0
            expected += b_quant @ a_quant
            updates.append(
                session.encrypt_client(
                    client_id,
                    1,
                    [
                        {
                            "layer_id": 0,
                            "name": "native.smoke",
                            "a": (a_quant / scale).astype(np.float32),
                            "b": (b_quant / scale).astype(np.float32),
                        }
                    ],
                )
            )

        skeleton = session.aggregate_round(1, updates)[0]
        reconstructed = skeleton.c @ np.linalg.solve(skeleton.m, skeleton.s)
        np.testing.assert_allclose(reconstructed, expected, atol=1e-8)
        self.assertGreaterEqual(skeleton.selected_rank, rank)
        self.assertLessEqual(skeleton.selected_rank, clients * rank)
        self.assertEqual(
            skeleton.projection_checks,
            skeleton.selected_rank - rank + 1,
        )
        encrypted_rows = int(0.25 * rows)
        encrypted_cols = int(0.25 * cols)
        self.assertEqual(
            skeleton.decrypted_cells,
            skeleton.selected_rank * (encrypted_rows + encrypted_cols),
        )
        self.assertGreater(updates[0].serialized_size_bytes, 0)
        session.close()


if __name__ == "__main__":
    unittest.main()
