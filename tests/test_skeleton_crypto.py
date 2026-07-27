import unittest

import numpy as np
import torch

from skeleton_crypto import SkeletonLoRACrypto


def _config(*, skeleton: bool, skeleton_rank: int = 4) -> dict:
    return {
        "scheme": "ckks",
        "mode": "full",
        "ratio": None,
        "skeleton": skeleton,
        "skeleton_rank": skeleton_rank,
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "global_scale": 2 ** 40,
        "cur_condition_threshold": 1e12,
    }


class SkeletonCryptoTest(unittest.TestCase):
    def test_ckks_round_trip_matches_mean_lora_product(self) -> None:
        for skeleton in (False, True):
            with self.subTest(skeleton=skeleton):
                torch.manual_seed(7)
                rank = 2
                shared_a = torch.randn(rank, 4)
                states = []
                for _ in range(2):
                    states.append(
                        {
                            "layer.lora_A.weight": shared_a.clone(),
                            "layer.lora_B.weight": torch.randn(5, rank),
                        }
                    )

                crypto = SkeletonLoRACrypto(
                    _config(skeleton=skeleton),
                    num_clients=2,
                    rank=rank,
                )
                ciphertexts = [
                    (client_id, crypto.encrypt(state, client_id, round_id=1))
                    for client_id, state in enumerate(states)
                ]
                aggregated = crypto.secure_aggregate(ciphertexts, round_id=1)

                actual = (
                    aggregated["layer.lora_B.weight"]
                    @ aggregated["layer.lora_A.weight"]
                ).numpy()
                expected = np.mean(
                    [
                        (state["layer.lora_B.weight"] @ state["layer.lora_A.weight"]).numpy()
                        for state in states
                    ],
                    axis=0,
                )
                np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)

    def test_rejects_missing_lora_pair(self) -> None:
        crypto = SkeletonLoRACrypto(_config(skeleton=False), num_clients=1, rank=2)
        with self.assertRaisesRegex(KeyError, "缺少与"):
            crypto.encrypt({"layer.lora_A.weight": torch.randn(2, 4)}, 0, 1)


if __name__ == "__main__":
    unittest.main()
