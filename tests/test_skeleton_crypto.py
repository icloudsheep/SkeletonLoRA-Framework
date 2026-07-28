import pickle
import unittest

import numpy as np
import torch

from skeleton_crypto import SkeletonLoRACrypto
from skeleton_crypto.bridge import _upload_size


def _config(
    *,
    skeleton: bool,
    skeleton_rank: int = 4,
    mode: str = "full",
    ratio: float | None = None,
) -> dict:
    return {
        "scheme": "ckks",
        "mode": mode,
        "ratio": ratio,
        "skeleton": skeleton,
        "skeleton_rank": skeleton_rank,
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "global_scale": 2 ** 40,
        "cur_condition_threshold": 1e12,
    }


class SkeletonCryptoTest(unittest.TestCase):
    def test_streaming_upload_size_matches_pickle_metric(self) -> None:
        upload = {
            "kind": "term",
            "operands": (("ct", b"ciphertext"), ("plain", np.arange(8))),
        }

        self.assertEqual(_upload_size(upload), len(pickle.dumps(upload)))

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

    def test_streaming_matches_one_shot_for_multiple_layers(self) -> None:
        for skeleton in (False, True):
            with self.subTest(skeleton=skeleton):
                torch.manual_seed(17)
                states = []
                for _ in range(2):
                    state = {}
                    for layer, shape in (("q", (5, 4)), ("v", (6, 4))):
                        out_features, in_features = shape
                        state[f"{layer}.lora_A.weight"] = torch.randn(2, in_features)
                        state[f"{layer}.lora_B.weight"] = torch.randn(out_features, 2)
                    states.append(state)

                one_shot_crypto = SkeletonLoRACrypto(
                    _config(skeleton=skeleton), num_clients=2, rank=2
                )
                ciphertexts = [
                    (client_id, one_shot_crypto.encrypt(state, client_id, 3))
                    for client_id, state in enumerate(states)
                ]
                one_shot = one_shot_crypto.secure_aggregate(ciphertexts, round_id=3)

                streaming_crypto = SkeletonLoRACrypto(
                    _config(skeleton=skeleton), num_clients=2, rank=2
                )
                events = []
                streaming, stats = streaming_crypto.secure_aggregate_streaming(
                    list(enumerate(states)), round_id=3, progress=events.append
                )

                self.assertEqual(set(streaming), set(one_shot))
                self.assertEqual(stats["strategy"], "layer_block_stream")
                self.assertTrue(any(event["event"] == "block_complete" for event in events))
                for a_key in (key for key in streaming if "lora_A" in key):
                    b_key = a_key.replace("lora_A", "lora_B", 1)
                    actual = (streaming[b_key] @ streaming[a_key]).numpy()
                    expected = (one_shot[b_key] @ one_shot[a_key]).numpy()
                    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)

    def test_streaming_rejects_mismatched_client_metadata(self) -> None:
        crypto = SkeletonLoRACrypto(_config(skeleton=False), num_clients=2, rank=2)
        first = {
            "layer.lora_A.weight": torch.randn(2, 4),
            "layer.lora_B.weight": torch.randn(5, 2),
        }
        second = {
            "layer.lora_A.weight": torch.randn(2, 3),
            "layer.lora_B.weight": torch.randn(5, 2),
        }
        with self.assertRaisesRegex(ValueError, "元数据不一致"):
            crypto.secure_aggregate_streaming([(0, first), (1, second)], round_id=1)

    def test_streaming_partial_ab_ratio_one_matches_mean_product(self) -> None:
        torch.manual_seed(23)
        shared_a = torch.randn(2, 5)
        states = [
            {
                "layer.lora_A.weight": shared_a.clone(),
                "layer.lora_B.weight": torch.randn(6, 2),
            }
            for _ in range(2)
        ]
        crypto = SkeletonLoRACrypto(
            _config(skeleton=False, mode="partial_AB", ratio=1),
            num_clients=2,
            rank=2,
        )

        aggregated, _ = crypto.secure_aggregate_streaming(
            list(enumerate(states)), round_id=1
        )

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
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)

    def test_rejects_invalid_streaming_memory_config(self) -> None:
        config = _config(skeleton=False)
        config["memory"] = {"max_rss_gb": 0}
        with self.assertRaisesRegex(ValueError, "max_rss_gb"):
            SkeletonLoRACrypto(config, num_clients=1, rank=2)

    def test_streaming_stops_before_exceeding_memory_budget(self) -> None:
        config = _config(skeleton=False)
        config["memory"] = {"max_rss_gb": 1e-12}
        crypto = SkeletonLoRACrypto(config, num_clients=1, rank=2)
        state = {
            "layer.lora_A.weight": torch.randn(2, 4),
            "layer.lora_B.weight": torch.randn(5, 2),
        }

        with self.assertRaisesRegex(MemoryError, "预计 RSS"):
            crypto.secure_aggregate_streaming([(0, state)], round_id=1)


if __name__ == "__main__":
    unittest.main()
