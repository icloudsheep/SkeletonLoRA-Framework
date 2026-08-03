"""LoRA 乘积低秩聚合的回归测试。"""

import importlib.util
import unittest
from pathlib import Path

import torch

_MODULE_PATH = Path(__file__).parents[1] / "utils" / "lora_product.py"
_SPEC = importlib.util.spec_from_file_location("lora_product_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
aggregate_lora_products = _MODULE.aggregate_lora_products
factorize_lora_product = _MODULE.factorize_lora_product


class LoraProductTest(unittest.TestCase):
    def test_factor_aggregation_matches_dense_truncated_svd(self) -> None:
        generator = torch.Generator().manual_seed(42)
        states = []
        for _ in range(3):
            states.append({
                "layer.lora_A.weight": torch.randn(2, 7, generator=generator),
                "layer.lora_B.weight": torch.randn(5, 2, generator=generator),
            })

        actual = aggregate_lora_products(states, rank=2)
        dense_mean = torch.stack([
            state["layer.lora_B.weight"] @ state["layer.lora_A.weight"]
            for state in states
        ]).mean(dim=0)
        expected_b, expected_a = factorize_lora_product(dense_mean, rank=2)

        self.assertTrue(torch.allclose(
            actual["layer.lora_B.weight"] @ actual["layer.lora_A.weight"],
            expected_b @ expected_a,
            rtol=1e-4,
            atol=1e-5,
        ))

    def test_non_lora_values_are_still_meaned(self) -> None:
        states = [
            {"other": torch.tensor([1.0, 3.0])},
            {"other": torch.tensor([3.0, 5.0])},
        ]

        actual = aggregate_lora_products(states, rank=1)

        self.assertTrue(torch.equal(actual["other"], torch.tensor([2.0, 4.0])))

    def test_aggregation_rejects_invalid_inputs(self) -> None:
        valid = {
            "layer.lora_A.weight": torch.randn(2, 7),
            "layer.lora_B.weight": torch.randn(5, 2),
        }
        mismatched_shape = {
            "layer.lora_A.weight": torch.randn(3, 7),
            "layer.lora_B.weight": torch.randn(5, 3),
        }

        with self.assertRaisesRegex(ValueError, "plaintexts"):
            aggregate_lora_products([], rank=2)
        with self.assertRaisesRegex(ValueError, "rank"):
            aggregate_lora_products([valid], rank=0)
        with self.assertRaisesRegex(ValueError, "参数集合"):
            aggregate_lora_products([valid, {"other": torch.ones(1)}], rank=2)
        with self.assertRaisesRegex(ValueError, "与 client 0 不一致"):
            aggregate_lora_products([valid, mismatched_shape], rank=2)


if __name__ == "__main__":
    unittest.main()
