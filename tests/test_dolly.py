"""Dolly 数据编码与分片的轻量回归测试。"""

import unittest

import torch

from datasets import build_dataloader
from datasets.dolly import (
    TokenizedCausalLMDataset,
    _encode_record,
    _format_prompt,
    _iid_uniform_shards,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict:
        token_ids = [3 + ord(char) % 29 for char in text]
        if add_special_tokens:
            token_ids.insert(0, 1)
        return {"input_ids": token_ids}


class DollyTest(unittest.TestCase):
    def test_prompt_includes_optional_context(self) -> None:
        without_context = _format_prompt("Do it", "")
        with_context = _format_prompt("Do it", "Useful facts")

        self.assertNotIn("### Context:", without_context)
        self.assertIn("### Context:\nUseful facts", with_context)
        self.assertTrue(with_context.endswith("### Response:\n"))

    def test_encoding_masks_prompt_and_padding(self) -> None:
        sample = _encode_record(
            {
                "instruction": "Add",
                "context": "",
                "response": "Done",
            },
            tokenizer=_FakeTokenizer(),
            pad_token_id=0,
            max_length=160,
            line_number=1,
        )

        self.assertEqual(sample["input_ids"].shape, torch.Size([160]))
        self.assertTrue(torch.any(sample["labels"] != -100))
        self.assertTrue(torch.all(sample["labels"][sample["attention_mask"] == 0] == -100))

    def test_long_prompt_and_response_keep_meaningful_supervision(self) -> None:
        sample = _encode_record(
            {
                "instruction": "I" * 200,
                "context": "C" * 200,
                "response": "R" * 100,
            },
            tokenizer=_FakeTokenizer(),
            pad_token_id=0,
            max_length=32,
            line_number=1,
        )

        supervised = sample["labels"][sample["labels"] != -100]
        self.assertEqual(len(supervised), 16)
        self.assertEqual(supervised[-1].item(), _FakeTokenizer.eos_token_id)
        self.assertEqual(sample["input_ids"][0].item(), 1)

    def test_iid_split_is_complete_balanced_and_repeatable(self) -> None:
        values = torch.arange(30).reshape(10, 3)
        dataset = TokenizedCausalLMDataset(values, values, values)

        first = _iid_uniform_shards(dataset, num_clients=3, seed=42)
        second = _iid_uniform_shards(dataset, num_clients=3, seed=42)

        self.assertEqual([len(shard) for shard in first], [4, 3, 3])
        self.assertEqual(
            [shard.indices for shard in first],
            [shard.indices for shard in second],
        )
        self.assertEqual(
            sorted(index for shard in first for index in shard.indices),
            list(range(10)),
        )

    def test_dataloader_shuffle_changes_between_rounds_and_is_repeatable(self) -> None:
        config = {
            "seed": 42,
            "federated": {"num_clients": 2},
            "train": {"batch_size": 1},
        }
        dataset = torch.arange(20)

        first = _loader_order(build_dataloader(config, dataset, round_id=1, client_id=0))
        repeated = _loader_order(build_dataloader(config, dataset, round_id=1, client_id=0))
        next_round = _loader_order(build_dataloader(config, dataset, round_id=2, client_id=0))

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_round)

    def test_dataloader_rejects_out_of_range_client_id(self) -> None:
        config = {
            "seed": 42,
            "federated": {"num_clients": 2},
            "train": {"batch_size": 1},
        }

        with self.assertRaisesRegex(ValueError, "client_id 必须小于"):
            build_dataloader(config, torch.arange(2), client_id=2)


def _loader_order(dataloader) -> list[int]:
    return [int(batch.item()) for batch in dataloader]


if __name__ == "__main__":
    unittest.main()
