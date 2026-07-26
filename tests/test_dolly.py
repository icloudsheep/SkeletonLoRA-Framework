"""Dolly 数据编码与分片的轻量回归测试。"""

import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
