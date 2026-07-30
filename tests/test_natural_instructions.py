"""Natural Instructions 数据校验、编码和读取顺序的回归测试。"""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from datasets.natural_instructions import (
    _encode_record,
    _format_prompt,
    _iter_records,
    _resolve_train_files,
    _validate_record,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict:
        token_ids = [3 + ord(char) % 29 for char in text]
        if add_special_tokens:
            token_ids.insert(0, 1)
        return {"input_ids": token_ids}


class NaturalInstructionsTest(unittest.TestCase):
    def test_prompt_contains_definition_input_and_response_marker(self) -> None:
        prompt = _format_prompt("Classify the text.", "A sample")

        self.assertIn("### Instruction:\nClassify the text.", prompt)
        self.assertIn("### Input:\nA sample", prompt)
        self.assertTrue(prompt.endswith("### Response:\n"))

    def test_encoding_masks_prompt_and_padding(self) -> None:
        sample = _encode_record(
            {
                "definition": "Classify the text.",
                "inputs": "A sample",
                "targets": "positive",
            },
            tokenizer=_FakeTokenizer(),
            pad_token_id=0,
            max_length=128,
            source="sample.jsonl:1",
        )

        supervised = sample["labels"] != -100
        self.assertEqual(sample["input_ids"].shape, torch.Size([128]))
        self.assertTrue(torch.any(supervised))
        self.assertTrue(torch.all(sample["labels"][sample["attention_mask"] == 0] == -100))
        first_supervised = int(torch.nonzero(supervised, as_tuple=False)[0].item())
        self.assertTrue(torch.all(sample["labels"][:first_supervised] == -100))

    def test_validation_rejects_missing_or_non_string_fields(self) -> None:
        valid = {
            "definition": "Classify.",
            "inputs": "Input",
            "targets": "label",
        }
        self.assertEqual(_validate_record(valid, "sample"), valid)
        with self.assertRaisesRegex(ValueError, "字段 targets"):
            _validate_record({**valid, "targets": ["label"]}, "sample")
        self.assertEqual(
            _validate_record({**valid, "inputs": ""}, "sample")["inputs"],
            "",
        )
        with self.assertRaisesRegex(ValueError, "字段 inputs"):
            _validate_record({**valid, "inputs": None}, "sample")

    def test_reader_uses_only_train_files_and_honors_max_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            train_dir = root / "train"
            test_dir = root / "test"
            train_dir.mkdir()
            test_dir.mkdir()
            for index in range(2):
                path = train_dir / f"task_{index}.jsonl"
                with path.open("w", encoding="utf-8") as stream:
                    for row in range(3):
                        stream.write(json.dumps({"id": f"{index}-{row}"}) + "\n")
            (test_dir / "task_test.jsonl").write_text('{"id":"test"}\n', encoding="utf-8")

            files = _resolve_train_files(str(root))
            records = list(_iter_records(files, seed=42, max_samples=4))

        self.assertEqual(len(files), 2)
        self.assertEqual(len(records), 4)
        self.assertEqual({record["id"].split("-")[0] for record, _ in records}, {"0", "1"})
        self.assertNotIn("test", {record["id"] for record, _ in records})


if __name__ == "__main__":
    unittest.main()
