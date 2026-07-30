"""MMLU auxiliary train 数据编码与默认配置的回归测试。"""

from pathlib import Path
import unittest

import torch
import yaml

from datasets.mmlu_train import _encode_record, _format_prompt, _validate_record


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict:
        token_ids = [3 + ord(char) % 29 for char in text]
        if add_special_tokens:
            token_ids.insert(0, 1)
        return {"input_ids": token_ids}


class MMLUTrainTest(unittest.TestCase):
    def test_prompt_contains_subject_question_and_all_choices(self) -> None:
        prompt = _format_prompt(
            {
                "subject": "abstract_algebra",
                "question": "Which option is correct?",
                "choices": ["First", "Second", "Third", "Fourth"],
            }
        )

        self.assertIn("Subject: abstract_algebra", prompt)
        self.assertIn("Question: Which option is correct?", prompt)
        for label, choice in zip("ABCD", ("First", "Second", "Third", "Fourth")):
            self.assertIn(f"{label}. {choice}", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_integer_answer_maps_to_choice_label(self) -> None:
        record = _validate_record(
            {
                "question": "Question",
                "choices": ["A1", "B1", "C1", "D1"],
                "answer": 2,
                "subject": "test",
            },
            line_number=1,
        )

        self.assertEqual(record["answer"], "C")

    def test_encoding_masks_prompt_and_padding(self) -> None:
        sample = _encode_record(
            {
                "question": "Question",
                "choices": ["A1", "B1", "C1", "D1"],
                "answer": 1,
                "subject": "test",
            },
            tokenizer=_FakeTokenizer(),
            pad_token_id=0,
            max_length=256,
            line_number=1,
        )

        supervised = sample["labels"] != -100
        self.assertEqual(sample["input_ids"].shape, torch.Size([256]))
        self.assertTrue(torch.any(supervised))
        self.assertTrue(torch.all(sample["labels"][sample["attention_mask"] == 0] == -100))
        first_supervised = int(torch.nonzero(supervised, as_tuple=False)[0].item())
        self.assertTrue(torch.all(sample["labels"][:first_supervised] == -100))
        self.assertEqual(sample["labels"][supervised][-1].item(), _FakeTokenizer.eos_token_id)

    def test_invalid_answer_and_choices_are_rejected(self) -> None:
        base = {
            "question": "Question",
            "choices": ["A1", "B1", "C1", "D1"],
            "answer": 0,
        }
        with self.assertRaisesRegex(ValueError, "choices 必须包含 4 个非空字符串"):
            _validate_record({**base, "choices": ["A1"]}, line_number=1)
        with self.assertRaisesRegex(ValueError, "answer 必须为"):
            _validate_record({**base, "answer": 4}, line_number=1)

    def test_default_config_uses_mmlu_train(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "configs" / "default.yaml").open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        self.assertEqual(config["dataset"]["kind"], "mmlu_train")
        self.assertEqual(
            config["dataset"]["path"],
            "./datasets/mmlu/mmlu_auxiliary_train.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
