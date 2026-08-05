import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.gsm8k import (
    extract_numeric_answer,
    extract_reference_answer,
    load_gsm8k_records,
)
from evaluation.mmlu import _format_prompt, evaluate_mmlu, load_mmlu_records


class EvaluationBenchmarkTest(unittest.TestCase):
    def test_loads_standard_mmlu_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test"
            path.mkdir()
            (path / "abstract_algebra_test.csv").write_text(
                "What is 1+1?,1,2,3,4,B\n",
                encoding="utf-8",
            )

            records = load_mmlu_records(Path(tmp_dir))

        self.assertEqual(1, len(records))
        self.assertEqual("abstract_algebra", records[0]["subject"])
        self.assertEqual("B", records[0]["answer"])

    def test_mmlu_prompt_contains_subject(self):
        prompt = _format_prompt(
            {
                "question": "What is 1+1?",
                "choices": ["1", "2", "3", "4"],
                "subject": "abstract_algebra",
            }
        )

        self.assertIn("Subject: abstract_algebra\nQuestion: What is 1+1?", prompt)

    def test_mmlu_jsonl_rejects_missing_or_empty_subject(self):
        invalid_records = [
            {"question": "Q", "choices": ["A", "B", "C", "D"], "answer": 0},
            {
                "question": "Q",
                "choices": ["A", "B", "C", "D"],
                "answer": 0,
                "subject": "",
            },
        ]
        for record in invalid_records:
            with self.subTest(record=record), tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "mmlu.jsonl"
                path.write_text(json.dumps(record) + "\n", encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "subject 无效"):
                    load_mmlu_records(path)

    @patch("evaluation.mmlu.load_local_tokenizer", return_value=object())
    @patch("evaluation.mmlu.completion_log_probability")
    @patch("evaluation.mmlu.load_mmlu_records")
    @patch("evaluation.mmlu.require_dataset_path", return_value=Path("unused"))
    def test_mmlu_result_contains_overall_and_subject_accuracy_rows(
        self,
        _require_path,
        load_records,
        completion_score,
        _load_tokenizer,
    ):
        load_records.return_value = [
            {"question": "Q1", "choices": ["1", "2", "3", "4"], "answer": "A", "subject": "math"},
            {"question": "Q2", "choices": ["1", "2", "3", "4"], "answer": "B", "subject": "math"},
            {"question": "Q3", "choices": ["1", "2", "3", "4"], "answer": "C", "subject": "history"},
        ]
        score_groups = iter(
            [
                [4.0, 3.0, 2.0, 1.0],
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 2.0, 4.0, 3.0],
            ]
        )
        active_scores = iter(())

        def next_score(*_args, **_kwargs):
            nonlocal active_scores
            try:
                return next(active_scores)
            except StopIteration:
                active_scores = iter(next(score_groups))
                return next(active_scores)

        completion_score.side_effect = next_score

        result = evaluate_mmlu(
            model=torch.nn.Module(),
            model_config={},
            benchmark_config={"path": "unused"},
            device=torch.device("cpu"),
            run_id="test-run",
        )

        summaries = {
            (row["row_type"], row["subject"]): row
            for row in result.rows
            if row["row_type"] != "question"
        }
        self.assertEqual(summaries[("overall", "ALL")]["correct_count"], 2)
        self.assertEqual(summaries[("overall", "ALL")]["total_count"], 3)
        self.assertAlmostEqual(summaries[("overall", "ALL")]["accuracy"], 2 / 3)
        self.assertEqual(summaries[("subject", "math")]["correct_count"], 1)
        self.assertEqual(summaries[("subject", "math")]["total_count"], 2)
        self.assertEqual(summaries[("subject", "math")]["accuracy"], 0.5)
        self.assertEqual(summaries[("subject", "history")]["accuracy"], 1.0)
        self.assertEqual(sum(row["row_type"] == "question" for row in result.rows), 3)
        self.assertEqual(
            {"subject_v1"},
            {row["prompt_version"] for row in result.rows},
        )

    def test_loads_standard_gsm8k_jsonl_and_extracts_answers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test.jsonl"
            path.write_text(
                json.dumps({"question": "What is 1+1?", "answer": "Work. #### 2"}) + "\n",
                encoding="utf-8",
            )

            records = load_gsm8k_records(Path(tmp_dir))

        self.assertEqual(1, len(records))
        self.assertEqual("2", extract_reference_answer(records[0]["answer"]))
        self.assertEqual("2000", extract_numeric_answer("Therefore #### 2,000"))
        self.assertEqual("2.5", extract_numeric_answer("The final answer is 2.50."))


if __name__ == "__main__":
    unittest.main()
