import json
import tempfile
import unittest
from pathlib import Path

from evaluation.gsm8k import (
    extract_numeric_answer,
    extract_reference_answer,
    load_gsm8k_records,
)
from evaluation.mmlu import load_mmlu_records


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
