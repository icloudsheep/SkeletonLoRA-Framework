import csv
import tempfile
import unittest
from pathlib import Path

from evaluation.mmlu_compare import compare_results


FIELDNAMES = ["row_type", "subject", "question_id", "prediction", "answer"]


class CompareMmluResultsTest(unittest.TestCase):
    def test_compares_matched_question_predictions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            old_path = Path(tmp_dir) / "old.csv"
            new_path = Path(tmp_dir) / "new.csv"
            self._write(
                old_path,
                [
                    ["question", "math", "0", "A", "A"],
                    ["question", "math", "1", "A", "B"],
                    ["question", "history", "2", "C", "C"],
                ],
            )
            self._write(
                new_path,
                [
                    ["question", "math", "0", "B", "A"],
                    ["question", "math", "1", "B", "B"],
                    ["question", "history", "2", "C", "C"],
                ],
            )

            result = compare_results(old_path, new_path)

        self.assertEqual(3, result["total"])
        self.assertEqual(2, result["old_correct"])
        self.assertEqual(2, result["new_correct"])
        self.assertEqual(2, result["changed_predictions"])
        self.assertEqual(1, result["improved"])
        self.assertEqual(1, result["regressed"])
        self.assertEqual(0.0, result["delta_percentage_points"])

    def test_rejects_different_question_sets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            old_path = Path(tmp_dir) / "old.csv"
            new_path = Path(tmp_dir) / "new.csv"
            self._write(old_path, [["question", "math", "0", "A", "A"]])
            self._write(new_path, [["question", "math", "1", "A", "A"]])

            with self.assertRaisesRegex(ValueError, "题目集合不一致"):
                compare_results(old_path, new_path)

    @staticmethod
    def _write(path: Path, rows: list[list[str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDNAMES)
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
