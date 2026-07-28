"""Hugging Face Parquet record conversion regression tests."""

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "convert_hf_parquet.py"
MODULE_SPEC = importlib.util.spec_from_file_location("convert_hf_parquet", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"cannot load conversion module: {MODULE_PATH}")
CONVERSION_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(CONVERSION_MODULE)
convert_record = CONVERSION_MODULE.convert_record


class ConvertHfParquetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mmlu_record = {
            "question": "Question",
            "choices": ["A1", "B1", "C1", "D1"],
            "answer": 0,
            "subject": "math",
        }

    def test_mmlu_train_replaces_missing_subject_with_unknown(self) -> None:
        for subject in (None, "", "   "):
            with self.subTest(subject=subject):
                converted = convert_record(
                    "mmlu_train",
                    {**self.mmlu_record, "subject": subject},
                    1,
                )
                self.assertEqual(converted["subject"], "unknown")

    def test_mmlu_evaluation_rejects_missing_subject(self) -> None:
        for subject in (None, "", "   "):
            with self.subTest(subject=subject):
                with self.assertRaisesRegex(ValueError, "invalid subject"):
                    convert_record(
                        "mmlu",
                        {**self.mmlu_record, "subject": subject},
                        1,
                    )

    def test_mmlu_train_preserves_valid_subject(self) -> None:
        converted = convert_record("mmlu_train", self.mmlu_record, 1)
        self.assertEqual(converted["subject"], "math")


if __name__ == "__main__":
    unittest.main()
