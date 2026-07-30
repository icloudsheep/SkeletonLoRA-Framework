import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from evaluate import (
    _add_model_mode,
    _build_evaluation_model,
    _output_filename,
)


class EvaluateModeTest(unittest.TestCase):
    @patch("evaluate.load_file")
    @patch("evaluate.build_peft_model")
    @patch("evaluate.build_model")
    def test_base_mode_does_not_load_adapter(
        self, build_model, build_peft_model, load_file
    ) -> None:
        base_model = MagicMock(spec=torch.nn.Module)
        moved_model = MagicMock(spec=torch.nn.Module)
        base_model.to.return_value = moved_model
        build_model.return_value = base_model
        config = {"model": {"kind": "open_llama"}, "lora": {}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _build_evaluation_model(
                config, Path(tmp_dir), "base", torch.device("cpu")
            )

        self.assertIs(result, moved_model)
        build_peft_model.assert_not_called()
        load_file.assert_not_called()

    @patch("evaluate.build_model")
    def test_adapter_mode_requires_final_checkpoint(self, build_model) -> None:
        config = {"model": {"kind": "open_llama"}, "lora": {}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "找不到 final adapter"):
                _build_evaluation_model(
                    config, Path(tmp_dir), "adapter", torch.device("cpu")
                )
        build_model.assert_not_called()

    def test_base_mode_uses_separate_output_and_csv_marker(self) -> None:
        self.assertEqual("mmlu.csv", _output_filename("mmlu", "adapter"))
        self.assertEqual("mmlu_base.csv", _output_filename("mmlu", "base"))
        self.assertEqual("eval_base.csv", _output_filename("train", "base"))

        fields, rows = _add_model_mode(
            ["run_id", "accuracy"],
            [{"run_id": "run-1", "accuracy": 0.25}],
            "base",
        )
        self.assertEqual(["run_id", "model_mode", "accuracy"], fields)
        self.assertEqual("base", rows[0]["model_mode"])


if __name__ == "__main__":
    unittest.main()
