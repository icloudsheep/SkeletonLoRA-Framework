"""训练指标辅助计算与 CSV 表头的回归测试。"""

import math
from pathlib import Path
import tempfile
import unittest

import torch

from training_progress import (
    _learning_rate_at_step,
    _perplexity,
    _supervised_token_count,
)
from utils.metrics import (
    CLIENT_ROUND_COLS,
    SECLORA_CLIENT_COLS,
    SECLORA_LAYER_COLS,
    SECLORA_ROUND_COLS,
    STEP_COLS,
    CsvWriters,
)


class TrainingMetricsTest(unittest.TestCase):
    def test_supervised_token_count_ignores_masked_labels(self) -> None:
        batch = {"labels": torch.tensor([[-100, 1, 2], [-100, -100, 3]])}
        self.assertEqual(_supervised_token_count(batch, "open_llama"), 3)

    def test_perplexity_is_exponential_cross_entropy(self) -> None:
        self.assertAlmostEqual(_perplexity(2.0, "open_llama"), math.exp(2.0))
        self.assertTrue(math.isnan(_perplexity(2.0, "dummy")))

    def test_cosine_schedule_uses_global_steps_and_minimum(self) -> None:
        config = {
            "learning_rate": 2.0e-4,
            "lr_scheduler": {
                "type": "cosine",
                "warmup_steps": 300,
                "min_learning_rate": 2.0e-5,
            },
        }
        self.assertAlmostEqual(
            _learning_rate_at_step(config, 299, 6000), 2.0e-4
        )
        self.assertAlmostEqual(
            _learning_rate_at_step(config, 5999, 6000), 2.0e-5
        )

    def test_csv_files_include_step_and_client_round_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            metrics_dir = Path(temporary_dir)
            writers = CsvWriters(metrics_dir)
            writers.close()

            step_header = (metrics_dir / "step.csv").read_text().splitlines()[0]
            round_header = (metrics_dir / "client_round.csv").read_text().splitlines()[0]
            seclora_client_header = (
                metrics_dir / "seclora_client.csv"
            ).read_text().splitlines()[0]
            seclora_round_header = (
                metrics_dir / "seclora_round.csv"
            ).read_text().splitlines()[0]
            seclora_layer_header = (
                metrics_dir / "seclora_layer.csv"
            ).read_text().splitlines()[0]

        self.assertEqual(step_header.split(","), STEP_COLS)
        self.assertEqual(round_header.split(","), CLIENT_ROUND_COLS)
        self.assertEqual(seclora_client_header.split(","), SECLORA_CLIENT_COLS)
        self.assertEqual(seclora_round_header.split(","), SECLORA_ROUND_COLS)
        self.assertEqual(seclora_layer_header.split(","), SECLORA_LAYER_COLS)


if __name__ == "__main__":
    unittest.main()
