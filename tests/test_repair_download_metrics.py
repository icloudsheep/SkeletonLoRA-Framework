import csv
import tempfile
import unittest
from pathlib import Path

from repair_download_metrics import _load_yaml, _mean_upload_size, _write_result


class RepairDownloadMetricsTest(unittest.TestCase):
    def test_mean_upload_size_uses_all_client_round_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "round.csv"
            path.write_text(
                "round,client_id,ciphertext_size\n"
                "1,0,100\n"
                "1,1,200\n"
                "2,0,300\n",
                encoding="utf-8",
            )

            result = _mean_upload_size(path)

        self.assertEqual(200.0, result)

    def test_rejects_negative_upload_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "round.csv"
            path.write_text(
                "ciphertext_size\n-1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "不能为负数"):
                _mean_upload_size(path)

    def test_load_yaml_requires_protocol_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.yaml"
            path.write_text("federated: {}\nlora: {}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "encryption"):
                _load_yaml(path)

    def test_write_result_preserves_column_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "result.csv"

            _write_result(path, {"run_id": "run", "download": 123})

            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
        self.assertEqual(["run_id", "download"], rows[0])
        self.assertEqual(["run", "123"], rows[1])


if __name__ == "__main__":
    unittest.main()
