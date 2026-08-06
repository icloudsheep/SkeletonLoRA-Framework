"""Plot publication-ready rolling-loss curves from one or more training runs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _parse_series(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("series must use LABEL=RUN_DIR")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("series must use LABEL=RUN_DIR")
    return label, Path(path).expanduser()


def load_rolling_curve(run_dir: Path) -> tuple[list[int], list[float]]:
    path = run_dir / "metrics" / "step.csv"
    if not path.is_file():
        raise FileNotFoundError(f"step metrics not found: {path}")
    values: dict[int, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"round", "step", "loss_moving_avg"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no observations")
    local_steps = max(int(row["step"]) for row in rows) + 1
    for row in rows:
        global_step = (int(row["round"]) - 1) * local_steps + int(row["step"]) + 1
        values[global_step].append(float(row["loss_moving_avg"]))
    steps = sorted(values)
    return steps, [sum(values[step]) / len(values[step]) for step in steps]


def plot(series: list[tuple[str, Path]], output: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
        }
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for label, run_dir in series:
        steps, losses = load_rolling_curve(run_dir)
        axis.plot(steps, losses, linewidth=1.6, label=label)
    axis.set_xlabel("Local training step")
    axis.set_ylabel("Rolling loss (300-step window)")
    axis.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.7)
    axis.legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        type=_parse_series,
        help="repeatable LABEL=RUN_DIR series",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/seclora_3b_modern_loss.pdf"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    plot(args.series, args.output, args.dpi)
    print(f"saved {args.output} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
