"""Plot the four Modern_Loss training curves as a publication-ready PDF."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "pdf" / "modern_loss_curves.pdf"
RUNS = {
    "3B": {
        "CKKS+Skeleton": "MODERN-LOSS-3B-AB-1-SKE-NI",
        "SHE-LoRA (CKKS)": "MODERN-LOSS-3B-AB-1-NI",
    },
    "7B": {
        "CKKS+Skeleton": "MODERN-LOSS-7B-AB-1-SKE-NI",
        "SHE-LoRA (CKKS)": "MODERN-LOSS-7B-AB-1-NI",
    },
}


def _token_weighted_loss(rows: list[dict[str, str]]) -> float:
    token_count = sum(int(row["supervised_tokens"]) for row in rows)
    if token_count <= 0:
        raise ValueError("supervised_tokens must be positive in every round")
    loss_sum = sum(
        float(row["token_weighted_mean_loss"])
        * int(row["supervised_tokens"])
        for row in rows
    )
    return loss_sum / token_count


def _client_mean(column: str) -> Callable[[list[dict[str, str]]], float]:
    def calculate(rows: list[dict[str, str]]) -> float:
        if not rows:
            raise ValueError("a round must contain at least one client record")
        return sum(float(row[column]) for row in rows) / len(rows)

    return calculate


METRICS: dict[str, tuple[str, Callable[[list[dict[str, str]]], float]]] = {
    "token-weighted": ("Token-weighted loss", _token_weighted_loss),
    "mean": ("Mean client loss", _client_mean("mean_loss")),
    "moving": ("Moving-average loss", _client_mean("final_moving_avg_loss")),
}


def load_curve(path: Path, metric: str) -> tuple[list[int], list[float]]:
    """Load client-round rows and aggregate the selected metric by round."""
    if metric not in METRICS:
        raise ValueError(f"unsupported metric: {metric}")
    if not path.is_file():
        raise FileNotFoundError(f"metrics file not found: {path}")

    rows_by_round: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "round",
            "mean_loss",
            "token_weighted_mean_loss",
            "final_moving_avg_loss",
            "supervised_tokens",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            round_id = int(row["round"])
            if round_id <= 0:
                raise ValueError(f"round must be positive in {path}: {round_id}")
            rows_by_round[round_id].append(row)

    if not rows_by_round:
        raise ValueError(f"metrics file contains no records: {path}")
    rounds = sorted(rows_by_round)
    expected = list(range(rounds[0], rounds[-1] + 1))
    if rounds != expected:
        missing_rounds = sorted(set(expected).difference(rounds))
        raise ValueError(f"{path} has missing rounds: {missing_rounds}")

    calculate = METRICS[metric][1]
    return rounds, [calculate(rows_by_round[round_id]) for round_id in rounds]


def plot_curves(metric: str, output: Path, dpi: int) -> None:
    """Render model loss curves and their signed method gaps in one PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_label = METRICS[metric][0]
    colors = {
        "CKKS+Skeleton": "#0072B2",
        "SHE-LoRA (CKKS)": "#D55E00",
    }
    line_styles = {
        "CKKS+Skeleton": "-",
        "SHE-LoRA (CKKS)": "--",
    }
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 7.0),
        sharex="col",
        gridspec_kw={"height_ratios": (2.2, 1.0)},
    )
    loss_axes = axes[0]
    gap_axes = axes[1]
    loss_axes[1].sharey(loss_axes[0])
    gap_axes[1].sharey(gap_axes[0])

    handles = []
    labels = []
    for loss_axis, gap_axis, (model, methods) in zip(
        loss_axes, gap_axes, RUNS.items()
    ):
        curves: dict[str, tuple[list[int], list[float]]] = {}
        for method, run_id in methods.items():
            metrics_path = (
                PROJECT_ROOT / "output" / run_id / "metrics" / "client_round.csv"
            )
            rounds, losses = load_curve(metrics_path, metric)
            curves[method] = (rounds, losses)
            line = loss_axis.plot(
                rounds,
                losses,
                color=colors[method],
                linestyle=line_styles[method],
                linewidth=1.8,
                label=method,
            )[0]
            loss_axis.scatter(
                rounds[-1], losses[-1], color=colors[method], s=18, zorder=3
            )
            loss_axis.annotate(
                f"{losses[-1]:.3f}",
                xy=(rounds[-1], losses[-1]),
                xytext=(-5, 6 if method == "CKKS+Skeleton" else -12),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=8,
                color=colors[method],
            )
            if model == "3B":
                handles.append(line)
                labels.append(method)

        skeleton_rounds, skeleton_losses = curves["CKKS+Skeleton"]
        ckks_rounds, ckks_losses = curves["SHE-LoRA (CKKS)"]
        if skeleton_rounds != ckks_rounds:
            raise ValueError(f"{model} runs do not contain matching rounds")
        gaps = [
            skeleton_loss - ckks_loss
            for skeleton_loss, ckks_loss in zip(skeleton_losses, ckks_losses)
        ]
        gap_axis.axhline(0.0, color="#555555", linewidth=0.9, zorder=1)
        gap_axis.plot(
            skeleton_rounds,
            gaps,
            color="#6A3D9A",
            linewidth=1.6,
            zorder=3,
        )
        gap_axis.fill_between(
            skeleton_rounds,
            gaps,
            0.0,
            where=[gap >= 0.0 for gap in gaps],
            color="#0072B2",
            alpha=0.18,
            interpolate=True,
        )
        gap_axis.fill_between(
            skeleton_rounds,
            gaps,
            0.0,
            where=[gap < 0.0 for gap in gaps],
            color="#D55E00",
            alpha=0.18,
            interpolate=True,
        )
        gap_axis.scatter(
            skeleton_rounds[-1], gaps[-1], color="#6A3D9A", s=18, zorder=4
        )
        gap_axis.annotate(
            f"{gaps[-1]:+.4f}",
            xy=(skeleton_rounds[-1], gaps[-1]),
            xytext=(-5, 7 if gaps[-1] >= 0.0 else -10),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8,
            color="#6A3D9A",
        )

        loss_axis.set_title(f"OpenLLaMA-v2 {model}")
        loss_axis.set_xlim(1, max(skeleton_rounds))
        loss_axis.grid(True, color="#D0D0D0", linewidth=0.6, alpha=0.7)
        gap_axis.set_title(
            r"Method gap: CKKS+Skeleton $-$ SHE-LoRA (CKKS)",
            fontsize=9,
        )
        gap_axis.set_xlabel("Federated round")
        gap_axis.grid(True, color="#D0D0D0", linewidth=0.6, alpha=0.7)

    loss_axes[0].set_ylabel(metric_label)
    gap_axes[0].set_ylabel(r"$\Delta$ loss")
    figure.suptitle(
        "Modern_Loss convergence at 1% partial-AB encryption "
        "(K=2, 60 local steps/client/round)",
        fontsize=12,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
        metadata={
            "Title": "Modern_Loss convergence and CKKS method gaps",
            "Subject": metric_label,
        },
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 3B and 7B Modern_Loss curves from client_round.csv."
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRICS),
        default="token-weighted",
        help="round-level loss statistic to plot (default: token-weighted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PDF output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".pdf":
        parser.error("--output must use the .pdf extension")
    plot_curves(args.metric, output, args.dpi)
    print(f"Saved {args.metric} loss curves to {output}")


if __name__ == "__main__":
    main()
