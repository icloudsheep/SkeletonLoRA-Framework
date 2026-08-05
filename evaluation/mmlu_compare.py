"""Compare question-level predictions from two MMLU evaluation CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


CHOICE_LABELS = {"A", "B", "C", "D"}


def load_question_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Load and validate the question rows used for an exact MMLU comparison."""
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"row_type", "subject", "question_id", "prediction", "answer"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} 缺少列: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            if row["row_type"] != "question":
                continue
            subject = row["subject"].strip()
            question_id = row["question_id"].strip()
            prediction = row["prediction"].strip().upper()
            answer = row["answer"].strip().upper()
            if not subject or not question_id:
                raise ValueError(f"{path} 第 {line_number} 行缺少 subject 或 question_id")
            if prediction not in CHOICE_LABELS or answer not in CHOICE_LABELS:
                raise ValueError(f"{path} 第 {line_number} 行 prediction 或 answer 无效")
            key = (subject, question_id)
            if key in rows:
                raise ValueError(f"{path} 存在重复题目: {key}")
            rows[key] = {"prediction": prediction, "answer": answer}
    if not rows:
        raise ValueError(f"{path} 不包含 question 行")
    return rows


def compare_results(old_path: Path, new_path: Path) -> dict:
    """Return overall and per-subject score changes between matched evaluations."""
    old_rows = load_question_rows(old_path)
    new_rows = load_question_rows(new_path)
    if old_rows.keys() != new_rows.keys():
        old_only = len(old_rows.keys() - new_rows.keys())
        new_only = len(new_rows.keys() - old_rows.keys())
        raise ValueError(f"题目集合不一致: old_only={old_only}, new_only={new_only}")

    old_correct = 0
    new_correct = 0
    changed = 0
    improved = 0
    regressed = 0
    subject_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for key, old_row in old_rows.items():
        new_row = new_rows[key]
        if old_row["answer"] != new_row["answer"]:
            raise ValueError(f"题目答案不一致: {key}")
        old_is_correct = old_row["prediction"] == old_row["answer"]
        new_is_correct = new_row["prediction"] == new_row["answer"]
        old_correct += int(old_is_correct)
        new_correct += int(new_is_correct)
        changed += int(old_row["prediction"] != new_row["prediction"])
        improved += int(not old_is_correct and new_is_correct)
        regressed += int(old_is_correct and not new_is_correct)
        counts = subject_counts[key[0]]
        counts[0] += int(old_is_correct)
        counts[1] += int(new_is_correct)
        counts[2] += 1

    total = len(old_rows)
    return {
        "total": total,
        "old_correct": old_correct,
        "new_correct": new_correct,
        "old_accuracy": old_correct / total,
        "new_accuracy": new_correct / total,
        "delta_percentage_points": (new_correct - old_correct) * 100 / total,
        "changed_predictions": changed,
        "improved": improved,
        "regressed": regressed,
        "subjects": [
            {
                "subject": subject,
                "old_correct": counts[0],
                "new_correct": counts[1],
                "total": counts[2],
                "delta_percentage_points": (counts[1] - counts[0]) * 100 / counts[2],
            }
            for subject, counts in sorted(subject_counts.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two MMLU result CSV files")
    parser.add_argument("old", type=Path, help="旧版无 subject prompt 的 mmlu.csv")
    parser.add_argument("new", type=Path, help="新版含 subject prompt 的 mmlu.csv")
    args = parser.parse_args()
    result = compare_results(args.old, args.new)

    print(
        "overall: "
        f"{result['old_correct']}/{result['total']} "
        f"({result['old_accuracy'] * 100:.4f}%) -> "
        f"{result['new_correct']}/{result['total']} "
        f"({result['new_accuracy'] * 100:.4f}%), "
        f"delta={result['delta_percentage_points']:+.4f} pp"
    )
    print(
        f"prediction changes: {result['changed_predictions']}; "
        f"improved={result['improved']}; regressed={result['regressed']}"
    )
    print("per subject:")
    for row in result["subjects"]:
        print(
            f"  {row['subject']}: "
            f"{row['old_correct']}/{row['total']} -> "
            f"{row['new_correct']}/{row['total']}, "
            f"delta={row['delta_percentage_points']:+.4f} pp"
        )


if __name__ == "__main__":
    main()
