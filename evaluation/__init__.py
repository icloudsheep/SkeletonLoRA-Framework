"""专业基准评测分派。"""

from evaluation.gsm8k import evaluate_gsm8k
from evaluation.mmlu import evaluate_mmlu


def run_benchmark(name: str, **kwargs):
    if name == "mmlu":
        return evaluate_mmlu(**kwargs)
    if name == "gsm8k":
        return evaluate_gsm8k(**kwargs)
    raise ValueError(f"未知的评测目标: {name}")


__all__ = ["run_benchmark"]
