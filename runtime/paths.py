"""RUN_ID 与目录准备,让 main 只调一个函数拿到全部路径。"""

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path


_TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"


@dataclass
class RunPaths:
    run_id: str
    log_dir: Path
    out_dir: Path
    metrics_dir: Path
    tb_dir: Path
    ckpt_root: Path


def prepare_run_paths(root: Path) -> RunPaths:
    run_id = _dt.datetime.now().strftime(_TIMESTAMP_FMT)
    log_dir = root / "logs" / run_id
    out_dir = root / "output" / run_id
    paths = RunPaths(
        run_id=run_id,
        log_dir=log_dir,
        out_dir=out_dir,
        metrics_dir=out_dir / "metrics",
        tb_dir=out_dir / "tensorboard",
        ckpt_root=out_dir / "checkpoints",
    )
    for d in (paths.log_dir, paths.metrics_dir, paths.tb_dir, paths.ckpt_root):
        d.mkdir(parents=True, exist_ok=True)
    return paths
