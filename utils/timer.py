"""基于 time.perf_counter() 的上下文管理器计时器。"""

import time
from contextlib import contextmanager


class _Elapsed:
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value: float = 0.0


@contextmanager
def perf_timer():
    """用法:
        with perf_timer() as t:
            do_work()
        elapsed_seconds = t.value
    """
    e = _Elapsed()
    start = time.perf_counter()
    try:
        yield e
    finally:
        e.value = time.perf_counter() - start
