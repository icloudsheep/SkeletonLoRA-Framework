"""统一的对象大小度量: pickle.dumps 后取长度,兼容 bytes / dict / 任意对象。

选它是为了让明文 / 密文 / 下发三种大小可直接比较,不引入不同度量。
"""

import pickle
from typing import Any


def sizeof(obj: Any) -> int:
    return len(pickle.dumps(obj))
