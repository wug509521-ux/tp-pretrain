from __future__ import annotations

"""
随机种子工具，确保实验可复现。
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    设置 Python / NumPy / PyTorch 的随机种子。

    参数：
    - seed: 随机种子整数
    - deterministic: 是否强制使用确定性 CUDA 算子（可能影响性能）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = ["set_seed"]



