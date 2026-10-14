"""
通用工具模块。

当前包含：
- JSONL 读写工具（见 io.py）
- 化学相关工具（见 chemistry.py）
- 随机种子与日志工具（见 seed.py, logging_utils.py）
"""

from .seed import set_seed
from .logging_utils import setup_logging

__all__ = ["set_seed", "setup_logging"]


