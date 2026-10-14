from __future__ import annotations

"""
统一日志配置：
- 同时输出到控制台与指定 log 文件
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logging(log_file: Optional[str | Path] = None, level: int = logging.INFO) -> None:
    """
    配置根 logger：
    - 控制台输出（StreamHandler）
    - 可选的文件输出（FileHandler）
    """
    # 清空已有 handler，避免重复输出
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 控制台
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # 文件
    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(p, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


__all__ = ["setup_logging"]



