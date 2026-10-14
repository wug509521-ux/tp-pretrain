"""
训练相关模块。

当前提供：
- 文本 seq2seq baseline 训练循环
- SOM 辅助监督的占位接口（可选择开启/关闭）
"""

from .losses import LossConfig
from .trainer import Seq2SeqTrainer

__all__ = ["LossConfig", "Seq2SeqTrainer"]



