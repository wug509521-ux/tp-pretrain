from __future__ import annotations

"""
路径展开终止策略的学习型 head 占位。

当前项目只实现基于规则的简单终止机制（score 阈值、无效率、与父节点相同），
本文件为未来的“学习型 stop head”预留接口。
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class StopHeadConfig:
    """
    学习型 stop head 的基础配置占位。

    - hidden_size: encoder / 路径表示的维度
    """

    hidden_size: int


class StopHead(nn.Module):
    """
    简单的二分类占位 head：给定某一步的表示，输出“是否终止”的 logits / 概率。

    当前项目暂不在训练中使用，只作为未来扩展的统一接口。
    """

    def __init__(self, config: StopHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, features: torch.Tensor) -> dict:
        """
        features: [batch, hidden_size]

        返回：
        - logits: [batch, 2]
        - probs: [batch, 2]
        """
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}


__all__ = ["StopHeadConfig", "StopHead"]



