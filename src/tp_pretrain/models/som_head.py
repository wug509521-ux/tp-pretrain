from __future__ import annotations

"""
Reaction center 辅助分类头。

当前不做 token 与原子的严格对齐，而是从 som_atoms 派生粗粒度标签，
在 encoder 的 pooled representation 上做一个简单的多分类任务。
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AuxHeadConfig:
    """
    辅助分类头配置。

    - hidden_size: encoder 输出维度
    - num_classes: 分类类别数（默认 4，对应 som_count_bucket: 0,1,2,>=3）
    - loss_weight: lambda_aux，辅助损失的权重
    """

    hidden_size: int
    num_classes: int = 4
    loss_weight: float = 1.0


class ReactionCenterAuxHead(nn.Module):
    """
    简单的 reaction center 辅助分类头。

    输入：
    - pooled: Tensor [batch_size, hidden_size]，encoder pooled representation
    - labels: Tensor [batch_size]，som_count_bucket（0,1,2,3）

    输出：
    - dict(loss=..., logits=...)
    """

    def __init__(self, config: AuxHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(
        self,
        pooled: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        logits = self.classifier(pooled)  # [B, C]
        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            out["loss"] = loss
        return out


__all__ = ["AuxHeadConfig", "ReactionCenterAuxHead"]


