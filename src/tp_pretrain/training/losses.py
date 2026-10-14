from __future__ import annotations

"""
损失函数与配置。

当前支持：
- 纯生成损失（来自 Transformers 模型自带的 seq2seq loss）
- 生成损失 + reaction center 辅助分类损失
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LossConfig:
    """
    训练损失配置。

    - use_aux_loss: 是否启用辅助损失
    - aux_loss_weight: lambda_aux 辅助损失权重
    """

    use_aux_loss: bool = False
    aux_loss_weight: float = 1.0


def combine_losses(
    gen_loss: torch.Tensor,
    aux_loss: Optional[torch.Tensor],
    config: LossConfig,
) -> torch.Tensor:
    """
    将生成损失与辅助损失进行加权合并。
    """
    total = gen_loss
    if config.use_aux_loss and aux_loss is not None:
        total = total + config.aux_loss_weight * aux_loss
    return total


__all__ = ["LossConfig", "combine_losses"]


