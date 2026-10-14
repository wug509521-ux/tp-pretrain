"""
模型相关模块。

当前仅提供 reaction center 辅助分类头接口，后续可在此目录中加入：
- 纯文本 seq2seq baseline 封装
- 图编码器（例如基于 RDKit / GNN 的分子表示）
"""

from .som_head import AuxHeadConfig, ReactionCenterAuxHead
from .stop_head import StopHeadConfig, StopHead
from .tokenizer import build_input_text

__all__ = ["AuxHeadConfig", "ReactionCenterAuxHead", "StopHeadConfig", "StopHead", "build_input_text"]


