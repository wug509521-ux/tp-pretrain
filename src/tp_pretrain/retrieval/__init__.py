"""
规则 / 模板检索模块。

当前实现一个基于 Morgan 指纹相似度的简单检索器，
用于根据 substrate SMILES 在规则库中检索候选规则。

后续可以在此基础上接入 RetroRules / MicrobeRX 等真实规则库，
并替换为更高效的索引结构（例如 Faiss、Annoy 等）。
"""

from .schema import RuleRecord
from .index import RuleLibraryIndex
from .retriever import RuleRetriever

__all__ = ["RuleRecord", "RuleLibraryIndex", "RuleRetriever"]



