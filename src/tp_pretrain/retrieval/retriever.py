from __future__ import annotations

"""
简单规则检索器。

根据 substrate SMILES 与规则示例分子的 Tanimoto 相似度进行检索，
返回 top-k 规则及其相似度分数。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from tp_pretrain.utils.chemistry import canonicalize_smiles, is_valid_smiles

from .index import RuleLibraryIndex
from .schema import RuleRecord
from .scorer import score_substrate_against_examples


@dataclass
class RetrievalConfig:
    top_k: int = 10
    min_similarity: float = 0.0


class RuleRetriever:
    """
    基于 Tanimoto 相似度的最小检索器。
    """

    def __init__(self, index: RuleLibraryIndex, config: Optional[RetrievalConfig] = None) -> None:
        self.index = index
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        substrate_smiles: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        根据 substrate_smiles 在规则库中检索 top-k 规则。

        返回结构：
        [
          {
            "rule": RuleRecord,
            "score": float,
          },
          ...
        ]
        """
        top_k = top_k or self.config.top_k

        can_sub = canonicalize_smiles(substrate_smiles)
        if not can_sub or not is_valid_smiles(can_sub):
            # 非法输入，直接返回空结果
            return []

        scores = score_substrate_against_examples(can_sub, self.index.example_smiles)

        scored_indices = [
            (i, s) for i, s in enumerate(scores) if s >= self.config.min_similarity
        ]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict] = []
        for idx, score in scored_indices[:top_k]:
            rule = self.index.get(idx)
            results.append({"rule": rule, "score": float(score)})

        return results


__all__ = ["RetrievalConfig", "RuleRetriever"]



