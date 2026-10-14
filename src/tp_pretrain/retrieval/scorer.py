from __future__ import annotations

"""
相似度打分工具。

当前使用 RDKit Morgan 指纹 + Tanimoto 相似度，
封装成对规则示例分子进行逐条打分的简单函数。
"""

from typing import Iterable, List, Optional, Sequence

from tp_pretrain.utils.chemistry import tanimoto_similarity


def score_substrate_against_examples(
    substrate_smiles: str,
    example_smiles_list: Sequence[Optional[str]],
) -> List[float]:
    """
    针对一批规则示例分子，计算与给定 substrate 的 Tanimoto 相似度。

    - 若示例分子 SMILES 为空或非法，相似度返回 0.0
    - 内部调用 utils.chemistry.tanimoto_similarity
    """
    scores: List[float] = []
    for ex_smi in example_smiles_list:
        if not ex_smi:
            scores.append(0.0)
            continue
        sim = tanimoto_similarity(substrate_smiles, ex_smi)
        scores.append(float(sim))
    return scores


__all__ = ["score_substrate_against_examples"]



