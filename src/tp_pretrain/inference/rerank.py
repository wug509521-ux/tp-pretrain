from __future__ import annotations

"""
预测结果 reranking 模块。

对模型生成的候选 predicted_smiles 进行打分重排，当前使用线性组合：
- model_score           （模型原始分数）
- rule_similarity       （substrate 与规则示例分子相似度的代表值，如 max）
- is_valid_smiles       （0/1）
- mass_change_is_ok     （0/1，基于分子量变化是否在合理范围）
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from tp_pretrain.utils.chemistry import exact_mass, is_valid_smiles
from tp_pretrain.retrieval.index import RuleLibraryIndex
from tp_pretrain.retrieval.scorer import score_substrate_against_examples


@dataclass
class RerankConfig:
    # 各特征的线性权重
    w_model_score: float = 1.0
    w_rule_similarity: float = 1.0
    w_valid_smiles: float = 1.0
    w_mass_change: float = 1.0

    # 判断“合理”分子量变化的阈值（单位：Da）
    min_mass_delta: float = 0.0
    max_mass_delta: float = 500.0


def _compute_rule_similarity(
    substrate_smiles: str,
    rule_index: Optional[RuleLibraryIndex],
) -> float:
    """
    计算 substrate 与规则示例分子的整体相似度。

    当前采用简单的 max Tanimoto，相当于“是否存在一个与底物相似的规则示例分子”。
    若未提供规则索引，则返回 0.0。
    """
    if rule_index is None or len(rule_index) == 0:
        return 0.0
    scores = score_substrate_against_examples(substrate_smiles, rule_index.example_smiles)
    return max(scores) if scores else 0.0


def _mass_change_feature(
    substrate_smiles: str,
    pred_smiles: str,
    cfg: RerankConfig,
) -> float:
    """
    基于精确分子质量的变化是否在 [min_delta, max_delta] 内产生一个 0/1 特征。
    """
    m_sub = exact_mass(substrate_smiles)
    m_pred = exact_mass(pred_smiles)
    if m_sub is None or m_pred is None:
        return 0.0
    delta = abs(m_pred - m_sub)
    return 1.0 if (cfg.min_mass_delta <= delta <= cfg.max_mass_delta) else 0.0


def rerank_predictions(
    prediction_items: Sequence[Dict],
    cfg: Optional[RerankConfig] = None,
    rule_index: Optional[RuleLibraryIndex] = None,
) -> List[Dict]:
    """
    对一批预测结果进行 rerank，返回带有 rerank_score / rerank_rank 的新对象。

    输入格式与 evaluate_predictions 中描述的一致：
    {
      "reaction_id": ...,
      "substrate_smiles": ...,
      "gold_product_smiles": ... 或 None,
      "predictions": [
        {"rank": 1, "predicted_smiles": "...", "score": float 或 None},
        ...
      ]
    }
    """
    cfg = cfg or RerankConfig()
    out_items: List[Dict] = []

    for item in prediction_items:
        substrate = item.get("substrate_smiles", "")
        preds = item.get("predictions", []) or []

        # 底物与规则库的相似度（对该样本内所有候选相同）
        rule_sim = _compute_rule_similarity(substrate, rule_index)

        # 为每个 prediction 计算 rerank_score
        for p in preds:
            model_score = float(p.get("score") or 0.0)
            smi = p.get("predicted_smiles") or ""

            valid_flag = 1.0 if is_valid_smiles(smi) else 0.0
            mass_ok = _mass_change_feature(substrate, smi, cfg)

            rerank_score = (
                cfg.w_model_score * model_score
                + cfg.w_rule_similarity * rule_sim
                + cfg.w_valid_smiles * valid_flag
                + cfg.w_mass_change * mass_ok
            )
            p["rerank_score"] = float(rerank_score)

        # 按 rerank_score 从大到小排序，给出 rerank_rank
        preds_sorted = sorted(preds, key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
        for new_rank, p in enumerate(preds_sorted, start=1):
            p["rerank_rank"] = new_rank

        # 保留原有结构，只是 predictions 内部元素增加了字段
        new_item = dict(item)
        new_item["predictions"] = preds_sorted
        out_items.append(new_item)

    return out_items


__all__ = ["RerankConfig", "rerank_predictions"]



