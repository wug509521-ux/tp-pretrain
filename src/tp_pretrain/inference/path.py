from __future__ import annotations

"""
多步路径展开与终止机制。

当前实现一个非常简化的单条路径展开器，并加入规则式终止条件：
- 若 top1 score 低于阈值则停止
- 若生成结果无效比例太高则停止
- 若生成结果与父节点完全相同则停止

同时预留未来接入学习型 stop head 的接口。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from tp_pretrain.inference.predictor import Seq2SeqTPPredictor
from tp_pretrain.utils.chemistry import canonicalize_smiles, is_valid_smiles


class TerminationReason(str, Enum):
    MAX_STEPS_REACHED = "max_steps_reached"
    LOW_TOP1_SCORE = "low_top1_score"
    HIGH_INVALID_RATIO = "high_invalid_ratio"
    NO_PROGRESS = "no_progress"  # 生成结果与父节点完全相同
    UNKNOWN = "unknown"


@dataclass
class PathExpansionConfig:
    max_steps: int = 3
    top1_score_threshold: float = -1e9  # 若模型分数低于该值则停止（根据具体打分范围调整）
    max_invalid_ratio: float = 0.8      # 若无效生成比例超过该值则停止
    stop_on_no_progress: bool = True    # 若 top1 生成与父节点相同则停止

    # 预留：未来可选择使用学习型 stop head 替代或补充规则
    use_learned_stop: bool = False


def _analyze_generation(
    parent_smiles: str,
    predictions: List[Dict],
    cfg: PathExpansionConfig,
) -> Dict:
    """
    分析一次生成结果，并根据规则判断是否需要终止。

    返回：
    {
      "terminate": bool,
      "reason": TerminationReason 或 None,
      "top1_score": float 或 None,
      "invalid_ratio": float,
      "top1_smiles": str 或 None,
    }
    """
    if not predictions:
        return {
            "terminate": True,
            "reason": TerminationReason.LOW_TOP1_SCORE,
            "top1_score": None,
            "invalid_ratio": 1.0,
            "top1_smiles": None,
        }

    # 以 rank=1 的候选作为 top1
    preds_sorted = sorted(predictions, key=lambda x: int(x.get("rank", 1)))
    top1 = preds_sorted[0]
    top1_score = float(top1.get("score") or 0.0)
    top1_smiles = top1.get("predicted_smiles") or ""

    # 统计无效比例
    total = len(preds_sorted)
    invalid = 0
    for p in preds_sorted:
        smi = p.get("predicted_smiles") or ""
        if not (is_valid_smiles(smi) and canonicalize_smiles(smi) is not None):
            invalid += 1
    invalid_ratio = invalid / total if total > 0 else 1.0

    # 终止规则 1：top1 score 过低
    if top1_score < cfg.top1_score_threshold:
        return {
            "terminate": True,
            "reason": TerminationReason.LOW_TOP1_SCORE,
            "top1_score": top1_score,
            "invalid_ratio": invalid_ratio,
            "top1_smiles": top1_smiles,
        }

    # 终止规则 2：无效比例过高
    if invalid_ratio > cfg.max_invalid_ratio:
        return {
            "terminate": True,
            "reason": TerminationReason.HIGH_INVALID_RATIO,
            "top1_score": top1_score,
            "invalid_ratio": invalid_ratio,
            "top1_smiles": top1_smiles,
        }

    # 终止规则 3：与父节点完全相同（无进展）
    parent_can = canonicalize_smiles(parent_smiles) or parent_smiles
    top1_can = canonicalize_smiles(top1_smiles) or top1_smiles
    if cfg.stop_on_no_progress and parent_can and top1_can and parent_can == top1_can:
        return {
            "terminate": True,
            "reason": TerminationReason.NO_PROGRESS,
            "top1_score": top1_score,
            "invalid_ratio": invalid_ratio,
            "top1_smiles": top1_smiles,
        }

    # 若以上规则都不触发，则继续展开
    return {
        "terminate": False,
        "reason": None,
        "top1_score": top1_score,
        "invalid_ratio": invalid_ratio,
        "top1_smiles": top1_smiles,
    }


def expand_path(
    predictor: Seq2SeqTPPredictor,
    start_smiles: str,
    cfg: Optional[PathExpansionConfig] = None,
) -> Dict:
    """
    从单个 substrate SMILES 展开一条多步路径（链式），并根据规则式终止机制停止。

    目前实现为：
    - 每一步只保留 top1 结果作为下一步的底物
    - 记录每一步的基本信息和终止原因

    返回结构示例：
    {
      "start_smiles": "...",
      "steps": [
        {"step": 0, "substrate": "...", "top1_product": "...", "top1_score": ...},
        {"step": 1, "substrate": "...", "top1_product": "...", "top1_score": ...},
      ],
      "termination_reason": "low_top1_score",
      "termination_step": 1
    }
    """
    from tp_pretrain.data.schema import TPReactionRecord  # 局部导入以避免循环

    cfg = cfg or PathExpansionConfig()
    steps: List[Dict] = []

    current_smiles = start_smiles
    termination_reason: TerminationReason = TerminationReason.UNKNOWN
    termination_step = 0

    for step in range(cfg.max_steps):
        record = TPReactionRecord(
            reaction_id=f"path_step_{step}",
            substrate_smiles=current_smiles,
            product_smiles="",
            domain="path_expansion",
            biosystem=None,
            condition={},
            template_id=None,
            rule_source=None,
            som_atoms=[],
            depth=1,
            metadata={},
        )

        gen_out = predictor.generate_for_record(record)
        preds = gen_out.get("predictions", []) or []

        analysis = _analyze_generation(current_smiles, preds, cfg)
        steps.append(
            {
                "step": step,
                "substrate": current_smiles,
                "top1_product": analysis.get("top1_smiles"),
                "top1_score": analysis.get("top1_score"),
                "invalid_ratio": analysis.get("invalid_ratio"),
            }
        )

        if analysis["terminate"]:
            termination_reason = analysis["reason"] or TerminationReason.UNKNOWN
            termination_step = step
            break

        # 更新当前底物，进入下一步
        current_smiles = analysis.get("top1_smiles") or current_smiles
        termination_step = step

    else:
        # for 循环正常结束（达到 max_steps），记为 MAX_STEPS_REACHED
        termination_reason = TerminationReason.MAX_STEPS_REACHED

    return {
        "start_smiles": start_smiles,
        "steps": steps,
        "termination_reason": termination_reason.value,
        "termination_step": termination_step,
    }


__all__ = ["TerminationReason", "PathExpansionConfig", "expand_path"]



