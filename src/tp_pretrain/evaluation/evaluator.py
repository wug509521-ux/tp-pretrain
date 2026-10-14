from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tp_pretrain.utils.chemistry import (
    canonicalize_smiles,
    is_valid_smiles,
    molecular_formula,
    tanimoto_similarity,
)


@dataclass
class EvaluatedExample:
    reaction_id: str
    substrate_smiles: str
    gold_smiles: Optional[str]
    top1_pred: Optional[str]
    top1_valid: bool
    top1_exact_match: bool
    top1_formula_match: bool
    tanimoto_top1: float
    exact_match_rank: Optional[int]
    formula_match_rank: Optional[int]


TOPK_LIST: Tuple[int, ...] = (1, 3, 5, 10)


def _safe_canonicalize(smiles: Optional[str]) -> Optional[str]:
    if not smiles:
        return None
    return canonicalize_smiles(smiles)


def evaluate_predictions(
    prediction_items: Sequence[Dict],
) -> Tuple[Dict[str, float], List[EvaluatedExample]]:
    """
    评估预测结果。

    prediction_items 每条记录格式示例：
    {
      "reaction_id": ...,
      "substrate_smiles": ...,
      "gold_product_smiles": ...,
      "predictions": [
        {"predicted_smiles": "...", "score": ..., ...},
        ...
      ]
    }

    注意：
    - 不依赖 predictions 中是否存在 rank 字段
    - 直接使用 predictions 列表顺序作为最终 rank（1-based）
    """
    n_examples = len(prediction_items)

    valid_smiles_total = 0
    total_predictions = 0

    exact_match_counts = {k: 0 for k in TOPK_LIST}
    formula_match_counts = {k: 0 for k in TOPK_LIST}

    tanimoto_sum_top1 = 0.0
    tanimoto_count_top1 = 0

    rerank_exact_match_counts = {1: 0, 5: 0}

    evaluated_examples: List[EvaluatedExample] = []

    for item in prediction_items:
        reaction_id = str(item.get("reaction_id", ""))
        substrate = str(item.get("substrate_smiles", ""))
        gold_raw = item.get("gold_product_smiles")

        gold_can = _safe_canonicalize(gold_raw)
        gold_formula = molecular_formula(gold_can) if gold_can else None

        preds = item.get("predictions", []) or []

        preds_can: List[Optional[str]] = []
        for p in preds:
            smi = p.get("predicted_smiles")
            total_predictions += 1
            can = _safe_canonicalize(smi)
            preds_can.append(can)
            if can is not None and is_valid_smiles(can):
                valid_smiles_total += 1

        top1_pred_can: Optional[str] = None
        top1_valid = False
        top1_exact = False
        top1_formula_eq = False
        tanimoto_top1 = 0.0

        if preds_can:
            top1_pred_can = preds_can[0]
            top1_valid = bool(top1_pred_can and is_valid_smiles(top1_pred_can))

        exact_rank: Optional[int] = None
        formula_rank: Optional[int] = None

        if gold_can:
            for i, can in enumerate(preds_can, start=1):
                if can is None or not is_valid_smiles(can):
                    continue

                if exact_rank is None and can == gold_can:
                    exact_rank = i

                if formula_rank is None and gold_formula:
                    pred_formula = molecular_formula(can)
                    if pred_formula and pred_formula == gold_formula:
                        formula_rank = i

                if exact_rank is not None and formula_rank is not None:
                    break

            if exact_rank is not None:
                for k in TOPK_LIST:
                    if exact_rank <= k:
                        exact_match_counts[k] += 1

            if formula_rank is not None:
                for k in TOPK_LIST:
                    if formula_rank <= k:
                        formula_match_counts[k] += 1

            if top1_pred_can and is_valid_smiles(top1_pred_can):
                top1_exact = top1_pred_can == gold_can
                if gold_formula:
                    pred_formula_top1 = molecular_formula(top1_pred_can)
                    top1_formula_eq = bool(
                        pred_formula_top1 and pred_formula_top1 == gold_formula
                    )

                sim = tanimoto_similarity(gold_can, top1_pred_can)
                tanimoto_top1 = sim
                tanimoto_sum_top1 += sim
                tanimoto_count_top1 += 1

            preds_with_rrank = []
            for i, p in enumerate(preds, start=1):
                rr = p.get("rerank_rank")
                if rr is None:
                    rr = i
                preds_with_rrank.append((int(rr), p))

            preds_rerank_sorted = sorted(preds_with_rrank, key=lambda x: x[0])

            rerank_hit_rank = None
            for rr, p in preds_rerank_sorted:
                can = _safe_canonicalize(p.get("predicted_smiles"))
                if can is None or not is_valid_smiles(can):
                    continue
                if can == gold_can:
                    rerank_hit_rank = rr
                    break

            if rerank_hit_rank is not None:
                for k in (1, 5):
                    if rerank_hit_rank <= k:
                        rerank_exact_match_counts[k] += 1

        evaluated_examples.append(
            EvaluatedExample(
                reaction_id=reaction_id,
                substrate_smiles=substrate,
                gold_smiles=gold_can,
                top1_pred=top1_pred_can,
                top1_valid=top1_valid,
                top1_exact_match=top1_exact,
                top1_formula_match=top1_formula_eq,
                tanimoto_top1=tanimoto_top1,
                exact_match_rank=exact_rank,
                formula_match_rank=formula_rank,
            )
        )

    metrics: Dict[str, float] = {}
    metrics["valid_smiles_rate"] = (
        valid_smiles_total / total_predictions if total_predictions > 0 else 0.0
    )

    for k in TOPK_LIST:
        metrics[f"exact_match_top{k}"] = (
            exact_match_counts[k] / n_examples if n_examples > 0 else 0.0
        )
        metrics[f"formula_match_top{k}"] = (
            formula_match_counts[k] / n_examples if n_examples > 0 else 0.0
        )

    metrics["avg_tanimoto_top1"] = (
        tanimoto_sum_top1 / tanimoto_count_top1 if tanimoto_count_top1 > 0 else 0.0
    )

    metrics["rerank_exact_match_top1"] = (
        rerank_exact_match_counts[1] / n_examples if n_examples > 0 else 0.0
    )
    metrics["rerank_exact_match_top5"] = (
        rerank_exact_match_counts[5] / n_examples if n_examples > 0 else 0.0
    )

    return metrics, evaluated_examples


def save_metrics_json(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def save_summary_csv(path: Path, evaluated_examples: Sequence[EvaluatedExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reaction_id",
        "substrate_smiles",
        "gold_smiles",
        "top1_pred",
        "top1_valid",
        "top1_exact_match",
        "top1_formula_match",
        "tanimoto_top1",
        "exact_match_rank",
        "formula_match_rank",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ex in evaluated_examples:
            writer.writerow(
                {
                    "reaction_id": ex.reaction_id,
                    "substrate_smiles": ex.substrate_smiles,
                    "gold_smiles": ex.gold_smiles or "",
                    "top1_pred": ex.top1_pred or "",
                    "top1_valid": int(bool(ex.top1_valid)),
                    "top1_exact_match": int(bool(ex.top1_exact_match)),
                    "top1_formula_match": int(bool(ex.top1_formula_match)),
                    "tanimoto_top1": ex.tanimoto_top1,
                    "exact_match_rank": ex.exact_match_rank if ex.exact_match_rank is not None else "",
                    "formula_match_rank": ex.formula_match_rank if ex.formula_match_rank is not None else "",
                }
            )


__all__ = [
    "EvaluatedExample",
    "evaluate_predictions",
    "save_metrics_json",
    "save_summary_csv",
]
