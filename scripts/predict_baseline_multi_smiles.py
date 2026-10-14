#!/usr/bin/env python
from __future__ import annotations
from types import SimpleNamespace
import argparse
import copy
import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from rdkit import Chem

from tp_pretrain.inference.predictor import GenerationConfig, Seq2SeqTPPredictor
from tp_pretrain.data.schema import TPReactionRecord

LOGGER = logging.getLogger("predict_baseline_multi_smiles")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, items: List[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def canonicalize_smiles(smiles: str) -> str:
    smiles = (smiles or "").strip()
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def enumerate_random_smiles(smiles: str, n: int, seed: int) -> List[str]:
    """
    生成唯一 SMILES 变体，固定采用：
    1 条 canonical + (n-1) 条 random unique

    说明：
    - 当 n=1 时，仅返回 canonical
    - 当 n>1 时，第 1 条永远是 canonical
    - 后续尽量补足随机 unique SMILES
    """
    smiles = (smiles or "").strip()
    if not smiles:
        return [""]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    canonical = Chem.MolToSmiles(mol, canonical=True)
    if n <= 1:
        return [canonical]

    rng = random.Random(seed)

    # 用 list + set，保证 canonical 固定排第一，同时维持唯一性
    variants = [canonical]
    seen = {canonical}

    # 适当放大尝试次数，避免 random 重复太多时补不满
    max_trials = max((n - 1) * 20, 100)

    for _ in range(max_trials):
        if len(variants) >= n:
            break
        try:
            _ = rng.random()  # 保持不同样本 seed 的确定性扰动
            s = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            if s and s not in seen:
                variants.append(s)
                seen.add(s)
        except Exception:
            continue

    return variants[:n]


def get_score(pred: Dict[str, Any]) -> float:
    for key in ("score", "logprob", "prob", "confidence"):
        val = pred.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except Exception:
            continue
    return 0.0



def _to_tp_record(item):
    """Convert a raw jsonl dict into TPReactionRecord safely."""
    if isinstance(item, TPReactionRecord):
        return item
    if not isinstance(item, dict):
        raise TypeError(f"Expected dict or TPReactionRecord, got {type(item)!r}")

    from dataclasses import fields, MISSING

    rec = dict(item)

    # 常见字段别名兜底
    alias_map = {
        "substrate_smiles": ["substrate", "parent_smiles", "reactant_smiles", "input_smiles", "smiles"],
        "product_smiles": ["product", "tp_smiles", "target_smiles"],
    }
    for target, aliases in alias_map.items():
        if not rec.get(target):
            for a in aliases:
                if rec.get(a):
                    rec[target] = rec[a]
                    break

    # sludge/elemdelta 数据里通常没有这些字段，这里统一补默认值
    defaults = {
        "domain": "sludge",
        "biosystem": "activated_sludge",
        "condition": "unknown",
        "template_id": "unknown",
        "rule_source": "unknown",
    }
    for k, v in defaults.items():
        if rec.get(k) in (None, ""):
            rec[k] = v

    # 只保留 TPReactionRecord 真正接受的字段，自动丢弃 reaction_text 等多余字段
    allowed = {f.name for f in fields(TPReactionRecord)}
    clean = {k: rec[k] for k in allowed if k in rec}

    # 检查仍然缺失的必填字段
    required = []
    for f in fields(TPReactionRecord):
        if not f.init:
            continue
        if f.default is MISSING and f.default_factory is MISSING and f.name not in clean:
            required.append(f.name)

    if required:
        raise KeyError(
            f"_to_tp_record still missing required TPReactionRecord fields: {required}; "
            f"available keys: {sorted(rec.keys())}"
        )

    return TPReactionRecord(**clean)



def _pick_root_atoms_heuristic(mol: Chem.Mol, max_roots: int = 8) -> List[int]:
    candidates = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        anum = atom.GetAtomicNum()
        score = 0

        if anum in {7, 8, 9, 15, 16, 17, 35, 53}:
            score += 10

        for bond in atom.GetBonds():
            nbr = bond.GetOtherAtom(atom)
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                if atom.GetAtomicNum() == 6 and nbr.GetAtomicNum() in {8, 7, 16}:
                    score += 8
                if atom.GetAtomicNum() in {8, 7, 16} and nbr.GetAtomicNum() == 6:
                    score += 8

        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() in {7, 8, 9, 15, 16, 17, 35, 53}:
                score += 3

        if atom.GetIsAromatic():
            for nbr in atom.GetNeighbors():
                if not nbr.GetIsAromatic():
                    score += 2
        else:
            for nbr in atom.GetNeighbors():
                if nbr.GetIsAromatic():
                    score += 2

        if anum > 1:
            score += 1

        candidates.append((score, idx))

    candidates.sort(key=lambda x: (-x[0], x[1]))

    root_atoms = []
    seen = set()
    for score, idx in candidates:
        if idx not in seen:
            root_atoms.append(idx)
            seen.add(idx)
        if len(root_atoms) >= max_roots:
            break

    return root_atoms


def enumerate_rooted_smiles(smiles: str, n: int, seed: int) -> List[str]:
    smiles = (smiles or "").strip()
    if not smiles:
        return [""]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    canonical = Chem.MolToSmiles(mol, canonical=True)
    roots = _pick_root_atoms_heuristic(mol, max_roots=max(n * 3, 8))

    variants = []
    seen = {canonical}

    rng = random.Random(seed)
    roots = roots[:]
    rng.shuffle(roots)

    for ridx in roots:
        if len(variants) >= n:
            break
        try:
            s = Chem.MolToSmiles(
                mol,
                canonical=False,
                rootedAtAtom=int(ridx),
            )
            if s and s not in seen:
                variants.append(s)
                seen.add(s)
        except Exception:
            continue

    return variants[:n]


def enumerate_hybrid_smiles(smiles: str, n: int, seed: int) -> List[str]:
    smiles = (smiles or "").strip()
    if not smiles:
        return [""]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    canonical = Chem.MolToSmiles(mol, canonical=True)
    if n <= 1:
        return [canonical]

    variants = [canonical]
    seen = {canonical}

    random_vars = enumerate_random_smiles(smiles, max(3, n), seed)
    added_random = 0
    for s in random_vars:
        if s not in seen:
            variants.append(s)
            seen.add(s)
            added_random += 1
        if added_random >= 2 or len(variants) >= n:
            break

    rooted_vars = enumerate_rooted_smiles(smiles, max(3, n), seed + 100003)
    added_rooted = 0
    for s in rooted_vars:
        if s not in seen:
            variants.append(s)
            seen.add(s)
            added_rooted += 1
        if added_rooted >= 2 or len(variants) >= n:
            break

    if len(variants) < n:
        extra_random = enumerate_random_smiles(smiles, max(n * 2, 10), seed + 200003)
        for s in extra_random:
            if s not in seen:
                variants.append(s)
                seen.add(s)
            if len(variants) >= n:
                break

    return variants[:n]


def aggregate_predictions(
    original_records: List[Dict[str, Any]],
    augmented_records: List[Dict[str, Any]],
    raw_outputs: List[Dict[str, Any]],
    num_return_sequences: int,
    final_topk: int,
    aggregation: str,
) -> List[Dict[str, Any]]:
    buckets: Dict[int, Dict[str, Any]] = defaultdict(dict)
    fallback_raw: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for aug_rec, out_item in zip(augmented_records, raw_outputs):
        orig_id = aug_rec["_orig_id"]
        preds = out_item.get("predictions", []) or []

        for rank_idx, pred in enumerate(preds, start=1):
            pred_smiles_raw = (pred.get("predicted_smiles") or "").strip()
            canon = canonicalize_smiles(pred_smiles_raw)
            score = get_score(pred)
            rank_points = max(0, num_return_sequences - rank_idx + 1)

            if canon:
                if canon not in buckets[orig_id]:
                    buckets[orig_id][canon] = {
                        "predicted_smiles": canon,
                        "count": 0,
                        "best_score": -math.inf,
                        "sum_score": 0.0,
                        "score_n": 0,
                        "rank_points": 0.0,
                        "aug_hits": set(),
                    }
                item = buckets[orig_id][canon]
                item["count"] += 1
                item["best_score"] = max(item["best_score"], score)
                item["sum_score"] += score
                item["score_n"] += 1
                item["rank_points"] += rank_points
                item["aug_hits"].add(aug_rec["_aug_id"])
            elif pred_smiles_raw:
                fallback_raw[orig_id].append(
                    {
                        "predicted_smiles": pred_smiles_raw,
                        "score": score,
                        "rank": rank_idx,
                    }
                )

    aggregated_outputs: List[Dict[str, Any]] = []
    for idx, record in enumerate(original_records):
        merged = []
        for _, item in buckets.get(idx, {}).items():
            avg_score = item["sum_score"] / max(1, item["score_n"])
            merged.append(
                {
                    "predicted_smiles": item["predicted_smiles"],
                    "score": item["best_score"],
                    "best_score": item["best_score"],
                    "avg_score": avg_score,
                    "count": item["count"],
                    "rank_points": item["rank_points"],
                    "aug_hit_count": len(item["aug_hits"]),
                }
            )

        if aggregation == "score":
            merged.sort(
                key=lambda x: (
                    x["best_score"],
                    x["avg_score"],
                    x["count"],
                    x["rank_points"],
                ),
                reverse=True,
            )
        elif aggregation == "borda":
            merged.sort(
                key=lambda x: (
                    x["rank_points"],
                    x["count"],
                    x["best_score"],
                    x["avg_score"],
                ),
                reverse=True,
            )
        else:  # vote
            merged.sort(
                key=lambda x: (
                    x["count"],
                    x["rank_points"],
                    x["best_score"],
                    x["avg_score"],
                ),
                reverse=True,
            )

        final_preds = merged[:final_topk]

        if not final_preds and fallback_raw.get(idx):
            final_preds = fallback_raw[idx][:final_topk]

        out_item = {
            "reaction_id": record.get("reaction_id", f"record_{idx}"),
            "substrate_smiles": record.get("substrate_smiles", ""),
            "gold_product_smiles": record.get("product_smiles", ""),
            "predictions": final_preds,
            "metadata": record.get("metadata", {}),
            "smiles_augmentation_n": record.get("_aug_n", 1),
            "aggregation": aggregation,
        }

        # 兼容某些已有字段
        if "reaction_text" in record:
            out_item["reaction_text"] = record.get("reaction_text")

        aggregated_outputs.append(out_item)

    return aggregated_outputs



def demote_unchanged_top1(
    aggregated_outputs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    如果 top1 候选与 substrate 完全相同，则将其下调到后面。
    仅调整 predictions 列表顺序，不改候选集合。
    """
    out = []

    changed_cases = 0
    total_cases = 0

    for item in aggregated_outputs:
        total_cases += 1
        obj = copy.deepcopy(item)

        substrate = canonicalize_smiles(obj.get("substrate_smiles", "") or "")
        preds = obj.get("predictions", []) or []

        if len(preds) <= 1 or not substrate:
            out.append(obj)
            continue

        def _pred_can(pred: Dict[str, Any]) -> str:
            return canonicalize_smiles(pred.get("predicted_smiles", "") or "")

        top1_can = _pred_can(preds[0])

        if top1_can and top1_can == substrate:
            same_idx = []
            diff_idx = []

            for i, p in enumerate(preds):
                p_can = _pred_can(p)
                if p_can and p_can == substrate:
                    same_idx.append(i)
                else:
                    diff_idx.append(i)

            new_preds = [preds[i] for i in diff_idx] + [preds[i] for i in same_idx]

            if new_preds != preds:
                changed_cases += 1
                obj["predictions"] = new_preds

        out.append(obj)

    LOGGER.info(
        "Post-filter demote_unchanged_top1 applied on %d/%d records",
        changed_cases,
        total_cases,
    )
    return out

def main() -> None:
    setup_logging()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--input-jsonl", default=None)
    ap.add_argument("--output-jsonl", default=None)
    ap.add_argument("--smiles-augment", type=int, default=1, help="每个底物生成多少个随机 SMILES 变体")
    ap.add_argument("--augment-mode", choices=["random", "hybrid"], default="random", help="SMILES augmentation mode")
    ap.add_argument("--aggregation", choices=["vote", "score", "borda"], default="vote")
    ap.add_argument("--final-topk", type=int, default=None, help="聚合后保留多少个候选；默认与 num_return_sequences 相同")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    infer_cfg = cfg.get("inference", {})

    model_dir = args.model_dir or infer_cfg.get("model_dir")
    input_path = args.input_jsonl or infer_cfg.get("input_jsonl")
    output_path = args.output_jsonl or infer_cfg.get("output_jsonl")
    num_beams = int(infer_cfg.get("num_beams", 5))
    num_return_sequences = int(infer_cfg.get("num_return_sequences", 5))
    max_new_tokens = int(infer_cfg.get("max_new_tokens", 256))
    final_topk = args.final_topk or num_return_sequences

    if not model_dir:
        raise ValueError("model_dir is required, either from --model-dir or config")
    if not input_path:
        raise ValueError("input_jsonl is required, either from --input-jsonl or config")
    if not output_path:
        raise ValueError("output_jsonl is required, either from --output-jsonl or config")

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    LOGGER.info("Loading records from %s", input_path)
    original_records = load_jsonl(input_path)
    LOGGER.info("Loaded %d records", len(original_records))

    augmented_records: List[Dict[str, Any]] = []
    for idx, rec in enumerate(original_records):
        base_smiles = (rec.get("substrate_smiles") or "").strip()
        n_aug = max(1, args.smiles_augment)

        if args.augment_mode == "hybrid":
            variants = enumerate_hybrid_smiles(base_smiles, n_aug, args.seed + idx)
        else:
            variants = enumerate_random_smiles(base_smiles, n_aug, args.seed + idx)
        for aug_id, variant in enumerate(variants):
            aug_rec = copy.deepcopy(rec)
            aug_rec["substrate_smiles"] = variant
            aug_rec["_orig_id"] = idx
            aug_rec["_aug_id"] = aug_id
            aug_rec["_aug_n"] = len(variants)
            augmented_records.append(aug_rec)

    LOGGER.info(
        "Expanded to %d augmented inputs (smiles_augment=%d)",
        len(augmented_records),
        max(1, args.smiles_augment),
    )

    infer_cfg = cfg.get("inference", {}) if isinstance(cfg, dict) else {}

    predictor_cfg = GenerationConfig(
        model_dir=str(model_dir),
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
    )

    predictor = Seq2SeqTPPredictor(predictor_cfg)
    LOGGER.info(
        "Running generation with num_beams=%d, num_return_sequences=%d, max_new_tokens=%d, aggregation=%s",
        num_beams,
        num_return_sequences,
        max_new_tokens,
        args.aggregation,
    )
    predictor_inputs = [_to_tp_record(x) for x in augmented_records]
    raw_outputs = predictor.generate_for_records(predictor_inputs)

    aggregated_outputs = aggregate_predictions(
        original_records=original_records,
        augmented_records=augmented_records,
        raw_outputs=raw_outputs,
        num_return_sequences=num_return_sequences,
        final_topk=final_topk,
        aggregation=args.aggregation,
    )

    aggregated_outputs = demote_unchanged_top1(aggregated_outputs)

    total_preds = sum(len(x.get("predictions", [])) for x in aggregated_outputs)
    valid_preds = sum(
        1
        for x in aggregated_outputs
        for p in x.get("predictions", [])
        if canonicalize_smiles(p.get("predicted_smiles", ""))
    )
    valid_rate = valid_preds / total_preds if total_preds else 0.0

    LOGGER.info(
        "Aggregation finished: total_predictions=%d, valid_predictions=%d, valid_rate=%.3f",
        total_preds,
        valid_preds,
        valid_rate,
    )
    LOGGER.info("Writing predictions to %s", output_path)
    write_jsonl(output_path, aggregated_outputs)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
