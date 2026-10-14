#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Top-k single-step predictions against reaction-pair labels."""

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def canonical(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    return Chem.MolToSmiles(mol, canonical=True) if mol else ""


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prediction_smiles(prediction):
    if isinstance(prediction, str):
        return prediction
    return (
        prediction.get("predicted_smiles")
        or prediction.get("product_smiles")
        or prediction.get("smiles")
        or ""
    )


def tanimoto(a, b):
    ma = Chem.MolFromSmiles(a) if a else None
    mb = Chem.MolFromSmiles(b) if b else None
    if ma is None or mb is None:
        return ""
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-jsonl", required=True)
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    gold_rows = load_jsonl(Path(args.gold_jsonl))
    pred_rows = load_jsonl(Path(args.prediction_jsonl))
    gold_by_id = {str(row.get("reaction_id")): row for row in gold_rows}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail = []
    max_k = 20
    for pred_row in pred_rows:
        reaction_id = str(pred_row.get("reaction_id"))
        gold_row = gold_by_id.get(reaction_id, {})
        gold = canonical(gold_row.get("product_smiles"))
        predictions = pred_row.get("predictions", []) or []
        candidates = []
        for prediction in predictions:
            value = canonical(prediction_smiles(prediction))
            if value and value not in candidates:
                candidates.append(value)
        max_k = max(max_k, len(candidates))
        first_rank = ""
        for rank, candidate in enumerate(candidates, 1):
            if candidate == gold:
                first_rank = rank
                break
        row = {
            "model": args.model_name,
            "reaction_id": reaction_id,
            "gold_canonical_smiles": gold,
            "top1_hit": int(bool(candidates and candidates[0] == gold)),
            "top5_hit": int(gold in candidates[:5]),
            "top10_hit": int(gold in candidates[:10]),
            "top20_hit": int(gold in candidates[:20]),
            "first_hit_rank": first_rank,
            "top1_tanimoto": tanimoto(candidates[0], gold) if candidates and gold else "",
            "n_valid_candidates": len(candidates),
        }
        for rank in range(1, max_k + 1):
            row[f"top{rank}_canonical_smiles"] = candidates[rank - 1] if rank <= len(candidates) else ""
        detail.append(row)

    detail_path = output_dir / f"{args.model_name}_per_sample.csv"
    if detail:
        fields = list(detail[0])
        with detail_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(detail)

    n = len(detail)
    metrics = {
        "model": args.model_name,
        "n_samples": n,
        "top1_hit_count": sum(row["top1_hit"] for row in detail),
        "top5_hit_count": sum(row["top5_hit"] for row in detail),
        "top10_hit_count": sum(row["top10_hit"] for row in detail),
        "top20_hit_count": sum(row["top20_hit"] for row in detail),
        "top1_hit_rate": sum(row["top1_hit"] for row in detail) / n if n else 0.0,
        "top5_hit_rate": sum(row["top5_hit"] for row in detail) / n if n else 0.0,
        "top10_hit_rate": sum(row["top10_hit"] for row in detail) / n if n else 0.0,
        "top20_hit_rate": sum(row["top20_hit"] for row in detail) / n if n else 0.0,
        "prediction_jsonl": str(Path(args.prediction_jsonl)),
        "gold_jsonl": str(Path(args.gold_jsonl)),
    }
    (output_dir / f"{args.model_name}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
