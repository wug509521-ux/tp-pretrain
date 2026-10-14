#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate reproduced multistep edges and paths against strict gold data."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem


def canonical(value):
    text = str(value or "").strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    return Chem.MolToSmiles(mol, canonical=True) if mol else ""


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_gold(gold_dir):
    gold_dir = Path(gold_dir)
    targets = defaultdict(set)
    target_rows = gold_dir / "gold_targets_by_step.jsonl"
    if target_rows.exists():
        for row in read_jsonl(target_rows):
            root = canonical(row.get("root_smiles"))
            step = int(row.get("step", 0))
            parents = row.get("parent_smiles", []) or []
            products = {canonical(x) for x in (row.get("gold_products", []) or [])}
            products.discard("")
            for parent in parents:
                parent_can = canonical(parent)
                if root and step and parent_can and products:
                    targets[(root, step, parent_can)].update(products)

    direct_edges = set()
    edge_path = gold_dir / "gold_edges.csv"
    if edge_path.exists():
        for row in read_csv(edge_path):
            parent = canonical(row.get("parent_smiles"))
            product = canonical(row.get("product_smiles"))
            if parent and product:
                direct_edges.add((parent, product))
    return targets, direct_edges


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(model_dir, targets, direct_edges):
    model_dir = Path(model_dir)
    nodes = read_csv(model_dir / "nodes.csv")
    edges = read_csv(model_dir / "edges.csv")
    node_root = {row["node_id"]: canonical(row.get("seed_smiles")) for row in nodes}
    edge_rows = []
    for row in edges:
        parent = canonical(row.get("parent_smiles"))
        product = canonical(row.get("child_smiles"))
        step = int(row.get("depth_to", 0))
        root = node_root.get(row.get("parent_node_id", ""), "")
        gold = targets.get((root, step, parent), set())
        match_mode = "root_step_parent" if gold else "direct_edge_fallback"
        matched = product in gold if gold else (parent, product) in direct_edges
        edge_rows.append({
            "model": model_dir.name,
            "edge_id": row.get("edge_id", ""),
            "seed_id": row.get("parent_node_id", "").split("-", 1)[0],
            "root_canonical_smiles": root,
            "step": step,
            "parent_canonical_smiles": parent,
            "predicted_canonical_smiles": product,
            "gold_label_available": int(bool(gold)),
            "matched": int(matched),
            "match_mode": match_mode,
            "candidate_rank": row.get("rank_within_parent", ""),
        })

    metrics = []
    for step in sorted({int(row["step"]) for row in edge_rows}):
        subset = [row for row in edge_rows if int(row["step"]) == step]
        labelled = [row for row in subset if row["gold_label_available"]]
        metrics.append({
            "model": model_dir.name,
            "level": "edge",
            "step": step,
            "n_edges": len(subset),
            "n_label_available": len(labelled),
            "n_hits": sum(row["matched"] for row in labelled),
            "hit_rate": (sum(row["matched"] for row in labelled) / len(labelled)) if labelled else "",
        })

    path_file = model_dir / "all_complete_paths.csv"
    path_rows = read_csv(path_file) if path_file.exists() else []
    path_results = []
    for row in path_rows:
        chain = [x.strip() for x in str(row.get("node_id_chain", "")).split("->") if x.strip()]
        edge_lookup = {str(x["child_node_id"]): x for x in edges}
        statuses = []
        for child_id in chain[1:]:
            edge = edge_lookup.get(child_id)
            if edge is None:
                statuses.append(0)
                continue
            parent = canonical(edge.get("parent_smiles"))
            product = canonical(edge.get("child_smiles"))
            root = node_root.get(edge.get("parent_node_id", ""), "")
            gold = targets.get((root, int(edge.get("depth_to", 0)), parent), set())
            statuses.append(int(product in gold if gold else (parent, product) in direct_edges))
        path_results.append({
            "model": model_dir.name,
            "seed_id": row.get("seed_id", ""),
            "leaf_node_id": row.get("leaf_node_id", ""),
            "path_length": row.get("path_length", ""),
            "all_edges_hit": int(bool(statuses) and all(statuses)),
            "hit_edge_count": sum(statuses),
            "edge_count": len(statuses),
            "node_id_chain": row.get("node_id_chain", ""),
        })
    if path_results:
        metrics.append({
            "model": model_dir.name,
            "level": "complete_path",
            "step": max(int(x.get("path_length", 0)) for x in path_results),
            "n_edges": len(path_results),
            "n_label_available": len(path_results),
            "n_hits": sum(x["all_edges_hit"] for x in path_results),
            "hit_rate": sum(x["all_edges_hit"] for x in path_results) / len(path_results),
        })
    return edge_rows, path_results, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    targets, direct_edges = load_gold(args.gold_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_edges, all_paths, all_metrics = [], [], []
    for model_dir in sorted(x for x in prediction_dir.iterdir() if x.is_dir() and (x / "edges.csv").exists()):
        edges, paths, metrics = evaluate_model(model_dir, targets, direct_edges)
        all_edges.extend(edges)
        all_paths.extend(paths)
        all_metrics.extend(metrics)
    write_csv(output_dir / "edge_evaluation.csv", all_edges, list(all_edges[0]) if all_edges else ["model"])
    write_csv(output_dir / "path_evaluation.csv", all_paths, list(all_paths[0]) if all_paths else ["model"])
    write_csv(output_dir / "multistep_metrics.csv", all_metrics, list(all_metrics[0]) if all_metrics else ["model"])
    summary = {
        "prediction_dir": str(prediction_dir),
        "gold_dir": str(Path(args.gold_dir)),
        "n_gold_parent_step_groups": len(targets),
        "n_gold_direct_edges": len(direct_edges),
        "models_evaluated": sorted({row["model"] for row in all_metrics}),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
