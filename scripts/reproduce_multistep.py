#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the released three-step multistep protocols.

Each active node receives a complete Top-k list.  Only the highest-scoring
valid three candidates are expanded, while every returned candidate remains
in nodes.csv and edges.csv for auditability.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from rdkit import Chem


MODEL_NAMES = (
    "model_1", "model_2", "model_3", "model_4",
    "reaction_site_oracle", "fusion",
)


def resolve_models(values):
    if values == ["all"]:
        return list(MODEL_NAMES)
    unknown = [value for value in values if value not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")
    return values


def canonical(value):
    text = str(value or "").strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    return Chem.MolToSmiles(mol, canonical=True) if mol else ""


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_seeds(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        seed_id = str(row.get("seed_id") or row.get("id") or "").strip()
        smiles = str(row.get("seed_smiles") or row.get("smiles") or "").strip()
        if not seed_id or not smiles:
            continue
        result.append({
            "seed_id": seed_id,
            "seed_smiles": smiles,
            "seed_canonical_smiles": canonical(
                row.get("seed_canonical_smiles") or smiles
            ),
        })
    if not result:
        raise RuntimeError(f"No usable seeds found in {path}")
    return result


def load_center_map(path):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Oracle center map not found: {path}")
    rows = load_jsonl(path) if path.suffix.lower() == ".jsonl" else None
    if rows is None:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    mapping = {}
    for row in rows:
        centers = row.get("center_atoms") or row.get("explicit_center_atoms")
        if isinstance(centers, str):
            try:
                centers = json.loads(centers)
            except json.JSONDecodeError:
                centers = [x for x in centers.replace(",", " ").split() if x]
        centers = [int(x) for x in (centers or [])]
        keys = [
            str(row.get("node_id") or "").strip(),
            str(row.get("canonical_smiles") or "").strip(),
            canonical(row.get("substrate_smiles") or row.get("smiles") or ""),
        ]
        for key in keys:
            if key:
                mapping[key] = centers
    return mapping


def write_csv(path, rows, fieldnames):
    path = Path(path)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value, default=-1.0e18):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def prediction_rows(output_path):
    rows = load_jsonl(output_path)
    result = {}
    for row in rows:
        rid = str(row.get("reaction_id") or row.get("id") or "")
        result[rid] = row.get("predictions", []) or []
    return result


def run_subprocess(command, env):
    print("RUN:", " ".join(str(x) for x in command), flush=True)
    subprocess.run(command, check=True, env=env)


def make_model_input(node, model_name, center_bundle, center_map):
    base = dict(
        reaction_id=node["node_id"],
        substrate_smiles=node["smiles"],
        product_smiles="C",
        domain="biotransformation",
        biosystem="activated_sludge",
        root_parent=node["seed_id"],
        parent_node_id=node["node_id"],
    )
    if model_name == "model_4":
        centers = center_bundle(node["smiles"], 5)
        base["substrate_smiles"] = node["smiles"] + (
            " CENTER_ATOMS " + " ".join(map(str, centers)) if centers else ""
        )
        base["explicit_center_atoms"] = centers
        base["center_prompt_type"] = "predicted_reaction_site_top5"
    elif model_name == "reaction_site_oracle":
        centers = center_map.get(node["node_id"], [])
        if not centers:
            centers = center_map.get(node["canonical_smiles"], [])
        if not centers:
            raise RuntimeError(
                "Missing true reaction-site centers for oracle node "
                f"{node['node_id']} ({node['canonical_smiles']})"
            )
        base["substrate_smiles"] = node["smiles"] + " CENTER_ATOMS " + " ".join(map(str, centers))
        base["explicit_center_atoms"] = centers
        base["center_prompt_type"] = "gold_reaction_site_oracle"
    return base


def run_one_model(model_name, args, root, seeds, env):
    out = Path(args.output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)
    center_map = load_center_map(args.oracle_center_map) if model_name == "reaction_site_oracle" else {}
    if model_name == "reaction_site_oracle" and not center_map:
        raise RuntimeError(
            "Oracle multistep reproduction requires --oracle-center-map. "
            "Strict gold products do not contain atom indices and cannot be silently "
            "converted into oracle centers."
        )

    center_bundle = None
    if model_name == "model_4":
        import joblib
        from predict_fusion_oofhead_newmergedc3tail import predict_topk_centers
        bundle = joblib.load(root / "models" / "center_predictor")
        center_bundle = lambda smiles, topk: predict_topk_centers(smiles, bundle, topk)

    model_dirs = {
        "model_1": root / "models" / "model_1_no_pretraining",
        "model_2": root / "models" / "model_2_single_step_pretraining",
        "model_3": root / "models" / "model_3_two_step_pretraining",
        "model_4": root / "models" / "model_4_reaction_site_assisted",
        "reaction_site_oracle": root / "models" / "reaction_site_oracle",
    }
    baseline = root / "scripts" / "predict_baseline_multi_smiles.py"
    fusion = root / "scripts" / "predict_fusion_oofhead_newmergedc3tail.py"
    config_path = root / "configs" / "infer_top20.yaml"

    node_rows = []
    edge_rows = []
    node_by_id = {}
    expansions = []
    for seed in seeds:
        node = {
            "node_id": seed["seed_id"],
            "seed_id": seed["seed_id"],
            "seed_smiles": seed["seed_smiles"],
            "smiles": seed["seed_smiles"],
            "canonical_smiles": seed["seed_canonical_smiles"],
            "depth": 0,
            "parent_node_id": "",
            "parent_smiles": "",
            "rank_within_parent": "",
            "score": 0.0,
            "path_score": 0.0,
            "ancestor_canonical_smiles": [seed["seed_canonical_smiles"]],
            "is_seed": 1,
            "is_unchanged": 0,
            "is_duplicate": 0,
            "is_cycle": 0,
            "expanded": 1,
            "prediction_status": "seed",
        }
        node_rows.append(node)
        node_by_id[node["node_id"]] = node
        expansions.append(node)

    cache = {}
    depth_summaries = []
    edge_counter = 0
    with tempfile.TemporaryDirectory(prefix=f"repro_{model_name}_", dir=str(out)) as tmp:
        tmp = Path(tmp)
        for depth in range(1, args.max_depth + 1):
            if not expansions:
                break
            input_rows = []
            cache_keys = []
            uncached = []
            for node in expansions:
                prompt = make_model_input(node, model_name, center_bundle, center_map)
                key = (node["seed_id"], node["canonical_smiles"], prompt["substrate_smiles"])
                cache_keys.append((node, key))
                if key not in cache:
                    cache[key] = None
                    input_rows.append(prompt)
                    uncached.append((node, key))
            cache_hits = len(cache_keys) - len(uncached)

            pred_by_key = {}
            if input_rows:
                input_path = tmp / f"{model_name}_depth{depth}_input.jsonl"
                pred_path = tmp / f"{model_name}_depth{depth}_predictions.jsonl"
                write_jsonl(input_path, input_rows)
                if model_name == "fusion":
                    command = [
                        sys.executable, str(fusion),
                        "--project-root", str(root), "--python-bin", sys.executable,
                        "--input-jsonl", str(input_path), "--output-jsonl", str(pred_path),
                        "--head-model-dir", str(root / "models" / "fusion_head"),
                        "--tail-model-dir", str(root / "models" / "fusion_tail"),
                        "--center-predictor", str(root / "models" / "center_predictor"),
                        "--center-topk", "5", "--final-topk", str(args.single_step_topk),
                        "--config", str(config_path), "--smiles-augment", str(args.smiles_augment),
                        "--augment-mode", args.augment_mode, "--aggregation", args.aggregation,
                    ]
                else:
                    command = [
                        sys.executable, str(baseline),
                        "--config", str(config_path), "--model-dir", str(model_dirs[model_name]),
                        "--input-jsonl", str(input_path), "--output-jsonl", str(pred_path),
                        "--smiles-augment", str(args.smiles_augment),
                        "--augment-mode", args.augment_mode, "--aggregation", args.aggregation,
                        "--final-topk", str(args.single_step_topk), "--seed", str(args.seed),
                    ]
                run_subprocess(command, env)
                pred_by_id = prediction_rows(pred_path)
                for node, key in uncached:
                    cache[key] = pred_by_id.get(node["node_id"], [])

            next_expansions = []
            counts = defaultdict(int)
            depth_seen_by_seed = defaultdict(set)
            for node, key in cache_keys:
                predictions = cache.get(key) or []
                counts["n_parent_nodes"] += 1
                seen_siblings = set()
                selected = 0
                for rank, pred in enumerate(predictions[:args.single_step_topk], start=1):
                    raw = str(pred.get("predicted_smiles") or "").strip()
                    child_can = canonical(raw)
                    score = safe_float(pred.get("score"), -1.0e18)
                    duplicate = bool(
                        child_can
                        and (child_can in seen_siblings or child_can in depth_seen_by_seed[node["seed_id"]])
                    )
                    if child_can:
                        seen_siblings.add(child_can)
                        depth_seen_by_seed[node["seed_id"]].add(child_can)
                    unchanged = bool(child_can and child_can == node["canonical_smiles"])
                    cycle = bool(child_can and child_can in node["ancestor_canonical_smiles"])
                    valid = bool(child_can)
                    child_id = f"{node['node_id']}-{rank}"
                    child = {
                        "node_id": child_id,
                        "seed_id": node["seed_id"],
                        "seed_smiles": node["seed_smiles"],
                        "smiles": raw,
                        "canonical_smiles": child_can,
                        "depth": depth,
                        "parent_node_id": node["node_id"],
                        "parent_smiles": node["smiles"],
                        "rank_within_parent": rank,
                        "score": score,
                        "path_score": node["path_score"] + score - args.depth_penalty,
                        "ancestor_canonical_smiles": node["ancestor_canonical_smiles"] + [child_can],
                        "is_seed": 0,
                        "is_unchanged": int(unchanged),
                        "is_duplicate": int(duplicate),
                        "is_cycle": int(cycle),
                        "expanded": 0,
                        "prediction_status": "valid" if valid else "invalid_smiles",
                    }
                    if unchanged:
                        counts["n_unchanged"] += 1
                    if duplicate:
                        counts["n_duplicate_sibling"] += 1
                    if cycle:
                        counts["n_cycle"] += 1
                    if not valid:
                        counts["n_invalid_smiles"] += 1
                    can_expand = (
                        depth < args.max_depth and valid and not unchanged
                        and not duplicate and not cycle and selected < args.per_parent_expand
                    )
                    if can_expand:
                        child["expanded"] = 1
                        selected += 1
                        next_expansions.append(child)
                        counts["n_expanded_to_next_depth"] += 1
                    node_rows.append(child)
                    node_by_id[child_id] = child
                    edge_counter += 1
                    edge_rows.append({
                        "edge_id": edge_counter,
                        "parent_node_id": node["node_id"],
                        "child_node_id": child_id,
                        "depth_from": depth - 1,
                        "depth_to": depth,
                        "parent_smiles": node["smiles"],
                        "child_smiles": raw,
                        "parent_canonical_smiles": node["canonical_smiles"],
                        "child_canonical_smiles": child_can,
                        "step_score": score,
                        "path_score_child": child["path_score"],
                        "rank_within_parent": rank,
                        "source": model_name,
                    })
                counts["n_candidates"] += len(predictions[:args.single_step_topk])
            before_beam = len(next_expansions)
            expansions = apply_beam_width(next_expansions, args.beam_width)
            depth_summaries.append({
                "depth": depth,
                **dict(counts),
                "n_cache_hits": cache_hits,
                "n_expansions_before_beam": before_beam,
                "n_expansions_after_beam": len(expansions),
            })

    path_rows = []
    for leaf in node_rows:
        if leaf["depth"] != args.max_depth:
            continue
        chain = []
        current = leaf
        while current:
            chain.append(current)
            current = node_by_id.get(current["parent_node_id"])
        chain.reverse()
        path_rows.append({
            "seed_id": leaf["seed_id"],
            "seed_smiles": leaf["seed_smiles"],
            "leaf_node_id": leaf["node_id"],
            "leaf_smiles": leaf["smiles"],
            "leaf_canonical_smiles": leaf["canonical_smiles"],
            "path_length": len(chain) - 1,
            "path_score": leaf["path_score"],
            "node_id_chain": " -> ".join(x["node_id"] for x in chain),
            "smiles_chain": " >> ".join(x["smiles"] for x in chain),
            "canonical_smiles_chain": " >> ".join(x["canonical_smiles"] for x in chain),
            "step_score_chain": " | ".join(str(x["score"]) for x in chain[1:]),
        })
    path_rows.sort(key=lambda x: (x["seed_id"], -safe_float(x["path_score"])))
    top_paths = []
    for seed_id, group in _groupby_seed(path_rows):
        for rank, row in enumerate(group[:args.paths_topk], 1):
            row = dict(row)
            row["path_rank"] = rank
            top_paths.append(row)

    node_export = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                    for k, v in row.items() if k != "ancestor_canonical_smiles"}
                   for row in node_rows]
    write_csv(out / "nodes.csv", node_export, list(node_export[0].keys()))
    write_csv(out / "edges.csv", edge_rows, list(edge_rows[0].keys()) if edge_rows else ["edge_id"])
    write_csv(out / "all_complete_paths.csv", path_rows, list(path_rows[0].keys()) if path_rows else ["seed_id"])
    write_csv(out / "paths_topk.csv", top_paths, list(top_paths[0].keys()) if top_paths else ["path_rank"])

    summary = {
        "model": model_name,
        "n_seeds": len(seeds),
        "n_nodes": len(node_rows),
        "n_edges": len(edge_rows),
        "n_complete_paths": len(path_rows),
        "n_top_paths": len(top_paths),
        "max_depth": args.max_depth,
        "single_step_topk": args.single_step_topk,
        "per_parent_expand": args.per_parent_expand,
        "beam_width": args.beam_width,
        "paths_topk": args.paths_topk,
        "depth_penalty": args.depth_penalty,
        "depth_summaries": depth_summaries,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    config = vars(args).copy()
    config["model"] = model_name
    config["package_relative_protocol"] = {
        "suppress_unchanged": True,
        "loop_block": True,
        "dedup_scope": "per_seed",
        "prob_mode": "softmax_score",
        "prob_temperature": 1.0,
        "offline_inference": True,
    }
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _groupby_seed(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["seed_id"]].append(row)
    return sorted(groups.items())


def apply_beam_width(nodes, beam_width):
    """Keep the highest path-score expansion states independently per seed."""
    kept = []
    for _, group in _groupby_seed(nodes):
        group.sort(key=lambda x: safe_float(x.get("path_score")), reverse=True)
        kept.extend(group[:beam_width])
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--seed-file", default="data/external_multistep_571_seeds/drug_list_2_multistep_seeds.csv")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES),
                        help="model_1 model_2 model_3 model_4 reaction_site_oracle fusion, or all")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--single-step-topk", type=int, default=10)
    parser.add_argument("--per-parent-expand", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=40)
    parser.add_argument("--paths-topk", type=int, default=120)
    parser.add_argument("--depth-penalty", type=float, default=0.05)
    parser.add_argument("--smiles-augment", type=int, default=1)
    parser.add_argument("--augment-mode", choices=["random", "hybrid"], default="hybrid")
    parser.add_argument("--aggregation", choices=["vote", "score", "borda"], default="score")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--oracle-center-map", default="")
    parser.add_argument("--output-dir", default="outputs/reproduction/multistep")
    args = parser.parse_args()
    if args.max_depth != 3:
        raise ValueError("The released protocol requires --max-depth 3.")
    if args.single_step_topk != 10 or args.per_parent_expand != 3:
        raise ValueError("The released protocol requires Top-10 generation and Top-3 expansion.")

    args.models = resolve_models(args.models)
    root = Path(args.package_root).resolve()
    seed_file = Path(args.seed_file)
    if not seed_file.is_absolute():
        seed_file = root / seed_file
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = load_seeds(seed_file)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + str(root / "scripts") + os.pathsep + env.get("PYTHONPATH", "")
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    summaries = []
    for model in args.models:
        summaries.append(run_one_model(model, args, root, seeds, env))
    (output_dir / "summary_all_models.json").write_text(
        json.dumps({"n_seeds": len(seeds), "models": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("MULTISTEP_REPRODUCTION_FINISHED")
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
