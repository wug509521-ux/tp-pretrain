# -*- coding: utf-8 -*-
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski


def load_jsonl(fp):
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(fp, rows):
    with open(fp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def canonicalize_smiles(smi: str) -> str:
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def mol_level_features(mol):
    if mol is None:
        return {
            "mol_heavy_atoms": 0,
            "mol_mw": 0.0,
            "mol_logp": 0.0,
            "mol_tpsa": 0.0,
            "mol_hbd": 0,
            "mol_hba": 0,
            "mol_rings": 0,
            "mol_rotb": 0,
        }
    return {
        "mol_heavy_atoms": mol.GetNumHeavyAtoms(),
        "mol_mw": float(Descriptors.MolWt(mol)),
        "mol_logp": float(Crippen.MolLogP(mol)),
        "mol_tpsa": float(Descriptors.TPSA(mol)),
        "mol_hbd": int(Lipinski.NumHDonors(mol)),
        "mol_hba": int(Lipinski.NumHAcceptors(mol)),
        "mol_rings": int(Lipinski.RingCount(mol)),
        "mol_rotb": int(Lipinski.NumRotatableBonds(mol)),
    }


def atom_features(mol, atom):
    idx = atom.GetIdx()
    n_atoms = max(mol.GetNumAtoms(), 1)
    neigh_atomic_nums = [n.GetAtomicNum() for n in atom.GetNeighbors()]

    feat = {
        "atom_idx_norm": idx / n_atoms,
        "atomic_num": atom.GetAtomicNum(),
        "degree": atom.GetDegree(),
        "total_degree": atom.GetTotalDegree(),
        "formal_charge": atom.GetFormalCharge(),
        "total_num_h": atom.GetTotalNumHs(),
        "total_valence": atom.GetTotalValence(),
        "num_radical_e": atom.GetNumRadicalElectrons(),
        "is_aromatic": int(atom.GetIsAromatic()),
        "is_in_ring": int(atom.IsInRing()),
        "mass": atom.GetMass(),
        "hybridization": int(atom.GetHybridization()),
        "chiral_tag": int(atom.GetChiralTag()),
        "is_hetero": int(atom.GetAtomicNum() not in [1, 6]),
    }

    for z in [6, 7, 8, 15, 16, 9, 17, 35, 53]:
        feat[f"neigh_Z{z}_count"] = sum(1 for x in neigh_atomic_nums if x == z)
    feat["neigh_hetero_count"] = sum(1 for x in neigh_atomic_nums if x not in [1, 6])

    feat.update(mol_level_features(mol))
    feat["atom_idx"] = idx
    return feat


def build_atom_feature_df(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.DataFrame()
    rows = []
    for atom in mol.GetAtoms():
        rows.append(atom_features(mol, atom))
    return pd.DataFrame(rows)


def predict_topk_centers(smiles: str, rf_bundle: dict, topk: int = 5):
    atom_df = build_atom_feature_df(smiles)
    if atom_df.empty:
        return []

    model = rf_bundle["model"]
    feature_cols = rf_bundle["feature_cols"]

    for c in feature_cols:
        if c not in atom_df.columns:
            atom_df[c] = 0

    X = atom_df[feature_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1]
    atom_df = atom_df.copy()
    atom_df["pred_prob"] = probs
    atom_df = atom_df.sort_values(["pred_prob", "atom_idx"], ascending=[False, True])

    return [int(x) for x in atom_df["atom_idx"].head(topk).tolist()]


def ensure_baseline_record(rec: dict):
    out = dict(rec)
    out.setdefault("reaction_id", out.get("_orig_id", "unknown"))
    out.setdefault("substrate_smiles", "")
    out.setdefault("product_smiles", "C")
    out.setdefault("domain", "biotransformation")
    out.setdefault("biosystem", "activated_sludge")
    out.setdefault("condition", "")
    out.setdefault("rule_source", "")
    out.setdefault("template_id", "")
    return out


def add_explicit_center_prompt(records, rf_bundle, topk=5, tag="CENTER_ATOMS"):
    out = []
    for r in records:
        rr = ensure_baseline_record(r)
        smi = rr["substrate_smiles"]
        centers = predict_topk_centers(smi, rf_bundle, topk=topk)
        if centers:
            rr["substrate_smiles"] = f"{smi} {tag} " + " ".join(map(str, centers))
        else:
            rr["substrate_smiles"] = smi
        rr["explicit_center_atoms"] = centers
        rr["center_prompt_type"] = f"oof_explicit_predcenter_top{len(centers)}"
        out.append(rr)
    return out


def resolve_path(project_root: str, maybe_path: str):
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return Path(project_root) / maybe_path


def run_single_predict(
    pybin,
    project_root,
    model_dir,
    input_jsonl,
    output_jsonl,
    config_path,
    smiles_augment,
    augment_mode,
    aggregation,
    final_topk,
):
    cmd = [
        pybin,
        str(Path(project_root) / "scripts" / "predict_baseline_multi_smiles.py"),
        "--config", str(resolve_path(project_root, config_path)),
        "--model-dir", str(model_dir),
        "--input-jsonl", str(input_jsonl),
        "--output-jsonl", str(output_jsonl),
        "--smiles-augment", str(smiles_augment),
        "--augment-mode", str(augment_mode),
        "--aggregation", str(aggregation),
        "--final-topk", str(final_topk),
    ]
    subprocess.run(cmd, check=True)


def normalize_pred_smiles(pred):
    if isinstance(pred, str):
        return canonicalize_smiles(pred)
    smi = (
        pred.get("predicted_smiles")
        or pred.get("product_smiles")
        or pred.get("smiles")
        or ""
    )
    return canonicalize_smiles(smi)


def pred_score(pred):
    if isinstance(pred, dict):
        if pred.get("score") is not None:
            return pred["score"]
        if pred.get("prob") is not None:
            return pred["prob"]
        if pred.get("path_score") is not None:
            return pred["path_score"]
    return None


def fuse_predictions(head_rows, tail_rows, final_topk=20):
    fused_rows = []
    for h, t in zip(head_rows, tail_rows):
        if h["reaction_id"] != t["reaction_id"]:
            raise RuntimeError(f"reaction_id mismatch: {h['reaction_id']} vs {t['reaction_id']}")

        fused = []
        seen = set()

        # head top1
        for p in h.get("predictions", [])[:1]:
            smi = normalize_pred_smiles(p)
            if smi and smi not in seen:
                q = dict(p) if isinstance(p, dict) else {}
                q["predicted_smiles"] = smi
                q["product_smiles"] = smi
                q["score"] = pred_score(p)
                q["fusion_source"] = "head_top1"
                fused.append(q)
                seen.add(smi)

        # tail fill
        for p in t.get("predictions", []):
            smi = normalize_pred_smiles(p)
            if smi and smi not in seen:
                q = dict(p) if isinstance(p, dict) else {}
                q["predicted_smiles"] = smi
                q["product_smiles"] = smi
                q["score"] = pred_score(p)
                q["fusion_source"] = "tail_fill"
                fused.append(q)
                seen.add(smi)
            if len(fused) >= final_topk:
                break

        for i, p in enumerate(fused, start=1):
            p["rank"] = i

        fused_rows.append({
            "reaction_id": h["reaction_id"],
            "substrate_smiles": t["substrate_smiles"],
            "predictions": fused[:final_topk],
            "source": t.get("source", h.get("source", "")),
        })
    return fused_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)

    # 下面这些参数既支持独立运行，也支持被多步主脚本调用
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--head-model-dir", required=True)
    ap.add_argument("--tail-model-dir", required=True)
    ap.add_argument("--center-predictor", required=True)
    ap.add_argument("--center-topk", type=int, default=5)
    ap.add_argument("--final-topk", type=int, default=20)
    ap.add_argument("--tag", default="CENTER_ATOMS")

    # 兼容多步主脚本会传进来的参数
    ap.add_argument("--config", default="configs/infer_top20.yaml")
    ap.add_argument("--model-dir", default="")   # 占位，忽略
    ap.add_argument("--smiles-augment", type=int, default=5)
    ap.add_argument("--augment-mode", default="hybrid")
    ap.add_argument("--aggregation", default="score")

    args, _ = ap.parse_known_args()

    rows = load_jsonl(args.input_jsonl)
    rf_bundle = joblib.load(args.center_predictor)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        head_in = td / "head_input.jsonl"
        head_out = td / "head_output.jsonl"
        tail_in = td / "tail_input.jsonl"
        tail_out = td / "tail_output.jsonl"

        head_rows_in = add_explicit_center_prompt(
            rows, rf_bundle, topk=args.center_topk, tag=args.tag
        )
        tail_rows_in = [ensure_baseline_record(r) for r in rows]

        write_jsonl(head_in, head_rows_in)
        write_jsonl(tail_in, tail_rows_in)

        run_single_predict(
            args.python_bin,
            args.project_root,
            args.head_model_dir,
            head_in,
            head_out,
            args.config,
            args.smiles_augment,
            args.augment_mode,
            args.aggregation,
            args.final_topk,
        )
        run_single_predict(
            args.python_bin,
            args.project_root,
            args.tail_model_dir,
            tail_in,
            tail_out,
            args.config,
            args.smiles_augment,
            args.augment_mode,
            args.aggregation,
            args.final_topk,
        )

        head_rows = load_jsonl(head_out)
        tail_rows = load_jsonl(tail_out)

        fused = fuse_predictions(head_rows, tail_rows, final_topk=args.final_topk)
        write_jsonl(args.output_jsonl, fused)


if __name__ == "__main__":
    main()
