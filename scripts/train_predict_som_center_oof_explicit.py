#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold


NEIGHBOR_ATOMS = [6, 7, 8, 15, 16, 9, 17, 35, 53]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["_row_id"] = str(i)
            rows.append(rec)
    return rows


def pick(rec, keys, default=None):
    for k in keys:
        if k in rec and rec[k] not in [None, ""]:
            return rec[k]
    return default


def parse_som_indices(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "[]"}:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return sorted(set(int(v) for v in obj))
    except Exception:
        pass
    nums = re.findall(r"-?\d+", s)
    return sorted(set(int(n) for n in nums))


def load_autosom_lookup(path):
    df = pd.read_csv(path)

    def norm(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s

    df["_sid"] = df["substrate_id"].map(norm)

    lookup = {}
    for _, r in df.iterrows():
        lookup[r["_sid"]] = {
            "soms": parse_som_indices(r.get("autosom_soms", "")),
            "rule": str(r.get("autosom_rule", "")),
            "status": str(r.get("autosom_status", "")),
        }
    return lookup


def mol_global_features(mol):
    try:
        return {
            "mol_heavy_atoms": mol.GetNumHeavyAtoms(),
            "mol_mw": Descriptors.MolWt(mol),
            "mol_logp": Descriptors.MolLogP(mol),
            "mol_tpsa": Descriptors.TPSA(mol),
            "mol_hbd": Descriptors.NumHDonors(mol),
            "mol_hba": Descriptors.NumHAcceptors(mol),
            "mol_rings": Descriptors.RingCount(mol),
            "mol_rotb": Descriptors.NumRotatableBonds(mol),
        }
    except Exception:
        return {
            "mol_heavy_atoms": np.nan,
            "mol_mw": np.nan,
            "mol_logp": np.nan,
            "mol_tpsa": np.nan,
            "mol_hbd": np.nan,
            "mol_hba": np.nan,
            "mol_rings": np.nan,
            "mol_rotb": np.nan,
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

    for z in NEIGHBOR_ATOMS:
        feat[f"neigh_Z{z}_count"] = sum(1 for x in neigh_atomic_nums if x == z)

    feat["neigh_hetero_count"] = sum(1 for x in neigh_atomic_nums if x not in [1, 6])
    feat.update(mol_global_features(mol))
    return feat


def build_atom_table(records, autosom_lookup=None, with_labels=True):
    rows = []

    for rec_idx, rec in enumerate(records):
        rid = pick(rec, ["reaction_id", "pair_id", "id"], rec["_row_id"])
        row_id = rec["_row_id"]
        smi = pick(rec, ["substrate_smiles", "reactant_smiles", "parent_smiles"])
        prod = pick(rec, ["product_smiles", "gold_product_smiles", "target_smiles"])

        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue

        gold_soms = []
        autosom_rule = ""
        autosom_status = ""

        if autosom_lookup is not None and row_id in autosom_lookup:
            gold_soms = autosom_lookup[row_id]["soms"]
            autosom_rule = autosom_lookup[row_id]["rule"]
            autosom_status = autosom_lookup[row_id]["status"]

        for atom in mol.GetAtoms():
            feat = atom_features(mol, atom)
            feat.update({
                "record_index": rec_idx,
                "reaction_id": rid,
                "row_id": row_id,
                "substrate_smiles": smi,
                "product_smiles": prod,
                "atom_idx": atom.GetIdx(),
                "gold_soms": json.dumps(gold_soms),
                "autosom_rule": autosom_rule,
                "autosom_status": autosom_status,
            })
            if with_labels:
                feat["som_label"] = int(atom.GetIdx() in set(gold_soms))
            rows.append(feat)

    return pd.DataFrame(rows)


def make_rf(seed):
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def get_feature_cols(atom_df):
    exclude = {
        "record_index",
        "reaction_id",
        "row_id",
        "substrate_smiles",
        "product_smiles",
        "gold_soms",
        "autosom_rule",
        "autosom_status",
        "som_label",
    }
    return [c for c in atom_df.columns if c not in exclude]


def fit_rf(atom_df, feature_cols, seed):
    X = atom_df[feature_cols].fillna(0)
    y = atom_df["som_label"].astype(int).values
    clf = make_rf(seed)
    clf.fit(X, y)
    return clf


def predict_atom_probs(clf, atom_df, feature_cols):
    X = atom_df[feature_cols].fillna(0)
    return clf.predict_proba(X)[:, 1]


def add_oof_probs(train_atoms, feature_cols, n_splits, seed):
    train_atoms = train_atoms.copy()
    train_atoms["pred_prob"] = np.nan

    record_indices = sorted(train_atoms["record_index"].unique())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(record_indices), start=1):
        train_rec_set = set(record_indices[i] for i in tr_idx)
        hold_rec_set = set(record_indices[i] for i in va_idx)

        fold_train = train_atoms[train_atoms["record_index"].isin(train_rec_set)].copy()
        fold_hold = train_atoms[train_atoms["record_index"].isin(hold_rec_set)].copy()

        clf = fit_rf(fold_train, feature_cols, seed + fold)
        probs = predict_atom_probs(clf, fold_hold, feature_cols)

        train_atoms.loc[fold_hold.index, "pred_prob"] = probs

        print(f"fold {fold}: train records={len(train_rec_set)}, hold records={len(hold_rec_set)}, hold atoms={len(fold_hold)}")

    if train_atoms["pred_prob"].isna().any():
        raise RuntimeError("Some OOF probabilities are missing.")

    return train_atoms


def assign_full_model_probs(atom_df, clf, feature_cols):
    atom_df = atom_df.copy()
    atom_df["pred_prob"] = predict_atom_probs(clf, atom_df, feature_cols)
    return atom_df


def build_record_predictions(atom_df, topk):
    pred = {}

    for (record_index, row_id, rid), g in atom_df.groupby(["record_index", "row_id", "reaction_id"], sort=False):
        gg = g.sort_values("pred_prob", ascending=False)
        centers = [int(x) for x in gg["atom_idx"].head(topk).tolist()]
        probs = [float(x) for x in gg["pred_prob"].head(topk).tolist()]

        gold_soms = []
        try:
            gold_soms = json.loads(str(gg.iloc[0]["gold_soms"]))
        except Exception:
            pass

        pred[str(row_id)] = {
            "reaction_id": rid,
            "predcenter_indices": centers,
            "predcenter_probs": probs,
            "gold_soms": gold_soms,
            "hit": int(len(set(centers) & set(gold_soms)) > 0) if gold_soms else None,
        }

    return pred


def write_explicit_jsonl(records, pred_lookup, out_jsonl, tag="CENTER_ATOMS"):
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    hits = []
    n = 0

    with open(out_jsonl, "w", encoding="utf-8") as out:
        for rec in records:
            row_id = rec["_row_id"]
            smi = pick(rec, ["substrate_smiles", "reactant_smiles", "parent_smiles"])
            prod = pick(rec, ["product_smiles", "gold_product_smiles", "target_smiles"])

            pr = pred_lookup.get(row_id, {})
            centers = pr.get("predcenter_indices", [])
            probs = pr.get("predcenter_probs", [])
            gold_soms = pr.get("gold_soms", [])
            hit = pr.get("hit", None)

            center_text = " ".join(str(int(x)) for x in centers)
            prompt = f"{smi} {tag} {center_text}".strip()

            new = dict(rec)
            new["substrate_smiles_original"] = smi
            new["substrate_smiles"] = prompt
            new["product_smiles"] = prod
            new["predcenter_indices"] = centers
            new["predcenter_probs"] = probs
            new["predcenter_gold_soms_for_eval"] = gold_soms
            new["predcenter_hit_for_eval"] = hit
            new["center_prompt_type"] = f"oof_explicit_predcenter_top{len(centers)}"
            new["explicit_center_prompt"] = prompt
            new["explicit_center_atoms"] = centers
            new.pop("_row_id", None)

            out.write(json.dumps(new, ensure_ascii=False) + "\n")
            n += 1
            if hit is not None:
                hits.append(hit)

    hit_rate = float(np.mean(hits)) if hits else None
    print("written:", out_jsonl, "n=", n, "center_hit=", hit_rate)
    return hit_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--train-autosom-csv", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--val-autosom-csv", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--test-autosom-csv", required=True, help="Only for center-hit evaluation, not used for model training.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260519)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl(args.train_jsonl)
    val_records = load_jsonl(args.val_jsonl)
    test_records = load_jsonl(args.test_jsonl)

    train_auto = load_autosom_lookup(args.train_autosom_csv)
    val_auto = load_autosom_lookup(args.val_autosom_csv)
    test_auto = load_autosom_lookup(args.test_autosom_csv)

    train_atoms = build_atom_table(train_records, train_auto, with_labels=True)
    val_atoms = build_atom_table(val_records, val_auto, with_labels=True)
    test_atoms = build_atom_table(test_records, test_auto, with_labels=True)

    feature_cols = get_feature_cols(train_atoms)

    train_atoms.to_csv(outdir / "train_atom_table_raw.csv", index=False, encoding="utf-8-sig")
    val_atoms.to_csv(outdir / "val_atom_table_raw.csv", index=False, encoding="utf-8-sig")
    test_atoms.to_csv(outdir / "test_atom_table_evalonly_raw.csv", index=False, encoding="utf-8-sig")

    # OOF predictions for train
    train_atoms_oof = add_oof_probs(train_atoms, feature_cols, args.n_splits, args.seed)

    # full train model for val/test
    full_clf = fit_rf(train_atoms, feature_cols, args.seed)
    val_atoms_pred = assign_full_model_probs(val_atoms, full_clf, feature_cols)
    test_atoms_pred = assign_full_model_probs(test_atoms, full_clf, feature_cols)

    joblib.dump(
        {"model": full_clf, "feature_cols": feature_cols},
        outdir / "som_atom_rf_fulltrain.joblib",
    )

    train_atoms_oof.to_csv(outdir / "train_atom_table_oof_scored.csv", index=False, encoding="utf-8-sig")
    val_atoms_pred.to_csv(outdir / "val_atom_table_scored.csv", index=False, encoding="utf-8-sig")
    test_atoms_pred.to_csv(outdir / "test_atom_table_evalonly_scored.csv", index=False, encoding="utf-8-sig")
    pd.Series(feature_cols).to_csv(outdir / "feature_cols.txt", index=False, header=False)

    train_pred = build_record_predictions(train_atoms_oof, args.topk)
    val_pred = build_record_predictions(val_atoms_pred, args.topk)
    test_pred = build_record_predictions(test_atoms_pred, args.topk)

    split_dir = outdir / f"split_pair622_oof_explicit_top{args.topk}"

    train_hit = write_explicit_jsonl(train_records, train_pred, split_dir / "train.jsonl")
    val_hit = write_explicit_jsonl(val_records, val_pred, split_dir / "val.jsonl")
    test_hit = write_explicit_jsonl(test_records, test_pred, split_dir / "test.jsonl")

    metrics = {
        "topk": args.topk,
        "n_splits": args.n_splits,
        "train_atom_rows": int(len(train_atoms)),
        "train_atom_pos": int(train_atoms["som_label"].sum()),
        "train_atom_neg": int(len(train_atoms) - train_atoms["som_label"].sum()),
        "train_center_hit_topk_oof": train_hit,
        "val_center_hit_topk_fulltrain": val_hit,
        "test_center_hit_topk_evalonly_fulltrain": test_hit,
    }

    for split, atoms, prob_col in [
        ("train_oof", train_atoms_oof, "pred_prob"),
        ("val", val_atoms_pred, "pred_prob"),
        ("test_evalonly", test_atoms_pred, "pred_prob"),
    ]:
        y = atoms["som_label"].astype(int).values
        p = atoms[prob_col].values
        if len(set(y)) > 1:
            metrics[f"{split}_atom_roc_auc"] = float(roc_auc_score(y, p))
            metrics[f"{split}_atom_pr_auc"] = float(average_precision_score(y, p))

    with open(outdir / "oof_predcenter_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("written:", outdir)


if __name__ == "__main__":
    main()
