#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def get_candidate_list(rec):
    for k in ["predictions", "candidates", "topk", "outputs"]:
        if k in rec and isinstance(rec[k], list):
            return k, rec[k]
    raise KeyError(f"Cannot find candidate list in record keys: {list(rec.keys())}")


def get_smiles(c):
    if isinstance(c, str):
        return c
    for k in [
        "predicted_smiles",
        "smiles",
        "product_smiles",
        "prediction",
        "pred_smiles",
        "canonical_smiles",
    ]:
        if k in c and c[k] not in [None, ""]:
            return str(c[k])
    return json.dumps(c, sort_keys=True, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-jsonl", required=True)
    ap.add_argument("--tail-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--n-head", type=int, default=1)
    ap.add_argument("--final-topk", type=int, default=20)
    ap.add_argument("--method-name", default="OOF-explicit-head-AutoSOM-rerank-tail")
    args = ap.parse_args()

    head_recs = load_jsonl(args.head_jsonl)
    tail_recs = load_jsonl(args.tail_jsonl)

    if len(head_recs) != len(tail_recs):
        raise ValueError(f"Length mismatch: head={len(head_recs)}, tail={len(tail_recs)}")

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for i, (h, t) in enumerate(zip(head_recs, tail_recs)):
            hk, hcands = get_candidate_list(h)
            tk, tcands = get_candidate_list(t)

            new_cands = []
            seen = set()

            # 1) 固定 head 前 n 个
            for c in hcands[:args.n_head]:
                smi = get_smiles(c)
                if smi not in seen:
                    cc = dict(c) if isinstance(c, dict) else {"predicted_smiles": c}
                    cc["fusion_source"] = "head"
                    cc["fusion_original_rank"] = len(new_cands) + 1
                    new_cands.append(cc)
                    seen.add(smi)

            # 2) 用 tail 补齐
            for c in tcands:
                smi = get_smiles(c)
                if smi in seen:
                    continue
                cc = dict(c) if isinstance(c, dict) else {"predicted_smiles": c}
                cc["fusion_source"] = "tail"
                cc["fusion_original_rank"] = len(new_cands) + 1
                new_cands.append(cc)
                seen.add(smi)

                if len(new_cands) >= args.final_topk:
                    break

            # 3) 如果 tail 不够，再从 head 后续补齐
            if len(new_cands) < args.final_topk:
                for c in hcands[args.n_head:]:
                    smi = get_smiles(c)
                    if smi in seen:
                        continue
                    cc = dict(c) if isinstance(c, dict) else {"predicted_smiles": c}
                    cc["fusion_source"] = "head_tail_fill"
                    cc["fusion_original_rank"] = len(new_cands) + 1
                    new_cands.append(cc)
                    seen.add(smi)

                    if len(new_cands) >= args.final_topk:
                        break

            rec = dict(h)
            rec[hk] = new_cands[:args.final_topk]
            rec["metadata"] = dict(rec.get("metadata", {}))
            rec["metadata"].update({
                "fusion_method": args.method_name,
                "n_head": args.n_head,
                "head_jsonl": args.head_jsonl,
                "tail_jsonl": args.tail_jsonl,
                "final_topk": args.final_topk,
            })

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("written:", args.out_jsonl)
    print("n_records:", len(head_recs))
    print("n_head:", args.n_head)
    print("final_topk:", args.final_topk)


if __name__ == "__main__":
    main()
