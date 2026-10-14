#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the released 111-sample reaction-site oracle input.

The published Model 4 OOF file contains both predicted centers and the
gold SOM indices used for the oracle evaluation.  This script deliberately
uses only the latter and never substitutes predicted centers.
"""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model4-input",
        default="data/activated_sludge_pair622_552/"
        "model4_oof_explicit_predcenter_top5_test.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/activated_sludge_pair622_552/oracle_center_test.jsonl",
    )
    parser.add_argument(
        "--audit-output",
        default="data/activated_sludge_pair622_552/oracle_center_test_audit.json",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = Path(args.model4_input)
    output = Path(args.output)
    audit = Path(args.audit_output)
    if not source.is_absolute():
        source = root / source
    if not output.is_absolute():
        output = root / output
    if not audit.is_absolute():
        audit = root / audit

    rows = load_jsonl(source)
    if not rows:
        raise RuntimeError(f"No records found: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    written = []
    missing = []
    for row in rows:
        centers = row.get("predcenter_gold_soms_for_eval")
        original = row.get("substrate_smiles_original")
        if original is None:
            original = str(row.get("substrate_smiles", "")).split(" CENTER_ATOMS ", 1)[0]
        if not original or not centers:
            missing.append(str(row.get("reaction_id", "")))
            continue
        centers = [int(x) for x in centers]
        record = dict(row)
        record["substrate_smiles_original"] = original
        record["substrate_smiles"] = original + " CENTER_ATOMS " + " ".join(map(str, centers))
        record["explicit_center_atoms"] = centers
        record["center_prompt_type"] = "gold_reaction_site_oracle"
        record["oracle_source_field"] = "predcenter_gold_soms_for_eval"
        record.pop("predcenter_indices", None)
        record.pop("predcenter_probs", None)
        record.pop("explicit_center_prompt", None)
        record.pop("predcenter_hit_for_eval", None)
        written.append(record)

    with output.open("w", encoding="utf-8") as handle:
        for row in written:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source": str(source),
        "output": str(output),
        "n_source": len(rows),
        "n_written": len(written),
        "n_missing_gold_center": len(missing),
        "missing_reaction_ids": missing,
        "center_definition": "predcenter_gold_soms_for_eval from the released Model 4 OOF file",
        "center_index_convention": "the atom-index convention already used by the released center prompt",
        "uses_predicted_centers": False,
    }
    audit.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
