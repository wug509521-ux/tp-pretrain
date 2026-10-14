#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare reproduced metrics with optional paper reference metrics."""

import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-step-dir", required=True)
    parser.add_argument("--multistep-dir", required=True)
    parser.add_argument("--reference-dir", default="results/reference_metrics")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    single = Path(args.single_step_dir)
    multi = Path(args.multistep_dir)
    reference = Path(args.reference_dir)
    rows = []
    for metrics_file in sorted(single.glob("*/**/*_metrics.json")):
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
        model = data.get("model", metrics_file.stem.replace("_metrics", ""))
        ref = reference / f"{model}_metrics.csv"
        ref_rows = read_csv(ref) if ref.exists() else []
        ref_row = ref_rows[0] if ref_rows else {}
        for metric, value in flatten(data).items():
            rows.append({
                "scope": "single_step",
                "model": model,
                "metric": metric,
                "reproduced": value,
                "reference": ref_row.get(metric, ""),
                "difference": (as_number(value) - as_number(ref_row.get(metric)))
                if as_number(value) is not None and as_number(ref_row.get(metric)) is not None else "",
                "source_file": str(metrics_file),
            })

    for summary_file in sorted(multi.glob("*/summary.json")):
        data = json.loads(summary_file.read_text(encoding="utf-8"))
        model = data.get("model", summary_file.parent.name)
        ref = reference / "multistep_metrics.csv"
        ref_rows = read_csv(ref) if ref.exists() else []
        ref_row = next((x for x in ref_rows if x.get("model") == model), {})
        for metric, value in flatten(data).items():
            rows.append({
                "scope": "multistep",
                "model": model,
                "metric": metric,
                "reproduced": value,
                "reference": ref_row.get(metric, ""),
                "difference": (as_number(value) - as_number(ref_row.get(metric)))
                if as_number(value) is not None and as_number(ref_row.get(metric)) is not None else "",
                "source_file": str(summary_file),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scope", "model", "metric", "reproduced", "reference", "difference", "source_file"]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("METRICS_COMPARISON_FINISHED")
    print("rows:", len(rows))
    print("output:", output)


if __name__ == "__main__":
    main()
