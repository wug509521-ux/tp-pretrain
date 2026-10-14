#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run all released single-step models on the fixed 111-reaction test set."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


MODEL_NAMES = (
    "model_1",
    "model_2",
    "model_3",
    "model_4",
    "reaction_site_oracle",
    "fusion",
)


def resolve_models(values):
    if values == ["all"]:
        return list(MODEL_NAMES)
    unknown = [value for value in values if value not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")
    return values


def run(command, env):
    print("RUN:", " ".join(str(x) for x in command), flush=True)
    subprocess.run(command, check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--input", default="data/activated_sludge_pair622_552/test.jsonl")
    parser.add_argument("--oracle-input", default="data/activated_sludge_pair622_552/oracle_center_test.jsonl")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES),
                        help="model_1 model_2 model_3 model_4 reaction_site_oracle fusion, or all")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--smiles-augment", type=int, default=1)
    parser.add_argument("--augment-mode", choices=["random", "hybrid"], default="hybrid")
    parser.add_argument("--aggregation", choices=["vote", "score", "borda"], default="score")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/reproduction/single_step")
    args = parser.parse_args()

    args.models = resolve_models(args.models)
    root = Path(args.package_root).resolve()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = {
        "package_root": str(root),
        "input": str(input_path),
        "models": args.models,
        "topk": args.topk,
        "num_beams": args.num_beams,
        "smiles_augment": args.smiles_augment,
        "augment_mode": args.augment_mode,
        "aggregation": args.aggregation,
        "seed": args.seed,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    baseline_script = root / "scripts" / "predict_baseline_multi_smiles.py"
    fusion_script = root / "scripts" / "predict_fusion_oofhead_newmergedc3tail.py"
    infer_config = root / "configs" / "infer_top20.yaml"
    model_dirs = {
        "model_1": root / "models" / "model_1_no_pretraining",
        "model_2": root / "models" / "model_2_single_step_pretraining",
        "model_3": root / "models" / "model_3_two_step_pretraining",
        "model_4": root / "models" / "model_4_reaction_site_assisted",
        "reaction_site_oracle": root / "models" / "reaction_site_oracle",
    }

    for model_name in args.models:
        model_output = output_dir / model_name
        model_output.mkdir(parents=True, exist_ok=True)
        prediction_path = model_output / "predictions.jsonl"

        if model_name == "fusion":
            command = [
                sys.executable,
                str(fusion_script),
                "--project-root", str(root),
                "--python-bin", sys.executable,
                "--input-jsonl", str(input_path),
                "--output-jsonl", str(prediction_path),
                "--head-model-dir", str(root / "models" / "fusion_head"),
                "--tail-model-dir", str(root / "models" / "fusion_tail"),
                "--center-predictor", str(root / "models" / "center_predictor"),
                "--center-topk", "5",
                "--final-topk", str(args.topk),
                "--config", str(infer_config),
                "--smiles-augment", str(args.smiles_augment),
                "--augment-mode", args.augment_mode,
                "--aggregation", args.aggregation,
            ]
        else:
            model_input = input_path
            if model_name == "model_4":
                model_input = root / "data" / "activated_sludge_pair622_552" / "model4_oof_explicit_predcenter_top5_test.jsonl"
            if model_name == "reaction_site_oracle":
                model_input = Path(args.oracle_input)
                if not model_input.is_absolute():
                    model_input = root / model_input
                if not model_input.exists():
                    raise FileNotFoundError(
                        "Oracle input is missing. Build it first with "
                        "scripts/build_oracle_center_test.py; refusing to use predicted centers."
                    )
            command = [
                sys.executable,
                str(baseline_script),
                "--config", str(infer_config),
                "--model-dir", str(model_dirs[model_name]),
                "--input-jsonl", str(model_input),
                "--output-jsonl", str(prediction_path),
                "--smiles-augment", str(args.smiles_augment),
                "--augment-mode", args.augment_mode,
                "--aggregation", args.aggregation,
                "--final-topk", str(args.topk),
                "--seed", str(args.seed),
            ]
        run(command, env)

        evaluate_script = root / "scripts" / "evaluate_single_step.py"
        run(
            [
                sys.executable,
                str(evaluate_script),
                "--gold-jsonl", str(input_path),
                "--prediction-jsonl", str(prediction_path),
                "--output-dir", str(model_output),
                "--model-name", model_name,
            ],
            env,
        )

    print("SINGLE_STEP_REPRODUCTION_FINISHED")
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
