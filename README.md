# TP Pretrain Reproducibility Package

This repository contains the released code, model checkpoints, tokenizer files, evaluation data and scripts required to reproduce the single-step and three-step transformation-product experiments.

## What is released

| Paper name | Release directory | Inference role |
|---|---|---|
| Model 1 | `models/model_1_no_pretraining` | No-pretraining baseline |
| Model 2 | `models/model_2_single_step_pretraining` | Single-step pretraining |
| Model 3 | `models/model_3_two_step_pretraining` | Two-step pretraining |
| Model 4 | `models/model_4_reaction_site_assisted` | OOF predicted-center assisted model |
| Reaction-site oracle | `models/reaction_site_oracle` | Single-step product-derived upper-bound ablation |
| Fusion/Ours | `models/fusion_head`, `models/fusion_tail`, `models/center_predictor` | Final reaction-site-assisted Fusion model |

The model directories contain the checkpoint, tokenizer and generation configuration. The Fusion model is a coordinated head/tail/center-predictor pipeline rather than one checkpoint file.

## 1. Create the environment

The recommended environment is Python 3.10 with PyTorch 2.5.1 and CUDA 12.1.

```bash
conda env create -f environment.yml
conda activate tp_pretrain_repro
python -m pip install -e .
```

For an existing environment:

```bash
python -m pip install -r requirements-repro.txt
python -m pip install -e .
```

The commands below should be executed from the repository root. The scripts resolve all model and data paths relative to that root and do not require the original server directory.

## 2. Single-step reproduction

The fixed activated-sludge benchmark contains 111 reaction pairs. Model 1–3 use ordinary substrate input, Model 4 uses the released out-of-fold predicted-center input, and the oracle uses a separately constructed product-derived center input.

```bash
python scripts/build_oracle_center_test.py
python scripts/reproduce_single_step.py \
  --input data/activated_sludge_pair622_552/test.jsonl \
  --models model_1 model_2 model_3 model_4 reaction_site_oracle fusion \
  --topk 20 \
  --output-dir outputs/reproduction/single_step
```

Each model directory contains `predictions.jsonl`, `metrics.csv`, a per-sample table, `run_config.json` and the model-specific metric JSON. The evaluator uses RDKit canonical SMILES matching and reports Top-1, Top-5, Top-10, Top-20 and similarity-based summaries.

To run the small example instead of the full benchmark:

```bash
bash examples/run_single_step_example.sh
```

## 3. Three-step multistep reproduction

The paper protocol uses 571 seeds, Top-10 generation, Top-3 expansion, beam width 40 and maximum depth 3.

```bash
python scripts/reproduce_multistep.py \
  --seed-file data/external_multistep_571_seeds/drug_list_2_multistep_seeds.csv \
  --models model_1 model_2 model_3 model_4 fusion \
  --max-depth 3 \
  --single-step-topk 10 \
  --per-parent-expand 3 \
  --beam-width 40 \
  --paths-topk 120 \
  --output-dir outputs/reproduction/multistep
```

Each model directory contains `nodes.csv`, `edges.csv`, `paths_topk.csv`, `all_complete_paths.csv`, `summary.json` and `run_config.json`.

The small example is:

```bash
bash examples/run_multistep_example.sh
```

## 4. Strict-gold evaluation

The 571 seeds reproduce the paper's main multistep prediction run. The 442-root/855-target directory is a separate strict-gold evaluation set and must not be merged with the 571-seed count.

```bash
python scripts/evaluate_multistep.py \
  --prediction-dir outputs/reproduction/multistep \
  --gold-dir data/external_multistep_gold_strict_442roots_855targets \
  --output-dir outputs/reproduction/multistep_evaluation

python scripts/compare_reproduced_metrics.py \
  --single-step-dir outputs/reproduction/single_step \
  --multistep-dir outputs/reproduction/multistep \
  --reference-dir results/reference_metrics \
  --output outputs/reproduction/metrics_comparison.csv
```

## 5. Reaction-site provenance and oracle scope

AutoSOM is the product-informed annotation procedure used to create reaction-site labels. The released center predictor is a substrate-only atom-level random-forest model that generates out-of-fold predicted centers for Model 4. The single-step reaction-site oracle uses product-derived center fields only as an upper-bound ablation.

The released external multistep records do not provide an independently defined product-derived center map for every intermediate. Therefore, no multistep oracle metric is claimed. Model 4 predicted centers must not be passed as a substitute for an oracle map. If an independently generated map becomes available, the oracle branch can be run with:

```bash
python scripts/reproduce_multistep.py \
  --seed-file data/external_multistep_571_seeds/drug_list_2_multistep_seeds.csv \
  --models reaction_site_oracle \
  --oracle-center-map path/to/oracle_center_map.jsonl \
  --output-dir outputs/reproduction/multistep_oracle
```

See `docs/REACTION_SITE_PROVENANCE.md` for the required map fields and leakage distinction.

## 6. Repository layout

```text
models/       released checkpoints and tokenizer files
data/         fixed benchmarks, raw-source provenance and gold records
configs/      minimal inference configuration files
scripts/      unified reproduction, prediction and evaluation scripts
examples/     small runnable examples
results/      paper reference metrics and source data
docs/         model mapping, baseline and reaction-site documentation
src/          reusable Python package code
```

## 7. Reproducibility and integrity

`SHA256SUMS.txt` contains the SHA256 digest of every released file except itself. Verify after extraction:

```bash
sha256sum -c SHA256SUMS.txt
```

The code in this repository is released under the Apache License 2.0; see `LICENSE`. The model checkpoints, tokenizer files, source datasets, and third-party baseline artifacts may have separate terms or redistribution restrictions and are not automatically covered by the code license. Review each data/model directory's provenance and license notes before redistribution. The citation metadata is in `CITATION.cff`; a DOI is intentionally omitted until the repository is archived and assigned one.

## 8. Citation

Please cite the associated manuscript and the exact release version. Update `CITATION.cff` with the final repository URL and DOI after archiving.
