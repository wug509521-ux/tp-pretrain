# TP Pretrain Reproducibility Package

This is the compact checkpoint-based release for reproducing the paper's single-step and three-step transformation-product experiments. It contains the six released model branches, the fixed inputs required by the reproduction scripts, evaluation data, source-data tables for reported results, and SHA256 integrity checks.

## Why JSON files are retained

The JSON files are structured reproducibility inputs, not redundant copies:

- `models/*/config.json`, `generation_config.json` and `tokenizer*.json` are required by the Hugging Face model loader.
- JSONL files under `data/` store machine-readable reaction records and fixed oracle inputs.
- JSON files under `results/` store run parameters and summary metrics so the reported results can be audited without parsing log text.
- `release_manifest.json` and `SHA256SUMS.txt` describe package contents and file integrity.

These files are kept separate because each has a different reader and purpose; merging them would make the reproduction scripts less reliable.
## Released model branches

| Paper name | Directory | Role |
|---|---|---|
| Model 1 | `models/model_1_no_pretraining` | No-pretraining baseline |
| Model 2 | `models/model_2_single_step_pretraining` | Single-step pretraining |
| Model 3 | `models/model_3_two_step_pretraining` | Two-step pretraining |
| Model 4 | `models/model_4_reaction_site_assisted` | Out-of-fold predicted-center assisted model |
| Reaction-site oracle | `models/reaction_site_oracle` | Product-derived single-step upper-bound ablation |
| Fusion/Ours | `models/fusion_head`, `models/fusion_tail`, `models/center_predictor` | Final reaction-site-assisted Fusion pipeline |

## Environment

```bash
conda env create -f environment.yml
conda activate tp_pretrain_repro
python -m pip install -e .
```

The released checkpoints are inference artifacts; GPU execution is recommended. All commands below are run from the repository root.

## Single-step reproduction

The fixed activated-sludge test set is stored in `data/activated_sludge_pair622_552/test.jsonl`. Model 1鈥? use ordinary substrate input, Model 4 uses the released out-of-fold predicted-center input, and the oracle uses the separately built product-derived center input.

```bash
python scripts/build_oracle_center_test.py
python scripts/reproduce_single_step.py \
  --input data/activated_sludge_pair622_552/test.jsonl \
  --models model_1 model_2 model_3 model_4 reaction_site_oracle fusion \
  --topk 20 \
  --output-dir outputs/reproduction/single_step
```

Each branch writes predictions, metrics, per-sample metrics and a run configuration.

## Three-step reproduction

The main multistep protocol uses the 571-seed file, Top-10 generation, Top-3 expansion, beam width 40 and maximum depth 3.

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

The separate `data/external_multistep_gold_strict_442roots_855targets/` directory is used only for strict-gold evaluation and must not be mixed with the 571-seed reproduction count.

## Reaction-site scope

Model 4 uses substrate-only out-of-fold predicted centers. The single-step oracle uses product-derived centers as an upper-bound ablation. An independently defined product-derived center map for every multistep intermediate is not released; therefore no multistep oracle metric is claimed. Predicted centers must not be substituted for oracle centers.

## What is intentionally not in the compact core package

Raw source workbooks are grouped under `data/raw/` and are included in the large release asset for provenance. They are not read by routine checkpoint-based inference. Large canonical audit tables, provenance manifests and debug-only gold inputs remain outside the compact package in `tp_pretrain_optional_provenance_backup_20261016` and may be distributed separately if licensing permits. This keeps the runnable release focused on model-result reproduction rather than data auditing.

## Layout

```text
models/       checkpoints, tokenizers and the Fusion components
data/         fixed inputs plus the raw source workbooks in `data/raw/`
configs/      shared inference configuration
scripts/      reproduction, inference and evaluation entrypoints
examples/     small runnable examples
results/      paper reference outputs and figure source data
docs/         model, baseline and reaction-site documentation
src/          reusable Python package code
```

## Integrity and license

Verify the release after extraction:

```bash
sha256sum -c SHA256SUMS.txt
```

The code is released under Apache-2.0; see `LICENSE`. Checkpoint, dataset and third-party baseline terms may differ. Citation metadata is in `CITATION.cff`; a DOI will be added after archival if one is assigned.