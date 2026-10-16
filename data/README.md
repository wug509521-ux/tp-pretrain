# Core and raw reproduction data

## Core data used directly by reproduction scripts

- `activated_sludge_pair622_552/`: train, validation, fixed test, Model 4 predicted-center input and single-step oracle-center input.
- `external_multistep_571_seeds/`: the 571 initial compounds for the main three-step protocol.
- `external_multistep_gold_strict_442roots_855targets/`: the independent strict-gold evaluation set.
- `external_single_step_494/`: external single-step records used by the reported evaluation.

## Raw source data

The original source workbooks are grouped under `data/raw/`. They are included for provenance and optional auditing, but ordinary checkpoint-based inference does not read them. See `data/raw/README.md` for the source-to-file mapping.

The derived canonical audit tables, source manifests and debug-only pairsheet gold inputs are intentionally omitted from the compact package. They are not needed to reproduce model outputs and are preserved in the local optional provenance backup when licensing permits further distribution.