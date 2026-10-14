# Configuration files

The public reproduction entry points use the following minimal configuration set:

- `data.yaml`: base data settings;
- `model.yaml`: T5 model settings;
- `infer.yaml`: general inference defaults;
- `infer_top20.yaml`: the released Top-k inference configuration.

Historical training and exploratory configurations are kept outside the public package because the release reproduces the published checkpoints rather than retraining every historical experiment. The released training code is retained under `src/` for provenance, while the reproducibility commands use the checkpoint files under `models/`.
