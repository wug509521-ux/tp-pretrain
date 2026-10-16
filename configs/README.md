# Reproduction configuration

The unified reproduction scripts use `infer_top20.yaml` for the shared beam-search settings. Model and input paths are supplied by the scripts so that the package can be moved between machines without editing server-specific paths.

Training-era YAML files and historical fixed-test variants were intentionally omitted from the core release. The published package is checkpoint-based: it reproduces the reported inference and evaluation results without requiring retraining.