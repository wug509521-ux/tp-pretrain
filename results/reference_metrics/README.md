# Reference metrics

This directory is reserved for the paper-reported reference metrics used by
`scripts/compare_reproduced_metrics.py`. Each CSV should contain one row for
the relevant model and use metric names matching the generated JSON files.

Reference values are kept separate from regenerated values so that a failed
reproduction cannot overwrite the published numbers.
## Single-step Top-k source data

`topk_prediction_results_different_models.xlsx` contains the 111-reaction fixed-test comparison for the Fusion model, BioTransformer 3, enviRule and the refined enviFormer implementation. It includes summary Top-1/3/5/10 counts, reaction-level hit indicators, standardized reaction pairs and retained candidate lists.
