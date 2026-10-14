# Reference metrics

This directory is reserved for the paper-reported reference metrics used by
`scripts/compare_reproduced_metrics.py`. Each CSV should contain one row for
the relevant model and use metric names matching the generated JSON files.

Reference values are kept separate from regenerated values so that a failed
reproduction cannot overwrite the published numbers.
