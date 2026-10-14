#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs/examples/single_step"
INPUT="$OUT/example_test_2.jsonl"
mkdir -p "$OUT"

head -n 2 "$ROOT/data/activated_sludge_pair622_552/test.jsonl" > "$INPUT"

python "$ROOT/scripts/reproduce_single_step.py" \
  --input "$INPUT" \
  --models model_1 model_2 model_3 fusion \
  --topk 5 \
  --num-beams 5 \
  --smiles-augment 1 \
  --augment-mode hybrid \
  --aggregation score \
  --output-dir "$OUT/run"

echo "SINGLE_STEP_EXAMPLE_FINISHED"
echo "output_dir: $OUT/run"
