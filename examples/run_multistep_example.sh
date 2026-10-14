#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs/examples/multistep"
SEEDS="$OUT/example_seeds_2.csv"
mkdir -p "$OUT"

{ head -n 1 "$ROOT/data/external_multistep_571_seeds/drug_list_2_multistep_seeds.csv"; head -n 3 "$ROOT/data/external_multistep_571_seeds/drug_list_2_multistep_seeds.csv" | tail -n 2; } > "$SEEDS"

python "$ROOT/scripts/reproduce_multistep.py" \
  --seed-file "$SEEDS" \
  --models model_1 model_2 model_3 model_4 fusion \
  --max-depth 3 \
  --single-step-topk 10 \
  --per-parent-expand 3 \
  --beam-width 40 \
  --paths-topk 20 \
  --smiles-augment 1 \
  --augment-mode hybrid \
  --aggregation score \
  --output-dir "$OUT/run"

echo "MULTISTEP_EXAMPLE_FINISHED"
echo "output_dir: $OUT/run"
