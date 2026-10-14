# Minimal examples

These examples use only a small subset of the bundled benchmark and do not change the released data. They verify installation, model loading and output layout before running the full 111-reaction or 571-seed experiments.

Run from the repository root after creating the environment:

```bash
bash examples/run_single_step_example.sh
bash examples/run_multistep_example.sh
```

The generated files are written below `outputs/examples/`. The examples use the same model directories, tokenizer files and inference parameters as the full reproduction commands.

The small single-step example runs Model 1–3 and Fusion. Model 4 and the single-step oracle are run together with the full 111-reaction benchmark because their released center-input files are paired with all 111 records.
