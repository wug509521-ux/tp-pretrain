# Reaction-site provenance and oracle scope

This release distinguishes three different objects that must not be conflated:

1. **AutoSOM annotations** are product-informed annotations obtained from an observed substrate–product pair. They are used as supervision for the center-predictor training step.
2. **The released center predictor** is a substrate-only atom-level random-forest model. During formal evaluation it produces out-of-fold predicted centers from the parent substrate, without reading the reference product.
3. **The reaction-site oracle** is a single-step upper-bound ablation. It supplies product-derived AutoSOM center annotations to the generator and is therefore not a blind prediction setting.

## Single-step benchmark

The fixed single-step benchmark contains 111 activated-sludge reaction pairs. Model 4 uses the released out-of-fold predicted-center input:

```text
data/activated_sludge_pair622_552/model4_oof_explicit_predcenter_top5_test.jsonl
```

The single-step oracle uses a separately constructed gold-center input:

```text
data/activated_sludge_pair622_552/oracle_center_test.jsonl
```

These two files are intentionally different. The Model 4 file contains predicted center indices, whereas the oracle file contains product-derived center fields. The oracle is used only to quantify an upper bound and is not used to select the final Fusion model.

## Blind multistep benchmark

The three-step external evaluation is a blind recursive inference protocol. The model generates candidates for the current parent, retains the prescribed Top-k candidates and expands the next generation without using literature products or product-derived centers to choose branches.

The released multistep pathway records do not contain an independently defined reaction-center map for every intermediate generated during recursion. Consequently, this release does **not** report a multistep reaction-site oracle metric. Model 4 predicted centers must not be substituted for the missing intermediate oracle map.

The multistep runner therefore supports an oracle branch only when the user supplies an external map through `--oracle-center-map`. The map must cover every intermediate that is evaluated and must document its source, atom-index convention and standardization procedure. Without that map, the valid reproducible multistep results are the blind Model 1–4 and Fusion runs, plus post hoc pathway evaluation against the released gold records.

## Reproducibility implication

The absence of a multistep oracle result is a data-availability limitation, not a missing checkpoint. The single-step oracle checkpoint and input are released and reproducible; the multistep oracle is intentionally withheld as an unclaimed analysis because its required intermediate-level product-derived center annotations are not available.
