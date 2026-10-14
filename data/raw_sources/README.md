# Raw source workbooks

These workbooks are preserved as byte-for-byte copies of the source files used
for data provenance review. Their contents are not silently substituted for the
processed canonical datasets already present in this release.

| Release file | Role |
|---|---|
| `reaction_pairs/Reaction_Pairs.xlsx` | Million-scale reaction-pair source; use `Sheet1` for the primary reaction table. |
| `environment_adaptation/combined_environment_dataset.xlsx` | Environmental adaptation reaction source. |
| `activated_sludge/EAWAG_Sludge_Reactions_20260422.xlsx` | Activated-sludge reactions and auxiliary rule/frequency sheets. |
| `multistep_candidate/near_20_year_wastewater_biotransformation_1218.xlsx` | Primary raw candidate for the multistep pathway dataset; use the `反应对` (`Reaction pairs`) sheet. |

The multistep construction chain is:

```text
反应对 sheet in the raw workbook
        ↓ canonical SMILES edge filtering
iterative_dataset_from_pairs_by_source_depth10/iterative_long.csv
        ↓ strict postprocessing
iterative_dataset_from_pairs_by_source_depth10_postproc/strict_iterative_long.csv
        ↓ gold conversion
external_multistep_gold_strict_442roots_855targets/
```

The current server audit found 1,613 valid unique raw edges and 1,568 unique
iterative edges, with 1,568 canonical edges in common and no iterative-only
edges. The 45 raw-only edges were not retained by the source-constrained
iterative build. The final strict gold directory is therefore a derived
evaluation artifact, not a byte-for-byte copy of this raw workbook.

See `data/manifests/raw_source_manifest.csv` for the release-time row counts,
hashes and source roles. That manifest currently contains an older count for
the nominal 1218 workbook (2,541 rows and 1,503 unique edges), while the
current server audit reports 2,628 rows and 1,613 unique edges. The manifest
and `SHA256SUMS.txt` must be regenerated after the source version is fixed.
