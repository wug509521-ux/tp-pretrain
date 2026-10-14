# Data release

This directory is organized by modelling stage and evaluation task. Each dataset must be accompanied by provenance, license/access information, a data dictionary and a SHA256 manifest.

The 111-reaction single-step benchmark is separate from the 1,910-product/855-parent multi-step validation set. The 494 and 552 activated-sludge counts must be explained as distinct processing stages if both are retained.

Raw HRMS files, restricted third-party data and files without redistribution permission must not be copied here. In that case, keep only de-identified processed data and an access description.

## Raw source workbooks

The original source workbooks used for provenance review are preserved under
`data/raw_sources/`. They include the million-scale `Reaction Pairs.xlsx`, the
environmental adaptation combined dataset, the EAWAG activated-sludge workbook,
and the candidate multistep wastewater biotransformation workbook. Their hashes,
primary sheets and row counts are recorded in
`data/manifests/raw_source_manifest.csv`.

The `near_20_year_wastewater_biotransformation_1218.xlsx` workbook is an
upstream candidate multistep source. Its `反应对` (`Reaction pairs`) sheet
contains the parent and transformation-product standardized SMILES used in the
multistep provenance audit. The source-constrained iterative dataset is stored
under:

```text
data/iterative_dataset_from_pairs_by_source_depth10/
├── iterative_long.csv
├── paths.csv
└── summary.json
```

For the current server copy, canonical edge comparison gave:

```text
raw workbook rows:              2628
raw valid unique edges:         1613
iterative_long rows:            6913
iterative valid unique edges:   1568
canonical overlap:              1568
raw-only edges:                 45
iterative-only edges:           0
```

Thus every unique reaction edge retained in `iterative_long.csv` is supported
by the raw `反应对` table. The long-format row count is larger because edges
are repeated across pathway IDs, contexts and depth settings. The iterative
dataset is a processed, source-constrained representation rather than a byte-
for-byte copy of the workbook.

The strict postprocessing stage is stored under:

```text
data/iterative_dataset_from_pairs_by_source_depth10_postproc/
└── strict_iterative_long.csv
```

This stage retains 1,466 unique edges and is the input used to create the
strict 442-root/855-target gold files. The strict gold release is therefore a
derived evaluation artifact, not a raw source dataset.

An older manifest entry reports 2,541 rows and 1,503 unique edges for the
nominal 1218 workbook, whereas the current server copy has 2,628 rows and
1,613 unique edges. This version discrepancy must be resolved before public
release; do not silently merge the two manifests.
