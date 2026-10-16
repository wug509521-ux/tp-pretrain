# Raw source data

These workbooks are the original source datasets retained for provenance and optional data auditing. Routine checkpoint-based reproduction does not read them during inference; they are included under `data/raw/` so that the data origin remains inspectable.

| File | Role |
|---|---|
| `reaction_pairs_raw.xlsx` | Million-scale reaction-pair source used for reaction-pair data provenance. |
| `environmental_domain_raw.xlsx` | Environmental-domain adaptation source. |
| `activated_sludge_reactions_raw.xlsx` | Activated-sludge reaction source. |
| `wastewater_biotransformation_1218_raw.xlsx` | Candidate multistep wastewater biotransformation source. The relevant raw worksheet is `Reaction pairs`. |

The derived safety-filtered workbook `1218_safe.xlsx` is not included because it is not required by the released reproduction scripts. Large source workbooks may be subject to separate data-licence terms.
