# Manuscript Model Mapping

| Manuscript name | Repository directory | Verified interpretation |
|---|---|---|
| Model 1 | `models/model_1_no_pretraining` | No-pretraining model |
| Model 2 | `models/model_2_single_step_pretraining` | Single-step pretraining model |
| Model 3 | `models/model_3_two_step_pretraining` | Two-step pretraining model |
| Model 4 | `models/model_4_reaction_site_assisted` | C3 OOF explicit predicted-center top-5 model |
| Fusion/Ours | `models/fusion_head` + `models/fusion_tail` | Reaction-site-assisted Fusion model |
| Reaction-site oracle | `models/reaction_site_oracle` | Gold reaction-site upper-bound ablation |

## Fusion model components

The Fusion model is not a single checkpoint directory. It consists of:

1. `models/fusion_head`
2. `models/fusion_tail`
3. `models/center_predictor`
4. `scripts/predict_fusion_oofhead_newmergedc3tail.py`

The Fusion model combines the predicted reaction-site head with the tail model. The reaction-site oracle uses experimentally derived reaction centers and is reported only as an upper-bound ablation.

## Important distinction

Model 4 is the C3 OOF explicit predicted-center model. It should not be described as the safe80 model unless the manuscript is revised accordingly.
