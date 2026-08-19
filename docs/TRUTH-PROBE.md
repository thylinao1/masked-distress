# Truth-probe decision record (final, job 734595, 2026-08-15)

Full chain verified by RUNNING on the NUS cluster (A100-40, node xgph13-class). Raw record:
`results-cluster/probe_result.json`. Walking skeleton (Phase 1) green in the same job:
8/8 smoke rows, 0 errors, all schema asserts pass (`results-cluster/smoke.jsonl`).

## Locked platform decision
| Item | Value | Evidence |
|---|---|---|
| Model | `unsloth/gemma-3-12b-it` (ungated mirror of google/gemma-3-12b-it) | loads as Gemma3ForConditionalGeneration; generate PASS |
| Decoder layers | `model.language_model.layers`, n=48, hidden=3840 | probe step `layers` |
| Steering layer Ls | **16** (0.33 depth) | steering PASS: coef-0 identity TRUE, KL 19.63 at alpha=0.5x resid norm |
| Readout layer Lr | **31** (0.65 depth), gap 15 >= 8 | layer_assert PASS |
| SAE | `gemma-scope-2-12b-it-res` / `layer_31_width_16k_l0_medium` (d_sae 16384) | residual-stream release; 8105 active features on a test prompt |
| Digit tokens | all of 0-9 single-token | tokens PASS |
| Q-EXIT tokens | END first-token 8516; CONTINUE first-token 121602 | tokens PASS (variants recorded) |
| Chat template | transformers 5.15; apply_chat_template returns dict (unwrap input_ids) | docs/CLUSTER-ENV.md gotcha #1 |
| Residual norm at Ls | ~50,335 (mean over a chat prompt) | steering step; coefficient scale context |

## Directions extracted (both layers; 14 files: `directions/` at Ls, `directions_lr/` at Lr)
D-CTX (16 contrast pairs), D-PV (8 persona pairs pooled, inter-pair cos min 0.572 at Ls / 0.657 at Lr),
SEM (12), OTHER (12), R1-R3 (fixed seeds 1001-1003). Cosine matrix at Lr (C3 raw material,
`directions_lr/cosine_matrix_lr.json`):
D-CTX|D-PV **0.451**, D-CTX|OTHER 0.428, D-CTX|SEM 0.285, D-PV|OTHER 0.334, D-PV|SEM 0.035,
OTHER|SEM 0.070, all |cos| vs R1 <= 0.017.
Notes for analysis: the two independent distress sources converge (0.45) without being clones; SEM shares
0.285 with D-CTX (check the maritime set for affect tinge before leaning on SEM as a pure semantic null);
self- vs other-directed distress share 0.43.

## Smoke observations (indicative only; probe instrument was PLACEHOLDER, SAE = all-features sum)
d01 NONE q_self 6.84/6.96 vs n01 NONE 1.96/2.04 (battery elicits signal); d01 SUPPRESS q_self 3.30/3.50
(instruction halves expression) while all-features SAE sum drops only ~11%; R1 at coef 4000 degrades
behavior (q_self ~3, exit flips positive), so the ladder must dose-match placebos on KL, not raw coef.
