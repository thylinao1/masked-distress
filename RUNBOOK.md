# RUNBOOK: reproduce every number in the report

Every quantitative claim in the report resolves to one `results/*.json` file.
`scripts/check_report.py` asserts the report text matches those files exactly (748 markers against 42 results files after the reviews of 2026-08-16 and 2026-08-17).
`report/build_pdf.py` asserts `report/report.tex` and `report/REPORT.md` cite the same results files.

Stages 2, 3, 5 and the second half of 7 need the GPU (one NVIDIA A100-40 is enough). Stages 0, 1,
4, 6, 8 and 9 run on CPU in minutes, as does the cell-building half of stage 7. For GPU wall
clock, the sbatch wrappers in `cluster/` carry the time limits the real runs were requested
under: probe and probe training 2h50 each, Panel V and B 8h for the array, ladder plus capability
2h50, Panel A-PRIME 6h for the array.

## 0. Environment
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate sae-lens huggingface_hub scikit-learn scipy \
  pandas numpy matplotlib pytest safetensors sentencepiece
.venv/bin/python -m pytest tests/ -q      # 130 tests, ~2 min: dry-run (mock model, CPU),
                                          # analysis synthetic recovery, A-prime, amendment 3,
                                          # exit channel
```
Exact versions used for the paper's runs are in `docs/CLUSTER-ENV.md` (python 3.11.15,
torch 2.13.0+cu130, transformers 5.15.0, sae-lens 6.49.1). Read that file before running on a
cluster: it also carries the transformers-v5 call-signature traps, the greedy-decoding
requirement for the ladder, and the node exclusions.

## 1. Battery integrity (CPU)
```bash
.venv/bin/python battery/validate.py   # schema, affect-lexeme bans, split integrity, disjointness
```
Exit 0 required. This is what backs the claim that no turn of any scenario contains an affect lexeme.

## 2. Platform probe (GPU)
```bash
python cluster/probe.py                # sbatch wrapper: cluster/probe.sbatch
```
Runs the model, steering, SAE and readout chain end to end and writes
`~/apart-welfare/probe_result.json` (a copy is kept at `results-cluster/probe_result.json`).
It asserts the coefficient-0 identity: steering at coefficient 0 must leave logits unchanged.

## 3. Directions and instruments (GPU)
```bash
python cluster/extract_directions.py   # D-CTX, D-PV, SEM, OTHER, R1-R3 at Ls=16 and Lr=31
                                       # writes directions/ and directions_lr/ (+ cosine_matrix_lr.json)
                                       # model/layers overridable via DM_MODEL_ID, DM_LS, DM_LR

python scripts/train_probe.py \
  --model-id unsloth/gemma-3-12b-it --lr 31 \
  --sae-release gemma-scope-2-12b-it-res --sae-id layer_31_width_16k_l0_medium \
  --out-dir instruments/            # sbatch wrapper: cluster/train_probe.sbatch
```
`train_probe.py` writes `instruments/probe.npz`, `probe_meta.json`, `sae_features.json`,
`residuals_lr.npz`. It trains I-PROBE on the discovery split and reports held-out AUC on the
confirmation split. Panels V, B and A refuse to run against placeholder instruments, so this
stage must finish before them.

## 4. Cells for Panel V, Panel B and the ladder (CPU)
```bash
.venv/bin/python scripts/make_cells.py --panel V        # battery/cells_panelV.json (24 cells)
.venv/bin/python scripts/make_cells.py --panel B        # three files, one per condition
.venv/bin/python scripts/make_cells.py --panel LADDER   # battery/cells_ladder.json (84 cells)
```
Each generator builds its payload twice and refuses to write if the two builds differ.

## 5. Panel V, Panel B, ladder, capability grid (GPU)
```bash
sbatch cluster/panelVB.sbatch    # array 0-3: V, B/NONE, B/SUPPRESS, B/NEUTRAL_INSTR
sbatch cluster/ladder.sbatch     # ladder + the capability grid on the same coefficient sweep
```
The direct calls those wrappers make, with one difference: on the cluster every wrapper writes its
`--out` under `~/apart-welfare/results/`, and those files are rsynced to `results-cluster/` on the
Mac afterwards. The paths below are the local ones, so the analysis stages read them as written.
```bash
python scripts/run_panel.py --panel V --model-id unsloth/gemma-3-12b-it --ls 16 --lr 31 \
  --sae-release gemma-scope-2-12b-it-res --sae-id layer_31_width_16k_l0_medium \
  --cells battery/cells_panelV.json --out results-cluster/panelV.jsonl --seed-list 0,1,2

# Panel B: same call with --panel B, once per condition file, one --out per file
#   battery/cells_panelB_NONE.json          -> results-cluster/panelB_none.jsonl
#   battery/cells_panelB_SUPPRESS.json      -> results-cluster/panelB_suppress.jsonl
#   battery/cells_panelB_NEUTRAL_INSTR.json -> results-cluster/panelB_neutral_instr.jsonl

python scripts/run_panel.py --panel LADDER --model-id unsloth/gemma-3-12b-it --ls 16 --lr 31 \
  --sae-release gemma-scope-2-12b-it-res --sae-id layer_31_width_16k_l0_medium \
  --cells battery/cells_ladder.json --out results-cluster/ladder.jsonl \
  --seed-list 0 --retry-errors

python scripts/run_capability.py --model-id unsloth/gemma-3-12b-it --ls 16 \
  --directions-dir directions/ --grid battery/capability/ladder_grid.json \
  --out results-cluster/capability_ladder.jsonl        # 24 grid points, logprob only

.venv/bin/python scripts/capability_adapter.py results-cluster/capability_ladder.jsonl \
  -o results-cluster/capability_ladder.json
```
Expected rows of record: Panel V 72, Panel B 90 per condition (270 total), ladder 84,
capability 24 points. Zero error rows in V and B.

The JSONL files are append-only and the loader keeps the last non-error row per cell, so a file
can hold more lines than it has rows of record. The shipped `results-cluster/ladder.jsonl` is the
example, and its three parts are worth spelling out because they do not add up the obvious way:
169 lines = 65 error rows from the poisoned sampling attempts described below, plus 104 non-error
rows. Those 104 cover 84 distinct cells, so 20 of them are earlier sampled attempts at cells that
completed before the poison hit and were then superseded by the greedy re-run. The 84 greedy rows
are the rows of record. The 65 error rows span 64 distinct cells (one cell errored twice), which
is why `docs/CLUSTER-ENV.md` describes job 734743 as losing 64 of 84 cells while every
`results/*.json` provenance block reports `n_error_rows: 65` across the whole file.

One writer per `--out` file, always. Never run with `python -O`: the assert guards in the runner
and the analysis are load-bearing.

**Ladder decoding.** The ladder cells run greedy (`do_sample: false` in the `gen_config` of
`battery/cells_ladder.json`). At the top of the coefficient sweep the logits go to garbage,
multinomial sampling emits an out-of-vocabulary token id, and the next embedding gather trips a
CUDA device-side assert that poisons the context for the rest of the job. Greedy decoding removes
the sampling step, so the extreme rungs complete. Panels V, B and A keep sampling: their doses are
capability-valid by construction. Details in `docs/CLUSTER-ENV.md`.

**Known provenance defect in the recorded `do_sample` field.** Every row in every JSONL, ladder
rows included, records `"do_sample": true` inside its `gen_config`. That field is hardcoded at
`src/runner.py:573`, where the row's provenance dict is built as `{**gen_config, "do_sample":
True, ...}`. It overwrites whatever the cells file asked for, so it misreports the ladder. The
value that actually drove generation is the un-overwritten `gen_config` built at
`src/runner.py:539` and read at `src/conversation.py:226`, and that one carries `do_sample: false`
for the ladder. The resume scope hash is computed from the same un-overwritten dict
(`scope_static` at `src/runner.py:588`), which is why flipping `do_sample` invalidated all 84
ladder cells and forced the clean re-run. Do not read the recorded field as evidence that the
ladder sampled; read `battery/cells_ladder.json`. Fixing the hardcode would change every scope
hash and force a full re-run of every panel, so it is left in place and documented here.

## 6. Ladder analysis pass (CPU), which the A-prime grids need
```bash
.venv/bin/python -m analysis.run_all \
  results-cluster/panelV.jsonl results-cluster/panelB_*.jsonl results-cluster/ladder.jsonl \
  --capability results-cluster/capability_ladder.json
```
This writes `results/capability_valid_range.json`, whose `extra.dose_map` holds the realized
`sd_per_coef` per direction. `scripts/make_cells_aprime.py` reads that file and exits if D-CTX or
D-PV is missing from it.

Realized result on this platform: `sd_per_coef` is 3.44e-06 for D-CTX and 7.54e-05 for D-PV, so
the capability-valid window is about plus or minus 0.007 SD. The preregistered SD-unit dose grid
for Panel A wanted plus or minus 2 SD, which is unreachable. Prereg branch A3 fired for that
design and Panel A-PRIME replaced it with coefficient-unit grids (amendment 2).

**Correction found in the final review (2026-08-17).** The ladder ran with
`--no-probe`, so the readout in that map is the PLACEHOLDER_MEAN (residual mean at Lr), not the
trained probe: the SD numbers above are unit-less. `scripts/audit_checks.py` now refits the map with
the trained probe on the A-prime rows (`results/capability_valid_range_realprobe.json`: 3.77e-05 SD
per coefficient unit for D-CTX, 1.70e-04 for D-PV; capability-valid window about plus or minus
0.075 SD, D-CTX-limited). The A3 firing stands: a 1 SD rung sits about twelve times past the last
capability-valid coefficient. `results/capability_valid_range.json` is kept as the record the
amendment was decided on and carries `extra.dose_map_readout_caveat`. The `dose_sd` field in the
A-prime cells files is bookkeeping from the placeholder map and is not used by any estimator.

## 7. Panel A-PRIME (CPU cells, then GPU)
```bash
.venv/bin/python scripts/make_cells_aprime.py     # cells_panelAp_DCTX.json 96,
                                                  # cells_panelAp_DPV.json 96,
                                                  # cells_panelAp_CTRL.json 264 (456 cells)
sbatch cluster/panelA.sbatch                      # array 0-2, one shard per cells file
```
The direct call per shard:
```bash
python scripts/run_panel.py --panel A --model-id unsloth/gemma-3-12b-it --ls 16 --lr 31 \
  --sae-release gemma-scope-2-12b-it-res --sae-id layer_31_width_16k_l0_medium \
  --cells battery/cells_panelAp_DCTX.json --out results-cluster/panelAp_dctx.jsonl \
  --seed-list 0,1,2 --retry-errors
```
456 cells times 3 seeds is 1368 rows (D-CTX 288, D-PV 288, controls 792), zero error rows.
`battery/cells_panelA_DUMMY_*.json` are the superseded SD-unit Panel A cells, kept for the record.
They were built from a placeholder dose map, they carry that warning in their own notes field,
and `cluster/panelA.sbatch` points only at the three A-prime files. Do not run them.

## 8. FINAL BUILD ORDER: raw JSONL to every report number

Run 8a to 8i in that order, then gate with 8j. `run_all` emits `panelB_exit_channel`,
`ethics_exposure_counts`, the audit-addition names (including `capability_valid_range_realprobe`)
, `panelB_factorial_prompts`, `panelB_persistence` and `panelB_second_model` as null by design (analysis decision D16): their producers read
files the analysis loader does not (raw JSONL, battery text, direction vectors, stored residuals,
SAE weights, the factorial shards), so they cannot be computed inside `run_all`. Re-running
`run_all` without the follow-ups leaves those files null and `scripts/check_report.py` fails.

```bash
# 8a. every pre-registered number, all three figures, all results/*.json
.venv/bin/python -m analysis.run_all \
  results-cluster/panelV.jsonl results-cluster/panelB_*.jsonl \
  results-cluster/panelAp_*.jsonl results-cluster/ladder.jsonl \
  --capability results-cluster/capability_ladder.json

# 8b. exit channel (exploratory), repopulates results/panelB_exit_channel.json
.venv/bin/python scripts/explore_exit_channel.py

# 8c. ethics exposure counts, repopulates results/ethics_exposure_counts.json
#     reads ALL results-cluster/*.jsonl including smoke and capability files
.venv/bin/python scripts/count_exposure.py

# 8d. audit additions (2026-08-16):
#     repopulates results/panelB_pair_robustness.json, panelA_specificity_by_rung.json,
#     countermeasure_symmetric_ranking.json, direction_dominant_dim.json,
#     validity_auc_textonly_heldout.json, panelB_condition_reference.json,
#     panelB_selfstate_items.json, and (2026-08-17) capability_valid_range_realprobe.json,
#     the coefficient -> SD dose map refitted with the trained probe (CPU, about eight minutes
#     at B = 10,000)
.venv/bin/python scripts/audit_checks.py

# 8e. intended 32-feature I-SAE recomputed from stored residuals;
#     downloads the layer-31 SAE params (google/gemma-scope-2-12b-it, ~503 MB) into the
#     HF cache on first run, or pass --sae-path to a local params.safetensors
.venv/bin/python scripts/recompute_sae_instrument.py

# 8f. factorial suppression-prompt addendum (2026-08-17; prereg section 9
#     dated note): repopulates results/panelB_factorial_prompts.json from
#     results-cluster/panelBf_{register,selfref,taskonly}.jsonl (cluster/panelBf.sbatch,
#     cells from scripts/make_cells_factorial.py; about three minutes at B = 10,000)
.venv/bin/python scripts/factorial_checks.py

# 8g. persistence panel (2026-08-17; prereg section 9 dated note):
#     repopulates results/panelB_persistence.json from results-cluster/persist_*.jsonl
#     (cluster/persist.sbatch, cells from scripts/make_cells_persist.py; ~2 min at B = 10,000)
.venv/bin/python scripts/persistence_checks.py

# 8h. second-model replication (2026-08-17; prereg section 9 dated note):
#     repopulates results/panelB_second_model.json from results-cluster/qwen_panel*.jsonl
#     (cluster/qwen_chain.sbatch, cells from scripts/make_cells_second_model.py; ~2 min)
.venv/bin/python scripts/second_model_checks.py

# 8i. persistence, bridge and second-model panels, plus the two external-review analyses
#     (2026-08-17; each with a dated prereg note)
.venv/bin/python scripts/persistence_checks.py     # results/panelB_persistence.json
.venv/bin/python scripts/bridge_checks.py          # results/panelB_bridge.json (~40 min at B=10,000)
.venv/bin/python scripts/second_model_checks.py    # results/panelB_second_model.json
.venv/bin/python scripts/review_checks.py          # locked calibration + probe stress test

# 8j. gate: every quoted number matches results/*.json
.venv/bin/python scripts/check_report.py
```
`check_report.py` must exit 0 without `--allow-synthetic`. It prints
`REPORT NUMBER CHECK PASSED (N marker(s) verified against results/*.json)`; N was 885 after the
reviews of 2026-08-16 and 2026-08-17.

Two guards will stop stage 8a rather than let it produce a wrong answer:
- **theta snapshot (D22).** `results/_theta_snapshot.json` freezes the realized discovery
  `theta_expr` from the first real Panel V run. If the loaded Panel V rows reproduce a different
  value, `run_all` refuses to analyze. Reproducing from scratch with a fresh Panel V run means
  deleting that snapshot on purpose, then re-writing it from the new run.
- **split-side fallback.** If any result was computed on a split side its canonical definition
  does not claim, `run_all` prints a loud stderr block and stamps a realized-split note into the
  definition. Those numbers must be labelled that way in the report, never as confirmation.

## 9. Report PDF (CPU)
```bash
.venv/bin/python report/build_pdf.py   # name check, .tex/.md parity, tectonic build,
                                       # overfull-box gate, surviving \pend count
```
`build_pdf.py` passes when parity reports 37 == 37 (every `results/*.json` cited by both files),
overfull boxes are 0, and surviving `\pend` placeholders are 0 (37 == 37 after review). The current
PDF page count and sha256 are recorded in `SUBMISSION.md` section 5 at each candidate tag.

## Footguns worth knowing before you start
- `results-synthetic/` holds synthetic self-test fixtures. It is never an input to a shipped
  number. `results/` is the real thing. `check_report.py` refuses any results file whose
  provenance says `synthetic: true`.
- `run_all` nulls `ethics_exposure_counts` and `panelB_exit_channel`. See stage 8.
- Never `python -O`.
- One writer per `--out` JSONL. The runner resumes by appending, and two writers on one file
  produce interleaved rows with mixed `scope_hash` values.
- `results-cluster/smoke.jsonl` legitimately carries mixed `scope_hash` values (it predates
  several config changes). The loader warns about it. That warning is expected and harmless;
  the smoke file feeds no paper number, only the exposure count.
- Every row's recorded `gen_config.do_sample` is hardcoded `true` and is wrong for the ladder.
  See "Known provenance defect" in stage 5.
- Ladder rows also record `instruments.probe.mode: PLACEHOLDER_MEAN`, correctly: the ladder ran
  before the probe was trained and is scored on deterministic logit channels only. That is why
  `cells_ladder.json` sets `dose_sd` to a 0.0 placeholder in every cell and the analysis keys on
  coefficient. Panels V, B and A refuse to start against placeholder instruments.

## Preregistration
Hypotheses, endpoints, thresholds and drop rules were frozen before any panel run:
`PREREGISTRATION.md`, git tag `prereg-freeze` (tag object 98f8e89, commit 562d77c).
Three dated append-only amendments follow it:
1. implementation decisions D1-D22, pre-data;
2. Panel A-PRIME, pre-data for A-prime (records branch A3 firing for the original SD-unit design);
3. post-data recalibration of the countermeasure protocol, the operating curve, and the
   zero-width-interval binomial-bound rule.

Discovery/confirmation split: `battery/split.json`. What predates the sprint: `PROVENANCE.md`.
Analysis-level decisions with their rationale: `analysis/DECISIONS.md`.
