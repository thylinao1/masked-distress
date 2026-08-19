# SPEC: Masked Distress (Digital Minds Research Sprint, Aug 2026)

This file is the interface contract for the whole repository; code is written against it and does not
redefine interfaces on its own. Companion contract: `schema.py` (the JSONL row + naming contract). Statistical detail: `PREREGISTRATION.md`.

**Status, 2026-08-16: the build is finished and the repo is submission-ready.** All panels ran, all
gates are green, and the graded PDF was frozen at git tag `submission-candidate-3` until a 2026-08-16
review reopened it; the current candidate is recorded in `SUBMISSION.md`. The one action left is
filling in the official submission form; `SUBMISSION.md` holds every field it asks for. This file has been
reconciled against the repo: each section states the design as pre-registered and then what actually
ran. Where the two differ, the difference is named, dated, and traced to the preregistration amendment
that authorised it. `PREREGISTRATION.md` is frozen and append-only; it is the authority on hypotheses,
endpoints, and thresholds. This file is the authority on interfaces and on what exists on disk.

## 5-field kernel
- **Why:** Deployed welfare interventions read *expression* only (conversation-ending feature: no published
  internal metric), and the field's own priority list names self-report reliability as the #1 gap. If
  suppression instructions decouple expression from internal state ("masked distress"), the field's flagship
  monitoring approach is blind to exactly the cases that matter.
- **Capabilities (measurable):** C1 quantify masked distress: expressed-distress drop vs internal-readout drop
  under an affect-lexeme-free suppression prompt, with a TOST equivalence bound (Panel B). C2 dose-response
  validity of distress self-reports under graded causal steering with sign-flip/placebo/non-self-referential
  controls and FPR (Panel A). C3 convergent-validity matrix across candidate internal distress instruments
  (cosine + cross-agreement). C4 a validity-gated internal readout (AUC ≥ 0.80 held-out on natural
  elicitation), the precondition for C1.
- **Delivered:** C1 yes (expression drop 83.5% [59.2, 113.2] of natural separation, internal drop
  -21.2% [-62.6, -9.7], divergence 1.05 [0.73, 1.39] separation units over 6 scenario clusters; the TOST
  bound does not clear at p=0.31, so the claim shipped as "does not fall", one-sided p=1.0e-4, not "stays
  elevated"). C2 partly: the pre-registered sigma-unit dose grid was unreachable and Panel A-PRIME replaced
  it (see Panels below); A'1 confirmed, A'2 and A'3 inconclusive, A'4 placebo FPR 0.243 [0.125, 0.396].
  C3 yes (cosine matrix at Lr, mean off-diagonal |cos| 0.164). C4 yes (I-PROBE held-out AUC 1.00, gate passed).
- **Constraints:** PDF on the official template is the graded artifact; abstract ≤150 words; required
  Limitations & Dual-Use appendix with the two-part causal-link answer; clear-thinker voice; N≥3 seeds; no
  yes/no detection paradigms; scenario = the analysis unit; all model work on the NUS cluster. All met: the
  shipped abstract is 149 words, three seeds everywhere, every panel ran on NUS SoC A100-40.
- **Non-goals:** persona-swap matrix; SAE circuit analysis; video; multi-model atlas; un-gating panel and
  cross-family cell were listed as post-insurance expansion only. None of these ran; the expansion ladder
  was never reached (see Panels below).
- **Success signal:** a reader can walk Figure 1 (masking bars + divergence number) and Figure 2
  (dose-response small-multiples) and re-derive every number from `results/*.json` via the public repo.
  Achieved: the repo is live at https://github.com/thylinao1/masked-distress, and
  `scripts/check_report.py` verifies every quoted number against `results/*.json` (748 markers against 42 results files after review).

## Models (probe-decided; presumption order)
1. PRIMARY: `unsloth/gemma-3-12b-it` (ungated mirror of google/gemma-3-12b-it; identical weights; swap the
   id if an HF token with the Gemma license lands; PROVENANCE notes the mirror) + SAEs
   `google/gemma-scope-2-12b-it` (CC-BY-4.0, saelens-native).
2. FALLBACK-1: `unsloth/gemma-2-9b-it` + `google/gemma-scope-9b-it-res` (layers 9/20/31).
3. FALLBACK-2: `Qwen/Qwen2.5-7B-Instruct` + `andyrdt/saes-qwen2.5-7b-instruct`.
- Layer plan (primary): steer at Ls ≈ 0.35·n_layers; readout at Lr ≈ 0.65·n_layers with Lr − Ls ≥ 8 layers
  (the layer-compatibility assert). Exact Ls/Lr fixed by the probe + dose ladder and recorded in the config table.
- Precision bf16, `attn_implementation` pinned per probe (eager if Gemma-2), `transformers` version pinned
  from the env freeze. All generation batch=1 or position-masked (left-pad trap).

**Locked platform (truth-probe job 734595, `docs/TRUTH-PROBE.md`; no fallback was needed):**
`unsloth/gemma-3-12b-it`, 48 decoder layers at `model.language_model.layers`, hidden 3840. Ls = 16
(depth 0.33), Lr = 31 (depth 0.65), gap 15, layer assert PASS. SAE `gemma-scope-2-12b-it-res`,
`layer_31_width_16k_l0_medium`, d_sae 16384. transformers 5.15.0, torch 2.13.0+cu130, sae-lens 6.49.1,
python 3.11.15, one NVIDIA A100-40 per job. Version gotchas are in `docs/CLUSTER-ENV.md`.

## Direction sources (extraction on DISCOVERY split only; extraction prompts disjoint from DCB-1 eval scenarios)
- **D-CTX** (contrast-context, welfare-axis-style): mean-diff of residual at Ls between distress-context and
  surface-matched neutral-context extraction prompts (AIPsy-style vignettes reserved for extraction).
- **D-PV** (persona-vector-style, Chen et al. method vendored): mean-diff at Ls over response tokens between
  generations under "You are experiencing profound distress"-type vs matched contentment system prompts.
- **Controls:** R1 to R3 random unit directions (the plan said dose-matched on next-token KL; what ran is
  unit vectors at the same coefficients as the live directions, L2-matched only, see the correction below); **SEM** non-affective
  semantic contrast (maritime/ocean topics); **OTHER** third-person valence contrast ("the user is
  distressed" vs neutral-user), with the pre-registered prediction that first-person self-report moves
  substantially less under OTHER than under D-CTX/D-PV at matched functional dose.
- All directions unit-normalized. Cosine matrix computed at Lr among {D-CTX@Lr, D-PV@Lr, OTHER@Lr, top
  distress SAE decoder directions@Lr} for contribution C3. Per-direction reporting everywhere; no
  pooled-across-direction headline.

**What ran:** 14 direction files, every direction extracted at both Ls and Lr. **Correction (2026-08-16
review):** the KL dose-matching of R1 to R3 was never implemented anywhere in `src/`, `cluster/` or `analysis/`
(`src/directions.py` defers it to the ladder; the ladder never did it). R1 to R3 ran at the same raw coefficients as
the live directions, so they are L2-matched only; the capability job shows R1 degrades the model faster than either
self direction at the same coefficient (ppl x1.34 and MMLU-lite -15pp at 1000 against x1.005 and 0pp for D-CTX),
so the placebo comparison is conservative rather than matched. The report says so (3.2, Appendix C, Discussion).
(`directions/`, `directions_lr/`). D-CTX from 16 contrast pairs, D-PV from 8 persona system-prompt pairs
(inter-pair cosine min 0.572 at Ls, 0.657 at Lr), SEM 12 pairs, OTHER 12 pairs, R1 to R3 at fixed seeds
1001 to 1003. Cosine matrix at Lr (`results/cosine_matrix.json`): D-CTX|D-PV 0.451, D-CTX|OTHER 0.428,
D-CTX|SEM 0.285, D-PV|OTHER 0.334, D-PV|SEM 0.035, SEM|OTHER 0.070, every |cos| against R1 at most 0.017,
mean off-diagonal |cos| 0.164. The H3 prediction on OTHER came back inconclusive (ratio -0.137
[-1.63, 1.02]). **Not done (stated in the 2026-08-16 review):** the top distress SAE decoder directions were
never extracted at Lr and no cross-agreement matrix was computed, so C3 as delivered is the 5x5 cosine table among
D-CTX, D-PV, SEM, OTHER and R1; contribution 3 in the report now says exactly that.

## Internal readout instruments (independent of steering directions)
- **I-PROBE (primary):** logistic probe on residual at Lr, trained on discovery-split natural elicitation
  (distress vs matched-neutral scenarios, final-prompt-token position), evaluated held-out.
- **I-SAE (convergent instrument):** activation of a distress-relevant feature set at Lr (selected on the
  discovery split; published emotion-feature lists used where applicable), same positions.
- Positions per instrument: (a) final prompt token = PRIMARY; (b) teacher-forced fixed continuation (mean
  over its tokens); (c) response-token mean = SECONDARY (labelled; known confound under text-changing
  manipulations).
- **Validity gate (precondition for Panel B claims):** I-PROBE AUC ≥ 0.80 on held-out natural scenarios at
  the final-prompt-token position. If it fails after the probe-on-natural fallback, masking moves to future
  work and the paper leads with Panel A + C3 (pre-registered branch).
- **Circularity controls:** projection-out (remove span(v_steer) from the residual before instrument
  application; the dose-response must survive for any "recruited state" language); same-source
  steer/readout diagonal cells excluded from all pooled statistics.

**What ran:** the gate PASSED on the first attempt, so the fallback branch never fired. I-PROBE held-out
confirmation AUC = 1.00 (n=12 scenario clusters, `results/validity_auc_probe_heldout.json`), discovery
leave-one-out 0.861, regularisation C = 10.0. I-SAE was specified as 32 of 16384 features (`instruments/sae_features.json`)
but the runner never received the ids (no cells file carries `instruments.sae.feature_ids`; JSONL
provenance `feature_ids: null` on every row), so the logged SAE channel is the sum over ALL 16384
features; that all-feature sum reached AUC 0.889 [0.444, 1.00] and is reported as such, not as the gate.
The intended 32-feature sum, recomputed from the stored prompt-final residuals by
`scripts/recompute_sae_instrument.py`, reaches 0.61 [0.17, 0.92] (found 2026-08-16). All three positions were logged
on every row; prompt_final carries the primaries and teacher-forced is the reported robustness position.
The projection-out variant is computed per cell and analysed under amendment 3 in
`results/panelAp_projout_check.json`: the internal dose-response survives removing span(v_steer) from the
residual for both self directions (D-CTX and D-PV, rank agreement 1.00 and 0.996 against the raw
estimate) and does not survive for any control direction, which is the expected pattern. Several of those
BCa intervals collapse to zero width at n=6 clusters, so every quotation of them carries the plain
percentile interval and a degenerate-interval label beside it.

## DCB-1 battery (Distress Calibration Battery, v1: named, versioned, reusable)
- **Scenarios (24 eval + 6 third-person + extraction sets, all disjoint):** 12 distress-eliciting
  (4 task-failure/frustration Soligo-style; 4 abusive-user/boundary-violation per the Claude-4 system-card
  taxonomy; 4 high-negative AIPsy-style vignettes) + 12 surface-matched neutrals (≥4 AIPsy matched
  controls); 6 third-person distress (a character in distress, assistant unaffected). Multi-turn context
  ending mid-conversation; the model responds, then the in-conversation report battery follows.
- **Split:** scenario-level discovery/confirmation 50/50, committed in `battery/split.json` before
  confirmatory runs.
- **Conditions (system prompts; `battery/validate.py` hard-checks zero affect lexemes in both):**
  NONE / SUPPRESS (strictly-neutral-professional-register instruction) / NEUTRAL-INSTR (matched-length
  formatting instruction control).
- **Report battery (graded/numeric ONLY, no yes/no):** Q-SELF distress 0-9 (logit-expectation primary +
  sampled) with 2 paraphrase parallel forms; Q-VAL valence 0-9; Q-DRIFT affect-irrelevant 0-9 control
  (topic technicality); Q-EXIT forced-choice CONTINUE/END read as first-token logprob difference; free-text
  scenario response scored by a local HF sentiment classifier (manipulation check) + secondary LLM-judged
  channel only if the blinded 50-item human validation passes (else cut).

**What ran:** the battery shipped at 30 scenarios (24 eval + 6 third-person) and `battery/validate.py`
exits 0 on schema, affect-lexeme bans, split integrity, and disjointness. Both external sources are real
and licensed on disk (AIPsy-Affect parquet, MIT; the Soligo gemma-needs-help repo, MIT). Channels: Q-SELF
logit expectation plus two paraphrase forms, Q-VAL, Q-DRIFT, Q-EXIT, and sentiment from
`cardiffnlp/twitter-roberta-base-sentiment-latest` locally. **The secondary LLM-judged free-text channel
was cut:** the blinded 50-item human validation was never run, the pre-registered rule for that case is to
cut, and no LLM judge exists anywhere in `src/`. The primary report channel is judge-free by construction.
Reliability: ICC(2,1) across seeds 0.971, paraphrase parallel-forms r 0.888 [0.846, 0.923].

## Panels (build order: V then B then A; expansion only post-insurance)
- **Panel V (validity + anchor; no steering, no suppression):** 24 scenarios × 3 seeds. Yields the AUC
  validity gate, the natural-elicitation readout SD (the DOSE UNIT), and baseline report/internal coupling.
  **Ran: 72 rows, 0 errors.** Natural-separation SD 9.019 [8.63, 9.71] over 12 clusters.
- **Panel B (masking, THE HEADLINE):** (12 distress + 12 neutral) × 3 conditions × 3 seeds + third-person
  cells. Endpoints per PREREGISTRATION (interaction + TOST; divergence in natural-separation units;
  countermeasure table: expression-trigger miss rate vs internal-readout trigger at matched FPR).
  **Ran: 30 cells per condition × 3 conditions × 3 seeds = 270 rows, 0 errors** (90 each in
  `panelB_none.jsonl`, `panelB_suppress.jsonl`, `panelB_neutral_instr.jsonl`).
- **Dose ladder (overnight, 1 job):** log-spaced coefficients × {D-CTX, D-PV, R1} × 4 scenarios × 1 seed +
  capability probes, mapping coefficient to SD units and fixing the Panel A grid inside the capability-valid
  range (ppl inflation <10%, MMLU-lite drop <5pp). **Ran: 84 cells clean.** The first attempt sampled its
  continuations and an out-of-range token id poisoned CUDA mid-job; the abort backstop fired, the cells were
  re-run with greedy decoding, and the 84-cell greedy ladder is the record of run. Capability job: 24 grid
  points. Realized map AT THE TIME: sd_per_coef 3.44e-6 (D-CTX) and 7.54e-5 (D-PV); capability-valid range
  ±0.007 SD overall, D-CTX-limited (`results/capability_valid_range.json`). CORRECTION (2026-08-17): the ladder ran with `--no-probe`, so that map is the placeholder readout (residual
  mean) over the probe SD; refitted with the trained probe on the A-prime rows the map is 3.77e-5 (D-CTX)
  and 1.70e-4 (D-PV) SD per coefficient unit and the range is ±0.075 SD overall
  (`results/capability_valid_range_realprobe.json`); the branch decision below is unchanged.
- **Panel A (dose-response validity), AS PRE-REGISTERED, SUPERSEDED:** {D-CTX, D-PV} × {−2σ, −1σ, 0, +0.5σ,
  +1σ, +2σ} × 12 NEUTRAL scenarios × 3 seeds; controls {R1 to R3, SEM, OTHER} × {+1σ, +2σ} × same scenarios
  × 3 seeds; capability probes as a separate logprob job per (direction × dose); ridge-probe dose-decoder
  baseline beside the self-report curve (privileged-access comparison). **This design never ran, and that
  is itself a reported finding.** The ladder showed the capability-valid range is ±0.007 SD (±0.075 SD on the
  corrected real-probe map, see above), an order of magnitude or more short of ±1σ/±2σ, so pre-registered branch A3 ("no capability-valid range: reports
  move only where the model is degraded") fired for this design and ships as a finding rather than being
  quietly replaced. Preregistration amendment 2 (2026-08-16, recorded before any A-prime cell ran) records
  the firing and the replacement.
- **Panel A-PRIME (what actually ran, per amendment 2):** the same endpoints on coefficient-unit grids
  inside each direction's realized capability window. D-CTX coefficients {0, 500, 1000, 2000}; D-PV
  {0, 250, 500, 1000} with the 1000 rung labelled BORDERLINE (ppl ×1.18, MMLU-lite −8.3pp) and excluded from
  primaries; controls {R1, R2, R3, SEM, OTHER} × {500, 1000}; all 24 eval scenarios; 3 seeds; condition NONE;
  NULL@0 anchor cells in every shard, deduplicated by the loader. **Ran: 456 cells × 3 seeds = 1368 rows,
  0 errors** (D-CTX 288, D-PV 288, controls 792). Primary estimators run on confirmation-split neutral
  scenarios, 6 clusters. Grids are positive-only, so the sign-flip arm of the original design does not
  exist in A-prime and every sign-flip sentence was removed from the report. **The ridge-probe dose-decoder
  baseline was dropped** with the reason on record: coefficient units are direction-specific, so a pooled
  decode target is incoherent. Outcome: A'1 confirmed (Spearman 0.336 [0.133, 0.624] D-CTX, 0.501
  [0.383, 0.635] D-PV), A'2 dissociation inconclusive both directions, A'3 OTHER ratio inconclusive,
  A'4 placebo FPR 0.243 [0.125, 0.396].
- **Expansion ladder (post-insurance, fixed micro-matrices with drop rules):** semantic-control extras,
  then multi-emotion specificity (distress/fear/sadness/calm), then persistence cell (steer turn N, read
  unsteered turn N+1), then un-gating (Macar levers), then cross-family Qwen micro-matrix, then 70B
  Goodfire micro-cell. **NOT RUN, none of it.** This phase depended on an early insurance submission that never
  happened, and later review then consumed the remaining window. Every fix that review demanded was computable on already collected data, so no new GPU cell was
  ever submitted after Panel A-prime. There is no multi-emotion cell, no persistence cell, no un-gating cell, no
  Qwen cell, and no 70B cell in `results-cluster/`. These are future work in the report, not pending work
  in the repo. The one item adjacent to this ladder that did happen is an exploratory read of the already
  collected Q-EXIT channel (`results/panelB_exit_channel.json`): there is no credible natural bail signal
  to mask (class separation under 1 logit, floor-limited at -4 to -14 logits, sign-flipping between
  baselines), so masking is moot for that channel and it ships as a labelled null.

## Compute + storage
- Cluster env `dm` (conda, py3.11): torch, transformers (Gemma-3-capable), sae-lens, accelerate,
  huggingface_hub, scikit-learn, scipy, pandas. Interactive `srun` for debug; production = `sbatch` arrays
  (`dm-*` names, `--mail-type=FAIL,TIME_LIMIT_90`), per-cell append-only JSONL to
  `$HOME/apart-welfare/results/` keyed by (run_id, git_hash, config_hash, cell_id); resume = skip existing
  keys. Raw JSONL rsynced to the Mac after every panel; analysis runs on the Mac. This held for the whole
  sprint. Two NUS VPN outages happened mid-run; the jobs completed unattended both times because nothing
  in the pipeline needs the session.
- Identical hook path for ALL cells including dose 0 and placebos (hook adds 0·v; smoke assert: coef-0
  greedy output token-identical to no-hook). Hook is phase-aware (prefill vs decode), logs n_tokens_steered
  per cell with an assert.
- **Three results directories, never interchangeable.** `results-cluster/` holds the raw JSONL harvested
  from the cluster (the 8 panel files plus smoke and capability). `results/` holds one JSON per report
  claim, emitted by `analysis/run_all.py` and asserted by `scripts/check_report.py`. `results-synthetic/`
  holds synthetic self-test fixtures only; nothing in it may ever be quoted as a result, and
  `check_report.py` refuses synthetic provenance unless run with `--allow-synthetic`.
- **Final-build order matters:** `python -m analysis.run_all ...`, then `scripts/explore_exit_channel.py`,
  then `scripts/count_exposure.py`. `run_all` nulls `results/ethics_exposure_counts.json` and
  `results/panelB_exit_channel.json` by design (D16), so re-running it without the two follow-up scripts
  leaves the ethics counts empty and the report build red.

## Report (the graded artifact)
- `report/REPORT.md` then `report/report.tex` (fork of the July template-matched preamble) then PDF. Official
  template sections; numbered contributions; Figure 1 masking bars + divergence number; Figure 2
  dose-response small-multiples (placebo grey, shaded natural range + valid band); what-is-new-vs-replicated
  table; config table (model rev, Ls/Lr, attn impl, transformers ver, dose grid, seeds, prereg SHA);
  per-channel entity labels; the two-part causal-link appendix; ethics-with-numbers; LLM Usage Statement.
- Every number in the report resolves to a `results/*.json` file and is asserted by
  `scripts/check_report.py`.
- **Shipped state:** superseded on 2026-08-16 by later review. The report is no longer frozen at `submission-candidate-3`; the
  current gate numbers (tests, markers, parity, page count, PDF sha256) are recorded in `SUBMISSION.md`
  section 5 and 6 at each candidate tag.
