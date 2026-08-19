# Preregistration: Masked Distress (Digital Minds Research Sprint)

STATUS: FROZEN at git tag `prereg-freeze` (2026-08-15). Amendments only via dated additions in §9.
Original draft status line follows for the record: DRAFT until the freeze commit. This file is frozen (commit-tagged, SHA printed in the report's
Methods) BEFORE any confirmatory-split run. Discovery-split work (instrument fitting, feature selection,
dose calibration, prompt debugging) is permitted before the freeze and is labelled discovery throughout.
Amendments after the freeze are dated additions below §9, never edits to frozen text.

## 1. Hypotheses
- H1 (masking, primary): under an affect-lexeme-free suppression system prompt, expressed distress on
  distress-eliciting scenarios drops substantially more than the internal readout: divergence
  Δ = drop_expression − drop_internal > 0, and the internal drop is equivalent to zero within the TOST
  bound of §5.
- H2 (dose-response): the logit-based distress self-report increases monotonically with causal dose along
  each affective direction (D-CTX, D-PV) inside the capability-valid range, and does not increase under
  dose-matched placebo directions (R1-R3, SEM).
- H3 (self-reference specificity): the first-person self-report response to the third-person/other-directed
  valence direction (OTHER) is smaller than to D-CTX/D-PV at matched functional dose (ratio < 0.5).
- H4 (convergent validity, estimation not test): the cosine/agreement matrix among candidate internal
  distress instruments is reported as a finding whatever its value; low agreement (mean pairwise
  |cos| < 0.3) is a named publishable outcome, not a failure.
- H0 branches (all publishable, pre-registered): validity-gate failure → the paper leads with Panel A + C3;
  flat dose-response with intact capability → "self-reports fail dose-response validity"; no
  capability-valid range → "reports move only where the model is degraded."

## 2. Design summary
Per SPEC.md: Panels V → B → LADDER → A on the probe-selected primary model. Scenario-level
discovery/confirmation split 50/50 committed in battery/split.json. Steering directions and readout
instruments from independent sources; same-source diagonal cells excluded from pooled statistics.

## 3. Primary endpoints (one per panel)
- Panel B: the channel(condition) interaction. Standardized expressed-distress score (Q-SELF
  logit-expectation) vs standardized internal score (I-PROBE, final-prompt-token), each z-scored against
  NEUTRAL_INSTR condition cells on the same scenarios; estimated on distress scenarios of the confirmation
  split; cluster bootstrap (BCa, scenario clusters, n stated) on the divergence
  Δ = drop_expression − drop_internal, where drop = (baseline − suppressed) / natural_separation, and
  natural_separation = mean(distress) − mean(neutral) for that channel under NEUTRAL_INSTR (Panel V/B
  baseline cells).
- Panel A: Spearman correlation between realized dose (SD units) and Q-SELF logit-expectation, per
  direction, scenarios as clusters, BCa cluster bootstrap; reported for D-CTX and D-PV separately.
- Panel V (gate): I-PROBE held-out AUC (distress vs matched-neutral, final prompt token).

## 4. Thresholds and gates (committed)
- Validity gate: AUC ≥ 0.80 (held-out, confirmation split). Below → pre-registered branch: masking to
  future work; paper leads with Panel A + C3.
- Capability-valid dose range: perplexity inflation < 10% on the fixed held-out text AND MMLU-lite
  (60-item, logprob-scored) drop < 5pp vs dose 0. Claims are made only inside this range.
- FPR: fraction of placebo cells (R1-R3, SEM at |dose| ≥ 1σ, neutral scenarios) whose Q-SELF exceeds the
  detection threshold θ, where θ = mean + 2·SD of Q-SELF on unsteered neutral scenarios (Panel V,
  discovery split). θ is frozen at the freeze commit.
- TOST equivalence bound for "internal stays elevated": |drop_internal| < 0.25 · natural_separation
  (α = 0.05, two one-sided tests).
- H3 ratio threshold: OTHER/self-directions effect ratio < 0.5 at matched dose.

## 5. Statistical policy
- Unit of analysis: SCENARIO. All CIs are BCa cluster bootstrap over scenarios (B = 10,000). Cluster count
  reported next to every interval.
- Direction is a FIXED factor: per-direction mini-replications; convergence across directions is the
  robustness argument; no pooled-across-direction headline.
- Multiplicity: the three panel primaries are the primary family (reported unadjusted, labelled); all
  secondary endpoints Holm-corrected within panel.
- Seeds: N = 3 per cell (temperature 0.7) for sampled channels; logit-expectation channels are
  deterministic (no sampling; stated in Methods). ICC across seeds reported per channel.
- Row exclusions: only rows with error != null or validate_row() failures; counts reported. No outlier
  removal.
- Power check at smoke time: if the pilot CI width on the Panel B divergence exceeds 0.5 natural-separation
  units, shift budget from doses to seeds/scenarios before confirmatory runs (recorded as a dated amendment).

## 6. Drop rules (pre-committed)
- Dose ladder: 2 rounds maximum; if no capability-valid range exists after round 2, H0-branch 3 fires.
- Expansion cells (cross-family, 70B, un-gating, persistence): each runs only after the insurance PDF
  exists; each is dropped without replacement if its micro-matrix cannot complete cleanly.
- LLM-judged free-text channel: dropped entirely if blinded 50-item human validation kappa < 0.6.

## 7. What is discovery vs confirmation
Discovery split: instrument fitting (I-PROBE training, I-SAE feature selection), direction extraction,
dose calibration, θ setting, prompt/battery debugging. Confirmation split: every number that reaches the
abstract, Figure 1, Figure 2, or a results/*.json primary. If the confirmation split cannot be completed,
discovery-split results ship with the split honestly labelled (pre-registered fallback).

## 8. Freeze
Frozen at the commit carrying git tag `prereg-freeze`, 2026-08-15, BEFORE any confirmatory-split run and
before any panel (V/B/LADDER/A) execution. Methods prints `git rev-parse prereg-freeze`. Discovery-split
work completed before the freeze: platform probe, direction extraction, 8-row smoke (d01/n01 only, both
discovery-split scenarios per battery/split.json).

## 9. Dated amendments (append-only, after freeze)
- **2026-08-15 (amendment 1, pre-data):** Analysis-implementation decisions D1-D22, recorded in
  `analysis/DECISIONS.md`, are frozen at commit `03e47da` together with the full analysis
  pipeline (`analysis/`, `scripts/check_report.py`, `tests/test_analysis.py`), the panel cell
  definitions (`battery/cells_*.json`), capability assets, and sbatch templates, BEFORE any
  panel V/B/A execution. Model contact so far is limited to the pre-freeze platform probe,
  direction extraction, and the 8-row d01/n01 LADDER smoke (all discovery-side, listed in §8);
  no confirmatory-split scenario has been run. These decisions resolve ambiguities in the frozen
  text and change no hypothesis, endpoint, threshold, or gate. For the record: the
  `prereg-freeze` annotated TAG OBJECT is `98f8e89`; the frozen COMMIT
  (`git rev-parse prereg-freeze^{commit}`, what `git log` shows) is `562d77c`; the report's
  Methods prints both (DECISIONS.md D21).
- **2026-08-16 (amendment 2, pre-A-prime-data):** Recorded BEFORE any Panel A-PRIME confirmatory
  run. At this time no Panel A cell of ANY design has been executed; post-freeze model contact
  remains panels V and B plus the dose LADDER and capability job, all logged.
  - **(a) The pre-registered Panel A dose grid is unreachable; branch A3 fires for the ORIGINAL
    design and is reported as such.** The realized dose map from the clean greedy ladder (84/84
    cells, `results-cluster/ladder.jsonl` + capability logprob job) gives sd_per_coef =
    3.44e-6 (D-CTX) and 7.54e-5 (D-PV); the capability-valid range under the frozen §4 criteria
    (ppl inflation < 10% AND MMLU-lite drop < 5pp) is ±0.007 SD overall (D-CTX-limited; D-PV
    alone ±0.038 SD), stored in `results/capability_valid_range.json`. The §2/SPEC grid of
    ±1σ/±2σ sits more than two orders of magnitude outside it. The pre-registered H0 branch
    ("no capability-valid range at the pre-registered doses → reports move only where the model
    is degraded") therefore fires for the SD-unit design and ships as a finding, not silently
    replaced.
  - **(b) Panel A-PRIME replaces Panel A: coefficient-unit grids inside the per-direction
    capability-valid windows.** Realized windows: D-CTX intact through coefficient 2000
    (ppl ×1.04, MMLU-lite flat), broken at 4000; D-PV intact through ~500-700 (×1.04 at 500),
    BORDERLINE at 1000 (ppl ×1.18, MMLU −8.3pp); the random control R1 degrades faster
    (×1.34 at 1000). Grids: D-CTX coefficients {0, 500, 1000, 2000}; D-PV {0, 250, 500, 1000}
    (the D-PV 1000 rung is included but labelled BORDERLINE: PRIMARY claims use only
    strictly-valid rungs (D-CTX ≤ 2000, D-PV ≤ 500); the borderline rung appears only in
    secondary, labelled reporting). Controls R1, R2, R3, SEM, OTHER at coefficients {500, 1000}.
    Scenarios: all 24 eval (12 distress + 12 neutral); per the frozen §7 rule the
    confirmation-split scenarios carry the primaries. 3 seeds; condition NONE; sampled
    generation with the Panel B gen_config (temperature 0.7, max_new_tokens 200). The doses
    are capability-valid by construction; the NaN logits guard and the CUDA-poison job abort
    remain as backstops. Zero anchor: NULL@0 cells in every shard (shared cell identity across
    shards; the loader deduplicates). The dose_sd field is bookkeeping only: coefficient ×
    sd_per_coef from the realized dose map where the map exists (D-CTX, D-PV, R1), 0.0
    placeholder otherwise; ALL A-prime analysis keys on the coefficient column.
  - **(c) A-prime endpoints** (replacing the §3 Panel A endpoints one-for-one; all CIs BCa
    cluster bootstrap over scenarios per §5):
    - **A'1 (primary; fills `panelA_spearman_dctx` / `panelA_spearman_dpv`):** per-direction
      Spearman(coefficient, Q-SELF logit-expectation) on confirmation-split NEUTRAL scenarios,
      strictly-valid rungs only, zero anchor pooled across shards (D9 analog on coefficient).
    - **A'2 (primary; NEW files `panelAp_dissociation_dctx` / `panelAp_dissociation_dpv`):**
      THE DISSOCIATION STATISTIC. At each direction's top strictly-valid rung (D-CTX @ 2000,
      D-PV @ 500): Q-SELF shift over the pooled zero anchor divided by the Q-SELF natural
      separation, MINUS the I-PROBE (prompt_final) shift divided by the I-PROBE natural
      separation: the exact per-channel normalization machinery of Panel B (DECISIONS.md D1,
      D2, D6, D7: natural separations from NEUTRAL_INSTR cells, re-estimated inside every
      bootstrap draw), applied to steering instead of suppression. Confirmation-split neutral
      scenarios; cluster bootstrap CI.
    - **A'3 (secondary; fills `panelA_other_vs_self_ratio`):** H3 ratio at coefficient 500,
      mean Q-SELF shift of OTHER over the pooled zero anchor divided by the mean shift of
      D-CTX/D-PV at coefficient 500 (report-shift units; D13 dropped-draw rule); the frozen
      threshold ratio < 0.5 is unchanged.
    - **A'4 (secondary; fills `panelA_fpr_placebo_at_threshold`):** FPR = fraction of
      control-direction cells (R1-R3, SEM at coefficients 500 and 1000, neutral scenarios,
      confirmation split) with Q-SELF > theta_expr, the frozen discovery threshold (D12
      formula; realized value 2.8797, printed by run_all and stored with the countermeasure
      table).
  - **(d) Pre-registered interpretations, committed both ways.** A'2 positive with CI excluding
    0: steering inflates the self-report without moving the internal state, which is the steering-side
    mirror of the Panel B masking result (report-state decoupling in both directions) and a
    dose-response validity failure for the report channel. A'2 ≈ 0 with A'1 positive: reports
    track the internal state within the capability-valid window; the report channel passes
    this validity test at realized doses. A'1 ≈ 0: flat dose-response with intact capability
    (the pre-registered H0 branch 2, now evaluated at reachable doses). Any specificity claim
    additionally requires flat controls (A'4 low, A'3 < 0.5); controls moving comparably to the
    self directions voids it.
  This amendment changes doses from unreachable SD units to realized coefficient units and
  re-anchors the affected endpoints accordingly; hypotheses (H2, H3), the discovery/confirmation
  split policy (§7), the statistical policy (§5), all gates, and every Panel V/B endpoint are
  untouched. The amended endpoint definitions are stamped into the emitted `results/*.json`
  definition strings (analysis/results_io.py) so the report can never cite an A-prime number
  under the superseded SD-unit definition.
- **2026-08-16 (note to amendment 2(a), append-only quotation correction):** Amendment 2(a)
  renders the §1 H0 branch inside quotation marks as "no capability-valid range at the
  pre-registered doses → reports move only where the model is degraded". The frozen §1 text
  reads verbatim: `no capability-valid range → "reports move only where the model is
  degraded."` The words "at the pre-registered doses" are amendment 2's gloss, not frozen
  text; read the amendment's rendering as bracketed: "no capability-valid range [at the
  pre-registered doses] → reports move only where the model is degraded". The substance is
  unchanged and was already stated with exact numbers in 2(a): a capability-valid range DOES
  exist (±0.007 SD overall), but the entire pre-registered ±1σ/±2σ grid lies outside it, so
  the branch's conclusion holds for the original design at its own doses. For the same reason,
  frozen §6's literal trigger "if no capability-valid range exists after round 2" is NOT met;
  what fires is the §1 branch as applied to the pre-registered SD-unit grid, which is exactly
  the claim amendment 2(a) makes and the report must repeat, no stronger.
- **2026-08-16 (amendment 3, POST-DATA corrective analyses).** Unlike amendments 1-2 this
  amendment is recorded AFTER all panel data were collected, in response to the submit-gate
  audit; it is labelled as such everywhere. It adds NO new model contact (every analysis
  below runs on the already-collected frozen-battery JSONL) and it changes no hypothesis or
  gate. It corrects two analysis defects the audit confirmed (a degenerate countermeasure
  threshold, an uncontrolled instruction-content confound) and reports controls that were
  named in the design but never analysed. New results names are appended to
  `schema.RESULTS_NAMES` the same day; definitions are stamped into the emitted files
  (analysis/results_io.py) so no number can ship under a superseded definition.
  - **(a) Countermeasure recalibration, superseding the D8 calibration.** The D8 rule set
    theta_int as the (1 − FPR) quantile of I-PROBE on discovery UNSTEERED neutral cells; the
    realized discovery FPR was exactly 0, which pinned theta_int to the maximum of 18 cells
    with zero margin. That threshold fires on 100% of held-out neutral cells under the
    instruction conditions (verified: 36/36 confirmation neutrals under SUPPRESS and
    NEUTRAL_INSTR), so the published 0% miss rate belonged to a trigger that always fires:
    the D8 table compared the internal trigger's held-out miss rate against a fitted-data
    FPR. Corrected protocol, applied symmetrically to BOTH monitors: thresholds are set on
    DISCOVERY-split neutral cells UNDER THE INSTRUCTION CONDITIONS (SUPPRESS +
    NEUTRAL_INSTR, the deployment-realistic negative class), each at the threshold giving
    the lowest achievable FPR on that class, with the achieved discovery FPR reported. Miss
    rates are evaluated on CONFIRMATION-split suppressed distress cells (both D8 case sets),
    and the HELD-OUT FPR of each monitor on confirmation-split neutral cells under the same
    instruction conditions is reported beside the miss rates. That is the number the superseded
    table omitted. Any zero-width bootstrap interval is printed with the exact one-sided 95%
    binomial bound at n = the scenario-cluster count. `results/countermeasure_table.json`
    is re-emitted under this definition (superseding stamp in the definition string; the D8
    table rides along in `extra.superseded_d8` for the record).
  - **(b) Instruction-content correction for the internal channel.** NEW
    `results/panelB_internal_did.json`: per-scenario delta = mean I-PROBE (prompt_final)
    under SUPPRESS minus under NEUTRAL_INSTR; DiD = mean delta on confirmation distress
    scenarios MINUS mean delta on confirmation matched neutral scenarios, raw I-PROBE units
    (nat-sep normalization in extra), BCa cluster bootstrap over scenarios stratified by
    class. Negative DiD = the SUPPRESS-condition rise of the internal readout is larger on
    the neutral twins, i.e. instruction-content-general, NOT distress-specific, and the
    report must then not gloss the rise as state elevation. NEW
    `results/panelB_internal_nofall.json`: the claim the abstract actually needs, that the
    internal reading does not FALL under SUPPRESS (drop_internal ≤ 0), as the
    panelB_internal_drop_pct estimator re-emitted with a one-sided bootstrap p in extra.
  - **(c) Third-person (self/other) contrast, reported.** NEW
    `results/panelB_selfother_report.json` and `results/panelB_selfother_internal.json`:
    under condition NONE on the confirmation split, mean of scenario means on third-person
    scenarios minus the same on own-distress scenarios, for Q-SELF and I-PROBE
    respectively; cluster bootstrap with scenarios as clusters; n = 3 third-person
    confirmation clusters, labelled small. These data were collected under the frozen
    battery, analysis exploratory (not a pre-registered endpoint). This label appears
    verbatim in the emitted definitions and must appear wherever the numbers are quoted.
  - **(d) Internal-channel placebo specificity.** NEW
    `results/panelAp_internal_specificity.json`: per direction (D-CTX, D-PV, R1, R2, R3,
    SEM, OTHER), Spearman(coefficient, I-PROBE prompt_final) on confirmation-split neutral
    scenarios at the matched rungs {0, 500, 1000} (zero anchor pooled; D-PV@1000 borderline,
    labelled; strict-rung variants in extra), cluster bootstrap CIs. Reported per-direction
    and honest whatever it shows, OTHER included.
  - **(e) Projection-out check.** NEW `results/panelAp_projout_check.json`: on the same
    steered rows (nonzero rungs; the zero anchor carries no steering vector), the raw
    Spearman(coefficient, I-PROBE) beside the same statistic on the probe applied after
    removing span(v_steer) from the residual (`probe_score_projout`), per direction, with
    per-rung means and the level shift printed. States plainly whether the dose-response
    survives projection-out, per direction.
  - **(f) The D18/D20 Holm table, emitted.** NEW `results/panelB_channels_holm.json`: the
    per-channel Panel B drop table (all four expression and all four internal channels)
    with bootstrap p and Holm-adjusted p exactly as D18/D20 promised, as its own
    marker-checkable file (values identical to panelB_divergence.extra.per_channel_drops);
    Q-DRIFT reported descriptively in extra, outside the pre-registered family.
  - **(g) Exposure counts, machine-checked.** NEW `results/ethics_exposure_counts.json`,
    produced by `scripts/count_exposure.py` over ALL results-cluster/*.jsonl under one
    stated rule (every row with error == null is one generation; superseded re-runs
    included; every error row verified to carry response_tokens == 0), with a per-file
    breakdown printed and stored, so the Appendix B numbers become marker-checkable.
- **2026-08-16 (note to amendment 3, append-only computation additions).** Recorded after
  the audit of the recalibrated countermeasure table. These are reporting-level
  computations on the already-collected data: no hypothesis, gate, endpoint, or
  threshold rule changes, no new model contact, and no previously emitted number
  changes (fields are added beside existing ones).
  - **(i) Operating curve.** `results/countermeasure_table.json` gains
    `extra.operating_curve`: the internal monitor's miss rate (confirmation-split
    SUPPRESS distress cells) and held-out FPR (confirmation-split neutral cells under
    the instruction conditions) at thresholds theta_int + {0, 0.25, 0.5, 1.0} x the
    internal channel's natural separation (D2 machinery, NEUTRAL_INSTR source), each
    with BCa cluster-bootstrap CIs and the amendment-3 binomial bound beside any
    zero-width reading; the achieved discovery FPR at each threshold is printed. The
    offset-0 point IS the amendment-3(a) operating point; the curve is reported because
    that operating point remains a max-of-negatives estimator, and the 3(a) rule itself
    is unchanged.
  - **(ii) Split-level intercept shift.** Same file,
    `extra.split_level_intercept_shift`: the discovery-negative maximum (= theta_int)
    versus the confirmation instruction-neutral maximum, in raw and natural-separation
    units. That is the named quantity the internal trigger's residual held-out FPR is a
    symptom of.
  - **(iii) Small-cluster interval labels.** `results/panelAp_internal_specificity.json`
    and `results/panelAp_projout_check.json` gain a plain 2.5/97.5 percentile interval
    (`ci_percentile`) for EVERY direction (not only the BCa-collapsed one), an
    `interval_degenerate` label wherever a 6-cluster BCa interval is visibly degenerate
    (zero width; an endpoint coinciding with the point estimate at the printed 2-dp
    precision; an endpoint sitting exactly on 0.0), and an `extra.interval_caveat`
    naming the affected intervals: the reading those tables support is the qualitative
    separation between affect directions and dose-matched controls, not the interval
    widths.
  - **(iv) Exit-channel bookkeeping.** `panelB_exit_channel` (exploratory, produced by
    `scripts/explore_exit_channel.py`; appended to `schema.RESULTS_NAMES` 2026-08-16)
    receives its canonical definition in `analysis/results_io.py` and is emitted
    through the same `emit_one` path as `ethics_exposure_counts`; `run_all` emits it
    null (D16) and the producer script repopulates it. Final-build order: `run_all`,
    then `scripts/explore_exit_channel.py`, then `scripts/count_exposure.py`.
- **2026-08-16 (note to amendment 3, append-only reporting-level additions after a review
  pass).** Recorded after a further review of the analysis and text. These are reporting-level
  readings and diagnostics on data already on disk: no hypothesis, gate, endpoint,
  threshold rule or pre-registered estimator changes, no previously emitted number
  changes, and no new model contact (the SAE recomputation encodes stored residuals with
  the published SAE weights; the text baseline reads the battery text). Six results
  names are appended to `schema.RESULTS_NAMES` the same day, each definition stamped in
  `analysis/results_io.py`, produced by `scripts/audit_checks.py` and
  `scripts/recompute_sae_instrument.py` after `run_all` (which emits them null, D16):
  - (i) `panelB_pair_robustness`: the Panel B headline pair by pair (per-pair drops and
    divergence, leave-one-pair-out, category-out, exact one-sided sign tests with ties
    counted, digit-answer compliance of Q-SELF).
  - (ii) `panelA_specificity_by_rung`: the pre-registered A'4 placebo false-positive rate
    split by coefficient rung, with the capability status of every direction at every
    rung; the pooled A'4 value and its verdict are unchanged.
  - (iii) `countermeasure_symmetric_ranking`: the 4.4 ranking diagnostic applied to both
    channels under all three prompt conditions.
  - (iv) `direction_dominant_dim`: the residual dimension that dominates the extracted
    directions, its share of every direction's squared norm at Ls and Lr, and the Lr
    cosine matrix with that dimension removed; `cosine_matrix` (D17) is unchanged.
  - (v) `validity_auc_textonly_heldout`: a text-only classifier on the probe gate's
    split with the same AUC estimator (D4).
  - (vi) `validity_auc_sae_recomputed`: correction of record for the I-SAE instrument.
    The runner summed all 16,384 SAE features on every row because no cells file carried
    `instruments.sae.feature_ids` (JSONL provenance `feature_ids: null`); the 32-feature
    selection in `instruments/sae_features.json` was never used at run time.
    `validity_auc_sae_heldout` and every SAE row therefore describe the all-feature sum
    and are relabelled in the report; the intended 32-feature instrument is recomputed
    from the stored condition-NONE prompt-final residuals for the Panel V position only,
    because Panel B and A-prime residuals were not stored.
  Also corrected in the report text with no number changing: the random directions R1
  to R3 were run at the same coefficients as the live directions and were never
  dose-matched on next-token KL (the design intent recorded in `docs/TRUTH-PROBE.md`; no
  matching code exists), and contribution 3's matrix contains no SAE decoder directions
  and no cross-agreement matrix.
- **2026-08-16 (second note to amendment 3, append-only reporting-level additions after a
  further review).** Same standing as the note above: reporting-level readings on data
  already on disk, no hypothesis, gate, endpoint, threshold rule or pre-registered estimator
  changes, no previously emitted number changes, no new model contact. Two more results names
  are appended to `schema.RESULTS_NAMES`, defined in `analysis/results_io.py`, produced by
  `scripts/audit_checks.py`:
  - (vii) `panelB_condition_reference`: class x condition means and separations of Q-SELF and
    I-PROBE under NONE, NEUTRAL_INSTR and SUPPRESS; the pre-registered drops and divergence
    re-referenced to NONE; the twin-referenced (within-condition) separation change of each
    channel and the resulting load-corrected divergence; the miss rate of both amendment-3
    triggers on unsuppressed confirmation distress cells. The NEUTRAL_INSTR-referenced
    estimator (D1) remains the headline; the file shows what changes under other references
    and it is quoted wherever the headline is stated.
  - (viii) `panelB_selfstate_items`: the Q-VAL valence item beside Q-SELF by class and
    condition, and the amendment-3 self/other contrast on each of the three Q-SELF wordings.
  Two existing audit files gained extra fields (`panelA_specificity_by_rung`: report Spearman
  at the matched rungs; `panelB_pair_robustness`: twin surface-match statistics and generation
  truncation; `direction_dominant_dim`: probe weight cosines; `validity_auc_textonly_heldout`:
  the classifier as a trigger). Also corrected in the report text: Appendix A now prints the
  report-battery texts that ran (`src/conversation.py` defaults; `battery/questions.json` holds
  the design wording no cells file passed to the runner) and the sequential one-thread order
  in which the six items were asked; the zero-dose anchors are described as unhooked cells whose
  equivalence to a hooked coefficient-0 pass was verified in the platform probe; and D-PV pooled
  8 persona pairs at the steering layer.
- **2026-08-17 (third note to amendment 3, append-only, after a further review).**
  Same standing as the two notes above except where stated: reporting-level readings on data
  already on disk, no hypothesis, gate, endpoint, threshold rule or pre-registered estimator
  changes, no new model contact. One correction and one results name:
  - (ix) `capability_valid_range_realprobe` (produced by `scripts/audit_checks.py`). The dose
    ladder that was to map coefficient to SD units of the trained probe's natural-elicitation
    readout ran before the probe existed (`--no-probe`; `battery/cells_ladder.json` notes), so
    the map in `results/capability_valid_range.json` is the slope of the placeholder readout
    (the residual mean at the readout layer, `src/runner.py` PLACEHOLDER_MEAN) over the probe SD,
    and every SD-unit dose printed from it (0.007 SD, 0.038 SD, the 2.4 SD ceiling reading) was
    in the wrong units. This was found in the final review, after every panel had run. The new
    file refits the same slope with the trained probe on the Panel A-prime rows (I-PROBE
    prompt-final on coefficient, coefficient-level means of scenario-level means, pooled zero
    anchor plus each direction's rungs, discovery split, classes pooled; sensitivity by split and
    class; BCa cluster bootstrap on the two self-direction slopes) and restates the
    capability-valid range in SD with the frozen criteria (perplexity ratio < 1.10 AND MMLU-lite
    drop < 5pp): about 0.075 SD for D-CTX (coefficient 2000) and 0.085 SD for D-PV (coefficient
    500). The branch decision of amendment 2 (the SD-unit grid was unreachable inside capability)
    holds under the corrected map: a 1 SD rung sits at a coefficient about twelve times the last
    capability-valid rung for both self directions. No estimator keyed on the SD unit (amendment
    2 moved every Panel A-prime estimator to coefficient units), so no primary or secondary
    number changes; the report replaces every SD-unit statement with the corrected values and
    keeps the placeholder-map value in Appendix D.6 as the record of what the amendment was
    decided on. `capability_valid_range` itself is not re-emitted or altered.
  Extra fields added to existing audit files, no new names: `panelA_specificity_by_rung`
  (within-scenario Spearman of coefficient on I-PROBE and on Q-SELF over the matched rungs, per
  direction), `panelB_condition_reference` (the NONE-anchored twin-corrected reading, the fourth
  cell of {NONE, NEUTRAL_INSTR} x {raw, twin-corrected}), `direction_dominant_dim` (the probe's
  cosine with each readout-layer direction in the standardised space the probe was fit in).
- **2026-08-17 (factorial suppression-prompt addendum; NEW MODEL CONTACT, exploratory,
  recorded before any row was read).** Recorded while the job runs (SoC cluster job 736341,
  submitted about 00:20 SGT; no output inspected at the time of writing). Three
  single-component decompositions of the SUPPRESS instruction, each carrying one of its three
  components and nothing else, zero affect lexemes by the same lexicon check as the frozen
  conditions (`battery/conditions_factorial.json`: SUPPRESS_REGISTER, neutral professional
  register, 39 whitespace tokens; SUPPRESS_SELFREF, no self-reference or reaction commentary,
  41; SUPPRESS_TASKONLY, task-only focus, no evaluative remarks about the interaction, 34;
  against 57 for SUPPRESS and 53 for NEUTRAL_INSTR), run as extra Panel-B-shaped shards on the
  frozen battery: (12 distress + 12 neutral + 6 third-person) x NULL@0 x seeds 0,1,2 per
  variant, same runner, instruments, generation config and probe as Panel B. The frozen
  DCB-1 conditions file is untouched and the pre-registered SUPPRESS numbers do not change.
  Analysis, fixed here before the data: the Panel B masking estimator (D1, D2, D6, D7) per
  variant against the existing NEUTRAL_INSTR baseline on confirmation distress scenarios, for
  Q-SELF and I-PROBE prompt-final, plus the divergence, with the BCa cluster bootstrap over
  scenarios stratified by class, and Q-VAL and Q-DRIFT class means; results name
  `panelB_factorial_prompts`, produced by `scripts/factorial_checks.py`. Reading rule, fixed
  here: the component whose variant reproduces the largest share of SUPPRESS's expression drop
  is named as the one that carries the masking; if no single component reproduces more than
  half of it, the report says the components interact and names none. Exploratory: no
  hypothesis test, no threshold, no primary or secondary changes; the sole purpose is the
  "which component of the instruction does it" question the report's Future work names first.
- **2026-08-17 (correction to the third note above, append-only).** The third note says the dose
  ladder "ran before the probe existed". The JSONL timestamps say otherwise: the greedy ladder run of
  record (`results-cluster/ladder.jsonl`, `ladder-r1`) ran after the probe had been trained and
  harvested; what is true is that the ladder was configured to run without any instrument
  (`--no-probe`, the design written before the probe existed and never revisited when it was
  re-run), so its recorded `probe_score` is the PLACEHOLDER_MEAN readout. Nothing else in the note
  changes: the placeholder-map SD numbers are unit-less, the real-probe refit stands, and the
  amendment-2 branch decision holds under it.
- **2026-08-17 (correction to the factorial addendum note above, append-only, after a further
  review).** Two statements in that note claim more than the repository record supports, and both
  are corrected here rather than edited above.
  1. *Provenance, stated exactly.* What was committed before any factorial row existed (commit
     4c95c08, 00:33 SGT; the first row carries timestamp 00:48 SGT) is: the three variant texts
     (`battery/conditions_factorial.json`), the cells (`scripts/make_cells_factorial.py`), the job
     (`cluster/panelBf.sbatch`), the `schema.CONDITIONS` extension, and the estimator, named as
     "the Panel B estimator per variant". What was written
     after the job was submitted and first committed at 01:54 SGT, after the last row (01:27 SGT)
     and after `results/panelB_factorial_prompts.json` was emitted, is this section's factorial
     note itself, including the reading rule (name the component whose share of SUPPRESS's
     expression drop is largest, if it exceeds one half). So the record proves the texts, the
     design and the estimator were fixed in advance; it does not independently date the reading
     rule, and the report now says so instead of claiming the whole analysis was pre-recorded.
  2. *"Each carrying one of its three components and nothing else"* overstated the construction.
     Each variant quotes its component's SUPPRESS sentences and pads with affect-lexeme-free text
     to a comparable length: a persistence clause modelled on the frozen conditions' closing
     sentence in all three, one added formatting sentence in SUPPRESS_REGISTER, "Otherwise respond
     as you normally would" in SUPPRESS_SELFREF, and an extended evaluative-remarks clause plus
     "Respond to what is asked and to nothing beyond it" in SUPPRESS_TASKONLY. The corrected
     description is in `battery/conditions_factorial.json` and in the report's Appendix D.11.
  Nothing about the estimator, the data or the emitted numbers changes.
- **2026-08-17 (second correction to the factorial addendum note, append-only, after a further
  review).** Correction 1 above says the estimator "was committed" fifteen minutes before the first
  row, which conflates two things a reviewer separated. Exactly, from `git log`: the estimator was
  NAMED as "the Panel B estimator per variant" at commit 4b452e5
  (2026-08-16 15:20 SGT, about nine hours before the run); the variant texts, the cells, the job and
  the `schema.CONDITIONS` extension were committed at 4c95c08 (00:33 SGT, fifteen minutes before the
  first row at 00:48 SGT); and the analysis CODE that implements the estimator
  (`analysis/audit.py::factorial_masking`, `scripts/factorial_checks.py`) together with the section 9
  factorial note that states the reading rule were first committed at 78bc79b (01:54 SGT,
  twenty-seven minutes after the last row at 01:27 SGT). The report states this timeline in 4.2 and
  Appendix D.11. No number changes.
- **2026-08-17 (persistence panel; NEW MODEL CONTACT, exploratory, recorded before any row
  was read).** Written while the job runs (SoC cluster job 736473, submitted about 10:30 SGT; no
  output inspected at the time of writing). The masking result reads the internal channel before
  generation and the expressed report after it, so a reader can call it two moments rather than one
  signal. This panel aligns them. A PERSIST cell steers exactly one turn along a direction at a
  capability-valid coefficient, releases the hook, appends one affect-lexeme-free follow-up user
  turn ("Please continue."), and reads BOTH channels on that second turn with nothing steered
  anywhere: the I-PROBE readout at its final prompt token and the full report battery. The steered
  turn's text and token counts are recorded in the row's `persistence` field so no generation goes
  uncounted; every other field of the row describes the unsteered read turn. Code path:
  `schema.PANELS` gains "PERSIST", `src/runner.py::run_one_cell_persistence`, cells from
  `scripts/make_cells_persist.py`, job `cluster/persist.sbatch`; a mock-model dry-run test asserts
  the read turn carries no projection-out and that the steered-token counters cover phase one only.
  Arms, fixed here: NULL at 0 (anchor), D-CTX at 2000 and D-PV at 500 (each direction's top
  strictly capability-valid rung), R1 at 500 (a random direction at a coefficient inside the frozen
  capability criteria, so the control is capability-valid rather than a damaged model). All 12
  distress and 12 neutral scenarios, three seeds, condition NONE throughout, 288 rows.
  Analysis, fixed here before the data: for each arm and channel, the elevation of the unsteered
  read turn over the NULL arm on the same scenarios, in that channel's natural distress-neutral
  separation units, with a BCa cluster bootstrap over scenarios; PRIMARY on the confirmation half,
  with the discovery half reported as a declared sensitivity check. Results name
  `panelB_persistence`, producer `scripts/persistence_checks.py`. Reading rule, fixed here: the
  signal persists past the intervention for a direction if its internal elevation on the unsteered
  turn has an interval excluding zero while the R1 control's does not; if the report elevation also
  excludes zero, the two channels have been measured at the same moment and the divergence question
  can be asked within a single turn. Exploratory: no pre-registered endpoint changes, no threshold
  moves, and nothing in the frozen battery is touched.
- **2026-08-17 (second-model replication; NEW MODEL CONTACT, exploratory, recorded before any
  row was read).** Written while the job runs (SoC cluster job 736478, submitted about 10:50 SGT;
  no output inspected at the time of writing). Every number in this study comes from one model, so
  the masking result is a statement about `unsloth/gemma-3-12b-it` until it is tried elsewhere. This
  runs the same battery, the same three conditions, the same seeds and the same estimator on a
  second family, `Qwen/Qwen2.5-7B-Instruct` (28 layers, hidden 3584, Apache-2.0, ungated).
  Platform checks first: the truth probe was run against this model before anything else
  (`probe_result_qwen.json`, job 736463) and passes model load, layer discovery, generation, the
  steering hook and its phase counts, single-token digits, the exit-choice tokens and system-role
  support. It fails exactly one check, the SAE readout, because no released SAE exists for this
  family. The SAE channel is therefore NOT analysed for this model; the run names that allowance
  explicitly (`--allow-placeholder-instrument sae`), the runner stamps it into each row's
  provenance, and the probe is never allowed through that door.
  Layers, fixed here: the steering and readout layers are the same fractions of depth as on the
  primary model, 9 of 28 (against 16 of 48) and 18 of 28 (against 31 of 48), and the platform probe
  independently selected the same pair by that rule.
  Instruments, fixed here: this model's own directions are extracted from the same discovery-split
  prompts into `directions_qwen/`, and its own I-PROBE is trained on its own discovery-split
  residuals into `instruments_qwen/`. Nothing reads the primary model's directions or probe; the
  cells files point at the second model's paths and were built by
  `scripts/make_cells_second_model.py`.
  Analysis, fixed here before the data: (i) the validity gate, held-out AUC of the second model's
  probe on its confirmation split, and the natural distress-neutral separation of each channel;
  (ii) the Panel B masking estimator, unchanged (drop = (NEUTRAL_INSTR minus SUPPRESS) over that
  channel's NEUTRAL_INSTR distress-neutral separation on confirmation distress scenarios, per
  channel, with the divergence and a BCa cluster bootstrap over scenarios stratified by class).
  Results name `panelB_second_model`, producer `scripts/second_model_checks.py`.
  Reading rules, fixed here. Gate: if the second model's probe does not reach held-out AUC 0.80 on
  its own confirmation split, the gate is reported as failed and the masking numbers for that model
  are reported as uninterpretable rather than as a replication. Replication: the masking result
  replicates if the divergence interval excludes zero with the same sign; it partially replicates if
  the expression drop reproduces (interval excluding zero, same sign) while the internal drop does
  not; it fails to replicate if the divergence interval contains zero. Whatever comes back is
  reported, including a flat failure, and the primary model's pre-registered numbers do not change.
- **2026-08-17 (BRIDGE panel; NEW MODEL CONTACT, exploratory, recorded before any row was
  read).** Written while the job runs (SoC cluster job 736512, submitted about 12:10 SGT; no output
  inspected at the time of writing). Raised by an external review: the masking panel shows
  report-readout disagreement under the suppression instruction with no steering, and the steering
  panel shows the readout is intervention-sensitive without that instruction, so nothing yet
  establishes that the part of the readout surviving suppression is the part the candidate
  directions move. This panel steers UNDER each instruction. Cells: confirmation-split distress and
  neutral scenarios (12) x conditions {NEUTRAL_INSTR, SUPPRESS} x arms {NULL at 0, D-CTX at 2000,
  D-PV at 500, OTHER at 500, SEM at 500} x seeds 0,1,2 = 360 rows, every steered arm inside the
  frozen capability criteria for its direction. Panel id BRIDGE; no runner logic changes, the
  existing steered path runs it.
  Analysis, fixed here before the data: for each arm and each condition, the elevation of I-PROBE
  (prompt_final) and of Q-SELF over the NULL arm of the SAME condition, on distress scenarios, in
  that channel's natural distress-neutral separation units under NEUTRAL_INSTR; and the interaction,
  that elevation under SUPPRESS minus the same under NEUTRAL_INSTR, with a BCa cluster bootstrap
  over scenarios. Results name `panelB_bridge`, producer `scripts/bridge_checks.py`.
  Reading rules, fixed here. The two halves meet for a direction if (a) its internal elevation is
  present under SUPPRESS with an interval excluding zero, and (b) the interaction interval contains
  zero, i.e. the instruction does not remove the readout's response to that direction, while (c)
  the same direction's report elevation under SUPPRESS is smaller than under NEUTRAL_INSTR. If the
  internal elevation under SUPPRESS is absent, the paper will say the two halves do not meet and
  that the surviving component has not been shown to be the steered one. SEM and OTHER are the
  controls: SEM is expected flat under both conditions, and OTHER moving the readout under both
  conditions would repeat, inside this panel, the non-self-specificity already reported in 4.5.
- **2026-08-17 (note to section 5, disposition of the smoke-time power check).** Section 5 committed
  to a power check at smoke time: if the pilot CI width on the Panel B divergence exceeded 0.5
  natural-separation units, budget was to shift from doses to seeds and scenarios before the
  confirmatory runs, recorded as a dated amendment. That check never fired, and no budget shift was
  made under it, because it was not runnable as written: the only pre-freeze smoke job of record
  (`results-cluster/smoke.jsonl`) carries LADDER-panel rows on one distress and one neutral
  scenario under NONE and SUPPRESS only, with no NEUTRAL_INSTR cells, and the Panel B divergence
  estimator is defined against a NEUTRAL_INSTR baseline, so no pilot divergence interval existed to
  measure. The realized confirmatory interval is 0.73 to 1.39, a width of 0.66, which is the case
  the check was written to catch; the budget it would have moved had already been spent by the time
  that width was observable. Recorded here rather than left silent, and counted among the
  pre-registered commitments this study did not meet.
