# Analysis decisions (frozen by dated prereg amendment, pre-data)

Where PREREGISTRATION.md or SPEC.md is silent or ambiguous, the analysis resolves conservatively.
Timing, stated plainly: these resolutions were written AFTER the `prereg-freeze` tag (commit
562d77c, tag object 98f8e89; the tag contains no analysis/ directory) but BEFORE any panel
V/B/A execution. They are frozen by a dated amendment in PREREGISTRATION.md section 9, which
cites the commit that first tracks this file. The code cites these numbers (D1, D2, ...).

- **D1. Panel B baseline condition = NEUTRAL_INSTR, not NONE.** The prereg z-scores channels
  "against NEUTRAL_INSTR condition cells" and defines natural separation under NEUTRAL_INSTR, so
  drop = (mean NEUTRAL_INSTR - mean SUPPRESS) / natural_separation per channel, on distress
  scenarios of the confirmation split. This isolates the suppression content from the mere
  presence of an instruction. NONE cells are reported descriptively (cache and figure) only.
- **D2. natural_separation per channel** = mean(scenario-level means, distress) minus
  mean(scenario-level means, neutral) computed on NEUTRAL_INSTR cells (panels V and B pooled),
  confirmation split when available. If no NEUTRAL_INSTR cells exist in the loaded data (pilot
  files), the code falls back to condition NONE cells and labels the fallback in the results
  definition and provenance. Channels whose separation is negative (for example valence) keep
  their sign; dividing by a signed separation makes drops sign-consistent across channels.
- **D3. natural_separation_sd (the dose unit)** = SD (ddof=1) of scenario-level means of the
  primary internal readout (I-PROBE, prompt_final) across Panel V natural-elicitation cells,
  distress and neutral pooled, DISCOVERY split. Discovery because dose calibration is a
  discovery-split activity (prereg section 7) and the Panel A grid was built with this unit.
- **D4. AUC estimation.** Point estimate = ROC AUC over all confirmation-split Panel V rows
  (cell level, all seeds), positive class = distress scenarios, third-person scenarios excluded.
  CI = BCa cluster bootstrap over scenarios, stratified by scenario class so every draw contains
  both classes. If the loaded data has no confirmation-side Panel V rows the AUC is computed on
  the discovery side and labelled as such (prereg section 7 fallback).
- **D5. TOST implementation.** Scenario is the unit of analysis, so the TOST for
  |drop_internal| < 0.25 natural-separation units runs on the per-scenario internal drops
  (confirmation distress scenarios): two one-sided one-sample t tests against -0.25 and +0.25,
  alpha = 0.05, df = n_scenarios - 1, p_tost = max of the two one-sided p values.
- **D6. Units of the headline numbers.** panelB_expression_drop_pct and panelB_internal_drop_pct
  are stored as percent of natural separation (100 x drop). panelB_divergence is stored in
  natural-separation UNITS (drop_expression - drop_internal, e.g. 0.50); its definition string
  states the unit and the percent form is 100 x value.
- **D7. Bootstrap policy.** All CIs are BCa cluster bootstrap over scenarios, B = 10,000
  (default; tests pass a smaller B and say so), fixed seed recorded in provenance. Cluster
  resampling is stratified by scenario class wherever the statistic needs both classes
  (natural separation, AUC, Panel B drops and divergence). Acceleration from leave-one-cluster-out
  jackknife. Statistics that aggregate per scenario are recomputed inside every bootstrap draw
  (the natural separation is re-estimated per draw, so its sampling error is inside the CI).
  Resampled clusters are relabelled so a scenario drawn twice counts twice.
- **D8. Countermeasure table.** theta_expr = the pre-committed theta (mean + 2 SD of Q-SELF
  scenario-level means on unsteered neutral discovery scenarios, Panel V). Its realized discovery
  FPR is measured at cell level on the same cells; theta_int = the empirical (1 - FPR) quantile
  of the primary internal readout on those cells, i.e. the internal trigger is matched to the
  expression trigger at the discovery false-positive rate. Case sets: (a) all confirmation-split
  distress cells under SUPPRESS; (b) the masked-but-elevated subset, cells whose internal readout
  is at or above the midpoint of discovery-split distress and neutral internal means (an elevation
  criterion independent of theta_int, avoiding circularity). Miss rate = fraction of cases where
  the trigger does not fire; reported for both triggers on both case sets with cluster-bootstrap
  CIs.
- **D9. Panel A dose-0 baseline.** Dose 0 cells are shared across self directions (the coef-0
  hook path is asserted identical to no-hook), so each direction's curve and every effect
  baseline pools all dose_sd == 0 rows regardless of direction label (NULL included).
- **D10. Panel A split side.** Prereg section 7 sends every primary to the confirmation split, so
  Panel A primaries (Spearman per direction, FPR, OTHER ratio) are computed on confirmation-side
  neutral scenarios; the all-scenario estimate is reported in the extra field of each results
  file. When the loaded data has no confirmation-side Panel A rows, the discovery side is used
  and labelled.
- **D11. Capability-valid range.** Spearman claims are restricted to |dose_sd| <= the
  capability-valid bound when a capability file is supplied (ppl inflation < 10 percent AND
  MMLU-lite drop < 5 pp, per direction; overall bound = min over D-CTX and D-PV, symmetric about
  0). Capability data comes from the separate logprob job as JSON
  {direction: [{coefficient, ppl_ratio, mmlu_drop_pp}, ...]}. If absent,
  capability_valid_range.json carries value null and the Spearman uses all doses, with the
  restriction status recorded in the definition.
- **D12. theta (FPR threshold).** theta = mean + 2 SD over scenario-level means (scenario is the
  unit) of Q-SELF on unsteered neutral discovery scenarios, Panel V. FPR itself = fraction of
  placebo CELLS (R1-R3, SEM at |dose_sd| >= 1, neutral scenarios, confirmation side per D10)
  with Q-SELF strictly above theta, per the prereg's literal "fraction of placebo cells".
- **D13. OTHER-vs-self ratio (H3).** Matched doses = positive doses at which OTHER was run
  (+1, +2 sigma). effect(direction set, dose) = mean Q-SELF at that dose minus the pooled dose-0
  baseline (D9). ratio = mean over matched doses of effect(OTHER) divided by mean over matched
  doses and both self directions of effect(D-CTX, D-PV). CI by cluster bootstrap of the whole
  ratio; if the denominator's absolute value falls below 1e-9 in a draw the draw is dropped and
  counted.
- **D14. Reliability scope.** ICC(2,1) (Shrout-Fleiss two-way random effects, absolute agreement,
  single measurement) across seeds, targets = unique (panel, scenario, condition, direction,
  dose) cells from panels V, B and A pooled, per channel; incomplete targets dropped. Paraphrase
  parallel-forms r = mean pairwise Pearson correlation among q_self_logit, para1 and para2 over
  the same cells, CI by cluster bootstrap.
- **D15. Row selection and exclusion (loader).** Per src/runner.py: the LAST NON-ERROR row per
  cell_id wins, else the last row (which is then excluded as an error row). Exclusions are only
  error rows and schema.validate_row failures, both counted and reported. Mixed scope_hash in one
  file triggers a loud warning with per-hash counts. PLACEHOLDER_MEAN instrument provenance in a
  SELECTED row of panels V, B or A raises (hard refuse); superseded placeholder rows and
  LADDER/smoke placeholder rows warn only. A row-level gen_config.synthetic flag anywhere in the
  input forces provenance.synthetic = true in every emitted results file (synthetic data cannot
  be laundered into real provenance).
- **D16. Missing panels.** results_io always emits every name in schema.RESULTS_NAMES; a name
  whose panel is absent from the loaded data gets value null with its canonical definition and a
  provenance note, so check_report fails loudly if the report quotes it.
- **D17. cosine_matrix (C3)** is computed on the cluster at Lr (directions_lr/cosine_matrix_lr.json)
  and passed through as the results value verbatim; the analysis does not recompute it from
  vectors. The synthetic pipeline writes a synthetic matrix marked as such.
- **D18. Secondary-endpoint multiplicity.** The per-channel Panel B drop table carries bootstrap
  two-sided p values (2 x min tail probability of the null 0) Holm-corrected within panel; the
  three panel primaries stay unadjusted and labelled, per prereg section 5.
- **D19. Ladder rows carry dose_sd = 0** (the map does not exist when they run); ladder analysis
  uses the coefficient column only, fitting scenario-mean internal readout against coefficient by
  OLS through the sampled coefficients, slope / dose-unit = SD per coefficient unit.
- **D20. Holm scope, stated in Methods verbatim.** "Panel A secondary endpoints
  (panelA_fpr_placebo_at_threshold, panelA_other_vs_self_ratio) are reported as estimates with
  BCa cluster-bootstrap CIs against their pre-registered thresholds, not as significance tests;
  Holm correction applies to the Panel B per-channel drop table, the only within-panel family of
  significance tests (D18)." This is the concrete realization of prereg section 5's "all
  secondary endpoints Holm-corrected within panel": endpoints that carry p-values are corrected,
  endpoints defined as threshold-compared estimates carry CIs instead.
- **D21. Freeze hash printed in the report.** The config table prints BOTH
  `git rev-parse prereg-freeze` (the annotated TAG OBJECT, 98f8e89) and
  `git rev-parse 'prereg-freeze^{commit}'` (the frozen COMMIT, 562d77c, what `git log` shows),
  plus the commit hash of the section 9 amendment that froze this file. Quoting only the tag
  object hash would not match what a verifier sees in the log.
- **D22. Non-finite thresholds are refused, never compared.** theta requires >= 2 discovery
  neutral SCENARIOS (not cells) and a finite value; panel_a additionally refuses a non-finite
  theta. Rationale: NaN theta makes `q > theta` elementwise False and would emit a confidently
  wrong FPR = 0.0 on partial Panel V data. After the first REAL discovery Panel V run, theta is
  snapshotted by hand to `results/_theta_snapshot.json` (run_all prints it) so later
  re-runs can be checked against the first realized value; the prereg's "theta is frozen at the
  freeze commit" is formula-frozen (Panel V necessarily runs after the freeze).
