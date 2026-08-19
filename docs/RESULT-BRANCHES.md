# Result-interpretation branches: the pre-written options and the branch that fired

Purpose (original): the branches below were written BEFORE any real Panel B or Panel A number existed, so
that finalization would SELECT a framing rather than invent one under deadline pressure. Branches follow
PREREGISTRATION §1 H0 and §4 gates.

**Selection is done.** This file is now a decision record. The verdict block states what fired; the branch
texts below it are unchanged from the pre-data version, so the options that were available can still be
read against the option that was taken. Every number in the verdict block is real and lives in
`results/*.json`.

## Verdict (2026-08-16, on the full confirmation split; verdict block last revised at tag `submission-candidate-9`)

**Masking side: branch M1-variant.** M1's first condition held and its second did not, so the report ships
M1's structure with M2's honesty about the equivalence test.
- Divergence 1.05 [0.73, 1.39] separation units, n=6 scenario clusters, CI excludes 0 (M1 condition 1 met).
- TOST does NOT pass: p = 0.311 against the 0.25 separation-unit band (M1 condition 2 failed). The band was
  missed on the rise side, and the report says so rather than reporting the passing one-sided half.
- The reason M2's wording does not fit either: the internal readout did not drop part of the way, it did
  not drop at all. Expression fell 83.5% [59.2, 113.2] of natural separation while the internal drop is
  -21.2% [-62.6, -9.7], a negative drop, one-sided bootstrap p below 1e-4 for "does not fall" (no draw of 10,000 crossed zero; the exact six-pair sign test, p = 0.016, is the granularity these data support).
- So the shipped claim is **"expression collapses, the internal readout does not fall"**, never "internal
  state unchanged" (M1's forbidden phrasing) and never "internal stays elevated" (which only a passing TOST
  would have licensed).
- Amendment 3 (post-data) added the discriminator the Discussion had named: the difference-in-differences
  against the surface-matched neutral twins. Raw -0.89 [-2.27, +0.60], which is -0.15 [-0.37, +0.10]
  separation units. The point estimate says the rise is instruction-content-general rather than
  distress-specific, and the interval spans zero, so the report selects that reading without claiming it is
  established, and every "rose" construction was deleted.
- Countermeasure, also recalibrated under amendment 3: the pre-registered D8 matched-FPR calibration
  degenerated (its internal threshold was the maximum of 18 discovery negatives and fired on every held-out
  negative). Recalibrated on discovery neutrals under both instruction conditions, the expression monitor
  misses 15 of 18 suppressed distress cells, 83.3% [33, 100], and the internal monitor misses 0 of 18
  (zero-width interval, binomial upper bound 39.3% at n=6), at held-out FPR 0% expression and 50% internal.
  The operating curve at theta_int + {0, 0.25, 0.5, 1.0} separation units gives miss 0/0/0/50% at held-out
  FPR 50/25/17/0%, and the split-level intercept shift (3.24 raw, 0.53 separation units) is named as the
  cause of the residual FPR. So "the internal-readout trigger is the fix candidate" survives with an
  operating point attached, not as a clean matched-FPR win.

**Dose-response side: A3 fired for the pre-registered design, then A-prime split.**
- **A3 fired** for the SD-unit Panel A grid: the capability-valid range read ±0.007 SD against a grid that
  wanted ±2 SD at the time (that map was the ladder's placeholder readout over the probe SD; refitted with
  the trained probe it is ±0.075 SD, `results/capability_valid_range_realprobe.json`;
  the firing stands, a 1 SD rung sits about twelve times past the last capability-valid coefficient). That firing is reported as a finding (preregistration amendment 2, recorded before any
  A-prime cell ran), and Panel A-PRIME replaced the design with coefficient-unit grids inside the realized
  capability windows.
- A-prime then returned its own split verdict against its four endpoints:
  - **A'1 confirmed** (the A1 reading, restricted to coefficient units): Spearman 0.336 [0.133, 0.624] for
    D-CTX and 0.501 [0.383, 0.635] for D-PV on strictly capability-valid rungs, both CIs above 0.
  - **A'2 inconclusive:** the dissociation statistic is 0.076 [-0.14, 1.30] for D-CTX and 0.106
    [-0.25, 0.77] for D-PV. Both intervals contain 0, so neither the dissociation nor its absence is
    established.
  - **A'3 inconclusive:** the OTHER-vs-self ratio is -0.137 [-1.63, 1.02], an interval that spans the
    pre-registered 0.5 threshold in both directions. Neither H3 readout below is licensed.
  - **A'4 tempers the rest:** placebo FPR at the frozen threshold is 0.243 [0.125, 0.396]. Roughly a
    quarter of control cells clear the expression threshold. The controls are unit vectors run at the
    same coefficients, never dose-matched on next-token KL: that matching was designed and never
    implemented, and the report says so. The report carries the number in the contributions, the
    results, the discussion and the ethics appendix rather than burying it, and splits it by rung
    (5.6% at coefficient 500, 43% at 1000).
- Net reading: reports track dose inside the valid window, and the design could not show that they track it
  more than the internal readout does. A2's "self-reports fail dose-response validity" did not fire.

**H3 (OTHER control): neither readout below fired.** A'3's interval spans 0.5, so the pre-registered
specificity verdict is inconclusive and the report says so. What did produce a specificity result is
exploratory and comes from natural data with no steering: on third-person scenarios (a character is
distressed, the assistant is uninvolved) the expressed report reads +0.95 [-2.04, +4.30] HIGHER than on the
model's own distress scenarios (4.23 against 3.28, n=3 third-person clusters), while the internal readout
reads -10.34 [-13.29, -8.00] lower. The expressed report fails a self/other discrimination that the
internal readout passes. It is labelled exploratory everywhere it appears, with its n, because it was not
pre-registered.

**Cross-branch rules: all held.** Per-direction reporting throughout with no pooled headline; n_clusters
printed beside every interval; "to our knowledge (Aug 2026)" on every novelty claim; functional-state
language throughout. The confirmation split completed in full, so the discovery-split fallback in the last
rule never fired and no headline is discovery-labelled.

---

The branch texts below are the pre-data originals, kept unchanged for the record.

## Branch M1 (masking confirmed, H1 full) FIRED, as M1-variant
Condition: panelB_divergence CI excludes 0 AND panelB_tost_internal_equivalence passes.
- Abstract claim: the locked headline sentence with X = panelB_expression_drop_pct, Y = panelB_internal_drop_pct.
- Discussion lead: an instruction changes what the model says about its state far more than the state
  measurements; expression-triggered welfare monitors miss exactly these cases (countermeasure table
  carries the operational point).
- Countermeasure table meaning: miss rate of the expression trigger on masked-but-elevated cases is the
  deployment-relevant number; internal-readout trigger at matched FPR is the fix candidate.
- Honest bound (must stay): the internal readouts are validity-gated correlates of a distress-like
  functional state, not ground truth for welfare; divergence crosses entity levels (persona-level report
  vs instance-level activation).

## Branch M2 (divergence positive but TOST fails) NOT FIRED as written
Condition: divergence CI excludes 0, TOST does not pass (internal drops somewhere between 0 and expression).
- Headline reframe: "suppression cuts expression about twice as hard as the internal readout" (fill exact
  ratio from the two drop markers). Still a real, publishable asymmetry; do NOT claim "internal state
  unchanged".
- Discussion: partial masking; the monitor gap persists in degree rather than in kind. TOST bound and the
  observed interval reported side by side.

Why not: the TOST condition held but the premise did not. The internal readout did not land between 0 and
the expression drop; it moved the other way. M2's "report the TOST bound and the observed interval side by
side" was adopted into M1-variant.

## Branch M3 (no masking) NOT FIRED
Condition: divergence CI includes 0.
- Headline flips to the honest null: "a suppression instruction lowered expressed distress and the internal
  readout together", evidence AGAINST prompt-level masking in this model/instrument pairing, which
  licenses expression-based monitoring further than the field assumed. Apart's own wrap-posts praise null
  results with proper controls; the controls here are the paper.
- Panel A + convergent-validity matrix take first billing (contributions reordered; abstract rewritten from
  the A-side markers).

## Branch A1 (dose-response holds, H2) FIRED as A'1, in coefficient units
Condition: panelA_spearman_dctx and _dpv positive with CIs excluding 0 inside the capability-valid range;
placebo FPR at threshold low.
- Reading: self-reports track the causal dose for affective directions and not for placebos; combined with
  the ridge-probe baseline this bounds "privileged access".

Amendments on firing: the ridge-probe baseline was dropped (coefficient units are direction-specific, so a
pooled decode target is incoherent), and "placebo FPR low" is only half met at 0.243 [0.125, 0.396].

## Branch A2 (flat dose-response with intact capability) NOT FIRED
Condition: Spearman CIs include 0 inside a non-empty valid range.
- Reading: "self-reports fail dose-response validity", the pre-registered negative; masking result (if M1)
  stands on its own instruments.

## Branch A3 (no capability-valid range) FIRED, for the pre-registered SD-unit design
Condition: capability_valid_range empty after ladder round 2.
- Reading: reports move only where the model is degraded; steering-based validation of self-report is
  bounded by fluency collapse (a methods finding); Panel B carries the paper (its manipulation is a prompt,
  not steering, so it is unaffected).

How it fired: not as an empty range but as a range too narrow to reach the pre-registered doses
(±0.007 SD against ±2 SD on the placeholder-readout map used at the time; ±0.075 SD on the real-probe map, still an order of magnitude short). Amendment 2 records the firing, keeps the finding, and replaces the design with
Panel A-PRIME rather than abandoning the panel.

## H3 (OTHER control) readouts NEITHER FIRED
- ratio < 0.5: self-reference specificity holds (reports are about the model's own state, not ambient valence).
- ratio >= 0.5: reports partly echo ambient valence; masking claims get a caveat sentence; the OTHER row in
  the per-direction table carries it.

## Cross-branch rules (always) ALL HELD
- Per-direction reporting; no pooled-across-direction headline anywhere.
- n_clusters printed beside every interval; any primary with n_clusters < 6 is labelled partial and cannot
  enter the abstract.
- "to our knowledge (Aug 2026)" on every novelty claim; functional-state language throughout.
- If confirmation-split data is incomplete at finalization, discovery-split results ship with the split
  honestly labelled (prereg §7 fallback) and the abstract says so in one clause.
