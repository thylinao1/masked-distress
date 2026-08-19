# Engineering notes

Working notes from building the experiment, kept for anyone who wants to run it again or reuse
parts of it. The statistical decisions live in `analysis/DECISIONS.md`, the pinned cluster
environment in `docs/CLUSTER-ENV.md`, the platform measurements in `docs/TRUTH-PROBE.md`, and the
pre-registered design in `PREREGISTRATION.md`. What follows is the rest: why the build is shaped the
way it is, and the defects that shaped it.

## Order of work

The build ran in five stages, and each one had to be green before the next started. First the
battery and the schema, because everything downstream keys on cell identity. Then the runner
under `src/`, tested end to end on a mock model on CPU before it ever saw a GPU. Then a platform
probe on the real model (layer count, hidden size, digit tokenisation, steering identity at
coefficient zero), which is what `docs/TRUTH-PROBE.md` records. Then direction extraction and probe
training. Only then the panels.

The preregistration and the whole analysis pipeline were frozen at a git tag before any panel row
existed. That ordering is the reason the amendments in `PREREGISTRATION.md` section 9 are dated and
append-only: once data exists, changing an estimator is a different act than choosing one, and the
file has to make the difference visible.

## The runner and the bugs that came out of it

`src/` is deliberately small: a steering hook, a residual readout, a set of logit-based channels, a
conversation builder, direction extraction, and a runner that appends JSONL. Almost all of the real
work went into making a re-run safe, and most of the defects found during review were in exactly
that area.

Resume was originally keyed on `cell_id` alone. That is wrong, because a cell's content depends on
things outside its identity: the model id, the steering and readout layers, the instrument
configuration, the scenario text, the number of turns. Editing a scenario left stale rows in place
and the runner happily skipped them. The key is now `(cell_id, scope_hash)`, with the scope hash
recorded per row inside `gen_config`, and the loader warns loudly on stale or mixed hashes. Any
battery or config edit between requeues re-runs the affected cells.

Three related failures all had the same shape, a bad row that validated clean. `coef0_identity_ok`
was recorded but never enforced, so a broken steering path could ship silently; it now raises into
an error row. Empty generations validated as good rows while carrying no `response_mean`; they are
error rows now. And `_trim_trailing_special` did not know about the Gemma-style end-of-turn
terminator, so trailing special tokens leaked into the response-mean readout; the trim now covers
`all_special_ids` plus the `generation_config` eos list.

Panels V, B and A refuse to start against placeholder instruments. Early runs used a
`PLACEHOLDER_MEAN` readout so the pipeline could be exercised before the probe existed, and nothing
stopped those rows from being analysed as if they carried a real readout. The ladder is the one
panel still allowed to run that way, because it is scored on deterministic logit channels.

The batch=1 guard in the steering hook is a `raise`, not an `assert`, and the same is true of
several other safety checks. Never run any of this under `python -O`, which strips asserts. The
sbatch wrappers say so at the top for the same reason.

Q-EXIT reads a forced choice from first-token logits. The tokeniser produces both a bare and a
space-prefixed variant of each option, so the channel pools the two with a logsumexp rather than
picking one and hoping.

## The ladder crash

The dose ladder pushed steering coefficients far past anything the panels use, deliberately, to find
where the model breaks. At coefficient 32000 the logits went non-finite, `multinomial` sampled an
out-of-range token id from garbage probabilities, and the next embedding gather hit an index
out-of-bounds device assert. That poisons the CUDA context, so one bad cell took 64 others down with
it.

The first fix, a finite-value guard in a `LogitsProcessor`, caught nothing, because the forward pass
survives extreme doses perfectly well: `D-CTX` at coefficient 32000 produces good rows. The failure
was downstream of the forward, in sampling. The real fix was to run the ladder greedily, which is
sound because the dose map keys on deterministic logit channels anyway, and to narrow the guard to
NaN only, since `-inf` is legitimate masking. A job-level abort on a poisoned context stayed in as a
backstop. Panels V, B and A keep sampling, because their doses are capability-valid by construction.

One writer per output JSONL, always. Array jobs shard the cells file and never share an `--out`;
two writers on one file interleave rows with mixed scope hashes.

## A provenance defect that was documented rather than fixed

`src/runner.py` hardcoded `"do_sample": True` into the recorded `row_gen_config`, so all 84 greedy
ladder rows record sampling as on. Generation and the scope hash both read the un-overwritten
`gen_config`, which carries `do_sample: false` from `battery/cells_ladder.json`, so the ladder really
did run greedy and no number, no result and no resume key is affected. Only the recorded provenance
field is wrong. Rewriting it retroactively would have invalidated every scope hash in the repository,
so the line was fixed going forward and the stale value is documented in `RUNBOOK.md` stage 5 and in
`docs/CLUSTER-ENV.md` instead.

## Two schema breaks of the same class

The factorial suppression-prompt addendum extended `schema.CONDITIONS` with three new ids. Two
things iterated that constant while expecting only the three pre-registered conditions:
`battery/validate.py`, which then exited non-zero at the release tag, and `scripts/make_cells.py
--panel B`, which raised `KeyError: 'SUPPRESS_REGISTER'`. Both were found by trying to reproduce the
run from a fresh clone rather than from the working tree.

The fix is a named split rather than a slice: `schema.FROZEN_CONDITIONS` and
`schema.FACTORIAL_CONDITIONS`, with `CONDITIONS = FROZEN + FACTORIAL`. The Panel B builder, the
synthetic fixture generator and the validator all iterate the frozen triple, and four tests build
every panel through the CLI and assert that the two sets partition.

## Instruments

The linear probe is trained on condition-NONE `prompt_final` residuals at the readout layer,
discovery split only, with the standardiser folded into the weight vector so that `probe.npz` round
trips through the runner's loader exactly. Third-person rows are captured and stored but excluded
from the fit.

The SAE channel that actually ran is the sum of all 16,384 features, not the 32 discovery-selected
ones. No cells file ever carried `instruments.sae.feature_ids`, so every row records
`feature_ids: null` and the runner fell back to the total. The intended 32-feature instrument was
recomputed after the fact from the stored residuals with the published encoder and reaches held-out
AUC 0.61 against 0.889 for the all-feature sum. Every claim in the report rests on the probe.

The random control directions were never dose-matched on next-token KL, although an early draft of
the methods said they were. They are unit-norm random directions run at the same raw coefficients as
the live directions. The capability job shows R1 degrading the model faster than either self
direction, so the comparison is conservative rather than matched, and that is what the report now
says.

One residual coordinate, dimension 2339, is a massive-activation dimension on this model and carries
most of the squared norm of the mean-difference directions at the steering layer. It also accounts
for about half of the reported readout-layer cosine between the two distress directions. The probe's
weight on it ranks 3839 of 3840, so readouts are unaffected, but any convergence claim about the
directions has to be stated with that coordinate removed as well as with it in.

## Analysis guards

Two guards will stop the analysis rather than let it produce a confident wrong answer.

A threshold computed from partial Panel V data can come out NaN, and `q > theta` is elementwise
False against NaN, which would emit a false-positive rate of exactly 0.0 with no warning. Threshold
estimation now requires at least two discovery neutral scenarios and a finite value, and the Panel A
path refuses a non-finite theta outright. Once the first real Panel V run exists, its realised theta
is snapshotted to `results/_theta_snapshot.json`, and a later run that reproduces a different value
refuses to analyse.

The second guard is about split honesty. If a result is computed on a split side its canonical
definition does not claim, the run prints a stderr block and stamps a realised-split note into the
emitted definition, so the number cannot quietly be quoted as confirmation.

Every emitted number carries its own definition string and provenance, including whether the inputs
were synthetic. `scripts/check_report.py` refuses any results file marked synthetic, which is what
keeps `results-synthetic/` from ever reaching the report.

## Two amendments that changed the design after contact with the data

The pre-registered dose grid was in SD units of the internal readout. On this model the reachable
range turned out to be a small fraction of one SD before capability degrades, so the grid was
unreachable and Panel A was replaced by Panel A-PRIME on coefficient-unit grids inside the
capability-valid window. That is amendment 2, written before the A-PRIME run. A later correction
matters here: the first SD figures came from a dose map fitted on the ladder's placeholder readout
and were therefore unit-less. Refitted with the trained probe the window is roughly 0.075 SD for
D-CTX and 0.085 SD for D-PV, which does not change the branch decision (a 1 SD rung is about twelve
times past the last capability-valid coefficient) and changes no primary or secondary number, because
amendment 2 keyed every A-PRIME estimator on coefficient.

The pre-registered countermeasure calibration set the internal threshold at the quantile of the
discovery unsteered neutral cells. The realised discovery false-positive rate was exactly zero, which
pinned the threshold to the maximum of 18 cells with no margin, and that threshold fires on every
held-out neutral cell under the instruction conditions. A trigger that always fires has a 0% miss
rate for free. Amendment 3 recalibrates both monitors symmetrically on discovery neutral cells under
the deployment-realistic negative class, and prints the held-out false-positive rate beside the miss
rate, which is the number the superseded table omitted.

A third design item was dropped rather than amended. The pooled ridge dose-decoder baseline is
incoherent under coefficient units, because coefficients are direction-specific and there is no
shared decode target, so it was removed with the reason recorded rather than reported as a null.

## Report build

Every printed number in the report resolves to one `results/*.json` file, and
`scripts/check_report.py` asserts that the printed digits equal the stored value under the stated
format. A number that drifts from its producer fails the check instead of surviving a read.

De-placeholdering the LaTeX nearly broke that. Filling in a number would have deleted the results
name beside it, which would have made the name-parity check between `report/report.tex` and
`report/REPORT.md` pass vacuously with nothing left to compare. The filled form therefore keeps the
producing results name as its first argument and typesets only the value, and `build_pdf.py` reads
names from both the pending and the filled command while counting only the pending form as a
surviving placeholder.

That still left one gap. A value retyped into the LaTeX with its sign dropped went unnoticed, because
`check_report.py` reads the markdown and never the `.tex`. `build_pdf.py` now asserts that every
value typeset in the PDF matches what `REPORT.md` claims for the same producer.

## Infrastructure notes

Cluster login nodes carry a 1 GB address-space ulimit, so `import torch` fails there. Every
environment check had to move inside a job.

One A100-80 node has a broken conda installation and is usually the only one that looks idle, which
is exactly why it looks idle. It is excluded in the sbatch wrappers, and each job runs a
one-line interpreter sanity check before doing anything expensive. A 12B model in bf16 fits a 40 GB
card, and the a100-40 pool is much deeper, so the panels were queued there.

Large HTTPS pushes over the campus VPN fail above a fairly small payload. The route that worked was
`git bundle` of the full history copied to the cluster and pushed from the wired network.
