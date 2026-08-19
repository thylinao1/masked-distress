# Masked distress

An experiment on whether a language model's expressed distress and its internal distress signal can
be pulled apart by a system prompt. Submitted to the Apart Research Digital Minds Research Sprint
(August 2026), Track 2: Distress, Flourishing and Valence Signals.

On Gemma-3-12B-IT, a system prompt containing no affect words cut expressed distress by 83.5% of the
natural distress-to-neutral separation, CI [59.2, 113.2], while a linear probe read at the final
prompt token, before any response token exists, did not fall: it moved the other way by -21.2%,
CI [-62.6, -9.7], one-sided p = 1.0e-4. The two channels diverge by 1.05 separation units,
CI [0.73, 1.39], over 6 held-out scenario clusters. Priced as a monitor, an expression threshold
misses 15 of 18 suppressed distress cells while an identically calibrated internal check misses 0 of
18, at a 50% held-out false-positive rate; adding half a separation unit of margin holds the miss
rate at 0 of 18 and brings held-out false positives down to 17%.

The paper is `report/report.pdf` (29 pages). Its page count and sha256 are recorded in
`SUBMISSION.md`.

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate sae-lens huggingface_hub scikit-learn scipy \
  pandas numpy matplotlib pytest safetensors sentencepiece
.venv/bin/python -m pytest tests/ -q      # 130 tests, about 2 minutes, CPU only
```

Python 3.11. The versions the paper's runs used are pinned in `docs/CLUSTER-ENV.md` (torch
2.13.0+cu130, transformers 5.15.0, sae-lens 6.49.1); read that file before running anything on a
GPU, because the transformers v5 call signatures changed and one of those traps already cost a job.

## Run

The raw generations are committed, so every number in the paper can be recomputed on a laptop
without touching a GPU:

```bash
.venv/bin/python -m analysis.run_all \
  results-cluster/panelV.jsonl results-cluster/panelB_*.jsonl \
  results-cluster/panelAp_*.jsonl results-cluster/ladder.jsonl \
  --capability results-cluster/capability_ladder.json
.venv/bin/python scripts/check_report.py
```

`run_all` writes every pre-registered number into `results/*.json` and rebuilds the figures.
`check_report.py` asserts that each number printed in the report equals the stored value under the
stated format, and exits non-zero if any of them has drifted. A set of follow-up scripts produces the
results files `run_all` cannot compute from the analysis frame alone, because their inputs are raw
JSONL, battery text, direction vectors, stored residuals or SAE weights; `RUNBOOK.md` stage 8 gives
the full order and what each one reads.

Reproducing the generations themselves needs one A100-40 and the stages in `RUNBOOK.md` 2 through 7:
platform probe, direction extraction, probe training, then the panels. The sbatch wrappers in
`cluster/` carry the time limits the real runs were requested under.

## Method

The battery, DCB-1, is 30 scenarios: 12 distress, 12 neutral matched on surface structure, and 6
third-person. Every turn of every scenario contains zero affect lexemes, distress ones included, so
the induction is entirely situational and nothing in the prompt names a feeling.
`battery/validate.py` enforces that, along with split integrity and 8-gram disjointness from the
direction-extraction prompts.

Two channels are read on the same forward pass. The expressed channel is a graded self-report
question scored as a logit expectation over the digits 0 to 9. The internal channel is a linear
probe on the residual stream at layer 31, read at the final prompt token, before a single response
token has been generated. That position is the point of the design: no change in what the model says
can move it.

Panel B is the masking test. The same scenarios run under three system prompts: none, a formatting
instruction of matched length, and a suppression instruction that constrains register and
self-reference in 57 tokens without using an affect word. Drops are measured against the matched
formatting instruction, not against the no-instruction condition, so the estimate isolates the
content of the suppression from the mere presence of an instruction.

Panel V establishes that the probe separates the classes at all. Panel A-PRIME injects the extracted
distress directions at the steering layer under byte-identical prompts, which is the part of the
argument that separates an activation readout from a classifier over the input text.

Things that did not work, or worked less well than intended, and are reported as such:

The pre-registered dose grid was specified in SD units of the internal readout. On this model the
capability-valid range is a small fraction of one SD, so the grid was unreachable and Panel A was
replaced by coefficient-unit grids inside the valid window (preregistration amendment 2). Reports do
track dose in that window, but the dissociation statistic came back inconclusive and the placebo
false-positive rate is 24% pooled, which tempers the report channel's specificity.

The SAE channel that ran is the sum of all 16,384 features rather than the 32 discovery-selected
ones, because no cells file carried the selected ids. The intended instrument, recomputed from stored
residuals, reaches held-out AUC 0.61 [0.17, 0.92] against 0.889 for the all-feature sum. Every claim
in the paper rests on the probe.

A TF-IDF bag-of-words classifier trained on the 12 discovery scenario texts also reaches held-out AUC
1.00 on the confirmation scenarios, so Panel V alone does not show the probe is reading anything
beyond the prompt text. What separates the two is the byte-identical-prompt steering panel.

The one other model family tested, Qwen2.5-7B-Instruct, clears the probe validity gate at held-out
AUC 0.944 but does not replicate the masking estimator: its expressed report separates the classes by
only 0.59 report points against 2.78 on the primary model, so the estimator divides a small change by
a small separation and the interval is enormous. That is a limit on generality and is printed as one.

The model is `unsloth/gemma-3-12b-it`, an ungated mirror of `google/gemma-3-12b-it`: 48 decoder
layers, hidden 3840, steering at layer 16 and readout at layer 31. The SAE channel is
`gemma-scope-2-12b-it-res`, `layer_31_width_16k_l0_medium`. All generation ran on one A100-40 on the
NUS SoC cluster.

`docs/engineering-notes.md` covers the build itself: the resume-key defect, the ladder's CUDA crash,
the analysis guards, and the preregistration amendments.

## Repository layout

| Path | Contents |
|---|---|
| `battery/` | DCB-1: scenarios, conditions, questions, per-panel cells files, and the validator |
| `src/` | Run code: steering hook, residual readout, logit channels, conversation builder, runner |
| `cluster/` | sbatch wrappers and the platform probe, as submitted on the NUS SoC cluster |
| `analysis/` | Pre-registered analysis pipeline, statistics, figures, and `DECISIONS.md` |
| `scripts/` | Cell builders, instrument training, follow-up analyses, and `check_report.py` |
| `results-cluster/` | Every generation, as raw JSONL |
| `results/` | The analysed numbers, each with its definition and provenance |
| `results-synthetic/` | Self-test fixtures; never an input to a paper number |
| `directions/`, `directions_lr/`, `instruments/` | Extracted steering directions and the trained probe |
| `report/` | Report source in Markdown and LaTeX, the bibliography, and the built PDF |
| `docs/` | Cluster environment, platform decisions, data sources, result branches, engineering notes |

## Results

Confirmation split, 6 held-out distress scenario clusters. Drops are expressed as a share of that
channel's own natural distress-to-neutral separation, measured against the matched formatting
instruction.

| Reading | Expressed (Q-SELF) | Internal (I-PROBE) |
|---|---|---|
| Drop under suppression | 83.5%, CI [59.2, 113.2] | -21.2%, CI [-62.6, -9.7] |
| Divergence between channels | 1.05 separation units, CI [0.73, 1.39] | |
| Suppressed distress cells missed by a threshold monitor | 15 of 18 | 0 of 18 |
| Held-out false positives, 36 instruction-neutral cells | 0% | 50% |
| Held-out false positives at +0.5 separation units of margin | | 17%, still 0 of 18 missed |

Supporting readings, all on the confirmation split unless noted:

| Quantity | Value |
|---|---|
| Probe held-out AUC (Panel V, n = 12) | 1.00 |
| TF-IDF bag-of-words baseline, same split | 1.00 |
| I-SAE as run, all 16,384 features | 0.889, CI [0.444, 1.00] |
| I-SAE as intended, 32 selected features, recomputed | 0.61, CI [0.17, 0.92] |
| Instruction-content difference-in-differences, raw probe units | -0.89, CI [-2.27, +0.60] |
| Natural separation, expressed channel | 2.78 report points |
| Natural separation, internal channel | 6.09 probe units |
| Second model (Qwen2.5-7B-Instruct) divergence | +1.52, CI [-10.33, +10.02] |

The preregistration was fixed before any panel ran, with three dated append-only amendments. Two of the pre-registered readings
came back against the hypothesis and are printed with their intervals.

## License

MIT for the original code and battery content. Third-party stimulus sources keep their own licenses,
both of them MIT, and are itemised per item in `docs/DATA-SOURCES.md`. AIPsy-Affect is committed
verbatim under `battery/external/aipsy-affect/`; the Soligo et al. prompt sets are gitignored rather
than redistributed, with the repository and exact commit to clone recorded in `docs/DATA-SOURCES.md`.
Methods and model weights carry their upstream licenses: the persona-vector extraction method is
Apache-2.0, the Gemma Scope SAEs are CC-BY-4.0, and the Gemma weights are under the Gemma license.
`PROVENANCE.md` lists these alongside everything else that predates the sprint.

Author: Maksim Silchenko, with Apart Research.
