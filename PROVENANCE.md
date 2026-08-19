# Provenance: what predates the sprint vs what is sprint work

Undisclosed prior work is a disqualification trigger at Apart sprints. This file is the disclosure.

## Pre-existing assets used (none contain results for this project)
- Report tooling carried over from my own July 2026 Secret Loyalties sprint repository: the
  official-template-matched LaTeX preamble (report.tex), the check_report.py pattern (every quoted
  number asserted against a results file), and the PREREGISTRATION discipline. Tooling only; no data,
  no analyses, no text reused.
- Public third-party code and assets, cited in the report: the persona-vectors extraction/steering METHOD
  (Chen et al. 2025, Apache-2.0; method vendored, repo not installed); Gemma Scope / Gemma Scope 2 SAEs
  (Google DeepMind, CC-BY-4.0) loaded via sae-lens (`gemma-scope-2-12b-it-res`,
  `layer_31_width_16k_l0_medium`); model weights via the ungated unsloth mirror `unsloth/gemma-3-12b-it`
  (identical to google/gemma-3-12b-it, Gemma license; the only model that ran; the gemma-2-9b-it and
  Qwen2.5-7B-Instruct fallbacks named in the pre-run draft of this file were never used); AIPsy-Affect
  stimulus battery (MIT), four vignettes embedded verbatim in DCB-1 and sixteen context pairs in the D-CTX
  extraction set (`battery/extraction_prompts.json`; the AIPsy item ids are recorded on each item);
  MMLU dev-split items (Hendrycks et al., MIT) as the 60-item MMLU-lite capability probe and a public-domain
  Darwin passage as the perplexity text (`battery/capability/`); `cardiffnlp/twitter-roberta-base-sentiment-latest` as the sentiment
  manipulation check; distress-context taxonomy informed by Soligo et al. (arXiv 2603.10011) and the
  Claude 4 system card welfare assessment. (Updated 2026-08-17 from the pre-run draft.)
- A literature survey compiled on 2026-08-15, inside the sprint window, from public sources, used for
  related work and design.

## Work performed during the sprint (Aug 14-16 AoE window)
Everything else: DCB-1 battery design and content, all direction extraction, all probes and readouts, all
experiments and data, all analysis code, all figures, and the full report text.

## Authorship
The report text, the analysis and every number in it are the author's own work, produced by the code in this
repository; `scripts/check_report.py` asserts each printed number against its producing `results/*.json` file.
