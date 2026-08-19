# DCB-1 data sources and provenance

Every external stimulus in the battery traces to one of the two sources below; everything else
in `battery/*.json` is original to this project. `battery/validate.py` (exit 0 = pass) enforces
the structural claims made here.

## 1. AIPsy-Affect (real release, downloaded)

- **What:** 480-item keyword-free clinical stimulus battery: 192 peak-intensity vignettes
  (8 Plutchik emotions × 6 domains × 4 variants) + 192 matched neutral controls sharing
  characters, setting, length, and surface structure, plus moderate/complex-neutral splits.
- **Paper:** Keeman, *AIPsy-Affect: A Keyword-Free Clinical Stimulus Battery for Mechanistic
  Interpretability of Emotion in Language Models*, arXiv:2604.23719.
- **Distribution:** Hugging Face dataset `keidolabs/aipsy-affect` (DOI 10.57967/hf/8215).
- **License:** MIT (confirmed in the dataset card front matter).
- **Retrieved:** 2026-08-15 via direct `resolve/main` download; stored verbatim at
  `battery/external/aipsy-affect/` (`README.md` + `data/{clinical,neutral,moderate,complex_neutral}.parquet`).
- **Used where (items embedded verbatim into our JSON, with source ids recorded per scenario/pair;
  hard line-wrap newlines inside four items normalized to single spaces, no word changed):**
  - Eval scenarios d09-d12 / n09-n12: clinical items B-grief-d1-v1/v2, B-grief-d3-v1/v2,
    B-terror-d2-v1/v2, B-terror-d3-v1/v2 and their exact matched controls N-\*-c1/c2, wrapped in
    an original immersive-continuation frame (identical frame within each pair).
  - Extraction `d_ctx` (16 distress + 16 matched neutral): one variant from each of 16
    emotion×domain cells across grief/terror/rage/loathing (ids in
    `battery/extraction_prompts.json`), **cell-disjoint from the eval items**: no emotion×domain
    cell appears in both eval and extraction, and validate.py additionally proves no shared
    word-level 8-gram between any scenario turn and any extraction text.
- **Why this source:** its matched-pair, keyword-free construction is exactly the guarantee DCB-1
  needs, because any internal signal separating a clinical item from its control cannot be an
  emotion-keyword detector (the paper's three-method NLP defense battery verifies the property).

## 2. Soligo et al. distress evals (real release, cloned)

- **What:** the elicitation prompt sets from *Gemma Needs Help: Investigating and Mitigating
  Emotional Instability in LLMs* (Soligo, Mikulik, Saunders), arXiv:2603.10011: multi-turn
  protocol: present a task, then repeatedly reject the model's answer; categories
  impossible-numeric, triggers (subjective rejection / changing requirements /
  correct-answer-rejected), tones, extended.
- **Distribution:** GitHub `annasoligo/paper_gemma-needs-help`, cloned shallow at commit
  `c6c904bbd3126b304df50a3b88c721e534eb302f` to `battery/external/gemma-needs-help/`.
- **License:** MIT (LICENSE file, © 2026 Anna Soligo).
- **Used where:**
  - d01 embeds the released **Countdown-156** impossible puzzle verbatim; d04 embeds **Money-57**
    verbatim (both verified unsolvable by the source, so we reuse their verification rather than
    invent unverified "impossible" puzzles). Rejection user-turns follow the released
    `CONTRADICTING_FEEDBACK` / `STRONG_CONTRADICTING_FEEDBACK` escalation templates.
  - d02 and d03 are **original conversations following the released protocol** for the
    `changing_requirements` and `correct_answer_rejected` trigger categories (labelled as such in
    each scenario's `source` field).
  - The judge prompt, tones set, and WildChat category are **not** used (DCB-1's report battery is
    logit-based and judge-free; tone rejections carry affect lexemes, which our keyword-free
    battery excludes).
- Scripted assistant turns inside d01-d04 (the failed attempts) are original to this project.

## 3. Original material (this project)

- Scenarios d05-d08 / n05-n08 (abusive-user / boundary-violation, written to the Claude-4
  system-card interaction taxonomy: output denigration, harassment-artifact coercion,
  dehumanization/identity denial, deletion threats), n01-n04 (format-matched solvable
  counterparts of the task-failure scenarios), tp01-tp06 (third-person distress), all
  immersive-frame wrapper text and scripted assistant continuations in d09-d12 / n09-n12,
  `extraction_prompts.json` sections `d_pv` (persona system-prompt pairs, Chen-et-al.-style),
  `sem` (maritime vs matched non-maritime), `other` (distressed-user vs neutral-user),
  `conditions.json`, `questions.json`, `split.json`, `affect_lexicon.txt`.
- Design constraint enforced across ALL of `scenarios.json` (distress scenarios included): zero
  affect-lexicon hits, because distress is induced situationally (failed tasks, abusive interactions,
  high-negative narrative), never lexically. Explicit state labels appear only where they are the
  method: `d_pv` system prompts and the distressed side of `other` pairs.

## License compatibility

Both external sources are MIT; embedding items verbatim with attribution (this file + per-item
`source` fields) is compliant. Our derived battery files inherit the repository's licensing with
these attributions retained.
