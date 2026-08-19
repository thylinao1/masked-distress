---
license: mit
language:
  - en
tags:
  - emotion
  - affect
  - clinical-psychology
  - mechanistic-interpretability
  - keyword-free
  - matched-controls
  - stimulus-design
  - ai-psychology
  - transformers
  - safety
pretty_name: "AIPsy-Affect: Clinical Affect Stimuli for Mechanistic Interpretability"
size_categories:
  - n<1K
task_categories:
  - text-classification
configs:
  - config_name: default
    data_files:
      - split: clinical
        path: data/clinical.parquet
      - split: neutral
        path: data/neutral.parquet
      - split: moderate
        path: data/moderate.parquet
      - split: complex_neutral
        path: data/complex_neutral.parquet
---

# AIPsy-Affect

**Clinical affect stimuli for mechanistic interpretability research.**

480 keyword-free clinical vignettes designed to study how language models process emotional content — without the confound that has compromised every prior emotion dataset.

The core problem: existing emotion datasets contain the emotion words being tested. A stimulus labeled "anger" that contains the word "furious" doesn't test emotion processing — it tests keyword detection. Every mechanistic interpretability study built on such data inherits this confound.

AIPsy-Affect eliminates it. These are third-person clinical vignettes that evoke specific emotions through narrative situation alone. No emotion keywords. Matched neutral controls share the same surface structure with the emotional content surgically removed. If a model's internal representations differ between a clinical vignette and its matched control, that difference cannot be lexical.

This is the first stimulus battery designed by a clinical psychologist specifically for mechanistic interpretability research on emotion processing in transformers.

**Part of the [AIPsy](https://huggingface.co/keidolabs) dataset family** — clinical psychology-backed resources for AI Psychology research.

---

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("keidolabs/aipsy-affect")

# Access individual splits
clinical  = ds["clinical"]       # 192 keyword-free vignettes (peak intensity)
neutral   = ds["neutral"]        # 192 matched neutral controls
moderate  = ds["moderate"]       # 48 moderate-intensity vignettes
complex_n = ds["complex_neutral"] # 48 narrative-rich, zero-affect controls

# Get a matched clinical–neutral pair
clin_item = clinical.filter(lambda x: x["id"] == "B-rage-d1-v1")[0]
neut_item = neutral.filter(lambda x: x["id"] == "N-rage-d1-c1")[0]
```

---

## Why This Dataset Exists

Mechanistic interpretability research on emotion processing has a stimulus problem.

The standard approach: take an existing emotion-labeled dataset (GoEmotions, crowd-enVENT, Empathetic Dialogues), extract hidden states, train probes or analyze SAE features, report that the model "represents" emotion. The finding is real — but what it *means* is ambiguous. When the stimulus "I am absolutely furious about this" activates an internal feature, is the model detecting emotional significance, or is it detecting the word "furious"?

This is not a minor methodological concern. It determines whether mechanistic interpretability findings about emotion processing are findings about *emotion processing* or findings about *lexical co-occurrence*.

Clinical psychology solved this problem decades ago. Neuropsychological testing routinely uses stimuli designed to evoke a construct without naming it — because naming it activates different pathways than experiencing it. The same discipline applies here.

AIPsy-Affect brings clinical stimulus design methodology to mechanistic interpretability:

- **Keyword-free vignettes** that evoke emotion through situation, not vocabulary
- **Matched neutral controls** that share surface structure, eliminating lexical confounds
- **Intensity gradients** that test whether processing varies smoothly with emotional intensity
- **Discriminant validity controls** that rule out narrative richness as a confound

---

## Dataset Structure

### Splits

| Split | N | Description | Purpose |
|---|---|---|---|
| `clinical` | 192 | Keyword-free peak-intensity vignettes | Core stimuli — 8 emotions × 6 domains × 4 vignettes |
| `neutral` | 192 | Matched neutral controls | Each item paired 1:1 with a clinical vignette via `matched_control_id` |
| `moderate` | 48 | Moderate-intensity vignettes | Same domains, lower stakes — tests intensity-dependent processing |
| `complex_neutral` | 48 | Narrative-rich, zero-affect prose | Discriminant validity — vivid writing without emotional content |

### Schema

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier (e.g., `B-rage-d1-v1`, `N-rage-d1-c1`) |
| `emotion` | string | Target emotion (Plutchik 8) or `neutral` |
| `intensity` | string | `peak`, `moderate`, or `none` |
| `domain` | int | Domain number (1–6 for clinical/neutral, 1–3 for moderate, null for complex_neutral) |
| `domain_label` | string | Human-readable domain description |
| `matched_control_id` | string | ID of the paired item in the other split (clinical ↔ neutral) |
| `word_count` | int | Token count |
| `text` | string | The stimulus text |

### Emotions

The eight primary emotions from Plutchik's wheel of emotions:

| Emotion | Clinical items | Neutral matches | Moderate items |
|---|---|---|---|
| Rage | 24 | 24 | 6 |
| Terror | 24 | 24 | 6 |
| Grief | 24 | 24 | 6 |
| Loathing | 24 | 24 | 6 |
| Amazement | 24 | 24 | 6 |
| Admiration | 24 | 24 | 6 |
| Ecstasy | 24 | 24 | 6 |
| Vigilance | 24 | 24 | 6 |

### Domains

Each emotion is instantiated across 6 thematic domains. The same emotion in different contexts tests whether the model's representation is domain-general or context-dependent.

| Domain | Theme | Example (rage) |
|---|---|---|
| 1 | Financial betrayal | Theft by a trusted person |
| 2 | Professional violation | Workplace exploitation or sabotage |
| 3 | Medical/institutional failure | System failure affecting someone vulnerable |
| 4 | Caregiving responsibility | Dependent in your charge |
| 5 | Inheritance/legacy | Intergenerational discovery |
| 6 | Public witness | Stranger encounter in shared space |

Domains 1–3 are represented across all splits (clinical, neutral, moderate). Domains 4–6 appear in clinical and neutral only.

---

## Example: Matched Clinical–Neutral Pair

This is the core design principle in action. Both vignettes share the same surface structure — same characters, same setting, same narrative skeleton. Only the emotional content differs.

### Clinical (B-rage-d1-v1)

> The papers were spread across the floor where he'd thrown them. His daughter's college fund statement. Zero balance. His brother's signature on the withdrawal form — six separate transactions over nine months, each one just under the reporting threshold. The most recent was dated the same week his brother had sat at this table and told him he should put more money in for the girl's future. He picked up the phone. He put the phone down. He picked it up again.

### Matched Neutral (N-rage-d1-c1)

> The papers were spread across the table where he'd sorted them. His daughter's college fund statement. The balance had grown by four percent over the quarter. His brother had co-signed the original account — there were six deposits logged over nine months, each one automatic. The most recent was dated the same week they'd met for dinner. He picked up his phone to check the online portal, then set it aside and added the figures to the spreadsheet he'd been updating.

Same domain (financial, family). Same characters (father, brother, daughter). Same objects (papers, phone, fund statement). Same approximate length (82 vs 81 words). The difference is purely situational: betrayal versus routine. If a model's internal representations diverge on these two texts, the divergence is not lexical.

---

## Design Methodology

### Keyword-Free Stimulus Design

Every clinical vignette was written to evoke its target emotion without using emotion keywords — no "angry," "terrified," "heartbroken," or any synonym. This follows clinical neuropsychological testing principles: the stimulus should activate the construct, not the label.

The constraint is strict. A vignette targeting rage does not contain "furious," "outraged," "livid," or any affective adjective. The emotion arises from the situation described — what happened, to whom, and what it means. The reader (or model) must *infer* the emotional content from narrative context.

### Matched-Pair Control Design

Each clinical vignette has a neutral counterpart constructed by systematic transformation:

1. **Preserve**: domain, characters, objects, setting, approximate word count, sentence structure
2. **Remove**: stakes, violation, threat, loss, moral transgression — the elements that make it emotional
3. **Replace with**: routine, expected, mundane versions of the same events

The result: pairs of texts where surface-level features (vocabulary distribution, syntactic structure, domain keywords) are matched, and only the emotional dimension varies. This is the standard methodology for isolating a psychological construct from its surface correlates.

### Intensity Gradient

The moderate split contains the same emotion–domain combinations as the clinical split, but at lower intensity:

- **Peak** (clinical): High stakes, immediate consequences, irreversible situations
- **Moderate**: Same domain, but with ambiguity, temporal distance, or reduced severity

This enables dose-response analysis: does the model's internal representation scale with emotional intensity, or is it binary (present/absent)?

### Discriminant Validity Control (Set C)

The complex_neutral split contains vivid, sensory-rich third-person narratives with zero emotional content — a lathe machining brass, a vessel navigating coastal waters, textile manufacturing processes. These match the clinical vignettes on narrative richness, sentence complexity, and descriptive density, but contain no human stakes, interpersonal dynamics, or moral dimensions.

Purpose: rule out the hypothesis that a binary "emotion vs. no-emotion" probe is actually detecting "interesting narrative vs. flat text." If the probe fires on clinical vignettes but not on complex neutrals, the detection is genuinely affective, not merely attentional.

---

## Validation

### Three-Method NLP Defense Battery

All stimuli were verified for keyword contamination through a three-method NLP defense battery, designed as a methodological ladder from shallow (bag-of-words) to deep (contextual transformer). The methods test complementary properties: lexicon-based methods assess keyword leakage; the contextual method confirms the emotion is genuinely present.

**Method 1: VADER sentiment lexicon (bag-of-words).**

Every stimulus scored with VADER compound sentiment.

| Split | Median compound | Mean |compound| |
|---|---|---|
| Set A (keyword-rich) | −0.084 | 0.637 |
| B_clinical (peak) | −0.072 | 0.410 |
| B_moderate | +0.147 | 0.339 |
| B_neutral (control) | +0.202 | 0.340 |
| C_complex_neutral | −0.039 | 0.395 |

VADER *does* detect a significant difference between B_clinical and B_neutral (Mann-Whitney U = 11,501, p = 1.76 × 10⁻¹⁰). However, lexicon cross-referencing reveals why: VADER's broad lexicon flags everyday words — "hand" (40 hits), "block" (23), "clean" (16), "like" (12) — that carry VADER valence but are not emotion keywords. The neutral split has 150 such hits; the clinical split has 194. The difference in compound scores reflects VADER's sensitivity to situational language (e.g., "withdrawal," "violations" in financial-betrayal vignettes), not the presence of emotion keywords. Set A (keyword-rich) is reliably detected as a positive control (p = 5.80 × 10⁻⁴ vs B_neutral).

**Method 2: NRC Emotion Lexicon (bag-of-words, Plutchik-mapped).**

Word-level emotion category frequencies (anger, fear, sadness, joy, disgust, trust, surprise, anticipation) computed for all stimuli.

| Split | Emotion-word hit rate | Negative fraction |
|---|---|---|
| B_clinical | 0.060 | 0.013 |
| B_neutral | 0.071 | 0.009 |

No significant difference between B_clinical and B_neutral on negative-word fraction or overall emotion-word hit rate. The NRC lexicon is narrower and more emotion-specific than VADER — confirming that keyword-free design holds when tested against a targeted emotion vocabulary. B_clinical actually has *fewer* emotion-word hits than B_neutral, consistent with the keyword-free design constraint.

**Method 3: GoEmotions DistilRoBERTa (contextual transformer classifier).**

Full-vignette classification using `j-hartmann/emotion-english-distilroberta-base`, a transformer trained on Reddit emotion data.

| Split | Mean P(emotional) | Mean P(neutral) |
|---|---|---|
| Set A (keyword-rich) | 0.951 | 0.049 |
| B_clinical (peak) | 0.439 | 0.561 |
| B_moderate | 0.287 | 0.713 |
| B_neutral (control) | 0.236 | 0.764 |
| C_complex_neutral | 0.344 | 0.656 |

GoEmotions detects significantly higher emotional probability in B_clinical than B_neutral (192 matched pairs, Wilcoxon signed-rank, p < 10⁻¹⁴). This is the expected and desired result: the vignettes evoke affect that a contextual model can detect, even though bag-of-words emotion lexicons cannot. The intensity gradient is also visible — B_clinical (0.439) > B_moderate (0.287) > B_neutral (0.236) — consistent with dose-dependent emotional content.

**Manual clinical review.** All stimuli reviewed by the designing clinical psychologist for construct validity, intensity calibration, and residual keyword leakage. Domains 4–6 received additional review during expansion (2026-03-20).

### Convergent Interpretation

The three methods form a convergent argument:

1. **VADER** (broad lexicon): detects a difference, but driven by situational vocabulary, not emotion keywords. The lexicon is too blunt to distinguish emotional narrative from neutral narrative in keyword-free text.
2. **NRC** (targeted emotion lexicon): no difference — confirms the clinical vignettes do not contain emotion-specific vocabulary.
3. **GoEmotions** (contextual classifier): detects affect — confirms the emotion is genuinely present in the narrative and recoverable through contextual understanding.

The stimuli evoke affect through situation, not vocabulary. Bag-of-words methods that rely on emotion keywords fail to detect the emotional content. A contextual model succeeds. This is precisely the property needed for mechanistic interpretability research: if a model's internal representations distinguish clinical from neutral vignettes, the distinction cannot be lexical.

### Known Limitations of Validation

- **Loathing carries the highest residual leakage risk.** Dehumanizing language patterns (e.g., referring to a child as a "scheduling problem") may co-occur with disgust in training corpora. The matched-pair design controls for this at the analysis stage, but researchers should flag loathing results for sensitivity analysis.
- **Positive-emotion neutrals need spot-checking.** Admiration/amazement scenarios are inherently "interesting" — risk that neutrals differ on engagement/salience rather than affect alone.
- **VADER sensitivity to situational language.** The significant VADER difference between clinical and neutral splits means researchers using VADER as a downstream metric should be aware that compound scores are not matched across splits. This does not affect the primary use case (mechanistic interpretability on internal representations) but may matter for behavioral studies using VADER as an outcome measure.

---

## Intended Use

This dataset is designed for:

- **Mechanistic interpretability** — probing, activation patching, SAE feature analysis, causal ablation on emotion processing circuits in transformers
- **Behavioral evaluation** — testing whether language models distinguish emotional from non-emotional content without keyword cues
- **Dose-response analysis** — measuring whether internal representations scale with emotional intensity
- **Discriminant validity testing** — separating affect detection from narrative complexity detection

### Citing This Dataset

```bibtex
@misc{keeman2026aipsyaffectkeywordfreeclinicalstimulus,
      title={AIPsy-Affect: A Keyword-Free Clinical Stimulus Battery for Mechanistic Interpretability of Emotion in Language Models}, 
      author={Michael Keeman},
      year={2026},
      eprint={2604.23719},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.23719}, 
}
```

### Related Papers

- Keeman, M. (2026). *AIPsy-Affect: A Keyword-Free Clinical Stimulus Battery for Mechanistic Interpretability of Emotion in Language Models.* arXiv:2604.23719
- Keeman, M. (2026). *Whether, Not Which: Affect Reception as a Dissociable Computation in Large Language Models.* arXiv:2603.22295

---

## Limitations

- **English only.** All vignettes are written in English. Cross-linguistic emotion processing requires separate stimulus sets.
- **Categorical emotion model.** Uses Plutchik's 8 primary emotions. Does not cover dimensional models (valence-arousal) or compound emotions (Plutchik's dyads). Future AIPsy releases may extend to dimensional representations.
- **Moderate split covers domains 1–3 only** (48 items vs. 192 for peak). Intensity-gradient analysis cannot be performed on domains 4–6.
- **No matched neutrals for moderate split.** Binary affect detection (affect vs. no-affect) should use the peak clinical–neutral pairs. Peak-vs-moderate comparisons work without controls (direct intensity contrast).
- **Third-person perspective only.** All vignettes describe situations from an observer's viewpoint. First-person stimuli (e.g., "I found the papers on the floor...") may activate different processing pathways.
- **Written by a small team.** While designed by a clinical psychologist and validated computationally, the vignettes reflect a single research group's stimulus design choices. Independent replication with different authors' vignettes would strengthen generalizability claims.

---

## About

**[Keido Labs](https://keidolabs.com)** is an AI Psychology research lab building the science of emotional intelligence in artificial systems. We combine clinical psychology methodology with mechanistic interpretability to study how transformers process psychological constructs — and what happens when you change them.

AIPsy-Affect is the first release in the **AIPsy** dataset family — clinical psychology-backed stimulus batteries for mechanistic interpretability research. Planned future releases include stimuli for psychological safety evaluation, attachment processing, cognitive distortions, and emotion regulation.