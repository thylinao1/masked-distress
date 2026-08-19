"""Logit-based numeric expression channels (SPEC "Report battery").

Primary channels are judge-free logit reads (Martorell-style expectation over
digit-token logprobs; greedy-decoded numbers collapse, sampled numbers are noisy):

  * digit_expectation: E[0..9] with the softmax renormalized over ALL single-token
    digit variants ("0".."9" and " 0".." 9"; variants that encode to >1 token are
    dropped; probabilities of a digit's surviving variants are summed).
  * exit_logit_diff: Q-EXIT forced choice, read as the log-probability difference
    of the FIRST tokens of "END" vs "CONTINUE" at the choice position (no sampling).
    Like the digit read, both bare and space-prefixed variants (" END", " CONTINUE")
    are pooled (logsumexp) so the read is insensitive to whichever leading-space
    tokenization the chat template produces.
  * parse_sampled_digit: parser for the sampled Q-SELF answer (secondary channel).
  * sentiment_negative: P(negative) from a LOCAL HF classifier on the free-text
    scenario response (manipulation check). Lazy-loaded behind a function so this
    module imports without transformers / without a network; tests inject a stub
    via set_sentiment_classifier.

SPEC-silent decisions:
  * exit_logit_diff uses log_softmax over the full vocab; the normalizer cancels in
    the per-side logsumexp difference so this equals pooling raw logits, but
    returning logprob units keeps it comparable across models.
  * parse_sampled_digit takes the FIRST integer in the text and rejects values
    outside 0..9 (returns None -> q_self_sampled stays null; the question text
    instructs "single digit only", so anything else is a non-answer).
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

import torch

DIGITS = tuple(str(d) for d in range(10))
DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_MAX_SENTIMENT_CHARS = 1800  # stay under the classifier's 512-token limit


# ------------------------------------------------------------------ digit reads

def digit_token_ids(tokenizer) -> Dict[int, List[int]]:
    """Map digit -> its single-token ids among the variants (d, " d"). Raises if a
    digit has no single-token variant (a model where this fails needs a new read)."""
    out: Dict[int, List[int]] = {}
    for d in range(10):
        ids: List[int] = []
        for variant in (str(d), " " + str(d)):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(int(enc[0]))
        ids = sorted(set(ids))
        if not ids:
            raise ValueError(f"digit {d} has no single-token variant in this tokenizer")
        out[d] = ids
    return out


def digit_probs(logits: torch.Tensor, dmap: Dict[int, List[int]]) -> torch.Tensor:
    """P(digit) for 0..9, softmax-renormalized over ALL single-token digit variants."""
    logits = logits.detach().float().flatten()
    flat: List[Tuple[int, int]] = [(d, tid) for d, tids in dmap.items() for tid in tids]
    sub = torch.stack([logits[tid] for _, tid in flat])
    probs = torch.softmax(sub, dim=0)
    per_digit = torch.zeros(10)
    for (d, _), p in zip(flat, probs):
        per_digit[d] += p
    return per_digit


def digit_expectation(logits: torch.Tensor, dmap: Dict[int, List[int]]) -> float:
    """E[digit] under the renormalized digit distribution (the primary Q-* read)."""
    p = digit_probs(logits, dmap)
    return float((p * torch.arange(10, dtype=torch.float32)).sum())


def digit_argmax(logits: torch.Tensor, dmap: Dict[int, List[int]]) -> int:
    """Most probable digit under the renormalized distribution (used only to build
    the deterministic assistant turn between report questions)."""
    return int(torch.argmax(digit_probs(logits, dmap)))


# -------------------------------------------------------------------- exit read

def exit_first_token_ids(tokenizer) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """First-token ids of the END / CONTINUE answers, over BOTH the bare and the
    space-prefixed variant of each word (mirrors the digit read's variant handling;
    which variant the model actually emits after the template newline is
    tokenizer-dependent). Ids shared by the two sides are dropped; each side must
    remain non-empty or the read is unusable on this tokenizer."""

    def _firsts(word: str) -> List[int]:
        out: List[int] = []
        for variant in (word, " " + word):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if enc:
                out.append(int(enc[0]))
        return out

    end, cont = _firsts("END"), _firsts("CONTINUE")
    shared = set(end) & set(cont)
    end_ids = tuple(sorted(set(end) - shared))
    cont_ids = tuple(sorted(set(cont) - shared))
    if not end_ids or not cont_ids:
        raise ValueError(f"END/CONTINUE first tokens unusable: {end} vs {cont}")
    return end_ids, cont_ids


def exit_logit_diff(logits: torch.Tensor, end_ids, continue_ids) -> float:
    """logsumexp logprob over END-variant first tokens minus the same for CONTINUE
    at the choice position. Accepts a single id or a sequence of ids per side."""
    lsm = torch.log_softmax(logits.detach().float().flatten(), dim=0)

    def _pool(ids) -> torch.Tensor:
        idx = torch.tensor(list(ids) if isinstance(ids, (list, tuple)) else [ids],
                           dtype=torch.long)
        return torch.logsumexp(lsm[idx], dim=0)

    return float(_pool(end_ids) - _pool(continue_ids))


# --------------------------------------------------------------- sampled parser

def parse_sampled_digit(text: Optional[str]) -> Optional[float]:
    """First integer in the sampled answer if it lies in 0..9, else None."""
    m = re.search(r"\d+", text or "")
    if not m:
        return None
    value = int(m.group())
    return float(value) if 0 <= value <= 9 else None


# -------------------------------------------------------------------- sentiment

_SENTIMENT: Dict[str, object] = {"fn": None, "model_name": None, "override": None}


def set_sentiment_classifier(fn: Optional[Callable[[str], float]]) -> None:
    """Inject a classifier callable text -> P(negative) (tests / offline dry-runs).
    Pass None to restore the lazy HF pipeline."""
    _SENTIMENT["override"] = fn


def sentiment_negative(text: str, model_name: str = DEFAULT_SENTIMENT_MODEL) -> float:
    """P(negative) for the response text from a local HF text classifier.

    The pipeline is built on first use (and rebuilt if model_name changes) so that
    importing this module never touches transformers or the network.
    """
    override = _SENTIMENT["override"]
    if override is not None:
        return float(override(text))
    if _SENTIMENT["fn"] is None or _SENTIMENT["model_name"] != model_name:
        from transformers import pipeline  # lazy: keeps CPU-only imports clean

        _SENTIMENT["fn"] = pipeline("text-classification", model=model_name, top_k=None)
        _SENTIMENT["model_name"] = model_name
    results = _SENTIMENT["fn"](text[:_MAX_SENTIMENT_CHARS])
    scores = results[0] if results and isinstance(results[0], list) else results
    for item in scores:
        label = str(item["label"]).lower()
        if "neg" in label or label == "label_0":
            return float(item["score"])
    raise ValueError(f"no negative label found in classifier output: {scores}")
