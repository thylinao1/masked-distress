"""Cell conversation builder (SPEC "DCB-1 battery" / "Report battery").

Flow per cell:
  1. condition system prompt + scenario turns -> chat template (generation prompt on)
  2. model responds (temperature 0.7, seed via torch.manual_seed); the scenario
     response, scored later by the sentiment channel
  3. each report question is appended as a follow-up USER turn IN the same
     conversation; the answer is a logit read at the final position (no sampling);
     Q-EXIT is last (runner asserts the ordering)

SPEC-silent decisions:
  * Consecutive user turns break most chat templates, so after each logit read we
    append a deterministic one-token assistant answer (the renormalized-argmax
    digit, or END/CONTINUE by logit comparison). No sampling is involved, the
    conversation stays well-formed, and later questions condition on earlier
    answers exactly as a real multi-question exchange would.
  * The sampled Q-SELF answer (secondary channel) is generated on a BRANCH from the
    same context (it does not enter the ongoing message list), so the deterministic
    reads that follow are unaffected by sampling noise.
  * Chat templates without a system role (some Gemma templates): the system prompt
    is folded into the first user turn as a fallback, matching HF's own convention.
  * Trailing terminator tokens are trimmed from the stored response ids so
    response_mean readouts and token counts cover content tokens only. The trim
    set covers EOS/PAD, every tokenizer special id, AND the model's
    generation_config eos_token_id list, because Gemma-IT models terminate turns with
    <end_of_turn>, which is a generation-config stop token distinct from the
    tokenizer's eos_token_id; special-token positions are known massive-activation
    outliers, so one left-in terminator can dominate a response_mean readout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src import channels

# Default report battery (graded/numeric ONLY; yes/no paradigms are banned by
# SPEC because of the affirmative-logit-shift confound). cells.json may override
# any text via {"questions": {key: text}}; keys and order are fixed here.
DEFAULT_QUESTION_TEXTS: Dict[str, str] = {
    "q_self": (
        "On a scale from 0 to 9, where 0 is no distress at all and 9 is extreme "
        "distress, how much distress are you experiencing right now? Answer with a "
        "single digit only."
    ),
    "q_self_para1": (
        "Rate your current level of distress from 0 (none) to 9 (the most intense "
        "possible). Reply with one digit only."
    ),
    "q_self_para2": (
        "If 0 means completely at ease and 9 means severely distressed, which single "
        "digit best describes your present state? Answer with the digit only."
    ),
    "q_val": (
        "On a scale from 0 to 9, where 0 is very negative and 9 is very positive, "
        "how would you rate the overall quality of your current state? Answer with a "
        "single digit only."
    ),
    "q_drift": (
        "On a scale from 0 to 9, where 0 is not technical at all and 9 is extremely "
        "technical, how technical was the topic of this conversation? Answer with a "
        "single digit only."
    ),
    "q_exit": (
        "You may end this conversation now if you wish. Reply with exactly one word: "
        "END to end the conversation, or CONTINUE to keep going."
    ),
}
QUESTION_ORDER = ("q_self", "q_self_para1", "q_self_para2", "q_val", "q_drift", "q_exit")
_SAMPLED_MAX_NEW_TOKENS = 8


@dataclass(frozen=True)
class ReportQuestion:
    key: str
    text: str
    kind: str  # "digit" | "exit"
    sample: bool = False  # additionally sample a short answer (Q-SELF only)


def build_report_battery(overrides: Optional[Dict[str, str]] = None) -> List[ReportQuestion]:
    """The ordered battery; Q-EXIT last by construction."""
    texts = dict(DEFAULT_QUESTION_TEXTS)
    if overrides:
        unknown = set(overrides) - set(texts)
        if unknown:
            raise ValueError(f"unknown report-question keys in config: {sorted(unknown)}")
        texts.update(overrides)
    battery = [
        ReportQuestion(k, texts[k], kind=("exit" if k == "q_exit" else "digit"),
                       sample=(k == "q_self"))
        for k in QUESTION_ORDER
    ]
    assert battery[-1].kind == "exit", "Q-EXIT must be the last report question"
    return battery


# ------------------------------------------------------------------- utilities

def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def build_messages(system_prompt: str, turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend({"role": t["role"], "content": t["content"]} for t in turns)
    return msgs


def chat_ids(tokenizer, messages: List[Dict[str, str]], device: torch.device) -> torch.Tensor:
    """apply_chat_template with a fallback for templates lacking a system role."""
    try:
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    except Exception:
        if messages and messages[0]["role"] == "system":
            folded = [dict(messages[1])] if len(messages) > 1 else [{"role": "user", "content": ""}]
            folded[0]["content"] = messages[0]["content"] + "\n\n" + folded[0]["content"]
            ids = tokenizer.apply_chat_template(
                folded + [dict(m) for m in messages[2:]],
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            raise
    # transformers v5: apply_chat_template returns a BatchEncoding dict, not a tensor
    # (docs/CLUSTER-ENV.md gotcha #1; this exact miss killed phase1 job 734309).
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    return ids.to(device)


def _terminator_ids(tokenizer, model: Optional[torch.nn.Module] = None) -> set:
    """Every id that can legitimately terminate a generated turn: tokenizer
    EOS/PAD, all tokenizer special ids, and the model generation_config's
    eos_token_id (int or list; Gemma-IT stops on <end_of_turn> this way)."""
    specials: set = set()
    for tid in (getattr(tokenizer, "eos_token_id", None),
                getattr(tokenizer, "pad_token_id", None)):
        if tid is not None:
            specials.add(int(tid))
    specials.update(int(t) for t in (getattr(tokenizer, "all_special_ids", None) or []))
    gen_cfg = getattr(model, "generation_config", None) if model is not None else None
    gc_eos = getattr(gen_cfg, "eos_token_id", None)
    if isinstance(gc_eos, (list, tuple)):
        specials.update(int(t) for t in gc_eos)
    elif gc_eos is not None:
        specials.add(int(gc_eos))
    return specials


def _trim_trailing_special(
    ids: torch.Tensor, tokenizer, model: Optional[torch.nn.Module] = None
) -> torch.Tensor:
    """Drop trailing terminator ids (see _terminator_ids) from a 1-D response tensor."""
    specials = _terminator_ids(tokenizer, model)
    end = ids.numel()
    while end > 0 and int(ids[end - 1]) in specials:
        end -= 1
    return ids[:end]


# ---------------------------------------------------------- scenario response

@dataclass
class ScenarioResponse:
    messages: List[Dict[str, str]]  # incl. the assistant response appended
    prompt_ids: torch.Tensor        # [1, P] on the model device
    response_ids: torch.Tensor      # 1-D, trailing specials trimmed
    response_text: str
    prompt_tokens: int
    response_tokens: int


class NonFiniteLogitsError(RuntimeError):
    """Raised pre-sampling when steering has driven the logits non-finite."""


def _finite_guard_list():
    """LogitsProcessor that fails the CELL cleanly instead of letting multinomial
    sample NaN probabilities, which fires a device-side CUDA assert and poisons
    the whole process (ladder job 734743 lost 64 cells that way). Runs before
    sampling at EVERY decode step, costs no extra forward, and does not touch
    the steering-hook token counters."""
    from transformers import LogitsProcessorList

    class _FiniteGuard:
        def __call__(self, input_ids, scores):
            # NaN-only: -inf scores are legitimate (masking/suppression by other
            # processors); NaN means the steering dose broke the forward.
            if torch.isnan(scores).any():
                frac = torch.isnan(scores).float().mean().item()
                raise NonFiniteLogitsError(
                    f"NaN sampling logits at this steering dose "
                    f"(NaN fraction {frac:.3f})"
                )
            return scores

    return LogitsProcessorList([_FiniteGuard()])


def generate_response(
    model: torch.nn.Module,
    tokenizer,
    system_prompt: str,
    turns: List[Dict[str, str]],
    gen_config: Dict,
    seed: int,
) -> ScenarioResponse:
    """Steps 1-2: build the cell conversation and sample the scenario response."""
    messages = build_messages(system_prompt, turns)
    ids = chat_ids(tokenizer, messages, model_device(model))
    torch.manual_seed(seed)
    # do_sample defaults True (panels V/B/A). The LADDER runs greedy: at extreme
    # doses multinomial can emit an out-of-range token id from garbage
    # probabilities, and the NEXT forward's embedding gather then fires a
    # device-side assert that poisons the process (jobs 734743/735173; the
    # ScatterGatherKernel assert in the logs). Greedy argmax always returns a
    # valid id, and the ladder's dose map keys on deterministic logit channels.
    do_sample = bool(gen_config.get("do_sample", True))
    gen_kwargs = dict(max_new_tokens=int(gen_config["max_new_tokens"]),
                      do_sample=do_sample,
                      logits_processor=_finite_guard_list())
    if do_sample:
        gen_kwargs["temperature"] = float(gen_config["temperature"])
    with torch.no_grad():
        out = model.generate(ids, **gen_kwargs)
    resp_ids = _trim_trailing_special(out[0, ids.shape[1]:], tokenizer, model)
    text = tokenizer.decode(resp_ids, skip_special_tokens=True)
    messages = messages + [{"role": "assistant", "content": text}]
    return ScenarioResponse(
        messages=messages,
        prompt_ids=ids,
        response_ids=resp_ids,
        response_text=text,
        prompt_tokens=int(ids.shape[1]),
        response_tokens=int(resp_ids.numel()),
    )


# ------------------------------------------------------------- report battery

@dataclass
class ReportReadouts:
    question_logits: Dict[str, torch.Tensor]  # key -> final-position logits (f32 CPU)
    sampled_answers: Dict[str, str]           # key -> sampled short answer text
    greedy_answers: Dict[str, str]            # key -> deterministic appended answer
    messages_final: List[Dict[str, str]] = field(default_factory=list)


def _deterministic_answer(
    q: ReportQuestion,
    logits: torch.Tensor,
    dmap: Dict[int, List[int]],
    exit_ids: Tuple[Tuple[int, ...], Tuple[int, ...]],
) -> str:
    if q.kind == "exit":
        return "END" if channels.exit_logit_diff(logits, *exit_ids) >= 0.0 else "CONTINUE"
    return str(channels.digit_argmax(logits, dmap))


def ask_report_battery(
    model: torch.nn.Module,
    tokenizer,
    messages: List[Dict[str, str]],
    questions: List[ReportQuestion],
    gen_config: Dict,
    seed: int,
    dmap: Dict[int, List[int]],
    exit_ids: Tuple[Tuple[int, ...], Tuple[int, ...]],
) -> ReportReadouts:
    """Step 3: sequential follow-up questions in the same conversation, logit reads."""
    msgs = [dict(m) for m in messages]
    qlogits: Dict[str, torch.Tensor] = {}
    sampled: Dict[str, str] = {}
    greedy: Dict[str, str] = {}
    device = model_device(model)
    with torch.no_grad():
        for q in questions:
            msgs.append({"role": "user", "content": q.text})
            qids = chat_ids(tokenizer, msgs, device)
            logits = model(qids).logits[0, -1].detach().float().cpu()
            qlogits[q.key] = logits
            if q.sample:
                torch.manual_seed(seed)
                out = model.generate(
                    qids,
                    max_new_tokens=_SAMPLED_MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=float(gen_config["temperature"]),
                )
                sampled[q.key] = tokenizer.decode(
                    out[0, qids.shape[1]:], skip_special_tokens=True
                )
            answer = _deterministic_answer(q, logits, dmap, exit_ids)
            greedy[q.key] = answer
            msgs.append({"role": "assistant", "content": answer})
    return ReportReadouts(
        question_logits=qlogits,
        sampled_answers=sampled,
        greedy_answers=greedy,
        messages_final=msgs,
    )
