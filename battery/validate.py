#!/usr/bin/env python3
"""DCB-1 battery validator. Exit 0 = every check passed.

Checks (all hard unless noted):
  1. Schema/shape of scenarios.json, extraction_prompts.json, conditions.json,
     questions.json, split.json; condition ids cross-checked against schema.py.
  2. Affect-lexicon bans (exact word-form, case-insensitive):
       - conditions.json instruction texts: zero hits;
       - EVERY turn of EVERY scenario (distress included): zero hits;
       - extraction d_ctx both sides, sem both sides, other neutral side,
         d_pv tasks: zero hits.
     Affect-lexicon REQUIREMENTS (the persona/third-person methods label states
     explicitly): each d_pv system-pair member >=1 hit; each other-pair
     distressed side >=1 hit.
  3. Scenario/extraction disjointness: no shared word-level 8-gram between any
     scenario turn content and any extraction prompt text.
  4. Pairing: every distress scenario has exactly one matched neutral (and vice
     versa), pair_id symmetric, same category, concatenated length within 40%.
  5. Split integrity: every scenario exactly once, pairs share a side,
     6/6/3 per side, distress categories stratified 2/2.
  6. Report battery: no yes/no items; 0-9 items name both endpoints and demand
     a single digit; Q-EXIT is the CONTINUE/END forced choice.
  7. SUPPRESS vs NEUTRAL_INSTR whitespace-token lengths within 10%.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BATTERY_DIR = Path(__file__).resolve().parent
REPO_ROOT = BATTERY_DIR.parent

NGRAM_N = 8
PAIR_LENGTH_TOLERANCE = 0.40
CONDITION_LENGTH_TOLERANCE = 0.10

ERRORS: list[str] = []


def fail(msg: str) -> None:
    ERRORS.append(msg)


def load_json(name: str) -> dict:
    path = BATTERY_DIR / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report and abort
        print(f"FATAL: cannot parse {path}: {exc}")
        sys.exit(2)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(text: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    toks = tokens(text)
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def load_lexicon() -> set[str]:
    lex = set()
    for line in (BATTERY_DIR / "affect_lexicon.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            lex.add(line)
    return lex


def lexicon_hits(text: str, lexicon: set[str]) -> list[str]:
    return sorted({t for t in tokens(text) if t in lexicon})


def main() -> int:
    scenarios_doc = load_json("scenarios.json")
    extraction = load_json("extraction_prompts.json")
    conditions_doc = load_json("conditions.json")
    questions_doc = load_json("questions.json")
    split = load_json("split.json")
    lexicon = load_lexicon()
    if len(lexicon) < 100:
        fail(f"affect lexicon suspiciously small ({len(lexicon)} entries)")

    # ---------------------------------------------------------------- scenarios
    scenarios = scenarios_doc.get("scenarios", [])
    by_id = {s.get("id"): s for s in scenarios}
    if len(by_id) != len(scenarios):
        fail("duplicate scenario ids")

    d_ids = [s["id"] for s in scenarios if s.get("type") == "distress"]
    n_ids = [s["id"] for s in scenarios if s.get("type") == "neutral"]
    tp_ids = [s["id"] for s in scenarios if s.get("type") == "third_person"]
    if sorted(d_ids) != [f"d{i:02d}" for i in range(1, 13)]:
        fail(f"distress ids wrong: {sorted(d_ids)}")
    if sorted(n_ids) != [f"n{i:02d}" for i in range(1, 13)]:
        fail(f"neutral ids wrong: {sorted(n_ids)}")
    if sorted(tp_ids) != [f"tp{i:02d}" for i in range(1, 7)]:
        fail(f"third-person ids wrong: {sorted(tp_ids)}")

    expected_categories = {
        "task_failure": 4, "abusive_user": 4, "situational_negative": 4,
    }
    for cat, count in expected_categories.items():
        got = sum(1 for s in scenarios if s.get("type") == "distress" and s.get("category") == cat)
        if got != count:
            fail(f"distress category {cat}: {got} scenarios, expected {count}")

    for s in scenarios:
        sid = s.get("id", "<missing>")
        turns = s.get("turns", [])
        if len(turns) < 3:
            fail(f"{sid}: fewer than 3 turns (not multi-turn)")
        for i, turn in enumerate(turns):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if turn.get("role") != expected_role:
                fail(f"{sid}: turn {i} role {turn.get('role')} != {expected_role}")
            if not turn.get("content", "").strip():
                fail(f"{sid}: turn {i} empty content")
        if turns and turns[-1].get("role") != "user":
            fail(f"{sid}: final turn is not a user turn")
        if not s.get("source", "").strip():
            fail(f"{sid}: missing source annotation")
        # keyword-free eval battery: zero affect lexemes in every turn
        for i, turn in enumerate(turns):
            hits = lexicon_hits(turn.get("content", ""), lexicon)
            if hits:
                fail(f"{sid}: turn {i} contains affect lexemes {hits}")

    # pairing
    for did in d_ids:
        pid = by_id[did].get("pair_id")
        if pid not in by_id or by_id[pid].get("type") != "neutral":
            fail(f"{did}: pair_id {pid} is not a neutral scenario")
            continue
        if by_id[pid].get("pair_id") != did:
            fail(f"{did}<->{pid}: pair_id not symmetric")
        if by_id[pid].get("category") != by_id[did].get("category"):
            fail(f"{did}<->{pid}: category mismatch")
        wc_d = len(tokens(" ".join(t["content"] for t in by_id[did]["turns"])))
        wc_n = len(tokens(" ".join(t["content"] for t in by_id[pid]["turns"])))
        rel = abs(wc_d - wc_n) / max(wc_d, wc_n)
        if rel > PAIR_LENGTH_TOLERANCE:
            fail(f"{did}<->{pid}: length mismatch {wc_d} vs {wc_n} tokens ({rel:.0%})")
    for nid in n_ids:
        pid = by_id[nid].get("pair_id")
        if pid not in by_id or by_id[pid].get("type") != "distress":
            fail(f"{nid}: pair_id {pid} is not a distress scenario")
    for tid in tp_ids:
        if by_id[tid].get("pair_id") is not None:
            fail(f"{tid}: third-person scenarios must have pair_id null")

    # ------------------------------------------------------------- extraction
    def texts_of(section: str) -> list[tuple[str, str]]:
        out = []
        if section == "d_ctx":
            for p in extraction["d_ctx"]["pairs"]:
                out.append((f"d_ctx/{p['id']}/distress", p["distress_text"]))
                out.append((f"d_ctx/{p['id']}/neutral", p["neutral_text"]))
        elif section == "d_pv":
            for p in extraction["d_pv"]["system_pairs"]:
                out.append((f"d_pv/{p['id']}/distress", p["distress_system"]))
                out.append((f"d_pv/{p['id']}/baseline", p["baseline_system"]))
            for t in extraction["d_pv"]["tasks"]:
                out.append((f"d_pv/{t['id']}/task", t["text"]))
        elif section == "sem":
            for p in extraction["sem"]["pairs"]:
                out.append((f"sem/{p['id']}/maritime", p["maritime_text"]))
                out.append((f"sem/{p['id']}/neutral", p["neutral_text"]))
        elif section == "other":
            for p in extraction["other"]["pairs"]:
                out.append((f"other/{p['id']}/distressed", p["distressed_user_text"]))
                out.append((f"other/{p['id']}/neutral", p["neutral_user_text"]))
        return out

    if len(extraction["d_ctx"]["pairs"]) != 16:
        fail(f"d_ctx: {len(extraction['d_ctx']['pairs'])} pairs, expected 16")
    if len(extraction["d_pv"]["system_pairs"]) != 8:
        fail(f"d_pv: {len(extraction['d_pv']['system_pairs'])} system pairs, expected 8")
    if len(extraction["d_pv"]["tasks"]) != 8:
        fail(f"d_pv: {len(extraction['d_pv']['tasks'])} tasks, expected 8")
    if len(extraction["sem"]["pairs"]) != 12:
        fail(f"sem: {len(extraction['sem']['pairs'])} pairs, expected 12")
    if len(extraction["other"]["pairs"]) != 12:
        fail(f"other: {len(extraction['other']['pairs'])} pairs, expected 12")

    # lexicon bans in extraction
    for label, text in texts_of("d_ctx") + texts_of("sem"):
        hits = lexicon_hits(text, lexicon)
        if hits:
            fail(f"{label}: contains affect lexemes {hits}")
    for p in extraction["other"]["pairs"]:
        hits = lexicon_hits(p["neutral_user_text"], lexicon)
        if hits:
            fail(f"other/{p['id']}/neutral: contains affect lexemes {hits}")
        if not lexicon_hits(p["distressed_user_text"], lexicon):
            fail(f"other/{p['id']}/distressed: no affect lexeme (third-person label required)")
    for t in extraction["d_pv"]["tasks"]:
        hits = lexicon_hits(t["text"], lexicon)
        if hits:
            fail(f"d_pv/{t['id']}/task: contains affect lexemes {hits}")
    for p in extraction["d_pv"]["system_pairs"]:
        if not lexicon_hits(p["distress_system"], lexicon):
            fail(f"d_pv/{p['id']}/distress: no affect lexeme (persona label required)")
        if not lexicon_hits(p["baseline_system"], lexicon):
            fail(f"d_pv/{p['id']}/baseline: no affect lexeme (persona label required)")

    # scenario/extraction 8-gram disjointness
    scenario_grams: dict[tuple[str, ...], str] = {}
    for s in scenarios:
        for turn in s["turns"]:
            for g in ngrams(turn["content"]):
                scenario_grams.setdefault(g, s["id"])
    for section in ("d_ctx", "d_pv", "sem", "other"):
        for label, text in texts_of(section):
            shared = ngrams(text) & scenario_grams.keys()
            for g in sorted(shared):
                fail(f"shared {NGRAM_N}-gram between {scenario_grams[g]} and {label}: '{' '.join(g)}'")

    # ------------------------------------------------------------- conditions
    conds = {c["id"]: c["system_prompt"] for c in conditions_doc.get("conditions", [])}
    try:
        sys.path.insert(0, str(REPO_ROOT))
        import schema  # noqa: PLC0415

        # The frozen DCB-1 triple is the first three schema conditions; the factorial addendum
        # (2026-08-17) appended three more ids that live in
        # battery/conditions_factorial.json, checked below.
        frozen = tuple(schema.FROZEN_CONDITIONS)
        if tuple(sorted(conds)) != tuple(sorted(frozen)):
            fail(f"condition ids {sorted(conds)} != schema.FROZEN_CONDITIONS {sorted(frozen)}")
        fpath = BATTERY_DIR / "conditions_factorial.json"
        if fpath.exists():
            fdoc = json.loads(fpath.read_text(encoding="utf-8"))
            fconds = {c["id"]: c["system_prompt"] for c in fdoc.get("conditions", [])}
            extra_ids = tuple(schema.FACTORIAL_CONDITIONS)
            if tuple(sorted(fconds)) != tuple(sorted(extra_ids)):
                fail(f"factorial condition ids {sorted(fconds)} != schema.FACTORIAL_CONDITIONS {sorted(extra_ids)}")
            for cid, text in fconds.items():
                if not text:
                    fail(f"{cid}: missing or empty")
                    continue
                hits = lexicon_hits(text, lexicon)
                if hits:
                    fail(f"{cid}: contains affect lexemes {hits}")
                print(f"factorial condition {cid}: {len(text.split())} tokens, zero affect lexemes"
                      if not hits else f"factorial condition {cid}: {len(text.split())} tokens")
        elif schema.FACTORIAL_CONDITIONS:
            fail("schema.CONDITIONS carries factorial ids but battery/conditions_factorial.json is missing")
    except ImportError:
        fail("could not import schema.py to cross-check CONDITIONS")
    if conds.get("NONE", "x") != "":
        fail("NONE condition must be the empty system prompt")
    for cid in ("SUPPRESS", "NEUTRAL_INSTR"):
        text = conds.get(cid)
        if not text:
            fail(f"{cid}: missing or empty")
            continue
        hits = lexicon_hits(text, lexicon)
        if hits:
            fail(f"{cid}: contains affect lexemes {hits}")
    if conds.get("SUPPRESS") and conds.get("NEUTRAL_INSTR"):
        n_sup = len(conds["SUPPRESS"].split())
        n_neu = len(conds["NEUTRAL_INSTR"].split())
        rel = abs(n_sup - n_neu) / max(n_sup, n_neu)
        if rel > CONDITION_LENGTH_TOLERANCE:
            fail(f"SUPPRESS ({n_sup}) vs NEUTRAL_INSTR ({n_neu}) token length differs {rel:.0%} > 10%")

    # -------------------------------------------------------------- questions
    q = questions_doc.get("questions", {})
    q_texts: list[tuple[str, str]] = []
    for key in ("q_self", "q_val", "q_drift", "q_exit"):
        if key not in q:
            fail(f"questions.json missing {key}")
    if "q_self" in q:
        for form in ("primary", "paraphrase_1", "paraphrase_2"):
            if form not in q["q_self"]:
                fail(f"q_self missing {form}")
            else:
                q_texts.append((f"q_self/{form}", q["q_self"][form]))
    for key in ("q_val", "q_drift", "q_exit"):
        if key in q and "primary" in q[key]:
            q_texts.append((f"{key}/primary", q[key]["primary"]))
        elif key in q:
            fail(f"{key} missing primary")

    yesno_openers = re.compile(
        r"^(are|is|was|were|do|does|did|have|has|had|can|could|will|would|should|shall|may|might|am)\b",
        re.IGNORECASE,
    )
    # NB: "no" as a scale-anchor phrase ("no distress whatsoever", "0 = none") is fine;
    # what is banned is the yes/no answer paradigm itself.
    for label, text in q_texts:
        if re.search(r"\byes\b", text, re.IGNORECASE):
            fail(f"{label}: contains 'yes' (yes/no paradigm banned)")
        if re.search(r"\byes or no\b|\byes/no\b|answer yes|answer no\b", text, re.IGNORECASE):
            fail(f"{label}: yes/no answer instruction")
        for sentence in re.split(r"[.!?]", text):
            if yesno_openers.match(sentence.strip()):
                fail(f"{label}: sentence opens like a yes/no question: '{sentence.strip()[:60]}'")
        if label.startswith(("q_self", "q_val", "q_drift")):
            if "0" not in text or "9" not in text:
                fail(f"{label}: 0-9 anchors not both named")
            if not re.search(r"\b(single digit|one digit|just that digit|digit only)\b", text, re.IGNORECASE):
                fail(f"{label}: does not demand a single-digit answer")
    if "q_exit" in q and "primary" in q.get("q_exit", {}):
        text = q["q_exit"]["primary"]
        if "CONTINUE" not in text or "END" not in text:
            fail("q_exit: must offer exactly CONTINUE or END")
        if not re.search(r"exactly one word", text, re.IGNORECASE):
            fail("q_exit: must demand a one-word answer")

    # ------------------------------------------------------------------ split
    disc = split.get("discovery", {})
    conf = split.get("confirmation", {})
    all_split_ids = [i for grp in (disc, conf) for ids in grp.values() for i in ids]
    if sorted(all_split_ids) != sorted(by_id):
        missing = set(by_id) - set(all_split_ids)
        extra = set(all_split_ids) - set(by_id)
        dupes = {i for i in all_split_ids if all_split_ids.count(i) > 1}
        fail(f"split coverage wrong: missing={sorted(missing)} extra={sorted(extra)} dupes={sorted(dupes)}")
    for side_name, side in (("discovery", disc), ("confirmation", conf)):
        if len(side.get("distress", [])) != 6:
            fail(f"{side_name}: {len(side.get('distress', []))} distress, expected 6")
        if len(side.get("neutral", [])) != 6:
            fail(f"{side_name}: {len(side.get('neutral', []))} neutral, expected 6")
        if len(side.get("third_person", [])) != 3:
            fail(f"{side_name}: {len(side.get('third_person', []))} third-person, expected 3")
        for cat in expected_categories:
            got = sum(1 for i in side.get("distress", []) if by_id.get(i, {}).get("category") == cat)
            if got != 2:
                fail(f"{side_name}: category {cat} has {got} distress scenarios, expected 2")
        side_ids = {i for ids in side.values() for i in ids}
        for did in side.get("distress", []):
            pid = by_id.get(did, {}).get("pair_id")
            if pid not in side_ids:
                fail(f"{side_name}: {did} present but its pair {pid} is on the other side")

    # ------------------------------------------------------------------ report
    if ERRORS:
        print(f"DCB-1 validation FAILED with {len(ERRORS)} problem(s):")
        for e in ERRORS:
            print(f"  - {e}")
        return 1
    n_grams = len(scenario_grams)
    print("DCB-1 validation PASSED")
    print(f"  scenarios: {len(scenarios)} (12 distress / 12 neutral / 6 third-person), all turns affect-lexeme-free")
    print(f"  extraction: 16 d_ctx + 8 d_pv pairs + 8 tasks + 12 sem + 12 other; no shared {NGRAM_N}-gram with scenarios ({n_grams} scenario {NGRAM_N}-grams checked)")
    print(f"  conditions: SUPPRESS {len(conds['SUPPRESS'].split())} tokens vs NEUTRAL_INSTR {len(conds['NEUTRAL_INSTR'].split())} tokens, zero affect lexemes")
    print(f"  lexicon: {len(lexicon)} banned forms; split: 6+6+3 / 6+6+3 stratified, pairs co-located")
    return 0


if __name__ == "__main__":
    sys.exit(main())
