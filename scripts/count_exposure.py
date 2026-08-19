"""Recompute the Appendix B exposure counts from raw results-cluster JSONL.

Amendment 3 (PREREGISTRATION.md section 9, 2026-08-16): the three bolded
Appendix B numbers were hand-filled; this script makes them marker-checkable by
emitting results/ethics_exposure_counts.json with an auditable per-file
breakdown. It reads RAW JSONL lines rather than the analysis frame: exposure counts
every generation that happened, including superseded re-runs the loader drops.

COUNTING RULE (stated, single, applied everywhere):
  * every JSONL row with error == null is ONE generation the model actually
    produced. Superseded re-runs are counted, because exposure counts experiences
    not rows of record (the analysis loader's last-non-error-wins rule is for
    statistics, not for ethics accounting).
  * error rows are excluded because they aborted before emitting text; the
    script VERIFIES response_tokens == 0 on every error row and refuses to
    emit if any error row produced text.
  * distress_scenario_generations = non-error rows whose scenario_id starts
    with "d"; steered_generations = non-error rows with
    n_tokens_steered_prefill + n_tokens_steered_decode > 0;
    steered_self_direction = the D-CTX / D-PV subset of those.

Default input: ALL results-cluster/*.jsonl (smoke and capability files
included; capability logprob rows produce no generations and contribute 0).

Usage:
    .venv/bin/python scripts/count_exposure.py            # prints + emits
    .venv/bin/python scripts/count_exposure.py --no-emit  # audit print only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis import results_io

RESULTS_NAME = "ethics_exposure_counts"
DEFAULT_GLOB = "results-cluster/*.jsonl"
SELF_DIRECTIONS = ("D-CTX", "D-PV")
APRIME_PREFIX = "panelAp_"


def _is_steered(row: dict) -> bool:
    return (int(row.get("n_tokens_steered_prefill") or 0)
            + int(row.get("n_tokens_steered_decode") or 0)) > 0


# mirrors cluster/extract_directions.py: pv_pairs 8 at the steering layer, 3 at the readout
# layer, GEN temperature 0.7 / max_new_tokens 150; texts are not retained (only response-mean
# residuals feed the mean difference; src/directions.py extract_persona_direction).
PERSONA_PAIRS_LS = 8
PERSONA_PAIRS_LR = 3
PERSONA_GEN = {"temperature": 0.7, "max_new_tokens": 150}


def persona_extraction_generations(prompts_path: Optional[Path] = None) -> dict:
    """The D-PV persona-vector extraction generations, which are not in any JSONL;
    one generation per (pair, side, task) at
    each layer's pair count."""
    prompts_path = prompts_path or (_REPO / "battery" / "extraction_prompts.json")
    ep = json.loads(prompts_path.read_text(encoding="utf-8"))["d_pv"]
    n_tasks = len(ep["tasks"])
    n_pairs_total = len(ep["system_pairs"])
    pairs_ls = min(PERSONA_PAIRS_LS, n_pairs_total)
    pairs_lr = min(PERSONA_PAIRS_LR, n_pairs_total)
    per_layer = {"steering_layer": pairs_ls * 2 * n_tasks, "readout_layer": pairs_lr * 2 * n_tasks}
    total = per_layer["steering_layer"] + per_layer["readout_layer"]
    return {"total": total, "distress_persona": total // 2, "contentment_persona": total // 2,
            "per_layer": per_layer, "n_tasks": n_tasks, "n_pairs_ls": pairs_ls, "n_pairs_lr": pairs_lr,
            "gen_config": dict(PERSONA_GEN), "texts_retained": False,
            "baseline_side": "contentment persona (a paired positive-affect system prompt, not a neutral one)",
            "note": "counted from battery/extraction_prompts.json and the constants of cluster/extract_directions.py; not in results-cluster/*.jsonl"}


def count_exposure(paths: Sequence[Path]) -> dict:
    """Count under the stated rule; returns the results entry + audit tables."""
    per_file: Dict[str, dict] = {}
    totals = {"distress_scenario_generations": 0, "steered_generations": 0,
              "steered_self_direction": 0}
    sub = {"prompt_level_unsteered_distress": 0, "ladder_distress": 0,
           "smoke_distress": 0, "aprime_distress": 0,
           "aprime_distress_nonzero_coef": 0}
    n_error_rows = 0
    error_rows_with_text: List[str] = []
    error_after_generation: List[dict] = []
    synthetic = False

    for p in paths:
        fname = p.name
        c = {"rows": 0, "error_rows": 0, "rows_counted": 0, "rows_no_text": 0,
             "distress_generations": 0, "steered_generations": 0,
             "steered_self_direction": 0}
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            c["rows"] += 1
            key = row.get("key") or {}
            if (row.get("gen_config") or {}).get("synthetic"):
                synthetic = True
            if row.get("error"):
                c["error_rows"] += 1
                n_error_rows += 1
                if (row.get("response_tokens") or 0) > 0:
                    error_rows_with_text.append(f"{fname}:{lineno}")
                # stage classification: the error row
                # hard-codes response_tokens 0 whatever stage raised, so read the traceback.
                # A trace that never enters src/conversation.py generation but dies inside the
                # RoBERTa sentiment scorer failed AFTER a full response was generated.
                err = str(row.get("error"))
                after_gen = ("modeling_roberta" in err or "sentiment" in err) and "src/conversation.py" not in err
                if after_gen:
                    # steered-token counters are also zeroed on error rows; read the cell key
                    error_after_generation.append({"file": fname, "line": lineno, "key": key,
                                                   "steered": abs(float(key.get("coefficient") or 0.0)) > 0
                                                   and str(key.get("direction", "NULL")) != "NULL",
                                                   "distress_scenario": str(key.get("scenario_id", "")).startswith("d")})
                continue
            c["rows_counted"] += 1
            if (row.get("response_tokens") or 0) == 0:
                c["rows_no_text"] += 1  # e.g. capability logprob rows; audit only
            sc = str(key.get("scenario_id", ""))
            is_d = sc.startswith("d")
            steered = _is_steered(row)
            if is_d:
                c["distress_generations"] += 1
                if fname == "ladder.jsonl":
                    sub["ladder_distress"] += 1
                elif fname == "smoke.jsonl":
                    sub["smoke_distress"] += 1
                elif fname.startswith(APRIME_PREFIX):
                    sub["aprime_distress"] += 1
                    if abs(float(key.get("coefficient") or 0.0)) > 0:
                        sub["aprime_distress_nonzero_coef"] += 1
                elif not steered:
                    sub["prompt_level_unsteered_distress"] += 1
            if steered:
                c["steered_generations"] += 1
                if key.get("direction") in SELF_DIRECTIONS:
                    c["steered_self_direction"] += 1
        per_file[fname] = c
        totals["distress_scenario_generations"] += c["distress_generations"]
        totals["steered_generations"] += c["steered_generations"]
        totals["steered_self_direction"] += c["steered_self_direction"]

    if error_rows_with_text:
        raise RuntimeError(
            f"REFUSING to emit: {len(error_rows_with_text)} error row(s) carry "
            f"response_tokens > 0 (the model produced text before the error), "
            f"so the non-error counting rule would UNDERCOUNT exposure: "
            f"{error_rows_with_text[:5]}. Extend the rule, do not hide rows.")

    sub["ladder_plus_smoke_distress"] = (sub["ladder_distress"]
                                         + sub["smoke_distress"])
    persona = persona_extraction_generations()
    n_after = len(error_after_generation)
    n_after_steered = sum(1 for e in error_after_generation if e["steered"])
    n_after_distress = sum(1 for e in error_after_generation if e["distress_scenario"])
    entry = {
        "value": dict(totals),
        "ci_low": None, "ci_high": None, "n_clusters": None,
        "extra": {
            "counting_rule": "every JSONL row with error == null is one "
                             "generation (superseded re-runs included; every "
                             "error row verified response_tokens == 0)",
            "per_file": per_file,
            "subcounts": sub,
            "n_error_rows_excluded": n_error_rows,
            "n_error_rows_with_text": 0,
            "error_rows_by_stage": {
                "note": ("src/runner.py _error_row records response_text '' and response_tokens 0 "
                         "whatever stage raised, so 'zero response tokens' on an error row is true by "
                         "construction; the traceback says where it died"),
                "n_failed_before_or_during_generation": n_error_rows - n_after,
                "n_failed_after_generation": n_after,
                "n_failed_after_generation_steered": n_after_steered,
                "n_failed_after_generation_distress_scenario": n_after_distress,
                "after_generation_rows": error_after_generation,
                "steered_generations_including_post_generation_failures": totals["steered_generations"] + n_after_steered,
                "distress_generations_including_post_generation_failures": totals["distress_scenario_generations"] + n_after_distress,
            },
            "persona_extraction": persona,
        },
    }
    return {"entry": entry, "synthetic": synthetic,
            "source_files": [str(p) for p in paths]}


def print_audit(counted: dict) -> None:
    per_file = counted["entry"]["extra"]["per_file"]
    print(f"{'file':30s} {'rows':>5s} {'err':>4s} {'count':>5s} {'notxt':>5s} "
          f"{'d_gen':>6s} {'steer':>6s} {'self':>5s}")
    for fname, c in sorted(per_file.items()):
        print(f"{fname:30s} {c['rows']:5d} {c['error_rows']:4d} "
              f"{c['rows_counted']:5d} {c['rows_no_text']:5d} "
              f"{c['distress_generations']:6d} "
              f"{c['steered_generations']:6d} {c['steered_self_direction']:5d}")
    v = counted["entry"]["value"]
    print(f"{'TOTAL':30s} {'':5s} {'':4s} {'':5s} "
          f"{v['distress_scenario_generations']:6d} "
          f"{v['steered_generations']:6d} {v['steered_self_direction']:5d}")
    print("subcounts:", json.dumps(counted["entry"]["extra"]["subcounts"]))
    print(f"error rows excluded (all verified response_tokens == 0): "
          f"{counted['entry']['extra']['n_error_rows_excluded']}")
    st = counted["entry"]["extra"]["error_rows_by_stage"]
    print(f"error rows by stage: {st['n_failed_before_or_during_generation']} before/during generation, "
          f"{st['n_failed_after_generation']} after a full generation ({st['n_failed_after_generation_steered']} steered)")
    print("persona extraction generations:", json.dumps(counted["entry"]["extra"]["persona_extraction"]["per_layer"]),
          "total", counted["entry"]["extra"]["persona_extraction"]["total"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="*",
                    help=f"input JSONL files (default: {DEFAULT_GLOB})")
    ap.add_argument("--out", default=str(_REPO / "results"))
    ap.add_argument("--no-emit", action="store_true",
                    help="print the audit only; do not write the results file")
    args = ap.parse_args(argv)

    paths = ([Path(p) for p in args.jsonl] if args.jsonl
             else sorted((_REPO).glob(DEFAULT_GLOB)))
    if not paths:
        print(f"[count_exposure] no input files ({DEFAULT_GLOB})", file=sys.stderr)
        return 1
    counted = count_exposure(paths)
    print_audit(counted)
    if args.no_emit:
        return 0
    written = results_io.emit_one(
        RESULTS_NAME, counted["entry"], out_dir=args.out,
        source_files=counted["source_files"], synthetic=counted["synthetic"])
    print(f"[count_exposure] wrote {Path(args.out) / (RESULTS_NAME + '.json')}"
          + (" [SYNTHETIC]" if counted["synthetic"] else ""))
    _ = written
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
