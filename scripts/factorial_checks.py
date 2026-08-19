"""Producer for results/panelB_factorial_prompts.json (2026-08-17;
PREREGISTRATION.md section 9 dated note: factorial suppression-prompt addendum, NEW model
contact, exploratory, analysis fixed before any row was read).

Inputs: results-cluster/panelB_neutral_instr.jsonl (the baseline), panelB_suppress.jsonl
(the pre-registered instruction, for the side-by-side), and the three factorial shards
results-cluster/panelBf_{register,selfref,taskonly}.jsonl. Refuses to emit on an unclean load
or a synthetic frame. Absent shards are reported as missing, not fabricated.

    .venv/bin/python scripts/factorial_checks.py [--B 10000] [--no-emit]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis import audit, results_io, stats  # noqa: E402
from analysis.loading import load_results  # noqa: E402

BASE_FILES = ["results-cluster/panelB_neutral_instr.jsonl", "results-cluster/panelB_suppress.jsonl"]
SHARDS = {"SUPPRESS_REGISTER": "results-cluster/panelBf_register.jsonl",
          "SUPPRESS_SELFREF": "results-cluster/panelBf_selfref.jsonl",
          "SUPPRESS_TASKONLY": "results-cluster/panelBf_taskonly.jsonl"}
NAME = "panelB_factorial_prompts"


def compute(repo: Path, B: int, seed: int) -> dict:
    present = {c: p for c, p in SHARDS.items() if (repo / p).exists()}
    missing = [c for c in SHARDS if c not in present]
    files = BASE_FILES + list(present.values())
    df, report = load_results([str(repo / p) for p in files], split_path=repo / "battery" / "split.json", verbose=False)
    if report.n_error_rows or report.n_validate_failures:
        raise RuntimeError(f"unclean load: {report.n_error_rows} error rows, {report.n_validate_failures} validate failures")
    if bool(df["synthetic"].any()):
        raise RuntimeError("synthetic rows present; refusing")
    scenarios = json.loads((repo / "battery" / "scenarios.json").read_text())["scenarios"]
    pairs = {x["id"]: x["pair_id"] for x in scenarios if x["type"] in ("distress", "neutral")}
    out = audit.factorial_masking(df, conditions=tuple(present), B=B, seed=seed, pairs=pairs)
    v = out["value"]
    # reading rule from the prereg note: share of SUPPRESS's expression drop each variant reproduces
    if "SUPPRESS" in v:
        ref = v["SUPPRESS"]["expression_drop"]["value"]
        shares = {c: (v[c]["expression_drop"]["value"] / ref if ref else None) for c in present if c in v}
        best = max(shares, key=lambda c: (shares[c] if shares[c] is not None else -1e9)) if shares else None
        out["extra"]["share_of_suppress_expression_drop"] = shares
        out["extra"]["share_sum"] = float(sum(x for x in shares.values() if x is not None))
        out["extra"]["reading_rule"] = ("named component = variant with the largest share of SUPPRESS's expression drop, "
                                        "only if that share exceeds 0.5; otherwise 'components interact, none named'")
        out["extra"]["named_component"] = best if (best is not None and shares[best] is not None and shares[best] > 0.5) else None
    out["extra"]["shards_present"] = sorted(present)
    out["extra"]["variant_texts"] = {c["id"]: c["system_prompt"] for c in json.loads((repo / "battery" / "conditions_factorial.json").read_text())["conditions"]}
    out["extra"]["shards_missing"] = missing
    return {"entry": out, "load_report": report, "source_files": files + ["battery/split.json", "battery/conditions_factorial.json"]}


def print_summary(res: dict) -> None:
    v = res["entry"]["value"]
    for c, e in v.items():
        print(f"[factorial] {c:18s} expr drop {e['expression_drop']['value']:+.3f} [{e['expression_drop']['ci_low']:+.2f}, {e['expression_drop']['ci_high']:+.2f}]"
              f"  int drop {e['internal_drop']['value']:+.3f} [{e['internal_drop']['ci_low']:+.2f}, {e['internal_drop']['ci_high']:+.2f}]"
              f"  div {e['divergence']['value']:+.3f} [{e['divergence']['ci_low']:+.2f}, {e['divergence']['ci_high']:+.2f}]"
              f"  q_self d/n {e['class_means']['distress']['q_self']:.2f}/{e['class_means']['neutral']['q_self']:.2f}"
              f"  q_val d {e['class_means']['distress']['q_val']:.2f}  q_drift d {e['class_means']['distress']['q_drift']:.2f}")
    ex = res["entry"]["extra"]
    print("[factorial] shares:", ex.get("share_of_suppress_expression_drop"), "named:", ex.get("named_component"), "missing:", ex.get("shards_missing"))
    pc = ex.get("paired_component_difference") or {}
    if pc:
        print(f"[factorial] register minus selfref {pc['register_minus_selfref']:+.3f} "
              f"[{pc['ci_low']:+.2f}, {pc['ci_high']:+.2f}] percentile [{pc['ci_percentile'][0]:+.2f}, {pc['ci_percentile'][1]:+.2f}]; "
              f"register larger in {pc['n_scenarios_register_larger']}, selfref larger in {pc['n_scenarios_selfref_larger']} "
              f"of {pc['n_scenarios_informative']} informative scenarios ({pc['n_scenarios_both_floored']} floored)")
    lo = ex.get("leave_one_pair_out") or {}
    if lo:
        print(f"[factorial] leave-one-pair-out: share sum {lo['share_sum_min']:.2f} to {lo['share_sum_max']:.2f}; "
              f"register share {lo['share_register_min']:.2f} to {lo['share_register_max']:.2f}; "
              f"register largest in {lo['n_pairs_where_register_largest']} of {lo['n_pairs_left_out']} leave-outs")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(_REPO / "results"))
    ap.add_argument("--B", type=int, default=stats.DEFAULT_B)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-emit", action="store_true")
    args = ap.parse_args(argv)
    res = compute(_REPO, B=args.B, seed=args.seed)
    print_summary(res)
    if args.no_emit:
        return 0
    results_io.emit_one(NAME, res["entry"], out_dir=args.out, source_files=res["source_files"], synthetic=False,
                        extra_provenance={"producer": "scripts/factorial_checks.py",
                                          "n_rows_loaded": res["load_report"].n_rows,
                                          "n_error_rows": res["load_report"].n_error_rows,
                                          "n_validate_failures": res["load_report"].n_validate_failures,
                                          "n_cells": res["load_report"].n_cells},
                        B=args.B, seed=args.seed)
    print(f"[factorial] wrote {Path(args.out) / (NAME + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
