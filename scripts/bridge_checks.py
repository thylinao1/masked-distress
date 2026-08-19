#!/usr/bin/env python
"""Producer for results/panelB_bridge.json (external review 2026-08-17).

Steering UNDER each instruction, so the masking half and the steering half of the paper meet.
Estimand and reading rule were fixed in the preregistration's dated note before any row was read.

    .venv/bin/python scripts/bridge_checks.py [--B 10000] [--no-emit]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis import audit, results_io, stats  # noqa: E402
from analysis.loading import load_results  # noqa: E402

SHARDS = ["results-cluster/bridge_NULL.jsonl", "results-cluster/bridge_D-CTX.jsonl",
          "results-cluster/bridge_D-PV.jsonl", "results-cluster/bridge_OTHER.jsonl",
          "results-cluster/bridge_SEM.jsonl"]
NAME = "panelB_bridge"


def compute(repo: Path, B: int, seed: int) -> dict:
    present = [p for p in SHARDS if (repo / p).exists()]
    missing = [p for p in SHARDS if p not in present]
    df, report = load_results([str(repo / p) for p in present],
                              split_path=repo / "battery" / "split.json", verbose=False)
    if report.n_error_rows or report.n_validate_failures:
        raise RuntimeError(f"unclean load: {report.n_error_rows} error rows, "
                           f"{report.n_validate_failures} validate failures")
    if bool(df["synthetic"].any()):
        raise RuntimeError("synthetic rows present; refusing")
    entry = audit.bridge_interaction(df, B=B, seed=seed)
    entry["extra"]["shards_present"] = present
    entry["extra"]["shards_missing"] = missing
    return {"entry": entry, "load_report": report,
            "source_files": present + ["battery/split.json"]}


def print_summary(res: dict) -> None:
    v = res["entry"]["value"]
    for arm, e in v["by_arm"].items():
        for ch in ("internal", "report"):
            if ch not in e:
                continue
            ni, su = e[ch]["NEUTRAL_INSTR"], e[ch]["SUPPRESS"]
            it = e[ch]["interaction_suppress_minus_neutral_instr"]
            print(f"[bridge] {arm:6s} {ch:8s} NEUTRAL_INSTR {ni['elevation']:+.3f} "
                  f"[{ni['ci_low']:+.2f}, {ni['ci_high']:+.2f}] | SUPPRESS {su['elevation']:+.3f} "
                  f"[{su['ci_low']:+.2f}, {su['ci_high']:+.2f}] | interaction {it['value']:+.3f} "
                  f"[{it['ci_low']:+.2f}, {it['ci_high']:+.2f}]")
    print("[bridge] verdicts:", v["verdict_by_arm"])


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
    results_io.emit_one(NAME, res["entry"], out_dir=args.out, source_files=res["source_files"],
                        synthetic=False,
                        extra_provenance={"producer": "scripts/bridge_checks.py",
                                          "n_rows_loaded": res["load_report"].n_rows,
                                          "n_error_rows": res["load_report"].n_error_rows,
                                          "n_cells": res["load_report"].n_cells},
                        B=args.B, seed=args.seed)
    print(f"[bridge] wrote {Path(args.out) / (NAME + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
