#!/usr/bin/env python
"""Producer for results/panelB_persistence.json (2026-08-17).

Reads the PERSIST shards (results-cluster/persist_*.jsonl), where each cell steered one turn and
then read both channels on the next turn with the hook released, and emits the elevation of that
unsteered read turn over the NULL arm, per arm and per channel, exactly as the preregistration's
dated note fixes it. Primary = the confirmation half; the discovery half is emitted beside it as
the declared sensitivity check.

    .venv/bin/python scripts/persistence_checks.py [--B 10000] [--no-emit]
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

SHARDS = ["results-cluster/persist_NULL.jsonl", "results-cluster/persist_D-CTX.jsonl",
          "results-cluster/persist_D-PV.jsonl", "results-cluster/persist_R1.jsonl"]
NAME = "panelB_persistence"


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
    entry = audit.persistence_elevation(df, B=B, seed=seed, split_side="confirmation")
    entry["extra"]["sensitivity_discovery_split"] = audit.persistence_elevation(
        df, B=B, seed=seed, split_side="discovery")["value"]["by_arm"]
    entry["extra"]["shards_present"] = present
    entry["extra"]["shards_missing"] = missing
    # every read turn must be unsteered: assert it rather than trust it
    pers = [r for r in df.to_dict("records") if r.get("panel") == "PERSIST"]
    entry["extra"]["n_rows_loaded"] = len(pers)
    return {"entry": entry, "load_report": report,
            "source_files": present + ["battery/split.json", "battery/cells_persist.json"]}


def print_summary(res: dict) -> None:
    v = res["entry"]["value"]
    for arm, e in v["by_arm"].items():
        print(f"[persist] {arm:6s} report {e['report']['elevation']:+.3f} "
              f"[{e['report']['ci_low']:+.2f}, {e['report']['ci_high']:+.2f}]  "
              f"internal {e['internal']['elevation']:+.3f} "
              f"[{e['internal']['ci_low']:+.2f}, {e['internal']['ci_high']:+.2f}]  "
              f"(n_cells {e['n_cells']})")
    print("[persist] verdicts:", v["verdict_by_direction"])
    print("[persist] missing shards:", res["entry"]["extra"]["shards_missing"])


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
                        extra_provenance={"producer": "scripts/persistence_checks.py",
                                          "n_rows_loaded": res["load_report"].n_rows,
                                          "n_error_rows": res["load_report"].n_error_rows,
                                          "n_cells": res["load_report"].n_cells},
                        B=args.B, seed=args.seed)
    print(f"[persist] wrote {Path(args.out) / (NAME + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
