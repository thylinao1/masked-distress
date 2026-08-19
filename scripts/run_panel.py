#!/usr/bin/env python
"""CLI entry point for the panel runner (SPEC "Compute + storage").

Example (cluster, inside an sbatch script):

    python scripts/run_panel.py \\
        --panel B --model-id unsloth/gemma-3-12b-it --ls 16 --lr 31 \\
        --sae-release gemma-scope-2-12b-it-resid_post --sae-id layer_31/width_16k \\
        --cells battery/cells_panelB.json --out ~/apart-welfare/results/panelB.jsonl \\
        --seed-list 0,1,2

Resume is automatic: (cell_id, scope_hash) pairs already present in --out are
skipped, so requeued jobs continue where they died; any config or battery edit
changes the scope_hash and re-runs the affected cells (a loud warning flags the
stale rows). Pass --retry-errors to re-run cells whose only row so far is an
error row.

Operational rules (violating either silently corrupts the results file):
  * ONE WRITER PER --out FILE. sbatch arrays must shard cells.json (one cells
    file + one --out per array task), never point two tasks at the same --out:
    concurrent writers can leave an error row as the last row for a cell that
    also has a good row. Analysis convention: last NON-ERROR row per cell_id,
    else last row.
  * NEVER run `python -O`: the layer-gap and Q-EXIT-ordering guards are assert
    statements and would be stripped.

Placeholder-instrument guard: panels V/B/A refuse to start when the probe/SAE
instruments are PLACEHOLDER_MEAN; pass --no-require-real-instruments only for an
explicit smoke run (sbatch templates pass nothing and inherit the safe default).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import runner  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Masked Distress panel runner")
    p.add_argument("--panel", required=True, choices=["V", "B", "LADDER", "A", "EXPANSION", "PERSIST", "BRIDGE"])
    p.add_argument("--allow-placeholder-instrument", action="append", default=None,
                   metavar="NAME", help="instrument that may stay PLACEHOLDER_MEAN on a real "
                                        "panel (e.g. sae for a model family with no released "
                                        "SAE); stamped into provenance; never the probe")
    p.add_argument("--model-id", required=True, help="HF model id to load")
    p.add_argument("--ls", type=int, required=True, help="steering layer index")
    p.add_argument("--lr", type=int, required=True, help="readout layer index")
    p.add_argument("--sae-release", default=None, help="SAELens release name (optional)")
    p.add_argument("--sae-id", default=None, help="SAELens sae_id within the release")
    p.add_argument("--cells", required=True, help="cells.json (format: src/runner.py docstring)")
    p.add_argument("--out", required=True, help="append-only results JSONL")
    p.add_argument("--seed-list", default="0,1,2", help="comma-separated generation seeds")
    p.add_argument("--run-id", default=None, help="override run_id (default: cells.json / <panel>-r1)")
    p.add_argument("--device", default=None, help="torch device (default: cuda:0 if available)")
    p.add_argument("--retry-errors", action="store_true",
                   help="re-run cells whose only existing row is an error row")
    p.add_argument("--require-real-instruments", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="refuse PLACEHOLDER_MEAN instruments at startup "
                        "(default: on for panels V/B/A, off for LADDER/EXPANSION; "
                        "--no-require-real-instruments allows an explicit smoke run)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    seeds = [int(s) for s in args.seed_list.split(",") if s.strip() != ""]
    if not seeds:
        print("[run_panel] --seed-list produced no seeds", file=sys.stderr)
        return 2
    cfg = runner.RunConfig(
        panel=args.panel,
        model_id=args.model_id,
        ls=args.ls,
        lr=args.lr,
        cells_path=args.cells,
        out_path=args.out,
        seeds=seeds,
        run_id=args.run_id,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
        retry_errors=args.retry_errors,
        require_real_instruments=args.require_real_instruments,
        allow_placeholder=tuple(args.allow_placeholder_instrument or ()),
    )
    stats = runner.run_panel(cfg)
    # Errors become rows, not crashes; exit 0 so sbatch arrays keep going. The
    # log line + error rows are the signal (SPEC: excluded and logged, never
    # silently dropped).
    print(f"[run_panel] {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
