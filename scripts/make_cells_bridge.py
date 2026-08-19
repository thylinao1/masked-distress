#!/usr/bin/env python
"""Cells for the BRIDGE panel (external review 2026-08-17).

The masking panel shows report-probe disagreement UNDER the suppression instruction, with no
steering. The steering panel shows the probe is intervention-sensitive WITHOUT the instruction.
Nothing so far joins them: the component of the probe that survives suppression has never been
shown to be the component the candidate directions move. This panel steers UNDER each instruction
and asks for the interaction.

Design, fixed before any row is read: confirmation-split distress and neutral scenarios (12) x
conditions {NEUTRAL_INSTR, SUPPRESS} x arms {NULL@0, D-CTX@2000, D-PV@500, OTHER@500, SEM@500}
x seeds 0,1,2 = 360 rows. Every steered arm sits at a coefficient inside the frozen capability
criteria for that direction (D-CTX 2000, everything else 500).

    python scripts/make_cells_bridge.py [--outdir battery/]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import make_cells as mc  # noqa: E402

ARMS = [("NULL", 0.0), ("D-CTX", 2000.0), ("D-PV", 500.0), ("OTHER", 500.0), ("SEM", 500.0)]
CONDITIONS = ["NEUTRAL_INSTR", "SUPPRESS"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(mc.BATTERY_DIR))
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    scenarios, prompts, split = mc.load_battery()
    conf = set(split["confirmation"]["distress"]) | set(split["confirmation"]["neutral"])
    eval_ids = sorted(s for s in (mc.scenario_ids_by_type(scenarios, "distress")
                                  + mc.scenario_ids_by_type(scenarios, "neutral")) if s in conf)
    written = []
    for direction, coef in ARMS:
        cells = [mc.make_cell(scenarios, prompts, sid, cond, direction, 0.0, coef)
                 for cond in CONDITIONS for sid in eval_ids]
        notes = (f"BRIDGE shard (external review 2026-08-17, PREREGISTRATION.md "
                 f"section 9 dated note): steering UNDER the instructions, arm {direction}@{coef:g}, "
                 f"conditions {CONDITIONS}, confirmation-split distress and neutral scenarios "
                 f"({len(eval_ids)}). Cross with --seed-list 0,1,2. Requires the real probe.")
        payload = mc.make_file_payload(f"bridge-{direction}-r1", cells, outdir, False, notes)
        path = outdir / f"cells_bridge_{direction}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append((path.name, len(cells)))
    print(f"[make_cells_bridge] wrote {len(written)} shard(s): " +
          ", ".join(f"{n} ({c} cells)" for n, c in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
