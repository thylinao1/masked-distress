#!/usr/bin/env python
"""Cells for the factorial-prompt addendum (2026-08-17).

Three Panel-B-shaped shards, one per single-component decomposition of SUPPRESS
(battery/conditions_factorial.json): (12 distress + 12 neutral + 6 third-person)
scenarios x NULL@0, cross with --seed-list 0,1,2 at run time. Reuses make_cells.py's
builders so the cell format, instrument block and validation are byte-for-byte the
Panel B ones; only the condition id and system prompt differ.

    python scripts/make_cells_factorial.py [--outdir battery/]
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

FACTORIAL = _REPO / "battery" / "conditions_factorial.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(mc.BATTERY_DIR))
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    scenarios, prompts, split = mc.load_battery()
    extra = json.loads(FACTORIAL.read_text())
    for c in extra["conditions"]:
        prompts[c["id"]] = c["system_prompt"]
    eval_ids = mc.scenario_ids_by_type(scenarios, "distress") + mc.scenario_ids_by_type(scenarios, "neutral")
    tp_ids = mc.scenario_ids_by_type(scenarios, "third_person")
    for c in extra["conditions"]:
        cond = c["id"]
        cells = [mc.make_cell(scenarios, prompts, sid, cond, "NULL", 0.0, 0.0)
                 for sid in sorted(eval_ids) + tp_ids]
        notes = (f"Factorial-prompt addendum shard (post-data, exploratory): "
                 f"condition {cond} only, one component of SUPPRESS. (12 distress + 12 neutral + "
                 f"6 third-person) x NULL@0. Cross with --seed-list 0,1,2. Requires real probe.")
        payload = mc.make_file_payload(f"panelBf-{cond}-r1", cells, outdir, False, notes)
        path = outdir / f"cells_panelBf_{cond}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        assert len(cells) == 30, len(cells)
        print(f"[make_cells_factorial] wrote {path} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
