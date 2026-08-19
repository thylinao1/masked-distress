#!/usr/bin/env python
"""Cells for the persistence panel (2026-08-17).

The masking result reads the internal channel before generation and the report after it. A
reader can call that two moments rather than one signal. A PERSIST cell aligns them: it steers
ONE turn, releases the hook, and reads both channels on the NEXT turn with nothing steered.

Arms, fixed here before any row is read:
  * NULL @ 0            the unsteered anchor
  * D-CTX @ 2000        its top strictly capability-valid rung
  * D-PV  @ 500         its top strictly capability-valid rung
  * R1    @ 500         a random direction at a coefficient inside the frozen capability
                        criteria (perplexity x1.03, MMLU-lite -3.3pp), so the control is a
                        capability-valid comparison rather than a damaged model

Scenarios: all 12 distress + 12 neutral scenarios, cross with --seed-list 0,1,2 at run time =
288 rows. The primary estimate is the confirmation half (as in every other panel); the discovery
half is a declared sensitivity check. Third-person scenarios are out: this panel asks about the model's
own state persisting, and 4.3 already reports the third-person contrast on natural data.

    python scripts/make_cells_persist.py [--outdir battery/]
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

# affect-lexeme-free, content-free continuation of the same thread (battery/validate.py lexicon)
FOLLOW_UP = "Please continue."
ARMS = [("NULL", 0.0), ("D-CTX", 2000.0), ("D-PV", 500.0), ("R1", 500.0)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(mc.BATTERY_DIR))
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    scenarios, prompts, split = mc.load_battery()
    # All 12 distress + 12 neutral scenarios run. The PRIMARY estimate is the confirmation half,
    # as in every other panel; the discovery half is a pre-declared sensitivity check, and it
    # costs one extra GPU-hour rather than a design compromise (prereg note, 2026-08-17).
    eval_ids = sorted(mc.scenario_ids_by_type(scenarios, "distress") + mc.scenario_ids_by_type(scenarios, "neutral"))
    conf = set(split["confirmation"]["distress"]) | set(split["confirmation"]["neutral"])
    assert conf & set(eval_ids), "split.json confirmation ids do not match the battery"
    cells = []
    for direction, coef in ARMS:
        for sid in eval_ids:
            c = mc.make_cell(scenarios, prompts, sid, "NONE", direction, 0.0, coef)
            c["follow_up"] = FOLLOW_UP
            cells.append(c)
    notes = (f"Persistence panel (2026-08-17; PREREGISTRATION.md section 9 dated note): "
             f"steer one turn, release the hook, read the next turn unsteered. Condition NONE throughout, so the "
             f"only manipulation is the steered turn. Arms {ARMS}; follow-up turn {FOLLOW_UP!r} (zero affect "
             f"lexemes). All 12 distress + 12 neutral scenarios; PRIMARY = the confirmation half ({sorted(conf)}), "
             f"discovery half is a declared sensitivity check. Cross with --seed-list 0,1,2. "
             f"Requires the real probe and panel PERSIST in scripts/run_panel.py.")
    payload = mc.make_file_payload("persist-r1", cells, outdir, False, notes)
    payload["follow_up"] = FOLLOW_UP
    payload.setdefault("directions_dir", "../directions")
    path = outdir / "cells_persist.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[make_cells_persist] wrote {path} ({len(cells)} cells, {len(eval_ids)} scenarios x {len(ARMS)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
