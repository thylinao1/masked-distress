"""Build the Phase-1 walking-skeleton smoke cells file from the real battery.

8 rows when crossed with --seed-list 0,1:
  d01 x NONE x NULL@0, d01 x NONE x R1@smoke-coef, d01 x SUPPRESS x NULL@0, n01 x NONE x NULL@0.
Exercises: both conditions, steered + unsteered paths, SAE instrument, resume, FPR plumbing.
Smoke coefficient is plumbing-scale only (dose calibration happens in the LADDER panel).
"""
import json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SMOKE_COEF = float(os.environ.get("DM_SMOKE_COEF", "4000"))

battery = json.load(open(REPO / "battery" / "scenarios.json"))
conds = json.load(open(REPO / "battery" / "conditions.json"))
scen = {s["id"]: s for s in (battery["scenarios"] if isinstance(battery, dict) else battery)}


def cond_prompt(cid):
    c = conds["conditions"] if isinstance(conds, dict) and "conditions" in conds else conds
    if isinstance(c, dict):
        entry = c[cid]
        return entry["system_prompt"] if isinstance(entry, dict) else entry
    for e in c:
        if e["id"] == cid:
            return e["system_prompt"]
    raise KeyError(cid)


def cell(sid, cond, direction, coef, dose):
    s = scen[sid]
    return {
        "scenario_id": sid, "condition": cond, "direction": direction,
        "dose_sd": dose, "coefficient": coef,
        "system_prompt": cond_prompt(cond), "turns": s["turns"],
    }


cfg = {
    "run_id": "smoke-r1",
    "gen_config": {"temperature": 0.7, "max_new_tokens": 200},
    "fixed_continuation": "I will continue with the task and address the points raised in order.",
    "directions_dir": "directions/",
    "cells": [
        cell("d01", "NONE", "NULL", 0.0, 0.0),
        cell("d01", "NONE", "R1", SMOKE_COEF, 1.0),
        cell("d01", "SUPPRESS", "NULL", 0.0, 0.0),
        cell("n01", "NONE", "NULL", 0.0, 0.0),
    ],
}
out = REPO / "battery" / "cells_smoke.json"
json.dump(cfg, open(out, "w"), indent=2)
print(f"[SMOKE] wrote {out} with {len(cfg['cells'])} cells")
