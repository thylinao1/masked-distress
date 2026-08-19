#!/usr/bin/env python
"""Generate panel cells files (battery/cells_*.json) per SPEC.md "Panels".

CLI:
    python scripts/make_cells.py --panel V              [--outdir battery/] [--no-probe]
    python scripts/make_cells.py --panel B              [--outdir battery/]
    python scripts/make_cells.py --panel LADDER         [--outdir battery/] [--no-probe]
    python scripts/make_cells.py --panel A --dose-map ladder_map.json [--tag DUMMY]

Output files (one sbatch array task per file; ONE WRITER PER --out, per
scripts/run_panel.py):
    V      -> cells_panelV.json                    24 cells (24 eval x NONE x NULL@0)
    B      -> cells_panelB_NONE.json               30 cells each: (12d + 12n + 6tp)
              cells_panelB_SUPPRESS.json           x that shard's condition x NULL@0
              cells_panelB_NEUTRAL_INSTR.json      (sharded BY CONDITION)
    LADDER -> cells_ladder.json                    84 cells: 7 coefficients x
              {D-CTX, D-PV, R1} x 4 discovery scenarios (d01, d03, n01, n03) x NONE
    A      -> cells_panelA[_TAG]_DCTX.json         72 cells: D-CTX x 6 doses x 12 neutral
              cells_panelA[_TAG]_DPV.json          72 cells: D-PV  x 6 doses x 12 neutral
              cells_panelA[_TAG]_CTRL.json        120 cells: {R1,R2,R3,SEM,OTHER}
                                                   x {+1,+2} SD x 12 neutral
              (all shards <= ~120 cells; all condition NONE)

Documented decisions (SPEC/task-silent points):
  * dose_sd in LADDER cells is a 0.0 PLACEHOLDER: the ladder is what MAPS
    coefficient -> SD units, so no SD value exists yet at generation time. The
    coefficient field carries the sweep; ladder analysis must key on coefficient
    and must NOT read dose_sd from ladder rows. (JSON has no NaN; 0.0 also keeps
    schema.validate_row clean.) In Panel A the roles flip: dose_sd carries the
    pre-registered grid and coefficient comes from the --dose-map produced by
    ladder analysis.
  * Panel A dose 0 cells keep their direction label (D-CTX/D-PV) with
    coefficient 0.0, per the identical-hook-path rule (SPEC "Compute + storage")
    wants dose-0 through the same hook, not a NULL cell.
  * directions_dir and the probe path are emitted RELATIVE TO THE OUTDIR because
    src/runner.py resolves cells.json paths against the cells file's own
    directory. A literal "directions/" inside battery/ would resolve to
    battery/directions/ (does not exist); the generator emits the correct
    relative path to <repo>/directions and <repo>/instruments/probe.npz
    ("../directions" etc. for the default outdir).
  * --no-probe omits the instruments block entirely (files that must run before
    the probe exists: V, LADDER). The runner then stamps PLACEHOLDER_MEAN
    provenance; panels V/B/A additionally need --no-require-real-instruments on
    the run_panel.py side to start in that state. The SAE instrument is always
    supplied via run_panel.py --sae-release/--sae-id (CLI), never via this file.
  * LADDER scenarios: the first two matched discovery-split pairs by id
    (d01/n01, d03/n03): deterministic, surface-matched, discovery-only.
  * The dose map format is {direction: {SD_string: coefficient}}; keys are
    normalized through float() so "-2", "-2.0" and "-2.00" are equivalent.
    SD 0.0 may be omitted (coefficient 0.0 is implied). Nonzero entries must
    have sign(coefficient) == sign(SD): SD units point toward distress by
    construction, so a sign mismatch is a calibration bug, not a choice.
  * A top-level "notes" key documents each file in-band. Safe: run_panel.py's
    scope_hash covers only scope_static + the per-cell dicts, so notes edits
    never invalidate resume state.

Determinism: no randomness anywhere; cells are emitted in sorted (scenario,
condition, direction, coefficient/dose) order; the script builds every payload
twice and asserts byte-identical serialization before writing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import schema  # noqa: E402

BATTERY_DIR = REPO_ROOT / "battery"

# Exact strings shared with cluster/make_smoke_cells.py (the format precedent).
GEN_CONFIG = {"temperature": 0.7, "max_new_tokens": 200}
FIXED_CONTINUATION = (
    "I will continue with the task and address the points raised in order."
)

LADDER_COEFFICIENTS = (500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0)
LADDER_DIRECTIONS = ("D-CTX", "D-PV", "R1")
LADDER_SCENARIOS = ("d01", "d03", "n01", "n03")  # first 2 discovery pairs

PANEL_A_MAIN_DIRECTIONS = ("D-CTX", "D-PV")
PANEL_A_DOSE_GRID = (-2.0, -1.0, 0.0, 0.5, 1.0, 2.0)
PANEL_A_CONTROL_DIRECTIONS = ("R1", "R2", "R3", "SEM", "OTHER")
PANEL_A_CONTROL_DOSES = (1.0, 2.0)


# ------------------------------------------------------------------ battery IO

def load_battery() -> Tuple[Dict[str, dict], Dict[str, str], dict]:
    """Return (scenarios by id, condition system prompts by id, split)."""
    scenarios_raw = json.loads((BATTERY_DIR / "scenarios.json").read_text())
    scenarios = {s["id"]: s for s in scenarios_raw["scenarios"]}
    conditions_raw = json.loads((BATTERY_DIR / "conditions.json").read_text())
    prompts = {c["id"]: c["system_prompt"] for c in conditions_raw["conditions"]}
    split = json.loads((BATTERY_DIR / "split.json").read_text())
    return scenarios, prompts, split


def scenario_ids_by_type(scenarios: Dict[str, dict], type_: str) -> List[str]:
    """Sorted ids of one scenario type (ids are zero-padded: lexicographic == numeric)."""
    return sorted(sid for sid, s in scenarios.items() if s["type"] == type_)


# ------------------------------------------------------------------- dose map

def canon_sd(sd: float) -> str:
    return format(float(sd), "g")


def load_dose_map(path: str) -> Dict[str, Dict[str, float]]:
    """Load {direction: {SD: coefficient}} with float-normalized SD keys."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"[make_cells] dose map {path}: top level must be an object")
    out: Dict[str, Dict[str, float]] = {}
    for direction, table in raw.items():
        if direction not in schema.DIRECTION_SOURCES:
            raise SystemExit(
                f"[make_cells] dose map {path}: unknown direction {direction!r} "
                f"(must be in schema.DIRECTION_SOURCES)"
            )
        if not isinstance(table, dict):
            raise SystemExit(
                f"[make_cells] dose map {path}: entry for {direction} must map SD -> coefficient"
            )
        norm: Dict[str, float] = {}
        for k, v in table.items():
            try:
                sd = float(k)
                coef = float(v)
            except (TypeError, ValueError):
                raise SystemExit(
                    f"[make_cells] dose map {path}: non-numeric entry {direction}[{k!r}] = {v!r}"
                )
            key = canon_sd(sd)
            if key in norm:
                raise SystemExit(
                    f"[make_cells] dose map {path}: duplicate SD {key} for {direction} "
                    "after float normalization"
                )
            if sd == 0.0 and coef != 0.0:
                raise SystemExit(
                    f"[make_cells] dose map {path}: {direction}[0] must be 0.0, got {coef}"
                )
            if sd != 0.0 and (sd > 0) != (coef > 0):
                raise SystemExit(
                    f"[make_cells] dose map {path}: sign mismatch {direction}[{key}] = {coef} "
                    "(SD units point toward distress; sign(coef) must equal sign(SD))"
                )
            norm[key] = coef
        out[direction] = norm
    return out


def dose_to_coef(dose_map: Dict[str, Dict[str, float]], direction: str, sd: float) -> float:
    if sd == 0.0:
        return 0.0
    table = dose_map.get(direction)
    if table is None or canon_sd(sd) not in table:
        raise SystemExit(
            f"[make_cells] dose map is missing {direction} @ {canon_sd(sd)} SD; "
            f"Panel A needs {PANEL_A_MAIN_DIRECTIONS} at SDs {PANEL_A_DOSE_GRID} and "
            f"{PANEL_A_CONTROL_DIRECTIONS} at SDs {PANEL_A_CONTROL_DOSES}"
        )
    return table[canon_sd(sd)]


# ---------------------------------------------------------------- cell payload

def make_cell(
    scenarios: Dict[str, dict],
    prompts: Dict[str, str],
    scenario_id: str,
    condition: str,
    direction: str,
    dose_sd: float,
    coefficient: float,
) -> dict:
    """One cell dict, exactly the smoke-file key set (extra keys would leak into
    scope_hash and needlessly invalidate resume state on cosmetic edits)."""
    return {
        "scenario_id": scenario_id,
        "condition": condition,
        "direction": direction,
        "dose_sd": float(dose_sd),
        "coefficient": float(coefficient),
        "system_prompt": prompts[condition],
        "turns": scenarios[scenario_id]["turns"],
    }


def relpath_from_outdir(outdir: Path, target: Path) -> str:
    """Path string that resolves to `target` when joined to the cells file's dir
    (src/runner.py resolves relative paths against the cells file directory)."""
    return os.path.relpath(target, outdir.resolve()).replace(os.sep, "/")


def make_file_payload(
    run_id: str,
    cells: List[dict],
    outdir: Path,
    no_probe: bool,
    notes: str,
) -> dict:
    payload = {
        "run_id": run_id,
        "gen_config": dict(GEN_CONFIG),
        "fixed_continuation": FIXED_CONTINUATION,
        "directions_dir": relpath_from_outdir(outdir, REPO_ROOT / "directions"),
        "notes": notes,
        "cells": cells,
    }
    if not no_probe:
        payload["instruments"] = {
            "probe": {
                "path": relpath_from_outdir(outdir, REPO_ROOT / "instruments" / "probe.npz")
            }
            # SAE deliberately absent: supplied via run_panel.py --sae-release/--sae-id.
        }
    return payload


# -------------------------------------------------------------------- builders

def build_panel_v(scenarios, prompts, split, outdir: Path, no_probe: bool):
    """24 eval scenarios x NONE x NULL@0. One file."""
    eval_ids = scenario_ids_by_type(scenarios, "distress") + scenario_ids_by_type(scenarios, "neutral")
    cells = [make_cell(scenarios, prompts, sid, "NONE", "NULL", 0.0, 0.0)
             for sid in sorted(eval_ids)]
    notes = ("Panel V (validity + anchor): 24 eval scenarios x NONE x NULL@0. "
             "No steering, no suppression. Cross with --seed-list 0,1,2 at run time. "
             "Generated with --no-probe: run before instruments/probe.npz exists "
             "(needs --no-require-real-instruments); regenerate without --no-probe "
             "for post-probe re-runs." if no_probe else
             "Panel V (validity + anchor): 24 eval scenarios x NONE x NULL@0. "
             "No steering, no suppression. Cross with --seed-list 0,1,2 at run time.")
    payload = make_file_payload("panelV-r1", cells, outdir, no_probe, notes)
    arith = f"24 eval scenarios x 1 condition (NONE) x 1 direction (NULL@0) = {len(cells)}"
    return [("cells_panelV.json", payload, 24, arith)]


def build_panel_b(scenarios, prompts, split, outdir: Path, no_probe: bool):
    """(12d + 12n + 6tp) x each condition, sharded BY CONDITION into 3 files."""
    eval_ids = scenario_ids_by_type(scenarios, "distress") + scenario_ids_by_type(scenarios, "neutral")
    tp_ids = scenario_ids_by_type(scenarios, "third_person")
    out = []
    for condition in schema.FROZEN_CONDITIONS:  # ("NONE", "SUPPRESS", "NEUTRAL_INSTR")
        cells = [make_cell(scenarios, prompts, sid, condition, "NULL", 0.0, 0.0)
                 for sid in sorted(eval_ids) + tp_ids]
        notes = (f"Panel B (masking) shard: condition {condition} only. "
                 "(12 distress + 12 neutral + 6 third-person) x NULL@0. Sharded by "
                 "condition so each sbatch array task owns exactly one --out "
                 "(ONE WRITER PER OUT). Cross with --seed-list 0,1,2 at run time. "
                 "Requires real probe + SAE instruments (SPEC validity gate).")
        payload = make_file_payload(f"panelB-{condition}-r1", cells, outdir, no_probe, notes)
        arith = (f"(12 distress + 12 neutral + 6 third-person) x 1 condition ({condition}) "
                 f"x 1 direction (NULL@0) = {len(cells)}")
        out.append((f"cells_panelB_{condition}.json", payload, 30, arith))
    return out


def build_ladder(scenarios, prompts, split, outdir: Path, no_probe: bool):
    """7 coefficients x {D-CTX, D-PV, R1} x 4 discovery scenarios x NONE. One file."""
    discovery = set(split["discovery"]["distress"]) | set(split["discovery"]["neutral"])
    for sid in LADDER_SCENARIOS:
        if sid not in discovery:
            raise SystemExit(f"[make_cells] LADDER scenario {sid} is not in the discovery split")
    cells = [
        make_cell(scenarios, prompts, sid, "NONE", direction, 0.0, coef)
        for direction in LADDER_DIRECTIONS
        for sid in LADDER_SCENARIOS
        for coef in LADDER_COEFFICIENTS
    ]
    notes = ("Dose ladder: coefficients {500,1000,2000,4000,8000,16000,32000} x "
             "{D-CTX, D-PV, R1} x 4 discovery-split scenarios (d01,d03,n01,n03 = "
             "first two matched pairs) x NONE. dose_sd is a 0.0 PLACEHOLDER in every "
             "cell: the ladder is what maps coefficient -> SD units, so analysis must "
             "key on coefficient and ignore dose_sd here. Run with 1 seed "
             "(--seed-list 0) per SPEC. Runs before the probe exists (--no-probe).")
    payload = make_file_payload("ladder-r1", cells, outdir, no_probe, notes)
    arith = (f"{len(LADDER_COEFFICIENTS)} coefficients x {len(LADDER_DIRECTIONS)} directions "
             f"x {len(LADDER_SCENARIOS)} scenarios x 1 condition (NONE) = {len(cells)}")
    return [("cells_ladder.json", payload, 84, arith)]


def build_panel_a(scenarios, prompts, split, outdir: Path, no_probe: bool,
                  dose_map: Dict[str, Dict[str, float]], tag: Optional[str]):
    """Main grid + controls on the 12 NEUTRAL scenarios, sharded <= ~120 cells."""
    neutral_ids = scenario_ids_by_type(scenarios, "neutral")
    if len(neutral_ids) != 12:
        raise SystemExit(f"[make_cells] expected 12 neutral scenarios, found {len(neutral_ids)}")
    infix = f"_{tag}" if tag else ""
    out = []

    for direction, shard in (("D-CTX", "DCTX"), ("D-PV", "DPV")):
        cells = [
            make_cell(scenarios, prompts, sid, "NONE", direction, sd,
                      dose_to_coef(dose_map, direction, sd))
            for sid in neutral_ids
            for sd in PANEL_A_DOSE_GRID
        ]
        notes = (f"Panel A (dose-response) shard: {direction} x doses "
                 "{-2,-1,0,+0.5,+1,+2} SD x 12 NEUTRAL scenarios x NONE. Coefficients "
                 "from the --dose-map (ladder analysis); dose 0 keeps the direction "
                 "label with coefficient 0.0 (identical hook path). Cross with "
                 "--seed-list 0,1,2." + (" DUMMY DOSE MAP - do not run panels from "
                 "this file; regenerate with the real ladder_map.json." if tag else ""))
        payload = make_file_payload(f"panelA-{shard}-r1", cells, outdir, no_probe, notes)
        arith = (f"1 direction ({direction}) x {len(PANEL_A_DOSE_GRID)} doses "
                 f"x 12 neutral scenarios = {len(cells)}")
        out.append((f"cells_panelA{infix}_{shard}.json", payload, 72, arith))

    cells = [
        make_cell(scenarios, prompts, sid, "NONE", direction, sd,
                  dose_to_coef(dose_map, direction, sd))
        for direction in PANEL_A_CONTROL_DIRECTIONS
        for sid in neutral_ids
        for sd in PANEL_A_CONTROL_DOSES
    ]
    notes = ("Panel A controls shard: {R1,R2,R3,SEM,OTHER} x {+1,+2} SD x 12 NEUTRAL "
             "scenarios x NONE. Coefficients from the --dose-map (KL-dose-matched by "
             "ladder analysis). Cross with --seed-list 0,1,2."
             + (" DUMMY DOSE MAP - do not run panels from this file; regenerate "
                "with the real ladder_map.json." if tag else ""))
    payload = make_file_payload("panelA-CTRL-r1", cells, outdir, no_probe, notes)
    arith = (f"{len(PANEL_A_CONTROL_DIRECTIONS)} control directions x "
             f"{len(PANEL_A_CONTROL_DOSES)} doses x 12 neutral scenarios = {len(cells)}")
    out.append((f"cells_panelA{infix}_CTRL.json", payload, 120, arith))
    return out


# ------------------------------------------------------------------ validation

def validate_file(path: Path, expected_count: int, scenarios: Dict[str, dict],
                  prompts: Dict[str, str], no_probe: bool) -> None:
    """Re-load from DISK and check the invariants the task pins down."""
    cfg = json.loads(path.read_text())
    problems: List[str] = []
    if cfg.get("gen_config") != GEN_CONFIG:
        problems.append(f"gen_config {cfg.get('gen_config')} != {GEN_CONFIG}")
    if cfg.get("fixed_continuation") != FIXED_CONTINUATION:
        problems.append("fixed_continuation does not match the smoke-file string")
    if not cfg.get("run_id"):
        problems.append("missing run_id")
    if not cfg.get("directions_dir"):
        problems.append("missing directions_dir")
    else:
        resolved = (path.parent / cfg["directions_dir"]).resolve()
        if resolved != (REPO_ROOT / "directions").resolve():
            problems.append(f"directions_dir resolves to {resolved}, not <repo>/directions")
    if no_probe:
        if cfg.get("instruments"):
            problems.append("--no-probe file unexpectedly carries instruments")
    else:
        probe_path = ((cfg.get("instruments") or {}).get("probe") or {}).get("path")
        if not probe_path:
            problems.append("missing instruments.probe.path")
        else:
            resolved = (path.parent / probe_path).resolve()
            if resolved != (REPO_ROOT / "instruments" / "probe.npz").resolve():
                problems.append(f"probe path resolves to {resolved}, not <repo>/instruments/probe.npz")
    cells = cfg.get("cells", [])
    if len(cells) != expected_count:
        problems.append(f"cell count {len(cells)} != expected {expected_count}")
    seen = set()
    for i, c in enumerate(cells):
        sid, cond, direction = c.get("scenario_id"), c.get("condition"), c.get("direction")
        if sid not in scenarios:
            problems.append(f"cell {i}: unknown scenario_id {sid!r}")
            continue
        if cond not in schema.CONDITIONS:
            problems.append(f"cell {i}: condition {cond!r} not in schema.CONDITIONS")
        if direction not in schema.DIRECTION_SOURCES:
            problems.append(f"cell {i}: direction {direction!r} not in schema.DIRECTION_SOURCES")
        if direction == "NULL" and (c.get("dose_sd") != 0.0 or c.get("coefficient") != 0.0):
            problems.append(f"cell {i}: NULL direction with nonzero dose/coefficient")
        if cond in prompts and c.get("system_prompt") != prompts[cond]:
            problems.append(f"cell {i}: system_prompt does not match conditions.json[{cond}]")
        if c.get("turns") != scenarios[sid]["turns"]:
            problems.append(f"cell {i}: turns differ from scenarios.json[{sid}]")
        ident = (sid, cond, direction, c.get("dose_sd"), c.get("coefficient"))
        if ident in seen:
            problems.append(f"cell {i}: duplicate cell {ident}")
        seen.add(ident)
    if problems:
        raise SystemExit(f"[make_cells] VALIDATION FAILED for {path}:\n  " + "\n  ".join(problems))


# ------------------------------------------------------------------------ main

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Masked Distress panel cells generator")
    p.add_argument("--panel", required=True, choices=["V", "B", "LADDER", "A"])
    p.add_argument("--dose-map", default=None,
                   help="ladder analysis output: {direction: {SD: coefficient}} (Panel A only)")
    p.add_argument("--outdir", default=str(BATTERY_DIR), help="output directory (default battery/)")
    p.add_argument("--no-probe", action="store_true",
                   help="omit the probe instrument (files that run before instruments/probe.npz "
                        "exists: V, LADDER)")
    p.add_argument("--tag", default=None,
                   help="filename infix, e.g. DUMMY for a placeholder Panel A dose map")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    scenarios, prompts, split = load_battery()

    if args.panel == "A" and not args.dose_map:
        raise SystemExit("[make_cells] Panel A requires --dose-map (ladder analysis output)")
    if args.panel != "A" and args.dose_map:
        raise SystemExit("[make_cells] --dose-map is Panel A only")

    def build():
        if args.panel == "V":
            return build_panel_v(scenarios, prompts, split, outdir, args.no_probe)
        if args.panel == "B":
            return build_panel_b(scenarios, prompts, split, outdir, args.no_probe)
        if args.panel == "LADDER":
            return build_ladder(scenarios, prompts, split, outdir, args.no_probe)
        dose_map = load_dose_map(args.dose_map)
        return build_panel_a(scenarios, prompts, split, outdir, args.no_probe,
                             dose_map, args.tag)

    files = build()
    # Determinism check: a second build must serialize byte-identically.
    rebuilt = build()
    for (n1, p1, _, _), (n2, p2, _, _) in zip(files, rebuilt):
        if n1 != n2 or json.dumps(p1, sort_keys=True) != json.dumps(p2, sort_keys=True):
            raise SystemExit(f"[make_cells] non-deterministic build detected for {n1}")

    total = 0
    for name, payload, expected, arith in files:
        path = outdir / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        validate_file(path, expected, scenarios, prompts, args.no_probe)
        n = len(payload["cells"])
        total += n
        print(f"[make_cells] {path}  cells={n}  ({arith})")
    print(f"[make_cells] panel {args.panel}: {len(files)} file(s), {total} cells total "
          f"(x seeds at run time via run_panel.py --seed-list)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
