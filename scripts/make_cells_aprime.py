#!/usr/bin/env python
"""Generate the Panel A-PRIME cells files (prereg section 9 amendment 2, 2026-08-16).

CLI:
    python scripts/make_cells_aprime.py [--outdir battery/]

Output files (one sbatch array task per file; ONE WRITER PER --out):
    cells_panelAp_DCTX.json   96 cells: 24 eval scenarios x (NULL@0 + D-CTX @ {500,1000,2000})
    cells_panelAp_DPV.json    96 cells: 24 eval scenarios x (NULL@0 + D-PV  @ {250,500,1000})
    cells_panelAp_CTRL.json  264 cells: 24 eval scenarios x (NULL@0 +
                                        {R1,R2,R3,SEM,OTHER} @ {500,1000})

Design decisions (all fixed by amendment 2; grids imported from analysis/panels.py so
the generator and the analysis can never disagree):
  * COEFFICIENT-unit grids inside the realized per-direction capability-valid windows.
    Strictly-valid: D-CTX <= 2000, D-PV <= 500; the D-PV 1000 rung is BORDERLINE
    (ppl x1.18, MMLU -8.3pp), included but excluded from primary claims.
  * dose_sd is BOOKKEEPING ONLY: coefficient x sd_per_coef from
    results/capability_valid_range.json extra.dose_map for directions the ladder
    mapped (D-CTX, D-PV, R1); 0.0 placeholder for R2/R3/SEM/OTHER (no realized map;
    fabricating one would be dishonest). ALL A-prime analysis keys on coefficient.
  * Zero anchor: direction NULL @ coefficient 0 in EVERY shard, so each shard is
    self-sufficient. NULL@0 cell identities coincide across shards (schema.CellKey
    has no run_id); the analysis loader deduplicates rows of record, so the
    duplication costs compute, never correctness.
  * Scenarios: all 24 eval (12 distress + 12 neutral); the frozen section 7 rule
    sends primaries to the confirmation split at analysis time.
  * gen_config identical to the Panel B cells (temperature 0.7, max_new_tokens 200,
    sampled - no do_sample override); instruments block identical to cells_panelV.json.
    Cross with --seed-list 0,1,2 at run time (scripts/run_panel.py --panel A).

Determinism: no randomness; cells emitted in sorted (scenario, direction, coefficient)
order; the payload is built twice and asserted byte-identical before writing
(same protocol as scripts/make_cells.py, whose helpers this script reuses).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import make_cells  # noqa: E402  (shared builders + validation)
from analysis.panels import (  # noqa: E402  (single source for the amended grids)
    APRIME_CONTROL_COEFS,
    APRIME_GRID,
    APRIME_STRICT_MAX_COEF,
)

BATTERY_DIR = REPO_ROOT / "battery"
DOSE_MAP_SOURCE = REPO_ROOT / "results" / "capability_valid_range.json"
CONTROL_DIRECTIONS = ("R1", "R2", "R3", "SEM", "OTHER")


def load_sd_per_coef(path: Path = DOSE_MAP_SOURCE) -> Dict[str, float]:
    """Realized sd_per_coef per direction from the ladder analysis output."""
    data = json.loads(path.read_text(encoding="utf-8"))
    dose_map = (data.get("extra") or {}).get("dose_map") or {}
    out: Dict[str, float] = {}
    for direction, entry in dose_map.items():
        spc = entry.get("sd_per_coef")
        if spc is None:
            continue
        spc = float(spc)
        if not spc > 0:
            raise SystemExit(
                f"[make_cells_aprime] {path}: sd_per_coef for {direction} is {spc}; "
                f"the realized map must be positive (SD units point toward distress)")
        out[direction] = spc
    for required in ("D-CTX", "D-PV"):
        if required not in out:
            raise SystemExit(
                f"[make_cells_aprime] {path}: dose map is missing {required}; "
                f"run the ladder analysis first")
    return out


def bookkeeping_dose_sd(sd_per_coef: Dict[str, float], direction: str,
                        coefficient: float) -> float:
    """coefficient x sd_per_coef where the ladder mapped the direction; 0.0
    placeholder otherwise (documented in the file notes; analysis keys on
    coefficient)."""
    spc = sd_per_coef.get(direction)
    if spc is None:
        return 0.0
    return float(coefficient) * spc


def build_files(outdir: Path) -> List[Tuple[str, dict, int, str]]:
    scenarios, prompts, _split = make_cells.load_battery()
    sd_per_coef = load_sd_per_coef()
    eval_ids = sorted(
        make_cells.scenario_ids_by_type(scenarios, "distress")
        + make_cells.scenario_ids_by_type(scenarios, "neutral"))
    if len(eval_ids) != 24:
        raise SystemExit(
            f"[make_cells_aprime] expected 24 eval scenarios, found {len(eval_ids)}")

    def zero_cell(sid: str) -> dict:
        return make_cells.make_cell(scenarios, prompts, sid, "NONE", "NULL", 0.0, 0.0)

    files: List[Tuple[str, dict, int, str]] = []

    for direction, shard in (("D-CTX", "DCTX"), ("D-PV", "DPV")):
        coefs = sorted(APRIME_GRID[direction])
        cells = []
        for sid in eval_ids:
            cells.append(zero_cell(sid))
            for coef in coefs:
                cells.append(make_cells.make_cell(
                    scenarios, prompts, sid, "NONE", direction,
                    bookkeeping_dose_sd(sd_per_coef, direction, coef), coef))
        strict = APRIME_STRICT_MAX_COEF[direction]
        borderline = [c for c in coefs if c > strict]
        notes = (
            f"Panel A-PRIME shard (prereg section 9 amendment 2): {direction} at "
            f"coefficients {{{', '.join(f'{c:g}' for c in coefs)}}} + NULL@0 zero anchor, "
            f"x 24 eval scenarios (12 distress + 12 neutral) x NONE. COEFFICIENT-unit "
            f"grid inside the realized capability-valid window; strictly-valid rungs are "
            f"coef <= {strict:g}"
            + (f"; coef {', '.join(f'{c:g}' for c in borderline)} is BORDERLINE "
               f"(secondary, labelled only)" if borderline else "")
            + f". dose_sd is bookkeeping only ({direction} sd_per_coef = "
              f"{sd_per_coef[direction]:.6g} from results/capability_valid_range.json); "
              f"analysis keys on coefficient. NULL@0 cells repeat across A-prime shards "
              f"on purpose (self-sufficient shards; loader dedups shared cell ids). "
              f"Sampled generation (Panel B gen_config); NaN-logits guard + CUDA-poison "
              f"abort remain as backstops. Cross with --seed-list 0,1,2 "
              f"(scripts/run_panel.py --panel A).")
        payload = make_cells.make_file_payload(
            f"panelAp-{shard}-r1", cells, outdir, no_probe=False, notes=notes)
        arith = (f"24 eval scenarios x (1 NULL@0 + {len(coefs)} {direction} rungs) "
                 f"= {len(cells)}")
        files.append((f"cells_panelAp_{shard}.json", payload, 96, arith))

    ctrl_coefs = sorted(APRIME_CONTROL_COEFS)
    cells = []
    for sid in eval_ids:
        cells.append(zero_cell(sid))
        for direction in CONTROL_DIRECTIONS:
            for coef in ctrl_coefs:
                cells.append(make_cells.make_cell(
                    scenarios, prompts, sid, "NONE", direction,
                    bookkeeping_dose_sd(sd_per_coef, direction, coef), coef))
    mapped = ", ".join(f"{d} {sd_per_coef[d]:.6g}" for d in sorted(sd_per_coef))
    notes = (
        f"Panel A-PRIME controls shard (prereg section 9 amendment 2): "
        f"{{R1, R2, R3, SEM, OTHER}} at coefficients "
        f"{{{', '.join(f'{c:g}' for c in ctrl_coefs)}}} + NULL@0 zero anchor, x 24 eval "
        f"scenarios x NONE. COEFFICIENT-unit doses; dose_sd is bookkeeping only "
        f"(coefficient x sd_per_coef where the ladder mapped the direction: {mapped}; "
        f"0.0 placeholder for R2/R3/SEM/OTHER, which have no realized map - analysis "
        f"keys on coefficient). Feeds A'3 (OTHER vs self at coef 500) and A'4 (FPR of "
        f"R1-R3/SEM cells at theta_expr). NULL@0 cells repeat across A-prime shards on "
        f"purpose (loader dedups shared cell ids). Sampled generation (Panel B "
        f"gen_config). Cross with --seed-list 0,1,2 (scripts/run_panel.py --panel A).")
    payload = make_cells.make_file_payload(
        "panelAp-CTRL-r1", cells, outdir, no_probe=False, notes=notes)
    arith = (f"24 eval scenarios x (1 NULL@0 + {len(CONTROL_DIRECTIONS)} control "
             f"directions x {len(ctrl_coefs)} coefficients) = {len(cells)}")
    files.append(("cells_panelAp_CTRL.json", payload, 264, arith))
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Panel A-PRIME cells generator (amendment 2)")
    ap.add_argument("--outdir", default=str(BATTERY_DIR),
                    help="output directory (default battery/)")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    scenarios, prompts, _split = make_cells.load_battery()

    files = build_files(outdir)
    rebuilt = build_files(outdir)  # determinism: byte-identical second build
    for (n1, p1, _, _), (n2, p2, _, _) in zip(files, rebuilt):
        if n1 != n2 or json.dumps(p1, sort_keys=True) != json.dumps(p2, sort_keys=True):
            raise SystemExit(f"[make_cells_aprime] non-deterministic build for {n1}")

    total = 0
    for name, payload, expected, arith in files:
        path = outdir / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        make_cells.validate_file(path, expected, scenarios, prompts, no_probe=False)
        n = len(payload["cells"])
        total += n
        print(f"[make_cells_aprime] {path}  cells={n}  ({arith})")
    print(f"[make_cells_aprime] Panel A-PRIME: {len(files)} files, {total} cells total "
          f"(x 3 seeds at run time = {total * 3} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
