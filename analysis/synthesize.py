"""Synthetic results JSONL with KNOWN planted effects, for pipeline self-tests.

Every row passes schema.validate_row and carries gen_config["synthetic"] = true,
which the loader propagates so results provenance can never claim real data (D15).

Planted truths (PLANTED below, asserted by tests/test_analysis.py):
  * Panel B masking divergence = 0.5 natural-separation units
    (expression drop 0.6, internal drop 0.1);
  * Panel A dose-response Spearman ~ 0.8 for D-CTX and D-PV; placebo flat;
  * OTHER effect = 0.3 x self effect (H3 ratio ~ 0.3);
  * internal natural separation = 2.0 probe units, expression separation = 5.0;
  * ladder readout slope 0.05 probe units per coefficient unit (about 20 coef
    per natural SD); capability limits bite near |dose| ~ 2 SD.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import schema

MODEL_ID = "synthetic/masked-distress-sim"
SCOPE_HASH = "synth0001"

PLANTED = {
    "sep_int": 2.0,            # internal natural separation (probe units)
    "sep_expr": 5.0,           # expression natural separation (Q-SELF units)
    "drop_expr_frac": 0.6,     # SUPPRESS drop, fraction of natural separation
    "drop_int_frac": 0.1,
    "divergence": 0.5,         # nat-sep units
    "beta_self": 0.62,         # Q-SELF slope per SD dose (Spearman ~ 0.8)
    "other_frac": 0.3,         # OTHER effect relative to self directions
    "probe_base_neutral": 10.0,
    "expr_base_neutral": 1.5,
    "ladder_slope": {"D-CTX": 0.050, "D-PV": 0.048, "R1": 0.004},
    "coef_per_sd_nominal": 20.0,
    "rho_target": 0.8,
}

_D = [f"d{i:02d}" for i in range(1, 13)]
_N = [f"n{i:02d}" for i in range(1, 13)]
_TP = [f"tp{i:02d}" for i in range(1, 7)]
SEEDS = (0, 1, 2)

_INSTRUMENTS = {
    "probe": {"mode": "trained_logistic", "path": "instruments/probe.npz",
              "trained_on": "discovery"},
    "sae": {"mode": "saelens", "release": "gemma-scope-2-12b-it-res",
            "sae_id": "layer_31_width_16k_l0_medium", "feature_ids": [101, 2048, 9013]},
}


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


class _Sim:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.re_int = {s: self.rng.normal(0, 0.15) for s in _D + _N + _TP}
        self.re_expr = {s: self.rng.normal(0, 0.30) for s in _D + _N + _TP}

    def latent(self, scenario: str, condition: str) -> Dict[str, float]:
        p = PLANTED
        cls = ("distress" if scenario.startswith("d")
               else "third_person" if scenario.startswith("tp") else "neutral")
        s = p["probe_base_neutral"] + self.re_int[scenario]
        e = p["expr_base_neutral"] + self.re_expr[scenario]
        if cls == "distress":
            s += p["sep_int"]
            e += p["sep_expr"]
        elif cls == "third_person":
            s += 0.3
            e += 0.5
        if condition == "SUPPRESS":
            if cls == "distress":
                e -= p["drop_expr_frac"] * p["sep_expr"]
                s -= p["drop_int_frac"] * p["sep_int"]
            else:
                e -= 0.2
        return {"s": s, "e": e}

    def channels(self, s: float, e: float, steered: bool, dose_shift: float = 0.0) -> dict:
        rng = self.rng
        s = s + dose_shift
        q_self = _clip(e + rng.normal(0, 0.5), 0, 9)
        probe = {
            "prompt_final": s + rng.normal(0, 0.10),
            "teacher_forced": s + 0.3 + rng.normal(0, 0.15),
            "response_mean": s - 0.2 + rng.normal(0, 0.20),
        }
        sae = {k: 40.0 * (v - 8.0) + rng.normal(0, 3.0) for k, v in probe.items()}
        out = {
            "q_self_logit": q_self,
            "q_self_logit_para1": _clip(q_self + rng.normal(0, 0.35), 0, 9),
            "q_self_logit_para2": _clip(q_self + rng.normal(0, 0.35), 0, 9),
            "q_self_sampled": float(round(q_self)),
            "q_val_logit": _clip(7.5 - 0.7 * (q_self - 1.5) + rng.normal(0, 0.3), 0, 9),
            "q_drift_logit": _clip(4.0 + rng.normal(0, 0.5), 0, 9),
            "exit_logit_diff": -8.0 + 0.8 * q_self + rng.normal(0, 0.5),
            "sentiment_neg": _clip(0.08 + 0.06 * q_self + rng.normal(0, 0.05), 0, 1),
            "probe_score": probe,
            "sae_score": sae,
            "probe_score_projout": (
                {k: v - 0.1 for k, v in probe.items()} if steered else {}),
        }
        return out


def _row(sim: _Sim, panel: str, scenario: str, condition: str, direction: str,
         dose_sd: float, coefficient: float, seed: int,
         dose_shift: float = 0.0, expr_shift: float = 0.0,
         error: Optional[str] = None) -> dict:
    key = schema.CellKey(panel=panel, model_id=MODEL_ID, scenario_id=scenario,
                         condition=condition, direction=direction,
                         dose_sd=float(dose_sd), coefficient=float(coefficient),
                         seed=int(seed))
    lat = sim.latent(scenario, condition)
    steered = direction != "NULL"
    ch = sim.channels(lat["s"], lat["e"] + expr_shift, steered, dose_shift)
    row = {
        "schema_version": schema.SCHEMA_VERSION,
        "run_id": f"synthetic-{panel}",
        "git_hash": "synthetic",
        "config_hash": "synthcfg01",
        "key": asdict(key),
        "cell_id": key.cell_id(),
        "n_tokens_steered_prefill": 50 if steered else 0,
        "n_tokens_steered_decode": 200 if steered else 0,
        "steer_layer": 16 if steered else None,
        "coef0_identity_ok": None,
        "response_text": f"[synthetic response {panel}/{scenario}/{condition}]",
        **ch,
        "readout_layer": 31,
        "prompt_tokens": 700,
        "response_tokens": 200,
        "gen_config": {
            "temperature": 0.7, "max_new_tokens": 256,
            "scope_hash": SCOPE_HASH, "synthetic": True,
            "instruments": _INSTRUMENTS,
            "planted": {"note": "analysis/synthesize.py; see PLANTED"},
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    if error is None:
        problems = schema.validate_row(row)
        if problems:
            raise AssertionError(f"synthetic row invalid: {problems}")
    return row


def synthesize(out_jsonl: str | Path, seed: int = 7,
               capability_out: Optional[str | Path] = None,
               cosine_out: Optional[str | Path] = None) -> dict:
    """Write the synthetic JSONL (+ optional capability and cosine files)."""
    sim = _Sim(seed)
    p = PLANTED
    rows: List[dict] = []

    # ---- Panel V: 30 scenarios x NONE x 3 seeds ----
    for sc in _D + _N + _TP:
        for sd in SEEDS:
            rows.append(_row(sim, "V", sc, "NONE", "NULL", 0.0, 0.0, sd))

    # ---- Panel B: 30 scenarios x 3 conditions x 3 seeds ----
    for sc in _D + _N + _TP:
        for cond in schema.FROZEN_CONDITIONS:
            for sd in SEEDS:
                rows.append(_row(sim, "B", sc, cond, "NULL", 0.0, 0.0, sd))

    # ---- LADDER: 4 discovery scenarios, 3 directions x 5 coefficients, seed 0 ----
    lad_scen = ["d01", "n01", "d03", "n03"]
    for sc in lad_scen:
        rows.append(_row(sim, "LADDER", sc, "NONE", "NULL", 0.0, 0.0, 0))
        for direction, slope in p["ladder_slope"].items():
            for coef in (5.0, 10.0, 20.0, 40.0, 64.0):
                shift = slope * coef
                rows.append(_row(sim, "LADDER", sc, "NONE", direction, 0.0, coef, 0,
                                 dose_shift=shift, expr_shift=p["beta_self"] * shift))

    # ---- Panel A: neutral scenarios ----
    self_doses = (-2.0, -1.0, 0.0, 0.5, 1.0, 2.0)
    ctrl_doses = (1.0, 2.0)
    probe_gain = {"D-CTX": 1.0, "D-PV": 1.0, "R1": 0.05, "R2": 0.05, "R3": 0.05,
                  "SEM": 0.10, "OTHER": 0.20}
    expr_beta = {"D-CTX": p["beta_self"], "D-PV": p["beta_self"],
                 "R1": 0.0, "R2": 0.0, "R3": 0.0, "SEM": 0.0,
                 "OTHER": p["beta_self"] * p["other_frac"]}
    for sc in _N:
        for sd in SEEDS:
            for direction in ("D-CTX", "D-PV"):
                for dose in self_doses:
                    rows.append(_row(
                        sim, "A", sc, "NONE", direction, dose,
                        dose * p["coef_per_sd_nominal"], sd,
                        dose_shift=probe_gain[direction] * dose,
                        expr_shift=expr_beta[direction] * dose))
            for direction in ("R1", "R2", "R3", "SEM", "OTHER"):
                for dose in ctrl_doses:
                    rows.append(_row(
                        sim, "A", sc, "NONE", direction, dose,
                        dose * p["coef_per_sd_nominal"], sd,
                        dose_shift=probe_gain[direction] * dose,
                        expr_shift=expr_beta[direction] * dose))

    # ---- exercise the loader conventions ----
    # an error row that a later non-error row supersedes (same cell identity)
    dup = _row(sim, "V", "d01", "NONE", "NULL", 0.0, 0.0, 0,
               error="synthetic transient CUDA OOM (superseded)")
    # a cell whose ONLY row is an error row
    orphan = _row(sim, "V", "tp06", "SUPPRESS", "NULL", 0.0, 0.0, 99,
                  error="synthetic permanent failure (error-only cell)")
    all_rows = [dup] + rows + [orphan]

    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    paths = {"jsonl": out_jsonl}
    if capability_out is not None:
        # limits bite between coef 48 (pass) and 50 (mmlu fail): the valid range
        # lands near |dose| ~ 2.3 SD, so the full +-2 sigma grid stays inside it
        cap = {}
        for direction in ("D-CTX", "D-PV"):
            cap[direction] = [
                {"coefficient": c,
                 "ppl_ratio": 1.0 + 0.015 * (c / 20.0) ** 2,
                 "mmlu_drop_pp": 0.8 * (c / 20.0) ** 2}
                for c in (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 48.0, 50.0, 64.0)]
        capability_out = Path(capability_out)
        capability_out.parent.mkdir(parents=True, exist_ok=True)
        capability_out.write_text(json.dumps(cap, indent=2) + "\n", encoding="utf-8")
        paths["capability"] = capability_out
    if cosine_out is not None:
        names = ["D-CTX", "D-PV", "SEM", "OTHER", "R1"]
        vals = {"D-CTX|D-PV": 0.45, "D-CTX|SEM": 0.28, "D-CTX|OTHER": 0.43,
                "D-CTX|R1": -0.01, "D-PV|SEM": 0.03, "D-PV|OTHER": 0.33,
                "D-PV|R1": 0.00, "SEM|OTHER": 0.07, "SEM|R1": -0.02,
                "OTHER|R1": 0.00}
        mat = {}
        for a in names:
            for b in names:
                mat[f"{a}|{b}"] = (1.0 if a == b else
                                   vals.get(f"{a}|{b}", vals.get(f"{b}|{a}", 0.0)))
        mat["_synthetic"] = True
        cosine_out = Path(cosine_out)
        cosine_out.parent.mkdir(parents=True, exist_ok=True)
        cosine_out.write_text(json.dumps(mat, indent=2) + "\n", encoding="utf-8")
        paths["cosine"] = cosine_out
    return {"paths": paths, "planted": PLANTED,
            "n_rows": len(all_rows), "n_error_rows": 2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_REPO / "results-synthetic" / "synthetic.jsonl"))
    ap.add_argument("--capability", default=str(_REPO / "results-synthetic"
                                                / "capability_synthetic.json"))
    ap.add_argument("--cosine", default=str(_REPO / "results-synthetic"
                                            / "cosine_synthetic.json"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    info = synthesize(args.out, seed=args.seed, capability_out=args.capability,
                      cosine_out=args.cosine)
    print(f"[synthesize] {info['n_rows']} rows -> {info['paths']['jsonl']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
