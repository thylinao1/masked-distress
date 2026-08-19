#!/usr/bin/env python
"""Producer for results/panelB_second_model.json (2026-08-17).

The Panel B masking estimator, unchanged, on the second model family, plus that model's own
validity gate. Both reading rules were fixed in the preregistration's dated note before any row
was read:

  * GATE: held-out AUC of this model's probe on its own confirmation split must reach 0.80,
    otherwise the masking numbers are reported as uninterpretable rather than as a replication.
  * REPLICATION: replicates if the divergence interval excludes zero with the same sign as the
    primary model; partially replicates if only the expression drop reproduces; fails if the
    divergence interval contains zero.

    .venv/bin/python scripts/second_model_checks.py [--tag qwen] [--B 10000] [--no-emit]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis import results_io, stats  # noqa: E402
from analysis.loading import load_results  # noqa: E402
from analysis.panels import EXPRESSION_PRIMARY, INTERNAL_PRIMARY, _divergence_stat, _mean_drop_stat  # noqa: E402

NAME = "panelB_second_model"
GATE_AUC_MIN = 0.80  # fixed in the prereg note before the data


def compute(repo: Path, tag: str, B: int, seed: int) -> dict:
    files = [f"results-cluster/{tag}_panelB_none.jsonl",
             f"results-cluster/{tag}_panelB_neutral_instr.jsonl",
             f"results-cluster/{tag}_panelB_suppress.jsonl"]
    vfile = f"results-cluster/{tag}_panelV.jsonl"
    present = [f for f in files if (repo / f).exists()]
    missing = [f for f in files if f not in present]
    if (repo / vfile).exists():
        present_all = present + [vfile]
    else:
        present_all = present
        missing.append(vfile)
    df, report = load_results([str(repo / p) for p in present_all],
                              split_path=repo / "battery" / "split.json", verbose=False)
    if report.n_error_rows or report.n_validate_failures:
        raise RuntimeError(f"unclean load: {report.n_error_rows} error rows, "
                           f"{report.n_validate_failures} validate failures")
    if bool(df["synthetic"].any()):
        raise RuntimeError("synthetic rows present; refusing")

    # ---- the gate, from this model's own probe metadata plus a recomputation on its Panel V rows
    meta_path = repo / f"instruments_{tag}" / "probe_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    gate_auc_trainer = meta.get("auc_confirmation")
    v = df[(df["panel"] == "V") & (df["split_side"] == "confirmation")
           & df["scenario_class"].isin(["distress", "neutral"])]
    gate_auc_panel = None
    if not v.empty:
        m = v.groupby(["scenario_id", "scenario_class"])[INTERNAL_PRIMARY].mean().reset_index()
        pos = m.loc[m["scenario_class"] == "distress", INTERNAL_PRIMARY].to_numpy()
        neg = m.loc[m["scenario_class"] == "neutral", INTERNAL_PRIMARY].to_numpy()
        if len(pos) and len(neg):
            wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
            gate_auc_panel = float(wins / (len(pos) * len(neg)))
    gate_value = gate_auc_panel if gate_auc_panel is not None else gate_auc_trainer
    gate_pass = bool(gate_value is not None and gate_value >= GATE_AUC_MIN)

    # ---- the masking estimator, unchanged
    b = df[(df["panel"] == "B") & (df["direction"] == "NULL") & (df["split_side"] == "confirmation")
           & df["scenario_class"].isin(["distress", "neutral"])]
    kw = dict(strata_col="scenario_class", B=B, seed=seed)
    masking = {}
    if not b.empty and (b["condition"] == "SUPPRESS").any() and (b["condition"] == "NEUTRAL_INSTR").any():
        re_ = stats.bca_cluster_bootstrap_frame(b, _mean_drop_stat(EXPRESSION_PRIMARY), **kw)
        ri_ = stats.bca_cluster_bootstrap_frame(b, _mean_drop_stat(INTERNAL_PRIMARY), **kw)
        rd_ = stats.bca_cluster_bootstrap_frame(b, _divergence_stat, **kw)
        piv = b.pivot_table(index=["scenario_id", "scenario_class"], columns="condition",
                            values=[EXPRESSION_PRIMARY, INTERNAL_PRIMARY], aggfunc="mean")

        def cm(ch, cond, cls):
            try:
                sel = piv[(ch, cond)]
                return float(sel[piv.index.get_level_values("scenario_class") == cls].mean())
            except KeyError:
                return float("nan")

        masking = {
            "expression_drop": {"value": re_.value, "ci_low": re_.ci_low, "ci_high": re_.ci_high},
            "internal_drop": {"value": ri_.value, "ci_low": ri_.ci_low, "ci_high": ri_.ci_high},
            "divergence": {"value": rd_.value, "ci_low": rd_.ci_low, "ci_high": rd_.ci_high},
            "n_clusters": rd_.n_clusters,
            "class_condition_means": {
                ch_name: {cond: {cls: cm(ch, cond, cls) for cls in ("distress", "neutral")}
                          for cond in ("NONE", "NEUTRAL_INSTR", "SUPPRESS")}
                for ch_name, ch in (("expression", EXPRESSION_PRIMARY), ("internal", INTERNAL_PRIMARY))},
        }

    div = masking.get("divergence", {})
    expr = masking.get("expression_drop", {})
    excl = lambda d: bool(d and math.isfinite(d.get("ci_low", float("nan")))
                          and math.isfinite(d.get("ci_high", float("nan")))
                          and (d["ci_low"] > 0 or d["ci_high"] < 0))
    if not gate_pass:
        verdict = "gate_failed_numbers_uninterpretable"
    elif excl(div) and div.get("value", 0) > 0:
        verdict = "replicates"
    elif excl(expr) and expr.get("value", 0) > 0:
        verdict = "partially_replicates_expression_only"
    else:
        verdict = "does_not_replicate"

    entry = {
        "value": {"model": meta.get("model_id") or "second model", "gate": {
            "auc_confirmation_recomputed_on_panelV": gate_auc_panel,
            "auc_confirmation_from_trainer": gate_auc_trainer,
            "threshold": GATE_AUC_MIN, "passed": gate_pass},
            "masking": masking, "verdict": verdict},
        "ci_low": None, "ci_high": None,
        "n_clusters": int(b["scenario_id"].nunique()) if not b.empty else 0,
        "extra": {"split_side": "confirmation", "tag": tag,
                  "estimator": "unchanged Panel B masking estimator (D1, D2, D6, D7)",
                  "sae_channel": "not analysed for this family: no released SAE, allowance stamped in row provenance",
                  "files_present": present_all, "files_missing": missing,
                  "reading_rules": ("fixed in the prereg note before the data: gate at held-out AUC "
                                    f"{GATE_AUC_MIN}; replicates if the divergence interval excludes zero "
                                    "with the same sign; partially replicates if only the expression drop "
                                    "reproduces; otherwise does not replicate"),
                  "probe_meta": {k: meta.get(k) for k in ("auc_discovery_loo", "auc_confirmation", "layer", "C")}},
    }
    return {"entry": entry, "load_report": report,
            "source_files": present_all + [f"instruments_{tag}/probe_meta.json", "battery/split.json"]}


def print_summary(res: dict) -> None:
    v = res["entry"]["value"]
    g = v["gate"]
    print(f"[2nd-model] gate AUC {g['auc_confirmation_recomputed_on_panelV']} "
          f"(trainer {g['auc_confirmation_from_trainer']}) threshold {g['threshold']} -> passed={g['passed']}")
    m = v.get("masking") or {}
    if m:
        for k in ("expression_drop", "internal_drop", "divergence"):
            d = m[k]
            print(f"[2nd-model] {k:16s} {d['value']:+.3f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]")
    print("[2nd-model] VERDICT:", v["verdict"], "| missing:", res["entry"]["extra"]["files_missing"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="qwen")
    ap.add_argument("--out", default=str(_REPO / "results"))
    ap.add_argument("--B", type=int, default=stats.DEFAULT_B)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-emit", action="store_true")
    args = ap.parse_args(argv)
    res = compute(_REPO, tag=args.tag, B=args.B, seed=args.seed)
    print_summary(res)
    if args.no_emit:
        return 0
    results_io.emit_one(NAME, res["entry"], out_dir=args.out, source_files=res["source_files"],
                        synthetic=False,
                        extra_provenance={"producer": "scripts/second_model_checks.py",
                                          "n_rows_loaded": res["load_report"].n_rows,
                                          "n_error_rows": res["load_report"].n_error_rows,
                                          "n_cells": res["load_report"].n_cells},
                        B=args.B, seed=args.seed)
    print(f"[2nd-model] wrote {Path(args.out) / (NAME + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
